"""Orchestration for nrl.com ingestion: season refresh and historical backfill.

refresh_season() is the hot path (runs before every prep/train/infer/predict):
  1. fetch the season draw -> fixture + bye rows
  2. fetch match centres for finals missing stats and the upcoming round
  3. rewrite the season's feed_cache_fixtures rows (carrying over odds and
     crowd, which this module does not own)
  4. rebuild the season's feed_cache_ladders / feed_cache_performance rows

backfill_seasons() fetches historical match centres (2012+) into the match_*
tables only; frozen feed_cache_* seasons are never rewritten.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

from . import store
from .cache_writer import replace_cache_year
from .draw import (
    FIXTURE_CACHE_COLUMNS,
    draw_to_bye_rows,
    draw_to_fixture_rows,
    fetch_round_draw,
    fetch_season_draw,
    load_venue_timezones,
)
from .ladder import build_season_ladder
from .match_centre import fetch_match_centre, parse_match_centre
from .performance import (
    build_season_performance,
    load_game_scoring,
    load_player_sums_by_game,
    load_team_stats_by_game,
)
from .web import FetchConfig, build_session

DEFAULT_VENUE_CSV = Path("data") / "reference" / "venue_locations.csv"
UPCOMING_FETCH_WINDOW_DAYS = 8
BACKFILL_FIRST_SEASON = 2012


def _print(message: str) -> None:
    print(f"[nrl-data] {message}", flush=True)


def _existing_fixture_extras(
    con: sqlite3.Connection, season: int
) -> dict[int, dict]:
    """Odds/crowd columns owned by other writers, keyed by game_id."""
    columns = [row[1] for row in con.execute("PRAGMA table_info(feed_cache_fixtures)")]
    carry = [
        col
        for col in columns
        if col == "crowd"
        or "odds" in col
        or "line_amount" in col
        or col.startswith("total_")
        or col.startswith("broadcast_channel")
    ]
    if not carry:
        return {}
    quoted = ", ".join(f'"{col}"' for col in carry)
    extras: dict[int, dict] = {}
    for row in con.execute(
        f"SELECT game_id, {quoted} FROM feed_cache_fixtures "
        "WHERE CAST(competition_year AS INTEGER) = ?",
        (int(season),),
    ):
        game_id = int(float(row[0]))
        extras[game_id] = {
            col: row[i + 1] for i, col in enumerate(carry) if row[i + 1] is not None
        }
    return extras


def _fetch_match_centres(
    session,
    config: FetchConfig,
    con: sqlite3.Connection,
    fixture_rows: list[dict],
    season: int,
    only_missing: bool = True,
    include_upcoming: bool = True,
    max_pages: int | None = None,
) -> tuple[int, list[str], dict[str, int]]:
    """Returns (pages fetched, errors, match_centre_url -> authoritative matchId)."""
    have_stats = store.games_with_team_stats(con) if only_missing else set()
    now_utc = dt.datetime.now(dt.timezone.utc).timestamp()
    upcoming_horizon = now_utc + UPCOMING_FETCH_WINDOW_DAYS * 86400

    pages = 0
    errors: list[str] = []
    url_to_match_id: dict[str, int] = {}
    for fixture in fixture_rows:
        raw_url = fixture.get("match_centre_url")
        if not raw_url:
            continue
        url = f"https://www.nrl.com{raw_url}" if raw_url.startswith("/") else raw_url
        game_id = int(float(fixture["game_id"]))
        state = fixture.get("game_state_name")

        if state == "Final":
            if only_missing and game_id in have_stats:
                continue
        elif include_upcoming and state == "Pre Game":
            kickoff = fixture.get("start_time_utc")
            if kickoff is None or not (now_utc - 86400 <= kickoff <= upcoming_horizon):
                continue
        else:
            continue

        if max_pages is not None and pages >= max_pages:
            break

        try:
            payload = fetch_match_centre(session, config, url)
            bundle = parse_match_centre(payload or {}, source_url=url)
        except Exception as exc:  # network/parse failures fail soft
            errors.append(f"{url}: {exc}")
            continue
        pages += 1
        if bundle is None:
            errors.append(f"{url}: no match centre payload")
            continue
        url_to_match_id[raw_url] = bundle["game_id"]
        store.upsert_match_bundle(
            con,
            bundle,
            competition_year=season,
            round_id=int(float(fixture["round_id"])),
            team_home=fixture.get("team_home"),
            team_away=fixture.get("team_away"),
        )
    return pages, errors, url_to_match_id


def _stored_match_ids_by_url(con: sqlite3.Connection) -> dict[str, int]:
    """match_centre_url (relative) -> matchId from previously stored stats."""
    mapping: dict[str, int] = {}
    for source_url, game_id in con.execute(
        "SELECT DISTINCT source_url, game_id FROM match_team_stats "
        "WHERE source_url IS NOT NULL AND source_url != ''"
    ):
        relative = source_url.replace("https://www.nrl.com", "")
        mapping[relative] = int(game_id)
    return mapping


def apply_game_id_corrections(
    fixture_rows: list[dict],
    url_to_match_id: dict[str, int],
) -> int:
    """Replace kickoff-order game_ids with nrl.com's own matchIds.

    Finals fixtures are numbered by bracket position, not kickoff order, so
    the reconstructed id can differ there; the match centre matchId is
    authoritative and uses the same year|111|round|game|0 scheme.
    """
    corrected = 0
    for row in fixture_rows:
        url = row.get("match_centre_url")
        match_id = url_to_match_id.get(url) if url else None
        if match_id is None:
            continue
        if int(float(row["game_id"])) != int(match_id):
            row["game_id"] = float(match_id)
            row["game_number"] = float(str(int(match_id))[-2])
            corrected += 1
    return corrected


def _crowd_by_game(con: sqlite3.Connection, season: int) -> dict[int, float]:
    return {
        int(row[0]): float(row[1])
        for row in con.execute(
            "SELECT game_id, attendance FROM match_context "
            "WHERE competition_year = ? AND attendance IS NOT NULL",
            (int(season),),
        )
    }


def rebuild_derived_caches(
    con: sqlite3.Connection,
    fixture_rows: list[dict],
    bye_rows: list[dict],
    season: int,
    min_writable_year: int,
) -> None:
    scoring = load_game_scoring(con, season)
    ladder_rows = build_season_ladder(fixture_rows, bye_rows, season, scoring)
    written = replace_cache_year(
        con, "feed_cache_ladders", season, ladder_rows, min_writable_year
    )
    _print(f"Rebuilt ladder cache for {season}: {written} rows.")

    performance_rows = build_season_performance(
        fixture_rows,
        bye_rows,
        season,
        load_team_stats_by_game(con, season),
        load_player_sums_by_game(con, season),
    )
    written = replace_cache_year(
        con, "feed_cache_performance", season, performance_rows, min_writable_year
    )
    _print(f"Rebuilt performance cache for {season}: {written} rows.")


def refresh_season(
    db_path: str | Path,
    season: int | None = None,
    config: FetchConfig | None = None,
    venue_csv: str | Path | None = None,
    max_pages: int | None = None,
) -> dict:
    """Refresh the current season from nrl.com. Returns run summary."""
    config = config or FetchConfig()
    season = season or dt.date.today().year
    started = store.utc_now_iso()

    con = sqlite3.connect(str(db_path))
    try:
        store.ensure_tables(con)
        store.load_venue_locations(con, venue_csv or DEFAULT_VENUE_CSV)
        venue_tz = load_venue_timezones(venue_csv or DEFAULT_VENUE_CSV)

        session = build_session()
        fixture_rows, bye_rows = fetch_season_draw(session, config, season, venue_tz)
        if not fixture_rows:
            store.record_ingest_run(
                con, "refresh", started, "no_fixtures", season, season, 0, []
            )
            con.commit()
            _print(f"No fixtures returned for {season}; nothing to refresh.")
            return {"status": "no_fixtures", "season": season}

        pages, errors, url_to_match_id = _fetch_match_centres(
            con=con,
            session=session,
            config=config,
            fixture_rows=fixture_rows,
            season=season,
            max_pages=max_pages,
        )

        # finals fixtures are numbered by bracket, not kickoff order; correct
        # ids from this run's match centres plus previously stored ones
        combined_ids = _stored_match_ids_by_url(con)
        combined_ids.update(url_to_match_id)
        corrected = apply_game_id_corrections(fixture_rows, combined_ids)
        if corrected:
            _print(f"Corrected {corrected} fixture game_id(s) from match centre matchIds.")

        # carry over columns owned by other writers (odds, crowd, broadcast)
        extras = _existing_fixture_extras(con, season)
        crowd = _crowd_by_game(con, season)
        for row in fixture_rows:
            game_id = int(float(row["game_id"]))
            for key, value in extras.get(game_id, {}).items():
                if row.get(key) is None:
                    row[key] = value
            if row.get("crowd") is None and game_id in crowd:
                row["crowd"] = crowd[game_id]

        cache_rows = [
            {key: row.get(key) for key in FIXTURE_CACHE_COLUMNS} for row in fixture_rows
        ]
        written = replace_cache_year(
            con, "feed_cache_fixtures", season, cache_rows, season
        )
        _print(f"Rebuilt fixture cache for {season}: {written} rows.")

        rebuild_derived_caches(con, fixture_rows, bye_rows, season, season)

        status = "completed" if not errors else "completed_with_errors"
        store.record_ingest_run(
            con, "refresh", started, status, season, season, pages, errors
        )
        con.commit()
        _print(
            f"Refresh complete: {len(fixture_rows)} fixtures, "
            f"{pages} match centre pages, {len(errors)} errors."
        )
        return {
            "status": status,
            "season": season,
            "fixtures": len(fixture_rows),
            "pages": pages,
            "errors": errors,
        }
    finally:
        con.close()


def backfill_seasons(
    db_path: str | Path,
    start_year: int,
    end_year: int,
    config: FetchConfig | None = None,
    venue_csv: str | Path | None = None,
    max_pages: int | None = None,
) -> dict:
    """Fetch historical match centres into match_* tables (no cache rewrites)."""
    config = config or FetchConfig()
    start_year = max(int(start_year), BACKFILL_FIRST_SEASON)
    end_year = int(end_year)
    started = store.utc_now_iso()

    con = sqlite3.connect(str(db_path))
    try:
        store.ensure_tables(con)
        store.load_venue_locations(con, venue_csv or DEFAULT_VENUE_CSV)
        venue_tz = load_venue_timezones(venue_csv or DEFAULT_VENUE_CSV)

        session = build_session()
        total_pages = 0
        all_errors: list[str] = []
        for season in range(start_year, end_year + 1):
            fixture_rows, _ = fetch_season_draw(session, config, season, venue_tz)
            _print(f"Backfill {season}: {len(fixture_rows)} fixtures in draw.")
            remaining = None if max_pages is None else max_pages - total_pages
            if remaining is not None and remaining <= 0:
                break
            pages, errors, _ = _fetch_match_centres(
                con=con,
                session=session,
                config=config,
                fixture_rows=fixture_rows,
                season=season,
                include_upcoming=False,
                max_pages=remaining,
            )
            total_pages += pages
            all_errors.extend(errors)
            con.commit()
            _print(f"Backfill {season}: fetched {pages} match centre pages.")

        status = "completed" if not all_errors else "completed_with_errors"
        store.record_ingest_run(
            con,
            "backfill",
            started,
            status,
            start_year,
            end_year,
            total_pages,
            all_errors,
        )
        con.commit()
        return {
            "status": status,
            "pages": total_pages,
            "errors": all_errors,
        }
    finally:
        con.close()


def fetch_single_round(
    db_path: str | Path,
    season: int,
    round_id: int,
    config: FetchConfig | None = None,
    venue_csv: str | Path | None = None,
) -> list[dict]:
    """Diagnostic helper: fixture rows for one round (no writes)."""
    config = config or FetchConfig()
    venue_tz = load_venue_timezones(venue_csv or DEFAULT_VENUE_CSV)
    session = build_session()
    payload = fetch_round_draw(session, config, season, round_id)
    if not payload:
        return []
    return draw_to_fixture_rows(payload, season, venue_tz)
