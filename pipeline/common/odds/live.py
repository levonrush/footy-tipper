"""Shared fixture matching and persistence for live odds providers."""

from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

from ..nrl_data.cache_writer import update_fixture_odds
from . import store
from .team_names import canonical_team
from .validity import valid_price_pair, validated_market_values

MAX_KICKOFF_DELTA = dt.timedelta(hours=6)


def parse_utc_datetime(value: object) -> dt.datetime | None:
    """Parse provider ISO timestamps or fixture-cache Unix timestamps."""
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        parsed = value
    elif isinstance(value, (int, float)):
        try:
            parsed = dt.datetime.fromtimestamp(float(value), tz=dt.timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed = dt.datetime.fromtimestamp(float(text), tz=dt.timezone.utc)
            except (OverflowError, OSError, TypeError, ValueError):
                return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def pre_game_fixtures(con: sqlite3.Connection) -> list[dict]:
    rows: list[dict] = []
    for row in con.execute(
        "SELECT game_id, competition_year, round_id, team_home, team_away, "
        "start_time_utc FROM feed_cache_fixtures "
        "WHERE game_state_name = 'Pre Game'"
    ):
        home = canonical_team(row[3]) or row[3]
        away = canonical_team(row[4]) or row[4]
        rows.append(
            {
                "game_id": int(float(row[0])),
                "competition_year": int(float(row[1])),
                "round_id": int(float(row[2])),
                "team_home": home,
                "team_away": away,
                "kickoff": parse_utc_datetime(row[5]),
            }
        )
    return rows


def match_fixture(
    snapshot: dict,
    fixtures: list[dict],
    max_delta: dt.timedelta = MAX_KICKOFF_DELTA,
) -> dict | None:
    """Match canonical teams and the nearest kickoff inside *max_delta*."""
    snapshot_kickoff = parse_utc_datetime(snapshot.get("commence_time"))
    if snapshot_kickoff is None:
        return None

    candidates: list[tuple[float, int, dict]] = []
    for fixture in fixtures:
        if (
            fixture["team_home"] != snapshot.get("home")
            or fixture["team_away"] != snapshot.get("away")
            or fixture.get("kickoff") is None
        ):
            continue
        delta = abs((fixture["kickoff"] - snapshot_kickoff).total_seconds())
        if delta <= max_delta.total_seconds():
            candidates.append((delta, fixture["game_id"], fixture))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def fixture_odds_update(values: dict) -> dict:
    """Translate ledger field names to fixture-cache column names."""
    update = {
        "team_head_to_head_odds_home": values.get("h2h_odds_home"),
        "team_head_to_head_odds_away": values.get("h2h_odds_away"),
        "team_line_odds_home": values.get("line_odds_home"),
        "team_line_odds_away": values.get("line_odds_away"),
        "team_line_amount_home": values.get("line_amount_home"),
        "total_line": values.get("total_line"),
        "total_over_odds": values.get("total_over_odds"),
        "total_under_odds": values.get("total_under_odds"),
    }
    if values.get("line_amount_home") is not None:
        update["team_line_amount_away"] = -values["line_amount_home"]
    return {key: value for key, value in update.items() if value is not None}


def _latest_complete_live_values(
    con: sqlite3.Connection,
    game_id: int,
) -> dict:
    """Return the newest atomic live snapshot with a valid H2H pair."""
    fields = tuple(store._NUMERIC_FIELDS)
    rows = con.execute(
        f"""
        SELECT id, snapshot_time_utc, {", ".join(fields)}
        FROM odds_history
        WHERE game_id = ?
          AND source IN ('the_odds_api', 'betfair')
          AND snapshot_kind = 'live'
        """,
        (int(game_id),),
    ).fetchall()
    candidates: list[tuple[dt.datetime, int, dict]] = []
    oldest = dt.datetime.min.replace(tzinfo=dt.timezone.utc)
    for row in rows:
        values = dict(zip(fields, row[2:]))
        if not valid_price_pair(
            values.get("h2h_odds_home"),
            values.get("h2h_odds_away"),
        ):
            continue
        candidates.append(
            (
                parse_utc_datetime(row[1]) or oldest,
                int(row[0]),
                validated_market_values(values),
            )
        )
    if not candidates:
        return {}
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def persist_live_snapshots(
    db_path: str | Path,
    source: str,
    snapshots: list[dict],
    observed_at: str | None = None,
    exclude_game_ids: set[int] | None = None,
) -> dict:
    """Atomically retain matched observations and update current fixtures.

    Each snapshot has canonical ``home``/``away`` names, ``commence_time``,
    provider ``values`` and optional non-secret ``raw_meta``.
    """
    con = sqlite3.connect(str(db_path))
    try:
        store.ensure_tables(con)
        fixtures = pre_game_fixtures(con)
        default_snapshot_time = observed_at or store.utc_now_iso()
        matched = 0
        inserted = 0
        h2h = 0
        line = 0
        totals = 0
        matched_game_ids: set[int] = set()
        excluded = exclude_game_ids or set()

        for snapshot in snapshots:
            fixture = match_fixture(snapshot, fixtures)
            if (
                fixture is None
                or fixture["game_id"] in excluded
                or fixture["game_id"] in matched_game_ids
            ):
                continue
            values = validated_market_values(snapshot.get("values") or {})
            if not values:
                continue
            snapshot_time = (
                snapshot.get("snapshot_time_utc") or default_snapshot_time
            )

            matched_game_ids.add(fixture["game_id"])
            matched += 1
            raw_meta = dict(snapshot.get("raw_meta") or {})
            raw_meta.setdefault("fetched_at_utc", default_snapshot_time)
            _, created = store.upsert_live_snapshot(
                con,
                fixture["game_id"],
                fixture["competition_year"],
                fixture["round_id"],
                source=source,
                snapshot_time_utc=snapshot_time,
                values=values,
                raw_meta=raw_meta,
            )
            inserted += int(created)
            # Rehydrate from the ledger's newest complete H2H observation.
            # This makes repeated quote times and out-of-order provider
            # responses unable to move the fixture cache away from the row
            # that the freshness gate and inference will select.
            update = fixture_odds_update(
                _latest_complete_live_values(con, fixture["game_id"])
            )
            if update:
                # Only complete market families enter `update`; absent or
                # partial provider fields therefore cannot erase valid cache
                # values from an earlier source.
                update_fixture_odds(con, fixture["game_id"], update)
            h2h += int("h2h_odds_home" in values)
            line += int("line_amount_home" in values)
            totals += int("total_line" in values)

        con.commit()
        return {
            "games_updated": matched,
            "snapshots_inserted": inserted,
            "h2h_games": h2h,
            "line_games": line,
            "totals_games": totals,
            "fixture_count": len(fixtures),
            "game_ids_updated": tuple(sorted(matched_game_ids)),
        }
    finally:
        con.close()
