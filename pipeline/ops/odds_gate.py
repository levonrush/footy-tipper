"""Read-only live-odds coverage checks for prediction delivery modes."""

from __future__ import annotations

import datetime as dt
import pathlib
import sqlite3
from dataclasses import dataclass

from pipeline.common.odds.validity import finite_number, valid_price_pair


LIVE_SOURCES = ("the_odds_api", "betfair")
DEFAULT_MAX_AGE_HOURS = 6.0


def _parse_timestamp(value) -> dt.datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


@dataclass(frozen=True)
class OddsCoverage:
    competition_year: int | None
    round_id: int | None
    round_name: str | None
    total_games: int
    covered_games: int
    missing_game_ids: tuple[int, ...] = ()
    stale_game_ids: tuple[int, ...] = ()
    fresh_game_ids: tuple[int, ...] = ()
    fresh_line_game_ids: tuple[int, ...] = ()
    fresh_total_game_ids: tuple[int, ...] = ()
    no_fixtures: bool = False
    error: str | None = None

    @property
    def complete(self) -> bool:
        if self.error:
            return False
        return self.no_fixtures or (
            self.total_games > 0 and self.covered_games == self.total_games
        )

    @property
    def label(self) -> str:
        if self.no_fixtures:
            return "no upcoming round"
        if self.round_name and self.competition_year:
            return f"{self.round_name} {self.competition_year}"
        return "current round"

    def message(self) -> str:
        if self.error:
            return f"Live odds coverage unavailable: {self.error}"
        if self.no_fixtures:
            return "No pre-game fixtures found; live odds coverage is not required."
        detail = (
            f"Fresh H2H odds coverage for {self.label}: "
            f"{self.covered_games}/{self.total_games}."
        )
        problems = []
        if self.missing_game_ids:
            problems.append(
                "missing games " + ", ".join(str(value) for value in self.missing_game_ids)
            )
        if self.stale_game_ids:
            problems.append(
                "stale games " + ", ".join(str(value) for value in self.stale_game_ids)
            )
        if problems:
            detail += " " + "; ".join(problems) + "."
        return detail


def current_round_odds_coverage(
    db_path: str | pathlib.Path,
    *,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
    now: dt.datetime | None = None,
) -> OddsCoverage:
    """Return fresh, paired H2H coverage for the next pre-game round."""
    path = pathlib.Path(db_path)
    if not path.exists():
        return OddsCoverage(
            None,
            None,
            None,
            0,
            0,
            error=f"runtime database does not exist at {path}",
        )

    now_utc = now or dt.datetime.now(dt.timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=dt.timezone.utc)
    now_utc = now_utc.astimezone(dt.timezone.utc)
    oldest_allowed = now_utc - dt.timedelta(hours=float(max_age_hours))
    newest_allowed = now_utc + dt.timedelta(minutes=5)

    try:
        with sqlite3.connect(str(path)) as con:
            table_names = {
                str(row[0])
                for row in con.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            if "feed_cache_fixtures" not in table_names:
                return OddsCoverage(
                    None,
                    None,
                    None,
                    0,
                    0,
                    error="feed_cache_fixtures is unavailable",
                )

            round_row = con.execute(
                """
                WITH latest_year AS (
                    SELECT MAX(CAST(competition_year AS INTEGER)) AS competition_year
                    FROM feed_cache_fixtures
                    WHERE game_state_name = 'Pre Game'
                ),
                next_round AS (
                    SELECT MIN(CAST(round_id AS INTEGER)) AS round_id
                    FROM feed_cache_fixtures
                    WHERE game_state_name = 'Pre Game'
                      AND CAST(competition_year AS INTEGER) =
                          (SELECT competition_year FROM latest_year)
                )
                SELECT
                    (SELECT competition_year FROM latest_year),
                    (SELECT round_id FROM next_round)
                """
            ).fetchone()
            if not round_row or round_row[0] is None or round_row[1] is None:
                return OddsCoverage(None, None, None, 0, 0, no_fixtures=True)

            competition_year = int(round_row[0])
            round_id = int(round_row[1])
            fixtures = con.execute(
                """
                SELECT
                    CAST(game_id AS INTEGER),
                    round_name
                FROM feed_cache_fixtures
                WHERE game_state_name = 'Pre Game'
                  AND CAST(competition_year AS INTEGER) = ?
                  AND CAST(round_id AS INTEGER) = ?
                ORDER BY CAST(start_time AS REAL), CAST(game_number AS REAL),
                         CAST(game_id AS INTEGER)
                """,
                (competition_year, round_id),
            ).fetchall()
            if not fixtures:
                return OddsCoverage(
                    competition_year,
                    round_id,
                    None,
                    0,
                    0,
                    no_fixtures=True,
                )

            round_name = str(fixtures[0][1] or f"Round {round_id}")
            game_ids = tuple(int(row[0]) for row in fixtures)
            if "odds_history" not in table_names:
                return OddsCoverage(
                    competition_year,
                    round_id,
                    round_name,
                    len(game_ids),
                    0,
                    missing_game_ids=game_ids,
                )

            placeholders = ", ".join("?" for _ in game_ids)
            source_placeholders = ", ".join("?" for _ in LIVE_SOURCES)
            rows = con.execute(
                f"""
                SELECT
                    CAST(game_id AS INTEGER),
                    snapshot_time_utc,
                    h2h_odds_home,
                    h2h_odds_away,
                    line_amount_home,
                    line_odds_home,
                    line_odds_away,
                    total_line,
                    total_over_odds,
                    total_under_odds,
                    id
                FROM odds_history
                WHERE snapshot_kind = 'live'
                  AND source IN ({source_placeholders})
                  AND CAST(game_id AS INTEGER) IN ({placeholders})
                ORDER BY id DESC
                """,
                (*LIVE_SOURCES, *game_ids),
            ).fetchall()
    except (OSError, sqlite3.Error, ValueError) as exc:
        return OddsCoverage(
            None,
            None,
            None,
            0,
            0,
            error=str(exc),
        )

    # Keep the latest complete H2H observation as one atomic bookmaker
    # snapshot. Spread/totals count as fresh only when that same selected
    # snapshot contains the complete family; do not splice a newer H2H quote
    # together with still-fresh lines from another bookmaker or fetch.
    latest_h2h: dict[
        int,
        tuple[dt.datetime | None, bool, bool, bool, int],
    ] = {}

    for (
        game_id,
        snapshot_time,
        home_odds,
        away_odds,
        line_amount,
        line_home_odds,
        line_away_odds,
        total_line,
        total_over_odds,
        total_under_odds,
        _row_id,
    ) in rows:
        game_id = int(game_id)
        parsed = _parse_timestamp(snapshot_time)
        line_complete = finite_number(line_amount) and valid_price_pair(
            line_home_odds, line_away_odds
        )
        total_complete = (
            finite_number(total_line)
            and float(total_line) > 0.0
            and valid_price_pair(total_over_odds, total_under_odds)
        )
        if not valid_price_pair(home_odds, away_odds):
            continue
        existing = latest_h2h.get(game_id)
        if existing is not None:
            existing_time, _, _, _, existing_row_id = existing
            if existing_time is not None and (
                parsed is None
                or parsed < existing_time
                or (parsed == existing_time and int(_row_id) <= existing_row_id)
            ):
                continue
        latest_h2h[game_id] = (
            parsed,
            bool(parsed and oldest_allowed <= parsed <= newest_allowed),
            bool(line_complete),
            bool(total_complete),
            int(_row_id),
        )

    missing = tuple(game_id for game_id in game_ids if game_id not in latest_h2h)
    stale = tuple(
        game_id
        for game_id in game_ids
        if game_id in latest_h2h and not latest_h2h[game_id][1]
    )
    covered = len(game_ids) - len(missing) - len(stale)
    fresh = tuple(
        game_id
        for game_id in game_ids
        if game_id not in missing and game_id not in stale
    )
    fresh_line = tuple(
        game_id
        for game_id in game_ids
        if game_id in latest_h2h
        and latest_h2h[game_id][1]
        and latest_h2h[game_id][2]
    )
    fresh_total = tuple(
        game_id
        for game_id in game_ids
        if game_id in latest_h2h
        and latest_h2h[game_id][1]
        and latest_h2h[game_id][3]
    )
    return OddsCoverage(
        competition_year,
        round_id,
        round_name,
        len(game_ids),
        covered,
        missing_game_ids=missing,
        stale_game_ids=stale,
        fresh_game_ids=fresh,
        fresh_line_game_ids=fresh_line,
        fresh_total_game_ids=fresh_total,
    )
