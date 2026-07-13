"""Parity validation: derived nrl.com data vs the cached feed tables.

Builds fixture/ladder/performance frames in memory for overlap seasons and
reconciles them column-by-column against feed_cache_* history. Nothing is
written to the caches; the output is a CSV report plus promotion-gate
booleans.

Gates (from the migration plan):
- fixtures: 100% key coverage and exact match on scores/state/kickoffs
- ladders: >= 99.5% on core numeric columns
- performance: per-column classification derivable / scaled / non-derivable
"""

from __future__ import annotations

import csv
import datetime as dt
import sqlite3
from collections import defaultdict
from pathlib import Path

from .draw import fetch_season_draw, load_venue_timezones
from .ladder import LADDER_COLUMNS, build_season_ladder
from .performance import (
    build_season_performance,
    load_game_scoring,
    load_player_sums_by_game,
    load_team_stats_by_game,
)
from .refresh import (
    DEFAULT_VENUE_CSV,
    _stored_match_ids_by_url,
    apply_game_id_corrections,
)
from .web import FetchConfig, build_session

NUMERIC_TOLERANCE = 0.051  # one-decimal rounding differences

FIXTURE_COMPARE_COLUMNS = [
    "round_id",
    "round_name",
    "game_number",
    "game_state_name",
    "start_time",
    "start_time_utc",
    "venue_name",
    "team_home",
    "team_away",
    "team_final_score_home",
    "team_final_score_away",
]

LADDER_CORE_COLUMNS = [
    "position",
    "wins",
    "draws",
    "losses",
    "byes",
    "competition_points",
    "points_for",
    "points_against",
    "points_difference",
    "home_wins",
    "home_losses",
    "away_wins",
    "away_losses",
    "close_games",
    "average_winning_margin",
    "average_losing_margin",
]


def _is_number(value) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _values_match(expected, actual) -> bool:
    if expected is None and actual is None:
        return True
    if _is_number(expected) and _is_number(actual):
        return abs(float(expected) - float(actual)) <= NUMERIC_TOLERANCE
    if expected is None or actual is None:
        return False
    return str(expected).strip() == str(actual).strip()


class ColumnTally:
    def __init__(self) -> None:
        self.compared = 0
        self.matched = 0
        self.expected_present = 0
        self.actual_present = 0

    def add(self, expected, actual) -> None:
        if expected is not None:
            self.expected_present += 1
        if actual is not None:
            self.actual_present += 1
        if expected is None and actual is None:
            return
        self.compared += 1
        if _values_match(expected, actual):
            self.matched += 1

    @property
    def match_rate(self) -> float:
        return self.matched / self.compared if self.compared else 0.0

    def classification(self) -> str:
        if self.expected_present and not self.actual_present:
            return "non-derivable"
        if self.compared == 0:
            return "no-data"
        if self.match_rate >= 0.995:
            return "derivable"
        if self.match_rate >= 0.5:
            return "close"
        return "mismatch"


def _load_cached(
    con: sqlite3.Connection, table: str, season: int
) -> list[dict]:
    cursor = con.execute(
        f"SELECT * FROM {table} WHERE CAST(competition_year AS INTEGER) = ?",
        (int(season),),
    )
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _compare_rows(
    expected_rows: dict,
    actual_rows: dict,
    columns: list[str],
    tallies: dict[str, ColumnTally],
    coverage: ColumnTally,
) -> None:
    for key, expected in expected_rows.items():
        actual = actual_rows.get(key)
        coverage.add(True, True if actual is not None else None)
        if actual is None:
            continue
        for column in columns:
            tallies[column].add(expected.get(column), actual.get(column))


def validate_seasons(
    db_path: str | Path,
    start_year: int,
    end_year: int,
    report_path: str | Path | None = None,
    config: FetchConfig | None = None,
    venue_csv: str | Path | None = None,
) -> dict:
    config = config or FetchConfig()
    venue_tz = load_venue_timezones(venue_csv or DEFAULT_VENUE_CSV)
    session = build_session()

    con = sqlite3.connect(str(db_path))
    fixture_tallies: dict[str, ColumnTally] = defaultdict(ColumnTally)
    ladder_tallies: dict[str, ColumnTally] = defaultdict(ColumnTally)
    perf_tallies: dict[str, ColumnTally] = defaultdict(ColumnTally)
    fixture_coverage = ColumnTally()
    ladder_coverage = ColumnTally()
    perf_coverage = ColumnTally()

    try:
        stored_match_ids = _stored_match_ids_by_url(con)
        for season in range(int(start_year), int(end_year) + 1):
            fixture_rows, bye_rows = fetch_season_draw(
                session, config, season, venue_tz
            )
            if not fixture_rows:
                print(f"[nrl-data] validate: no draw data for {season}, skipping.")
                continue
            apply_game_id_corrections(fixture_rows, stored_match_ids)

            cached_fixtures = {
                int(float(row["game_id"])): row
                for row in _load_cached(con, "feed_cache_fixtures", season)
            }
            derived_fixtures = {
                int(float(row["game_id"])): row for row in fixture_rows
            }
            _compare_rows(
                cached_fixtures,
                derived_fixtures,
                FIXTURE_COMPARE_COLUMNS,
                fixture_tallies,
                fixture_coverage,
            )

            scoring = load_game_scoring(con, season)
            derived_ladder = {
                (row["round_id"], row["team"]): row
                for row in build_season_ladder(fixture_rows, bye_rows, season, scoring)
            }
            cached_ladder = {
                (int(float(row["round_id"])), row["team"]): row
                for row in _load_cached(con, "feed_cache_ladders", season)
            }
            _compare_rows(
                cached_ladder,
                derived_ladder,
                [col for col in LADDER_COLUMNS if col not in ("team", "round_id", "competition_year")],
                ladder_tallies,
                ladder_coverage,
            )

            derived_perf = {
                (row["round_id"], row["team"]): row
                for row in build_season_performance(
                    fixture_rows,
                    bye_rows,
                    season,
                    load_team_stats_by_game(con, season),
                    load_player_sums_by_game(con, season),
                )
            }
            cached_perf_rows = _load_cached(con, "feed_cache_performance", season)
            cached_perf = {
                (int(float(row["round_id"])), row["team"]): row
                for row in cached_perf_rows
            }
            perf_columns = sorted(
                {
                    column
                    for row in cached_perf_rows
                    for column in row
                    if column not in ("team", "round_id", "competition_year", "season_id", "round_number")
                }
            )
            _compare_rows(
                cached_perf, derived_perf, perf_columns, perf_tallies, perf_coverage
            )
            print(f"[nrl-data] validate: compared season {season}.")
    finally:
        con.close()

    gates = {
        "fixtures_keys_pct": round(fixture_coverage.match_rate * 100, 2),
        "fixtures_pass": fixture_coverage.match_rate == 1.0
        and all(
            fixture_tallies[col].match_rate == 1.0
            for col in (
                "team_final_score_home",
                "team_final_score_away",
                "game_state_name",
                "start_time_utc",
            )
            if fixture_tallies[col].compared
        ),
        "ladder_core_pct": round(
            100
            * (
                sum(ladder_tallies[c].matched for c in LADDER_CORE_COLUMNS)
                / max(1, sum(ladder_tallies[c].compared for c in LADDER_CORE_COLUMNS))
            ),
            2,
        ),
    }
    gates["ladder_pass"] = gates["ladder_core_pct"] >= 99.5
    gates["performance_non_derivable"] = sorted(
        column
        for column, tally in perf_tallies.items()
        if tally.classification() == "non-derivable"
    )

    if report_path is None:
        stamp = dt.date.today().isoformat()
        report_path = Path("reports") / f"nrl_data_parity_{stamp}.csv"
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["table", "column", "compared", "matched", "match_rate", "classification"]
        )
        for table, tallies in (
            ("fixtures", fixture_tallies),
            ("ladders", ladder_tallies),
            ("performance", perf_tallies),
        ):
            for column in sorted(tallies):
                tally = tallies[column]
                writer.writerow(
                    [
                        table,
                        column,
                        tally.compared,
                        tally.matched,
                        round(tally.match_rate, 4),
                        tally.classification(),
                    ]
                )
    gates["report_path"] = str(report_path)
    print(f"[nrl-data] validate: report written to {report_path}")
    print(
        f"[nrl-data] validate: fixture keys {gates['fixtures_keys_pct']}%, "
        f"fixtures_pass={gates['fixtures_pass']}, "
        f"ladder core {gates['ladder_core_pct']}% (pass={gates['ladder_pass']})"
    )
    return gates
