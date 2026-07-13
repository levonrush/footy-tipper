"""SQLite storage for match centre ingestion.

Team stats are long (stat titles vary by era); player stats are wide (stable
key set since 2012, with ALTER TABLE fallback for future additions), matching
the auto-evolving column behaviour of the prepared tables in db-write.R.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path

from ..lineups.normalization import normalize_player_name

PLAYER_STAT_COLUMNS = [
    "all_run_metres",
    "all_runs",
    "bomb_kicks",
    "conversion_attempts",
    "conversions",
    "cross_field_kicks",
    "dummy_half_run_metres",
    "dummy_half_runs",
    "dummy_passes",
    "errors",
    "fantasy_points_total",
    "field_goals",
    "forced_drop_out_kicks",
    "forty_twenty_kicks",
    "goal_conversion_rate",
    "goals",
    "grubber_kicks",
    "handling_errors",
    "hit_up_run_metres",
    "hit_ups",
    "ineffective_tackles",
    "intercepts",
    "kick_metres",
    "kick_return_metres",
    "kicks",
    "kicks_dead",
    "kicks_defused",
    "line_break_assists",
    "line_breaks",
    "line_engaged_runs",
    "minutes_played",
    "missed_tackles",
    "offloads",
    "offside_within_ten_metres",
    "on_report",
    "one_on_one_lost",
    "one_on_one_steal",
    "one_point_field_goals",
    "passes",
    "passes_to_run_ratio",
    "penalties",
    "penalty_goals",
    "play_the_ball_average_speed",
    "play_the_ball_total",
    "points",
    "post_contact_metres",
    "receipts",
    "ruck_infringements",
    "send_offs",
    "sin_bins",
    "stint_one",
    "tackle_breaks",
    "tackle_efficiency",
    "tackles_made",
    "tries",
    "try_assists",
    "twenty_forty_kicks",
    "two_point_field_goals",
]

_PLAYER_IDENTITY_KEYS = {
    "side",
    "player_id",
    "player_name",
    "jersey_number",
    "position",
}


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def ensure_tables(con: sqlite3.Connection) -> None:
    player_stat_cols = ",\n            ".join(
        f"{col} REAL" for col in PLAYER_STAT_COLUMNS
    )
    con.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS match_team_stats (
            game_id INTEGER NOT NULL,
            competition_year INTEGER,
            round_id INTEGER,
            team TEXT,
            side TEXT NOT NULL,
            stat_name TEXT NOT NULL,
            value REAL,
            source_url TEXT,
            ingested_at_utc TEXT,
            PRIMARY KEY (game_id, side, stat_name)
        );
        CREATE TABLE IF NOT EXISTS match_player_stats (
            game_id INTEGER NOT NULL,
            competition_year INTEGER,
            round_id INTEGER,
            team TEXT,
            side TEXT NOT NULL,
            player_id INTEGER NOT NULL,
            player_key TEXT,
            player_name TEXT,
            jersey_number INTEGER,
            position TEXT,
            {player_stat_cols},
            ingested_at_utc TEXT,
            PRIMARY KEY (game_id, side, player_id)
        );
        CREATE TABLE IF NOT EXISTS match_context (
            game_id INTEGER PRIMARY KEY,
            competition_year INTEGER,
            round_id INTEGER,
            weather_label TEXT,
            ground_condition TEXT,
            attendance INTEGER,
            source_url TEXT,
            ingested_at_utc TEXT
        );
        CREATE TABLE IF NOT EXISTS match_officials (
            game_id INTEGER NOT NULL,
            competition_year INTEGER,
            round_id INTEGER,
            role TEXT NOT NULL,
            official_name TEXT NOT NULL,
            profile_id INTEGER,
            ingested_at_utc TEXT,
            PRIMARY KEY (game_id, role, official_name)
        );
        CREATE TABLE IF NOT EXISTS nrl_ingest_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mode TEXT,
            started_at_utc TEXT,
            completed_at_utc TEXT,
            status TEXT,
            requested_start_year INTEGER,
            requested_end_year INTEGER,
            pages_fetched INTEGER,
            errors_json TEXT
        );
        CREATE TABLE IF NOT EXISTS venue_locations (
            venue_name TEXT PRIMARY KEY,
            city TEXT,
            latitude REAL,
            longitude REAL,
            timezone TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_match_player_stats_player
            ON match_player_stats (player_id, competition_year, round_id);
        CREATE INDEX IF NOT EXISTS idx_match_team_stats_team
            ON match_team_stats (team, competition_year, round_id);
        """
    )


def _table_columns(con: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in con.execute(f"PRAGMA table_info({table})")}


def _ensure_stat_columns(con: sqlite3.Connection, rows: list[dict]) -> list[str]:
    existing = _table_columns(con, "match_player_stats")
    stat_keys: set[str] = set()
    for row in rows:
        stat_keys.update(
            key for key in row if key not in _PLAYER_IDENTITY_KEYS
        )
    for key in sorted(stat_keys - existing):
        con.execute(f"ALTER TABLE match_player_stats ADD COLUMN {key} REAL")
        existing.add(key)
    return sorted(stat_keys & existing)


def load_venue_locations(con: sqlite3.Connection, csv_path: str | Path) -> int:
    import csv as csv_module

    path = Path(csv_path)
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv_module.DictReader(handle))
    con.executemany(
        """
        INSERT INTO venue_locations (venue_name, city, latitude, longitude, timezone)
        VALUES (:venue_name, :city, :latitude, :longitude, :timezone)
        ON CONFLICT(venue_name) DO UPDATE SET
            city = excluded.city,
            latitude = excluded.latitude,
            longitude = excluded.longitude,
            timezone = excluded.timezone
        """,
        rows,
    )
    return len(rows)


def upsert_match_bundle(
    con: sqlite3.Connection,
    bundle: dict,
    competition_year: int,
    round_id: int,
    team_home: str | None,
    team_away: str | None,
) -> None:
    """Write one parsed match centre bundle (idempotent per game)."""
    game_id = bundle["game_id"]
    now = utc_now_iso()
    source_url = bundle.get("source_url") or ""
    team_by_side = {"home": team_home, "away": team_away}

    if bundle["team_stats"]:
        con.execute("DELETE FROM match_team_stats WHERE game_id = ?", (game_id,))
        con.executemany(
            """
            INSERT INTO match_team_stats
                (game_id, competition_year, round_id, team, side, stat_name,
                 value, source_url, ingested_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    game_id,
                    competition_year,
                    round_id,
                    team_by_side.get(stat["side"]),
                    stat["side"],
                    stat["stat_name"],
                    stat["value"],
                    source_url,
                    now,
                )
                for stat in bundle["team_stats"]
            ],
        )

    player_rows = bundle["player_stats"]
    if player_rows:
        stat_columns = _ensure_stat_columns(con, player_rows)
        con.execute("DELETE FROM match_player_stats WHERE game_id = ?", (game_id,))
        columns = [
            "game_id",
            "competition_year",
            "round_id",
            "team",
            "side",
            "player_id",
            "player_key",
            "player_name",
            "jersey_number",
            "position",
            *stat_columns,
            "ingested_at_utc",
        ]
        placeholders = ", ".join("?" for _ in columns)
        con.executemany(
            f"INSERT INTO match_player_stats ({', '.join(columns)}) "
            f"VALUES ({placeholders})",
            [
                (
                    game_id,
                    competition_year,
                    round_id,
                    team_by_side.get(row["side"]),
                    row["side"],
                    row["player_id"],
                    normalize_player_name(row.get("player_name")),
                    row.get("player_name"),
                    row.get("jersey_number"),
                    row.get("position"),
                    *(row.get(col) for col in stat_columns),
                    now,
                )
                for row in player_rows
            ],
        )

    context = bundle.get("context") or {}
    if any(value is not None for value in context.values()):
        con.execute(
            """
            INSERT INTO match_context
                (game_id, competition_year, round_id, weather_label,
                 ground_condition, attendance, source_url, ingested_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(game_id) DO UPDATE SET
                weather_label = excluded.weather_label,
                ground_condition = excluded.ground_condition,
                attendance = excluded.attendance,
                source_url = excluded.source_url,
                ingested_at_utc = excluded.ingested_at_utc
            """,
            (
                game_id,
                competition_year,
                round_id,
                context.get("weather_label"),
                context.get("ground_condition"),
                context.get("attendance"),
                source_url,
                now,
            ),
        )

    officials = bundle.get("officials") or []
    if officials:
        # Appointments change pre-game; replace the game's rows wholesale.
        con.execute("DELETE FROM match_officials WHERE game_id = ?", (game_id,))
        con.executemany(
            """
            INSERT INTO match_officials
                (game_id, competition_year, round_id, role, official_name,
                 profile_id, ingested_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    game_id,
                    competition_year,
                    round_id,
                    official["role"],
                    official["official_name"],
                    official.get("profile_id"),
                    now,
                )
                for official in officials
            ],
        )


def games_with_team_stats(con: sqlite3.Connection) -> set[int]:
    return {
        int(row[0])
        for row in con.execute("SELECT DISTINCT game_id FROM match_team_stats")
    }


def record_ingest_run(
    con: sqlite3.Connection,
    mode: str,
    started_at_utc: str,
    status: str,
    start_year: int | None,
    end_year: int | None,
    pages_fetched: int,
    errors: list[str],
) -> None:
    con.execute(
        """
        INSERT INTO nrl_ingest_runs
            (mode, started_at_utc, completed_at_utc, status,
             requested_start_year, requested_end_year, pages_fetched, errors_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            mode,
            started_at_utc,
            utc_now_iso(),
            status,
            start_year,
            end_year,
            pages_fetched,
            json.dumps(errors[:50]),
        ),
    )


def completed_backfill_exists(con: sqlite3.Connection) -> bool:
    row = con.execute(
        "SELECT COUNT(*) FROM nrl_ingest_runs "
        "WHERE mode = 'backfill' AND status = 'completed'"
    ).fetchone()
    return bool(row and row[0])
