"""SQLite storage for odds history (backfill + live snapshots).

`odds_history` keeps every observation with its source and timing kind so the
movement features (open vs latest) and any later open-vs-close experiments
have raw material. The `odds_snapshots` ledger written by R remains the
prediction-time observation contract.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def ensure_tables(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS odds_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id INTEGER NOT NULL,
            competition_year INTEGER,
            round_id INTEGER,
            source TEXT NOT NULL,
            snapshot_kind TEXT NOT NULL,
            snapshot_time_utc TEXT,
            h2h_odds_home REAL,
            h2h_odds_away REAL,
            h2h_odds_home_min REAL,
            h2h_odds_home_max REAL,
            h2h_odds_away_min REAL,
            h2h_odds_away_max REAL,
            line_amount_home REAL,
            line_odds_home REAL,
            line_odds_away REAL,
            total_line REAL,
            total_over_odds REAL,
            total_under_odds REAL,
            raw_meta TEXT,
            UNIQUE (game_id, source, snapshot_kind, snapshot_time_utc)
        );
        CREATE INDEX IF NOT EXISTS idx_odds_history_game
            ON odds_history (game_id, source, snapshot_kind);
        """
    )


_NUMERIC_FIELDS = [
    "h2h_odds_home",
    "h2h_odds_away",
    "h2h_odds_home_min",
    "h2h_odds_home_max",
    "h2h_odds_away_min",
    "h2h_odds_away_max",
    "line_amount_home",
    "line_odds_home",
    "line_odds_away",
    "total_line",
    "total_over_odds",
    "total_under_odds",
]


def insert_snapshot(
    con: sqlite3.Connection,
    game_id: int,
    competition_year: int | None,
    round_id: int | None,
    source: str,
    snapshot_kind: str,
    snapshot_time_utc: str | None,
    values: dict,
    raw_meta: dict | None = None,
) -> bool:
    """Insert one odds observation; returns False when it already exists."""
    row = [
        int(game_id),
        competition_year,
        round_id,
        source,
        snapshot_kind,
        snapshot_time_utc,
    ]
    row.extend(values.get(field) for field in _NUMERIC_FIELDS)
    row.append(json.dumps(raw_meta) if raw_meta else None)
    columns = (
        "game_id, competition_year, round_id, source, snapshot_kind, "
        "snapshot_time_utc, " + ", ".join(_NUMERIC_FIELDS) + ", raw_meta"
    )
    placeholders = ", ".join("?" for _ in range(len(row)))
    cursor = con.execute(
        f"INSERT OR IGNORE INTO odds_history ({columns}) VALUES ({placeholders})",
        row,
    )
    return cursor.rowcount > 0


def latest_live_snapshots(con: sqlite3.Connection) -> dict[int, dict]:
    """Most recent betfair live observation per game."""
    fields = ", ".join(_NUMERIC_FIELDS)
    result: dict[int, dict] = {}
    for row in con.execute(
        f"""
        SELECT game_id, {fields}
        FROM odds_history
        WHERE source = 'betfair' AND snapshot_kind = 'live'
          AND id IN (
              SELECT MAX(id) FROM odds_history
              WHERE source = 'betfair' AND snapshot_kind = 'live'
              GROUP BY game_id
          )
        """
    ):
        result[int(row[0])] = {
            field: row[i + 1]
            for i, field in enumerate(_NUMERIC_FIELDS)
            if row[i + 1] is not None
        }
    return result
