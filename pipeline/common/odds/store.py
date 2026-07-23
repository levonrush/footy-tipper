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
        # SQLite UNIQUE treats NULLs as distinct; un-timestamped workbook
        # facts use '' so open/close rows deduplicate per game+source+kind
        snapshot_time_utc or "",
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


def upsert_live_snapshot(
    con: sqlite3.Connection,
    game_id: int,
    competition_year: int | None,
    round_id: int | None,
    source: str,
    snapshot_time_utc: str | None,
    values: dict,
    raw_meta: dict | None = None,
) -> tuple[int, bool]:
    """Store the exact provider quote and return ``(row_id, created)``.

    Provider quote timestamps are freshness evidence, not unique fetch
    identities. A repeated timestamp can legitimately arrive with corrected
    prices, so live ingestion updates that ledger row instead of changing only
    the fixture cache and leaving the two stores inconsistent.
    """
    snapshot_time = snapshot_time_utc or ""
    existing = con.execute(
        """
        SELECT id
        FROM odds_history
        WHERE game_id = ?
          AND source = ?
          AND snapshot_kind = 'live'
          AND snapshot_time_utc = ?
        """,
        (int(game_id), source, snapshot_time),
    ).fetchone()
    row = [
        int(game_id),
        competition_year,
        round_id,
        source,
        "live",
        snapshot_time,
    ]
    row.extend(values.get(field) for field in _NUMERIC_FIELDS)
    row.append(json.dumps(raw_meta) if raw_meta else None)
    columns = (
        "game_id, competition_year, round_id, source, snapshot_kind, "
        "snapshot_time_utc, " + ", ".join(_NUMERIC_FIELDS) + ", raw_meta"
    )
    assignments = ", ".join(
        [
            "competition_year = excluded.competition_year",
            "round_id = excluded.round_id",
            *(
                f"{field} = excluded.{field}"
                for field in _NUMERIC_FIELDS
            ),
            "raw_meta = excluded.raw_meta",
        ]
    )
    placeholders = ", ".join("?" for _ in row)
    cursor = con.execute(
        f"""
        INSERT INTO odds_history ({columns})
        VALUES ({placeholders})
        ON CONFLICT (game_id, source, snapshot_kind, snapshot_time_utc)
        DO UPDATE SET {assignments}
        """,
        row,
    )
    row_id = (
        int(existing[0])
        if existing is not None
        else int(cursor.lastrowid)
    )
    return row_id, existing is None


def latest_live_snapshots(con: sqlite3.Connection) -> dict[int, dict]:
    """Most recent live-provider observation per game."""
    fields = ", ".join(_NUMERIC_FIELDS)
    result: dict[int, dict] = {}
    for row in con.execute(
        f"""
        SELECT game_id, {fields}
        FROM odds_history
        WHERE source IN ('the_odds_api', 'betfair') AND snapshot_kind = 'live'
          AND id IN (
              SELECT MAX(id) FROM odds_history
              WHERE source IN ('the_odds_api', 'betfair')
                AND snapshot_kind = 'live'
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
