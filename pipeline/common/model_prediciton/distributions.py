"""Persistence for per-game simulated-distribution summaries.

A finals email has one to four games instead of eight, so it can afford to go
deep on each one: how likely a one-score game is, whether the model thinks the
favourite covers, whether the total goes over. All of that is already implicit in
the simulation that produced the tip, and none of it is in `predictions_table`.

This follows `prediction_explanations` exactly, and for the same reason: that
table is the published tips contract, and widening it means touching two SQL
files, a view, a contract test and the Drive CSV. A separate sibling table means
a missing or broken distributions table costs the email a section rather than a
send.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pandas as pd

DISTRIBUTION_SCHEMA_VERSION = 1
TABLE_NAME = "prediction_distributions"

# Additive migration, defined once here.
_EXPECTED_COLUMNS = {
    "schema_version": "INTEGER",
    "generated_at": "TEXT",
    "p_draw_full_time": "REAL",
    "p_one_score_game": "REAL",
    "p_home_by_1_6": "REAL",
    "p_home_by_7_12": "REAL",
    "p_home_by_13_plus": "REAL",
    "p_away_by_1_6": "REAL",
    "p_away_by_7_12": "REAL",
    "p_away_by_13_plus": "REAL",
    "median_total": "REAL",
    "total_p10": "REAL",
    "total_p90": "REAL",
    "p_home_covers_line": "REAL",
    "p_line_push": "REAL",
    "p_total_over": "REAL",
    "p_total_push": "REAL",
}

VALUE_COLUMNS = tuple(
    column
    for column in _EXPECTED_COLUMNS
    if column not in {"schema_version", "generated_at"}
)


def _ensure_table(con) -> None:
    con.execute(
        f"CREATE TABLE IF NOT EXISTS {TABLE_NAME} (game_id INTEGER PRIMARY KEY)"
    )
    existing = {row[1] for row in con.execute(f"PRAGMA table_info({TABLE_NAME})")}
    for column, ddl in _EXPECTED_COLUMNS.items():
        if column not in existing:
            con.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN {column} {ddl}")


def save_distributions(distributions, db_path) -> int:
    """Upsert one row per game. Returns the number of rows written."""
    if distributions is None or getattr(distributions, "empty", True):
        return 0
    if "game_id" not in distributions.columns:
        return 0

    generated_at = datetime.now(timezone.utc).isoformat()
    columns = ["game_id", "schema_version", "generated_at", *VALUE_COLUMNS]
    placeholders = ", ".join("?" for _ in columns)
    assignments = ", ".join(
        f"{column} = excluded.{column}" for column in columns if column != "game_id"
    )

    rows = []
    for _, record in distributions.iterrows():
        rows.append(
            (
                int(record["game_id"]),
                DISTRIBUTION_SCHEMA_VERSION,
                generated_at,
                *(_optional(record.get(column)) for column in VALUE_COLUMNS),
            )
        )

    con = sqlite3.connect(str(db_path))
    try:
        _ensure_table(con)
        con.executemany(
            f"INSERT INTO {TABLE_NAME} ({', '.join(columns)}) VALUES ({placeholders}) "
            f"ON CONFLICT(game_id) DO UPDATE SET {assignments}",
            rows,
        )
        con.commit()
    finally:
        con.close()
    return len(rows)


def load_distributions(db_path, game_ids=None) -> pd.DataFrame:
    """Stored summaries for the given games, or an empty frame.

    Never raises: a missing table is the normal state before the first inference
    run on a new database.
    """
    try:
        con = sqlite3.connect(str(db_path))
        try:
            if not _table_exists(con):
                return pd.DataFrame()
            query = f"SELECT * FROM {TABLE_NAME}"
            params = ()
            if game_ids is not None:
                identifiers = [int(value) for value in game_ids]
                if not identifiers:
                    return pd.DataFrame()
                query += f" WHERE game_id IN ({', '.join('?' for _ in identifiers)})"
                params = tuple(identifiers)
            return pd.read_sql_query(query, con, params=params)
        finally:
            con.close()
    except Exception as exc:
        print(f"Distribution lookup failed ({exc}).")
        return pd.DataFrame()


def _table_exists(con) -> bool:
    return (
        con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (TABLE_NAME,)
        ).fetchone()
        is not None
    )


def _optional(value):
    if value is None or pd.isna(value):
        return None
    return float(value)
