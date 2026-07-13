"""Writers for the R-owned feed_cache_* tables.

Replicates replace_cached_feed_year() from data-prep/feed-cache.R: align
columns both ways (ALTER TABLE ADD COLUMN for new incoming columns, NULL-fill
for columns the incoming batch lacks), then delete + append the year's rows.

A freeze rule protects feed history: seasons at or below `min_writable_year - 1`
are never rewritten, because the retired XML feed is the only source for them
(2008-2011 predate nrl.com match centre coverage).
"""

from __future__ import annotations

import sqlite3

DEFAULT_MIN_WRITABLE_YEAR = 2026


class FrozenSeasonError(ValueError):
    pass


def _table_columns(con: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in con.execute(f"PRAGMA table_info({table})")]


def _sqlite_type(values: list) -> str:
    for value in values:
        if isinstance(value, bool):
            return "INTEGER"
        if isinstance(value, (int, float)):
            return "REAL"
        if value is not None:
            return "TEXT"
    return "REAL"


def align_columns(
    con: sqlite3.Connection,
    table: str,
    rows: list[dict],
) -> list[str]:
    """Ensure the table has every incoming column; return final column order."""
    existing = _table_columns(con, table)
    if not existing:
        raise ValueError(
            f"Cache table '{table}' does not exist; it is created by the R "
            "prep and must be present before Python writes to it."
        )
    incoming: list[str] = []
    for row in rows:
        for key in row:
            if key not in incoming:
                incoming.append(key)
    for column in incoming:
        if column not in existing:
            column_type = _sqlite_type([row.get(column) for row in rows])
            con.execute(f'ALTER TABLE "{table}" ADD COLUMN "{column}" {column_type}')
            existing.append(column)
    return existing


def replace_cache_year(
    con: sqlite3.Connection,
    table: str,
    year: int,
    rows: list[dict],
    min_writable_year: int = DEFAULT_MIN_WRITABLE_YEAR,
) -> int:
    """Replace one season's rows in a feed cache table. Returns rows written."""
    year = int(year)
    if year < min_writable_year:
        raise FrozenSeasonError(
            f"Refusing to rewrite frozen season {year} in {table} "
            f"(min writable year is {min_writable_year})."
        )

    year_rows = [
        row
        for row in rows
        if row.get("competition_year") is not None
        and int(row["competition_year"]) == year
    ]
    if not year_rows:
        return 0

    columns = align_columns(con, table, year_rows)
    con.execute(f'DELETE FROM "{table}" WHERE competition_year = ?', (year,))
    quoted = ", ".join(f'"{col}"' for col in columns)
    placeholders = ", ".join("?" for _ in columns)
    con.executemany(
        f'INSERT INTO "{table}" ({quoted}) VALUES ({placeholders})',
        [tuple(row.get(col) for col in columns) for row in year_rows],
    )
    return len(year_rows)


def update_fixture_odds(
    con: sqlite3.Connection,
    game_id: int | float,
    odds: dict,
    only_when_null: bool = False,
) -> bool:
    """Update odds columns on one feed_cache_fixtures row.

    `only_when_null` guards historical backfill: it will not overwrite odds
    that came from the original feed.
    """
    allowed = {
        "team_head_to_head_odds_home",
        "team_head_to_head_odds_away",
        "team_line_odds_home",
        "team_line_odds_away",
        "team_line_amount_home",
        "team_line_amount_away",
        "total_line",
        "total_over_odds",
        "total_under_odds",
    }
    updates = {key: value for key, value in odds.items() if key in allowed}
    if not updates:
        return False

    existing = _table_columns(con, "feed_cache_fixtures")
    for column in updates:
        if column not in existing:
            con.execute(
                f'ALTER TABLE feed_cache_fixtures ADD COLUMN "{column}" REAL'
            )
            existing.append(column)

    if only_when_null:
        assignments = ", ".join(
            f'"{col}" = COALESCE("{col}", ?)' for col in updates
        )
    else:
        assignments = ", ".join(f'"{col}" = ?' for col in updates)
    cursor = con.execute(
        f"UPDATE feed_cache_fixtures SET {assignments} WHERE game_id = ?",
        (*updates.values(), game_id),
    )
    return cursor.rowcount > 0
