"""Additive SQLite schema for the Club Context event registry.

Repository databases do not use a migration framework.  This module follows
the existing ingestion/store convention: create missing tables, add new
columns in a stable order, then create indexes.  Existing columns and rows are
never rewritten.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable


CONTEXT_SCHEMA_VERSION = 1
CONTEXT_FEATURE_VERSION = 2


TABLE_COLUMNS: dict[str, tuple[tuple[str, str], ...]] = {
    "context_article_snapshots": (
        ("snapshot_id", "INTEGER PRIMARY KEY AUTOINCREMENT"),
        ("canonical_url", "TEXT"),
        ("source_url", "TEXT"),
        ("publisher_key", "TEXT"),
        ("independence_key", "TEXT"),
        ("source_name", "TEXT"),
        ("source_class", "TEXT"),
        ("title", "TEXT"),
        ("snippet", "TEXT"),
        ("source_published_at_utc", "TEXT"),
        ("source_modified_at_utc", "TEXT"),
        ("first_observed_at_utc", "TEXT"),
        ("fetched_at_utc", "TEXT"),
        ("content_hash", "TEXT"),
        ("rights_status", "TEXT"),
        ("automated_use_allowed", "INTEGER NOT NULL DEFAULT 0"),
        ("acquisition_method", "TEXT NOT NULL DEFAULT 'manual'"),
        ("extraction_version", "TEXT NOT NULL DEFAULT '1'"),
        ("parse_status", "TEXT NOT NULL DEFAULT 'ok'"),
        ("parse_error", "TEXT"),
        ("metadata_json", "TEXT NOT NULL DEFAULT '{}'"),
    ),
    "context_events": (
        ("event_id", "INTEGER PRIMARY KEY AUTOINCREMENT"),
        ("event_key", "TEXT"),
        ("category", "TEXT"),
        ("phase", "TEXT"),
        ("known_at_utc", "TEXT"),
        ("known_at_basis", "TEXT NOT NULL DEFAULT 'source_published_at'"),
        ("effective_from_utc", "TEXT"),
        ("expires_at_utc", "TEXT"),
        ("confidence", "REAL NOT NULL DEFAULT 0"),
        ("salience", "REAL NOT NULL DEFAULT 0"),
        ("sensitivity", "TEXT NOT NULL DEFAULT 'standard'"),
        ("factual_summary", "TEXT"),
        ("confirmation_status", "TEXT NOT NULL DEFAULT 'unconfirmed'"),
        ("review_status", "TEXT NOT NULL DEFAULT 'pending'"),
        ("taxonomy_version", "INTEGER NOT NULL DEFAULT 1"),
        ("extractor_version", "TEXT NOT NULL DEFAULT 'manual-v1'"),
        ("created_at_utc", "TEXT"),
        ("updated_at_utc", "TEXT"),
        # v2 factual attributes.  Defaults keep every previously stored row
        # eligible and unchanged in meaning.
        ("disposition", "TEXT NOT NULL DEFAULT 'undetermined'"),
        ("availability_impact", "INTEGER NOT NULL DEFAULT 0"),
        ("magnitude", "REAL NOT NULL DEFAULT 0"),
    ),
    "context_attention_series": (
        ("team_key", "TEXT"),
        ("observed_date", "TEXT"),
        ("source_key", "TEXT"),
        ("article_count", "REAL NOT NULL DEFAULT 0"),
        ("corpus_norm", "REAL NOT NULL DEFAULT 0"),
        ("fetched_at_utc", "TEXT"),
    ),
    "context_attention_runs": (
        ("run_id", "TEXT PRIMARY KEY"),
        ("source_key", "TEXT"),
        ("started_at_utc", "TEXT"),
        ("completed_at_utc", "TEXT"),
        ("status", "TEXT"),
        ("start_year", "INTEGER NOT NULL DEFAULT 0"),
        ("end_year", "INTEGER NOT NULL DEFAULT 0"),
        ("team_count", "INTEGER NOT NULL DEFAULT 0"),
        ("observation_count", "INTEGER NOT NULL DEFAULT 0"),
        ("error_count", "INTEGER NOT NULL DEFAULT 0"),
        ("errors_json", "TEXT NOT NULL DEFAULT '[]'"),
    ),
    "context_event_sources": (
        ("event_id", "INTEGER"),
        ("snapshot_id", "INTEGER"),
        ("evidence_role", "TEXT"),
        ("publisher_key", "TEXT"),
        ("independence_key", "TEXT"),
        ("is_official", "INTEGER NOT NULL DEFAULT 0"),
        ("is_reputable", "INTEGER NOT NULL DEFAULT 0"),
        ("linked_at_utc", "TEXT"),
    ),
    "context_event_entities": (
        ("event_id", "INTEGER"),
        ("entity_type", "TEXT"),
        ("entity_key", "TEXT"),
        ("display_name", "TEXT"),
        ("team_key", "TEXT"),
        ("relationship", "TEXT NOT NULL DEFAULT 'affected'"),
        ("external_id", "TEXT NOT NULL DEFAULT ''"),
        ("linked_at_utc", "TEXT"),
    ),
    "context_ingestion_runs": (
        ("run_id", "TEXT PRIMARY KEY"),
        ("mode", "TEXT"),
        ("started_at_utc", "TEXT"),
        ("completed_at_utc", "TEXT"),
        ("status", "TEXT"),
        ("candidate_count", "INTEGER NOT NULL DEFAULT 0"),
        ("snapshot_count", "INTEGER NOT NULL DEFAULT 0"),
        ("event_count", "INTEGER NOT NULL DEFAULT 0"),
        ("eligible_event_count", "INTEGER NOT NULL DEFAULT 0"),
        ("error_count", "INTEGER NOT NULL DEFAULT 0"),
        ("config_json", "TEXT NOT NULL DEFAULT '{}'"),
        ("errors_json", "TEXT NOT NULL DEFAULT '[]'"),
    ),
    "context_prediction_runs": (
        ("prediction_run_id", "TEXT PRIMARY KEY"),
        ("mode", "TEXT"),
        ("competition_year", "INTEGER"),
        ("round_id", "INTEGER"),
        ("round_target_at_utc", "TEXT"),
        ("decision_at_utc", "TEXT"),
        ("created_at_utc", "TEXT"),
        ("model_release", "TEXT"),
        ("fixture_ids_json", "TEXT NOT NULL DEFAULT '[]'"),
        ("context_hash", "TEXT"),
        ("schema_version", "INTEGER NOT NULL DEFAULT 1"),
        ("taxonomy_version", "INTEGER NOT NULL DEFAULT 1"),
        ("feature_version", "INTEGER NOT NULL DEFAULT 1"),
        ("shadow_mode", "INTEGER NOT NULL DEFAULT 1"),
    ),
    "prediction_context": (
        ("prediction_run_id", "TEXT"),
        ("game_id", "INTEGER"),
        ("competition_year", "INTEGER"),
        ("round_id", "INTEGER"),
        ("team_home", "TEXT"),
        ("team_away", "TEXT"),
        ("team_home_key", "TEXT"),
        ("team_away_key", "TEXT"),
        ("decision_at_utc", "TEXT"),
        ("eligible_event_ids_json", "TEXT NOT NULL DEFAULT '[]'"),
        ("observed_context_json", "TEXT NOT NULL DEFAULT '[]'"),
        ("features_json", "TEXT NOT NULL DEFAULT '{}'"),
        ("created_at_utc", "TEXT"),
        ("shadow_mode", "INTEGER NOT NULL DEFAULT 1"),
    ),
}


_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS context_article_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_url TEXT NOT NULL,
    source_url TEXT NOT NULL,
    publisher_key TEXT NOT NULL,
    independence_key TEXT NOT NULL,
    source_name TEXT NOT NULL,
    source_class TEXT NOT NULL,
    title TEXT NOT NULL,
    snippet TEXT,
    source_published_at_utc TEXT,
    source_modified_at_utc TEXT,
    first_observed_at_utc TEXT NOT NULL,
    fetched_at_utc TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    rights_status TEXT NOT NULL,
    automated_use_allowed INTEGER NOT NULL DEFAULT 0,
    acquisition_method TEXT NOT NULL DEFAULT 'manual',
    extraction_version TEXT NOT NULL DEFAULT '1',
    parse_status TEXT NOT NULL DEFAULT 'ok',
    parse_error TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS context_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_key TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL,
    phase TEXT NOT NULL,
    known_at_utc TEXT NOT NULL,
    known_at_basis TEXT NOT NULL DEFAULT 'source_published_at',
    effective_from_utc TEXT NOT NULL,
    expires_at_utc TEXT,
    confidence REAL NOT NULL DEFAULT 0,
    salience REAL NOT NULL DEFAULT 0,
    sensitivity TEXT NOT NULL DEFAULT 'standard',
    factual_summary TEXT NOT NULL,
    confirmation_status TEXT NOT NULL DEFAULT 'unconfirmed',
    review_status TEXT NOT NULL DEFAULT 'pending',
    taxonomy_version INTEGER NOT NULL DEFAULT 1,
    extractor_version TEXT NOT NULL DEFAULT 'manual-v1',
    created_at_utc TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL,
    disposition TEXT NOT NULL DEFAULT 'undetermined',
    availability_impact INTEGER NOT NULL DEFAULT 0,
    magnitude REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS context_attention_series (
    team_key TEXT NOT NULL,
    observed_date TEXT NOT NULL,
    source_key TEXT NOT NULL,
    article_count REAL NOT NULL DEFAULT 0,
    corpus_norm REAL NOT NULL DEFAULT 0,
    fetched_at_utc TEXT NOT NULL,
    PRIMARY KEY (team_key, observed_date, source_key)
);

CREATE TABLE IF NOT EXISTS context_attention_runs (
    run_id TEXT PRIMARY KEY,
    source_key TEXT NOT NULL,
    started_at_utc TEXT NOT NULL,
    completed_at_utc TEXT,
    status TEXT NOT NULL,
    start_year INTEGER NOT NULL DEFAULT 0,
    end_year INTEGER NOT NULL DEFAULT 0,
    team_count INTEGER NOT NULL DEFAULT 0,
    observation_count INTEGER NOT NULL DEFAULT 0,
    error_count INTEGER NOT NULL DEFAULT 0,
    errors_json TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS context_event_sources (
    event_id INTEGER NOT NULL,
    snapshot_id INTEGER NOT NULL,
    evidence_role TEXT NOT NULL,
    publisher_key TEXT NOT NULL,
    independence_key TEXT NOT NULL,
    is_official INTEGER NOT NULL DEFAULT 0,
    is_reputable INTEGER NOT NULL DEFAULT 0,
    linked_at_utc TEXT NOT NULL,
    PRIMARY KEY (event_id, snapshot_id)
);

CREATE TABLE IF NOT EXISTS context_event_entities (
    event_id INTEGER NOT NULL,
    entity_type TEXT NOT NULL,
    entity_key TEXT NOT NULL,
    display_name TEXT NOT NULL,
    team_key TEXT NOT NULL,
    relationship TEXT NOT NULL DEFAULT 'affected',
    external_id TEXT NOT NULL DEFAULT '',
    linked_at_utc TEXT NOT NULL,
    PRIMARY KEY (event_id, entity_type, entity_key, team_key, relationship)
);

CREATE TABLE IF NOT EXISTS context_ingestion_runs (
    run_id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    started_at_utc TEXT NOT NULL,
    completed_at_utc TEXT,
    status TEXT NOT NULL,
    candidate_count INTEGER NOT NULL DEFAULT 0,
    snapshot_count INTEGER NOT NULL DEFAULT 0,
    event_count INTEGER NOT NULL DEFAULT 0,
    eligible_event_count INTEGER NOT NULL DEFAULT 0,
    error_count INTEGER NOT NULL DEFAULT 0,
    config_json TEXT NOT NULL DEFAULT '{}',
    errors_json TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS context_prediction_runs (
    prediction_run_id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    competition_year INTEGER NOT NULL,
    round_id INTEGER NOT NULL,
    round_target_at_utc TEXT NOT NULL,
    decision_at_utc TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    model_release TEXT,
    fixture_ids_json TEXT NOT NULL DEFAULT '[]',
    context_hash TEXT NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 1,
    taxonomy_version INTEGER NOT NULL DEFAULT 1,
    feature_version INTEGER NOT NULL DEFAULT 1,
    shadow_mode INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS prediction_context (
    prediction_run_id TEXT NOT NULL,
    game_id INTEGER NOT NULL,
    competition_year INTEGER NOT NULL,
    round_id INTEGER NOT NULL,
    team_home TEXT NOT NULL,
    team_away TEXT NOT NULL,
    team_home_key TEXT NOT NULL,
    team_away_key TEXT NOT NULL,
    decision_at_utc TEXT NOT NULL,
    eligible_event_ids_json TEXT NOT NULL DEFAULT '[]',
    observed_context_json TEXT NOT NULL DEFAULT '[]',
    features_json TEXT NOT NULL DEFAULT '{}',
    created_at_utc TEXT NOT NULL,
    shadow_mode INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (prediction_run_id, game_id)
);
"""


_INDEX_AND_TRIGGER_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_context_article_url_hash_version
    ON context_article_snapshots (canonical_url, content_hash, extraction_version);
CREATE INDEX IF NOT EXISTS idx_context_article_published
    ON context_article_snapshots (source_published_at_utc, first_observed_at_utc);
CREATE INDEX IF NOT EXISTS idx_context_article_publisher
    ON context_article_snapshots (publisher_key, independence_key, source_class);
CREATE UNIQUE INDEX IF NOT EXISTS idx_context_events_key
    ON context_events (event_key);
CREATE INDEX IF NOT EXISTS idx_context_events_window
    ON context_events (known_at_utc, effective_from_utc, expires_at_utc);
CREATE INDEX IF NOT EXISTS idx_context_events_state
    ON context_events (review_status, confirmation_status, confidence);
CREATE INDEX IF NOT EXISTS idx_context_event_sources_event
    ON context_event_sources (event_id, independence_key);
CREATE UNIQUE INDEX IF NOT EXISTS idx_context_event_sources_unique
    ON context_event_sources (event_id, snapshot_id);
CREATE INDEX IF NOT EXISTS idx_context_event_entities_team
    ON context_event_entities (team_key, event_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_context_event_entities_unique
    ON context_event_entities (
        event_id, entity_type, entity_key, team_key, relationship
    );
CREATE INDEX IF NOT EXISTS idx_context_ingestion_runs_window
    ON context_ingestion_runs (mode, started_at_utc);
CREATE INDEX IF NOT EXISTS idx_context_prediction_runs_round
    ON context_prediction_runs (competition_year, round_id, created_at_utc);
CREATE INDEX IF NOT EXISTS idx_prediction_context_game
    ON prediction_context (game_id, created_at_utc);
CREATE INDEX IF NOT EXISTS idx_context_attention_team_date
    ON context_attention_series (team_key, observed_date);
CREATE UNIQUE INDEX IF NOT EXISTS idx_prediction_context_run_game
    ON prediction_context (prediction_run_id, game_id);

CREATE TRIGGER IF NOT EXISTS trg_context_prediction_runs_no_update
BEFORE UPDATE ON context_prediction_runs
BEGIN
    SELECT RAISE(ABORT, 'context prediction runs are immutable');
END;
CREATE TRIGGER IF NOT EXISTS trg_context_prediction_runs_no_delete
BEFORE DELETE ON context_prediction_runs
BEGIN
    SELECT RAISE(ABORT, 'context prediction runs are immutable');
END;
CREATE TRIGGER IF NOT EXISTS trg_prediction_context_no_update
BEFORE UPDATE ON prediction_context
BEGIN
    SELECT RAISE(ABORT, 'prediction context snapshots are immutable');
END;
CREATE TRIGGER IF NOT EXISTS trg_prediction_context_no_delete
BEFORE DELETE ON prediction_context
BEGIN
    SELECT RAISE(ABORT, 'prediction context snapshots are immutable');
END;
"""


def _table_columns(con: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in con.execute(f"PRAGMA table_info({table})")}


def _ensure_columns(
    con: sqlite3.Connection,
    table: str,
    expected: Iterable[tuple[str, str]],
) -> None:
    existing = _table_columns(con, table)
    for column, ddl in expected:
        if column in existing:
            continue
        # SQLite cannot add a primary-key column.  A legacy table bearing the
        # package's name must at least retain its identity column.
        if "PRIMARY KEY" in ddl:
            raise sqlite3.OperationalError(
                f"{table} is missing required identity column {column}"
            )
        con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
        existing.add(column)


def ensure_context_tables(con: sqlite3.Connection) -> None:
    """Create/migrate every Club Context table without rewriting data."""

    con.executescript(_CREATE_SQL)
    for table, columns in TABLE_COLUMNS.items():
        _ensure_columns(con, table, columns)
    con.executescript(_INDEX_AND_TRIGGER_SQL)


def context_tables_present(con: sqlite3.Connection) -> bool:
    try:
        found = {
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        return set(TABLE_COLUMNS).issubset(found)
    except sqlite3.Error:
        return False


def context_schema_current(con: sqlite3.Connection) -> bool:
    """Whether every v1 table and additive column is already available."""

    if not context_tables_present(con):
        return False
    try:
        return all(
            {column for column, _ in expected}.issubset(_table_columns(con, table))
            for table, expected in TABLE_COLUMNS.items()
        )
    except sqlite3.Error:
        return False
