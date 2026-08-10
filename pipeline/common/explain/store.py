"""Persistence for per-game explanations.

Explanations live in their own table rather than as extra columns on
predictions_table. That table is the published tips contract: ten columns
spread across two SQL files, a view, a contract test and two duplicated
column-migration helpers, and a Drive CSV whose shape depends on it. The
payload here is also variable width (N drivers per game), which is a JSON blob,
not scalar columns.

The separation buys the important property: a missing or broken explanations
table degrades the email to exactly today's output instead of breaking a send.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import sqlite3
from datetime import datetime, timezone

import pandas as pd

from pipeline.common.explain import families as fam
from pipeline.common.explain import game as xgame
from pipeline.common.explain import trace as xt

EXPLANATION_SCHEMA_VERSION = 1
TABLE_NAME = "prediction_explanations"

# Additive migration, defined once here. predictions_table's equivalent is
# duplicated across prediction_functions and distribution; that duplication is
# on the send path and is deliberately left alone rather than refactored here.
_EXPECTED_COLUMNS = {
    "schema_version": "INTEGER",
    "taxonomy_version": "INTEGER",
    "generated_at": "TEXT",
    "model_release": "TEXT",
    "why_line": "TEXT",
    "prob_route": "TEXT",
    "attribution_source": "TEXT",
    "guard_fired": "INTEGER",
    "reconciled": "INTEGER",
    "line_applied": "INTEGER",
    "total_applied": "INTEGER",
    "published_cond": "REAL",
    "tier_a_prob": "REAL",
    "tier_b_prob": "REAL",
    "tier_c_prob": "REAL",
    "market_prob": "REAL",
    "pooled_logit": "REAL",
    "temperature": "REAL",
    "feature_multiplier": "REAL",
    "mu_model_home": "REAL",
    "mu_model_away": "REAL",
    "mu_baseline_home": "REAL",
    "mu_baseline_away": "REAL",
    "mu_blended_home": "REAL",
    "mu_blended_away": "REAL",
    "mu_final_home": "REAL",
    "mu_final_away": "REAL",
    "tier_a_attack_home": "REAL",
    "tier_a_defence_home": "REAL",
    "tier_a_attack_away": "REAL",
    "tier_a_defence_away": "REAL",
    "prob_drivers_json": "TEXT",
    "margin_drivers_json": "TEXT",
    "family_prob_json": "TEXT",
    "family_margin_json": "TEXT",
    "trace_json": "TEXT",
}

_SQL_DIR = pathlib.Path(__file__).resolve().parents[1] / "sql"


def _ensure_explanations_table_columns(con) -> None:
    existing = {row[1] for row in con.execute(f"PRAGMA table_info({TABLE_NAME})").fetchall()}
    for column, ddl in _EXPECTED_COLUMNS.items():
        if column not in existing:
            con.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN {column} {ddl}")


def _table_exists(con) -> bool:
    row = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (TABLE_NAME,)
    ).fetchone()
    return row is not None


def _drivers_json(drivers):
    return json.dumps([driver.as_dict() for driver in drivers])


def _row_values(explanation, *, generated_at, model_release):
    probability = explanation.probability
    score = explanation.score
    return (
        int(explanation.game_id),
        EXPLANATION_SCHEMA_VERSION,
        fam.FAMILY_TAXONOMY_VERSION,
        generated_at,
        model_release,
        explanation.why_line,
        probability.route,
        probability.attribution_source,
        int(bool(probability.guard_fired)),
        int(bool(score.reconciled)),
        int(bool(score.line_applied)),
        int(bool(score.total_applied)),
        float(probability.published_cond),
        float(probability.tier_a),
        float(probability.tier_b),
        float(probability.tier_c),
        float(probability.market),
        float(probability.pooled_logit),
        float(probability.temperature),
        float(probability.feature_multiplier),
        float(score.mu_model_home),
        float(score.mu_model_away),
        float(score.mu_baseline_home),
        float(score.mu_baseline_away),
        float(score.mu_blended_home),
        float(score.mu_blended_away),
        float(score.mu_final_home),
        float(score.mu_final_away),
        float(score.tier_a_attack_home),
        float(score.tier_a_defence_home),
        float(score.tier_a_attack_away),
        float(score.tier_a_defence_away),
        _drivers_json(explanation.prob_drivers),
        _drivers_json(explanation.margin_drivers),
        _drivers_json(explanation.prob_families),
        _drivers_json(explanation.margin_families),
        json.dumps(
            {
                "team_home": explanation.team_home,
                "team_away": explanation.team_away,
                "probability": probability.as_dict(),
                "score": score.as_dict(),
                "meta": dict(explanation.meta),
            },
            default=str,
        ),
    )


def _release_label(model_release) -> str | None:
    """A release identifier as text; the manifest stores a dict, not a string."""
    if model_release is None:
        return None
    if isinstance(model_release, dict):
        return str(
            model_release.get("release_id")
            or model_release.get("git_sha")
            or ""
        ) or None
    return str(model_release)


def save_explanations(explanations, db_path, project_root, *, model_release=None) -> int:
    """Upsert one row per game. Returns the number written."""
    if not explanations:
        return 0
    model_release = _release_label(model_release)
    project_root = pathlib.Path(project_root)
    create_sql = (project_root / "pipeline/common/sql/create_explanations_table.sql").read_text()
    insert_sql = (
        project_root / "pipeline/common/sql/insert_into_explanations_table.sql"
    ).read_text()
    generated_at = datetime.now(timezone.utc).isoformat()

    con = sqlite3.connect(str(db_path))
    try:
        con.execute(create_sql)
        _ensure_explanations_table_columns(con)
        for explanation in explanations:
            con.execute(
                insert_sql,
                _row_values(
                    explanation, generated_at=generated_at, model_release=model_release
                ),
            )
        con.commit()
    finally:
        con.close()
    return len(explanations)


def load_explanations(db_path, game_ids=None) -> pd.DataFrame:
    """Stored explanations as a frame. Empty frame when the table is absent."""
    try:
        con = sqlite3.connect(str(db_path))
    except sqlite3.Error:
        return pd.DataFrame()
    try:
        if not _table_exists(con):
            return pd.DataFrame()
        query = f"SELECT * FROM {TABLE_NAME}"
        params = ()
        if game_ids is not None:
            ids = [int(value) for value in game_ids]
            if not ids:
                return pd.DataFrame()
            query += f" WHERE game_id IN ({','.join('?' * len(ids))})"
            params = tuple(ids)
        return pd.read_sql_query(query, con, params=params)
    except Exception:
        return pd.DataFrame()
    finally:
        con.close()


def why_lines(db_path, game_ids=None) -> dict:
    """game_id -> why_line, for the email and site. Never raises."""
    frame = load_explanations(db_path, game_ids)
    if frame.empty or "why_line" not in frame.columns:
        return {}
    return {
        int(row["game_id"]): str(row["why_line"])
        for _, row in frame.iterrows()
        if row.get("why_line")
    }


def _construct(cls, payload):
    """Build a dataclass from stored JSON, ignoring keys it does not declare.

    as_dict() emits derived values (tipped_home) and a future schema may add
    more, so rehydration filters rather than trusting the payload's shape.
    """
    allowed = {field.name for field in dataclasses.fields(cls)}
    return cls(**{key: value for key, value in payload.items() if key in allowed})


def _drivers_from_json(payload):
    try:
        records = json.loads(payload) if payload else []
    except (TypeError, ValueError):
        return ()
    return tuple(xgame.Driver(**record) for record in records)


def load_game_explanations(db_path, game_ids=None) -> list:
    """Rehydrate stored rows into GameExplanation objects for the CLI/site."""
    frame = load_explanations(db_path, game_ids)
    if frame.empty:
        return []

    explanations = []
    for _, row in frame.iterrows():
        try:
            payload = json.loads(row.get("trace_json") or "{}")
            probability = _construct(xt.ProbabilityTrace, payload["probability"])
            score = _construct(xt.ScoreTrace, payload["score"])
        except Exception:
            # A row written by an older schema is skipped rather than crashing
            # the reader: explanations are diagnostics, not a contract.
            continue
        explanations.append(
            xgame.GameExplanation(
                game_id=int(row["game_id"]),
                team_home=payload.get("team_home", ""),
                team_away=payload.get("team_away", ""),
                probability=probability,
                score=score,
                prob_drivers=_drivers_from_json(row.get("prob_drivers_json")),
                margin_drivers=_drivers_from_json(row.get("margin_drivers_json")),
                prob_families=_drivers_from_json(row.get("family_prob_json")),
                margin_families=_drivers_from_json(row.get("family_margin_json")),
                why_line=str(row.get("why_line") or ""),
                meta=payload.get("meta", {}),
            )
        )
    return explanations
