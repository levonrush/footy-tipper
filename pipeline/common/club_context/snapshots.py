"""Immutable, per-prediction-run Club Context snapshots."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any

import pandas as pd

from ..lineups.normalization import normalize_team_name
from .features import CONTEXT_FEATURE_COLUMNS, _matches_frame, resolve_context_for_matches
from .schema import CONTEXT_FEATURE_VERSION, CONTEXT_SCHEMA_VERSION, ensure_context_tables
from .taxonomy import CONTEXT_TAXONOMY_VERSION, PredictionMode, enum_value
from .time import capture_live_decision_at_utc, round_target_at_utc, utc_iso, utc_now


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _release_label(model_release: Any) -> str | None:
    if model_release is None:
        return None
    if isinstance(model_release, dict):
        return str(
            model_release.get("release_id")
            or model_release.get("git_sha")
            or ""
        ) or None
    return str(model_release)


def create_prediction_context_snapshot(
    db_path: str | Path,
    matches: Any,
    *,
    mode: PredictionMode,
    decision_at_utc: Any | None = None,
    model_release: Any = None,
    prediction_run_id: str | None = None,
    strict: bool = False,
) -> str | None:
    """Freeze observed context and features for one round/run.

    Operational modes require the one decision timestamp captured immediately
    before feature construction; this prevents a post-inference recapture from
    drifting from the facts the model actually saw.  Historical/evaluation
    modes default to the deterministic 11:00 Sydney round target.

    Returns the new run id.  By default any diagnostics failure returns None so
    Club Context cannot cost a prediction or send; ingestion/maintenance tools
    may pass ``strict=True`` to surface the original exception.
    """

    try:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(str(path))) as con, con:
            ensure_context_tables(con)
            mode_value = enum_value(PredictionMode, mode, "mode")
            frame = _matches_frame(con, matches)
            if frame.empty:
                raise ValueError("at least one match is required")
            required = {
                "game_id",
                "competition_year",
                "round_id",
                "team_home",
                "team_away",
                "start_time_utc",
            }
            if not required.issubset(frame.columns):
                missing = ", ".join(sorted(required - set(frame.columns)))
                raise ValueError(f"matches are missing: {missing}")
            identities = {
                (int(float(row.competition_year)), int(float(row.round_id)))
                for row in frame.itertuples()
            }
            if len(identities) != 1:
                raise ValueError("one prediction context snapshot must contain one round")
            year, round_id = next(iter(identities))
            target = round_target_at_utc(frame.to_dict("records"))
            if decision_at_utc is None:
                if mode_value not in {
                    PredictionMode.HISTORICAL.value,
                    PredictionMode.EVALUATION.value,
                }:
                    raise ValueError(
                        "operational snapshots require the decision_at_utc "
                        "captured before feature construction"
                    )
                decision = target
            else:
                decision = capture_live_decision_at_utc(decision_at_utc)
            features, observed = resolve_context_for_matches(
                con, frame, decision_at_utc=decision
            )
            feature_by_game = {
                int(float(row["game_id"])): {
                    column: float(row[column])
                    for column in CONTEXT_FEATURE_COLUMNS
                    if column != "game_id"
                }
                for row in features.to_dict("records")
            }
            fixture_ids = sorted(int(float(value)) for value in frame["game_id"])
            created = utc_iso(utc_now())
            decision_iso = utc_iso(decision)
            target_iso = utc_iso(target)
            run_id = prediction_run_id or str(uuid.uuid4())
            snapshot_rows = []
            for match in sorted(
                frame.to_dict("records"), key=lambda row: int(float(row["game_id"]))
            ):
                game_id = int(float(match["game_id"]))
                cards = observed.get(game_id, [])
                event_ids = sorted({int(card["event_id"]) for card in cards})
                snapshot_rows.append(
                    {
                        "prediction_run_id": run_id,
                        "game_id": game_id,
                        "competition_year": year,
                        "round_id": round_id,
                        "team_home": str(match["team_home"]),
                        "team_away": str(match["team_away"]),
                        "team_home_key": normalize_team_name(match["team_home"]),
                        "team_away_key": normalize_team_name(match["team_away"]),
                        "decision_at_utc": decision_iso,
                        "eligible_event_ids_json": _json(event_ids),
                        "observed_context_json": _json(cards),
                        "features_json": _json(feature_by_game[game_id]),
                        "created_at_utc": created,
                        "shadow_mode": 1,
                    }
                )

            digest_payload = {
                "mode": mode_value,
                "competition_year": year,
                "round_id": round_id,
                "round_target_at_utc": target_iso,
                "decision_at_utc": decision_iso,
                "model_release": _release_label(model_release),
                "fixture_ids": fixture_ids,
                "rows": [
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {"created_at_utc", "prediction_run_id"}
                    }
                    for row in snapshot_rows
                ],
                "schema_version": CONTEXT_SCHEMA_VERSION,
                "taxonomy_version": CONTEXT_TAXONOMY_VERSION,
                "feature_version": CONTEXT_FEATURE_VERSION,
                "shadow_mode": 1,
            }
            context_hash = hashlib.sha256(
                _json(digest_payload).encode("utf-8")
            ).hexdigest()
            con.execute(
                """
                INSERT INTO context_prediction_runs (
                    prediction_run_id, mode, competition_year, round_id,
                    round_target_at_utc, decision_at_utc, created_at_utc,
                    model_release, fixture_ids_json, context_hash,
                    schema_version, taxonomy_version, feature_version, shadow_mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    run_id,
                    digest_payload["mode"],
                    year,
                    round_id,
                    target_iso,
                    decision_iso,
                    created,
                    digest_payload["model_release"],
                    _json(fixture_ids),
                    context_hash,
                    CONTEXT_SCHEMA_VERSION,
                    CONTEXT_TAXONOMY_VERSION,
                    CONTEXT_FEATURE_VERSION,
                ),
            )
            columns = tuple(snapshot_rows[0])
            con.executemany(
                f"INSERT INTO prediction_context ({','.join(columns)}) "
                f"VALUES ({','.join('?' for _ in columns)})",
                [tuple(row[column] for column in columns) for row in snapshot_rows],
            )
            return run_id
    except Exception:
        if strict:
            raise
        return None


def load_prediction_context(
    db_path: str | Path,
    game_ids: Any | None = None,
    *,
    prediction_run_id: str | None = None,
) -> pd.DataFrame:
    """Load an exact run, or the latest run containing requested games."""

    path = Path(db_path)
    if not path.exists():
        return pd.DataFrame()
    try:
        ids = None if game_ids is None else [int(float(value)) for value in game_ids]
        if ids == []:
            return pd.DataFrame()
        with closing(sqlite3.connect(str(path))) as con:
            tables = {
                row[0]
                for row in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if not {"prediction_context", "context_prediction_runs"}.issubset(tables):
                return pd.DataFrame()
            run_id = prediction_run_id
            if run_id is None:
                where = ""
                params: tuple[Any, ...] = ()
                if ids is not None:
                    where = f"WHERE pc.game_id IN ({','.join('?' for _ in ids)})"
                    params = tuple(ids)
                found = con.execute(
                    f"""
                    SELECT pc.prediction_run_id
                    FROM prediction_context pc
                    JOIN context_prediction_runs pr
                      ON pr.prediction_run_id = pc.prediction_run_id
                    {where}
                    ORDER BY pr.created_at_utc DESC, pr.rowid DESC
                    LIMIT 1
                    """,
                    params,
                ).fetchone()
                if found is None:
                    return pd.DataFrame()
                run_id = str(found[0])
            query = """
                SELECT pc.*, pr.mode, pr.round_target_at_utc, pr.model_release,
                       pr.context_hash, pr.schema_version, pr.taxonomy_version,
                       pr.feature_version
                FROM prediction_context pc
                JOIN context_prediction_runs pr
                  ON pr.prediction_run_id = pc.prediction_run_id
                WHERE pc.prediction_run_id = ?
            """
            params_list: list[Any] = [run_id]
            if ids is not None:
                query += f" AND pc.game_id IN ({','.join('?' for _ in ids)})"
                params_list.extend(ids)
            query += " ORDER BY pc.game_id"
            return pd.read_sql_query(query, con, params=params_list)
    except Exception:
        return pd.DataFrame()
