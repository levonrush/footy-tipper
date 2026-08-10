"""Serialise cohort analyses to reports/explain-*.json.

Mirrors the conventions evaluate.py already uses for its own reports: a
timestamped file plus a stable ``-latest`` copy, an env override for the path,
and a write that never fails the run that produced it. Diagnostics that can
break the thing they diagnose do not get run.

Artifacts go to reports/, never models/: model_release._build_receipt hashes
every file in the models directory and an unexpected file fails release
validation.
"""

from __future__ import annotations

import json
import os
import pathlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pipeline.common.explain import families as fam

EXPLAIN_REPORT_SCHEMA_VERSION = 1

LATEST_FILENAME = "explain-latest.json"
PATH_ENV = "FOOTY_TIPPER_EXPLAIN_REPORT_PATH"

# Bounded so the artifact stays readable; per-game arrays never survive.
MAX_LISTED_FEATURES = 400
MAX_WORST_GAMES = 25


def _jsonable(value):
    if isinstance(value, pd.DataFrame):
        return value.replace({np.nan: None}).to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.replace({np.nan: None}).tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def build_explain_report(results: dict, *, source: str, config: dict = None) -> dict:
    """Turn run_analyses output into a JSON-safe document."""
    report = {
        "schema_version": EXPLAIN_REPORT_SCHEMA_VERSION,
        "taxonomy_version": fam.FAMILY_TAXONOMY_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "honest": source != "in-sample-deployed",
        "config": _jsonable(config or {}),
        "n_games": int(results.get("n_games", 0) or 0),
        "meta": _jsonable(results.get("meta", {})),
    }

    if "families" in results:
        report["families"] = _jsonable(results["families"])

    if "dead" in results:
        dead = dict(results["dead"])
        per_feature = dead.pop("per_feature", None)
        if per_feature is not None:
            dead["per_feature"] = _jsonable(per_feature.head(MAX_LISTED_FEATURES))
        report["dead"] = _jsonable(dead)

    if "coverage" in results:
        report["coverage"] = _jsonable(results["coverage"])

    if "disagreement" in results:
        report["disagreement"] = _jsonable(results["disagreement"])

    if "confident-wrong" in results:
        confident = dict(results["confident-wrong"])
        confident["worst_games"] = confident.get("worst_games", [])[:MAX_WORST_GAMES]
        report["confident_wrong"] = _jsonable(confident)

    return report


def write_explain_report(report: dict, project_root) -> pathlib.Path | None:
    """Write the report. Returns the path written, or None on any failure."""
    try:
        override = os.getenv(PATH_ENV)
        if override:
            path = pathlib.Path(override)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            return path

        reports_dir = pathlib.Path(project_root) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        payload = json.dumps(report, indent=2, default=str)
        stamped = reports_dir / f"explain-{stamp}.json"
        stamped.write_text(payload, encoding="utf-8")
        (reports_dir / LATEST_FILENAME).write_text(payload, encoding="utf-8")
        return stamped
    except Exception as exc:  # pragma: no cover - diagnostics must never fail a run
        print(f"Explain report not written ({exc}).")
        return None


def load_explain_report(path) -> dict:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def latest_report_path(project_root) -> pathlib.Path:
    return pathlib.Path(project_root) / "reports" / LATEST_FILENAME
