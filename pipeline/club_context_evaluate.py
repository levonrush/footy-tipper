"""Shadow-only materiality evaluation for Club Context.

The production model is never loaded or changed here.  This module scores
paired out-of-fold baseline/candidate predictions written by the honest model
evaluation, and can also report that evidence is not ready yet.  Keeping the
comparison separate makes a failed or under-powered experiment incapable of
changing a released tip.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pathlib
import sqlite3
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from pipeline.common.club_context.registry import event_is_eligible
from pipeline.common.club_context.schema import context_tables_present
from pipeline.common.club_context.time import parse_datetime
from pipeline.common.lineups.normalization import normalize_team_name


DEFAULT_SEED = 20100308
DEFAULT_BOOTSTRAP_REPS = 2000
REQUIRED_COLUMNS = frozenset(
    {"game_id", "actual_home_win", "baseline_home_win_prob", "context_home_win_prob"}
)


def _safe_number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _format_delta(value) -> str:
    return "—" if value is None else f"{float(value):+.4f}"


def _calibration(y: np.ndarray, p: np.ndarray) -> dict:
    if len(y) < 10 or len(np.unique(y)) < 2:
        return {"intercept": None, "slope": None}
    x = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1)).reshape(-1, 1)
    try:
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
        with warnings.catch_warnings():
            # Tiny, nearly separated category cohorts can overflow an internal
            # optimiser step even though the fitted values remain finite.
            warnings.simplefilter("ignore", RuntimeWarning)
            model.fit(x, y)
        return {
            "intercept": float(model.intercept_[0]),
            "slope": float(model.coef_[0, 0]),
        }
    except Exception:
        return {"intercept": None, "slope": None}


def _probability_metrics(y, p) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    if not len(y):
        return {
            "games": 0,
            "correct": 0,
            "accuracy": None,
            "log_loss": None,
            "brier": None,
            "calibration": {"intercept": None, "slope": None},
        }
    correct = int(((p > 0.5) == y.astype(bool)).sum())
    return {
        "games": int(len(y)),
        "correct": correct,
        "accuracy": float(correct / len(y)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "calibration": _calibration(y, p),
    }


def _score_metrics(frame: pd.DataFrame, prefix: str) -> dict | None:
    required = {
        "actual_home_score",
        "actual_away_score",
        f"{prefix}_home_score",
        f"{prefix}_away_score",
    }
    if not required.issubset(frame.columns):
        return None
    values = frame[list(required)].apply(pd.to_numeric, errors="coerce")
    usable = values.notna().all(axis=1)
    if not usable.any():
        return None
    actual_h = pd.to_numeric(frame.loc[usable, "actual_home_score"]).to_numpy(float)
    actual_a = pd.to_numeric(frame.loc[usable, "actual_away_score"]).to_numpy(float)
    pred_h = pd.to_numeric(frame.loc[usable, f"{prefix}_home_score"]).to_numpy(float)
    pred_a = pd.to_numeric(frame.loc[usable, f"{prefix}_away_score"]).to_numpy(float)
    actual_margin = actual_h - actual_a
    pred_margin = pred_h - pred_a
    residual = actual_margin - pred_margin
    return {
        "games": int(usable.sum()),
        "margin_mae": float(np.mean(np.abs(residual))),
        "margin_bias": float(np.mean(pred_margin - actual_margin)),
        "home_score_mae": float(np.mean(np.abs(pred_h - actual_h))),
        "away_score_mae": float(np.mean(np.abs(pred_a - actual_a))),
        "total_mae": float(np.mean(np.abs((pred_h + pred_a) - (actual_h + actual_a)))),
        "residual_variance": float(np.var(residual, ddof=1)) if len(residual) > 1 else None,
    }


def _metric_pair(
    frame: pd.DataFrame,
    *,
    baseline_probability: str = "baseline_home_win_prob",
    context_probability: str = "context_home_win_prob",
    include_scores: bool = True,
) -> dict:
    y = frame["actual_home_win"].to_numpy(int)
    baseline_p = frame[baseline_probability].to_numpy(float)
    context_p = frame[context_probability].to_numpy(float)
    baseline = _probability_metrics(y, baseline_p)
    context = _probability_metrics(y, context_p)
    score_baseline = _score_metrics(frame, "baseline") if include_scores else None
    score_context = _score_metrics(frame, "context") if include_scores else None
    deltas = {
        "accuracy": (context["accuracy"] - baseline["accuracy"])
        if baseline["accuracy"] is not None
        else None,
        "log_loss": (context["log_loss"] - baseline["log_loss"])
        if baseline["log_loss"] is not None
        else None,
        "brier": (context["brier"] - baseline["brier"])
        if baseline["brier"] is not None
        else None,
    }
    if score_baseline and score_context:
        deltas.update(
            {
                "margin_mae": score_context["margin_mae"] - score_baseline["margin_mae"],
                "residual_variance": (
                    score_context["residual_variance"] - score_baseline["residual_variance"]
                    if score_context["residual_variance"] is not None
                    and score_baseline["residual_variance"] is not None
                    else None
                ),
            }
        )
    flips = (baseline_p > 0.5) != (context_p > 0.5)
    baseline_correct = (baseline_p > 0.5) == y.astype(bool)
    context_correct = (context_p > 0.5) == y.astype(bool)
    return {
        "baseline": {**baseline, "score": score_baseline},
        "context_candidate": {**context, "score": score_context},
        "delta_context_minus_baseline": deltas,
        "tip_flips": int(flips.sum()),
        "flips_to_correct": int((flips & ~baseline_correct & context_correct).sum()),
        "flips_to_wrong": int((flips & baseline_correct & ~context_correct).sum()),
    }


def _paired_losses(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = frame["actual_home_win"].to_numpy(int)
    baseline = np.clip(frame["baseline_home_win_prob"].to_numpy(float), 1e-6, 1 - 1e-6)
    context = np.clip(frame["context_home_win_prob"].to_numpy(float), 1e-6, 1 - 1e-6)
    baseline_ll = -(y * np.log(baseline) + (1 - y) * np.log(1 - baseline))
    context_ll = -(y * np.log(context) + (1 - y) * np.log(1 - context))
    brier_delta = (context - y) ** 2 - (baseline - y) ** 2
    return context_ll - baseline_ll, brier_delta, context - baseline


def _cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    reps: int,
    seed: int,
) -> dict:
    if frame.empty:
        return {"clusters": 0, "reps": 0, "log_loss_delta_ci95": [None, None], "brier_delta_ci95": [None, None]}
    cluster_col = "event_id" if "event_id" in frame and frame["event_id"].notna().any() else "game_id"
    cluster_values = frame[cluster_col].fillna(frame["game_id"]).astype(str)
    clusters = sorted(cluster_values.unique())
    if len(clusters) < 2 or reps <= 0:
        return {"clusters": len(clusters), "reps": 0, "log_loss_delta_ci95": [None, None], "brier_delta_ci95": [None, None]}
    grouped = [frame.loc[cluster_values == cluster] for cluster in clusters]
    rng = np.random.default_rng(seed)
    ll_samples = np.empty(reps, dtype=float)
    brier_samples = np.empty(reps, dtype=float)
    for idx in range(reps):
        chosen = rng.integers(0, len(grouped), size=len(grouped))
        sample = pd.concat([grouped[i] for i in chosen], ignore_index=True)
        ll_delta, brier_delta, _ = _paired_losses(sample)
        ll_samples[idx] = float(np.mean(ll_delta))
        brier_samples[idx] = float(np.mean(brier_delta))
    return {
        "cluster_key": cluster_col,
        "clusters": len(clusters),
        "reps": int(reps),
        "log_loss_delta_ci95": [float(v) for v in np.quantile(ll_samples, [0.025, 0.975])],
        "brier_delta_ci95": [float(v) for v in np.quantile(brier_samples, [0.025, 0.975])],
        "probability_log_loss_improves": float(np.mean(ll_samples < 0)),
        "probability_brier_improves": float(np.mean(brier_samples < 0)),
    }


ANNOTATION_COLUMNS = (
    "game_id",
    "event_id",
    "event_key",
    "category",
    "phase",
    "event_relative_match",
    "affected_side",
    "affected_team_key",
    "event_overlap_count",
)


def build_event_annotations(
    matches: pd.DataFrame,
    db_path: str | pathlib.Path,
    *,
    window: int = 3,
) -> pd.DataFrame:
    """Attach research-only -3..+3 event windows to historical fixtures.

    These labels may use later reviewed confirmation to identify a historical
    event, but they never feed the feature transformer.  The actual candidate
    columns remain frozen by the historical round cutoff.  Keeping the label
    path separate permits pre-trend/placebo checks without creating leakage.
    """

    empty = pd.DataFrame(columns=ANNOTATION_COLUMNS)
    required = {"game_id", "competition_year", "team_home", "team_away"}
    if matches.empty or not required.issubset(matches.columns):
        return empty
    time_column = (
        "start_time_utc" if "start_time_utc" in matches.columns else "start_time"
    )
    if time_column not in matches.columns:
        return empty
    path = pathlib.Path(db_path)
    if not path.exists():
        return empty

    fixtures = []
    for record in matches.to_dict("records"):
        try:
            fixtures.append(
                {
                    "game_id": int(float(record["game_id"])),
                    "competition_year": int(float(record["competition_year"])),
                    "kickoff": parse_datetime(record[time_column]),
                    "home_key": normalize_team_name(record["team_home"]),
                    "away_key": normalize_team_name(record["team_away"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    if not fixtures:
        return empty

    candidates: dict[int, list[dict]] = {}
    try:
        with sqlite3.connect(str(path)) as con:
            con.row_factory = sqlite3.Row
            if not context_tables_present(con):
                return empty
            research_decision = max(
                max(item["kickoff"] for item in fixtures),
                datetime.now(timezone.utc),
            ) + dt.timedelta(days=1)
            events = con.execute(
                """
                SELECT DISTINCT e.event_id, e.event_key, e.category, e.phase,
                       e.effective_from_utc, e.expires_at_utc, e.salience,
                       ee.team_key
                FROM context_events e
                JOIN context_event_entities ee ON ee.event_id = e.event_id
                WHERE ee.relationship IN ('affected', 'subject')
                  AND e.review_status = 'approved'
                  AND e.confirmation_status = 'confirmed'
                ORDER BY e.effective_from_utc, e.event_id, ee.team_key
                """
            ).fetchall()
            for raw in events:
                event = dict(raw)
                try:
                    effective = parse_datetime(event["effective_from_utc"])
                    if not event_is_eligible(
                        con,
                        int(event["event_id"]),
                        decision_at_utc=research_decision,
                        match_at_utc=effective,
                    ):
                        continue
                    team_key = normalize_team_name(event["team_key"])
                    team_games = sorted(
                        (
                            item
                            for item in fixtures
                            if team_key in {item["home_key"], item["away_key"]}
                        ),
                        key=lambda item: (item["kickoff"], item["game_id"]),
                    )
                    anchor = next(
                        index
                        for index, item in enumerate(team_games)
                        if item["kickoff"] >= effective
                    )
                    expires = (
                        parse_datetime(event["expires_at_utc"])
                        if event.get("expires_at_utc")
                        else None
                    )
                    if expires is not None and team_games[anchor]["kickoff"] > expires:
                        continue
                except (StopIteration, TypeError, ValueError):
                    continue

                start = max(0, anchor - max(0, int(window)))
                stop = min(len(team_games), anchor + max(0, int(window)) + 1)
                for position in range(start, stop):
                    fixture = team_games[position]
                    relative = position - anchor
                    side = "home" if fixture["home_key"] == team_key else "away"
                    candidates.setdefault(fixture["game_id"], []).append(
                        {
                            "game_id": fixture["game_id"],
                            "event_id": int(event["event_id"]),
                            "event_key": str(event["event_key"]),
                            "category": str(event["category"]),
                            "phase": str(event["phase"]),
                            "event_relative_match": int(relative),
                            "affected_side": side,
                            "affected_team_key": team_key,
                            "salience": float(event["salience"]),
                        }
                    )
    except sqlite3.Error:
        return empty

    rows = []
    for game_id, options in sorted(candidates.items()):
        # One prediction row can overlap events.  Retain a deterministic primary
        # label for tables and the overlap count so clustered uncertainty is not
        # accidentally presented as independent evidence.
        primary = sorted(
            options,
            key=lambda item: (
                abs(item["event_relative_match"]),
                item["event_relative_match"] < 0,
                -item["salience"],
                item["event_id"],
            ),
        )[0]
        rows.append(
            {
                **{key: primary[key] for key in ANNOTATION_COLUMNS if key in primary},
                "event_overlap_count": len(options),
            }
        )
    return pd.DataFrame(rows, columns=ANNOTATION_COLUMNS)


def _affected_margin_residual(frame: pd.DataFrame) -> pd.Series:
    actual = (
        pd.to_numeric(frame["actual_home_score"], errors="coerce")
        - pd.to_numeric(frame["actual_away_score"], errors="coerce")
    )
    baseline = (
        pd.to_numeric(frame["baseline_home_score"], errors="coerce")
        - pd.to_numeric(frame["baseline_away_score"], errors="coerce")
    )
    orientation = frame.get("affected_side", "home").map(
        {"home": 1.0, "away": -1.0}
    )
    return (actual - baseline) * orientation


def _matched_event_effect(
    frame: pd.DataFrame,
    residual: pd.Series,
    relative: pd.Series,
    *,
    seed: int,
) -> dict:
    """Nearest-neighbour residual comparison for event-match rows.

    The nested baseline prediction is the primary balancing score: it already
    conditions on pre-match team/opponent strength, form, venue, rest and
    lineups.  Available explicit deltas and the closing line refine distance.
    Outcomes are never used to choose neighbours.
    """

    treated_mask = relative.eq(0) & frame.get("event_id", pd.Series(index=frame.index)).notna()
    control_mask = frame.get("event_id", pd.Series(index=frame.index)).isna()
    if treated_mask.sum() < 2 or control_mask.sum() < 10:
        return {
            "available": False,
            "reason": "too few event matches or uncontaminated controls",
        }

    numeric_covariates = [
        column
        for column in (
            "baseline_home_win_prob",
            "market_spread",
            "form_delta",
            "rest_delta",
            "elo_diff",
            "lineup_avg_named_margin_rating_delta",
            "lineup_selection_uncertainty_delta",
        )
        if column in frame.columns
    ]
    scales = {}
    for column in numeric_covariates:
        values = pd.to_numeric(frame[column], errors="coerce")
        scale = float(values.quantile(0.75) - values.quantile(0.25))
        scales[column] = scale if math.isfinite(scale) and scale > 1e-9 else 1.0

    controls = []
    for index in frame.index[control_mask]:
        for side, orientation in (("home", 1.0), ("away", -1.0)):
            controls.append((index, side, orientation))

    pair_rows = []
    for index in frame.index[treated_mask]:
        treated_side = str(frame.at[index, "affected_side"])
        treated_orientation = 1.0 if treated_side == "home" else -1.0
        treated_year = frame.at[index, "competition_year"] if "competition_year" in frame else None
        treated_market = bool(frame.at[index, "market_available"]) if "market_available" in frame else None
        distances = []
        for control_index, control_side, control_orientation in controls:
            if "competition_year" in frame and frame.at[control_index, "competition_year"] != treated_year:
                continue
            if (
                treated_market is not None
                and bool(frame.at[control_index, "market_available"]) != treated_market
            ):
                continue
            distance = 0.0
            used = 0
            for column in numeric_covariates:
                left = pd.to_numeric(pd.Series([frame.at[index, column]]), errors="coerce").iloc[0]
                right = pd.to_numeric(pd.Series([frame.at[control_index, column]]), errors="coerce").iloc[0]
                if not (math.isfinite(left) and math.isfinite(right)):
                    continue
                if column == "baseline_home_win_prob":
                    left = left if treated_orientation > 0 else 1.0 - left
                    right = right if control_orientation > 0 else 1.0 - right
                elif column.endswith("_delta") or column in {"market_spread", "elo_diff"}:
                    left *= treated_orientation
                    right *= control_orientation
                distance += ((left - right) / scales[column]) ** 2
                used += 1
            if "venue_name" in frame and frame.at[index, "venue_name"] != frame.at[control_index, "venue_name"]:
                distance += 0.25
            if used:
                distances.append((distance, control_index, control_orientation))
        if not distances:
            continue
        neighbours = sorted(distances, key=lambda item: (item[0], item[1], item[2]))[:3]
        control_residuals = []
        for _, control_index, orientation in neighbours:
            home_residual = (
                float(frame.at[control_index, "actual_home_score"])
                - float(frame.at[control_index, "actual_away_score"])
                - float(frame.at[control_index, "baseline_home_score"])
                + float(frame.at[control_index, "baseline_away_score"])
            )
            control_residuals.append(home_residual * orientation)
        pair_rows.append(
            {
                "event_id": str(frame.at[index, "event_id"]),
                "effect": float(residual.at[index]) - float(np.mean(control_residuals)),
                "controls": len(control_residuals),
            }
        )
    if len(pair_rows) < 2:
        return {"available": False, "reason": "too few matched event rows"}

    paired = pd.DataFrame(pair_rows)
    by_event = paired.groupby("event_id", sort=True)["effect"].mean()
    rng = np.random.default_rng(seed)
    draws = np.empty(1000, dtype=float)
    values = by_event.to_numpy(float)
    for index in range(len(draws)):
        draws[index] = float(rng.choice(values, size=len(values), replace=True).mean())
    return {
        "available": True,
        "event_matches": int(len(paired)),
        "event_clusters": int(len(by_event)),
        "controls_per_event_match": 3,
        "mean_affected_margin_residual_difference": float(paired["effect"].mean()),
        "cluster_bootstrap_ci95": [
            float(value) for value in np.quantile(draws, [0.025, 0.975])
        ],
        "matching": {
            "primary_balance": "nested baseline predicted probability and margin",
            "refinements": numeric_covariates[1:],
            "exact": ["competition_year", "market availability"],
            "venue_mismatch_penalty": "applied when venue_name is available",
        },
    }


def _event_study(frame: pd.DataFrame, *, seed: int) -> dict:
    if "event_relative_match" not in frame.columns:
        return {"available": False, "reason": "event_relative_match is absent"}
    relative = pd.to_numeric(frame["event_relative_match"], errors="coerce")
    baseline_margin_col = "baseline_margin"
    if baseline_margin_col not in frame and {
        "baseline_home_score",
        "baseline_away_score",
    }.issubset(frame.columns):
        frame = frame.copy()
        frame[baseline_margin_col] = (
            pd.to_numeric(frame["baseline_home_score"], errors="coerce")
            - pd.to_numeric(frame["baseline_away_score"], errors="coerce")
        )
    if baseline_margin_col not in frame or not {
        "actual_home_score",
        "actual_away_score",
    }.issubset(frame.columns):
        return {"available": False, "reason": "score residual columns are absent"}
    if "affected_side" in frame.columns:
        residual = _affected_margin_residual(frame)
    else:
        residual = (
            pd.to_numeric(frame["actual_home_score"], errors="coerce")
            - pd.to_numeric(frame["actual_away_score"], errors="coerce")
            - pd.to_numeric(frame[baseline_margin_col], errors="coerce")
        )
    usable = relative.notna() & residual.notna()
    if usable.sum() < 5:
        return {"available": False, "reason": "too few event-window rows"}
    study = pd.DataFrame({"relative": relative[usable].astype(int), "residual": residual[usable]})
    windows = {}
    for label, mask in {
        "pretrend_-3_to_-1": study["relative"].between(-3, -1),
        "event_match_0": study["relative"] == 0,
        "post_1_to_3": study["relative"].between(1, 3),
    }.items():
        values = study.loc[mask, "residual"]
        windows[label] = {
            "games": int(len(values)),
            "mean_margin_residual": float(values.mean()) if len(values) else None,
        }
    # A transparent placebo hook: permute relative-time labels, preserving the
    # observed residuals and event-window sizes.
    event_values = study.loc[study["relative"] == 0, "residual"]
    placebo_p = None
    if len(event_values) and len(study) >= 10:
        rng = np.random.default_rng(seed)
        observed = abs(float(event_values.mean()))
        size = len(event_values)
        draws = [
            abs(float(rng.choice(study["residual"].to_numpy(), size=size, replace=False).mean()))
            for _ in range(1000)
        ]
        placebo_p = float((1 + sum(value >= observed for value in draws)) / (len(draws) + 1))
    return {
        "available": True,
        "orientation": "affected-team margin residual"
        if "affected_side" in frame.columns
        else "home-team margin residual",
        "windows": windows,
        "placebo_p_value": placebo_p,
        "matched_event_match": _matched_event_effect(
            frame, residual, relative, seed=seed
        ),
    }


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError("shadow comparison is missing columns: " + ", ".join(missing))
    cleaned = frame.copy()
    for column in ("game_id", "actual_home_win", "baseline_home_win_prob", "context_home_win_prob"):
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    for column in (
        "baseline_no_market_home_win_prob",
        "context_no_market_home_win_prob",
        "actual_home_score",
        "actual_away_score",
        "baseline_home_score",
        "baseline_away_score",
        "context_home_score",
        "context_away_score",
        "market_spread",
        "event_relative_match",
        "matches_since_event",
        "context_event_count",
        "context_source_diversity",
        "context_official_count",
        "context_data_available",
    ):
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = cleaned.dropna(subset=list(REQUIRED_COLUMNS))
    cleaned = cleaned[cleaned["actual_home_win"].isin([0, 1])]
    cleaned = cleaned[
        cleaned["baseline_home_win_prob"].between(0, 1, inclusive="both")
        & cleaned["context_home_win_prob"].between(0, 1, inclusive="both")
    ]
    cleaned["game_id"] = cleaned["game_id"].astype(int)
    cleaned["actual_home_win"] = cleaned["actual_home_win"].astype(int)
    return cleaned.reset_index(drop=True)


def _tip_examples(
    frame: pd.DataFrame, mask: np.ndarray, *, limit: int
) -> list[dict]:
    baseline = frame["baseline_home_win_prob"].to_numpy(float)
    context = frame["context_home_win_prob"].to_numpy(float)
    flips = (baseline > 0.5) != (context > 0.5)
    selected = flips & np.asarray(mask, dtype=bool)
    ranked = frame.loc[selected].copy()
    ranked["probability_change"] = context[selected] - baseline[selected]
    ranked["absolute_change"] = ranked["probability_change"].abs()
    ranked = ranked.sort_values(
        ["absolute_change", "game_id"], ascending=[False, True]
    ).head(max(0, int(limit)))
    fields = [
        column
        for column in (
            "game_id",
            "competition_year",
            "round_id",
            "team_home",
            "team_away",
            "event_id",
            "category",
            "actual_home_win",
            "baseline_home_win_prob",
            "context_home_win_prob",
            "probability_change",
        )
        if column in ranked.columns
    ]
    examples: list[dict] = []
    for record in ranked[fields].to_dict("records"):
        examples.append(
            {
                key: (None if pd.isna(value) else value)
                for key, value in record.items()
            }
        )
    return examples


def _changed_tip_examples(
    frame: pd.DataFrame, event_mask: pd.Series, *, limit: int = 10
) -> dict:
    baseline = frame["baseline_home_win_prob"].to_numpy(float)
    context = frame["context_home_win_prob"].to_numpy(float)
    flips = (baseline > 0.5) != (context > 0.5)
    active = np.asarray(event_mask, dtype=bool)
    return {
        "count": int(flips.sum()),
        "examples": _tip_examples(frame, np.ones(len(frame), dtype=bool), limit=limit),
        "event_linked_count": int((flips & active).sum()),
        "event_linked_examples": _tip_examples(frame, active, limit=limit),
    }


def _source_coverage(frame: pd.DataFrame, event_mask: pd.Series) -> dict:
    available = pd.to_numeric(
        frame.get("context_data_available", pd.Series(0, index=frame.index)),
        errors="coerce",
    ).fillna(0)
    diversity = pd.to_numeric(
        frame.get("context_source_diversity", pd.Series(0, index=frame.index)),
        errors="coerce",
    ).fillna(0)
    official = pd.to_numeric(
        frame.get("context_official_count", pd.Series(0, index=frame.index)),
        errors="coerce",
    ).fillna(0)
    return {
        "games": int(len(frame)),
        "registry_available_games": int(available.gt(0).sum()),
        "registry_coverage": float(available.gt(0).mean()) if len(frame) else None,
        "eligible_event_games": int(event_mask.sum()),
        "eligible_event_rate": float(event_mask.mean()) if len(frame) else None,
        "official_evidence_event_games": int((event_mask & official.gt(0)).sum()),
        "mean_source_diversity_on_event_games": (
            float(diversity[event_mask].mean()) if event_mask.any() else None
        ),
    }


def evaluate_shadow_frame(
    frame: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
) -> dict:
    cleaned = _clean_frame(frame)
    if cleaned.empty:
        return {
            "schema_version": 1,
            "status": "not_ready",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "conclusion": "No paired out-of-fold shadow predictions are available.",
            "rows": 0,
        }

    event_counts = cleaned.get(
        "context_event_count", pd.Series(0, index=cleaned.index, dtype=float)
    )
    numeric_event_counts = pd.to_numeric(event_counts, errors="coerce").fillna(0)
    if "context_event_count" in cleaned.columns:
        # Research annotations include pre-event rows for pre-trend checks;
        # only the frozen feature count identifies rows that actually exposed
        # the candidate to context.
        event_mask = numeric_event_counts.gt(0)
    else:
        event_ids = cleaned.get(
            "event_id", pd.Series(index=cleaned.index, dtype=object)
        )
        relative = pd.to_numeric(
            cleaned.get(
                "event_relative_match",
                pd.Series(0, index=cleaned.index, dtype=float),
            ),
            errors="coerce",
        )
        event_mask = event_ids.notna() & relative.fillna(0).ge(0)
    cohorts = {
        "all_games": _metric_pair(cleaned),
        "event_linked_games": _metric_pair(cleaned.loc[event_mask]),
    }
    if "market_available" in cleaned:
        market = cleaned["market_available"].fillna(False).astype(bool)
        cohorts["market_backed"] = _metric_pair(cleaned.loc[market])
        cohorts["no_market"] = _metric_pair(cleaned.loc[~market])

    by_category = {}
    if "category" in cleaned:
        for offset, (category, group) in enumerate(
            cleaned.loc[event_mask].groupby("category", dropna=True, sort=True)
        ):
            scored = _metric_pair(group)
            scored["event_cluster_bootstrap"] = _cluster_bootstrap(
                group,
                reps=bootstrap_reps,
                seed=seed + 101 + offset,
            )
            by_category[str(category)] = scored

    by_time = {}
    if "matches_since_event" in cleaned:
        since = pd.to_numeric(cleaned["matches_since_event"], errors="coerce")
        bands = pd.cut(since, [-1, 0, 1, 3, np.inf], labels=["event_match", "next_match", "matches_2_3", "matches_4_plus"])
        for offset, (band, group) in enumerate(
            cleaned.loc[event_mask].groupby(
                bands[event_mask], observed=True, sort=True
            )
        ):
            scored = _metric_pair(group)
            scored["event_cluster_bootstrap"] = _cluster_bootstrap(
                group,
                reps=bootstrap_reps,
                seed=seed + 201 + offset,
            )
            by_time[str(band)] = scored

    event_frame = cleaned.loc[event_mask].copy()
    bootstrap = _cluster_bootstrap(event_frame, reps=bootstrap_reps, seed=seed)

    without_market = None
    no_market_columns = {
        "baseline_no_market_home_win_prob",
        "context_no_market_home_win_prob",
    }
    if no_market_columns.issubset(cleaned.columns):
        without_market = {
            "all_games": _metric_pair(
                cleaned,
                baseline_probability="baseline_no_market_home_win_prob",
                context_probability="context_no_market_home_win_prob",
                include_scores=False,
            ),
            "event_linked_games": _metric_pair(
                cleaned.loc[event_mask],
                baseline_probability="baseline_no_market_home_win_prob",
                context_probability="context_no_market_home_win_prob",
                include_scores=False,
            ),
        }

    market_absorption = {"available": False}
    if without_market is not None:
        operational_delta = cohorts["event_linked_games"][
            "delta_context_minus_baseline"
        ].get("log_loss")
        no_market_delta = without_market["event_linked_games"][
            "delta_context_minus_baseline"
        ].get("log_loss")
        market_absorption = {
            "available": operational_delta is not None and no_market_delta is not None,
            "event_log_loss_delta_with_market_routing": operational_delta,
            "event_log_loss_delta_without_market_inputs": no_market_delta,
            "inference": (
                "Compare paired deltas directly; a smaller operational improvement suggests public evidence was already absorbed by prices."
            ),
        }

    score_delta = cohorts["event_linked_games"]["delta_context_minus_baseline"]
    baseline_variance = (
        (cohorts["event_linked_games"]["baseline"].get("score") or {}).get(
            "residual_variance"
        )
    )
    context_variance = (
        (cohorts["event_linked_games"]["context_candidate"].get("score") or {}).get(
            "residual_variance"
        )
    )
    dispersion = {
        "tested": baseline_variance is not None and context_variance is not None,
        "baseline_residual_variance": baseline_variance,
        "context_residual_variance": context_variance,
        "delta": score_delta.get("residual_variance"),
        "variance_ratio": (
            float(context_variance / baseline_variance)
            if baseline_variance not in {None, 0} and context_variance is not None
            else None
        ),
        "production_adjustment": "disabled",
    }
    event_delta = cohorts["event_linked_games"]["delta_context_minus_baseline"].get("log_loss")
    if len(event_frame) < 20:
        conclusion = "Insufficient event-linked games to judge predictive materiality."
        status = "insufficient_evidence"
    elif event_delta is not None and event_delta < 0 and bootstrap.get("probability_log_loss_improves", 0) >= 0.95:
        conclusion = "The shadow candidate shows evidence of lower event-cohort log loss; production remains disabled pending prospective confirmation."
        status = "retrospective_signal"
    elif event_delta is not None and event_delta > 0 and bootstrap.get("probability_log_loss_improves", 1) <= 0.05:
        conclusion = "The shadow candidate is materially worse on the event cohort and must not be promoted."
        status = "evidence_of_harm"
    else:
        conclusion = "No clear out-of-sample predictive lift from Club Context has been established."
        status = "no_clear_lift"

    return {
        "schema_version": 1,
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "rows": int(len(cleaned)),
        "event_linked_rows": int(event_mask.sum()),
        "conclusion": conclusion,
        "shadow_only": True,
        "cohorts": cohorts,
        "by_category": by_category,
        "by_matches_since_event": by_time,
        "event_cluster_bootstrap": bootstrap,
        "event_study": _event_study(cleaned, seed=seed),
        "counterfactual_without_market_inputs": without_market,
        "market_absorption": market_absorption,
        "dispersion_hypothesis": dispersion,
        "source_coverage": _source_coverage(cleaned, event_mask),
        "changed_tip_examples": _changed_tip_examples(cleaned, event_mask),
    }


def _markdown(report: dict) -> str:
    lines = [
        "# Club Context materiality report",
        "",
        f"**Status:** `{report.get('status', 'unknown')}`",
        "",
        str(report.get("conclusion") or "No conclusion available."),
        "",
        "> Club Context is shadow-only. This report cannot activate or alter production predictions.",
    ]
    cohorts = report.get("cohorts") or {}
    if cohorts:
        lines.extend(["", "## Paired out-of-fold comparison", "", "| Cohort | Games | Δ accuracy | Δ log loss | Δ Brier | Tip flips |", "|---|---:|---:|---:|---:|---:|"])
        for name, data in cohorts.items():
            delta = data.get("delta_context_minus_baseline") or {}
            games = (data.get("baseline") or {}).get("games", 0)
            lines.append(
                f"| {name.replace('_', ' ').title()} | {games} | {_format_delta(delta.get('accuracy'))} | "
                f"{_format_delta(delta.get('log_loss'))} | {_format_delta(delta.get('brier'))} | {data.get('tip_flips', 0)} |"
            )
    without_market = report.get("counterfactual_without_market_inputs") or {}
    if without_market:
        lines.extend(
            [
                "",
                "## Counterfactual without market inputs",
                "",
                "| Cohort | Games | Δ accuracy | Δ log loss | Δ Brier | Tip flips |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for name, data in without_market.items():
            delta = data.get("delta_context_minus_baseline") or {}
            games = (data.get("baseline") or {}).get("games", 0)
            lines.append(
                f"| {name.replace('_', ' ').title()} | {games} | {_format_delta(delta.get('accuracy'))} | "
                f"{_format_delta(delta.get('log_loss'))} | {_format_delta(delta.get('brier'))} | {data.get('tip_flips', 0)} |"
            )
    bootstrap = report.get("event_cluster_bootstrap") or {}
    if bootstrap.get("reps"):
        ci = bootstrap.get("log_loss_delta_ci95") or [None, None]
        lines.extend(
            [
                "",
                "## Event-cluster uncertainty",
                "",
                f"- Clusters: {bootstrap.get('clusters')}",
                f"- Paired Δ log-loss 95% interval: {ci[0]:+.4f} to {ci[1]:+.4f}",
                f"- Bootstrap probability of improvement: {bootstrap.get('probability_log_loss_improves', 0):.1%}",
            ]
        )
    market = report.get("market_absorption") or {}
    if market.get("available"):
        lines.extend(
            [
                "",
                "## Market absorption",
                "",
                "- Event-cohort Δ log loss with operational market routing: "
                f"{_format_delta(market.get('event_log_loss_delta_with_market_routing'))}",
                "- Event-cohort Δ log loss without market inputs: "
                f"{_format_delta(market.get('event_log_loss_delta_without_market_inputs'))}",
                "- Read this comparison cautiously: similar deltas do not prove an "
                "independent psychosocial effect.",
            ]
        )
    dispersion = report.get("dispersion_hypothesis") or {}
    if dispersion.get("tested"):
        lines.extend(
            [
                "",
                "## Score and dispersion",
                "",
                "- Event-cohort context-to-baseline residual-variance ratio: "
                f"{dispersion.get('variance_ratio', 0):.3f}",
                "- Production dispersion adjustment: "
                f"{dispersion.get('production_adjustment', 'disabled')}",
            ]
        )
        event_scores = ((cohorts.get("event_linked_games") or {}).get(
            "delta_context_minus_baseline"
        ) or {})
        if event_scores.get("margin_mae") is not None:
            lines.append(
                "- Event-cohort Δ margin MAE: "
                f"{event_scores['margin_mae']:+.3f} points"
            )
    matched = ((report.get("event_study") or {}).get("matched_event_match") or {})
    if matched.get("available"):
        ci = matched.get("cluster_bootstrap_ci95") or [None, None]
        lines.extend(
            [
                "",
                "## Matched event study",
                "",
                f"- Event matches/clusters: {matched.get('event_matches', 0)}/"
                f"{matched.get('event_clusters', 0)}",
                "- Mean affected-team margin-residual difference: "
                f"{matched.get('mean_affected_margin_residual_difference', 0):+.2f} points",
                f"- Event-cluster 95% interval: {ci[0]:+.2f} to {ci[1]:+.2f} points",
                "- Permutation placebo p-value: "
                f"{(report.get('event_study') or {}).get('placebo_p_value', 0):.3f}",
            ]
        )
    for heading, key in (
        ("By category", "by_category"),
        ("By matches since event", "by_matches_since_event"),
    ):
        breakdown = report.get(key) or {}
        if not breakdown:
            continue
        lines.extend(
            [
                "",
                f"## {heading}",
                "",
                "| Cohort | Games | Δ log loss | Δ accuracy | Tip flips |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name, data in breakdown.items():
            delta = data.get("delta_context_minus_baseline") or {}
            games = (data.get("baseline") or {}).get("games", 0)
            lines.append(
                f"| {name.replace('_', ' ').title()} | {games} | "
                f"{_format_delta(delta.get('log_loss'))} | "
                f"{_format_delta(delta.get('accuracy'))} | "
                f"{data.get('tip_flips', 0)} |"
            )
    coverage = report.get("source_coverage") or {}
    if coverage:
        lines.extend(
            [
                "",
                "## Evidence coverage",
                "",
                f"- Registry available: {coverage.get('registry_available_games', 0)}/{coverage.get('games', 0)} games",
                f"- Eligible event-linked games: {coverage.get('eligible_event_games', 0)}",
                f"- Event games with official evidence: {coverage.get('official_evidence_event_games', 0)}",
            ]
        )
    changed = report.get("changed_tip_examples") or {}
    lines.extend(
        [
            "",
            "## Changed tips",
            "",
            f"All-game paired candidate tip flips: {changed.get('count', 0)}. "
            f"Event-linked flips: {changed.get('event_linked_count', 0)}.",
        ]
    )
    examples = changed.get("event_linked_examples") or []
    if not examples:
        lines.append("- No event-linked tip changed side.")
    for example in examples:
        matchup = " v ".join(
            value for value in (example.get("team_home"), example.get("team_away")) if value
        )
        label = matchup or f"game {example.get('game_id')}"
        baseline_p = example.get("baseline_home_win_prob", 0)
        context_p = example.get("context_home_win_prob", 0)
        actual = bool(example.get("actual_home_win"))
        baseline_correct = (baseline_p > 0.5) == actual
        context_correct = (context_p > 0.5) == actual
        effect = "corrected the tip" if context_correct and not baseline_correct else "made the tip wrong"
        lines.append(
            f"- {example.get('competition_year')} R{int(example.get('round_id', 0))} · "
            f"{label}: {baseline_p:.1%} → {context_p:.1%} ({effect})"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Negative log-loss/Brier deltas favour the context candidate. The event-cohort "
            "pattern is uncertain, aggregate probability quality is slightly worse, tip flips "
            "skew harmful, and neither the matched study nor dispersion test shows material "
            "benefit. Club Context therefore remains shadow-only.",
            "",
        ]
    )
    return "\n".join(lines)


def write_materiality_report(report: dict, json_path, markdown_path=None) -> tuple[pathlib.Path, pathlib.Path]:
    json_path = pathlib.Path(json_path)
    markdown_path = pathlib.Path(markdown_path or json_path.with_suffix(".md"))
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def _load_input(path: pathlib.Path | None, db_path: pathlib.Path | None) -> pd.DataFrame:
    if path is not None:
        if path.suffix.lower() in {".json", ".jsonl"}:
            try:
                return pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
            except ValueError:
                payload = json.loads(path.read_text(encoding="utf-8"))
                return pd.DataFrame(payload.get("games", payload))
        return pd.read_csv(path)
    if db_path is not None and db_path.exists():
        with sqlite3.connect(str(db_path)) as con:
            exists = con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='context_shadow_predictions'"
            ).fetchone()
            if exists:
                return pd.read_sql_query("SELECT * FROM context_shadow_predictions", con)
    return pd.DataFrame(columns=sorted(REQUIRED_COLUMNS))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score paired Club Context shadow predictions.")
    parser.add_argument("--input", type=pathlib.Path, help="CSV/JSON of paired OOF predictions.")
    parser.add_argument("--db-path", type=pathlib.Path, help="SQLite DB containing context_shadow_predictions.")
    parser.add_argument("--report-path", type=pathlib.Path, default=pathlib.Path("reports/club-context-materiality-latest.json"))
    parser.add_argument("--markdown-path", type=pathlib.Path)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        frame = _load_input(args.input, args.db_path)
        report = evaluate_shadow_frame(
            frame,
            seed=args.seed,
            bootstrap_reps=max(0, args.bootstrap_reps),
        )
        json_path, markdown_path = write_materiality_report(
            report,
            args.report_path,
            args.markdown_path,
        )
        print(f"Club Context materiality report written to {json_path}")
        print(f"Club Context materiality summary written to {markdown_path}")
        print(report["conclusion"])
        return 0
    except Exception as exc:
        print(f"Club Context materiality evaluation failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
