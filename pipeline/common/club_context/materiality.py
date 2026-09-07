"""Cohort-restricted materiality test for Club Context.

The shipped ablation asks "does adding fifty-one columns improve the whole
model?".  At 2.8% exposure that question is nearly unanswerable: refitting
LightGBM with the extra columns perturbs every one of the 3,180 held-out rows,
so the resulting tip flips are dominated by refit noise on games with no event
at all.

This module asks the narrower question the evidence can actually answer.  It
holds the production baseline fixed and fits a small, pre-declared offset on the
baseline logit, using signed exposure so the offset is structurally zero
wherever no event applies.  Unexposed rows therefore stay byte-identical to
baseline and every reported flip is genuinely event-linked.

Estimation is leave-one-event-cluster-out, so no game contributes to the
coefficient that scores it, and a club's whole event window moves together
rather than leaking across its own matches.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


EPSILON = 1e-6

# Pre-declared specifications, reported together.  Listing them in advance and
# publishing all three is the guard against picking a winner after seeing the
# fit; the primary is the simplest one.
OFFSET_SPECIFICATIONS: dict[str, tuple[str, ...]] = {
    "side": ("club_context_affected_side",),
    "exposure": ("club_context_affected_exposure",),
    "side_and_exposure": (
        "club_context_affected_side",
        "club_context_affected_exposure",
    ),
}
PRIMARY_SPECIFICATION = "side"

DEFAULT_RIDGE = 1.0
DEFAULT_BOOTSTRAP_REPS = 2000
DEFAULT_SEED = 20100308


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), EPSILON, 1.0 - EPSILON)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(value, dtype=float)))


def fit_offset_coefficients(
    design: np.ndarray,
    outcome: np.ndarray,
    offset: np.ndarray,
    *,
    ridge: float = DEFAULT_RIDGE,
    max_iterations: int = 100,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """Ridge-penalised logistic regression with a fixed offset, no intercept.

    Newton/IRLS.  The problem is a handful of parameters over a few thousand
    rows, so an explicit solve is both exact and instant.  There is deliberately
    no intercept: one would shift every fixture in the competition, including
    the 97% with no event.
    """

    design = np.asarray(design, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    offset = np.asarray(offset, dtype=float)
    if design.ndim == 1:
        design = design.reshape(-1, 1)
    beta = np.zeros(design.shape[1], dtype=float)
    if design.shape[0] == 0:
        return beta
    penalty = ridge * np.eye(design.shape[1])
    for _ in range(max_iterations):
        # Clipping the linear predictor keeps a wide-scaled column from
        # overflowing the exponential on the first Newton step; the ridge alone
        # does not bound it.
        linear = np.clip(offset + design @ beta, -30.0, 30.0)
        mu = _sigmoid(linear)
        weights = np.clip(mu * (1.0 - mu), 1e-9, None)
        gradient = design.T @ (outcome - mu) - ridge * beta
        hessian = design.T @ (design * weights[:, None]) + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)):
            break
        # Trust region: a separated cohort can otherwise take an enormous first
        # step and never come back.
        largest = float(np.max(np.abs(step)))
        if largest > 1.0:
            step = step / largest
            largest = 1.0
        beta = beta + step
        if largest < tolerance:
            break
    return beta


def _out_of_fold_offsets(
    design: np.ndarray,
    outcome: np.ndarray,
    offset: np.ndarray,
    clusters: np.ndarray,
    *,
    ridge: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Leave-one-event-cluster-out linear offsets for every row.

    Rows outside every cluster (no event) carry a zero design row, so they
    neither influence the fit nor receive an adjustment.  They are still scored,
    which is what makes the all-games numbers honest.
    """

    linear = np.zeros(design.shape[0], dtype=float)
    labels = pd.Series(clusters).fillna("").astype(str).to_numpy()
    unique = sorted({label for label in labels if label})
    folds = []
    for label in unique:
        held = labels == label
        beta = fit_offset_coefficients(
            design[~held], outcome[~held], offset[~held], ridge=ridge
        )
        linear[held] = design[held] @ beta
        folds.append({"cluster": label, "coefficients": [float(v) for v in beta]})
    full = fit_offset_coefficients(design, outcome, offset, ridge=ridge)
    return linear, {
        "clusters": len(unique),
        "full_sample_coefficients": [float(value) for value in full],
        "fold_coefficient_mean": (
            [
                float(np.mean([fold["coefficients"][index] for fold in folds]))
                for index in range(design.shape[1])
            ]
            if folds
            else [0.0] * design.shape[1]
        ),
    }


def _probability_metrics(outcome: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    probability = np.clip(probability, EPSILON, 1.0 - EPSILON)
    losses = -(
        outcome * np.log(probability) + (1.0 - outcome) * np.log(1.0 - probability)
    )
    return {
        "games": int(outcome.size),
        "log_loss": float(losses.mean()) if outcome.size else float("nan"),
        "brier": float(np.mean((probability - outcome) ** 2)) if outcome.size else float("nan"),
        "accuracy": float(np.mean((probability > 0.5) == (outcome > 0.5)))
        if outcome.size
        else float("nan"),
    }


def _cluster_bootstrap(
    values: np.ndarray,
    clusters: np.ndarray,
    *,
    reps: int,
    seed: int,
) -> dict[str, float]:
    labels = pd.Series(clusters).astype(str).to_numpy()
    groups = {label: values[labels == label] for label in sorted(set(labels))}
    keys = list(groups)
    if not keys:
        return {"available": False}
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, dtype=float)
    for index in range(reps):
        picked = rng.choice(keys, size=len(keys), replace=True)
        draws[index] = float(np.concatenate([groups[key] for key in picked]).mean())
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "available": True,
        "clusters": len(keys),
        "mean": float(values.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "probability_of_improvement": float((draws < 0).mean()),
    }


def _annotated_side(frame: pd.DataFrame) -> np.ndarray:
    """Orientation for every annotated row, event window or not.

    ``club_context_affected_side`` is zero outside the event window by design.
    The placebo needs the same club's orientation on its pre-event games, which
    the paired frame carries as the textual ``affected_side`` annotation.
    """

    side = frame.get("affected_side")
    if side is None:
        return np.zeros(len(frame), dtype=float)
    return (
        side.map({"home": 1.0, "away": -1.0}).fillna(0.0).to_numpy(float)
    )


def _prevalence(frame: pd.DataFrame, exposed: np.ndarray) -> dict[str, Any]:
    seasons = int(pd.to_numeric(frame.get("competition_year"), errors="coerce").nunique())
    seasons = max(seasons, 1)
    games = int(len(frame))
    exposed_games = int(exposed.sum())
    return {
        "games": games,
        "exposed_games": exposed_games,
        "exposure_rate": float(exposed_games / games) if games else 0.0,
        "seasons": seasons,
        "games_per_season": float(games / seasons),
        "exposed_games_per_season": float(exposed_games / seasons),
    }


def _season_impact(prevalence: dict[str, Any], exposed_delta: dict[str, float]) -> dict[str, Any]:
    """Translate a cohort effect into what it is worth over a whole season.

    A layer can be statistically real and still not matter.  This is the
    arithmetic that decides which of those two a result is.
    """

    per_season = prevalence["exposed_games_per_season"]
    # Accuracy delta is candidate minus baseline, so a positive delta is more
    # correct tips.  Log loss runs the other way and is left as measured.
    accuracy_delta = exposed_delta.get("accuracy", 0.0)
    log_loss_delta = exposed_delta.get("log_loss", 0.0)
    tips = accuracy_delta * per_season
    return {
        "exposed_games_per_season": per_season,
        "extra_correct_tips_per_season": float(tips),
        "seasons_per_extra_tip": float(1.0 / tips) if abs(tips) > 1e-9 else None,
        "season_log_loss_delta": float(
            log_loss_delta * per_season / max(prevalence["games_per_season"], 1.0)
        ),
    }


def _minimum_detectable_effect(residuals: np.ndarray, clusters: np.ndarray) -> dict[str, Any]:
    """Roughly, the smallest affected-side bias this sample could resolve.

    Uses the between-cluster standard error, because the games inside one event
    window are not independent observations of that window.
    """

    labels = pd.Series(clusters).astype(str).to_numpy()
    means = np.array(
        [residuals[labels == label].mean() for label in sorted(set(labels))],
        dtype=float,
    )
    if means.size < 2:
        return {"available": False}
    standard_error = float(means.std(ddof=1) / math.sqrt(means.size))
    return {
        "available": True,
        "clusters": int(means.size),
        "cluster_standard_error": standard_error,
        # 1.96 for 95% coverage plus 0.84 for 80% power.
        "minimum_detectable_effect": float(2.80 * standard_error),
        "observed_effect": float(means.mean()),
    }


def evaluate_context_offset(
    frame: pd.DataFrame,
    *,
    ridge: float = DEFAULT_RIDGE,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    seed: int = DEFAULT_SEED,
    specifications: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    """Score every pre-declared offset specification against the fixed baseline."""

    specs = specifications or OFFSET_SPECIFICATIONS
    required = {"baseline_home_win_prob", "actual_home_win"}
    missing = sorted(required - set(frame.columns))
    if missing:
        return {
            "status": "not_ready",
            "reason": f"paired frame is missing: {', '.join(missing)}",
        }

    work = frame.copy()
    outcome = pd.to_numeric(work["actual_home_win"], errors="coerce").to_numpy(float)
    baseline = pd.to_numeric(work["baseline_home_win_prob"], errors="coerce").to_numpy(float)
    valid = np.isfinite(outcome) & np.isfinite(baseline)
    work = work.loc[valid].reset_index(drop=True)
    outcome = outcome[valid]
    baseline = baseline[valid]
    if work.empty:
        return {"status": "not_ready", "reason": "no scoreable paired rows"}

    offset = _logit(baseline)
    clusters = (
        work["event_key"].astype("string")
        if "event_key" in work.columns
        else pd.Series([pd.NA] * len(work), dtype="string")
    )
    if clusters.isna().all() and "event_id" in work.columns:
        clusters = work["event_id"].astype("string")
    clusters = clusters.fillna("").to_numpy()

    available = [
        name
        for name, columns in specs.items()
        if all(column in work.columns for column in columns)
    ]
    if not available:
        return {
            "status": "not_ready",
            "reason": "no orientation columns are present in the paired frame",
        }

    # An event annotation covers the games around an event, including the ones
    # before it that serve as its own controls.  Exposure means the event
    # actually applied to this fixture, which is exactly where the offset is
    # allowed to move anything.
    orientation = (
        pd.to_numeric(work.get("club_context_affected_side"), errors="coerce")
        .fillna(0.0)
        .to_numpy(float)
    )
    if not np.any(orientation != 0) and "context_event_count" in work.columns:
        active = (
            pd.to_numeric(work["context_event_count"], errors="coerce").fillna(0.0) > 0
        ).to_numpy()
    else:
        active = orientation != 0
    exposed = active & (clusters != "")
    prevalence = _prevalence(work, exposed)
    baseline_metrics = {
        "all_games": _probability_metrics(outcome, baseline),
        "exposed": _probability_metrics(outcome[exposed], baseline[exposed]),
    }

    # Affected-side calibration: the raw question of whether the baseline is
    # biased on these games at all, before any model is fitted to them.
    affected = exposed & (orientation != 0)
    affected_probability = np.where(orientation > 0, baseline, 1.0 - baseline)
    affected_outcome = np.where(orientation > 0, outcome, 1.0 - outcome)
    affected_residual = (affected_outcome - affected_probability)[affected]
    calibration = _cluster_bootstrap(
        affected_residual,
        clusters[affected],
        reps=bootstrap_reps,
        seed=seed,
    )
    calibration["direction"] = (
        "affected side underperforms the baseline probability"
        if calibration.get("mean", 0.0) < 0
        else "affected side outperforms the baseline probability"
    )
    power = _minimum_detectable_effect(affected_residual, clusters[affected])

    # Pre-event placebo.  Events are selected on underperformance, so the same
    # clubs were probably already losing more than the model expected before
    # anything happened.  If the affected-side residual is just as negative
    # before the event as after it, the layer is measuring selection, not the
    # event.
    placebo = {"available": False, "reason": "no relative-match annotation"}
    if "event_relative_match" in work.columns:
        relative = pd.to_numeric(work["event_relative_match"], errors="coerce").to_numpy()
        annotated_side = (
            pd.to_numeric(work.get("affected_side_sign"), errors="coerce")
            .fillna(0.0)
            .to_numpy(float)
            if "affected_side_sign" in work.columns
            else _annotated_side(work)
        )
        before = (
            np.isfinite(relative)
            & (relative < 0)
            & (clusters != "")
            & (annotated_side != 0)
        )
        if before.sum() >= 5:
            placebo_probability = np.where(annotated_side > 0, baseline, 1.0 - baseline)
            placebo_outcome = np.where(annotated_side > 0, outcome, 1.0 - outcome)
            placebo_residual = (placebo_outcome - placebo_probability)[before]
            placebo = _cluster_bootstrap(
                placebo_residual,
                clusters[before],
                reps=bootstrap_reps,
                seed=seed,
            )
            placebo["games"] = int(before.sum())
            placebo["reading"] = (
                "the same bias is present before the event, so this cohort is "
                "selected on underperformance"
                if placebo.get("ci95_high", 1.0) < 0
                else "no comparable bias before the event"
            )
        else:
            placebo = {"available": False, "reason": "too few pre-event rows"}

    results: dict[str, Any] = {}
    for name in available:
        columns = specs[name]
        design = np.column_stack(
            [
                pd.to_numeric(work[column], errors="coerce").fillna(0.0).to_numpy(float)
                for column in columns
            ]
        )
        # Enforce the no-event-no-change contract at the design matrix, so it
        # cannot be broken by an annotation that reaches beyond the event window.
        design[~exposed] = 0.0
        fold_labels = np.where(exposed, clusters, "")
        linear, fit = _out_of_fold_offsets(
            design, outcome, offset, fold_labels, ridge=ridge
        )
        adjusted = _sigmoid(offset + linear)
        # The contract that makes this test honest: no event, no change.
        untouched = int(np.sum(np.abs(adjusted[~exposed] - baseline[~exposed]) > 1e-12))
        flips = (baseline > 0.5) != (adjusted > 0.5)
        paired_delta = (
            -(
                outcome * np.log(np.clip(adjusted, EPSILON, 1 - EPSILON))
                + (1 - outcome) * np.log(np.clip(1 - adjusted, EPSILON, 1 - EPSILON))
            )
            + (
                outcome * np.log(np.clip(baseline, EPSILON, 1 - EPSILON))
                + (1 - outcome) * np.log(np.clip(1 - baseline, EPSILON, 1 - EPSILON))
            )
        )
        cohorts = {}
        for label, mask in (("all_games", np.ones_like(exposed)), ("exposed", exposed)):
            after = _probability_metrics(outcome[mask], adjusted[mask])
            before = _probability_metrics(outcome[mask], baseline[mask])
            cohorts[label] = {
                "baseline": before,
                "candidate": after,
                "delta": {
                    key: float(after[key] - before[key])
                    for key in ("log_loss", "brier", "accuracy")
                },
            }
        correct_before = (baseline > 0.5) == (outcome > 0.5)
        correct_after = (adjusted > 0.5) == (outcome > 0.5)
        results[name] = {
            "columns": list(columns),
            "fit": fit,
            "cohorts": cohorts,
            "unexposed_rows_changed": untouched,
            "tip_flips": int(flips.sum()),
            "tip_flips_outside_event_cohort": int(np.sum(flips & ~exposed)),
            "flips_to_correct": int(np.sum(flips & correct_after & ~correct_before)),
            "flips_to_wrong": int(np.sum(flips & correct_before & ~correct_after)),
            "paired_log_loss_delta": _cluster_bootstrap(
                paired_delta[exposed], clusters[exposed], reps=bootstrap_reps, seed=seed
            ),
            "season_impact": _season_impact(
                prevalence, cohorts["exposed"]["delta"]
            ),
        }

    primary = PRIMARY_SPECIFICATION if PRIMARY_SPECIFICATION in results else available[0]
    return {
        "status": "ok",
        "primary_specification": primary,
        "estimation": {
            "method": "leave-one-event-cluster-out ridge logistic offset",
            "ridge": float(ridge),
            "intercept": False,
            "seed": int(seed),
            "bootstrap_reps": int(bootstrap_reps),
        },
        "prevalence": prevalence,
        "baseline": baseline_metrics,
        "affected_side_calibration": calibration,
        "pre_event_placebo": placebo,
        "power": power,
        "specifications": results,
        "shadow_only": True,
    }


# ---------------------------------------------------------------------------
# High-prevalence comparator
# ---------------------------------------------------------------------------


MIN_PRIOR_GAMES = 6


def _team_game_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match, with the baseline oriented to that team."""

    required = {
        "competition_year",
        "round_id",
        "game_id",
        "team_home",
        "team_away",
        "baseline_home_win_prob",
        "actual_home_win",
    }
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    home = pd.DataFrame(
        {
            "competition_year": frame["competition_year"],
            "round_id": frame["round_id"],
            "game_id": frame["game_id"],
            "team": frame["team_home"],
            "probability": pd.to_numeric(frame["baseline_home_win_prob"], errors="coerce"),
            "outcome": pd.to_numeric(frame["actual_home_win"], errors="coerce"),
        }
    )
    away = home.copy()
    away["team"] = frame["team_away"]
    away["probability"] = 1.0 - home["probability"]
    away["outcome"] = 1.0 - home["outcome"]
    stacked = pd.concat([home, away], ignore_index=True)
    return stacked.dropna(subset=["probability", "outcome"])


def season_shortfall(frame: pd.DataFrame) -> pd.DataFrame:
    """Season-to-date wins minus the model's own expected wins, per team-game.

    Both terms are cumulative over games strictly before the one being scored,
    so the value is available at prediction time. It asks a narrow question: has
    this club, so far this season, done better or worse than the model itself
    said it would?
    """

    stacked = _team_game_frame(frame)
    if stacked.empty:
        return stacked
    ordered = stacked.sort_values(["team", "competition_year", "round_id", "game_id"])
    grouped = ordered.groupby(["team", "competition_year"], group_keys=False)
    ordered["prior_games"] = grouped.cumcount()
    ordered["expected_prior"] = grouped["probability"].transform(
        lambda values: values.shift(1).expanding().sum()
    )
    ordered["actual_prior"] = grouped["outcome"].transform(
        lambda values: values.shift(1).expanding().sum()
    )
    ordered["shortfall_rate"] = (
        ordered["actual_prior"] - ordered["expected_prior"]
    ) / ordered["prior_games"].replace(0, np.nan)
    ordered["cluster"] = (
        ordered["team"].astype(str) + "-" + ordered["competition_year"].astype(str)
    )
    return ordered


def evaluate_form_shortfall(
    frame: pd.DataFrame,
    *,
    ridge: float = DEFAULT_RIDGE,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """The same offset method applied to a comparator that touches most games.

    Club Context reaches 2.8% of fixtures, which caps what it can be worth no
    matter how real its effect is.  This measures the identical mechanism, a
    club performing away from what the model expects, on a state variable
    defined for every club in every round after the opening month.  It exists to
    make the prevalence argument concrete and reproducible, and it activates
    nothing.
    """

    stacked = season_shortfall(frame)
    if stacked.empty:
        return {"available": False, "reason": "paired frame lacks team/outcome columns"}
    measured = stacked[stacked["prior_games"] >= MIN_PRIOR_GAMES].dropna(
        subset=["shortfall_rate"]
    )
    if len(measured) < 200:
        return {"available": False, "reason": "too few measured team-games"}

    residual = (measured["outcome"] - measured["probability"]).to_numpy(float)
    shortfall = measured["shortfall_rate"].to_numpy(float)
    clusters = measured["cluster"].to_numpy()

    labels = sorted(set(clusters))
    groups = {label: np.where(clusters == label)[0] for label in labels}
    rng = np.random.default_rng(seed)
    slopes = np.empty(min(bootstrap_reps, 2000), dtype=float)
    for index in range(slopes.size):
        picked = np.concatenate(
            [groups[label] for label in rng.choice(labels, size=len(labels), replace=True)]
        )
        slopes[index] = float(np.polyfit(shortfall[picked], residual[picked], 1)[0])
    low, high = np.quantile(slopes, [0.025, 0.975])

    lookup = measured.set_index(["game_id", "team"])["shortfall_rate"]
    home = pd.MultiIndex.from_arrays([frame["game_id"], frame["team_home"]])
    away = pd.MultiIndex.from_arrays([frame["game_id"], frame["team_away"]])
    delta = lookup.reindex(home).to_numpy(float) - lookup.reindex(away).to_numpy(float)
    outcome = pd.to_numeric(frame["actual_home_win"], errors="coerce").to_numpy(float)
    baseline = pd.to_numeric(frame["baseline_home_win_prob"], errors="coerce").to_numpy(float)
    seasons = pd.to_numeric(frame["competition_year"], errors="coerce").to_numpy(float)
    usable = np.isfinite(delta) & np.isfinite(outcome) & np.isfinite(baseline)
    if usable.sum() < 200:
        return {"available": False, "reason": "too few matches with both sides measured"}

    delta, outcome, baseline, seasons = (
        delta[usable],
        outcome[usable],
        baseline[usable],
        seasons[usable],
    )
    offset = _logit(baseline)
    design = delta.reshape(-1, 1)
    adjusted_linear = np.zeros(delta.size, dtype=float)
    # Leave-one-season-out rather than leave-one-cluster-out: the comparator is
    # a season-shaped state variable, so a whole season is the honest fold.
    for season in sorted(set(seasons)):
        held = seasons == season
        beta = fit_offset_coefficients(
            design[~held], outcome[~held], offset[~held], ridge=ridge
        )
        adjusted_linear[held] = design[held] @ beta
    adjusted = _sigmoid(offset + adjusted_linear)

    before = _probability_metrics(outcome, baseline)
    after = _probability_metrics(outcome, adjusted)
    season_count = max(len(set(seasons)), 1)
    return {
        "available": True,
        "prevalence": {
            "measured_matches": int(usable.sum()),
            "paired_matches": int(len(frame)),
            "coverage": float(usable.sum() / max(len(frame), 1)),
            "minimum_prior_games": MIN_PRIOR_GAMES,
        },
        "slope": {
            "value": float(np.polyfit(shortfall, residual, 1)[0]),
            "ci95_low": float(low),
            "ci95_high": float(high),
            "clusters": len(labels),
            "team_games": int(len(measured)),
        },
        "baseline": before,
        "candidate": after,
        "delta": {
            key: float(after[key] - before[key])
            for key in ("log_loss", "brier", "accuracy")
        },
        "tip_flips": int(np.sum((baseline > 0.5) != (adjusted > 0.5))),
        "extra_correct_tips_per_season": float(
            (after["accuracy"] - before["accuracy"]) * usable.sum() / season_count
        ),
        "estimation": "leave-one-season-out ridge logistic offset, no intercept",
        "shadow_only": True,
    }
