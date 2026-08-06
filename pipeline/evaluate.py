# Description: Honest nested season-out evaluation of the full prediction stack.
#
# train.py's printed metrics are slightly optimistic: the blend weights,
# stacker, and calibrator are fitted on OOF rows that include the most recent
# season. This script evaluates each held-out season Y by fitting the entire
# meta-layer (blend weights -> stacker -> calibrator) only on seasons < Y.
# The Tier-B/Tier-C inputs come from the expanding-window OOF generators, so
# predictions for season Y only ever use models trained on seasons < Y.
print("Running the evaluate.py script...")

import json
import os
import pathlib
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.metrics import brier_score_loss, log_loss

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from pipeline.common import console
from pipeline.common.model_prediciton import prediction_functions as pf
from pipeline.common.lineups import features as lf
from pipeline.common.model_training import calibration as calib
from pipeline.common.model_training import comp_sim
from pipeline.common.model_training import distributional_metrics as dm
from pipeline.common.model_training import modelling_functions as mf
from pipeline.common.model_training import tier_a_baseline as tb
from pipeline.common.model_training import training_config as tc
from pipeline.runtime_paths import (
    database_path,
    models_path,
    project_root as configured_project_root,
)


# Central-interval half-widths for a normal, in standard deviations.
NORMAL_50_Z = 0.6744897501960817
NORMAL_90_Z = 1.6448536269514722

# What inference serves. The reconciliation scorecard scores every combination
# and labels this one `shipped`, so the report always says which is deployed.
RECONCILE_MODE = "on_conflict"
DISPLAY_MODE = "median"
RECONCILIATION_VARIANTS = (
    "on_conflict_median",
    "on_conflict_mode",
    "always_median",
    "always_mode",
    "legacy",
)


def _prediction_metrics(y, probabilities):
    """Return directly comparable binary-probability metrics."""
    y = np.asarray(y, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(y) != len(probabilities):
        raise ValueError("labels and probabilities must have the same length")
    if not len(y):
        return {
            "games": 0,
            "correct": 0,
            "accuracy": None,
            "log_loss": None,
            "brier": None,
        }
    correct = int(((probabilities > 0.5) == y.astype(bool)).sum())
    return {
        "games": int(len(y)),
        "correct": correct,
        "accuracy": float(correct / len(y)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "brier": float(brier_score_loss(y, probabilities)),
    }


def _legacy_reconciled_prediction(
    mu_home,
    mu_away,
    calibrated_cond,
    n_simulations,
    lambda3,
    dispersion_home,
    dispersion_away,
    rng,
):
    """The reconciliation the constraint-native solve replaced.

    Kept here, and nowhere else, purely so the fix can be priced. It draws from
    the *raw* score means and then importance-reweights the simulated games so
    the reported margin carries the calibrated probability: home wins get
    `cal / raw_cond` and away wins the complement, leaving a discontinuity in
    the reported margin at zero. Nothing in the pipeline calls this.

    Returns the modal scoreline on the tipped side, the weighted median margin,
    and the weighted ensemble, so the point predictions and the distribution it
    implied can both be scored.
    """
    home, away = pf.draw_score_samples(
        mu_home,
        mu_away,
        n_simulations,
        lambda3=lambda3,
        dispersion_home=dispersion_home,
        dispersion_away=dispersion_away,
        rng=rng,
    )
    margins = home - away
    home_wins = int((margins > 0).sum())
    away_wins = int((margins < 0).sum())

    cal = float(np.clip(calibrated_cond, 1e-6, 1 - 1e-6))
    raw_cond = float(
        np.clip(home_wins / max(1, home_wins + away_wins), 1e-6, 1 - 1e-6)
    )
    weights = np.ones(n_simulations)
    weights[margins > 0] = cal / raw_cond
    weights[margins < 0] = (1.0 - cal) / (1.0 - raw_cond)

    order = np.argsort(margins, kind="stable")
    cumulative = np.cumsum(weights[order])
    median_margin = int(
        margins[order][int(np.searchsorted(cumulative, 0.5 * cumulative[-1]))]
    )

    # Strict > matched the caller's tie-break: cal == 0.5 tipped the away side.
    tip_mask = margins > 0 if cal > 0.5 else margins < 0
    if tip_mask.any():
        scoreline = Counter(zip(home[tip_mask], away[tip_mask])).most_common(1)[0][0]
    else:
        modal = Counter(zip(home, away)).most_common(1)[0][0]
        ordered = (max(modal), min(modal)) if cal > 0.5 else (min(modal), max(modal))
        scoreline = (
            ordered
            if ordered[0] != ordered[1]
            else (
                (ordered[0] + 1, ordered[1])
                if cal > 0.5
                else (ordered[0], ordered[1] + 1)
            )
        )

    return scoreline, median_margin, margins, weights


def _point_prediction_metrics(
    scorelines, median_margins, actual_home, actual_away, crps_values=None
):
    """Score the three integers that actually reach the predictions table."""
    scorelines = np.asarray(scorelines, dtype=float)
    predicted_home = scorelines[:, 0]
    predicted_away = scorelines[:, 1]
    predicted_margin = predicted_home - predicted_away
    actual_home = np.asarray(actual_home, dtype=float)
    actual_away = np.asarray(actual_away, dtype=float)
    actual_margin = actual_home - actual_away

    scored = {
        "games": int(predicted_margin.size),
        "margin_mae": float(np.mean(np.abs(predicted_margin - actual_margin))),
        "margin_bias": float(np.mean(predicted_margin - actual_margin)),
        "home_score_mae": float(np.mean(np.abs(predicted_home - actual_home))),
        "away_score_mae": float(np.mean(np.abs(predicted_away - actual_away))),
        "total_mae": float(
            np.mean(np.abs((predicted_home + predicted_away) - (actual_home + actual_away)))
        ),
        "exact_scoreline_rate": float(
            np.mean((predicted_home == actual_home) & (predicted_away == actual_away))
        ),
        "median_margin_mae": float(
            np.mean(np.abs(np.asarray(median_margins, dtype=float) - actual_margin))
        ),
    }
    if crps_values is not None:
        crps_values = np.asarray(crps_values, dtype=float)
        finite = np.isfinite(crps_values)
        scored["crps"] = float(np.mean(crps_values[finite])) if finite.any() else None
    return scored


def _score_reconciliation_point_predictions(
    raw_pairs,
    calibrated_cond,
    actual_home,
    actual_away,
    game_ids,
    n_samples,
    lambda3,
    dispersion_home,
    dispersion_away,
):
    """Price every available way of producing the three displayed integers.

    The distribution scorecard measures the margin's full predictive
    distribution. This measures what a reader of the tips actually sees: a
    scoreline and the score difference taken from it. Four live variants are
    scored, crossing when the score means are moved onto the calibrated
    probability (`always` or only `on_conflict`) with how the sample cloud is
    reduced to a scoreline (`mode` or `median`), plus `legacy`, which replays the
    importance-reweighting the constraint-native solve replaced.

    Every variant runs off the same per-game seed, so they differ only in the two
    decisions under test, not in the draws. Each carries its own margin CRPS from
    those matched seeds; the `methods` table above draws off a shared stream, so
    only these numbers are a clean head-to-head.
    """
    cond = np.asarray(calibrated_cond, dtype=float)
    usable = np.isfinite(cond)
    if not usable.any():
        return None

    variants = {
        name: {"scorelines": [], "medians": [], "crps": []}
        for name in (
            "always_mode",
            "always_median",
            "on_conflict_mode",
            "on_conflict_median",
            "legacy",
        )
    }
    kept_home = []
    kept_away = []

    def _draws(mu_h, mu_a, game_id):
        # salt=1 is the inference seed, so these are the numbers that would ship.
        return pf.draw_score_samples(
            mu_h,
            mu_a,
            n_samples,
            lambda3=lambda3,
            dispersion_home=dispersion_home,
            dispersion_away=dispersion_away,
            rng=pf.rng_for_game(game_id, salt=1),
        )

    for i in np.flatnonzero(usable):
        mu_h, mu_a = raw_pairs[i]
        game_id = None if game_ids is None else game_ids[i]
        actual = actual_home[i] - actual_away[i]
        cal = float(np.clip(cond[i], 1e-6, 1 - 1e-6))
        tipped_home = cal > 0.5

        solved_h, solved_a = pf.solve_score_means_for_probability(mu_h, mu_a, cal)
        solved_home, solved_away = _draws(solved_h, solved_a, game_id)
        solved_crps = dm.crps_ensemble(solved_home - solved_away, actual)
        solved_median = int(round(float(np.median(solved_home - solved_away))))

        # `on_conflict` only differs where the score model puts the other side
        # in front, so elsewhere it reuses the raw draws.
        conflict = (pf.conditional_home_win_prob(mu_h, mu_a) > 0.5) != tipped_home
        if conflict:
            raw_home, raw_away = solved_home, solved_away
            raw_crps, raw_median = solved_crps, solved_median
        else:
            raw_home, raw_away = _draws(mu_h, mu_a, game_id)
            raw_crps = dm.crps_ensemble(raw_home - raw_away, actual)
            raw_median = int(round(float(np.median(raw_home - raw_away))))

        for name, (home_sim, away_sim, crps, median), display in (
            ("always_mode", (solved_home, solved_away, solved_crps, solved_median), "mode"),
            ("always_median", (solved_home, solved_away, solved_crps, solved_median), "median"),
            ("on_conflict_mode", (raw_home, raw_away, raw_crps, raw_median), "mode"),
            ("on_conflict_median", (raw_home, raw_away, raw_crps, raw_median), "median"),
        ):
            variants[name]["scorelines"].append(
                pf.scoreline_from_samples(
                    home_sim, away_sim, tipped_home=tipped_home, display=display
                )
            )
            variants[name]["medians"].append(median)
            variants[name]["crps"].append(crps)

        legacy_scoreline, legacy_median, legacy_margins, weights = (
            _legacy_reconciled_prediction(
                mu_h,
                mu_a,
                cal,
                n_samples,
                lambda3,
                dispersion_home,
                dispersion_away,
                pf.rng_for_game(game_id, salt=1),
            )
        )
        variants["legacy"]["scorelines"].append(legacy_scoreline)
        variants["legacy"]["medians"].append(legacy_median)
        variants["legacy"]["crps"].append(
            dm.crps_weighted_ensemble(legacy_margins, weights, actual)
        )

        kept_home.append(actual_home[i])
        kept_away.append(actual_away[i])

    scored = {
        name: _point_prediction_metrics(
            parts["scorelines"], parts["medians"], kept_home, kept_away, parts["crps"]
        )
        for name, parts in variants.items()
    }
    scored["games"] = scored["always_mode"]["games"]
    # Name the deployed arrangement, so a report can never describe a
    # configuration it did not measure.
    scored["deployed"] = f"{RECONCILE_MODE}_{DISPLAY_MODE}"
    scored["shipped"] = scored[scored["deployed"]]
    return scored


def _score_margin_distributions(
    blended_h,
    blended_a,
    actual_home,
    actual_away,
    actual_margin,
    market_spread,
    calibrated_cond,
    prior_mask,
    test_mask,
    n_samples,
    seed,
    game_ids=None,
):
    """Score the margin's predictive distribution against strong baselines.

    Everything else in the pipeline scores the binary win probability, so the
    score distribution itself, and with it `lambda3`, the negative-binomial
    dispersion, and the market score blends, has never been measured. This
    closes that gap with CRPS, randomised PIT, and coverage-with-width.

    Following the thesis evaluation discipline, the comparators are strong
    rather than a floor: a normal approximation, an empirical replay of real
    past errors, and the market line. Beating a uniform guess would prove
    nothing. The over-dispersion and shared-component parameters are refitted
    on prior seasons only, so the held-out season stays honest.

    `model` is the raw score model. `model_reconciled` is what actually ships,
    after the means are shifted onto the calibrated win probability, so the
    difference between them prices that reconciliation step.
    """
    rng = np.random.default_rng(seed)

    prior_margin = actual_margin[prior_mask]
    prior_model_margin = (blended_h - blended_a)[prior_mask]
    test_margin = actual_margin[test_mask]
    model_margin = (blended_h - blended_a)[test_mask]
    n_test = int(test_mask.sum())
    if n_test == 0 or prior_mask.sum() < 2:
        return None

    # Distribution parameters from prior seasons only.
    lambda3 = mf.estimate_lambda3(
        actual_home[prior_mask],
        actual_away[prior_mask],
        blended_h[prior_mask],
        blended_a[prior_mask],
    )
    dispersion_home = mf.estimate_dispersion(actual_home[prior_mask], blended_h[prior_mask])
    dispersion_away = mf.estimate_dispersion(actual_away[prior_mask], blended_a[prior_mask])

    def _margin_samples(mu_pairs):
        sets = []
        for mu_h, mu_a in mu_pairs:
            home, away = pf.draw_score_samples(
                mu_h,
                mu_a,
                n_samples,
                lambda3=lambda3,
                dispersion_home=dispersion_home,
                dispersion_away=dispersion_away,
                rng=rng,
            )
            sets.append(home - away)
        return sets

    methods = {}

    raw_pairs = list(zip(blended_h[test_mask], blended_a[test_mask]))
    methods["model"] = dm.score_sample_forecasts(
        _margin_samples(raw_pairs), test_margin, rng=rng
    )

    # What ships: means shifted onto the calibrated win probability.
    test_cond = np.asarray(calibrated_cond, dtype=float)
    reconciled_pairs = [
        pf.solve_score_means_for_probability(mu_h, mu_a, cond)
        if np.isfinite(cond)
        else (mu_h, mu_a)
        for (mu_h, mu_a), cond in zip(raw_pairs, test_cond)
    ]
    methods["model_reconciled"] = dm.score_sample_forecasts(
        _margin_samples(reconciled_pairs), test_margin, rng=rng
    )

    # Baseline: normal approximation with prior-season residual spread.
    residuals = prior_margin - prior_model_margin
    sigma = float(np.std(residuals, ddof=1)) if residuals.size > 1 else float("nan")
    if np.isfinite(sigma) and sigma > 0:
        crps_values = [
            dm.crps_normal(mu, sigma, y) for mu, y in zip(model_margin, test_margin)
        ]
        # Continuous, so the PIT needs no randomisation.
        pit_values = norm.cdf((test_margin - model_margin) / sigma)
        methods["normal_approximation"] = {
            "games": n_test,
            "crps": float(np.mean(crps_values)),
            "sigma": sigma,
            "pit": dm.pit_histogram(pit_values),
            "intervals": [
                dm.interval_coverage(
                    model_margin - sigma * z, model_margin + sigma * z, test_margin,
                    level=level,
                )
                for level, z in ((0.5, NORMAL_50_Z), (0.9, NORMAL_90_Z))
            ],
        }

    # Baseline: replay real past errors around the model's margin. The direct
    # analogue of the thesis empirical trajectory baseline, and the bar to beat.
    if residuals.size >= 30:
        replay_sets = [
            mu + rng.choice(residuals, size=n_samples, replace=True)
            for mu in model_margin
        ]
        methods["empirical_replay"] = dm.score_sample_forecasts(
            replay_sets, test_margin, rng=rng
        )

    # Market line as a point forecast. Point CRPS is MAE, which is what puts
    # deterministic and probabilistic methods on one scale.
    spread = market_spread[test_mask]
    has_line = np.isfinite(spread)
    market_comparison = {"games": int(has_line.sum())}
    if has_line.any():
        market_comparison["market_line_crps"] = float(
            np.mean(np.abs(spread[has_line] - test_margin[has_line]))
        )
        # Same rows, so the head-to-head is like for like.
        model_on_line_rows = dm.score_sample_forecasts(
            _margin_samples([raw_pairs[i] for i in np.flatnonzero(has_line)]),
            test_margin[has_line],
            rng=rng,
        )
        market_comparison["model_crps"] = model_on_line_rows["crps"]

    # What the fix bought the displayed scoreline, in points.
    test_game_ids = None if game_ids is None else np.asarray(game_ids)[test_mask]
    reconciliation = _score_reconciliation_point_predictions(
        raw_pairs,
        test_cond,
        actual_home[test_mask],
        actual_away[test_mask],
        test_game_ids,
        n_samples,
        lambda3,
        dispersion_home,
        dispersion_away,
    )

    return {
        "games": n_test,
        "sim_samples": int(n_samples),
        "lambda3": float(lambda3),
        "dispersion_home": dispersion_home,
        "dispersion_away": dispersion_away,
        "methods": methods,
        "market_comparison": market_comparison,
        "reconciliation": reconciliation,
    }


def _expert_metrics(y, candidates):
    """Score named experts on the same rows as the candidate."""
    metrics = {}
    for name, probabilities in candidates.items():
        scored = _prediction_metrics(y, probabilities)
        metrics[name] = {
            "accuracy": scored["accuracy"],
            "log_loss": scored["log_loss"],
            "brier": scored["brier"],
        }
    return metrics


def _pool_probability_results(results):
    """Pool operational and counterfactual held-out probability results."""
    pooled_p = np.concatenate([res["model_p"] for res in results])
    pooled_no_market_p = np.concatenate(
        [res["no_market_counterfactual_p"] for res in results]
    )
    pooled_y = np.concatenate([res["y_test"] for res in results])
    pooled_tier_a = np.concatenate([res["tier_a_p"] for res in results])
    pooled_tier_b = np.concatenate([res["tier_b_p"] for res in results])
    pooled_tier_c = np.concatenate([res["tier_c_p"] for res in results])
    pooled_market = np.concatenate([res["market_p"] for res in results])
    pooled_valid_market = np.concatenate(
        [res["valid_market"] for res in results]
    ).astype(bool)

    operational = _prediction_metrics(pooled_y, pooled_p)
    actual_market = _prediction_metrics(
        pooled_y[pooled_valid_market],
        pooled_p[pooled_valid_market],
    )
    actual_no_market = _prediction_metrics(
        pooled_y[~pooled_valid_market],
        pooled_p[~pooled_valid_market],
    )
    no_market_counterfactual = _prediction_metrics(
        pooled_y,
        pooled_no_market_p,
    )

    expert_candidates = {
        "tier_a": pooled_tier_a,
        "tier_b": pooled_tier_b,
        "tier_c": pooled_tier_c,
    }
    expert_metrics = _expert_metrics(pooled_y, expert_candidates)
    market_regime_experts = (
        _expert_metrics(
            pooled_y[pooled_valid_market],
            {
                **{
                    name: probabilities[pooled_valid_market]
                    for name, probabilities in expert_candidates.items()
                },
                "market": pooled_market[pooled_valid_market],
            },
        )
        if pooled_valid_market.any()
        else {}
    )
    no_market_regime_experts = (
        _expert_metrics(
            pooled_y[~pooled_valid_market],
            {
                name: probabilities[~pooled_valid_market]
                for name, probabilities in expert_candidates.items()
            },
        )
        if (~pooled_valid_market).any()
        else {}
    )
    no_market_counterfactual_experts = _expert_metrics(
        pooled_y,
        expert_candidates,
    )

    global_acceptance = calib.acceptance_against_experts(
        operational,
        expert_metrics,
    )
    market_acceptance = calib.acceptance_against_experts(
        actual_market,
        market_regime_experts,
    )
    no_market_counterfactual_acceptance = calib.acceptance_against_experts(
        no_market_counterfactual,
        no_market_counterfactual_experts,
    )
    acceptance = {
        "accuracy_tolerance": 0.01,
        "loss_tolerance": 0.005,
        "global": global_acceptance,
        "market_regime": market_acceptance,
        "no_market_counterfactual": no_market_counterfactual_acceptance,
        "passed": bool(
            global_acceptance["passed"]
            and market_acceptance["passed"]
            and no_market_counterfactual_acceptance["passed"]
        ),
    }

    # The competition view: what each candidate would actually have won.
    # Seasons are scored separately and then averaged, because P(first)
    # saturates and pooling would wash out exactly the effect that matters.
    # One fold is one season; fall back to the fold index when a caller has not
    # labelled the year.
    pooled_groups = np.concatenate(
        [
            np.full(len(res["y_test"]), float(res.get("year", index)))
            for index, res in enumerate(results)
        ]
    )
    comp_candidates = {
        "deployed": pooled_p,
        "no_market_counterfactual": pooled_no_market_p,
        "tier_a": pooled_tier_a,
        "tier_b": pooled_tier_b,
        "tier_c": pooled_tier_c,
        "market": pooled_market,
    }
    competition = {}
    for name, probabilities in comp_candidates.items():
        placement = calib.comp_placement_metrics(
            probabilities, pooled_market, pooled_y, pooled_groups
        )
        if placement is not None:
            competition[name] = placement

    return {
        **operational,
        "market_regime": actual_market,
        "no_market_regime": actual_no_market,
        "actual_route_regimes": {
            "market": actual_market,
            "no_market": actual_no_market,
        },
        "no_market_counterfactual": no_market_counterfactual,
        "expert_metrics": expert_metrics,
        "market_regime_expert_metrics": market_regime_experts,
        "no_market_regime_expert_metrics": no_market_regime_experts,
        "no_market_counterfactual_expert_metrics": (no_market_counterfactual_experts),
        "acceptance": acceptance,
        "competition": competition,
        "margin_distribution": _pool_margin_distributions(results),
    }


def _pool_margin_distributions(results):
    """Game-weighted pooling of the per-season margin-distribution scorecard."""
    seasons = [
        res["margin_distribution"]
        for res in results
        if res.get("margin_distribution")
    ]
    if not seasons:
        return None

    def _weighted(values_and_weights):
        pairs = [(v, w) for v, w in values_and_weights if v is not None and w]
        if not pairs:
            return None
        total = float(sum(w for _, w in pairs))
        return float(sum(v * w for v, w in pairs) / total) if total else None

    method_names = sorted({name for s in seasons for name in s["methods"]})
    methods = {}
    for name in method_names:
        entries = [s["methods"][name] for s in seasons if name in s["methods"]]
        pooled_method = {
            "games": int(sum(e.get("games") or 0 for e in entries)),
            "crps": _weighted([(e.get("crps"), e.get("games")) for e in entries]),
        }
        # Coverage and width travel together, by level, and never apart.
        levels = sorted(
            {
                interval["level"]
                for e in entries
                for interval in e.get("intervals", [])
            }
        )
        pooled_method["intervals"] = [
            {
                "level": level,
                "coverage": _weighted(
                    [
                        (interval.get("coverage"), interval.get("games"))
                        for e in entries
                        for interval in e.get("intervals", [])
                        if interval["level"] == level
                    ]
                ),
                "width": _weighted(
                    [
                        (interval.get("width"), interval.get("games"))
                        for e in entries
                        for interval in e.get("intervals", [])
                        if interval["level"] == level
                    ]
                ),
            }
            for level in levels
        ]
        uniformity = [
            (e["pit"].get("uniformity_mae"), e.get("games"))
            for e in entries
            if e.get("pit")
        ]
        if uniformity:
            pooled_method["pit_uniformity_mae"] = _weighted(uniformity)
        methods[name] = pooled_method

    reconciliation_entries = [
        s["reconciliation"] for s in seasons if s.get("reconciliation")
    ]
    reconciliation = None
    if reconciliation_entries:
        point_fields = (
            "margin_mae",
            "margin_bias",
            "home_score_mae",
            "away_score_mae",
            "total_mae",
            "exact_scoreline_rate",
            "median_margin_mae",
            "crps",
        )
        reconciliation = {
            "games": int(sum(e.get("games") or 0 for e in reconciliation_entries)),
            "deployed": f"{RECONCILE_MODE}_{DISPLAY_MODE}",
        }
        for variant in RECONCILIATION_VARIANTS:
            reconciliation[variant] = {
                "games": int(
                    sum(e[variant].get("games") or 0 for e in reconciliation_entries)
                ),
                **{
                    field: _weighted(
                        [
                            (e[variant].get(field), e[variant].get("games"))
                            for e in reconciliation_entries
                        ]
                    )
                    for field in point_fields
                },
            }
        reconciliation["shipped"] = reconciliation[reconciliation["deployed"]]

    market_entries = [s["market_comparison"] for s in seasons]
    return {
        "games": int(sum(s["games"] for s in seasons)),
        "methods": methods,
        "reconciliation": reconciliation,
        "market_comparison": {
            "games": int(sum(e.get("games") or 0 for e in market_entries)),
            "market_line_crps": _weighted(
                [(e.get("market_line_crps"), e.get("games")) for e in market_entries]
            ),
            "model_crps": _weighted(
                [(e.get("model_crps"), e.get("games")) for e in market_entries]
            ),
        },
    }


def _load_training_frame(project_root, db_path, baseline_cfg=None):
    """Load and feature-merge the training frame the same way train.py does."""
    predictors = tc.filter_predictors(
        include_performance=tc.include_performance, predictor_list=tc.predictors
    )
    data = mf.get_training_data(
        db_path=db_path,
        sql_file=project_root / "pipeline/common/sql/training_data.sql",
    )
    if data.empty:
        raise RuntimeError("Training data is empty. Run data prep first.")

    if baseline_cfg is None:
        baseline_cfg = tb.default_baseline_config_from_env()
    baseline_features = tb.compute_tier_a_baseline_features(data, baseline_cfg)
    data = data.merge(baseline_features, on="game_id", how="left")

    base_home = float(data["team_final_score_home"].mean())
    base_away = float(data["team_final_score_away"].mean())
    data["baseline_mu_home"] = pd.to_numeric(
        data["baseline_mu_home"], errors="coerce"
    ).fillna(base_home)
    data["baseline_mu_away"] = pd.to_numeric(
        data["baseline_mu_away"], errors="coerce"
    ).fillna(base_away)
    data["baseline_home_win_prob_conditional"] = pd.to_numeric(
        data["baseline_home_win_prob_conditional"], errors="coerce"
    ).fillna(0.5)

    try:
        years = sorted(
            pd.to_numeric(data["competition_year"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        lineup_entries = lf.load_lineup_entries(db_path, years=years)
        lineup_features = lf.build_lineup_match_features(data, lineup_entries)
        data = data.merge(lineup_features, on="game_id", how="left")
        for col in lf.LINEUP_FEATURE_COLUMNS:
            if col == "game_id":
                continue
            if col in {"lineup_home_players", "lineup_away_players"}:
                data[col] = data[col].fillna("")
            else:
                data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0)
    except Exception as exc:
        print(f"Lineup feature merge skipped ({exc}).")

    try:
        from pipeline.common.nrl_data import features as ctx

        data = ctx.merge_match_context_features(data, db_path)
    except Exception as exc:
        print(f"Match-context feature merge skipped ({exc}).")

    return data, predictors


def _evaluate_season(
    test_year,
    year_col,
    non_draw,
    genuine_oof,
    y_full,
    baseline_mu_home,
    baseline_mu_away,
    home_mu_oof,
    away_mu_oof,
    tier_a_cond,
    tier_c_cond_oof,
    market_cond,
    valid_market_all,
    home_odds,
    away_odds,
    actual_margin,
    market_spread,
    actual_home_score=None,
    actual_away_score=None,
    sim_samples=20000,
    game_ids=None,
    lineup_unc_home=None,
    lineup_unc_away=None,
    lineup_mc_samples=64,
    lineup_mu_noise_scale=0.12,
    edge_threshold=0.05,
):
    prior_mask = non_draw & genuine_oof & (year_col < test_year)
    test_mask = non_draw & genuine_oof & (year_col == test_year)

    if prior_mask.sum() < 100 or test_mask.sum() < 5:
        return None

    # Blend weights selected on prior seasons only.
    wh, wa, _, _ = mf.select_blend_weights_by_log_loss(
        y_full[prior_mask],
        baseline_mu_home[prior_mask],
        baseline_mu_away[prior_mask],
        home_mu_oof[prior_mask],
        away_mu_oof[prior_mask],
    )

    blended_h = np.maximum((1.0 - wh) * baseline_mu_home + wh * home_mu_oof, 1e-6)
    blended_a = np.maximum((1.0 - wa) * baseline_mu_away + wa * away_mu_oof, 1e-6)
    # Marginalise over lineup uncertainty exactly as train.py fits and
    # inference.py serves, so this evaluation measures the deployed stack
    # rather than a slightly different one.
    tier_b_cond = pf.marginalized_conditional_home_win_prob_vec(
        blended_h,
        blended_a,
        lineup_unc_home,
        lineup_unc_away,
        game_ids=game_ids,
        n_samples=lineup_mc_samples,
        mu_noise_scale=lineup_mu_noise_scale,
    )

    # Probability-stack v3, fitted strictly on seasons before the held-out
    # year. The market pool sees only genuine paired H2H prices; the no-market
    # pool is trained counterfactually on every prior OOF row.
    market_prior_mask = prior_mask & valid_market_all
    market_pool = None
    market_calibrator = None
    market_selection = {
        "selected": "tier_b",
        "selection_rows": 0,
        "reason": "insufficient_market_rows",
    }
    if market_prior_mask.sum() >= 50:
        candidate = calib.SimplexLogitPool(include_market=True).fit(
            tier_a=tier_a_cond[market_prior_mask],
            tier_b=tier_b_cond[market_prior_mask],
            tier_c=tier_c_cond_oof[market_prior_mask],
            market=market_cond[market_prior_mask],
            y=y_full[market_prior_mask],
        )
        if candidate._is_fitted:
            market_pool = candidate
            market_calibrator = calib.TemperatureCalibrator()
            market_expert_probabilities = {
                "tier_a": tier_a_cond[market_prior_mask],
                "tier_b": tier_b_cond[market_prior_mask],
                "tier_c": tier_c_cond_oof[market_prior_mask],
                "market": market_cond[market_prior_mask],
            }
            market_fallback_expert = calib.strongest_deployable_expert(
                market_expert_probabilities,
                y_full[market_prior_mask],
                year_col[market_prior_mask],
                market=market_cond[market_prior_mask],
            )
            market_loso_path = calib.loso_simplex_pool_predictions(
                tier_a=tier_a_cond[market_prior_mask],
                tier_b=tier_b_cond[market_prior_mask],
                tier_c=tier_c_cond_oof[market_prior_mask],
                market=market_cond[market_prior_mask],
                y=y_full[market_prior_mask],
                groups=year_col[market_prior_mask],
                include_market=True,
                shrinkage_grid=calib.DEFAULT_SHRINKAGE_GRID,
                fallback_expert=market_fallback_expert,
            )
            market_loso = None if market_loso_path is None else market_loso_path.get(1.0)
            if market_loso is not None:
                market_loso_rows = np.isfinite(market_loso)
                market_calibrator.fit(
                    market_loso[market_loso_rows],
                    y_full[market_prior_mask][market_loso_rows],
                )
            market_nested = calib.nested_loso_simplex_predictions(
                tier_a=tier_a_cond[market_prior_mask],
                tier_b=tier_b_cond[market_prior_mask],
                tier_c=tier_c_cond_oof[market_prior_mask],
                market=market_cond[market_prior_mask],
                y=y_full[market_prior_mask],
                groups=year_col[market_prior_mask],
                include_market=True,
                shrinkage_grid=calib.DEFAULT_SHRINKAGE_GRID,
                fallback_expert=market_fallback_expert,
            )
            market_selection = calib.select_market_pool(
                market_pool,
                market_nested,
                y_full[market_prior_mask],
                market_expert_probabilities,
                groups=year_col[market_prior_mask],
                market_probabilities=market_cond[market_prior_mask],
            )
            market_calibrator = calib.fit_selected_market_calibrator(
                market_selection,
                market_expert_probabilities,
                y_full[market_prior_mask],
                market_calibrator,
                loso_path_predictions=market_loso_path,
            )

    no_market_pool = calib.SimplexLogitPool(include_market=False).fit(
        tier_a=tier_a_cond[prior_mask],
        tier_b=tier_b_cond[prior_mask],
        tier_c=tier_c_cond_oof[prior_mask],
        y=y_full[prior_mask],
    )
    no_market_learned_weights = no_market_pool.weight_map
    no_market_calibrator = calib.TemperatureCalibrator()

    no_market_expert_probabilities = {
        "tier_a": tier_a_cond[prior_mask],
        "tier_b": tier_b_cond[prior_mask],
        "tier_c": tier_c_cond_oof[prior_mask],
    }
    no_market_fallback_expert = calib.strongest_deployable_expert(
        no_market_expert_probabilities,
        y_full[prior_mask],
        year_col[prior_mask],
        market=market_cond[prior_mask],
    )
    no_market_loso_path = calib.loso_simplex_pool_predictions(
        tier_a=tier_a_cond[prior_mask],
        tier_b=tier_b_cond[prior_mask],
        tier_c=tier_c_cond_oof[prior_mask],
        y=y_full[prior_mask],
        groups=year_col[prior_mask],
        include_market=False,
        shrinkage_grid=calib.DEFAULT_SHRINKAGE_GRID,
        fallback_expert=no_market_fallback_expert,
    )
    no_market_loso = None if no_market_loso_path is None else no_market_loso_path.get(1.0)
    if no_market_loso is not None:
        no_market_loso_rows = np.isfinite(no_market_loso)
        no_market_calibrator.fit(
            no_market_loso[no_market_loso_rows],
            y_full[prior_mask][no_market_loso_rows],
        )

    nested_no_market = calib.nested_loso_simplex_predictions(
        tier_a=tier_a_cond[prior_mask],
        tier_b=tier_b_cond[prior_mask],
        tier_c=tier_c_cond_oof[prior_mask],
        y=y_full[prior_mask],
        groups=year_col[prior_mask],
        include_market=False,
        shrinkage_grid=calib.DEFAULT_SHRINKAGE_GRID,
        fallback_expert=no_market_fallback_expert,
    )
    no_market_selection = calib.select_no_market_pool(
        no_market_pool,
        nested_no_market,
        y_full[prior_mask],
        no_market_expert_probabilities,
        groups=year_col[prior_mask],
        market_probabilities=market_cond[prior_mask],
    )
    no_market_strategy = no_market_selection["strategy"]
    no_market_calibrator = calib.fit_selected_pool_calibrator(
        no_market_selection,
        no_market_expert_probabilities,
        y_full[prior_mask],
        no_market_calibrator,
        loso_path_predictions=no_market_loso_path,
    )
    no_market_eligibility = no_market_selection["eligibility"]
    no_market_pool_log_loss = no_market_eligibility["pool_log_loss"]
    no_market_tier_b_log_loss = no_market_eligibility["tier_b_log_loss"]

    test_tier_a = tier_a_cond[test_mask]
    test_tier_b = tier_b_cond[test_mask]
    test_tier_c = tier_c_cond_oof[test_mask]
    test_market = market_cond[test_mask]
    test_valid_market = valid_market_all[test_mask]
    model_p, probability_routes = calib.predict_probability_regimes(
        tier_a=test_tier_a,
        tier_b=test_tier_b,
        tier_c=test_tier_c,
        market=test_market,
        valid_market=test_valid_market,
        market_stacker=market_pool,
        market_calibrator=market_calibrator,
        no_market_stacker=no_market_pool,
        no_market_calibrator=no_market_calibrator,
        no_market_strategy=no_market_strategy,
    )
    model_p = np.clip(model_p, 1e-6, 1 - 1e-6)

    # Independently score the selected no-market strategy on every outer
    # held-out row. Forcing the market mask false exercises the same runtime
    # routing and A/B/C-only consensus guard without leaking market evidence.
    no_market_counterfactual_p, no_market_counterfactual_routes = (
        calib.predict_probability_regimes(
            tier_a=test_tier_a,
            tier_b=test_tier_b,
            tier_c=test_tier_c,
            market=test_market,
            valid_market=np.zeros(test_mask.sum(), dtype=bool),
            market_stacker=market_pool,
            market_calibrator=market_calibrator,
            no_market_stacker=no_market_pool,
            no_market_calibrator=no_market_calibrator,
            no_market_strategy=no_market_strategy,
        )
    )
    no_market_counterfactual_p = np.clip(
        no_market_counterfactual_p,
        1e-6,
        1 - 1e-6,
    )

    y_test = y_full[test_mask]
    market_p = test_market
    valid_market = test_valid_market

    operational_metrics = _prediction_metrics(y_test, model_p)
    market_regime = _prediction_metrics(
        y_test[valid_market],
        model_p[valid_market],
    )
    no_market_regime = _prediction_metrics(
        y_test[~valid_market],
        model_p[~valid_market],
    )
    no_market_counterfactual = _prediction_metrics(
        y_test,
        no_market_counterfactual_p,
    )

    result = {
        "year": int(test_year),
        "games": operational_metrics["games"],
        "correct": operational_metrics["correct"],
        "log_loss": operational_metrics["log_loss"],
        "brier": operational_metrics["brier"],
        "blend_wh": float(wh),
        "blend_wa": float(wa),
        "market_games": int(valid_market.sum()),
        "market_correct": int(
            ((market_p[valid_market] > 0.5) == y_test[valid_market].astype(bool)).sum()
        ),
        "model_p": model_p,
        "no_market_counterfactual_p": no_market_counterfactual_p,
        "y_test": y_test,
        "market_p": market_p,
        "tier_a_p": test_tier_a,
        "tier_b_p": test_tier_b,
        "tier_c_p": test_tier_c,
        "valid_market": valid_market,
        "market_regime": market_regime,
        "no_market_regime": no_market_regime,
        "actual_route_regimes": {
            "market": market_regime,
            "no_market": no_market_regime,
        },
        "no_market_counterfactual": no_market_counterfactual,
        "market_selection": market_selection,
        "no_market_strategy": no_market_strategy,
        "no_market_selection": no_market_selection,
        "no_market_eligibility": no_market_eligibility,
        "no_market_learned_weights": no_market_learned_weights,
        "no_market_selected_weights": no_market_pool.weight_map,
        "no_market_selection_pool_log_loss": no_market_pool_log_loss,
        "no_market_selection_tier_b_log_loss": no_market_tier_b_log_loss,
        "probability_routes": probability_routes,
        "no_market_counterfactual_routes": (no_market_counterfactual_routes),
    }

    # Margin metrics for the comp's tie-breaker: model margin from the
    # blended mus, market margin from the negated line handicap.
    model_margin = blended_h[test_mask] - blended_a[test_mask]
    margin_actual = actual_margin[test_mask]
    result["margin_mae"] = float(np.mean(np.abs(model_margin - margin_actual)))
    result["margin_bias"] = float(np.mean(model_margin - margin_actual))
    spread = market_spread[test_mask]
    has_line = np.isfinite(spread)
    result["market_margin_games"] = int(has_line.sum())
    result["market_margin_mae"] = (
        float(np.mean(np.abs(spread[has_line] - margin_actual[has_line])))
        if has_line.any()
        else None
    )

    # Season-out gate for the ridge margin blend (mirrors train.py's fit):
    # fit on prior seasons with a line, score on the whole test season with
    # model-margin fallback where the line is missing.
    result["margin_blend_mae"] = None
    blend_fit_mask = prior_mask & np.isfinite(market_spread)
    if blend_fit_mask.sum() >= 100:
        model_margin_full = blended_h - blended_a
        tier_a_margin_full = baseline_mu_home - baseline_mu_away
        X_margin = np.column_stack(
            [model_margin_full, market_spread, tier_a_margin_full]
        )
        margin_model = Ridge(alpha=1.0)
        margin_model.fit(X_margin[blend_fit_mask], actual_margin[blend_fit_mask])
        safe_x_margin = np.nan_to_num(X_margin, nan=0.0, posinf=0.0, neginf=0.0)
        blend_pred = np.where(
            np.isfinite(market_spread),
            margin_model.predict(safe_x_margin),
            model_margin_full,
        )
        result["margin_blend_mae"] = float(
            np.mean(np.abs(blend_pred[test_mask] - margin_actual))
        )

    # Proper scoring rules on the margin distribution, which point MAE above
    # cannot see: CRPS, randomised PIT, and coverage reported with width.
    # Needs the realised scores, since the over-dispersion and shared-component
    # parameters are refitted per season rather than taken from the manifest.
    result["margin_distribution"] = None
    if actual_home_score is not None and actual_away_score is not None:
        result["margin_distribution"] = _score_margin_distributions(
            blended_h,
            blended_a,
            actual_home_score,
            actual_away_score,
            actual_margin,
            market_spread,
            model_p,
            prior_mask,
            test_mask,
            sim_samples,
            seed=int(test_year),
            game_ids=game_ids,
        )

    # Flat-stake ROI at the edge threshold.
    edge = np.where(valid_market, model_p - market_p, np.nan)
    oh = home_odds[test_mask]
    oa = away_odds[test_mask]
    total_bets, wins, profit = 0, 0, 0.0
    for e, act, odds_h, odds_a, has_market in zip(edge, y_test, oh, oa, valid_market):
        if not has_market:
            continue
        if e > edge_threshold and np.isfinite(odds_h) and odds_h > 1.0:
            total_bets += 1
            profit += (odds_h - 1.0) if act == 1 else -1.0
            wins += int(act == 1)
        elif e < -edge_threshold and np.isfinite(odds_a) and odds_a > 1.0:
            total_bets += 1
            profit += (odds_a - 1.0) if act == 0 else -1.0
            wins += int(act == 0)
    result.update({"bets": total_bets, "bet_wins": wins, "profit": float(profit)})
    return result


def _git_sha(project_root):
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _manifest_fingerprint(project_root, models_dir=None):
    try:
        models_dir = pathlib.Path(models_dir or (project_root / "models"))
        with open(models_dir / "model_manifest.json") as fh:
            manifest = json.load(fh)
        return {
            "blend_weight_home": manifest.get("blend_weight_home"),
            "blend_weight_away": manifest.get("blend_weight_away"),
            "lambda3": manifest.get("lambda3"),
            "tier_a_baseline": manifest.get("tier_a_baseline"),
            "predictor_count": len(manifest.get("predictors") or []),
            "probability_stack": manifest.get("probability_stack"),
        }
    except Exception:
        return None


def _print_margin_distribution(pooled):
    """Print the margin scorecard: CRPS first, then calibration with width.

    Ordered worst-CRPS-last so the winner is obvious, and the baselines sit in
    the same table as the model rather than in a footnote. A method that wins
    CRPS while badly under-covering has bought sharpness with honesty, which is
    only visible when the two columns sit side by side.
    """
    if not pooled or not pooled.get("methods"):
        return

    print("\n── Margin distribution (CRPS, lower is better) ──")
    print(
        f"  {'Method':<22} {'Games':>6} {'CRPS':>8} "
        f"{'50% cov':>8} {'90% cov':>8} {'90% width':>10} {'PIT dev':>8}"
    )

    def _sort_key(item):
        crps = item[1].get("crps")
        return crps if crps is not None else float("inf")

    def _fmt(value, width, precision):
        if value is None:
            return format("n/a", f">{width}")
        return format(value, f">{width}.{precision}f")

    for name, scored in sorted(pooled["methods"].items(), key=_sort_key):
        by_level = {
            interval["level"]: interval for interval in scored.get("intervals", [])
        }
        print(
            f"  {name:<22} {scored.get('games', 0):>6} "
            f"{_fmt(scored.get('crps'), 8, 2)} "
            f"{_fmt(by_level.get(0.5, {}).get('coverage'), 8, 2)} "
            f"{_fmt(by_level.get(0.9, {}).get('coverage'), 8, 2)} "
            f"{_fmt(by_level.get(0.9, {}).get('width'), 10, 1)} "
            f"{_fmt(scored.get('pit_uniformity_mae'), 8, 4)}"
        )

    market = pooled.get("market_comparison") or {}
    if market.get("market_line_crps") is not None:
        print(
            f"  On the {market['games']} games with a line: "
            f"model CRPS {market['model_crps']:.2f} vs "
            f"market line {market['market_line_crps']:.2f} "
            "(a point forecast, so its CRPS is its MAE)"
        )

    reconciliation = pooled.get("reconciliation")
    if reconciliation:
        print(
            "\n── Displayed scoreline: when to reconcile, and what to display ──"
        )
        print(
            f"  {'Variant':<22} {'Games':>6} {'Margin MAE':>11} {'Home MAE':>9} "
            f"{'Away MAE':>9} {'Total MAE':>10} {'CRPS':>7}"
        )
        deployed = reconciliation.get("deployed")
        ranked = sorted(
            (
                (name, reconciliation[name])
                for name in RECONCILIATION_VARIANTS
                if reconciliation.get(name)
            ),
            key=lambda item: (
                item[1].get("margin_mae")
                if item[1].get("margin_mae") is not None
                else float("inf")
            ),
        )
        for name, scored in ranked:
            label = name + (" *" if name == deployed else "")
            print(
                f"  {label:<22} {scored.get('games', 0):>6} "
                f"{_fmt(scored.get('margin_mae'), 11, 2)} "
                f"{_fmt(scored.get('home_score_mae'), 9, 2)} "
                f"{_fmt(scored.get('away_score_mae'), 9, 2)} "
                f"{_fmt(scored.get('total_mae'), 10, 2)} "
                f"{_fmt(scored.get('crps'), 7, 2)}"
            )
        print(f"  * deployed. Best on margin MAE: {ranked[0][0]}.")


def _build_report(results, pooled, config):
    seasons = [
        {k: v for k, v in res.items() if not isinstance(v, np.ndarray)}
        for res in results
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "seasons": seasons,
        "pooled": pooled,
    }


def _write_report(report, project_root):
    """Write the eval report; failure to write must never fail the eval."""
    try:
        override = os.getenv("FOOTY_TIPPER_EVAL_REPORT_PATH")
        if override:
            paths = [pathlib.Path(override)]
        else:
            reports_dir = project_root / "reports"
            reports_dir.mkdir(exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            paths = [
                reports_dir / f"eval-{stamp}.json",
                reports_dir / "eval-latest.json",
            ]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as fh:
                json.dump(report, fh, indent=2, default=float)
        return paths[-1]
    except Exception as exc:
        print(f"Eval report not written ({exc}).")
        return None


def main():
    project_root = configured_project_root()
    db_path = database_path(project_root)
    models_dir = models_path(project_root)
    n_seasons = int(os.getenv("FOOTY_TIPPER_EVAL_SEASONS", "3"))
    # CRPS converges quickly, so evaluation needs far fewer draws than the
    # 100k inference uses for a modal scoreline.
    sim_samples = int(os.getenv("FOOTY_TIPPER_EVAL_SIM_SAMPLES", "20000"))

    home_model = pf.load_models("home_model", project_root, models_dir=models_dir)
    away_model = pf.load_models("away_model", project_root, models_dir=models_dir)

    # Optional honest Tier-A tuning: the grid only ever sees seasons strictly
    # before the earliest held-out season, so no test year informs the choice.
    baseline_cfg = None
    if os.getenv("FOOTY_TIPPER_TUNE_TIER_A", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "n",
        "off",
    }:
        raw = mf.get_training_data(
            db_path=db_path,
            sql_file=project_root / "pipeline/common/sql/training_data.sql",
        )
        years_all = sorted(
            pd.to_numeric(raw["competition_year"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        if len(years_all) > n_seasons + 1:
            cutoff = years_all[-n_seasons]
            tune_df = raw[
                pd.to_numeric(raw["competition_year"], errors="coerce") < cutoff
            ]
            baseline_cfg, tier_a_grid = tb.tune_baseline_hyperparams(tune_df)
            if not tier_a_grid.empty:
                print(
                    f"Tier-A tuned on seasons < {cutoff}: alpha={baseline_cfg.alpha:.2f}, "
                    f"carryover={baseline_cfg.carryover:.2f}"
                )

    data, configured_predictors = _load_training_frame(
        project_root, db_path, baseline_cfg=baseline_cfg
    )
    try:
        candidate_manifest = json.loads(
            (models_dir / "model_manifest.json").read_text(encoding="utf-8")
        )
        selected = candidate_manifest.get("predictors")
        if (
            not isinstance(selected, list)
            or not selected
            or not all(isinstance(name, str) and name for name in selected)
        ):
            raise ValueError("candidate predictor contract is invalid")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(
            "Cannot evaluate candidate without its valid staged predictor contract"
        ) from exc
    data = tc.align_predictor_columns(data, configured_predictors)
    data = tc.align_predictor_columns(data, selected)

    year_col = pd.to_numeric(data["competition_year"], errors="coerce").to_numpy()
    non_draw = data["team_final_score_home"].to_numpy(dtype=float) != data[
        "team_final_score_away"
    ].to_numpy(dtype=float)
    y_full = (
        data["team_final_score_home"].to_numpy(dtype=float)
        > data["team_final_score_away"].to_numpy(dtype=float)
    ).astype(int)

    console.emit_progress("generating expanding-window out-of-fold predictions (slow)")
    print("Generating expanding-window OOF predictions (this is the slow part)...")
    home_mu_oof, home_mask = mf.generate_oof_score_predictions(
        data, selected, home_model, "team_final_score_home", return_mask=True
    )
    away_mu_oof, away_mask = mf.generate_oof_score_predictions(
        data, selected, away_model, "team_final_score_away", return_mask=True
    )
    best_params = dict(home_model.named_steps["hyperparamtuning"].best_params_)
    preprocessor_steps = home_model[:-1]
    tier_c_oof, binary_mask = mf.generate_oof_binary_predictions(
        data, non_draw, selected, preprocessor_steps, best_params, return_mask=True
    )
    tier_c_cond_oof = np.clip(tier_c_oof, 1e-6, 1 - 1e-6)
    genuine_oof = home_mask & away_mask & binary_mask

    tier_a_cond = np.clip(
        pd.to_numeric(data["baseline_home_win_prob_conditional"], errors="coerce")
        .fillna(0.5)
        .to_numpy(dtype=float),
        1e-6,
        1 - 1e-6,
    )
    market_cond = pf.derive_market_home_probability(data)
    valid_market_all = calib.valid_h2h_mask(data)

    home_odds = pd.to_numeric(
        data.get("team_head_to_head_odds_home", np.nan), errors="coerce"
    ).to_numpy(dtype=float)
    away_odds = pd.to_numeric(
        data.get("team_head_to_head_odds_away", np.nan), errors="coerce"
    ).to_numpy(dtype=float)

    baseline_mu_home = data["baseline_mu_home"].to_numpy(dtype=float)
    baseline_mu_away = data["baseline_mu_away"].to_numpy(dtype=float)

    # Same lineup-uncertainty settings train.py and inference.py use.
    lineup_mc_samples = int(os.getenv("FOOTY_TIPPER_LINEUP_MONTE_CARLO_SAMPLES", "64"))
    lineup_mu_noise_scale = float(os.getenv("FOOTY_TIPPER_LINEUP_MU_NOISE_SCALE", "0.12"))
    lineup_unc_home = (
        pd.to_numeric(data.get("lineup_selection_uncertainty_home", 0.0), errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    lineup_unc_away = (
        pd.to_numeric(data.get("lineup_selection_uncertainty_away", 0.0), errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    actual_margin = data["team_final_score_home"].to_numpy(dtype=float) - data[
        "team_final_score_away"
    ].to_numpy(dtype=float)
    # Market's expected home margin: the line handicap is negative when the
    # home side is favourite, so the expected margin is its negation.
    market_spread = -pd.to_numeric(
        data.get("implied_spread_home", np.nan), errors="coerce"
    ).to_numpy(dtype=float)
    eval_years = sorted({int(y) for y in year_col[genuine_oof & ~np.isnan(year_col)]})[
        -n_seasons:
    ]
    print(f"Evaluating held-out seasons: {eval_years}")

    results = []
    for test_year in eval_years:
        console.emit_progress(f"scoring held-out season {test_year}")
        res = _evaluate_season(
            test_year,
            year_col,
            non_draw,
            genuine_oof,
            y_full,
            baseline_mu_home,
            baseline_mu_away,
            home_mu_oof,
            away_mu_oof,
            tier_a_cond,
            tier_c_cond_oof,
            market_cond,
            valid_market_all,
            home_odds,
            away_odds,
            actual_margin,
            market_spread,
            data["team_final_score_home"].to_numpy(dtype=float),
            data["team_final_score_away"].to_numpy(dtype=float),
            sim_samples,
            game_ids=data["game_id"].to_numpy(),
            lineup_unc_home=lineup_unc_home,
            lineup_unc_away=lineup_unc_away,
            lineup_mc_samples=lineup_mc_samples,
            lineup_mu_noise_scale=lineup_mu_noise_scale,
        )
        if res is None:
            print(f"  {test_year}: skipped (not enough prior or test rows).")
            continue
        res["comp_sim"] = comp_sim.simulate_comp_placement(
            res["model_p"], res["market_p"], res["y_test"]
        )
        results.append(res)

    if not results:
        print("No seasons could be evaluated. Train on more seasons first.")
        return 1

    print("\n── Honest nested evaluation (meta-layer never sees the test season) ──")
    print(
        f"  {'Season':<8} {'Tips':>9} {'Acc':>7} {'Mkt acc':>8} "
        f"{'M/N':>9} {'M-pool':>8} {'No-mkt':>8} "
        f"{'LogLoss':>8} {'Brier':>7} {'ROI':>8}"
    )
    for res in results:
        acc = res["correct"] / res["games"]
        mkt = (
            (res["market_correct"] / res["market_games"])
            if res["market_games"]
            else float("nan")
        )
        roi = (100.0 * res["profit"] / res["bets"]) if res["bets"] else float("nan")
        roi_text = f"{roi:+.1f}%" if res["bets"] else "n/a"
        regime_text = (
            f"{res['market_regime']['games']}/{res['no_market_regime']['games']}"
        )
        print(
            f"  {res['year']:<8} {res['correct']:>4}/{res['games']:<4} {acc:>7.1%} {mkt:>8.1%} "
            f"{regime_text:>9} {res['market_selection']['selected']:>8} "
            f"{res['no_market_selection']['selected']:>8} "
            f"{res['log_loss']:>8.4f} {res['brier']:>7.4f} {roi_text:>8}"
        )

    probability_pooled = _pool_probability_results(results)
    pooled_games = probability_pooled["games"]
    pooled_correct = probability_pooled["correct"]
    pooled_mkt_games = int(sum(res["market_games"] for res in results))
    pooled_mkt_correct = int(sum(res["market_correct"] for res in results))
    pooled_bets = int(sum(res["bets"] for res in results))
    pooled_profit = float(sum(res["profit"] for res in results))

    pooled_log_loss = probability_pooled["log_loss"]
    pooled_brier = probability_pooled["brier"]
    market_regime_pooled = probability_pooled["market_regime"]
    no_market_regime_pooled = probability_pooled["no_market_regime"]
    no_market_counterfactual_pooled = probability_pooled["no_market_counterfactual"]
    expert_metrics = probability_pooled["expert_metrics"]
    acceptance = probability_pooled["acceptance"]

    print("\n── Pooled across held-out seasons ──")
    print(
        f"  Tipping accuracy: {pooled_correct}/{pooled_games} ({pooled_correct / pooled_games:.1%})"
    )
    if pooled_mkt_games:
        print(
            f"  Market favourite: {pooled_mkt_correct}/{pooled_mkt_games} ({pooled_mkt_correct / pooled_mkt_games:.1%})"
        )
    print(f"  Log-loss: {pooled_log_loss:.4f}   Brier: {pooled_brier:.4f}")
    for regime_name, regime in (
        ("Market-backed", market_regime_pooled),
        ("Model-only", no_market_regime_pooled),
    ):
        if regime["games"]:
            print(
                f"  {regime_name}: {regime['games']} games, "
                f"accuracy {regime['accuracy']:.1%}, "
                f"log-loss {regime['log_loss']:.4f}, "
                f"Brier {regime['brier']:.4f}"
            )
    print(
        "  Counterfactual model-only: "
        f"{no_market_counterfactual_pooled['games']} games, "
        f"accuracy {no_market_counterfactual_pooled['accuracy']:.1%}, "
        f"log-loss {no_market_counterfactual_pooled['log_loss']:.4f}, "
        f"Brier {no_market_counterfactual_pooled['brier']:.4f}"
    )
    print("  Single-expert benchmarks:")
    for name, metrics in expert_metrics.items():
        print(
            f"    {name}: accuracy {metrics['accuracy']:.1%}, "
            f"log-loss {metrics['log_loss']:.4f}, Brier {metrics['brier']:.4f}"
        )
    if pooled_bets:
        print(
            f"  Edge bets: {pooled_bets}, flat-stake ROI: {100.0 * pooled_profit / pooled_bets:+.1f}%"
        )

    print(
        "  Acceptance gate: "
        + (
            "PASS"
            if acceptance["passed"]
            else "FAIL (candidate trails the strongest applicable expert beyond tolerance)"
        )
    )
    print(
        "    global="
        f"{'pass' if acceptance['global']['passed'] else 'fail'}, "
        "market="
        f"{'pass' if acceptance['market_regime']['passed'] else 'fail'}, "
        "counterfactual-model-only="
        f"{'pass' if acceptance['no_market_counterfactual']['passed'] else 'fail'}"
    )

    # Margin metrics pooled by game count (season MAEs are per-game means).
    pooled_margin_mae = float(
        sum(res["margin_mae"] * res["games"] for res in results) / pooled_games
    )
    market_margin_games = int(sum(res["market_margin_games"] for res in results))
    pooled_market_margin_mae = (
        float(
            sum(
                res["market_margin_mae"] * res["market_margin_games"]
                for res in results
                if res["market_margin_mae"] is not None
            )
            / market_margin_games
        )
        if market_margin_games
        else None
    )
    blend_results = [res for res in results if res.get("margin_blend_mae") is not None]
    pooled_margin_blend_mae = (
        float(
            sum(res["margin_blend_mae"] * res["games"] for res in blend_results)
            / sum(res["games"] for res in blend_results)
        )
        if blend_results
        else None
    )
    print(f"  Margin MAE (tie-breaker): model {pooled_margin_mae:.2f}", end="")
    if pooled_market_margin_mae is not None:
        print(
            f" vs market line {pooled_market_margin_mae:.2f} ({market_margin_games} games)",
            end="",
        )
    if pooled_margin_blend_mae is not None:
        print(f" vs ridge blend {pooled_margin_blend_mae:.2f}", end="")
    print()

    _print_margin_distribution(probability_pooled.get("margin_distribution"))

    # Lift the headline numbers onto the operator's console; this script's stdout
    # is captured by the parent, so without a result marker the step shows only a tick.
    summary_rows = [
        (
            "Tipping accuracy",
            f"{pooled_correct / pooled_games:.1%}  ({pooled_correct}/{pooled_games})",
        ),
        ("Log loss / Brier", f"{pooled_log_loss:.4f} / {pooled_brier:.4f}"),
    ]
    if pooled_mkt_games:
        summary_rows.append(
            (
                "Market favourite",
                f"{pooled_mkt_correct / pooled_mkt_games:.1%}  ({pooled_mkt_correct}/{pooled_mkt_games})",
            )
        )
    summary_rows.append(
        (
            "Margin MAE",
            f"model {pooled_margin_mae:.2f}"
            + (
                f"  vs market line {pooled_market_margin_mae:.2f}"
                if pooled_market_margin_mae is not None
                else ""
            ),
        )
    )
    margin_pooled = probability_pooled.get("margin_distribution")
    if margin_pooled and margin_pooled.get("methods"):
        ranked = sorted(
            margin_pooled["methods"].items(),
            key=lambda kv: kv[1]["crps"] if kv[1].get("crps") is not None else float("inf"),
        )
        best_name, best = ranked[0]
        summary_rows.append(("Best margin CRPS", f"{best_name} {best['crps']:.2f}"))
        model_scored = margin_pooled["methods"].get("model")
        if model_scored and model_scored.get("crps") is not None:
            summary_rows.append(("Score model CRPS", f"{model_scored['crps']:.2f}"))
    if pooled_bets:
        summary_rows.append(
            ("Flat-stake ROI", f"{100.0 * pooled_profit / pooled_bets:+.1f}%  ({pooled_bets} bets)")
        )
    # The headline: what each candidate would have won, not how well it was
    # calibrated. Deployed first, then the comparators worth arguing with.
    competition = probability_pooled.get("competition") or {}
    for label, name in (
        ("Deployed", "deployed"),
        ("Tier C alone", "tier_c"),
        ("Market favourite", "market"),
    ):
        placement = competition.get(name)
        if not placement:
            continue
        summary_rows.append(
            (
                f"P(win comp): {label}",
                f"{placement['mean_p_first']:.3f} mean/season, "
                f"rank {placement['mean_expected_rank']:.1f}, "
                f"{placement['tips_correct']}/{placement['games']} tips",
            )
        )
    summary_rows.append(
        ("Acceptance gate", "PASS" if acceptance["passed"] else "FAIL")
    )
    console.emit_result("evaluation_summary", rows=summary_rows)

    comp_results = [res["comp_sim"] for res in results if res.get("comp_sim")]
    pooled_p_first = (
        float(np.mean([c["p_first"] for c in comp_results])) if comp_results else None
    )
    pooled_expected_rank = (
        float(np.mean([c["expected_rank"] for c in comp_results]))
        if comp_results
        else None
    )
    if comp_results:
        print(
            f"  Comp placement (field of {comp_results[0]['field_size']}): "
            f"P(first) {pooled_p_first:.1%}, expected rank {pooled_expected_rank:.1f}"
        )

    pooled = {
        **probability_pooled,
        "market_games": pooled_mkt_games,
        "market_correct": pooled_mkt_correct,
        "market_accuracy": (pooled_mkt_correct / pooled_mkt_games)
        if pooled_mkt_games
        else None,
        "bets": pooled_bets,
        "profit": pooled_profit,
        "roi_pct": (100.0 * pooled_profit / pooled_bets) if pooled_bets else None,
        "margin_mae": pooled_margin_mae,
        "market_margin_mae": pooled_market_margin_mae,
        "market_margin_games": market_margin_games,
        "margin_blend_mae": pooled_margin_blend_mae,
        "comp_p_first": pooled_p_first,
        "comp_expected_rank": pooled_expected_rank,
    }
    config = {
        "eval_seasons": [res["year"] for res in results],
        "n_seasons_requested": n_seasons,
        "rows": int(len(data)),
        "selected_predictor_count": int(len(selected)),
        "git_sha": _git_sha(project_root),
        "manifest": _manifest_fingerprint(project_root, models_dir=models_dir),
        "env": {
            key: os.environ[key]
            for key in sorted(os.environ)
            if key.startswith("FOOTY_TIPPER_")
            and "PASSWORD" not in key
            and "KEY" not in key
        },
    }
    report_path = _write_report(_build_report(results, pooled, config), project_root)
    if report_path is not None:
        print(f"\nReport written to {report_path}")

    if acceptance["passed"]:
        print("\nEvaluation complete.")
        return 0
    print("\nEvaluation complete, but the probability candidate failed acceptance.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
