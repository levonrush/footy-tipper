# Description: This script trains score models and saves artefacts used at inference.
print("Running the train.py script...")

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import brier_score_loss, log_loss

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from pipeline.common import console
from pipeline.common.model_prediciton import prediction_functions as pf
from pipeline.common.lineups import features as lf
from pipeline.common.model_training import calibration as calib
from pipeline.common.model_training import joker_policy as jp
from pipeline.common.model_training import modelling_functions as mf
from pipeline.common.model_training import tier_a_baseline as tb
from pipeline.common.model_training import training_config as tc
from pipeline.runtime_paths import (
    database_path,
    models_path,
    project_root as configured_project_root,
)


def _select_blend_weight_by_log_loss(
    y_selected,
    selection_mask,
    baseline_mu_home,
    baseline_mu_away,
    oof_mu_home,
    oof_mu_away,
):
    """OOF blend weight selection; see mf.select_blend_weights_by_log_loss."""
    return mf.select_blend_weights_by_log_loss(
        y_selected,
        np.asarray(baseline_mu_home, dtype=float)[selection_mask],
        np.asarray(baseline_mu_away, dtype=float)[selection_mask],
        np.asarray(oof_mu_home, dtype=float)[selection_mask],
        np.asarray(oof_mu_away, dtype=float)[selection_mask],
    )


# Shared with evaluate.py, which refits them per held-out season.
_estimate_lambda3 = mf.estimate_lambda3
_estimate_dispersion = mf.estimate_dispersion


def _non_draw_mask(df: pd.DataFrame) -> np.ndarray:
    return df["team_final_score_home"].to_numpy(dtype=float) != df[
        "team_final_score_away"
    ].to_numpy(dtype=float)


project_root = configured_project_root()
db_path = database_path(project_root)
models_dir = models_path(project_root)
models_dir.mkdir(parents=True, exist_ok=True)

predictors = tc.filter_predictors(
    include_performance=tc.include_performance, predictor_list=tc.predictors
)

console.emit_progress("loading training data")
print("Get Training Data")
training_data = mf.get_training_data(
    db_path=db_path,
    sql_file=project_root / "pipeline/common/sql/training_data.sql",
)

if training_data.empty:
    raise RuntimeError("Training data is empty. Run data prep first.")

console.emit_progress("computing Tier-A baseline ratings")
print("Computing Tier-A baseline features")
baseline_cfg = tb.default_baseline_config_from_env()
# On by default: the honest eval showed tuned alpha/carryover beat the
# hard-coded 0.2/0.6 on log-loss and Brier. Disable with =false.
if os.getenv("FOOTY_TIPPER_TUNE_TIER_A", "true").strip().lower() not in {
    "0",
    "false",
    "no",
    "n",
    "off",
}:
    print(
        "Tuning Tier-A alpha/carryover (sequential ratings, first season excluded)..."
    )
    baseline_cfg, tier_a_grid = tb.tune_baseline_hyperparams(
        training_data, config_template=baseline_cfg
    )
    if not tier_a_grid.empty:
        best_row = tier_a_grid.sort_values("log_loss").iloc[0]
        print(
            f"Tier-A tuned: alpha={baseline_cfg.alpha:.2f}, carryover={baseline_cfg.carryover:.2f} "
            f"(log-loss {best_row['log_loss']:.4f}, acc {best_row['accuracy']:.1%} on {int(best_row['games'])} games)"
        )
baseline_features = tb.compute_tier_a_baseline_features(training_data, baseline_cfg)
training_data = training_data.merge(baseline_features, on="game_id", how="left")

base_home = float(training_data["team_final_score_home"].mean())
base_away = float(training_data["team_final_score_away"].mean())

training_data["baseline_mu_home"] = pd.to_numeric(
    training_data["baseline_mu_home"], errors="coerce"
).fillna(base_home)
training_data["baseline_mu_away"] = pd.to_numeric(
    training_data["baseline_mu_away"], errors="coerce"
).fillna(base_away)
training_data["baseline_draw_prob"] = pd.to_numeric(
    training_data["baseline_draw_prob"], errors="coerce"
).fillna(0.0)
training_data["baseline_home_win_prob_conditional"] = pd.to_numeric(
    training_data["baseline_home_win_prob_conditional"], errors="coerce"
).fillna(0.5)

console.emit_progress("merging lineup features")
print("Merging lineup-derived features")
try:
    training_years = sorted(
        pd.to_numeric(training_data["competition_year"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    lineup_entries = lf.load_lineup_entries(db_path, years=training_years)
    lineup_features = lf.build_lineup_match_features(training_data, lineup_entries)
    training_data = training_data.merge(lineup_features, on="game_id", how="left")

    training_data = lf.fill_lineup_feature_columns(training_data)

    lineup_coverage = lf.lineup_coverage_fraction(training_data)
    print(f"Lineup features merged. Coverage={lineup_coverage:.1%}")
except Exception as exc:
    import traceback

    traceback.print_exc()
    if os.getenv("FOOTY_TIPPER_LINEUP_FEATURES_STRICT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }:
        raise RuntimeError(
            "Lineup feature merge failed and FOOTY_TIPPER_LINEUP_FEATURES_STRICT is set."
        ) from exc
    print(
        f"Lineup feature merge skipped ({exc}). "
        "The model will train/predict WITHOUT lineup features — "
        "set FOOTY_TIPPER_LINEUP_FEATURES_STRICT=true to make this fatal."
    )

print("Merging match-context features (form/referee/weather/travel + player form)")
try:
    from pipeline.common.nrl_data import features as ctx

    training_data = ctx.merge_match_context_features(training_data, db_path)
except Exception as exc:
    print(f"Match-context feature merge skipped ({exc}).")

training_data = tc.align_predictor_columns(training_data, predictors)
selected_predictors = tc.prune_sparse_predictors(training_data, predictors)
training_data = tc.align_predictor_columns(training_data, selected_predictors)

print(f"Training with {len(selected_predictors)} predictors")

console.emit_progress("training home-score model")
print("Training the model for home team scores")
home_model = mf.train_and_select_best_model(
    training_data,
    selected_predictors,
    "team_final_score_home",
    tc.use_rfe,
    tc.num_folds,
    tc.opt_metric,
)

console.emit_progress("training away-score model")
print("Training the model for away team scores")
away_model = mf.train_and_select_best_model(
    training_data,
    selected_predictors,
    "team_final_score_away",
    tc.use_rfe,
    tc.num_folds,
    tc.opt_metric,
)

print("Blending Tier-A baseline with Tier-B model outputs")
home_model_mu = np.maximum(home_model.predict(training_data[selected_predictors]), 1e-6)
away_model_mu = np.maximum(away_model.predict(training_data[selected_predictors]), 1e-6)

baseline_mu_home = training_data["baseline_mu_home"].to_numpy(dtype=float)
baseline_mu_away = training_data["baseline_mu_away"].to_numpy(dtype=float)

non_draw = _non_draw_mask(training_data)
y_full = (
    training_data["team_final_score_home"].to_numpy(dtype=float)
    > training_data["team_final_score_away"].to_numpy(dtype=float)
).astype(int)

# Generate OOF score predictions BEFORE blend weight selection so weights are
# chosen on unbiased OOF tipping accuracy rather than in-sample Poisson deviance.
print(
    "Generating OOF score predictions for blend weight selection and stacker training..."
)
home_model_mu_oof, home_oof_mask = mf.generate_oof_score_predictions(
    training_data,
    selected_predictors,
    home_model,
    "team_final_score_home",
    return_mask=True,
)
away_model_mu_oof, away_oof_mask = mf.generate_oof_score_predictions(
    training_data,
    selected_predictors,
    away_model,
    "team_final_score_away",
    return_mask=True,
)

# Select weights only on rows whose Tier-B predictions are genuinely OOF:
# first-season rows fall back to in-sample predictions and would bias the
# grid toward Tier B (evaluate.py already masks this way).
blend_mask = non_draw & home_oof_mask & away_oof_mask
if blend_mask.sum() < 50:
    print(
        "Too few genuine-OOF rows for blend selection; falling back to all non-draw rows."
    )
    blend_mask = non_draw
home_weight, away_weight, blend_ll, blend_acc = _select_blend_weight_by_log_loss(
    y_full[blend_mask],
    blend_mask,
    baseline_mu_home,
    baseline_mu_away,
    home_model_mu_oof,
    away_model_mu_oof,
)

blended_mu_home = np.maximum(
    (1.0 - home_weight) * baseline_mu_home + home_weight * home_model_mu, 1e-6
)
blended_mu_away = np.maximum(
    (1.0 - away_weight) * baseline_mu_away + away_weight * away_model_mu, 1e-6
)

lambda3 = _estimate_lambda3(
    training_data["team_final_score_home"].to_numpy(dtype=float),
    training_data["team_final_score_away"].to_numpy(dtype=float),
    blended_mu_home,
    blended_mu_away,
)

# Negative-binomial dispersion from OOF residuals: NRL points are lumpy
# (2/4/6), so the simulation needs fatter tails than Poisson.
disp_mask = home_oof_mask & away_oof_mask
dispersion_home = _estimate_dispersion(
    training_data["team_final_score_home"].to_numpy(dtype=float)[disp_mask],
    np.maximum(
        (1.0 - home_weight) * baseline_mu_home + home_weight * home_model_mu_oof, 1e-6
    )[disp_mask],
)
dispersion_away = _estimate_dispersion(
    training_data["team_final_score_away"].to_numpy(dtype=float)[disp_mask],
    np.maximum(
        (1.0 - away_weight) * baseline_mu_away + away_weight * away_model_mu_oof, 1e-6
    )[disp_mask],
)
print(
    f"Estimated NB dispersion: home k={dispersion_home if dispersion_home is None else f'{dispersion_home:.2f}'}, "
    f"away k={dispersion_away if dispersion_away is None else f'{dispersion_away:.2f}'} "
    "(None = no over-dispersion, plain Poisson)"
)

print(
    f"Selected blend weights: home={home_weight:.2f}, away={away_weight:.2f} "
    f"(OOF log-loss={blend_ll:.4f}, tipping accuracy at chosen weights={blend_acc:.1%})"
)
print(f"Estimated bivariate shared component lambda3={lambda3:.4f}")

console.emit_progress("fitting probability pools")
print("Fitting market and no-market simplex probability pools")
lineup_mc_samples = int(os.getenv("FOOTY_TIPPER_LINEUP_MONTE_CARLO_SAMPLES", "64"))
lineup_mu_noise_scale = float(os.getenv("FOOTY_TIPPER_LINEUP_MU_NOISE_SCALE", "0.12"))

lineup_unc_home = (
    pd.to_numeric(
        training_data.get("lineup_selection_uncertainty_home", 0.0), errors="coerce"
    )
    .fillna(0.0)
    .to_numpy(dtype=float)
)
lineup_unc_away = (
    pd.to_numeric(
        training_data.get("lineup_selection_uncertainty_away", 0.0), errors="coerce"
    )
    .fillna(0.0)
    .to_numpy(dtype=float)
)

tier_a_cond = np.clip(
    training_data["baseline_home_win_prob_conditional"].to_numpy(dtype=float),
    1e-6,
    1 - 1e-6,
)

train_game_ids = training_data["game_id"].to_numpy()
market_cond = pf.derive_market_home_probability(training_data)

# OOF blended mus for stacker training (use weights selected above).
blended_mu_home_oof = np.maximum(
    (1.0 - home_weight) * baseline_mu_home + home_weight * home_model_mu_oof, 1e-6
)
blended_mu_away_oof = np.maximum(
    (1.0 - away_weight) * baseline_mu_away + away_weight * away_model_mu_oof, 1e-6
)

# Marginalise over lineup uncertainty here, not just at inference. The pools
# and calibrator are fitted on this array, and inference.py serves the
# marginalised Tier-B probability, so computing it plainly here would fit the
# meta-layer on a different input from the one it receives in production.
# Deterministic per game via rng_for_game, so re-runs never flip a tip.
tier_b_cond_oof = pf.marginalized_conditional_home_win_prob_vec(
    blended_mu_home_oof,
    blended_mu_away_oof,
    lineup_unc_home,
    lineup_unc_away,
    game_ids=train_game_ids,
    n_samples=lineup_mc_samples,
    mu_noise_scale=lineup_mu_noise_scale,
)

# ── Tier-C: binary LightGBM (OOF) ────────────────────────────────────────────
# Trains a direct binary win/loss classifier using the same hyperparameters as
# the Poisson models. OOF predictions are used for stacker training to avoid bias.
console.emit_progress("generating out-of-fold predictions for the stacker")
print("Generating OOF binary predictions for stacker training...")
best_params = dict(home_model.named_steps["hyperparamtuning"].best_params_)
preprocessor_steps = home_model[:-1]

binary_model_oof, binary_oof_mask = mf.generate_oof_binary_predictions(
    training_data,
    non_draw,
    selected_predictors,
    preprocessor_steps,
    best_params,
    return_mask=True,
)
tier_c_cond_oof = np.clip(binary_model_oof, 1e-6, 1 - 1e-6)

console.emit_progress("training the final classifier + calibrator")
print("Training final binary classifier...")
training_data["_y_binary_col"] = (
    training_data["team_final_score_home"].to_numpy(dtype=float)
    > training_data["team_final_score_away"].to_numpy(dtype=float)
).astype(int)
binary_model = mf.train_binary_classifier(
    training_data[non_draw],
    selected_predictors,
    "_y_binary_col",
    best_params,
    preprocessor_steps,
)
training_data.drop(columns=["_y_binary_col"], inplace=True)

# ── Probability pools (trained on OOF Tier-B + OOF Tier-C) ───────────────────
# Restrict meta-model training to rows whose tier inputs are genuinely
# out-of-fold; first-season rows carry in-sample fallbacks that would bias
# the pools towards the overfit tiers.
genuine_oof = home_oof_mask & away_oof_mask & binary_oof_mask
stacker_fit_mask = non_draw & genuine_oof
if stacker_fit_mask.sum() < 50:
    print(
        f"Only {int(stacker_fit_mask.sum())} genuine-OOF rows available; "
        "falling back to all non-draw rows for meta-model training."
    )
    stacker_fit_mask = non_draw

comp_years_all = pd.to_numeric(
    training_data["competition_year"], errors="coerce"
).to_numpy()

# Market availability is determined from complete raw decimal prices, never
# from the historical 0.5 probability fallback.
valid_market_all = calib.valid_h2h_mask(training_data)
market_fit_mask = stacker_fit_mask & valid_market_all
if market_fit_mask.sum() < 50:
    raise RuntimeError(
        "Fewer than 50 genuine-OOF rows have complete H2H odds; "
        "cannot train the market probability pool safely."
    )

# Market regime: A/B/C plus genuine market probability.
stacker = calib.SimplexLogitPool(include_market=True)
stacker.fit(
    tier_a=tier_a_cond[market_fit_mask],
    tier_b=tier_b_cond_oof[market_fit_mask],
    market=market_cond[market_fit_mask],
    tier_c=tier_c_cond_oof[market_fit_mask],
    y=y_full[market_fit_mask],
)
if not stacker._is_fitted:
    raise RuntimeError("Market simplex probability pool could not be fitted.")
# Captured before selection, which collapses the pool in place. Without this
# the manifest cannot say what a rejected pool had actually learned, and
# diagnosing a collapse means re-reading a stale training log.
market_learned_weights = dict(stacker.weight_map)
print(
    "Market pool weights: "
    + ", ".join(f"{name}={weight:.3f}" for name, weight in stacker.weight_map.items())
)

market_loso = calib.loso_simplex_pool_predictions(
    tier_a=tier_a_cond[market_fit_mask],
    tier_b=tier_b_cond_oof[market_fit_mask],
    market=market_cond[market_fit_mask],
    tier_c=tier_c_cond_oof[market_fit_mask],
    y=y_full[market_fit_mask],
    groups=comp_years_all[market_fit_mask],
    include_market=True,
)
calibrator = calib.TemperatureCalibrator()
if market_loso is not None:
    market_loso_rows = np.isfinite(market_loso)
    calibrator.fit(
        market_loso[market_loso_rows],
        y_full[market_fit_mask][market_loso_rows],
    )
    print(
        "Market temperature calibrated on "
        f"{int(market_loso_rows.sum())} LOSO predictions "
        f"(T={calibrator.temperature_:.3f})."
    )
else:
    print(
        "Market LOSO calibration unavailable (<3 season groups); using identity temperature."
    )

market_nested = calib.nested_loso_simplex_predictions(
    tier_a=tier_a_cond[market_fit_mask],
    tier_b=tier_b_cond_oof[market_fit_mask],
    tier_c=tier_c_cond_oof[market_fit_mask],
    market=market_cond[market_fit_mask],
    y=y_full[market_fit_mask],
    groups=comp_years_all[market_fit_mask],
    include_market=True,
)
market_expert_probabilities = {
    "tier_a": tier_a_cond[market_fit_mask],
    "tier_b": tier_b_cond_oof[market_fit_mask],
    "tier_c": tier_c_cond_oof[market_fit_mask],
    "market": market_cond[market_fit_mask],
}
market_selection = calib.select_market_pool(
    stacker,
    market_nested,
    y_full[market_fit_mask],
    market_expert_probabilities,
    groups=comp_years_all[market_fit_mask],
)
calibrator = calib.fit_selected_market_calibrator(
    market_selection,
    market_expert_probabilities,
    y_full[market_fit_mask],
    calibrator,
)
print(
    "Market pool nested selection: "
    f"{market_selection['selected']} "
    f"({market_selection['selection_rows']} rows)."
)

# Counterfactual no-market regime: A/B/C on every genuine-OOF row, regardless
# of whether that historical fixture happened to have market prices.
stacker_no_market = calib.SimplexLogitPool(include_market=False)
stacker_no_market.fit(
    tier_a=tier_a_cond[stacker_fit_mask],
    tier_b=tier_b_cond_oof[stacker_fit_mask],
    tier_c=tier_c_cond_oof[stacker_fit_mask],
    y=y_full[stacker_fit_mask],
)
if not stacker_no_market._is_fitted:
    raise RuntimeError("No-market simplex probability pool could not be fitted.")
no_market_learned_weights = stacker_no_market.weight_map
print(
    "No-market learned pool weights: "
    + ", ".join(
        f"{name}={weight:.3f}" for name, weight in no_market_learned_weights.items()
    )
)

no_market_loso = calib.loso_simplex_pool_predictions(
    tier_a=tier_a_cond[stacker_fit_mask],
    tier_b=tier_b_cond_oof[stacker_fit_mask],
    tier_c=tier_c_cond_oof[stacker_fit_mask],
    y=y_full[stacker_fit_mask],
    groups=comp_years_all[stacker_fit_mask],
    include_market=False,
)
calibrator_no_market = calib.TemperatureCalibrator()
if no_market_loso is not None:
    no_market_loso_rows = np.isfinite(no_market_loso)
    calibrator_no_market.fit(
        no_market_loso[no_market_loso_rows],
        y_full[stacker_fit_mask][no_market_loso_rows],
    )
else:
    print("No-market LOSO calibration unavailable; eligibility will fall back safely.")

no_market_nested = calib.nested_loso_simplex_predictions(
    tier_a_cond[stacker_fit_mask],
    tier_b_cond_oof[stacker_fit_mask],
    y_full[stacker_fit_mask],
    comp_years_all[stacker_fit_mask],
    tier_c=tier_c_cond_oof[stacker_fit_mask],
    include_market=False,
)
no_market_expert_probabilities = {
    "tier_a": tier_a_cond[stacker_fit_mask],
    "tier_b": tier_b_cond_oof[stacker_fit_mask],
    "tier_c": tier_c_cond_oof[stacker_fit_mask],
}
no_market_selection = calib.select_no_market_pool(
    stacker_no_market,
    no_market_nested,
    y_full[stacker_fit_mask],
    no_market_expert_probabilities,
    groups=comp_years_all[stacker_fit_mask],
)
no_market_strategy = no_market_selection["strategy"]
calibrator_no_market = calib.fit_selected_pool_calibrator(
    no_market_selection,
    no_market_expert_probabilities,
    y_full[stacker_fit_mask],
    calibrator_no_market,
)
no_market_eligibility = no_market_selection["eligibility"]
no_market_pool_log_loss = no_market_eligibility["pool_log_loss"]
no_market_tier_b_log_loss = no_market_eligibility["tier_b_log_loss"]
no_market_selection_rows = no_market_selection["selection_rows"]
print(
    "No-market nested selection: "
    f"strategy={no_market_strategy}, "
    f"selected={no_market_selection['selected']}, "
    f"eligible={no_market_selection['eligible']} "
    f"on {no_market_selection_rows} rows "
    f"(pool log-loss={no_market_pool_log_loss}, "
    f"Tier-B log-loss={no_market_tier_b_log_loss}, "
    f"T={calibrator_no_market.temperature_:.3f})."
)
print(
    "No-market selected pool weights: "
    + ", ".join(
        f"{name}={weight:.3f}" for name, weight in stacker_no_market.weight_map.items()
    )
)

market_stacked_all = stacker.predict(
    tier_a_cond,
    tier_b_cond_oof,
    tier_c=tier_c_cond_oof,
    market=market_cond,
)
market_calibrated_all = calibrator.predict(market_stacked_all)
no_market_stacked_all = stacker_no_market.predict(
    tier_a_cond,
    tier_b_cond_oof,
    tier_c=tier_c_cond_oof,
)
no_market_calibrated_all = calibrator_no_market.predict(no_market_stacked_all)
calibrated_oof = np.where(
    valid_market_all,
    market_calibrated_all,
    no_market_calibrated_all if no_market_strategy == "simplex" else tier_b_cond_oof,
)
calibrated_oof, training_consensus_guard = calib.apply_consensus_guard(
    calibrated_oof,
    tier_a_cond,
    tier_b_cond_oof,
    tier_c=tier_c_cond_oof,
    market=market_cond,
    valid_market=valid_market_all,
)
if training_consensus_guard.any():
    print(
        "Consensus guard replaced "
        f"{int(training_consensus_guard.sum())} side-reversing OOF prediction(s) with Tier B."
    )

# ── Margin blend for the comp tie-breaker ─────────────────────────────────────
# Small ridge on honest inputs: OOF model margin, the line market's expected
# margin, and the Tier-A margin. Three coefficients, so in-sample MAE is a
# fair guide; the season-out gate lives in evaluate.py.
margin_blend = None
try:
    actual_margin_all = training_data["team_final_score_home"].to_numpy(
        dtype=float
    ) - training_data["team_final_score_away"].to_numpy(dtype=float)
    model_margin_oof = blended_mu_home_oof - blended_mu_away_oof
    tier_a_margin = baseline_mu_home - baseline_mu_away
    market_spread_arr = -pd.to_numeric(
        training_data.get("implied_spread_home", np.nan), errors="coerce"
    ).to_numpy(dtype=float)
    margin_mask = (
        genuine_oof & np.isfinite(market_spread_arr) & np.isfinite(actual_margin_all)
    )
    if margin_mask.sum() >= 100:
        X_margin = np.column_stack(
            [model_margin_oof, market_spread_arr, tier_a_margin]
        )[margin_mask]
        margin_model = Ridge(alpha=1.0)
        margin_model.fit(X_margin, actual_margin_all[margin_mask])
        blend_mae = float(
            np.mean(
                np.abs(margin_model.predict(X_margin) - actual_margin_all[margin_mask])
            )
        )
        model_only_mae = float(
            np.mean(
                np.abs(model_margin_oof[margin_mask] - actual_margin_all[margin_mask])
            )
        )
        market_only_mae = float(
            np.mean(
                np.abs(market_spread_arr[margin_mask] - actual_margin_all[margin_mask])
            )
        )
        margin_blend = {
            "intercept": float(margin_model.intercept_),
            "coef_model_margin": float(margin_model.coef_[0]),
            "coef_market_spread": float(margin_model.coef_[1]),
            "coef_tier_a_margin": float(margin_model.coef_[2]),
            "fit_rows": int(margin_mask.sum()),
            "fit_mae": blend_mae,
            "fit_mae_model_only": model_only_mae,
            "fit_mae_market_only": market_only_mae,
        }
        print(
            f"Margin blend fitted: MAE {blend_mae:.2f} vs model-only {model_only_mae:.2f} "
            f"vs market-only {market_only_mae:.2f} ({int(margin_mask.sum())} rows)"
        )
    else:
        print(
            "Margin blend skipped (too few rows with line odds and genuine-OOF margins)."
        )
except Exception as exc:
    print(f"Margin blend skipped ({exc}).")

# ── Totals blend for the scoreline simulation ─────────────────────────────────
# The totals market carries information about game pace/conditions the score
# models may miss. Fit a two-coefficient ridge (OOF model total, market total
# line) on honest inputs; inference rescales both lambdas toward the blended
# expected total, preserving the margin split.
total_blend = None
try:
    actual_total_all = training_data["team_final_score_home"].to_numpy(
        dtype=float
    ) + training_data["team_final_score_away"].to_numpy(dtype=float)
    model_total_oof = blended_mu_home_oof + blended_mu_away_oof
    market_total_arr = pd.to_numeric(
        training_data.get("market_total_line", np.nan), errors="coerce"
    ).to_numpy(dtype=float)
    total_mask = (
        genuine_oof & np.isfinite(market_total_arr) & np.isfinite(actual_total_all)
    )
    if total_mask.sum() >= 100:
        X_total = np.column_stack([model_total_oof, market_total_arr])[total_mask]
        total_model = Ridge(alpha=1.0)
        total_model.fit(X_total, actual_total_all[total_mask])
        total_mae = float(
            np.mean(np.abs(total_model.predict(X_total) - actual_total_all[total_mask]))
        )
        model_total_mae = float(
            np.mean(np.abs(model_total_oof[total_mask] - actual_total_all[total_mask]))
        )
        market_total_mae = float(
            np.mean(np.abs(market_total_arr[total_mask] - actual_total_all[total_mask]))
        )
        total_blend = {
            "intercept": float(total_model.intercept_),
            "coef_model_total": float(total_model.coef_[0]),
            "coef_market_total": float(total_model.coef_[1]),
            "fit_rows": int(total_mask.sum()),
            "fit_mae": total_mae,
            "fit_mae_model_only": model_total_mae,
            "fit_mae_market_only": market_total_mae,
        }
        print(
            f"Total blend fitted: MAE {total_mae:.2f} vs model-only {model_total_mae:.2f} "
            f"vs market-only {market_total_mae:.2f} ({int(total_mask.sum())} rows)"
        )
    else:
        print(
            "Total blend skipped (too few rows with totals lines and genuine-OOF totals)."
        )
except Exception as exc:
    print(f"Total blend skipped ({exc}).")

# ── Evaluation metrics ────────────────────────────────────────────────────────
try:
    # Evaluate only on genuine-OOF rows; first-season fallback rows are
    # in-sample and would flatter every number below.
    eval_mask = stacker_fit_mask
    y_eval = y_full[eval_mask]
    nd_preds = np.clip(calibrated_oof[eval_mask], 1e-6, 1 - 1e-6)
    market_nd = market_cond[eval_mask]
    # Raw, complete two-sided prices identify games with genuine market data.
    valid_market = valid_market_all[eval_mask]

    # ── PRIMARY: Tipping accuracy ─────────────────────────────────────────────
    tip_correct = (nd_preds > 0.5) == y_eval.astype(bool)
    tip_acc = tip_correct.mean()
    naive_home_acc = float(y_eval.mean())  # always-pick-home baseline

    print("\n── Tipping accuracy (genuine OOF, non-draw) ────────────────────────────")
    print(
        f"  Model:       {tip_acc:.1%}  ({tip_correct.sum()}/{len(tip_correct)} correct)"
    )
    print(f"  Always home: {naive_home_acc:.1%}")

    if valid_market.sum() >= 10:
        market_tip = (market_nd[valid_market] > 0.5) == y_eval[valid_market].astype(
            bool
        )
        model_tip_on_mkt = (nd_preds[valid_market] > 0.5) == y_eval[
            valid_market
        ].astype(bool)
        diff = model_tip_on_mkt.mean() - market_tip.mean()
        print(
            f"  Market fav:  {market_tip.mean():.1%}  (on {valid_market.sum()} games with odds)"
        )
        print(
            f"  Model (same games): {model_tip_on_mkt.mean():.1%}  ({'▲' if diff > 0 else '▼'} {abs(diff):.1%} vs market)"
        )

    # ── SECONDARY: Probabilistic calibration ──────────────────────────────────
    nd_log_loss = log_loss(y_eval, nd_preds)
    nd_brier = brier_score_loss(y_eval, nd_preds)
    print("\n── Calibration (genuine OOF, non-draw) ─────────────────────────────────")
    print(f"  Log-loss  (model):   {nd_log_loss:.4f}")
    print(f"  Brier     (model):   {nd_brier:.4f}")

    if valid_market.sum() >= 10:
        market_ll = log_loss(
            y_eval[valid_market], np.clip(market_nd[valid_market], 1e-6, 1 - 1e-6)
        )
        market_br = brier_score_loss(
            y_eval[valid_market], np.clip(market_nd[valid_market], 1e-6, 1 - 1e-6)
        )
        model_ll = log_loss(
            y_eval[valid_market], np.clip(nd_preds[valid_market], 1e-6, 1 - 1e-6)
        )
        model_br = brier_score_loss(
            y_eval[valid_market], np.clip(nd_preds[valid_market], 1e-6, 1 - 1e-6)
        )
        print(
            f"  Log-loss  (market benchmark): {market_ll:.4f}  |  model: {model_ll:.4f}  ({'▲ better' if model_ll < market_ll else '▼ worse'})"
        )
        print(
            f"  Brier     (market benchmark): {market_br:.4f}  |  model: {model_br:.4f}  ({'▲ better' if model_br < market_br else '▼ worse'})"
        )

    # Lift the headline numbers onto the operator's console. The parent captures
    # this script's stdout, so without a result marker a seven-minute train
    # finishes showing nothing but a tick.
    summary_rows = [
        ("Tipping accuracy (OOF)", f"{tip_acc:.1%}  ({int(tip_correct.sum())}/{len(tip_correct)})"),
        ("Log loss / Brier", f"{nd_log_loss:.4f} / {nd_brier:.4f}"),
        ("Blend weights (home/away)", f"{home_weight:.2f} / {away_weight:.2f}"),
        ("NB dispersion (home/away)", f"{dispersion_home:.2f} / {dispersion_away:.2f}"
            if dispersion_home and dispersion_away else "plain Poisson"),
        ("Shared component lambda3", f"{lambda3:.4f}"),
        ("Market pool", str(market_selection.get("selected", "n/a"))),
        ("No-market pool", str(no_market_selection.get("selected", "n/a"))),
    ]
    if valid_market.sum() >= 10:
        summary_rows.insert(
            1,
            (
                "vs market favourite",
                f"{model_tip_on_mkt.mean():.1%} model vs {market_tip.mean():.1%} market "
                f"({'+' if diff > 0 else ''}{diff:.1%} on {int(valid_market.sum())} games)",
            ),
        )
    console.emit_result("training_summary", rows=summary_rows)

    # Calibration reliability table (predicted probability bins vs actual win rate).
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    print(f"\n── Calibration reliability (non-draw, {n_bins} bins) ───────────────")
    print(f"  {'Pred range':<14} {'Pred mean':>10} {'Actual':>8} {'Count':>7}")
    for i in range(n_bins):
        mask = (nd_preds >= bins[i]) & (nd_preds < bins[i + 1])
        if mask.sum() < 3:
            continue
        pred_mean = nd_preds[mask].mean()
        actual_mean = y_eval[mask].mean()
        gap = actual_mean - pred_mean
        flag = "  ◄ over" if gap < -0.05 else ("  ► under" if gap > 0.05 else "")
        print(
            f"  {bins[i]:.1f}–{bins[i + 1]:.1f}         {pred_mean:>10.3f} {actual_mean:>8.3f} {mask.sum():>7}{flag}"
        )

except Exception as exc:
    import traceback

    traceback.print_exc()
    print(f"Skipped evaluation metrics ({exc}).")

# ── Holdout year evaluation (last year) ───────────────────────────────────────
try:
    comp_years = pd.to_numeric(training_data["competition_year"], errors="coerce")
    last_year = int(comp_years.max())
    holdout_mask = (comp_years == last_year).values

    nd_holdout = non_draw & holdout_mask
    if nd_holdout.sum() >= 5:
        model_p = calibrated_oof[nd_holdout]
        market_p = market_cond[nd_holdout]
        actuals = y_full[nd_holdout]

        # ── PRIMARY: Tipping accuracy on holdout ──────────────────────────────
        tip_correct_holdout = (model_p > 0.5) == actuals.astype(bool)
        tip_acc_holdout = tip_correct_holdout.mean()

        valid_mkt_holdout = valid_market_all[nd_holdout]
        print(
            f"\n── Tipping accuracy ({last_year} holdout, non-draw) ─────────────────"
        )
        print(
            f"  Model:       {tip_acc_holdout:.1%}  ({tip_correct_holdout.sum()}/{len(tip_correct_holdout)} correct)"
        )
        if valid_mkt_holdout.sum() >= 3:
            mkt_tip_holdout = (market_p[valid_mkt_holdout] > 0.5) == actuals[
                valid_mkt_holdout
            ].astype(bool)
            print(
                f"  Market fav:  {mkt_tip_holdout.mean():.1%}  (on {valid_mkt_holdout.sum()} games with odds)"
            )

        # ── SECONDARY: ROI betting simulation ─────────────────────────────────
        home_odds_col = (
            training_data["team_head_to_head_odds_home"]
            if "team_head_to_head_odds_home" in training_data.columns
            else pd.Series(np.nan, index=training_data.index)
        )
        away_odds_col = (
            training_data["team_head_to_head_odds_away"]
            if "team_head_to_head_odds_away" in training_data.columns
            else pd.Series(np.nan, index=training_data.index)
        )
        home_odds_raw = pd.to_numeric(home_odds_col, errors="coerce").values[nd_holdout]
        away_odds_raw = pd.to_numeric(away_odds_col, errors="coerce").values[nd_holdout]

        edge = model_p - market_p
        threshold = 0.05
        total_bets, wins, profit = 0, 0, 0.0

        for e, act, oh, oa in zip(edge, actuals, home_odds_raw, away_odds_raw):
            if e > threshold and np.isfinite(oh):
                total_bets += 1
                profit += (oh - 1.0) if act == 1 else -1.0
                wins += int(act == 1)
            elif e < -threshold and np.isfinite(oa):
                total_bets += 1
                profit += (oa - 1.0) if act == 0 else -1.0
                wins += int(act == 0)

        print(
            f"\n── ROI simulation ({last_year} holdout, ≥{threshold:.0%} edge) ──────────────"
        )
        print(
            f"  Games in holdout: {nd_holdout.sum()},  edge bets placed: {total_bets}"
        )
        if total_bets > 0:
            roi_pct = 100.0 * profit / total_bets
            win_rate = 100.0 * wins / total_bets
            print(
                f"  Wins: {wins}/{total_bets} ({win_rate:.1f}%),  flat-stake ROI: {roi_pct:+.1f}%"
            )
        else:
            print("  No games exceeded the edge threshold.")
        print(
            "  NOTE: blend weights, stacker, and calibrator were fitted using OOF rows "
            "from this season too — run `footy-tipper advanced model evaluate` "
            "for fully held-out numbers."
        )
except Exception as exc:
    import traceback

    traceback.print_exc()
    print(f"Holdout evaluation skipped ({exc}).")

print("Save model artefacts")
mf.save_models(home_model, "home_model", project_root, models_dir=models_dir)
mf.save_models(away_model, "away_model", project_root, models_dir=models_dir)
mf.save_models(binary_model, "binary_model", project_root, models_dir=models_dir)
calib.save_artifact(stacker, models_dir / "stacker.pkl")
calib.save_artifact(calibrator, models_dir / "win_prob_calibrator.pkl")
calib.save_artifact(stacker_no_market, models_dir / "stacker_no_market.pkl")
calib.save_artifact(
    calibrator_no_market,
    models_dir / "win_prob_calibrator_no_market.pkl",
)

manifest = {
    "predictors": selected_predictors,
    "blend_weight_home": home_weight,
    "blend_weight_away": away_weight,
    "lambda3": lambda3,
    "lineup_monte_carlo_samples": lineup_mc_samples,
    "lineup_mu_noise_scale": lineup_mu_noise_scale,
    "tier_a_baseline": tb.baseline_config_to_dict(baseline_cfg, base_home, base_away),
    "margin_blend": margin_blend,
    "total_blend": total_blend,
    "market_extra_version": calib.MARKET_EXTRA_VERSION,
    # Baked into the fitted transformer, recorded here so the deployed feature
    # space is auditable from the manifest alone.
    "nan_passthrough": mf.nan_passthrough_enabled(),
    # Recorded so a release can be reproduced exactly from this manifest.
    "training_seed": mf.training_seed(),
    "probability_stack": {
        "schema_version": calib.PROBABILITY_STACK_VERSION,
        "market": {
            "strategy": "simplex",
            "stacker_file": "stacker.pkl",
            "calibrator_file": "win_prob_calibrator.pkl",
            "fit_rows": int(market_fit_mask.sum()),
            "experts": list(stacker.expert_names_),
            "learned_weights": market_learned_weights,
            "weights": stacker.weight_map,
            "temperature": float(calibrator.temperature_),
            "selection": market_selection,
        },
        "no_market": {
            "strategy": no_market_strategy,
            "stacker_file": "stacker_no_market.pkl",
            "calibrator_file": "win_prob_calibrator_no_market.pkl",
            "fit_rows": int(stacker_fit_mask.sum()),
            "selection": no_market_selection,
            "eligibility": no_market_eligibility,
            "selection_rows": no_market_selection_rows,
            "pool_log_loss": no_market_pool_log_loss,
            "tier_b_log_loss": no_market_tier_b_log_loss,
            "experts": list(stacker_no_market.expert_names_),
            "learned_weights": no_market_learned_weights,
            "weights": stacker_no_market.weight_map,
            "temperature": float(calibrator_no_market.temperature_),
        },
    },
    "dispersion_home": dispersion_home,
    "dispersion_away": dispersion_away,
}

print("Running joker strategy backtest")
try:
    joker_policy_path = models_dir / "joker_policy.json"
    joker_policy = jp.save_joker_policy(training_data, joker_policy_path)
    manifest["joker_policy_file"] = str(joker_policy_path.name)
    manifest["joker_policy_default_strategy"] = joker_policy.get(
        "default_strategy", "points"
    )
    manifest["joker_policy_seasons_evaluated"] = int(
        joker_policy.get("seasons_evaluated", 0)
    )
    print(
        "Joker policy saved "
        f"(default={manifest['joker_policy_default_strategy']}, "
        f"seasons={manifest['joker_policy_seasons_evaluated']})"
    )
except Exception as exc:
    print(f"Joker policy backtest skipped ({exc}).")

manifest_path = models_dir / "model_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Saved manifest to {manifest_path}")

print("Model training complete!")
