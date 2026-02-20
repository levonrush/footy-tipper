# Description: This script trains score models and saves artefacts used at inference.
print("Running the train.py script...")

import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_poisson_deviance

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from pipeline.common.model_prediciton import prediction_functions as pf
from pipeline.common.model_training import calibration as calib
from pipeline.common.model_training import modelling_functions as mf
from pipeline.common.model_training import tier_a_baseline as tb
from pipeline.common.model_training import training_config as tc


def _select_blend_weight(y_true, baseline_mu, model_mu):
    candidates = np.linspace(0.0, 1.0, 21)
    best_weight = 1.0
    best_score = np.inf

    y_true = np.asarray(y_true, dtype=float)
    baseline_mu = np.asarray(baseline_mu, dtype=float)
    model_mu = np.asarray(model_mu, dtype=float)

    for weight in candidates:
        blended = np.maximum((1.0 - weight) * baseline_mu + weight * model_mu, 1e-6)
        score = mean_poisson_deviance(y_true, blended)
        if score < best_score:
            best_score = score
            best_weight = float(weight)

    return best_weight, best_score


def _estimate_lambda3(y_home, y_away, mu_home, mu_away):
    y_home = np.asarray(y_home, dtype=float)
    y_away = np.asarray(y_away, dtype=float)
    mu_home = np.asarray(mu_home, dtype=float)
    mu_away = np.asarray(mu_away, dtype=float)

    if len(y_home) < 2:
        return 0.0

    resid_home = y_home - mu_home
    resid_away = y_away - mu_away
    cov = float(np.cov(resid_home, resid_away, ddof=1)[0, 1])
    lambda3 = max(0.0, cov)

    # Cap to keep most matches in a feasible shared-component range.
    cap = float(np.quantile(np.minimum(mu_home, mu_away), 0.25) * 0.8)
    return float(max(0.0, min(lambda3, max(cap, 0.0))))


def _non_draw_mask(df: pd.DataFrame) -> np.ndarray:
    return (df["team_final_score_home"].to_numpy(dtype=float) != df["team_final_score_away"].to_numpy(dtype=float))


project_root = pathlib.Path().absolute()
db_path = project_root / "data" / "footy-tipper-db.sqlite"

predictors = tc.filter_predictors(include_performance=tc.include_performance, predictor_list=tc.predictors)

print("Get Training Data")
training_data = mf.get_training_data(
    db_path=db_path,
    sql_file=project_root / "pipeline/common/sql/training_data.sql",
)

if training_data.empty:
    raise RuntimeError("Training data is empty. Run data prep first.")

print("Computing Tier-A baseline features")
baseline_cfg = tb.default_baseline_config_from_env()
baseline_features = tb.compute_tier_a_baseline_features(training_data, baseline_cfg)
training_data = training_data.merge(baseline_features, on="game_id", how="left")

base_home = float(training_data["team_final_score_home"].mean())
base_away = float(training_data["team_final_score_away"].mean())

training_data["baseline_mu_home"] = pd.to_numeric(training_data["baseline_mu_home"], errors="coerce").fillna(base_home)
training_data["baseline_mu_away"] = pd.to_numeric(training_data["baseline_mu_away"], errors="coerce").fillna(base_away)
training_data["baseline_draw_prob"] = pd.to_numeric(training_data["baseline_draw_prob"], errors="coerce").fillna(0.0)
training_data["baseline_home_win_prob_conditional"] = (
    pd.to_numeric(training_data["baseline_home_win_prob_conditional"], errors="coerce").fillna(0.5)
)

training_data = tc.align_predictor_columns(training_data, predictors)
selected_predictors = tc.prune_sparse_predictors(training_data, predictors)
training_data = tc.align_predictor_columns(training_data, selected_predictors)

print(f"Training with {len(selected_predictors)} predictors")

print("Training the model for home team scores")
home_model = mf.train_and_select_best_model(
    training_data,
    selected_predictors,
    "team_final_score_home",
    tc.use_rfe,
    tc.num_folds,
    tc.opt_metric,
)

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

home_weight, home_dev = _select_blend_weight(
    training_data["team_final_score_home"].to_numpy(dtype=float),
    baseline_mu_home,
    home_model_mu,
)
away_weight, away_dev = _select_blend_weight(
    training_data["team_final_score_away"].to_numpy(dtype=float),
    baseline_mu_away,
    away_model_mu,
)

blended_mu_home = np.maximum((1.0 - home_weight) * baseline_mu_home + home_weight * home_model_mu, 1e-6)
blended_mu_away = np.maximum((1.0 - away_weight) * baseline_mu_away + away_weight * away_model_mu, 1e-6)

lambda3 = _estimate_lambda3(
    training_data["team_final_score_home"].to_numpy(dtype=float),
    training_data["team_final_score_away"].to_numpy(dtype=float),
    blended_mu_home,
    blended_mu_away,
)

print(f"Selected blend weights: home={home_weight:.2f}, away={away_weight:.2f}")
print(f"In-sample blended deviance: home={home_dev:.4f}, away={away_dev:.4f}")
print(f"Estimated bivariate shared component lambda3={lambda3:.4f}")

print("Fitting stacking model and beta calibrator")
tier_a_cond = np.clip(training_data["baseline_home_win_prob_conditional"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
tier_b_cond = np.array(
    [
        pf.conditional_home_win_prob(mh, ma)
        for mh, ma in zip(blended_mu_home, blended_mu_away)
    ],
    dtype=float,
)
market_cond = pf.derive_market_home_probability(training_data)
if "odds_missing" in training_data.columns:
    odds_missing = pd.to_numeric(training_data["odds_missing"], errors="coerce").fillna(0).to_numpy(dtype=float)
else:
    odds_missing = np.zeros(len(training_data), dtype=float)

non_draw = _non_draw_mask(training_data)
y_binary = (
    training_data.loc[non_draw, "team_final_score_home"].to_numpy(dtype=float)
    > training_data.loc[non_draw, "team_final_score_away"].to_numpy(dtype=float)
).astype(int)

stacker = calib.LogisticStacker()
stacker.fit(
    tier_a=tier_a_cond[non_draw],
    tier_b=tier_b_cond[non_draw],
    market=market_cond[non_draw],
    odds_missing=odds_missing[non_draw],
    y=y_binary,
)
stacked_cond = stacker.predict(tier_a_cond, tier_b_cond, market_cond, odds_missing)

calibrator = calib.BetaCalibrator()
calibrator.fit(stacked_cond[non_draw], y_binary)
calibrated_cond = calibrator.predict(stacked_cond)

try:
    nd_log_loss = log_loss(y_binary, np.clip(calibrated_cond[non_draw], 1e-6, 1 - 1e-6))
    print(f"Non-draw calibrated log loss (train): {nd_log_loss:.4f}")
except Exception:
    print("Skipped non-draw log-loss calculation (insufficient class variation).")

print("Save model artefacts")
mf.save_models(home_model, "home_model", project_root)
mf.save_models(away_model, "away_model", project_root)
calib.save_artifact(stacker, project_root / "models" / "stacker.pkl")
calib.save_artifact(calibrator, project_root / "models" / "win_prob_calibrator.pkl")

manifest = {
    "predictors": selected_predictors,
    "blend_weight_home": home_weight,
    "blend_weight_away": away_weight,
    "lambda3": lambda3,
    "tier_a_baseline": tb.baseline_config_to_dict(baseline_cfg, base_home, base_away),
}

manifest_path = project_root / "models" / "model_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Saved manifest to {manifest_path}")

print("Model training complete!")
