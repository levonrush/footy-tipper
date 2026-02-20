# Description: Make predictions using trained artefacts and save them to SQLite.
print("Running the inference.py script...")

import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from pipeline.common.model_prediciton import prediction_functions as pf
from pipeline.common.model_training import calibration as calib
from pipeline.common.model_training import tier_a_baseline as tb
from pipeline.common.model_training import training_config as tc


project_root = pathlib.Path().absolute()
db_path = project_root / "data" / "footy-tipper-db.sqlite"

manifest_path = project_root / "models" / "model_manifest.json"
manifest = {}
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

predictors = manifest.get(
    "predictors",
    tc.filter_predictors(include_performance=tc.include_performance, predictor_list=tc.predictors),
)
blend_weight_home = float(manifest.get("blend_weight_home", 1.0))
blend_weight_away = float(manifest.get("blend_weight_away", 1.0))
lambda3 = float(manifest.get("lambda3", 0.0))

baseline_cfg_payload = manifest.get("tier_a_baseline", {})
if baseline_cfg_payload:
    baseline_cfg = tb.baseline_config_from_dict(baseline_cfg_payload)
else:
    baseline_cfg = tb.default_baseline_config_from_env()

home_model = pf.load_models("home_model", project_root)
away_model = pf.load_models("away_model", project_root)

stacker = None
calibrator = None
stacker_path = project_root / "models" / "stacker.pkl"
calibrator_path = project_root / "models" / "win_prob_calibrator.pkl"
if stacker_path.exists():
    stacker = calib.load_artifact(stacker_path)
if calibrator_path.exists():
    calibrator = calib.load_artifact(calibrator_path)

inference_data = pf.get_inference_data(db_path, project_root / "pipeline/common/sql/inference_data.sql")
if inference_data.empty:
    print("Inference dataset is empty. Writing empty prediction batch.")
    outcomes, margins = pf.predict_match_outcome_and_scoreline_with_bayes(inference_data=inference_data)
    outcome_df = pd.merge(outcomes, margins, on="game_id", how="left")
    pf.save_predictions_to_db(
        outcome_df,
        db_path,
        project_root / "pipeline/common/sql/create_table.sql",
        project_root / "pipeline/common/sql/insert_into_table.sql",
    )
    print("Predictions saved to the database!")
    raise SystemExit(0)

# Build Tier-A baseline on full context so pre-game rows inherit latest team state.
try:
    context_data = pf.get_table_data(db_path, "footy_tipping_data")
except Exception:
    context_data = inference_data.copy()

baseline_features = tb.compute_tier_a_baseline_features(context_data, baseline_cfg)
inference_data = inference_data.merge(baseline_features, on="game_id", how="left")

default_home = float(getattr(baseline_cfg, "base_home", 22.0) or 22.0)
default_away = float(getattr(baseline_cfg, "base_away", 20.0) or 20.0)

inference_data["baseline_mu_home"] = pd.to_numeric(inference_data["baseline_mu_home"], errors="coerce").fillna(default_home)
inference_data["baseline_mu_away"] = pd.to_numeric(inference_data["baseline_mu_away"], errors="coerce").fillna(default_away)
inference_data["baseline_draw_prob"] = pd.to_numeric(inference_data["baseline_draw_prob"], errors="coerce").fillna(0.0)
inference_data["baseline_home_win_prob_conditional"] = (
    pd.to_numeric(inference_data["baseline_home_win_prob_conditional"], errors="coerce").fillna(0.5)
)

inference_data = tc.align_predictor_columns(inference_data, predictors)

home_model_mu = np.maximum(pf.predict_scores(home_model, inference_data[predictors]), 1e-6)
away_model_mu = np.maximum(pf.predict_scores(away_model, inference_data[predictors]), 1e-6)

baseline_mu_home = inference_data["baseline_mu_home"].to_numpy(dtype=float)
baseline_mu_away = inference_data["baseline_mu_away"].to_numpy(dtype=float)

blended_mu_home = np.maximum((1.0 - blend_weight_home) * baseline_mu_home + blend_weight_home * home_model_mu, 1e-6)
blended_mu_away = np.maximum((1.0 - blend_weight_away) * baseline_mu_away + blend_weight_away * away_model_mu, 1e-6)

# Tier A / Tier B / market conditional win probabilities for stacking.
tier_a_cond = np.clip(inference_data["baseline_home_win_prob_conditional"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
tier_b_cond = np.array([pf.conditional_home_win_prob(mh, ma) for mh, ma in zip(blended_mu_home, blended_mu_away)], dtype=float)
market_cond = pf.derive_market_home_probability(inference_data)
if "odds_missing" in inference_data.columns:
    odds_missing = pd.to_numeric(inference_data["odds_missing"], errors="coerce").fillna(0).to_numpy(dtype=float)
else:
    odds_missing = np.zeros(len(inference_data), dtype=float)

if stacker is not None:
    stacked_cond = stacker.predict(tier_a_cond, tier_b_cond, market_cond, odds_missing)
else:
    stacked_cond = tier_b_cond

if calibrator is not None:
    calibrated_cond = calibrator.predict(stacked_cond)
else:
    calibrated_cond = stacked_cond

outcomes, margins = pf.predict_match_outcome_and_scoreline_with_bayes(
    inference_data=inference_data,
    mu_home=blended_mu_home,
    mu_away=blended_mu_away,
    lambda3=lambda3,
    calibrated_home_win_conditional=calibrated_cond,
)
outcome_df = pd.merge(outcomes, margins, on="game_id")

pf.save_predictions_to_db(
    outcome_df,
    db_path,
    project_root / "pipeline/common/sql/create_table.sql",
    project_root / "pipeline/common/sql/insert_into_table.sql",
)

print("Predictions saved to the database!")
