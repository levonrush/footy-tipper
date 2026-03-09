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
from pipeline.common.lineups import features as lf
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
lineup_mc_samples = int(manifest.get("lineup_monte_carlo_samples", os.getenv("FOOTY_TIPPER_LINEUP_MONTE_CARLO_SAMPLES", "64")))
lineup_mu_noise_scale = float(manifest.get("lineup_mu_noise_scale", os.getenv("FOOTY_TIPPER_LINEUP_MU_NOISE_SCALE", "0.12")))

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

print("Merging lineup-derived features")
try:
    context_years = sorted(
        pd.to_numeric(context_data["competition_year"], errors="coerce").dropna().astype(int).unique().tolist()
    )
    lineup_entries = lf.load_lineup_entries(db_path, years=context_years)
    context_lineup_features = lf.build_lineup_match_features(context_data, lineup_entries)
    inference_lineup_features = context_lineup_features[context_lineup_features["game_id"].isin(inference_data["game_id"])]
    inference_data = inference_data.merge(inference_lineup_features, on="game_id", how="left")

    for col in lf.LINEUP_FEATURE_COLUMNS:
        if col == "game_id":
            continue
        if col in {"lineup_home_players", "lineup_away_players"}:
            inference_data[col] = inference_data[col].fillna("")
        else:
            inference_data[col] = pd.to_numeric(inference_data[col], errors="coerce").fillna(0.0)

    lineup_coverage = 0.0
    if "lineup_features_missing" in inference_data.columns and len(inference_data) > 0:
        lineup_coverage = float((inference_data["lineup_features_missing"] <= 0).mean())
    print(f"Lineup features merged for inference. Coverage={lineup_coverage:.1%}")
except Exception as exc:
    print(f"Lineup feature merge skipped ({exc}).")

inference_data = tc.align_predictor_columns(inference_data, predictors)

home_model_mu = np.maximum(pf.predict_scores(home_model, inference_data[predictors]), 1e-6)
away_model_mu = np.maximum(pf.predict_scores(away_model, inference_data[predictors]), 1e-6)

baseline_mu_home = inference_data["baseline_mu_home"].to_numpy(dtype=float)
baseline_mu_away = inference_data["baseline_mu_away"].to_numpy(dtype=float)

blended_mu_home = np.maximum((1.0 - blend_weight_home) * baseline_mu_home + blend_weight_home * home_model_mu, 1e-6)
blended_mu_away = np.maximum((1.0 - blend_weight_away) * baseline_mu_away + blend_weight_away * away_model_mu, 1e-6)

# Tier A / Tier B / market conditional win probabilities for stacking.
tier_a_cond = np.clip(inference_data["baseline_home_win_prob_conditional"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
lineup_unc_home = pd.to_numeric(inference_data.get("lineup_selection_uncertainty_home", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
lineup_unc_away = pd.to_numeric(inference_data.get("lineup_selection_uncertainty_away", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
tier_b_cond = np.array(
    [
        pf.marginalized_conditional_home_win_prob(
            mh,
            ma,
            lineup_uncertainty_home=uh,
            lineup_uncertainty_away=ua,
            n_samples=lineup_mc_samples,
            mu_noise_scale=lineup_mu_noise_scale,
        )
        for mh, ma, uh, ua in zip(blended_mu_home, blended_mu_away, lineup_unc_home, lineup_unc_away)
    ],
    dtype=float,
)
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

# Log tips summary (primary) and edge summary (secondary).
try:
    model_prob = calibrated_cond
    edge = model_prob - market_cond
    edge_threshold = 0.05

    teams_home = inference_data["team_home"].to_numpy() if "team_home" in inference_data.columns else ["?"] * len(inference_data)
    teams_away = inference_data["team_away"].to_numpy() if "team_away" in inference_data.columns else ["?"] * len(inference_data)

    # ── PRIMARY: Tips ─────────────────────────────────────────────────────────
    print(f"\n── Tips ({len(inference_data)} game(s)) ──────────────────────────────────────")
    for th, ta, mp in zip(teams_home, teams_away, model_prob):
        tip = th if mp > 0.5 else ta
        tip_prob = mp if mp > 0.5 else 1.0 - mp
        confidence = "HIGH" if tip_prob >= 0.70 else ("MED" if tip_prob >= 0.55 else "LOW")
        print(f"  TIP [{confidence}] {th} vs {ta}: {tip} ({tip_prob:.1%})")

    # ── SECONDARY: Betting edge vs market ────────────────────────────────────
    value_home = (edge > edge_threshold).sum()
    value_away = (edge < -edge_threshold).sum()
    if value_home + value_away > 0:
        print(f"\n── Market edge (threshold ±{edge_threshold:.0%}) ────────────────────────────")
        for th, ta, mp, mkp, e in zip(teams_home, teams_away, model_prob, market_cond, edge):
            if abs(e) > edge_threshold:
                direction = "HOME" if e > 0 else "AWAY"
                print(f"  [{direction}] {th} vs {ta}: model={mp:.1%}, market={mkp:.1%}, edge={e:+.1%}")
except Exception:
    pass

print("Predictions saved to the database!")
