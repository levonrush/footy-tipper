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

from pipeline.common import console
from pipeline.common.model_prediciton import prediction_functions as pf
from pipeline.common.model_prediciton.market_score_blend import (
    apply_market_score_mean_blends,
)
from pipeline.common.lineups import features as lf
from pipeline.common.model_training import calibration as calib
from pipeline.common.model_training import tier_a_baseline as tb
from pipeline.common.model_training import training_config as tc
from pipeline.ops.odds_gate import current_round_odds_coverage
from pipeline.runtime_paths import database_path, models_path, project_root as configured_project_root


project_root = configured_project_root()
db_path = database_path(project_root)
models_dir = models_path(project_root)

manifest_path = models_dir / "model_manifest.json"
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
dispersion_home = manifest.get("dispersion_home")
dispersion_away = manifest.get("dispersion_away")
lineup_mc_samples = int(manifest.get("lineup_monte_carlo_samples", os.getenv("FOOTY_TIPPER_LINEUP_MONTE_CARLO_SAMPLES", "64")))
lineup_mu_noise_scale = float(manifest.get("lineup_mu_noise_scale", os.getenv("FOOTY_TIPPER_LINEUP_MU_NOISE_SCALE", "0.12")))

baseline_cfg_payload = manifest.get("tier_a_baseline", {})
if baseline_cfg_payload:
    baseline_cfg = tb.baseline_config_from_dict(baseline_cfg_payload)
else:
    baseline_cfg = tb.default_baseline_config_from_env()

home_model = pf.load_models("home_model", project_root, models_dir=models_dir)
away_model = pf.load_models("away_model", project_root, models_dir=models_dir)

stacker = None
calibrator = None
stacker_no_market = None
calibrator_no_market = None
binary_model = None
stacker_path = models_dir / "stacker.pkl"
calibrator_path = models_dir / "win_prob_calibrator.pkl"
stacker_no_market_path = models_dir / "stacker_no_market.pkl"
calibrator_no_market_path = models_dir / "win_prob_calibrator_no_market.pkl"
binary_model_path = models_dir / "binary_model.pkl"
if stacker_path.exists():
    stacker = calib.load_artifact(stacker_path)
if calibrator_path.exists():
    calibrator = calib.load_artifact(calibrator_path)
if stacker_no_market_path.exists():
    stacker_no_market = calib.load_artifact(stacker_no_market_path)
if calibrator_no_market_path.exists():
    calibrator_no_market = calib.load_artifact(calibrator_no_market_path)
if binary_model_path.exists():
    binary_model = calib.load_artifact(binary_model_path)

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
    import traceback
    traceback.print_exc()
    if os.getenv("FOOTY_TIPPER_LINEUP_FEATURES_STRICT", "").strip().lower() in {"1", "true", "yes", "y"}:
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

    inference_data = ctx.merge_match_context_features(inference_data, db_path)
except Exception as exc:
    print(f"Match-context feature merge skipped ({exc}).")

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
infer_game_ids = inference_data["game_id"].to_numpy()
tier_b_cond = np.array(
    [
        pf.marginalized_conditional_home_win_prob(
            mh,
            ma,
            lineup_uncertainty_home=uh,
            lineup_uncertainty_away=ua,
            n_samples=lineup_mc_samples,
            mu_noise_scale=lineup_mu_noise_scale,
            # Deterministic per-game RNG: re-running inference on the same
            # data must never flip a tip.
            rng=pf.rng_for_game(gid, salt=2),
        )
        for mh, ma, uh, ua, gid in zip(
            blended_mu_home, blended_mu_away, lineup_unc_home, lineup_unc_away, infer_game_ids
        )
    ],
    dtype=float,
)
market_cond = pf.derive_market_home_probability(inference_data)
odds_coverage = current_round_odds_coverage(db_path)
fresh_game_ids = (
    set(odds_coverage.fresh_game_ids) if not odds_coverage.error else set()
)
fresh_line_game_ids = (
    set(odds_coverage.fresh_line_game_ids) if not odds_coverage.error else set()
)
fresh_total_game_ids = (
    set(odds_coverage.fresh_total_game_ids) if not odds_coverage.error else set()
)
fresh_market = calib.fresh_game_mask(inference_data, fresh_game_ids)
fresh_line_market = calib.fresh_game_mask(inference_data, fresh_line_game_ids)
fresh_total_market = calib.fresh_game_mask(inference_data, fresh_total_game_ids)
valid_market = calib.valid_fresh_h2h_mask(inference_data, fresh_game_ids)
if int(valid_market.sum()) < len(inference_data):
    print(
        "WARNING: probability market regime is available for only "
        f"{int(valid_market.sum())}/{len(inference_data)} game(s) with fresh "
        "paired H2H snapshots; remaining games are model-only."
    )

tier_c_cond = None
if binary_model is not None:
    try:
        tier_c_cond = np.clip(
            binary_model.predict_proba(inference_data[predictors])[:, 1], 1e-6, 1 - 1e-6
        )
    except Exception as exc:
        print(f"Binary model prediction skipped ({exc}).")

probability_stack = manifest.get("probability_stack")
if not isinstance(probability_stack, dict):
    probability_stack = {}
probability_stack_version = int(probability_stack.get("schema_version", 0) or 0)
no_market_config = probability_stack.get("no_market")
if not isinstance(no_market_config, dict):
    no_market_config = {}
no_market_strategy = str(no_market_config.get("strategy", "tier_b"))

# The line-market feature layout is needed only by legacy logistic stackers.
# New simplex pools consume genuine H2H probability as their sole market
# expert and route missing rows to an independent model-only regime.
line_extra = None
if (
    probability_stack_version < calib.PROBABILITY_STACK_VERSION
    and valid_market.any()
):
    legacy_market_frame = inference_data.copy()
    stale_line = ~fresh_line_market
    stale_total = ~fresh_total_market
    for column in (
        "home_line_cover_prob_shin",
        "home_line_cover_prob_power",
        "home_line_cover_prob_basic",
        "line_overround_basic",
        "implied_spread_home",
        "line_move_points",
    ):
        if column in legacy_market_frame.columns:
            legacy_market_frame.loc[stale_line, column] = np.nan
    if "line_odds_missing" in legacy_market_frame.columns:
        legacy_market_frame.loc[stale_line, "line_odds_missing"] = 1.0
    for column in ("market_total_line", "total_line"):
        if column in legacy_market_frame.columns:
            legacy_market_frame.loc[stale_total, column] = np.nan
    if "totals_missing" in legacy_market_frame.columns:
        legacy_market_frame.loc[stale_total, "totals_missing"] = 1.0
    line_extra = calib.build_line_market_features(
        legacy_market_frame, blended_mu_home - blended_mu_away
    )
    manifest_extra_version = manifest.get("market_extra_version")
    if (
        manifest_extra_version is not None
        and int(manifest_extra_version) != calib.MARKET_EXTRA_VERSION
    ):
        print(
            "WARNING: stacker market-extra layout mismatch "
            f"(manifest v{manifest_extra_version} vs code v{calib.MARKET_EXTRA_VERSION}). "
            "Retrain before trusting stacked probabilities."
        )

calibrated_cond, probability_routes = calib.predict_probability_regimes(
    tier_a=tier_a_cond,
    tier_b=tier_b_cond,
    tier_c=tier_c_cond,
    market=market_cond,
    valid_market=valid_market,
    market_stacker=stacker,
    market_calibrator=calibrator,
    no_market_stacker=stacker_no_market,
    no_market_calibrator=calibrator_no_market,
    no_market_strategy=no_market_strategy,
    legacy_extra=line_extra,
)
print(
    "Probability routes: "
    f"market={probability_routes['market']}, "
    f"no-market-pool={probability_routes['no_market_pool']}, "
    f"Tier-B-fallback={probability_routes['tier_b']}."
)
if probability_routes["consensus_guarded"]:
    print(
        "WARNING: consensus guard replaced "
        f"{probability_routes['consensus_guarded']} side-reversing ensemble "
        "prediction(s) with Tier B."
    )

margin_blend = manifest.get("margin_blend")
total_blend = manifest.get("total_blend")
try:
    blended_mu_home, blended_mu_away, score_market_diagnostics = (
        apply_market_score_mean_blends(
            inference_data,
            blended_mu_home,
            blended_mu_away,
            baseline_mu_home,
            baseline_mu_away,
            fresh_market=fresh_market,
            fresh_line_market=fresh_line_market,
            fresh_total_market=fresh_total_market,
            margin_blend=margin_blend,
            total_blend=total_blend,
        )
    )
    print(
        "Score-mean market blends: "
        f"line={score_market_diagnostics['line_count']}/{len(inference_data)}, "
        f"total={score_market_diagnostics['total_count']}/{len(inference_data)}."
    )
except Exception as exc:
    print(f"Score-mean market blends skipped ({exc}).")

# Both chosen on the held-out margin scorecard, not by preference: see
# `margin_distribution.reconciliation` in reports/eval-latest.json, which scores
# every combination of these two switches on the same per-game seeds.
outcomes, margins = pf.predict_match_outcome_and_scoreline_with_bayes(
    inference_data=inference_data,
    mu_home=blended_mu_home,
    mu_away=blended_mu_away,
    lambda3=lambda3,
    calibrated_home_win_conditional=calibrated_cond,
    dispersion_home=dispersion_home,
    dispersion_away=dispersion_away,
    reconcile="on_conflict",
    display="median",
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
    # Missing odds are a model-only prediction regime, not a synthetic 50%
    # market. Never report a betting edge for those rows.
    edge = np.where(valid_market, model_prob - market_cond, np.nan)
    edge_threshold = 0.05

    teams_home = inference_data["team_home"].to_numpy() if "team_home" in inference_data.columns else ["?"] * len(inference_data)
    teams_away = inference_data["team_away"].to_numpy() if "team_away" in inference_data.columns else ["?"] * len(inference_data)

    # ── PRIMARY: Tips ─────────────────────────────────────────────────────────
    print(f"\n── Tips ({len(inference_data)} game(s)) ──────────────────────────────────────")
    tip_records = []
    for th, ta, mp in zip(teams_home, teams_away, model_prob):
        tip = th if mp > 0.5 else ta
        tip_prob = mp if mp > 0.5 else 1.0 - mp
        confidence = "HIGH" if tip_prob >= 0.70 else ("MED" if tip_prob >= 0.55 else "LOW")
        print(f"  TIP [{confidence}] {th} vs {ta}: {tip} ({tip_prob:.1%})")
        tip_records.append(
            {"home": str(th), "away": str(ta), "tip": str(tip), "prob": float(tip_prob)}
        )
    if tip_records:
        console.emit_result("tips", games=tip_records)

    # ── SECONDARY: Betting edge vs market ────────────────────────────────────
    value_home = int(np.nansum(edge > edge_threshold))
    value_away = int(np.nansum(edge < -edge_threshold))
    if value_home + value_away > 0:
        print(f"\n── Market edge (threshold ±{edge_threshold:.0%}) ────────────────────────────")
        for th, ta, mp, mkp, e, has_market in zip(
            teams_home,
            teams_away,
            model_prob,
            market_cond,
            edge,
            valid_market,
        ):
            if has_market and abs(e) > edge_threshold:
                direction = "HOME" if e > 0 else "AWAY"
                print(f"  [{direction}] {th} vs {ta}: model={mp:.1%}, market={mkp:.1%}, edge={e:+.1%}")
except Exception:
    pass

print("Predictions saved to the database!")
