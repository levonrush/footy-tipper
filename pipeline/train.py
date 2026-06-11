# Description: This script trains score models and saves artefacts used at inference.
print("Running the train.py script...")

import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from pipeline.common.model_prediciton import prediction_functions as pf
from pipeline.common.lineups import features as lf
from pipeline.common.model_training import calibration as calib
from pipeline.common.model_training import joker_policy as jp
from pipeline.common.model_training import modelling_functions as mf
from pipeline.common.model_training import tier_a_baseline as tb
from pipeline.common.model_training import training_config as tc


def _select_blend_weight_by_log_loss(y_binary, non_draw, baseline_mu_home, baseline_mu_away, oof_mu_home, oof_mu_away):
    """Joint grid search over (w_home, w_away) minimising OOF log-loss.

    Uses OOF score predictions so the criterion is unbiased — in-sample would
    almost always select w=1.0 because the fitted LightGBM memorises training
    scores. Log-loss is a smooth proper scoring rule, so the argmin is far more
    stable than maximising 0/1 tipping accuracy (which is flat almost
    everywhere and jumps at thresholds). Accuracy at the chosen weights is
    returned for reporting.
    """
    candidates = np.linspace(0.0, 1.0, 11)
    best_wh, best_wa, best_ll = 1.0, 1.0, np.inf

    bh = np.asarray(baseline_mu_home, dtype=float)[non_draw]
    ba = np.asarray(baseline_mu_away, dtype=float)[non_draw]
    oh = np.asarray(oof_mu_home, dtype=float)[non_draw]
    oa = np.asarray(oof_mu_away, dtype=float)[non_draw]
    y = np.asarray(y_binary, dtype=int)

    for wh in candidates:
        blended_h = np.maximum((1.0 - wh) * bh + wh * oh, 1e-6)
        for wa in candidates:
            blended_a = np.maximum((1.0 - wa) * ba + wa * oa, 1e-6)
            win_probs = np.clip(pf.conditional_home_win_prob_vec(blended_h, blended_a), 1e-6, 1 - 1e-6)
            ll = log_loss(y, win_probs)
            if ll < best_ll:
                best_ll, best_wh, best_wa = float(ll), float(wh), float(wa)

    best_h = np.maximum((1.0 - best_wh) * bh + best_wh * oh, 1e-6)
    best_a = np.maximum((1.0 - best_wa) * ba + best_wa * oa, 1e-6)
    best_probs = pf.conditional_home_win_prob_vec(best_h, best_a)
    best_acc = float(((best_probs > 0.5) == y.astype(bool)).mean())

    return best_wh, best_wa, best_ll, best_acc


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

print("Merging lineup-derived features")
try:
    training_years = sorted(
        pd.to_numeric(training_data["competition_year"], errors="coerce").dropna().astype(int).unique().tolist()
    )
    lineup_entries = lf.load_lineup_entries(db_path, years=training_years)
    lineup_features = lf.build_lineup_match_features(training_data, lineup_entries)
    training_data = training_data.merge(lineup_features, on="game_id", how="left")

    for col in lf.LINEUP_FEATURE_COLUMNS:
        if col == "game_id":
            continue
        if col in {"lineup_home_players", "lineup_away_players"}:
            training_data[col] = training_data[col].fillna("")
        else:
            training_data[col] = pd.to_numeric(training_data[col], errors="coerce").fillna(0.0)

    lineup_coverage = 0.0
    if "lineup_features_missing" in training_data.columns and len(training_data) > 0:
        lineup_coverage = float((training_data["lineup_features_missing"] <= 0).mean())
    print(f"Lineup features merged. Coverage={lineup_coverage:.1%}")
except Exception as exc:
    print(f"Lineup feature merge skipped ({exc}).")

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

non_draw = _non_draw_mask(training_data)
y_full = (
    training_data["team_final_score_home"].to_numpy(dtype=float)
    > training_data["team_final_score_away"].to_numpy(dtype=float)
).astype(int)
y_binary = y_full[non_draw]

# Generate OOF score predictions BEFORE blend weight selection so weights are
# chosen on unbiased OOF tipping accuracy rather than in-sample Poisson deviance.
print("Generating OOF score predictions for blend weight selection and stacker training...")
home_model_mu_oof, home_oof_mask = mf.generate_oof_score_predictions(
    training_data, selected_predictors, home_model, "team_final_score_home", return_mask=True
)
away_model_mu_oof, away_oof_mask = mf.generate_oof_score_predictions(
    training_data, selected_predictors, away_model, "team_final_score_away", return_mask=True
)

home_weight, away_weight, blend_ll, blend_acc = _select_blend_weight_by_log_loss(
    y_binary, non_draw, baseline_mu_home, baseline_mu_away,
    home_model_mu_oof, away_model_mu_oof,
)

blended_mu_home = np.maximum((1.0 - home_weight) * baseline_mu_home + home_weight * home_model_mu, 1e-6)
blended_mu_away = np.maximum((1.0 - away_weight) * baseline_mu_away + away_weight * away_model_mu, 1e-6)

lambda3 = _estimate_lambda3(
    training_data["team_final_score_home"].to_numpy(dtype=float),
    training_data["team_final_score_away"].to_numpy(dtype=float),
    blended_mu_home,
    blended_mu_away,
)

print(
    f"Selected blend weights: home={home_weight:.2f}, away={away_weight:.2f} "
    f"(OOF log-loss={blend_ll:.4f}, tipping accuracy at chosen weights={blend_acc:.1%})"
)
print(f"Estimated bivariate shared component lambda3={lambda3:.4f}")

print("Fitting stacking model and beta calibrator")
lineup_mc_samples = int(os.getenv("FOOTY_TIPPER_LINEUP_MONTE_CARLO_SAMPLES", "64"))
lineup_mu_noise_scale = float(os.getenv("FOOTY_TIPPER_LINEUP_MU_NOISE_SCALE", "0.12"))

lineup_unc_home = pd.to_numeric(training_data.get("lineup_selection_uncertainty_home", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
lineup_unc_away = pd.to_numeric(training_data.get("lineup_selection_uncertainty_away", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)

tier_a_cond = np.clip(training_data["baseline_home_win_prob_conditional"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)

# In-sample Tier B predictions (used for inference at prediction time).
train_game_ids = training_data["game_id"].to_numpy()
tier_b_cond = np.array(
    [
        pf.marginalized_conditional_home_win_prob(
            mh,
            ma,
            lineup_uncertainty_home=uh,
            lineup_uncertainty_away=ua,
            n_samples=lineup_mc_samples,
            mu_noise_scale=lineup_mu_noise_scale,
            rng=pf.rng_for_game(gid, salt=2),
        )
        for mh, ma, uh, ua, gid in zip(
            blended_mu_home, blended_mu_away, lineup_unc_home, lineup_unc_away, train_game_ids
        )
    ],
    dtype=float,
)
market_cond = pf.derive_market_home_probability(training_data)
if "odds_missing" in training_data.columns:
    odds_missing = pd.to_numeric(training_data["odds_missing"], errors="coerce").fillna(0).to_numpy(dtype=float)
else:
    odds_missing = np.zeros(len(training_data), dtype=float)

# OOF blended mus for stacker training (use weights selected above).
blended_mu_home_oof = np.maximum(
    (1.0 - home_weight) * baseline_mu_home + home_weight * home_model_mu_oof, 1e-6
)
blended_mu_away_oof = np.maximum(
    (1.0 - away_weight) * baseline_mu_away + away_weight * away_model_mu_oof, 1e-6
)
tier_b_cond_oof = np.array(
    [pf.conditional_home_win_prob(mh, ma) for mh, ma in zip(blended_mu_home_oof, blended_mu_away_oof)],
    dtype=float,
)

# ── Tier-C: binary LightGBM (OOF) ────────────────────────────────────────────
# Trains a direct binary win/loss classifier using the same hyperparameters as
# the Poisson models. OOF predictions are used for stacker training to avoid bias.
print("Generating OOF binary predictions for stacker training...")
best_params = dict(home_model.named_steps["hyperparamtuning"].best_params_)
preprocessor_steps = home_model[:-1]

binary_model_oof, binary_oof_mask = mf.generate_oof_binary_predictions(
    training_data, non_draw, selected_predictors, preprocessor_steps, best_params, return_mask=True
)
tier_c_cond_oof = np.clip(binary_model_oof, 1e-6, 1 - 1e-6)

print("Training final binary classifier...")
training_data["_y_binary_col"] = (
    training_data["team_final_score_home"].to_numpy(dtype=float)
    > training_data["team_final_score_away"].to_numpy(dtype=float)
).astype(int)
binary_model = mf.train_binary_classifier(
    training_data[non_draw], selected_predictors, "_y_binary_col",
    best_params, preprocessor_steps,
)
training_data.drop(columns=["_y_binary_col"], inplace=True)

# ── Stacker (trained on OOF Tier-B + OOF Tier-C) ─────────────────────────────
# Restrict meta-model training to rows whose tier inputs are genuinely
# out-of-fold; first-season rows carry in-sample fallbacks that would bias
# the stacker towards the overfit tiers.
genuine_oof = home_oof_mask & away_oof_mask & binary_oof_mask
stacker_fit_mask = non_draw & genuine_oof
if stacker_fit_mask.sum() < 50:
    print(
        f"Only {int(stacker_fit_mask.sum())} genuine-OOF rows available; "
        "falling back to all non-draw rows for meta-model training."
    )
    stacker_fit_mask = non_draw

comp_years_all = pd.to_numeric(training_data["competition_year"], errors="coerce").to_numpy()

stacker = calib.LogisticStacker()
stacker.fit(
    tier_a=tier_a_cond[stacker_fit_mask],
    tier_b=tier_b_cond_oof[stacker_fit_mask],
    market=market_cond[stacker_fit_mask],
    odds_missing=odds_missing[stacker_fit_mask],
    tier_c=tier_c_cond_oof[stacker_fit_mask],
    y=y_full[stacker_fit_mask],
    groups=comp_years_all[stacker_fit_mask],
)

# Log selected regularisation strength and coefficients.
if stacker._is_fitted and hasattr(stacker._model, "coef_"):
    if hasattr(stacker._model, "C_"):
        print(f"Stacker selected C={stacker._model.C_[0]:.4f} (cross-validated from {calib.LogisticStacker._DEFAULT_CS})")
    coef_names = ["tier_a", "tier_b", "market", "odds_missing", "disagree_tier_a", "disagree_tier_b", "tier_c"]
    coef_vals = stacker._model.coef_[0]
    coef_str = ", ".join(f"{n}={v:.3f}" for n, v in zip(coef_names, coef_vals))
    print(f"Stacker coefficients: {coef_str}")

# ── Calibrator (trained on OOF stacked predictions) ───────────────────────────
stacked_cond_oof = stacker.predict(tier_a_cond, tier_b_cond_oof, market_cond, odds_missing, tier_c=tier_c_cond_oof)
calibrator = calib.BetaCalibrator()
calibrator.fit(stacked_cond_oof[stacker_fit_mask], y_full[stacker_fit_mask])
calibrated_oof = calibrator.predict(stacked_cond_oof)

# ── Evaluation metrics ────────────────────────────────────────────────────────
try:
    # Evaluate only on genuine-OOF rows; first-season fallback rows are
    # in-sample and would flatter every number below.
    eval_mask = stacker_fit_mask
    y_eval = y_full[eval_mask]
    nd_preds = np.clip(calibrated_oof[eval_mask], 1e-6, 1 - 1e-6)
    market_nd = market_cond[eval_mask]
    # Use odds_missing flag to identify games with real odds (not 0.5 fallback).
    # market_cond clips everything to (0,1) so range checks can't detect missing odds.
    odds_missing_nd = odds_missing[eval_mask].astype(bool)
    valid_market = ~odds_missing_nd

    # ── PRIMARY: Tipping accuracy ─────────────────────────────────────────────
    tip_correct = (nd_preds > 0.5) == y_eval.astype(bool)
    tip_acc = tip_correct.mean()
    naive_home_acc = float(y_eval.mean())  # always-pick-home baseline

    print(f"\n── Tipping accuracy (genuine OOF, non-draw) ────────────────────────────")
    print(f"  Model:       {tip_acc:.1%}  ({tip_correct.sum()}/{len(tip_correct)} correct)")
    print(f"  Always home: {naive_home_acc:.1%}")

    if valid_market.sum() >= 10:
        market_tip = (market_nd[valid_market] > 0.5) == y_eval[valid_market].astype(bool)
        model_tip_on_mkt = (nd_preds[valid_market] > 0.5) == y_eval[valid_market].astype(bool)
        diff = model_tip_on_mkt.mean() - market_tip.mean()
        print(f"  Market fav:  {market_tip.mean():.1%}  (on {valid_market.sum()} games with odds)")
        print(f"  Model (same games): {model_tip_on_mkt.mean():.1%}  ({'▲' if diff > 0 else '▼'} {abs(diff):.1%} vs market)")

    # ── SECONDARY: Probabilistic calibration ──────────────────────────────────
    nd_log_loss = log_loss(y_eval, nd_preds)
    nd_brier = brier_score_loss(y_eval, nd_preds)
    print(f"\n── Calibration (genuine OOF, non-draw) ─────────────────────────────────")
    print(f"  Log-loss  (model):   {nd_log_loss:.4f}")
    print(f"  Brier     (model):   {nd_brier:.4f}")

    if valid_market.sum() >= 10:
        market_ll = log_loss(y_eval[valid_market], np.clip(market_nd[valid_market], 1e-6, 1 - 1e-6))
        market_br = brier_score_loss(y_eval[valid_market], np.clip(market_nd[valid_market], 1e-6, 1 - 1e-6))
        model_ll  = log_loss(y_eval[valid_market], np.clip(nd_preds[valid_market], 1e-6, 1 - 1e-6))
        model_br  = brier_score_loss(y_eval[valid_market], np.clip(nd_preds[valid_market], 1e-6, 1 - 1e-6))
        print(f"  Log-loss  (market benchmark): {market_ll:.4f}  |  model: {model_ll:.4f}  ({'▲ better' if model_ll < market_ll else '▼ worse'})")
        print(f"  Brier     (market benchmark): {market_br:.4f}  |  model: {model_br:.4f}  ({'▲ better' if model_br < market_br else '▼ worse'})")

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
        print(f"  {bins[i]:.1f}–{bins[i+1]:.1f}         {pred_mean:>10.3f} {actual_mean:>8.3f} {mask.sum():>7}{flag}")

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

        odds_missing_holdout = odds_missing[nd_holdout].astype(bool)
        valid_mkt_holdout = ~odds_missing_holdout
        print(f"\n── Tipping accuracy ({last_year} holdout, non-draw) ─────────────────")
        print(f"  Model:       {tip_acc_holdout:.1%}  ({tip_correct_holdout.sum()}/{len(tip_correct_holdout)} correct)")
        if valid_mkt_holdout.sum() >= 3:
            mkt_tip_holdout = (market_p[valid_mkt_holdout] > 0.5) == actuals[valid_mkt_holdout].astype(bool)
            print(f"  Market fav:  {mkt_tip_holdout.mean():.1%}  (on {valid_mkt_holdout.sum()} games with odds)")

        # ── SECONDARY: ROI betting simulation ─────────────────────────────────
        home_odds_col = training_data["team_head_to_head_odds_home"] if "team_head_to_head_odds_home" in training_data.columns else pd.Series(np.nan, index=training_data.index)
        away_odds_col = training_data["team_head_to_head_odds_away"] if "team_head_to_head_odds_away" in training_data.columns else pd.Series(np.nan, index=training_data.index)
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

        print(f"\n── ROI simulation ({last_year} holdout, ≥{threshold:.0%} edge) ──────────────")
        print(f"  Games in holdout: {nd_holdout.sum()},  edge bets placed: {total_bets}")
        if total_bets > 0:
            roi_pct = 100.0 * profit / total_bets
            win_rate = 100.0 * wins / total_bets
            print(f"  Wins: {wins}/{total_bets} ({win_rate:.1f}%),  flat-stake ROI: {roi_pct:+.1f}%")
        else:
            print("  No games exceeded the edge threshold.")
        print(
            "  NOTE: blend weights, stacker, and calibrator were fitted using OOF rows "
            "from this season too — run `footy-tipper evaluate` for fully held-out numbers."
        )
except Exception as exc:
    import traceback
    traceback.print_exc()
    print(f"Holdout evaluation skipped ({exc}).")

print("Save model artefacts")
mf.save_models(home_model, "home_model", project_root)
mf.save_models(away_model, "away_model", project_root)
mf.save_models(binary_model, "binary_model", project_root)
calib.save_artifact(stacker, project_root / "models" / "stacker.pkl")
calib.save_artifact(calibrator, project_root / "models" / "win_prob_calibrator.pkl")

manifest = {
    "predictors": selected_predictors,
    "blend_weight_home": home_weight,
    "blend_weight_away": away_weight,
    "lambda3": lambda3,
    "lineup_monte_carlo_samples": lineup_mc_samples,
    "lineup_mu_noise_scale": lineup_mu_noise_scale,
    "tier_a_baseline": tb.baseline_config_to_dict(baseline_cfg, base_home, base_away),
}

print("Running joker strategy backtest")
try:
    joker_policy_path = project_root / "models" / "joker_policy.json"
    joker_policy = jp.save_joker_policy(training_data, joker_policy_path)
    manifest["joker_policy_file"] = str(joker_policy_path.name)
    manifest["joker_policy_default_strategy"] = joker_policy.get("default_strategy", "points")
    manifest["joker_policy_seasons_evaluated"] = int(joker_policy.get("seasons_evaluated", 0))
    print(
        "Joker policy saved "
        f"(default={manifest['joker_policy_default_strategy']}, "
        f"seasons={manifest['joker_policy_seasons_evaluated']})"
    )
except Exception as exc:
    print(f"Joker policy backtest skipped ({exc}).")

manifest_path = project_root / "models" / "model_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Saved manifest to {manifest_path}")

print("Model training complete!")
