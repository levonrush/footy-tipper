# Description: Honest nested season-out evaluation of the full prediction stack.
#
# train.py's printed metrics are slightly optimistic: the blend weights,
# stacker, and calibrator are fitted on OOF rows that include the most recent
# season. This script evaluates each held-out season Y by fitting the entire
# meta-layer (blend weights -> stacker -> calibrator) only on seasons < Y.
# The Tier-B/Tier-C inputs come from the expanding-window OOF generators, so
# predictions for season Y only ever use models trained on seasons < Y.
print("Running the evaluate.py script...")

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
from pipeline.common.model_training import modelling_functions as mf
from pipeline.common.model_training import tier_a_baseline as tb
from pipeline.common.model_training import training_config as tc


def _load_training_frame(project_root, db_path):
    """Load and feature-merge the training frame the same way train.py does."""
    predictors = tc.filter_predictors(include_performance=tc.include_performance, predictor_list=tc.predictors)
    data = mf.get_training_data(
        db_path=db_path,
        sql_file=project_root / "pipeline/common/sql/training_data.sql",
    )
    if data.empty:
        raise RuntimeError("Training data is empty. Run data prep first.")

    baseline_cfg = tb.default_baseline_config_from_env()
    baseline_features = tb.compute_tier_a_baseline_features(data, baseline_cfg)
    data = data.merge(baseline_features, on="game_id", how="left")

    base_home = float(data["team_final_score_home"].mean())
    base_away = float(data["team_final_score_away"].mean())
    data["baseline_mu_home"] = pd.to_numeric(data["baseline_mu_home"], errors="coerce").fillna(base_home)
    data["baseline_mu_away"] = pd.to_numeric(data["baseline_mu_away"], errors="coerce").fillna(base_away)
    data["baseline_home_win_prob_conditional"] = (
        pd.to_numeric(data["baseline_home_win_prob_conditional"], errors="coerce").fillna(0.5)
    )

    try:
        years = sorted(
            pd.to_numeric(data["competition_year"], errors="coerce").dropna().astype(int).unique().tolist()
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
    odds_missing,
    home_odds,
    away_odds,
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
    tier_b_cond = pf.conditional_home_win_prob_vec(blended_h, blended_a)

    # Stacker + calibrator fitted on prior seasons only.
    stacker = calib.LogisticStacker()
    stacker.fit(
        tier_a=tier_a_cond[prior_mask],
        tier_b=tier_b_cond[prior_mask],
        market=market_cond[prior_mask],
        odds_missing=odds_missing[prior_mask],
        tier_c=tier_c_cond_oof[prior_mask],
        y=y_full[prior_mask],
        groups=year_col[prior_mask],
    )
    stacked_prior = stacker.predict(
        tier_a_cond[prior_mask],
        tier_b_cond[prior_mask],
        market_cond[prior_mask],
        odds_missing[prior_mask],
        tier_c=tier_c_cond_oof[prior_mask],
    )
    calibrator = calib.BetaCalibrator()
    calibrator.fit(stacked_prior, y_full[prior_mask])

    stacked_test = stacker.predict(
        tier_a_cond[test_mask],
        tier_b_cond[test_mask],
        market_cond[test_mask],
        odds_missing[test_mask],
        tier_c=tier_c_cond_oof[test_mask],
    )
    model_p = np.clip(calibrator.predict(stacked_test), 1e-6, 1 - 1e-6)
    y_test = y_full[test_mask]
    market_p = market_cond[test_mask]
    valid_market = ~odds_missing[test_mask].astype(bool)

    result = {
        "year": int(test_year),
        "games": int(test_mask.sum()),
        "correct": int(((model_p > 0.5) == y_test.astype(bool)).sum()),
        "log_loss": float(log_loss(y_test, model_p)),
        "brier": float(brier_score_loss(y_test, model_p)),
        "blend_wh": float(wh),
        "blend_wa": float(wa),
        "market_games": int(valid_market.sum()),
        "market_correct": int(
            ((market_p[valid_market] > 0.5) == y_test[valid_market].astype(bool)).sum()
        ),
        "model_p": model_p,
        "y_test": y_test,
    }

    # Flat-stake ROI at the edge threshold.
    edge = model_p - market_p
    oh = home_odds[test_mask]
    oa = away_odds[test_mask]
    total_bets, wins, profit = 0, 0, 0.0
    for e, act, odds_h, odds_a in zip(edge, y_test, oh, oa):
        if e > edge_threshold and np.isfinite(odds_h):
            total_bets += 1
            profit += (odds_h - 1.0) if act == 1 else -1.0
            wins += int(act == 1)
        elif e < -edge_threshold and np.isfinite(odds_a):
            total_bets += 1
            profit += (odds_a - 1.0) if act == 0 else -1.0
            wins += int(act == 0)
    result.update({"bets": total_bets, "bet_wins": wins, "profit": float(profit)})
    return result


def main():
    project_root = pathlib.Path().absolute()
    db_path = project_root / "data" / "footy-tipper-db.sqlite"
    n_seasons = int(os.getenv("FOOTY_TIPPER_EVAL_SEASONS", "3"))

    home_model = pf.load_models("home_model", project_root)
    away_model = pf.load_models("away_model", project_root)

    data, predictors = _load_training_frame(project_root, db_path)
    data = tc.align_predictor_columns(data, predictors)
    selected = tc.prune_sparse_predictors(data, predictors)
    data = tc.align_predictor_columns(data, selected)

    year_col = pd.to_numeric(data["competition_year"], errors="coerce").to_numpy()
    non_draw = (
        data["team_final_score_home"].to_numpy(dtype=float)
        != data["team_final_score_away"].to_numpy(dtype=float)
    )
    y_full = (
        data["team_final_score_home"].to_numpy(dtype=float)
        > data["team_final_score_away"].to_numpy(dtype=float)
    ).astype(int)

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
        pd.to_numeric(data["baseline_home_win_prob_conditional"], errors="coerce").fillna(0.5).to_numpy(dtype=float),
        1e-6,
        1 - 1e-6,
    )
    market_cond = pf.derive_market_home_probability(data)
    if "odds_missing" in data.columns:
        odds_missing = pd.to_numeric(data["odds_missing"], errors="coerce").fillna(0).to_numpy(dtype=float)
    else:
        odds_missing = np.zeros(len(data), dtype=float)

    home_odds = pd.to_numeric(data.get("team_head_to_head_odds_home", np.nan), errors="coerce").to_numpy(dtype=float)
    away_odds = pd.to_numeric(data.get("team_head_to_head_odds_away", np.nan), errors="coerce").to_numpy(dtype=float)

    baseline_mu_home = data["baseline_mu_home"].to_numpy(dtype=float)
    baseline_mu_away = data["baseline_mu_away"].to_numpy(dtype=float)

    eval_years = sorted({int(y) for y in year_col[genuine_oof & ~np.isnan(year_col)]})[-n_seasons:]
    print(f"Evaluating held-out seasons: {eval_years}")

    results = []
    for test_year in eval_years:
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
            odds_missing,
            home_odds,
            away_odds,
        )
        if res is None:
            print(f"  {test_year}: skipped (not enough prior or test rows).")
            continue
        results.append(res)

    if not results:
        print("No seasons could be evaluated. Train on more seasons first.")
        return 1

    print("\n── Honest nested evaluation (meta-layer never sees the test season) ──")
    print(f"  {'Season':<8} {'Tips':>9} {'Acc':>7} {'Mkt acc':>8} {'LogLoss':>8} {'Brier':>7} {'ROI':>8}")
    for res in results:
        acc = res["correct"] / res["games"]
        mkt = (res["market_correct"] / res["market_games"]) if res["market_games"] else float("nan")
        roi = (100.0 * res["profit"] / res["bets"]) if res["bets"] else float("nan")
        roi_text = f"{roi:+.1f}%" if res["bets"] else "n/a"
        print(
            f"  {res['year']:<8} {res['correct']:>4}/{res['games']:<4} {acc:>7.1%} {mkt:>8.1%} "
            f"{res['log_loss']:>8.4f} {res['brier']:>7.4f} {roi_text:>8}"
        )

    pooled_p = np.concatenate([res["model_p"] for res in results])
    pooled_y = np.concatenate([res["y_test"] for res in results])
    pooled_games = int(sum(res["games"] for res in results))
    pooled_correct = int(sum(res["correct"] for res in results))
    pooled_mkt_games = int(sum(res["market_games"] for res in results))
    pooled_mkt_correct = int(sum(res["market_correct"] for res in results))
    pooled_bets = int(sum(res["bets"] for res in results))
    pooled_profit = float(sum(res["profit"] for res in results))

    print("\n── Pooled across held-out seasons ──")
    print(f"  Tipping accuracy: {pooled_correct}/{pooled_games} ({pooled_correct / pooled_games:.1%})")
    if pooled_mkt_games:
        print(f"  Market favourite: {pooled_mkt_correct}/{pooled_mkt_games} ({pooled_mkt_correct / pooled_mkt_games:.1%})")
    print(f"  Log-loss: {log_loss(pooled_y, pooled_p):.4f}   Brier: {brier_score_loss(pooled_y, pooled_p):.4f}")
    if pooled_bets:
        print(f"  Edge bets: {pooled_bets}, flat-stake ROI: {100.0 * pooled_profit / pooled_bets:+.1f}%")
    print("\nEvaluation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
