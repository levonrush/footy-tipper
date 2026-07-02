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
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import brier_score_loss, log_loss

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from pipeline.common.model_prediciton import prediction_functions as pf
from pipeline.common.lineups import features as lf
from pipeline.common.model_training import calibration as calib
from pipeline.common.model_training import comp_sim
from pipeline.common.model_training import modelling_functions as mf
from pipeline.common.model_training import tier_a_baseline as tb
from pipeline.common.model_training import training_config as tc


def _load_training_frame(project_root, db_path, baseline_cfg=None):
    """Load and feature-merge the training frame the same way train.py does."""
    predictors = tc.filter_predictors(include_performance=tc.include_performance, predictor_list=tc.predictors)
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
    actual_margin,
    market_spread,
    line_frame,
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
    line_extra = calib.build_line_market_features(line_frame, blended_h - blended_a)

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
        extra=line_extra[prior_mask],
    )
    calibrator = calib.BetaCalibrator()
    # Mirror train.py: fit the calibrator on leave-one-season-out stacker
    # predictions within the prior seasons, falling back to in-sample.
    loso_prior = calib.loso_stacker_predictions(
        tier_a=tier_a_cond[prior_mask],
        tier_b=tier_b_cond[prior_mask],
        market=market_cond[prior_mask],
        odds_missing=odds_missing[prior_mask],
        y=y_full[prior_mask],
        groups=year_col[prior_mask],
        tier_c=tier_c_cond_oof[prior_mask],
        extra=line_extra[prior_mask],
    )
    if loso_prior is not None:
        loso_rows = np.isfinite(loso_prior)
        calibrator.fit(loso_prior[loso_rows], y_full[prior_mask][loso_rows])
    else:
        stacked_prior = stacker.predict(
            tier_a_cond[prior_mask],
            tier_b_cond[prior_mask],
            market_cond[prior_mask],
            odds_missing[prior_mask],
            tier_c=tier_c_cond_oof[prior_mask],
            extra=line_extra[prior_mask],
        )
        calibrator.fit(stacked_prior, y_full[prior_mask])

    stacked_test = stacker.predict(
        tier_a_cond[test_mask],
        tier_b_cond[test_mask],
        market_cond[test_mask],
        odds_missing[test_mask],
        tier_c=tier_c_cond_oof[test_mask],
        extra=line_extra[test_mask],
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
        "market_p": market_p,
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
        float(np.mean(np.abs(spread[has_line] - margin_actual[has_line]))) if has_line.any() else None
    )

    # Season-out gate for the ridge margin blend (mirrors train.py's fit):
    # fit on prior seasons with a line, score on the whole test season with
    # model-margin fallback where the line is missing.
    result["margin_blend_mae"] = None
    blend_fit_mask = prior_mask & np.isfinite(market_spread)
    if blend_fit_mask.sum() >= 100:
        model_margin_full = blended_h - blended_a
        tier_a_margin_full = baseline_mu_home - baseline_mu_away
        X_margin = np.column_stack([model_margin_full, market_spread, tier_a_margin_full])
        margin_model = Ridge(alpha=1.0)
        margin_model.fit(X_margin[blend_fit_mask], actual_margin[blend_fit_mask])
        blend_pred = np.where(
            np.isfinite(market_spread),
            margin_model.predict(np.nan_to_num(X_margin, nan=0.0)),
            model_margin_full,
        )
        result["margin_blend_mae"] = float(np.mean(np.abs(blend_pred[test_mask] - margin_actual)))

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


def _git_sha(project_root):
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root, capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _manifest_fingerprint(project_root):
    try:
        with open(project_root / "models" / "model_manifest.json") as fh:
            manifest = json.load(fh)
        return {
            "blend_weight_home": manifest.get("blend_weight_home"),
            "blend_weight_away": manifest.get("blend_weight_away"),
            "lambda3": manifest.get("lambda3"),
            "tier_a_baseline": manifest.get("tier_a_baseline"),
            "predictor_count": len(manifest.get("predictors") or []),
        }
    except Exception:
        return None


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
            paths = [reports_dir / f"eval-{stamp}.json", reports_dir / "eval-latest.json"]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as fh:
                json.dump(report, fh, indent=2, default=float)
        return paths[-1]
    except Exception as exc:
        print(f"Eval report not written ({exc}).")
        return None


def main():
    project_root = pathlib.Path().absolute()
    db_path = project_root / "data" / "footy-tipper-db.sqlite"
    n_seasons = int(os.getenv("FOOTY_TIPPER_EVAL_SEASONS", "3"))

    home_model = pf.load_models("home_model", project_root)
    away_model = pf.load_models("away_model", project_root)

    # Optional honest Tier-A tuning: the grid only ever sees seasons strictly
    # before the earliest held-out season, so no test year informs the choice.
    baseline_cfg = None
    if os.getenv("FOOTY_TIPPER_TUNE_TIER_A", "true").strip().lower() not in {"0", "false", "no", "n", "off"}:
        raw = mf.get_training_data(
            db_path=db_path,
            sql_file=project_root / "pipeline/common/sql/training_data.sql",
        )
        years_all = sorted(
            pd.to_numeric(raw["competition_year"], errors="coerce").dropna().astype(int).unique().tolist()
        )
        if len(years_all) > n_seasons + 1:
            cutoff = years_all[-n_seasons]
            tune_df = raw[pd.to_numeric(raw["competition_year"], errors="coerce") < cutoff]
            baseline_cfg, tier_a_grid = tb.tune_baseline_hyperparams(tune_df)
            if not tier_a_grid.empty:
                print(
                    f"Tier-A tuned on seasons < {cutoff}: alpha={baseline_cfg.alpha:.2f}, "
                    f"carryover={baseline_cfg.carryover:.2f}"
                )

    data, predictors = _load_training_frame(project_root, db_path, baseline_cfg=baseline_cfg)
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

    actual_margin = (
        data["team_final_score_home"].to_numpy(dtype=float)
        - data["team_final_score_away"].to_numpy(dtype=float)
    )
    # Market's expected home margin: the line handicap is negative when the
    # home side is favourite, so the expected margin is its negation.
    market_spread = -pd.to_numeric(
        data.get("implied_spread_home", np.nan), errors="coerce"
    ).to_numpy(dtype=float)
    line_cols = [
        "home_line_cover_prob_shin",
        "home_line_cover_prob_power",
        "home_line_cover_prob_basic",
        "line_overround_basic",
        "implied_spread_home",
    ]
    line_frame = data[[c for c in line_cols if c in data.columns]]

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
            actual_margin,
            market_spread,
            line_frame,
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

    pooled_log_loss = float(log_loss(pooled_y, pooled_p))
    pooled_brier = float(brier_score_loss(pooled_y, pooled_p))

    print("\n── Pooled across held-out seasons ──")
    print(f"  Tipping accuracy: {pooled_correct}/{pooled_games} ({pooled_correct / pooled_games:.1%})")
    if pooled_mkt_games:
        print(f"  Market favourite: {pooled_mkt_correct}/{pooled_mkt_games} ({pooled_mkt_correct / pooled_mkt_games:.1%})")
    print(f"  Log-loss: {pooled_log_loss:.4f}   Brier: {pooled_brier:.4f}")
    if pooled_bets:
        print(f"  Edge bets: {pooled_bets}, flat-stake ROI: {100.0 * pooled_profit / pooled_bets:+.1f}%")

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
        print(f" vs market line {pooled_market_margin_mae:.2f} ({market_margin_games} games)", end="")
    if pooled_margin_blend_mae is not None:
        print(f" vs ridge blend {pooled_margin_blend_mae:.2f}", end="")
    print()

    comp_results = [res["comp_sim"] for res in results if res.get("comp_sim")]
    pooled_p_first = float(np.mean([c["p_first"] for c in comp_results])) if comp_results else None
    pooled_expected_rank = (
        float(np.mean([c["expected_rank"] for c in comp_results])) if comp_results else None
    )
    if comp_results:
        print(
            f"  Comp placement (field of {comp_results[0]['field_size']}): "
            f"P(first) {pooled_p_first:.1%}, expected rank {pooled_expected_rank:.1f}"
        )

    pooled = {
        "games": pooled_games,
        "correct": pooled_correct,
        "accuracy": pooled_correct / pooled_games,
        "market_games": pooled_mkt_games,
        "market_correct": pooled_mkt_correct,
        "market_accuracy": (pooled_mkt_correct / pooled_mkt_games) if pooled_mkt_games else None,
        "log_loss": pooled_log_loss,
        "brier": pooled_brier,
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
        "manifest": _manifest_fingerprint(project_root),
        "env": {
            key: os.environ[key]
            for key in sorted(os.environ)
            if key.startswith("FOOTY_TIPPER_") and "PASSWORD" not in key and "KEY" not in key
        },
    }
    report_path = _write_report(_build_report(results, pooled, config), project_root)
    if report_path is not None:
        print(f"\nReport written to {report_path}")

    print("\nEvaluation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
