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
from pipeline.runtime_paths import (
    database_path,
    models_path,
    project_root as configured_project_root,
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
            market_loso = calib.loso_simplex_pool_predictions(
                tier_a=tier_a_cond[market_prior_mask],
                tier_b=tier_b_cond[market_prior_mask],
                tier_c=tier_c_cond_oof[market_prior_mask],
                market=market_cond[market_prior_mask],
                y=y_full[market_prior_mask],
                groups=year_col[market_prior_mask],
                include_market=True,
            )
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
            )
            market_expert_probabilities = {
                "tier_a": tier_a_cond[market_prior_mask],
                "tier_b": tier_b_cond[market_prior_mask],
                "tier_c": tier_c_cond_oof[market_prior_mask],
                "market": market_cond[market_prior_mask],
            }
            market_selection = calib.select_market_pool(
                market_pool,
                market_nested,
                y_full[market_prior_mask],
                market_expert_probabilities,
                groups=year_col[market_prior_mask],
            )
            market_calibrator = calib.fit_selected_market_calibrator(
                market_selection,
                market_expert_probabilities,
                y_full[market_prior_mask],
                market_calibrator,
            )

    no_market_pool = calib.SimplexLogitPool(include_market=False).fit(
        tier_a=tier_a_cond[prior_mask],
        tier_b=tier_b_cond[prior_mask],
        tier_c=tier_c_cond_oof[prior_mask],
        y=y_full[prior_mask],
    )
    no_market_learned_weights = no_market_pool.weight_map
    no_market_calibrator = calib.TemperatureCalibrator()

    no_market_loso = calib.loso_simplex_pool_predictions(
        tier_a=tier_a_cond[prior_mask],
        tier_b=tier_b_cond[prior_mask],
        tier_c=tier_c_cond_oof[prior_mask],
        y=y_full[prior_mask],
        groups=year_col[prior_mask],
        include_market=False,
    )
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
    )
    no_market_expert_probabilities = {
        "tier_a": tier_a_cond[prior_mask],
        "tier_b": tier_b_cond[prior_mask],
        "tier_c": tier_c_cond_oof[prior_mask],
    }
    no_market_selection = calib.select_no_market_pool(
        no_market_pool,
        nested_no_market,
        y_full[prior_mask],
        no_market_expert_probabilities,
        groups=year_col[prior_mask],
    )
    no_market_strategy = no_market_selection["strategy"]
    no_market_calibrator = calib.fit_selected_pool_calibrator(
        no_market_selection,
        no_market_expert_probabilities,
        y_full[prior_mask],
        no_market_calibrator,
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
