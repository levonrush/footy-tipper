from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


VALID_STRATEGIES = {"points", "protect", "chase"}


def _coerce_env_float(name: str, default: float, minimum: float | None = None) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except Exception:
        value = float(default)
    if minimum is not None:
        value = max(float(minimum), value)
    return value


def _coerce_env_int(name: str, default: int, minimum: int | None = None) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(float(raw))
    except Exception:
        value = int(default)
    if minimum is not None:
        value = max(int(minimum), value)
    return value


def _strategy_alias(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    aliases = {
        "points": "points",
        "expected": "points",
        "ev": "points",
        "protect": "protect",
        "lead": "protect",
        "conservative": "protect",
        "chase": "chase",
        "aggressive": "chase",
        "swing": "chase",
    }
    return aliases.get(raw, "points")


def _safe_int(value) -> int:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return 0
    return int(parsed)


def _build_historical_round_metrics(
    training_data: pd.DataFrame,
    min_round_coverage: float = 0.9,
    min_matches_per_round: int = 4,
) -> pd.DataFrame:
    output_columns = [
        "competition_year",
        "round_id",
        "matches_total",
        "matches_priced",
        "odds_coverage",
        "mu",
        "variance",
        "sigma",
    ]
    if training_data.empty:
        return pd.DataFrame(columns=output_columns)

    required_cols = {
        "game_id",
        "competition_year",
        "round_id",
        "team_head_to_head_odds_home",
        "team_head_to_head_odds_away",
    }
    if not required_cols.issubset(set(training_data.columns)):
        return pd.DataFrame(columns=output_columns)

    base = training_data.copy()
    base["competition_year"] = pd.to_numeric(base["competition_year"], errors="coerce")
    base["round_id"] = pd.to_numeric(base["round_id"], errors="coerce")
    base = base[base["competition_year"].notna() & base["round_id"].notna()].copy()
    if base.empty:
        return pd.DataFrame(columns=output_columns)

    round_totals = (
        base.groupby(["competition_year", "round_id"], as_index=False)
        .agg(matches_total=("game_id", "count"))
    )

    base["odds_home"] = pd.to_numeric(base["team_head_to_head_odds_home"], errors="coerce")
    base["odds_away"] = pd.to_numeric(base["team_head_to_head_odds_away"], errors="coerce")
    priced = base[(base["odds_home"] > 1.0) & (base["odds_away"] > 1.0)].copy()
    if priced.empty:
        return pd.DataFrame(columns=output_columns)

    q_home = 1.0 / priced["odds_home"]
    q_away = 1.0 / priced["odds_away"]
    overround = q_home + q_away
    priced = priced[overround > 0].copy()
    if priced.empty:
        return pd.DataFrame(columns=output_columns)

    q_home = 1.0 / priced["odds_home"]
    q_away = 1.0 / priced["odds_away"]
    overround = q_home + q_away
    priced["p_home"] = q_home / overround
    priced["p_away"] = q_away / overround
    priced["p_tip_correct"] = priced[["p_home", "p_away"]].max(axis=1)
    priced["match_variance"] = priced["p_tip_correct"] * (1.0 - priced["p_tip_correct"])

    metrics = (
        priced.groupby(["competition_year", "round_id"], as_index=False)
        .agg(
            matches_priced=("game_id", "count"),
            mu=("p_tip_correct", "sum"),
            variance=("match_variance", "sum"),
        )
    )
    metrics = metrics.merge(round_totals, on=["competition_year", "round_id"], how="left")
    metrics["matches_total"] = pd.to_numeric(metrics["matches_total"], errors="coerce").fillna(0).astype(int)
    metrics["matches_priced"] = pd.to_numeric(metrics["matches_priced"], errors="coerce").fillna(0).astype(int)
    metrics["odds_coverage"] = metrics["matches_priced"] / metrics["matches_total"].replace(0, pd.NA)
    metrics["odds_coverage"] = pd.to_numeric(metrics["odds_coverage"], errors="coerce").fillna(0.0)
    metrics["sigma"] = metrics["variance"].clip(lower=0.0).pow(0.5)

    metrics = metrics[
        (metrics["odds_coverage"] >= float(min_round_coverage))
        & (metrics["matches_priced"] >= int(min_matches_per_round))
    ].copy()
    metrics = metrics.sort_values(["competition_year", "round_id"]).reset_index(drop=True)
    if metrics.empty:
        return pd.DataFrame(columns=output_columns)

    return metrics[output_columns]


def _choose_round(metrics: pd.DataFrame, strategy: str, risk_lambda: float) -> int:
    strategy = _strategy_alias(strategy)
    if strategy == "protect":
        scores = metrics["mu"] - (risk_lambda * metrics["sigma"])
    elif strategy == "chase":
        scores = metrics["variance"]
    else:
        scores = metrics["mu"]
    idx = int(pd.Series(scores).idxmax())
    return int(metrics.iloc[idx]["round_id"])


def _simulate_round_win_probabilities(
    metrics: pd.DataFrame,
    points_gap: float,
    n_simulations: int,
    field_size: int,
    crowd_mean_penalty: float,
    crowd_skill_sigma: float,
    rng_seed: int,
) -> tuple[dict[int, float], float]:
    if metrics.empty:
        return {}, float("nan")

    m = len(metrics)
    round_ids = metrics["round_id"].astype(int).to_numpy()
    mu = metrics["mu"].to_numpy(dtype=float)
    sigma = metrics["sigma"].to_numpy(dtype=float)
    matches_cap = metrics["matches_priced"].astype(int).to_numpy(dtype=int)

    user_sigma = np.maximum(sigma, 0.8)
    field_sigma = np.maximum(sigma * 1.08, 0.9)

    rng = np.random.default_rng(int(rng_seed))

    user_round_scores = rng.normal(loc=mu.reshape(1, m), scale=user_sigma.reshape(1, m), size=(n_simulations, m))
    user_round_scores = np.rint(np.clip(user_round_scores, 0.0, matches_cap.reshape(1, m))).astype(float)

    crowd_skill = rng.normal(loc=0.0, scale=max(crowd_skill_sigma, 1e-6), size=(n_simulations, field_size, 1))
    field_mu = mu.reshape(1, 1, m) - crowd_mean_penalty + crowd_skill
    field_round_scores = rng.normal(loc=field_mu, scale=field_sigma.reshape(1, 1, m), size=(n_simulations, field_size, m))
    field_round_scores = np.rint(np.clip(field_round_scores, 0.0, matches_cap.reshape(1, 1, m))).astype(float)

    field_base_totals = field_round_scores.sum(axis=2)
    field_joker_round = rng.integers(0, m, size=(n_simulations, field_size))
    sim_idx = np.arange(n_simulations)[:, None]
    entrant_idx = np.arange(field_size)[None, :]
    field_bonus = field_round_scores[sim_idx, entrant_idx, field_joker_round]
    field_totals = field_base_totals + field_bonus
    field_best = field_totals.max(axis=1)

    user_base_totals = user_round_scores.sum(axis=1)

    # Every candidate round is scored against the same simulated draws, so
    # strategy comparisons are paired: only the chosen round differs.
    results = {}
    for idx, round_id in enumerate(round_ids):
        user_total = user_base_totals + user_round_scores[:, idx] - float(points_gap)
        wins = (user_total > field_best).astype(float)
        ties = (user_total == field_best).astype(float) * 0.5
        results[int(round_id)] = float((wins + ties).mean())

    no_joker_total = user_base_totals - float(points_gap)
    no_joker_wins = (no_joker_total > field_best).astype(float)
    no_joker_ties = (no_joker_total == field_best).astype(float) * 0.5
    win_prob_no_joker = float((no_joker_wins + no_joker_ties).mean())
    return results, win_prob_no_joker


def run_joker_policy_backtest(training_data: pd.DataFrame) -> dict:
    risk_lambda = _coerce_env_float("FOOTY_TIPPER_JOKER_RISK_LAMBDA", 1.0, minimum=0.0)
    min_round_coverage = _coerce_env_float("FOOTY_TIPPER_JOKER_MIN_ROUND_COVERAGE", 0.95, minimum=0.0)
    min_matches_per_round = _coerce_env_int("FOOTY_TIPPER_JOKER_MIN_MATCHES_PER_ROUND", 4, minimum=2)
    min_rounds_per_season = _coerce_env_int("FOOTY_TIPPER_JOKER_MIN_ROUNDS_PER_SEASON", 6, minimum=2)

    n_simulations = _coerce_env_int("FOOTY_TIPPER_JOKER_BACKTEST_SIMULATIONS", 20000, minimum=500)
    field_size = _coerce_env_int("FOOTY_TIPPER_JOKER_BACKTEST_FIELD_SIZE", 75, minimum=10)
    crowd_mean_penalty = _coerce_env_float("FOOTY_TIPPER_JOKER_BACKTEST_CROWD_MEAN_PENALTY", 0.05, minimum=0.0)
    crowd_skill_sigma = _coerce_env_float("FOOTY_TIPPER_JOKER_BACKTEST_CROWD_SKILL_SIGMA", 0.12, minimum=0.0)
    seed = _coerce_env_int("FOOTY_TIPPER_JOKER_BACKTEST_SEED", 2026, minimum=1)

    lead_gap = _coerce_env_float("FOOTY_TIPPER_JOKER_SCENARIO_LEAD_GAP", -6.0)
    neutral_gap = _coerce_env_float("FOOTY_TIPPER_JOKER_SCENARIO_NEUTRAL_GAP", 0.0)
    chase_gap = _coerce_env_float("FOOTY_TIPPER_JOKER_SCENARIO_CHASE_GAP", 6.0)

    lead_max_gap = _coerce_env_float("FOOTY_TIPPER_JOKER_LEAD_MAX_GAP", -3.0)
    chase_min_gap = _coerce_env_float("FOOTY_TIPPER_JOKER_CHASE_MIN_GAP", 3.0)

    rounds = _build_historical_round_metrics(
        training_data,
        min_round_coverage=min_round_coverage,
        min_matches_per_round=min_matches_per_round,
    )

    scenarios = {
        "lead": float(lead_gap),
        "neutral": float(neutral_gap),
        "chase": float(chase_gap),
    }
    strategies = ["points", "protect", "chase"]

    if rounds.empty:
        return {
            "version": 2,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": "training_backtest_market_odds",
            "status": "insufficient_data",
            "default_strategy": "points",
            "recommended_strategy_by_scenario": {k: "points" for k in scenarios.keys()},
            "state_thresholds": {
                "lead_max_gap": float(lead_max_gap),
                "chase_min_gap": float(chase_min_gap),
            },
            "simulation_config": {
                "risk_lambda": float(risk_lambda),
                "min_round_coverage": float(min_round_coverage),
                "min_matches_per_round": int(min_matches_per_round),
                "min_rounds_per_season": int(min_rounds_per_season),
                "simulations": int(n_simulations),
                "field_size": int(field_size),
                "crowd_mean_penalty": float(crowd_mean_penalty),
                "crowd_skill_sigma": float(crowd_skill_sigma),
                "seed": int(seed),
            },
            "seasons_evaluated": 0,
            "rounds_evaluated": 0,
            "scenario_results": [],
            "note": "No historical rounds met joker backtest coverage requirements.",
        }

    round_groups = rounds.groupby("competition_year")
    scenario_seed_offset = {"lead": 101, "neutral": 202, "chase": 303}
    season_records = []
    for season_key, season_df in round_groups:
        season_metrics = season_df.sort_values("round_id").reset_index(drop=True)
        if len(season_metrics) < min_rounds_per_season:
            continue
        season = _safe_int(season_key)
        for scenario_name, points_gap in scenarios.items():
            seed_offset = int(seed + (season * 37) + scenario_seed_offset.get(scenario_name, 0))
            win_prob_by_round, win_prob_no_joker = _simulate_round_win_probabilities(
                season_metrics,
                points_gap=points_gap,
                n_simulations=n_simulations,
                field_size=field_size,
                crowd_mean_penalty=crowd_mean_penalty,
                crowd_skill_sigma=crowd_skill_sigma,
                rng_seed=seed_offset,
            )
            if not win_prob_by_round:
                continue
            for strategy in strategies:
                chosen_round = _choose_round(season_metrics, strategy, risk_lambda)
                season_records.append(
                    {
                        "competition_year": season,
                        "scenario": scenario_name,
                        "strategy": strategy,
                        "chosen_round_id": int(chosen_round),
                        "win_prob": float(win_prob_by_round.get(chosen_round, np.nan)),
                        "win_prob_no_joker": float(win_prob_no_joker),
                    }
                )

    results_df = pd.DataFrame.from_records(season_records)
    if results_df.empty:
        return {
            "version": 2,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": "training_backtest_market_odds",
            "status": "insufficient_data",
            "default_strategy": "points",
            "recommended_strategy_by_scenario": {k: "points" for k in scenarios.keys()},
            "state_thresholds": {
                "lead_max_gap": float(lead_max_gap),
                "chase_min_gap": float(chase_min_gap),
            },
            "simulation_config": {
                "risk_lambda": float(risk_lambda),
                "min_round_coverage": float(min_round_coverage),
                "min_matches_per_round": int(min_matches_per_round),
                "min_rounds_per_season": int(min_rounds_per_season),
                "simulations": int(n_simulations),
                "field_size": int(field_size),
                "crowd_mean_penalty": float(crowd_mean_penalty),
                "crowd_skill_sigma": float(crowd_skill_sigma),
                "seed": int(seed),
            },
            "seasons_evaluated": 0,
            "rounds_evaluated": int(len(rounds)),
            "scenario_results": [],
            "note": "Backtest skipped because no seasons met minimum priced-round requirements.",
        }

    scenario_summary = (
        results_df.groupby(["scenario", "strategy"], as_index=False)
        .agg(
            mean_win_prob=("win_prob", "mean"),
            median_win_prob=("win_prob", "median"),
            mean_win_prob_no_joker=("win_prob_no_joker", "mean"),
            season_count=("competition_year", "nunique"),
        )
        .sort_values(["scenario", "mean_win_prob"], ascending=[True, False])
        .reset_index(drop=True)
    )
    scenario_summary["mean_joker_lift"] = (
        scenario_summary["mean_win_prob"] - scenario_summary["mean_win_prob_no_joker"]
    )

    # Strategies that differ by less than the tie epsilon are within Monte
    # Carlo noise; prefer plain expected-points in that case rather than
    # letting the 4th decimal pick the policy.
    tie_epsilon = _coerce_env_float("FOOTY_TIPPER_JOKER_TIE_EPSILON", 0.002, minimum=0.0)
    recommended_by_scenario = {}
    for scenario_name in scenarios.keys():
        subset = scenario_summary[scenario_summary["scenario"] == scenario_name]
        if subset.empty:
            recommended_by_scenario[scenario_name] = "points"
            continue
        best = subset.iloc[0]
        points_rows = subset[subset["strategy"] == "points"]
        if (
            not points_rows.empty
            and float(best["mean_win_prob"]) - float(points_rows.iloc[0]["mean_win_prob"]) <= tie_epsilon
        ):
            recommended_by_scenario[scenario_name] = "points"
        else:
            recommended_by_scenario[scenario_name] = _strategy_alias(str(best["strategy"]))

    default_strategy = _strategy_alias(recommended_by_scenario.get("neutral", "points"))

    return {
        "version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "training_backtest_market_odds",
        "status": "ok",
        "default_strategy": default_strategy,
        "recommended_strategy_by_scenario": recommended_by_scenario,
        "state_thresholds": {
            "lead_max_gap": float(lead_max_gap),
            "chase_min_gap": float(chase_min_gap),
        },
        "simulation_config": {
            "risk_lambda": float(risk_lambda),
            "min_round_coverage": float(min_round_coverage),
            "min_matches_per_round": int(min_matches_per_round),
            "min_rounds_per_season": int(min_rounds_per_season),
            "simulations": int(n_simulations),
            "field_size": int(field_size),
            "crowd_mean_penalty": float(crowd_mean_penalty),
            "crowd_skill_sigma": float(crowd_skill_sigma),
            "seed": int(seed),
            "tie_epsilon": float(tie_epsilon),
            "scenario_points_gap": scenarios,
        },
        "seasons_evaluated": int(results_df["competition_year"].nunique()),
        "rounds_evaluated": int(len(rounds)),
        "scenario_results": scenario_summary.to_dict(orient="records"),
    }


def save_joker_policy(training_data: pd.DataFrame, output_path: Path) -> dict:
    policy = run_joker_policy_backtest(training_data)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    return policy
