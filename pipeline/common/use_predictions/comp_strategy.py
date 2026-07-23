"""Competition-aware tip selection: maximize P(win the comp), not tips.

Pure argmax tipping maximizes expected correct tips, but a tipping comp is
won on *relative* score. When you lead, mirroring the field's tips removes
relative variance and protects the lead; when you trail, high-EV contrarian
picks buy the variance you need. This module searches small deviations from
the model's argmax tips and keeps the set that maximizes simulated P(win),
using the calibrated model probabilities as reality and a rival field that
tips market favourites with per-rival skill noise.

Modes (FOOTY_TIPPER_COMP_STRATEGY): off | advisory (default) | auto.
Only auto changes the tips that are rendered/sent; the predictions_table is
never modified, and every decision is persisted to comp_strategy_decisions.
"""

import itertools
import os
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pipeline.common.model_prediciton.prediction_functions import GAME_SEED_BASE
from pipeline.common.use_predictions.joker import (
    compute_joker_round_metrics,
    get_joker_round_candidates,
)

DEFAULT_FIELD_SIZE = 75
DEFAULT_N_SIMS = 8000
DEFAULT_MAX_FLIPS = 2
DEFAULT_FLIP_BAND = (0.38, 0.62)
DEFAULT_RIVAL_FLIP_MEAN = 0.12
DEFAULT_RIVAL_FLIP_SIGMA = 0.06
# Minimum simulated P(win) gain before deviating from the model's tips.
DEFAULT_MIN_PWIN_GAIN = 0.002
# Expected extra correct tips per future round vs tipping market favourites
# (honest eval: ~63.4% model vs 61.2% market over ~8 games -> ~0.17/round).
DEFAULT_USER_EDGE_PER_ROUND = 0.15


def _env_float(name, default):
    try:
        return float(os.getenv(name, str(default)).strip())
    except Exception:
        return float(default)


def _env_int(name, default, minimum=None):
    try:
        value = int(float(os.getenv(name, str(default)).strip()))
    except Exception:
        value = int(default)
    if minimum is not None:
        value = max(int(minimum), value)
    return value


def resolve_comp_strategy_mode():
    mode = os.getenv("FOOTY_TIPPER_COMP_STRATEGY", "advisory").strip().lower()
    if mode in {"off", "advisory", "auto"}:
        return mode
    return "advisory"


def _future_round_metrics(db_path, project_root, current_round_id):
    """Market mu/sigma for rounds after the current one (may be empty)."""
    try:
        fixtures = get_joker_round_candidates(db_path, project_root)
        metrics = compute_joker_round_metrics(fixtures)
        if metrics.empty:
            return metrics
        round_ids = pd.to_numeric(metrics["round_id"], errors="coerce")
        return metrics[round_ids > float(current_round_id)].reset_index(drop=True)
    except Exception as exc:
        print(f"Comp strategy: future round metrics unavailable ({exc}).")
        return pd.DataFrame()


def simulate_comp_outcomes(
    model_p,
    market_p,
    points_gap,
    future_mu,
    future_sigma,
    future_caps,
    future_edges,
    field_size=DEFAULT_FIELD_SIZE,
    n_sims=DEFAULT_N_SIMS,
    flip_mean=DEFAULT_RIVAL_FLIP_MEAN,
    flip_sigma=DEFAULT_RIVAL_FLIP_SIGMA,
    user_edge=DEFAULT_USER_EDGE_PER_ROUND,
    seed=GAME_SEED_BASE,
):
    """Simulate everything except the user's current-round tips.

    Returns (outcomes, rival_totals_without_current_score + rival current
    score, gap) pieces so candidate tip sets can be scored cheaply against
    identical draws: only the user's current-round score changes between
    candidates, making comparisons exactly paired.
    """
    model_p = np.asarray(model_p, dtype=float)
    market_p = np.asarray(market_p, dtype=float)
    n_games = model_p.size
    rng = np.random.default_rng(int(seed))

    outcomes = rng.random((n_sims, n_games)) < model_p.reshape(1, -1)

    flip_rates = np.clip(
        rng.normal(flip_mean, flip_sigma, size=(n_sims, field_size)), 0.02, 0.5
    )
    has_fav = np.isfinite(market_p) & (market_p != 0.5)
    fav_tip = market_p > 0.5
    flips = rng.random((n_sims, field_size, n_games)) < flip_rates[:, :, None]
    rival_tips = np.where(
        has_fav.reshape(1, 1, -1),
        fav_tip.reshape(1, 1, -1) ^ flips,
        rng.random((n_sims, field_size, n_games)) < 0.5,
    )
    rival_scores = (rival_tips == outcomes[:, None, :]).sum(axis=2).astype(float)

    # Future rounds: normal approximations from market round metrics. Rivals
    # with higher flip rates lose expected points proportional to the round's
    # mean favourite edge.
    user_future = np.zeros(n_sims)
    rival_future = np.zeros((n_sims, field_size))
    for mu, sigma, cap, edge in zip(future_mu, future_sigma, future_caps, future_edges):
        sigma_u = max(float(sigma), 0.8)
        user_future += np.clip(rng.normal(mu + float(user_edge), sigma_u, size=n_sims), 0.0, cap)
        rival_mu = mu - flip_rates * max(float(edge), 0.0)
        rival_future += np.clip(
            rng.normal(rival_mu, max(float(sigma) * 1.08, 0.9), size=(n_sims, field_size)),
            0.0,
            cap,
        )

    rival_totals = rival_scores + rival_future
    return outcomes, rival_totals, user_future


def score_tip_candidate(tips, outcomes, rival_totals, user_future, points_gap):
    """P(win comp) for one tip vector against the shared simulated draws."""
    tips = np.asarray(tips, dtype=bool)
    user_score = (tips.reshape(1, -1) == outcomes).sum(axis=1).astype(float)
    user_total = user_score + user_future - float(points_gap)
    best_rival = rival_totals.max(axis=1)
    wins = user_total > best_rival
    ties = user_total == best_rival
    return float(np.mean(wins + 0.5 * ties))


def _unavailable(reason, mode):
    return {
        "available": False,
        "mode": mode,
        "status": "unavailable",
        "headline": "Comp strategy unavailable",
        "detail": reason,
        "deviations": [],
        "p_win_baseline": None,
        "p_win_adjusted": None,
        "tips_changed": 0,
    }


def get_comp_strategy_recommendation(db_path, project_root, predictions):
    """Recommend (or in auto mode, apply) competition-aware tip deviations.

    Never raises: any failure returns an unavailable payload and the caller
    keeps the pure model tips.
    """
    mode = resolve_comp_strategy_mode()
    if mode == "off":
        return {**_unavailable("Comp strategy disabled (FOOTY_TIPPER_COMP_STRATEGY=off).", mode), "status": "off"}

    try:
        return _recommend(db_path, project_root, predictions, mode)
    except Exception as exc:
        print(f"Comp strategy failed soft ({exc}).")
        return _unavailable(f"Comp strategy failed ({exc}).", mode)


def _recommend(db_path, project_root, predictions, mode):
    required = {"game_id", "home_team_win_prob", "home_team_lose_prob", "round_id"}
    if predictions is None or predictions.empty or not required.issubset(predictions.columns):
        return _unavailable("No predictions with win probabilities available.", mode)

    frame = predictions.reset_index(drop=True)
    win = pd.to_numeric(frame["home_team_win_prob"], errors="coerce")
    lose = pd.to_numeric(frame["home_team_lose_prob"], errors="coerce")
    denom = win + lose
    model_p = np.clip((win / denom).where(denom > 0, 0.5).fillna(0.5).to_numpy(dtype=float), 1e-6, 1 - 1e-6)

    market_p = np.full(len(frame), np.nan)
    if {"team_head_to_head_odds_home", "team_head_to_head_odds_away"}.issubset(frame.columns):
        oh = pd.to_numeric(frame["team_head_to_head_odds_home"], errors="coerce")
        oa = pd.to_numeric(frame["team_head_to_head_odds_away"], errors="coerce")
        qh, qa = 1.0 / oh, 1.0 / oa
        market_p = (qh / (qh + qa)).where((oh > 1.0) & (oa > 1.0)).to_numpy(dtype=float)

    points_gap = _env_float("FOOTY_TIPPER_COMP_GAP", _env_float("FOOTY_TIPPER_JOKER_POINTS_GAP", 0.0))
    field_size = _env_int("FOOTY_TIPPER_COMP_FIELD_SIZE", DEFAULT_FIELD_SIZE, minimum=2)
    max_flips = _env_int("FOOTY_TIPPER_COMP_MAX_FLIPS", DEFAULT_MAX_FLIPS, minimum=0)
    n_sims = _env_int("FOOTY_TIPPER_COMP_SIMULATIONS", DEFAULT_N_SIMS, minimum=1000)
    band_lo = _env_float("FOOTY_TIPPER_COMP_FLIP_BAND_LO", DEFAULT_FLIP_BAND[0])
    band_hi = _env_float("FOOTY_TIPPER_COMP_FLIP_BAND_HI", DEFAULT_FLIP_BAND[1])
    min_gain = _env_float("FOOTY_TIPPER_COMP_MIN_PWIN_GAIN", DEFAULT_MIN_PWIN_GAIN)
    rounds_left_env = _env_int("FOOTY_TIPPER_COMP_ROUNDS_LEFT", -1)

    current_round_id = pd.to_numeric(frame["round_id"], errors="coerce").iloc[0]
    future = _future_round_metrics(db_path, project_root, current_round_id)
    if not future.empty:
        future_mu = future["mu"].to_numpy(dtype=float)
        future_sigma = future["sigma"].to_numpy(dtype=float)
        future_caps = future["matches_considered"].to_numpy(dtype=float)
        # Expected points a rival loses per unit flip rate: sum of (2q-1).
        future_edges = (2.0 * future_mu - future_caps)
        if rounds_left_env >= 0:
            future_mu = future_mu[:rounds_left_env]
            future_sigma = future_sigma[:rounds_left_env]
            future_caps = future_caps[:rounds_left_env]
            future_edges = future_edges[:rounds_left_env]
    else:
        future_mu = np.array([])
        future_sigma = np.array([])
        future_caps = np.array([])
        future_edges = np.array([])

    seed = int(GAME_SEED_BASE + 7919 * int(current_round_id if np.isfinite(current_round_id) else 0))
    outcomes, rival_totals, user_future = simulate_comp_outcomes(
        model_p,
        market_p,
        points_gap,
        future_mu,
        future_sigma,
        future_caps,
        future_edges,
        field_size=field_size,
        n_sims=n_sims,
        user_edge=_env_float("FOOTY_TIPPER_COMP_USER_EDGE", DEFAULT_USER_EDGE_PER_ROUND),
        seed=seed,
    )

    baseline_tips = model_p > 0.5
    p_win_baseline = score_tip_candidate(baseline_tips, outcomes, rival_totals, user_future, points_gap)

    flippable = [i for i, p in enumerate(model_p) if band_lo <= p <= band_hi]
    best_tips = baseline_tips
    best_p_win = p_win_baseline
    for size in range(1, max_flips + 1):
        for combo in itertools.combinations(flippable, size):
            candidate = baseline_tips.copy()
            for i in combo:
                candidate[i] = not candidate[i]
            p_win = score_tip_candidate(candidate, outcomes, rival_totals, user_future, points_gap)
            if p_win > best_p_win:
                best_tips, best_p_win = candidate, p_win

    if best_p_win - p_win_baseline < min_gain:
        best_tips, best_p_win = baseline_tips, p_win_baseline

    deviations = []
    for i in np.flatnonzero(best_tips != baseline_tips):
        home = str(frame.iloc[i].get("team_home", "Home"))
        away = str(frame.iloc[i].get("team_away", "Away"))
        deviations.append(
            {
                "game_id": frame.iloc[i]["game_id"],
                "team_home": home,
                "team_away": away,
                "model_tip": home if baseline_tips[i] else away,
                "strategy_tip": home if best_tips[i] else away,
                "model_p_home": float(model_p[i]),
                "market_p_home": float(market_p[i]) if np.isfinite(market_p[i]) else None,
            }
        )

    scenario = "lead" if points_gap <= -3 else ("chase" if points_gap >= 3 else "neutral")
    if deviations:
        applied = "APPLIED" if mode == "auto" else "recommended (advisory only)"
        headline = f"COMP STRATEGY: {len(deviations)} tip deviation(s) {applied}"
        names = ", ".join(f"{d['model_tip']} -> {d['strategy_tip']}" for d in deviations)
        detail = (
            f"Scenario {scenario} (gap {points_gap:+.0f}): flipping {names} lifts simulated "
            f"P(win comp) {p_win_baseline:.1%} -> {best_p_win:.1%} against a field of {field_size}."
        )
    else:
        headline = "COMP STRATEGY: model tips already optimal"
        detail = (
            f"Scenario {scenario} (gap {points_gap:+.0f}): no deviation beat the model's tips "
            f"(P(win comp) {p_win_baseline:.1%} against a field of {field_size})."
        )

    return {
        "available": True,
        "mode": mode,
        "status": "ok",
        "headline": headline,
        "detail": detail,
        "scenario": scenario,
        "points_gap": float(points_gap),
        "field_size": int(field_size),
        "n_sims": int(n_sims),
        "future_rounds_modelled": int(len(future_mu)),
        "p_win_baseline": float(p_win_baseline),
        "p_win_adjusted": float(best_p_win),
        "deviations": deviations,
        "tips_changed": int(len(deviations)),
        "baseline_tips_home": [bool(t) for t in baseline_tips],
        "strategy_tips_home": [bool(t) for t in best_tips],
    }


def apply_comp_strategy_to_predictions(predictions, recommendation):
    """Return a copy of the predictions frame with strategy tips applied.

    Only used in auto mode. Flips home_team_result for deviated games and
    re-clamps predicted_margin/scoreline signs to the new tip. Probabilities
    are left honest (unchanged).
    """
    if not recommendation.get("available") or not recommendation.get("deviations"):
        return predictions

    adjusted = predictions.copy()
    flipped_ids = {d["game_id"] for d in recommendation["deviations"]}
    for idx in adjusted.index:
        if adjusted.loc[idx, "game_id"] not in flipped_ids:
            continue
        new_result = "Loss" if str(adjusted.loc[idx].get("home_team_result")) == "Win" else "Win"
        adjusted.loc[idx, "home_team_result"] = new_result
        if "predicted_margin" in adjusted.columns:
            margin = pd.to_numeric(pd.Series([adjusted.loc[idx, "predicted_margin"]]), errors="coerce").iloc[0]
            if pd.notna(margin):
                if new_result == "Win" and margin <= 0:
                    adjusted.loc[idx, "predicted_margin"] = 1
                elif new_result == "Loss" and margin >= 0:
                    adjusted.loc[idx, "predicted_margin"] = -1
        if {"predicted_home_score", "predicted_away_score"}.issubset(adjusted.columns):
            hs = pd.to_numeric(pd.Series([adjusted.loc[idx, "predicted_home_score"]]), errors="coerce").iloc[0]
            as_ = pd.to_numeric(pd.Series([adjusted.loc[idx, "predicted_away_score"]]), errors="coerce").iloc[0]
            if pd.notna(hs) and pd.notna(as_):
                winner_ahead = hs > as_ if new_result == "Win" else as_ > hs
                if not winner_ahead:
                    if hs == as_:
                        adjusted.loc[idx, "predicted_home_score"] = (
                            hs + 1 if new_result == "Win" else hs
                        )
                        adjusted.loc[idx, "predicted_away_score"] = (
                            as_ + 1 if new_result == "Loss" else as_
                        )
                    else:
                        adjusted.loc[idx, "predicted_home_score"] = as_
                        adjusted.loc[idx, "predicted_away_score"] = hs
                if "predicted_margin" in adjusted.columns:
                    adjusted.loc[idx, "predicted_margin"] = int(
                        adjusted.loc[idx, "predicted_home_score"]
                        - adjusted.loc[idx, "predicted_away_score"]
                    )
    return adjusted


def persist_comp_strategy_decision(db_path, recommendation, predictions):
    """Log baseline and strategy tips per game; fail-soft, never raises."""
    if not recommendation.get("available"):
        return False
    try:
        year = None
        round_id = None
        if predictions is not None and not predictions.empty:
            year_val = pd.to_numeric(pd.Series([predictions.iloc[0].get("competition_year")]), errors="coerce").iloc[0]
            round_val = pd.to_numeric(pd.Series([predictions.iloc[0].get("round_id")]), errors="coerce").iloc[0]
            year = int(year_val) if pd.notna(year_val) else None
            round_id = int(round_val) if pd.notna(round_val) else None

        con = sqlite3.connect(str(db_path))
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS comp_strategy_decisions (
                    created_at_utc TEXT NOT NULL,
                    competition_year INTEGER,
                    round_id INTEGER,
                    game_id INTEGER,
                    mode TEXT,
                    scenario TEXT,
                    points_gap REAL,
                    baseline_tip_home INTEGER,
                    strategy_tip_home INTEGER,
                    p_win_baseline REAL,
                    p_win_adjusted REAL
                )
                """
            )
            now = datetime.now(timezone.utc).isoformat()
            baseline = recommendation.get("baseline_tips_home", [])
            strategy = recommendation.get("strategy_tips_home", [])
            game_ids = predictions["game_id"].tolist() if predictions is not None else []
            for gid, b, s in zip(game_ids, baseline, strategy):
                con.execute(
                    "INSERT INTO comp_strategy_decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        now,
                        year,
                        round_id,
                        int(gid),
                        recommendation.get("mode"),
                        recommendation.get("scenario"),
                        recommendation.get("points_gap"),
                        int(bool(b)),
                        int(bool(s)),
                        recommendation.get("p_win_baseline"),
                        recommendation.get("p_win_adjusted"),
                    ),
                )
            con.commit()
        finally:
            con.close()
        return True
    except Exception as exc:
        print(f"Comp strategy decision log skipped ({exc}).")
        return False
