from __future__ import annotations

import sqlite3
import pathlib
from collections import Counter

import dill as pickle
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import poisson, skellam

from pipeline.common.odds.validity import valid_decimal_odds

# Fixed base so every run over the same fixtures produces the same tips.
GAME_SEED_BASE = 20100308

# Share of eighty-minute ties that survive golden point and are recorded as
# draws. The simulation's raw tie rate (about 3.9%) matches the pre-golden-point
# era, so it is the extra-time period that was missing, not the tie estimate.
# Calibrated against the realised NRL draw rate: 14 in 3577 games since 2009
# (0.39%), 7 in 2170 since 2016 (0.32%), against a 3.9% simulated tie rate.
GOLDEN_POINT_UNRESOLVED_SHARE = 0.10


def rng_for_game(game_id, salt=0):
    """Deterministic per-game RNG so re-runs never flip a tip."""
    try:
        seed = GAME_SEED_BASE + int(game_id) * 1009 + int(salt)
    except Exception:
        seed = GAME_SEED_BASE + int(salt)
    return np.random.default_rng(seed)


def get_inference_data(db_path, sql_file):
    """Retrieve data for inference from an SQLite database."""
    print("Getting inference data...")
    con = sqlite3.connect(str(db_path))
    with open(sql_file, "r") as file:
        query = file.read()
    inference_data = pd.read_sql_query(query, con)
    con.close()
    return inference_data


def get_table_data(db_path, table_name):
    """Read a full table from SQLite."""
    con = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", con)
    con.close()
    return df


def predict_scores(model, data):
    """Predict expected scores."""
    return model.predict(data)


def compute_outcome_probs_independent(mu_home, mu_away):
    draw_prob = float(skellam.pmf(0, mu_home, mu_away))
    home_win_prob = float(1.0 - skellam.cdf(0, mu_home, mu_away))
    away_win_prob = float(max(0.0, 1.0 - home_win_prob - draw_prob))
    return home_win_prob, away_win_prob, draw_prob


def conditional_home_win_prob(mu_home, mu_away):
    home_win, away_win, _ = compute_outcome_probs_independent(mu_home, mu_away)
    non_draw = max(1e-9, home_win + away_win)
    return home_win / non_draw


def conditional_home_win_prob_vec(mu_home, mu_away):
    """Vectorised p(home win | non-draw) under independent Poisson scores."""
    mu_home = np.maximum(np.asarray(mu_home, dtype=float), 1e-9)
    mu_away = np.maximum(np.asarray(mu_away, dtype=float), 1e-9)
    draw_prob = skellam.pmf(0, mu_home, mu_away)
    home_win = 1.0 - skellam.cdf(0, mu_home, mu_away)
    away_win = np.maximum(0.0, 1.0 - home_win - draw_prob)
    non_draw = np.maximum(1e-9, home_win + away_win)
    return home_win / non_draw


def marginalized_conditional_home_win_prob(
    mu_home,
    mu_away,
    lineup_uncertainty_home=0.0,
    lineup_uncertainty_away=0.0,
    n_samples=64,
    mu_noise_scale=0.12,
    rng=None,
):
    """
    Approximate p(home win | non-draw) by marginalising over lineup uncertainty.

    The uncertainty terms are expected to be in [0, 0.25] from p(1-p) style
    features. We convert them to multiplicative score-mean noise and average the
    conditional win probabilities across Monte Carlo draws.
    """
    base = conditional_home_win_prob(mu_home, mu_away)
    n_samples = int(max(1, n_samples))
    mu_noise_scale = float(max(0.0, mu_noise_scale))
    if n_samples <= 1 or mu_noise_scale <= 0:
        return base

    if rng is None:
        rng = np.random.default_rng()

    uh = float(max(0.0, lineup_uncertainty_home))
    ua = float(max(0.0, lineup_uncertainty_away))
    std_home = mu_noise_scale * np.sqrt(uh)
    std_away = mu_noise_scale * np.sqrt(ua)
    if std_home <= 1e-9 and std_away <= 1e-9:
        return base

    # Lognormal multipliers keep score means positive and centred near 1.0.
    mult_home = np.exp(rng.normal(loc=-0.5 * (std_home ** 2), scale=std_home, size=n_samples))
    mult_away = np.exp(rng.normal(loc=-0.5 * (std_away ** 2), scale=std_away, size=n_samples))

    probs = []
    for mh, ma in zip(mult_home, mult_away):
        sample_mu_home = max(1e-6, float(mu_home) * float(mh))
        sample_mu_away = max(1e-6, float(mu_away) * float(ma))
        probs.append(conditional_home_win_prob(sample_mu_home, sample_mu_away))

    return float(np.mean(probs)) if probs else base


def marginalized_conditional_home_win_prob_vec(
    mu_home,
    mu_away,
    lineup_uncertainty_home=None,
    lineup_uncertainty_away=None,
    game_ids=None,
    n_samples=64,
    mu_noise_scale=0.12,
):
    """Vectorised `marginalized_conditional_home_win_prob` over many matches.

    Produces exactly the values the scalar version would, because the per-game
    multipliers still come from `rng_for_game(game_id, salt=2)`, but evaluates
    the Skellam conditional across all matches at once instead of one scalar
    call per draw. That is the difference between 64 array operations and
    tens of thousands of scipy calls, which is what makes marginalising
    affordable in train.py and evaluate.py rather than inference only.
    """
    mu_home = np.maximum(np.asarray(mu_home, dtype=float), 1e-9)
    mu_away = np.maximum(np.asarray(mu_away, dtype=float), 1e-9)
    n_rows = mu_home.size
    base = conditional_home_win_prob_vec(mu_home, mu_away)

    n_samples = int(max(1, n_samples))
    mu_noise_scale = float(max(0.0, mu_noise_scale))
    if n_rows == 0 or n_samples <= 1 or mu_noise_scale <= 0:
        return base

    unc_home = np.maximum(
        np.nan_to_num(np.asarray(
            np.zeros(n_rows) if lineup_uncertainty_home is None
            else lineup_uncertainty_home, dtype=float
        )), 0.0,
    )
    unc_away = np.maximum(
        np.nan_to_num(np.asarray(
            np.zeros(n_rows) if lineup_uncertainty_away is None
            else lineup_uncertainty_away, dtype=float
        )), 0.0,
    )
    std_home = mu_noise_scale * np.sqrt(unc_home)
    std_away = mu_noise_scale * np.sqrt(unc_away)

    # Rows with no lineup uncertainty are an exact no-op, as in the scalar path.
    active = (std_home > 1e-9) | (std_away > 1e-9)
    if not active.any():
        return base

    if game_ids is None:
        game_ids = np.arange(n_rows)

    mult_home = np.ones((n_rows, n_samples), dtype=float)
    mult_away = np.ones((n_rows, n_samples), dtype=float)
    for i in np.flatnonzero(active):
        rng = rng_for_game(game_ids[i], salt=2)
        mult_home[i] = np.exp(
            rng.normal(-0.5 * std_home[i] ** 2, std_home[i], size=n_samples)
        )
        mult_away[i] = np.exp(
            rng.normal(-0.5 * std_away[i] ** 2, std_away[i], size=n_samples)
        )

    totals = np.zeros(n_rows, dtype=float)
    for draw in range(n_samples):
        totals += conditional_home_win_prob_vec(
            np.maximum(mu_home * mult_home[:, draw], 1e-6),
            np.maximum(mu_away * mult_away[:, draw], 1e-6),
        )
    marginalised = totals / float(n_samples)

    return np.where(active, marginalised, base)


def derive_market_home_probability(df: pd.DataFrame) -> np.ndarray:
    """Get market-implied home win probability without fabricating missing rows.

    Preference order: Shin-adjusted → power-normalised → basic-normalised → raw odds ratio.
    Shin (1993) adjustment is most principled as it corrects for insider-trading bias
    in the bookmaker overround. Raw paired prices remain the availability
    contract; unavailable rows return NaN, never a neutral-looking 50%.
    """
    p = pd.Series(np.nan, index=df.index)

    if "home_market_prob_shin" in df.columns:
        p = p.fillna(pd.to_numeric(df["home_market_prob_shin"], errors="coerce"))

    if "home_market_prob_power" in df.columns:
        p = p.fillna(pd.to_numeric(df["home_market_prob_power"], errors="coerce"))

    if "home_market_prob_basic" in df.columns:
        p = p.fillna(pd.to_numeric(df["home_market_prob_basic"], errors="coerce"))

    if (
        "team_head_to_head_odds_home" not in df.columns
        or "team_head_to_head_odds_away" not in df.columns
    ):
        return np.full(len(df), np.nan, dtype=float)

    home_odds = pd.to_numeric(
        df["team_head_to_head_odds_home"], errors="coerce"
    )
    away_odds = pd.to_numeric(
        df["team_head_to_head_odds_away"], errors="coerce"
    )
    valid = np.fromiter(
        (
            valid_decimal_odds(home_price)
            and valid_decimal_odds(away_price)
            for home_price, away_price in zip(home_odds, away_odds)
        ),
        dtype=bool,
        count=len(df),
    )
    if valid.any():
        qh = 1.0 / home_odds
        qa = 1.0 / away_odds
        p_basic = qh / (qh + qa)
        p = p.fillna(p_basic)
    p = p.where(valid, np.nan)

    values = pd.to_numeric(p, errors="coerce").to_numpy(dtype=float, copy=True)
    finite = np.isfinite(values)
    values[finite] = np.clip(values[finite], 1e-6, 1 - 1e-6)
    values[~finite] = np.nan
    return values


def draw_score_samples(
    mu_home,
    mu_away,
    n_simulations,
    lambda3=0.0,
    dispersion_home=None,
    dispersion_away=None,
    rng=None,
):
    """Draw correlated, optionally over-dispersed score pairs.

    Rugby-league points arrive in 2/4/6 lumps, so raw Poisson understates the
    margin variance; a per-side negative-binomial `k` (gamma-mixed Poisson,
    var = mu + mu^2/k) widens it. `lambda3` is a shared Poisson component that
    gives the two scores a positive covariance.

    Both apply at once. The independent part of each score carries the gamma
    mixing, with its dispersion rescaled by (lam/mu)^2 so the marginal variance
    still lands on mu + mu^2/k while the shared component holds the covariance
    at lambda3. Previously a non-zero lambda3 silently discarded the dispersion.
    """
    if rng is None:
        rng = np.random.default_rng()

    mu_home = float(max(mu_home, 1e-9))
    mu_away = float(max(mu_away, 1e-9))
    lambda3 = float(max(lambda3, 0.0))

    shared = min(lambda3, 0.95 * min(mu_home, mu_away)) if lambda3 > 0 else 0.0
    lam_home = max(mu_home - shared, 1e-9)
    lam_away = max(mu_away - shared, 1e-9)

    def _draw(lam, mu, dispersion):
        if dispersion is not None and np.isfinite(dispersion) and dispersion > 0:
            k = float(dispersion) * (lam / mu) ** 2
            rate = rng.gamma(shape=k, scale=lam / k, size=n_simulations)
            return rng.poisson(np.maximum(rate, 1e-12))
        return rng.poisson(lam, size=n_simulations)

    home_goals_sim = _draw(lam_home, mu_home, dispersion_home)
    away_goals_sim = _draw(lam_away, mu_away, dispersion_away)

    if shared > 0:
        shared_sim = rng.poisson(shared, size=n_simulations)
        home_goals_sim = home_goals_sim + shared_sim
        away_goals_sim = away_goals_sim + shared_sim

    return home_goals_sim, away_goals_sim


def scoreline_from_samples(home_sim, away_sim, tipped_home=None, display="median"):
    """Reduce a simulated score cloud to the one scoreline that gets displayed.

    Two reductions, because they are not equally good and the difference is
    measurable:

    * `"median"` takes the median margin and the median total, which are each
      the MAE-optimal point estimate of their own quantity, and splits the total
      around the margin. The total is nudged one point onto the parity the margin
      needs, toward whichever neighbour the simulation actually favours.
    * `"mode"` takes the most common exact scoreline on the tipped side. It is a
      mode of a two-dimensional discrete distribution, so it carries far more
      sampling noise than either median, and it is retained only so the two can
      be compared.

    `tipped_home` fixes the side the displayed margin must fall on. A margin of
    zero, or one whose sign contradicts the tip, is pushed to a single point in
    the tipped direction: the scoreline and the tip are one prediction and must
    never disagree. Both cases only arise inside the near-tie band.
    """
    home_sim = np.asarray(home_sim)
    away_sim = np.asarray(away_sim)
    margins = home_sim - away_sim

    if tipped_home is None:
        home_wins = int((margins > 0).sum())
        away_wins = int((margins < 0).sum())
        if home_wins == away_wins:
            tipped_home = None
        else:
            tipped_home = home_wins > away_wins

    if display == "mode":
        if tipped_home is None:
            return tuple(int(v) for v in Counter(zip(home_sim, away_sim)).most_common(1)[0][0])
        tip_mask = margins > 0 if tipped_home else margins < 0
        if tip_mask.any():
            modal = Counter(zip(home_sim[tip_mask], away_sim[tip_mask])).most_common(1)[0][0]
            return int(modal[0]), int(modal[1])
        # Unreachable once the means agree with the tip, but a degenerate call
        # must still return an ordered scoreline rather than raise.
        modal = Counter(zip(home_sim, away_sim)).most_common(1)[0][0]
        ordered = (max(modal), min(modal)) if tipped_home else (min(modal), max(modal))
        if ordered[0] == ordered[1]:
            ordered = (
                (ordered[0] + 1, ordered[1]) if tipped_home else (ordered[0], ordered[1] + 1)
            )
        return int(ordered[0]), int(ordered[1])

    if display != "median":
        raise ValueError(f"unknown display mode: {display!r}")

    margin = int(round(float(np.median(margins))))
    if tipped_home is not None and (
        (tipped_home and margin <= 0) or (not tipped_home and margin >= 0)
    ):
        margin = 1 if tipped_home else -1

    totals = home_sim + away_sim
    total = int(round(float(np.median(totals))))
    if (total + margin) % 2:
        up = int((totals == total + 1).sum())
        down = int((totals == total - 1).sum())
        total = total + 1 if up >= down else total - 1

    home = (total + margin) // 2
    away = total - home
    # Clamping preserves the margin, which is the headline number.
    if away < 0:
        home, away = margin, 0
    if home < 0:
        home, away = 0, -margin
    return int(home), int(away)


def solve_score_means_for_probability(mu_home, mu_away, target_cond, min_mean=1e-3):
    """Shift the score means so their own win probability equals the target.

    The total `mu_home + mu_away` is held fixed and the difference is moved,
    because p(home win | non-draw) under Skellam is monotone in the mean
    difference at fixed total. Bisection therefore converges cleanly.

    This is what makes the calibrated probability and the score distribution a
    single object. The alternative, reconciling them after the fact with
    importance weights, distorts the shape of the distribution it corrects and
    needs a special case for calibration tipping a side the simulation never
    produced; moving the means removes both problems by construction.
    """
    mu_home = float(mu_home)
    mu_away = float(mu_away)
    total = mu_home + mu_away
    target = float(np.clip(target_cond, 1e-9, 1.0 - 1e-9))
    if not np.isfinite(total) or total <= 2.0 * min_mean:
        return mu_home, mu_away

    limit = total - 2.0 * min_mean

    def gap(diff):
        return (
            conditional_home_win_prob((total + diff) / 2.0, (total - diff) / 2.0)
            - target
        )

    # Clamp when the target sits outside what positive means can express.
    if gap(limit) <= 0.0:
        diff = limit
    elif gap(-limit) >= 0.0:
        diff = -limit
    else:
        try:
            diff = float(brentq(gap, -limit, limit, xtol=1e-9, maxiter=200))
        except (ValueError, RuntimeError):
            return mu_home, mu_away

    return (total + diff) / 2.0, (total - diff) / 2.0


def simulate_game(
    home_score_avg,
    away_score_avg,
    n_simulations=100000,
    lambda3=0.0,
    rng=None,
    calibrated_cond=None,
    dispersion_home=None,
    dispersion_away=None,
    reconcile="on_conflict",
    display="median",
):
    """Simulate outcomes and scoreline under Poisson-family score models.

    `calibrated_cond` is the calibrated p(home win | non-draw) that decides the
    tip. `reconcile` says when the score means are moved onto it:

    * `"on_conflict"` moves them only when the score model would otherwise put
      the wrong side in front. The requirement the scoreline has to meet is that
      it never contradicts the tip, and on most games it already does. Held-out
      seasons say the score model is the better margin estimator, so leaving it
      alone where there is nothing to fix is worth roughly a third of a point of
      margin error.
    * `"always"` moves them on every game, so the simulated win probability
      equals the calibrated one everywhere. Stronger coherence, measurably worse
      margins, since it imports the tip model's view onto games where the two
      already agreed.

    Either way the means move along a total-preserving ray, so the margin and
    scoreline fall out of one distribution rather than being reweighted onto the
    calibrated probability afterwards.

    `display` picks how the cloud is reduced to a scoreline; see
    `scoreline_from_samples`.
    """
    if rng is None:
        rng = np.random.default_rng()

    if reconcile not in {"always", "on_conflict"}:
        raise ValueError(f"unknown reconcile mode: {reconcile!r}")

    mu_home = float(max(home_score_avg, 1e-9))
    mu_away = float(max(away_score_avg, 1e-9))

    tipped_home = None
    reconciled = False
    if calibrated_cond is not None and np.isfinite(calibrated_cond):
        # Strict > matches the caller's tie-break: cal == 0.5 tips the away side.
        cal = float(np.clip(calibrated_cond, 1e-6, 1 - 1e-6))
        tipped_home = cal > 0.5
        # Analytic, so the conflict test uses the same quantity the solver
        # targets and costs nothing.
        conflict = (conditional_home_win_prob(mu_home, mu_away) > 0.5) != tipped_home
        if reconcile == "always" or conflict:
            mu_home, mu_away = solve_score_means_for_probability(mu_home, mu_away, cal)
            reconciled = True

    home_goals_sim, away_goals_sim = draw_score_samples(
        mu_home,
        mu_away,
        n_simulations,
        lambda3=lambda3,
        dispersion_home=dispersion_home,
        dispersion_away=dispersion_away,
        rng=rng,
    )

    margins = home_goals_sim - away_goals_sim
    home_wins = int((margins > 0).sum())
    away_wins = int((margins < 0).sum())
    draws = int((margins == 0).sum())

    # A tie after eighty minutes is not a drawn game: golden point resolves
    # nearly all of them. Send the tied mass to extra time, split by the
    # game's own non-draw strength, and keep only the small share that
    # survives it.
    home_wins_eff = float(home_wins)
    away_wins_eff = float(away_wins)
    draws_eff = float(draws)
    if draws:
        unresolved = draws * GOLDEN_POINT_UNRESOLVED_SHARE
        resolved = draws - unresolved
        decided = home_wins + away_wins
        p_home_extra_time = (home_wins / decided) if decided else 0.5
        home_wins_eff += resolved * p_home_extra_time
        away_wins_eff += resolved * (1.0 - p_home_extra_time)
        draws_eff = unresolved

    total_games = float(n_simulations)
    probabilities = {
        "home_win_prob": home_wins_eff / total_games,
        "away_win_prob": away_wins_eff / total_games,
        "draw_prob": draws_eff / total_games,
        # Median margin is a far more stable point estimate than the margin of
        # the modal exact scoreline.
        "median_margin": int(round(float(np.median(margins)))),
        "reconciled": reconciled,
    }

    predicted_scoreline = scoreline_from_samples(
        home_goals_sim, away_goals_sim, tipped_home=tipped_home, display=display
    )
    return probabilities, predicted_scoreline


def calculate_bayes_factor(probabilities):
    """Posterior odds in favour of the tipped side, clipped to stay finite.

    Historically this was home/away (so away tips read as "negative evidence"
    and a zero away prob produced inf); it is now symmetric in the tip.
    """
    home = max(float(probabilities["home_win_prob"]), 1e-9)
    away = max(float(probabilities["away_win_prob"]), 1e-9)
    return float(min(max(home, away) / min(home, away), 999.0))


def map_bayes_factor_to_evidence(bayes_factor):
    """Plain confidence wording for the tipped side's posterior odds."""
    if bayes_factor < 1.5:
        return "Coin flip"
    if bayes_factor < 2.5:
        return "Slight lean"
    if bayes_factor < 4.0:
        return "Confident"
    if bayes_factor < 9.0:
        return "Strong"
    return "Near lock"


_DIAGNOSTIC_COLUMNS = [
    "game_id",
    "reconciled",
    "mu_home_used",
    "mu_away_used",
    "sim_draw_prob",
]


def predict_match_outcome_and_scoreline_with_bayes(
    home_model=None,
    away_model=None,
    inference_data=None,
    predictors=None,
    n_simulations=100000,
    mu_home=None,
    mu_away=None,
    lambda3=0.0,
    calibrated_home_win_conditional=None,
    dispersion_home=None,
    dispersion_away=None,
    reconcile="on_conflict",
    display="median",
    return_diagnostics=False,
):
    """
    Predict match outcomes and scorelines.

    Backward compatible mode:
    - pass home_model/away_model/inference_data/predictors.

    Enhanced mode:
    - pass inference_data with precomputed mu_home/mu_away arrays and optional
      calibrated_home_win_conditional. Market line and totals information must
      be applied to those score means before calling this function.

    return_diagnostics adds a third frame describing how each scoreline was
    produced (whether reconciliation moved the score means, and the means
    actually simulated). It is opt-in and reads values out of the simulation
    that already happened: it never re-simulates, so the tip cannot move.
    """
    if inference_data is None:
        raise ValueError("inference_data is required.")

    if inference_data.empty:
        empty_outcomes = pd.DataFrame(
            columns=[
                "game_id",
                "home_team_result",
                "home_team_win_prob",
                "home_team_lose_prob",
                "draw_prob",
                "bayes_factor",
                "evidence_strength",
            ]
        )
        empty_margins = pd.DataFrame(
            columns=["game_id", "predicted_home_score", "predicted_away_score", "predicted_margin"]
        )
        if return_diagnostics:
            return empty_outcomes, empty_margins, pd.DataFrame(columns=_DIAGNOSTIC_COLUMNS)
        return empty_outcomes, empty_margins

    working = inference_data.copy().reset_index(drop=True)

    if mu_home is None or mu_away is None:
        if home_model is None or away_model is None or predictors is None:
            raise ValueError("Need either precomputed mu arrays or models + predictors.")
        working["home_goals_avg"] = predict_scores(home_model, working[predictors])
        working["away_goals_avg"] = predict_scores(away_model, working[predictors])
    else:
        working["home_goals_avg"] = np.asarray(mu_home, dtype=float)
        working["away_goals_avg"] = np.asarray(mu_away, dtype=float)

    if calibrated_home_win_conditional is None:
        calibrated_home_win_conditional = np.full(len(working), np.nan)
    calibrated_home_win_conditional = np.asarray(calibrated_home_win_conditional, dtype=float)

    results = []
    for idx, row in working.iterrows():
        calibrated_cond = calibrated_home_win_conditional[idx]
        if not np.isnan(calibrated_cond):
            calibrated_cond = float(np.clip(calibrated_cond, 1e-6, 1 - 1e-6))

        # Per-game deterministic RNG: identical inputs always yield the same
        # tip, scoreline, and margin across re-runs.
        rng = rng_for_game(row.get("game_id"), salt=1)
        probabilities, predicted_scoreline = simulate_game(
            row["home_goals_avg"],
            row["away_goals_avg"],
            n_simulations=n_simulations,
            lambda3=lambda3,
            rng=rng,
            calibrated_cond=None if np.isnan(calibrated_cond) else calibrated_cond,
            dispersion_home=dispersion_home,
            dispersion_away=dispersion_away,
            reconcile=reconcile,
            display=display,
        )

        if not np.isnan(calibrated_cond):
            non_draw = max(0.0, 1.0 - probabilities["draw_prob"])
            probabilities["home_win_prob"] = calibrated_cond * non_draw
            probabilities["away_win_prob"] = (1.0 - calibrated_cond) * non_draw

        home_team_result = "Win" if probabilities["home_win_prob"] > probabilities["away_win_prob"] else "Loss"
        bayes_factor = calculate_bayes_factor(probabilities)
        evidence_strength = map_bayes_factor_to_evidence(bayes_factor)

        # Scoreline and margin are one public prediction.  A separate median or
        # post-simulation line override can otherwise render contradictions
        # such as "17-14" alongside "by 1".  Market margin information should
        # influence the score means before simulation; persistence is always
        # the displayed score difference.
        predicted_margin = int(predicted_scoreline[0] - predicted_scoreline[1])

        results.append(
            {
                "game_id": row["game_id"],
                "home_team_win_prob": probabilities["home_win_prob"],
                "home_team_lose_prob": probabilities["away_win_prob"],
                "draw_prob": probabilities["draw_prob"],
                "predicted_home_score": predicted_scoreline[0],
                "predicted_away_score": predicted_scoreline[1],
                "predicted_margin": predicted_margin,
                "home_team_result": home_team_result,
                "bayes_factor": bayes_factor,
                "evidence_strength": evidence_strength,
                # Diagnostics. outcome_df/margin_df are built by explicit
                # column selection below, so these are invisible to callers
                # that did not ask for them.
                "reconciled": bool(probabilities.get("reconciled", False)),
                "mu_home_used": float(row["home_goals_avg"]),
                "mu_away_used": float(row["away_goals_avg"]),
                "sim_draw_prob": float(probabilities["draw_prob"]),
            }
        )

    results_df = pd.DataFrame(results)
    outcome_df = results_df[
        [
            "game_id",
            "home_team_result",
            "home_team_win_prob",
            "home_team_lose_prob",
            "draw_prob",
            "bayes_factor",
            "evidence_strength",
        ]
    ]
    margin_df = results_df[["game_id", "predicted_home_score", "predicted_away_score", "predicted_margin"]]
    if return_diagnostics:
        return outcome_df, margin_df, results_df[_DIAGNOSTIC_COLUMNS]
    return outcome_df, margin_df


def get_predictions(db_path, sql_file):
    con = sqlite3.connect(str(db_path))
    with open(sql_file, "r") as file:
        query = file.read()
    predictions = pd.read_sql_query(query, con)
    con.close()
    return predictions


def load_models(model_name, project_root, models_dir=None):
    model_path = (
        pathlib.Path(models_dir) if models_dir is not None else project_root / "models"
    ) / f"{model_name}.pkl"
    try:
        with open(model_path, "rb") as f:
            pipeline = pickle.load(f)
        print(f"{model_name} model pipeline loaded successfully.")
    except Exception as e:
        print(f"Error loading {model_name} model: {e}")
        raise
    return pipeline


def _ensure_prediction_table_columns(con):
    expected_columns = {
        "draw_prob": "REAL",
        "bayes_factor": "REAL",
        "evidence_strength": "TEXT",
        "predicted_home_score": "INTEGER",
        "predicted_away_score": "INTEGER",
        "predicted_margin": "INTEGER",
    }
    existing_columns = {row[1] for row in con.execute("PRAGMA table_info(predictions_table)").fetchall()}
    for column_name, column_ddl in expected_columns.items():
        if column_name not in existing_columns:
            con.execute(f"ALTER TABLE predictions_table ADD COLUMN {column_name} {column_ddl}")


def save_predictions_to_db(predictions_df, db_path, create_table_sql_file, insert_into_table_sql_file):
    print("Saving predictions to database...")
    con = sqlite3.connect(str(db_path))

    with open(create_table_sql_file, "r") as file:
        create_table_query = file.read()
    con.execute(create_table_query)
    _ensure_prediction_table_columns(con)

    with open(insert_into_table_sql_file, "r") as file:
        insert_into_table_query = file.read()

    for _, row in predictions_df.iterrows():
        con.execute(
            insert_into_table_query,
            (
                row["game_id"],
                row["home_team_result"],
                row["home_team_win_prob"],
                row["home_team_lose_prob"],
                row["draw_prob"],
                row["bayes_factor"],
                row["evidence_strength"],
                row["predicted_home_score"],
                row["predicted_away_score"],
                row["predicted_margin"],
            ),
        )

    con.commit()
    con.close()
