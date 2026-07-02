# Description: Simulate tipping-competition placement over a realized season.
#
# Given per-game model probabilities, market probabilities, and actual
# outcomes for games that have already been played, score the model's tips
# against a synthetic field of rivals who tip the market favourite with
# per-rival skill noise. This turns raw tipping accuracy into the metric
# that actually decides a comp: P(finish first) and expected rank.

import numpy as np

DEFAULT_FIELD_SIZE = 75
DEFAULT_N_SIMS = 4000
# Rivals tip the market favourite but flip to the underdog at a per-rival
# rate drawn once per rival: some rivals are sharp, some are chaotic.
DEFAULT_FLIP_RATE_MEAN = 0.12
DEFAULT_FLIP_RATE_SIGMA = 0.06
COMP_SIM_SEED = 20100308


def simulate_comp_placement(
    model_p,
    market_p,
    outcomes,
    tips=None,
    field_size=DEFAULT_FIELD_SIZE,
    n_sims=DEFAULT_N_SIMS,
    flip_rate_mean=DEFAULT_FLIP_RATE_MEAN,
    flip_rate_sigma=DEFAULT_FLIP_RATE_SIGMA,
    seed=COMP_SIM_SEED,
):
    """Score tips (default: model_p > 0.5) against a simulated rival field.

    The season is realized: outcomes are fixed, so the user's score is
    deterministic and only the rivals' tips are random. Rivals tip the
    market favourite per game and flip to the underdog with a per-rival
    flip rate; games with no clear favourite are coin-flips for them.

    Ties at the top count as half a win (the comp's margin tie-breaker is
    modelled as a coin-flip here). Returns None for empty input.
    """
    model_p = np.asarray(model_p, dtype=float)
    market_p = np.asarray(market_p, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)

    valid = np.isfinite(model_p) & np.isfinite(outcomes)
    model_p = model_p[valid]
    market_p = market_p[valid]
    outcomes = outcomes[valid].astype(int)
    if model_p.size == 0:
        return None

    if tips is None:
        tips = model_p > 0.5
    else:
        tips = np.asarray(tips, dtype=bool)[valid]
    user_score = int((tips == outcomes.astype(bool)).sum())

    has_favourite = np.isfinite(market_p) & (market_p != 0.5)
    favourite_tip = market_p > 0.5
    fav_correct = int((favourite_tip[has_favourite] == outcomes[has_favourite].astype(bool)).sum())
    fav_wrong = int(has_favourite.sum()) - fav_correct
    n_unclear = int((~has_favourite).sum())

    rng = np.random.default_rng(seed)
    flip_rates = np.clip(
        rng.normal(flip_rate_mean, flip_rate_sigma, size=(n_sims, field_size)), 0.0, 0.5
    )
    # Rival is correct on a favourite-won game unless they flipped, and
    # correct on an upset game only if they flipped.
    rival_scores = (
        rng.binomial(fav_correct, 1.0 - flip_rates)
        + rng.binomial(fav_wrong, flip_rates)
        + rng.binomial(n_unclear, 0.5, size=(n_sims, field_size))
    )

    best_rival = rival_scores.max(axis=1)
    strict_win = user_score > best_rival
    tie_top = user_score == best_rival
    beaten_by = (rival_scores > user_score).sum(axis=1)
    tied_with = (rival_scores == user_score).sum(axis=1)

    return {
        "games": int(model_p.size),
        "user_score": user_score,
        "field_size": int(field_size),
        "n_sims": int(n_sims),
        "market_favourite_score": fav_correct,
        "p_first": float(np.mean(strict_win + 0.5 * tie_top)),
        "p_top": float(np.mean(strict_win | tie_top)),
        "expected_rank": float(1.0 + np.mean(beaten_by + 0.5 * tied_with)),
        "mean_rival_score": float(rival_scores.mean()),
        "mean_best_rival_score": float(best_rival.mean()),
    }
