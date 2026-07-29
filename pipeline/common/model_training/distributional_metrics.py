"""Proper scoring rules for sample-based predictive distributions.

The score models draw from a Poisson family for every match, but only three
integers ever reach the predictions table. Every scoring rule elsewhere in the
pipeline (log loss, Brier, Poisson deviance) scores the binary win probability,
so the score distribution itself, and with it `lambda3`, the negative-binomial
dispersion, and the market score blends, currently goes unmeasured.

These are the sample-based rules that close that gap:

* CRPS scores accuracy and honesty together, and reduces to MAE for a point
  forecast, which is what puts deterministic and probabilistic methods on one
  scale.
* Randomised PIT reports calibration shape. Margins are integers, so the plain
  PIT is not uniform even under a perfectly specified model; the randomised
  form is required to avoid manufacturing a false miscalibration finding.
* Coverage is only ever returned alongside width, so sharpness cannot be
  claimed without paying for it in calibration.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def crps_ensemble(samples, y):
    """CRPS of an empirical predictive distribution against a scalar outcome.

    Uses the estimator

        CRPS = mean|x_s - y| - (1 / 2S^2) * sum_s sum_s' |x_s - x_s'|

    evaluated through the sorted-sample identity

        sum_s sum_s' |x_s - x_s'| = 2 * sum_i (2i - S - 1) * x_(i)

    so the cost is the sort rather than the S^2 double sum. This matches the
    1/(2S^2) convention used in the thesis evaluation, not the 1/(S(S-1))
    unbiased variant, so numbers stay comparable across the two projects.
    """
    x = np.sort(np.asarray(samples, dtype=float))
    n_samples = x.size
    if n_samples == 0 or not np.isfinite(y):
        return float("nan")

    mean_abs_error = float(np.mean(np.abs(x - float(y))))
    if n_samples == 1:
        return mean_abs_error

    ranks = np.arange(1, n_samples + 1, dtype=float)
    spread = float(np.dot(2.0 * ranks - n_samples - 1.0, x)) / float(n_samples**2)
    return mean_abs_error - spread


def crps_normal(mu, sigma, y):
    """Closed-form CRPS for a Gaussian predictive distribution.

    Used for the normal-approximation baseline, where an exact value is both
    cheaper and less noisy than sampling.
    """
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0 or not np.isfinite(mu) or not np.isfinite(y):
        return float("nan")
    z = (float(y) - float(mu)) / sigma
    return float(
        sigma
        * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    )


def randomised_pit(samples, y, rng=None):
    """Randomised probability integral transform for a discrete outcome.

    For an integer-valued predictive distribution the ordinary PIT cannot be
    uniform, because F jumps at every attainable value. Randomising within the
    jump,

        u = F(y-) + v * (F(y) - F(y-)),   v ~ U(0, 1)

    restores uniformity under a correctly specified model (Czado et al.). A
    U-shaped histogram then means the distribution is too narrow, and a hump
    means it is too wide.
    """
    x = np.asarray(samples, dtype=float)
    if x.size == 0 or not np.isfinite(y):
        return float("nan")
    if rng is None:
        rng = np.random.default_rng()

    y = float(y)
    f_below = float(np.mean(x < y))
    f_at = float(np.mean(x <= y))
    return float(f_below + float(rng.random()) * (f_at - f_below))


def pit_histogram(pit_values, n_bins=10):
    """Bin PIT values onto [0, 1] and report the deviation from uniform.

    `uniformity_mae` is the mean absolute difference between observed and
    expected bin frequency, so 0 is perfectly calibrated and larger is worse.
    It gives the histogram a single number that a gate can read.
    """
    values = np.asarray(pit_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"bins": n_bins, "counts": [], "frequencies": [], "uniformity_mae": None}

    counts, _ = np.histogram(values, bins=n_bins, range=(0.0, 1.0))
    frequencies = counts / float(values.size)
    expected = 1.0 / float(n_bins)
    return {
        "bins": int(n_bins),
        "counts": [int(c) for c in counts],
        "frequencies": [float(f) for f in frequencies],
        "uniformity_mae": float(np.mean(np.abs(frequencies - expected))),
    }


def predictive_interval(samples, level=0.9):
    """Central predictive interval at the given level."""
    x = np.asarray(samples, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan")
    tail = (1.0 - float(level)) / 2.0
    return float(np.quantile(x, tail)), float(np.quantile(x, 1.0 - tail))


def interval_coverage(lowers, uppers, y, level=0.9):
    """Empirical coverage reported together with mean interval width.

    Coverage alone is trivially gamed by widening the interval, so the two are
    returned as one object and never separately.
    """
    lowers = np.asarray(lowers, dtype=float)
    uppers = np.asarray(uppers, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(lowers) & np.isfinite(uppers) & np.isfinite(y)
    if not valid.any():
        return {"level": float(level), "coverage": None, "width": None, "games": 0}

    inside = (y[valid] >= lowers[valid]) & (y[valid] <= uppers[valid])
    return {
        "level": float(level),
        "coverage": float(np.mean(inside)),
        "width": float(np.mean(uppers[valid] - lowers[valid])),
        "games": int(valid.sum()),
    }


def score_sample_forecasts(sample_sets, actuals, levels=(0.5, 0.9), rng=None):
    """Score a set of per-game sample draws against the realised outcomes.

    `sample_sets` is a sequence of 1-D sample arrays, one per game, and
    `actuals` the matching realised values. Returns the CRPS, the randomised
    PIT summary, and coverage-with-width at each requested level.
    """
    actuals = np.asarray(actuals, dtype=float)
    if rng is None:
        rng = np.random.default_rng()

    crps_values = []
    pit_values = []
    bounds = {level: ([], []) for level in levels}

    for samples, actual in zip(sample_sets, actuals):
        crps_values.append(crps_ensemble(samples, actual))
        pit_values.append(randomised_pit(samples, actual, rng=rng))
        for level in levels:
            low, high = predictive_interval(samples, level=level)
            bounds[level][0].append(low)
            bounds[level][1].append(high)

    crps_values = np.asarray(crps_values, dtype=float)
    finite = np.isfinite(crps_values)

    return {
        "games": int(finite.sum()),
        "crps": float(np.mean(crps_values[finite])) if finite.any() else None,
        "pit": pit_histogram(pit_values),
        "intervals": [
            interval_coverage(lows, highs, actuals, level=level)
            for level, (lows, highs) in bounds.items()
        ],
    }
