"""Convert link-scale contributions into units a human can act on.

LightGBM hands back contributions on the link scale: log-odds for the binary
classifier, log-mean for the Poisson score models. Neither is readable. These
helpers map them to probability points and points of margin.

The link is nonlinear, so linearised points do NOT sum to (prediction - base).
Every surface therefore reports two numbers and says which is which:

* ``points``  linearised magnitude. Physically meaningful and comparable across
  games, but only approximately additive.
* ``share``   signed fraction of the model's total absolute departure from its
  base value. Exact by construction, sums to +/-1, safe for ranking.
"""

from __future__ import annotations

import numpy as np

# Probabilities are reported in percentage points, the unit tips are read in.
PCT_SCALE = 100.0


def prob_points(delta_logit, p_ref):
    """Probability points from a log-odds contribution, evaluated at p_ref.

    First-order: dp = p(1-p) dz. Multiply by 100 for percentage points.
    """
    delta_logit = np.asarray(delta_logit, dtype=float)
    p_ref = np.asarray(p_ref, dtype=float)
    return p_ref * (1.0 - p_ref) * delta_logit


def score_points(delta_log_mu, mu_ref):
    """Points of expected score from a log-mean contribution: d(mu) = mu d(log mu)."""
    return np.asarray(mu_ref, dtype=float) * np.asarray(delta_log_mu, dtype=float)


def margin_points(contrib_home, mu_home, contrib_away, mu_away):
    """Points of expected margin from paired home/away log-mean contributions.

    d(mu_h - mu_a) = mu_h * d(log mu_h) - mu_a * d(log mu_a). A feature that
    lifts both scores equally moves the total, not the margin, and correctly
    nets out near zero here.
    """
    return score_points(contrib_home, mu_home) - score_points(contrib_away, mu_away)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype=float), -700.0, 700.0)))


def logit(p, eps=1e-12):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))
