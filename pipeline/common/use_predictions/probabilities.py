"""One convention for every published probability: two-way, draws excluded.

`predictions_table` stores a coherent three-way triple (home win, away win,
draw).  The stored win probabilities are the calibrated conditional scaled by
`(1 - draw_prob)`, so reading either of them on its own understates the tipped
side by the draw mass.

Every reader-facing number -- confidence badges, fair prices, edges, prompt
text -- uses the draw-excluded conditional `win / (win + lose)` instead:

* it recovers `calibrated_cond`, which is the quantity the model is actually
  fitted, calibrated and scored on;
* it matches `derive_market_home_probability`, which already returns a two-way
  normalised price, so model and market are compared in the same units;
* the simulated `draw_prob` averages 3.9% while the real NRL draw rate is 0.32%
  (14 in 3577 games), because the simulation prices an 80-minute tie and golden
  point resolves nearly all of them.  Deflating a calibrated probability by an
  uncalibrated quantity roughly twelve times too large is worse than either
  alternative.

`comp_strategy` and `joker` have always done this internally; this module is
that same rule, applied everywhere.
"""

import numpy as np
import pandas as pd

HOME_WIN_COLUMN = "home_team_win_prob"
AWAY_WIN_COLUMN = "home_team_lose_prob"


def two_way_home_probability(win, lose):
    """P(home wins | not a draw), for scalars or Series alike.

    Returns NaN wherever the pair is missing or does not sum to something
    positive.  A missing probability is never silently replaced with 0.5 --
    callers that need a neutral default must say so themselves.
    """
    if isinstance(win, pd.Series) or isinstance(lose, pd.Series):
        win_values = pd.to_numeric(win, errors="coerce")
        lose_values = pd.to_numeric(lose, errors="coerce")
        denominator = win_values + lose_values
        return (win_values / denominator).where(denominator > 0)

    win_value = pd.to_numeric(pd.Series([win]), errors="coerce").iloc[0]
    lose_value = pd.to_numeric(pd.Series([lose]), errors="coerce").iloc[0]
    denominator = win_value + lose_value
    if pd.isna(denominator) or denominator <= 0:
        return np.nan
    return float(win_value) / float(denominator)


def home_probability(row):
    """Two-way home win probability for one prediction row."""
    return two_way_home_probability(row.get(HOME_WIN_COLUMN), row.get(AWAY_WIN_COLUMN))


def tip_probability(row):
    """Confidence in the side the model actually tipped, draws excluded."""
    home_prob = home_probability(row)
    if pd.isna(home_prob):
        return np.nan
    if row.get("home_team_result") == "Win":
        return home_prob
    return 1.0 - home_prob


def tipped_home(win, lose):
    """Whether the home side is tipped, from the probabilities alone.

    Compares the two sides directly rather than testing the stored home
    probability against 0.5: because both sides carry the same `(1 - draw)`
    factor, `win > lose` is the same predicate the tip was written with, while
    `win > 0.5` mis-attributes games whose conditional sits just above a half.
    """
    win_values = pd.to_numeric(win, errors="coerce")
    lose_values = pd.to_numeric(lose, errors="coerce")
    return win_values > lose_values
