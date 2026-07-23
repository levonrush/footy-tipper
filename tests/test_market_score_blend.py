import unittest

import numpy as np
import pandas as pd

from pipeline.common.model_prediciton import prediction_functions as pf
from pipeline.common.model_prediciton.market_score_blend import (
    apply_market_score_mean_blends,
)


MARGIN_MARKET_ONLY = {
    "intercept": 0.0,
    "coef_model_margin": 0.0,
    "coef_market_spread": 1.0,
    "coef_tier_a_margin": 0.0,
}
TOTAL_MARKET_ONLY = {
    "intercept": 0.0,
    "coef_model_total": 0.0,
    "coef_market_total": 1.0,
}


class MarketScoreBlendTests(unittest.TestCase):
    def _frame(self):
        return pd.DataFrame(
            {
                "implied_spread_home": [20.0, -20.0, 0.0, 0.0],
                "team_line_odds_home": [1.91, 1.91, 0.0, 1.91],
                "team_line_odds_away": [1.91, 1.91, 0.0, 1.91],
                "line_odds_missing": [0.0, 0.0, 1.0, 0.0],
                "market_total_line": [40.0, 40.0, 60.0, 50.0],
                "total_over_odds": [1.91, 1.91, 0.0, 1.91],
                "total_under_odds": [1.91, 1.91, 0.0, 1.91],
                "totals_missing": [0.0, 0.0, 1.0, 0.0],
            }
        )

    def test_fresh_opposite_spreads_move_identical_means_before_simulation(self):
        frame = self._frame().iloc[:2].reset_index(drop=True)

        home, away, diagnostics = apply_market_score_mean_blends(
            frame,
            [20.0, 20.0],
            [20.0, 20.0],
            [20.0, 20.0],
            [20.0, 20.0],
            fresh_market=[True, True],
            margin_blend=MARGIN_MARKET_ONLY,
        )

        np.testing.assert_allclose(home, [10.0, 30.0])
        np.testing.assert_allclose(away, [30.0, 10.0])
        np.testing.assert_allclose(home + away, [40.0, 40.0])
        self.assertEqual(diagnostics["line_count"], 2)

        _, scorelines = pf.predict_match_outcome_and_scoreline_with_bayes(
            inference_data=pd.DataFrame({"game_id": [101, 102]}),
            mu_home=home,
            mu_away=away,
            calibrated_home_win_conditional=np.array([0.2, 0.8]),
            n_simulations=20_000,
        )
        margins = scorelines["predicted_margin"].to_numpy(dtype=int)
        self.assertLess(margins[0], 0)
        self.assertGreater(margins[1], 0)
        np.testing.assert_array_equal(
            margins,
            scorelines["predicted_home_score"].to_numpy(dtype=int)
            - scorelines["predicted_away_score"].to_numpy(dtype=int),
        )

    def test_stale_and_placeholder_markets_cannot_change_score_means(self):
        frame = self._frame().iloc[[0, 2]].reset_index(drop=True)

        home, away, diagnostics = apply_market_score_mean_blends(
            frame,
            [20.0, 20.0],
            [20.0, 20.0],
            [20.0, 20.0],
            [20.0, 20.0],
            fresh_market=[False, True],
            margin_blend=MARGIN_MARKET_ONLY,
            total_blend=TOTAL_MARKET_ONLY,
        )

        np.testing.assert_allclose(home, [20.0, 20.0])
        np.testing.assert_allclose(away, [20.0, 20.0])
        self.assertEqual(diagnostics["line_count"], 0)
        self.assertEqual(diagnostics["total_count"], 0)

    def test_fresh_h2h_does_not_make_old_line_or_total_fresh(self):
        frame = self._frame().iloc[[0]].reset_index(drop=True)

        home, away, diagnostics = apply_market_score_mean_blends(
            frame,
            [20.0],
            [20.0],
            [20.0],
            [20.0],
            fresh_market=[True],
            fresh_line_market=[False],
            fresh_total_market=[False],
            margin_blend=MARGIN_MARKET_ONLY,
            total_blend=TOTAL_MARKET_ONLY,
        )

        np.testing.assert_allclose(home, [20.0])
        np.testing.assert_allclose(away, [20.0])
        self.assertEqual(diagnostics["line_count"], 0)
        self.assertEqual(diagnostics["total_count"], 0)

    def test_valid_pickem_and_total_adjust_the_same_expected_score_pair(self):
        frame = self._frame().iloc[[3]].reset_index(drop=True)

        home, away, diagnostics = apply_market_score_mean_blends(
            frame,
            [20.0],
            [20.0],
            [20.0],
            [20.0],
            fresh_market=[True],
            margin_blend=MARGIN_MARKET_ONLY,
            total_blend=TOTAL_MARKET_ONLY,
        )

        np.testing.assert_allclose(home, [25.0])
        np.testing.assert_allclose(away, [25.0])
        self.assertEqual(diagnostics["line_count"], 1)
        self.assertEqual(diagnostics["total_count"], 1)


if __name__ == "__main__":
    unittest.main()
