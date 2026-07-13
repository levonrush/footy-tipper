import unittest

import numpy as np
import pandas as pd

from pipeline.common.model_training import calibration as calib


class MarketExtraFeatureTests(unittest.TestCase):
    def test_width_matches_names(self):
        df = pd.DataFrame({"anything": [1.0, 2.0]})
        X = calib.build_line_market_features(df, np.array([3.0, -2.0]))
        self.assertEqual(X.shape, (2, len(calib.LINE_MARKET_FEATURE_NAMES)))
        self.assertEqual(len(calib.LINE_MARKET_FEATURE_NAMES), 8)

    def test_soft_fail_on_empty_columns(self):
        df = pd.DataFrame({"anything": [1.0]})
        X = calib.build_line_market_features(df, np.array([np.nan]))
        names = calib.LINE_MARKET_FEATURE_NAMES
        row = dict(zip(names, X[0]))
        self.assertEqual(row["line_cover_logit"], 0.0)
        self.assertEqual(row["line_missing"], 1.0)
        self.assertEqual(row["h2h_move_logit"], 0.0)
        self.assertEqual(row["total_line_centered"], 0.0)
        self.assertEqual(row["totals_missing"], 1.0)

    def test_new_inputs_flow_through(self):
        df = pd.DataFrame(
            {
                "home_line_cover_prob_shin": [0.55],
                "line_overround_basic": [1.05],
                "implied_spread_home": [-4.5],
                "h2h_move_logit": [0.3],
                "line_move_points": [2.0],
                "market_total_line": [45.0],
            }
        )
        X = calib.build_line_market_features(df, np.array([6.0]))
        row = dict(zip(calib.LINE_MARKET_FEATURE_NAMES, X[0]))
        self.assertGreater(row["line_cover_logit"], 0.0)
        self.assertEqual(row["line_missing"], 0.0)
        self.assertAlmostEqual(row["h2h_move_logit"], 0.3)
        self.assertAlmostEqual(row["line_move_points"], 0.2)  # scaled by 10
        self.assertAlmostEqual(row["total_line_centered"], 0.5)  # (45-40)/10
        self.assertEqual(row["totals_missing"], 0.0)

    def test_movement_clipped(self):
        df = pd.DataFrame({"h2h_move_logit": [5.0], "line_move_points": [-40.0]})
        X = calib.build_line_market_features(df, np.array([0.0]))
        row = dict(zip(calib.LINE_MARKET_FEATURE_NAMES, X[0]))
        self.assertEqual(row["h2h_move_logit"], 2.0)
        self.assertEqual(row["line_move_points"], -1.2)

    def test_stacker_roundtrip_with_wider_extra(self):
        rng = np.random.default_rng(7)
        n = 400
        y = rng.integers(0, 2, n)
        base = np.clip(0.5 + (y - 0.5) * 0.2 + rng.normal(0, 0.1, n), 0.05, 0.95)
        extra = rng.normal(0, 0.5, (n, len(calib.LINE_MARKET_FEATURE_NAMES)))
        groups = np.repeat(np.arange(4), n // 4)

        stacker = calib.LogisticStacker()
        stacker.fit(
            tier_a=base,
            tier_b=base,
            market=base,
            odds_missing=np.zeros(n),
            y=y,
            groups=groups,
            extra=extra,
        )
        preds = stacker.predict(
            base, base, base, np.zeros(n), extra=extra
        )
        self.assertEqual(len(preds), n)
        self.assertTrue(np.all((preds > 0) & (preds < 1)))


if __name__ == "__main__":
    unittest.main()
