import unittest
import warnings

import numpy as np

from pipeline.common.model_training.calibration import (
    BetaCalibrator,
    LogisticStacker,
    loso_stacker_predictions,
)


class CalibrationTests(unittest.TestCase):
    def test_beta_calibrator_handles_extreme_probabilities_without_runtime_warning(self):
        calibrator = BetaCalibrator()
        probs = np.array([0.0, 1.0, 1e-12, 1 - 1e-12, 0.25, 0.75])
        y = np.array([0, 1, 0, 1, 0, 1])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            calibrator.fit(probs, y)
            preds = calibrator.predict(probs)

        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(runtime_warnings, [])
        self.assertTrue(np.isfinite(preds).all())
        self.assertTrue(((preds >= 0.0) & (preds <= 1.0)).all())

    def test_logistic_stacker_handles_non_finite_inputs(self):
        stacker = LogisticStacker()
        tier_a = np.array([0.0, 1.0, np.nan, 0.6, 0.4, 1e-12])
        tier_b = np.array([1.0, 0.0, 0.55, np.inf, 0.3, 1 - 1e-12])
        market = np.array([0.5, np.nan, 0.52, 0.48, np.inf, 0.1])
        odds_missing = np.array([0, 1, 0, np.nan, 0, 1])
        y = np.array([1, 0, 1, 0, 1, 0])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stacker.fit(tier_a, tier_b, market, odds_missing, y)
            preds = stacker.predict(tier_a, tier_b, market, odds_missing)

        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(runtime_warnings, [])
        self.assertTrue(np.isfinite(preds).all())
        self.assertTrue(((preds >= 0.0) & (preds <= 1.0)).all())

    def test_logistic_stacker_grouped_cv_fit(self):
        rng = np.random.default_rng(11)
        n = 240
        tier_a = rng.uniform(0.2, 0.8, n)
        tier_b = np.clip(tier_a + rng.normal(0, 0.05, n), 0.01, 0.99)
        market = np.clip(tier_a + rng.normal(0, 0.05, n), 0.01, 0.99)
        odds_missing = np.zeros(n)
        y = (rng.uniform(0, 1, n) < tier_a).astype(int)
        groups = np.repeat(np.arange(2018, 2026), n // 8)

        stacker = LogisticStacker()
        stacker.fit(tier_a, tier_b, market, odds_missing, y, groups=groups)
        self.assertTrue(stacker._is_fitted)
        preds = stacker.predict(tier_a, tier_b, market, odds_missing)
        self.assertTrue(((preds >= 0.0) & (preds <= 1.0)).all())


class LosoStackerPredictionTests(unittest.TestCase):
    def _synthetic(self, n_per_season=120, seasons=(2021, 2022, 2023, 2024), seed=5):
        rng = np.random.default_rng(seed)
        n = n_per_season * len(seasons)
        tier_a = rng.uniform(0.2, 0.8, n)
        tier_b = np.clip(tier_a + rng.normal(0, 0.05, n), 0.01, 0.99)
        market = np.clip(tier_a + rng.normal(0, 0.05, n), 0.01, 0.99)
        odds_missing = np.zeros(n)
        y = (rng.uniform(0, 1, n) < tier_a).astype(int)
        groups = np.repeat(np.array(seasons, dtype=float), n_per_season)
        return tier_a, tier_b, market, odds_missing, y, groups

    def test_returns_finite_out_of_sample_predictions(self):
        tier_a, tier_b, market, odds_missing, y, groups = self._synthetic()
        preds = loso_stacker_predictions(tier_a, tier_b, market, odds_missing, y, groups)
        self.assertIsNotNone(preds)
        self.assertTrue(np.isfinite(preds).all())
        self.assertTrue(((preds >= 0.0) & (preds <= 1.0)).all())

    def test_requires_three_season_groups(self):
        tier_a, tier_b, market, odds_missing, y, groups = self._synthetic(seasons=(2023, 2024))
        preds = loso_stacker_predictions(tier_a, tier_b, market, odds_missing, y, groups)
        self.assertIsNone(preds)

    def test_held_out_season_labels_do_not_change_its_predictions(self):
        tier_a, tier_b, market, odds_missing, y, groups = self._synthetic()
        hold = groups == 2024.0
        preds_a = loso_stacker_predictions(tier_a, tier_b, market, odds_missing, y, groups)
        y_flipped = y.copy()
        y_flipped[hold] = 1 - y_flipped[hold]
        preds_b = loso_stacker_predictions(tier_a, tier_b, market, odds_missing, y_flipped, groups)
        np.testing.assert_allclose(preds_a[hold], preds_b[hold])

    def test_differs_from_in_sample_stacker_predictions(self):
        tier_a, tier_b, market, odds_missing, y, groups = self._synthetic()
        stacker = LogisticStacker()
        stacker.fit(tier_a, tier_b, market, odds_missing, y, groups=groups)
        in_sample = stacker.predict(tier_a, tier_b, market, odds_missing)
        loso = loso_stacker_predictions(tier_a, tier_b, market, odds_missing, y, groups)
        self.assertGreater(float(np.max(np.abs(in_sample - loso))), 0.0)


if __name__ == "__main__":
    unittest.main()
