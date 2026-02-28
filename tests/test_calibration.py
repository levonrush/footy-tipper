import unittest
import warnings

import numpy as np

from pipeline.common.model_training.calibration import BetaCalibrator, LogisticStacker


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


if __name__ == "__main__":
    unittest.main()
