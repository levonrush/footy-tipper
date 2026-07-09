import unittest

import numpy as np

try:
    from pipeline.common.model_training import modelling_functions as mf
    HAS_MODEL_DEPS = True
except ImportError:
    HAS_MODEL_DEPS = False


@unittest.skipUnless(HAS_MODEL_DEPS, "lightgbm/skopt not installed")
class BlendWeightSelectionTests(unittest.TestCase):
    def test_prefers_informative_model_over_flat_baseline(self):
        rng = np.random.default_rng(42)
        n = 400
        # True scoring rates differ per game; outcomes drawn from them.
        true_home = rng.uniform(14, 30, n)
        true_away = rng.uniform(14, 30, n)
        scores_home = rng.poisson(true_home)
        scores_away = rng.poisson(true_away)
        keep = scores_home != scores_away
        y = (scores_home[keep] > scores_away[keep]).astype(int)

        # Baseline knows nothing (flat league average); model knows the truth.
        flat = np.full(keep.sum(), 22.0)
        wh, wa, ll, acc = mf.select_blend_weights_by_log_loss(
            y, flat, flat, true_home[keep], true_away[keep]
        )
        self.assertGreaterEqual(wh, 0.8)
        self.assertGreaterEqual(wa, 0.8)
        self.assertGreater(acc, 0.6)

    def test_prefers_baseline_over_noise_model(self):
        rng = np.random.default_rng(7)
        n = 400
        true_home = rng.uniform(14, 30, n)
        true_away = rng.uniform(14, 30, n)
        scores_home = rng.poisson(true_home)
        scores_away = rng.poisson(true_away)
        keep = scores_home != scores_away
        y = (scores_home[keep] > scores_away[keep]).astype(int)

        noise_home = rng.uniform(14, 30, keep.sum())
        noise_away = rng.uniform(14, 30, keep.sum())
        wh, wa, ll, acc = mf.select_blend_weights_by_log_loss(
            y, true_home[keep], true_away[keep], noise_home, noise_away
        )
        self.assertLessEqual(wh, 0.3)
        self.assertLessEqual(wa, 0.3)


if __name__ == "__main__":
    unittest.main()
