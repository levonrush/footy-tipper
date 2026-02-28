import unittest

from pipeline.common.model_prediciton import prediction_functions as pf


class LineupUncertaintyTests(unittest.TestCase):
    def test_marginalized_prob_matches_base_when_noise_disabled(self):
        base = pf.conditional_home_win_prob(24.0, 20.0)
        marginalized = pf.marginalized_conditional_home_win_prob(
            24.0,
            20.0,
            lineup_uncertainty_home=0.2,
            lineup_uncertainty_away=0.2,
            n_samples=128,
            mu_noise_scale=0.0,
        )
        self.assertAlmostEqual(base, marginalized, places=8)

    def test_marginalized_prob_is_valid_probability(self):
        value = pf.marginalized_conditional_home_win_prob(
            24.0,
            20.0,
            lineup_uncertainty_home=0.22,
            lineup_uncertainty_away=0.08,
            n_samples=128,
            mu_noise_scale=0.15,
        )
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()

