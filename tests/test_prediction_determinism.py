import unittest

import numpy as np
import pandas as pd

from pipeline.common.model_prediciton import prediction_functions as pf


class DeterminismTests(unittest.TestCase):
    def test_rng_for_game_is_stable_per_game(self):
        a = pf.rng_for_game(101, salt=1).integers(0, 1_000_000)
        b = pf.rng_for_game(101, salt=1).integers(0, 1_000_000)
        c = pf.rng_for_game(102, salt=1).integers(0, 1_000_000)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_marginalized_prob_repeatable(self):
        kwargs = dict(lineup_uncertainty_home=0.2, lineup_uncertainty_away=0.1, n_samples=32)
        first = pf.marginalized_conditional_home_win_prob(22, 20, rng=pf.rng_for_game(7, salt=2), **kwargs)
        second = pf.marginalized_conditional_home_win_prob(22, 20, rng=pf.rng_for_game(7, salt=2), **kwargs)
        self.assertEqual(first, second)

    def test_full_prediction_run_is_repeatable(self):
        frame = pd.DataFrame(
            [
                {"game_id": 11, "mu_h": 24.0, "mu_a": 18.0},
                {"game_id": 12, "mu_h": 16.0, "mu_a": 17.5},
            ]
        )

        def run():
            outcomes, margins = pf.predict_match_outcome_and_scoreline_with_bayes(
                inference_data=frame,
                mu_home=frame["mu_h"].to_numpy(),
                mu_away=frame["mu_a"].to_numpy(),
                lambda3=2.0,
                n_simulations=20000,
            )
            return outcomes.merge(margins, on="game_id")

        first = run()
        second = run()
        pd.testing.assert_frame_equal(first, second)

    def test_margin_is_median_of_simulations(self):
        probs, scoreline = pf.simulate_game(26, 14, n_simulations=50000, rng=pf.rng_for_game(3))
        self.assertIn("median_margin", probs)
        # For mu_home=26 vs mu_away=14 the median margin must be near 12.
        self.assertTrue(8 <= probs["median_margin"] <= 16)
        self.assertEqual(len(scoreline), 2)


class CalibratedConsistencyTests(unittest.TestCase):
    def test_tip_never_contradicts_margin_or_scoreline(self):
        # Raw mus favour the home side, but calibration flips the tip away.
        frame = pd.DataFrame([{"game_id": 21, "mu_h": 22.0, "mu_a": 19.0}])
        outcomes, margins = pf.predict_match_outcome_and_scoreline_with_bayes(
            inference_data=frame,
            mu_home=frame["mu_h"].to_numpy(),
            mu_away=frame["mu_a"].to_numpy(),
            n_simulations=20000,
            calibrated_home_win_conditional=np.array([0.35]),
        )
        row = outcomes.merge(margins, on="game_id").iloc[0]
        self.assertEqual(row["home_team_result"], "Loss")
        self.assertLess(row["predicted_margin"], 0)
        self.assertLess(row["predicted_home_score"], row["predicted_away_score"])

    def test_calibrated_margin_moves_toward_calibration(self):
        # Same raw mus; a stronger calibrated home prob must not shrink the margin.
        frame = pd.DataFrame([{"game_id": 22, "mu_h": 23.0, "mu_a": 20.0}])

        def margin_for(cal):
            _, margins = pf.predict_match_outcome_and_scoreline_with_bayes(
                inference_data=frame,
                mu_home=frame["mu_h"].to_numpy(),
                mu_away=frame["mu_a"].to_numpy(),
                n_simulations=20000,
                calibrated_home_win_conditional=np.array([cal]),
            )
            return margins["predicted_margin"].iloc[0]

        self.assertGreaterEqual(margin_for(0.85), margin_for(0.55))
        self.assertGreater(margin_for(0.85), 0)
        self.assertLess(margin_for(0.15), 0)

    def test_extreme_calibration_flip_still_orders_scoreline(self):
        # mus so lopsided the simulation never produces an away win.
        frame = pd.DataFrame([{"game_id": 23, "mu_h": 60.0, "mu_a": 2.0}])
        outcomes, margins = pf.predict_match_outcome_and_scoreline_with_bayes(
            inference_data=frame,
            mu_home=frame["mu_h"].to_numpy(),
            mu_away=frame["mu_a"].to_numpy(),
            n_simulations=5000,
            calibrated_home_win_conditional=np.array([0.05]),
        )
        row = outcomes.merge(margins, on="game_id").iloc[0]
        self.assertEqual(row["home_team_result"], "Loss")
        self.assertLess(row["predicted_margin"], 0)
        self.assertLess(row["predicted_home_score"], row["predicted_away_score"])


class VectorisedWinProbTests(unittest.TestCase):
    def test_matches_scalar_implementation(self):
        mu_home = np.array([22.0, 18.5, 30.0, 5.0])
        mu_away = np.array([20.0, 24.0, 12.0, 5.0])
        vec = pf.conditional_home_win_prob_vec(mu_home, mu_away)
        scalar = [pf.conditional_home_win_prob(h, a) for h, a in zip(mu_home, mu_away)]
        np.testing.assert_allclose(vec, scalar, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
