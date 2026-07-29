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
        self.assertEqual(
            row["predicted_margin"],
            row["predicted_home_score"] - row["predicted_away_score"],
        )

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


class LineupMarginalisationTests(unittest.TestCase):
    """train.py fits the pools on the marginalised Tier-B probability now.

    It used to compute the marginalised value, discard it, and fit on the plain
    one while inference served the marginalised one. These pin the properties
    that made that mismatch matter.
    """

    def test_is_an_exact_no_op_without_lineup_uncertainty(self):
        # Bounds the old skew: with no lineups the two paths agreed exactly,
        # so only games with real lineup uncertainty were ever affected.
        plain = pf.conditional_home_win_prob(24.0, 20.0)
        marginalised = pf.marginalized_conditional_home_win_prob(
            24.0,
            20.0,
            lineup_uncertainty_home=0.0,
            lineup_uncertainty_away=0.0,
            rng=pf.rng_for_game(51, salt=2),
        )
        self.assertEqual(plain, marginalised)

    def test_vectorised_marginalisation_matches_the_scalar_path(self):
        # train.py and evaluate.py use the vectorised form; inference uses the
        # scalar one. They must agree exactly or the meta-layer is again being
        # fitted on a different input from the one production serves.
        mu_h = np.array([24.0, 18.0, 30.0, 21.0])
        mu_a = np.array([20.0, 22.0, 12.0, 21.0])
        unc_h = np.array([0.20, 0.0, 0.10, 0.25])
        unc_a = np.array([0.05, 0.0, 0.24, 0.15])
        game_ids = np.array([11, 12, 13, 14])

        vectorised = pf.marginalized_conditional_home_win_prob_vec(
            mu_h, mu_a, unc_h, unc_a, game_ids=game_ids, n_samples=64
        )
        scalar = np.array(
            [
                pf.marginalized_conditional_home_win_prob(
                    h,
                    a,
                    lineup_uncertainty_home=uh,
                    lineup_uncertainty_away=ua,
                    n_samples=64,
                    rng=pf.rng_for_game(gid, salt=2),
                )
                for h, a, uh, ua, gid in zip(mu_h, mu_a, unc_h, unc_a, game_ids)
            ]
        )
        np.testing.assert_allclose(vectorised, scalar, rtol=0, atol=1e-12)

    def test_vectorised_marginalisation_is_a_no_op_without_uncertainty(self):
        mu_h = np.array([24.0, 18.0])
        mu_a = np.array([20.0, 22.0])
        np.testing.assert_array_equal(
            pf.marginalized_conditional_home_win_prob_vec(
                mu_h, mu_a, np.zeros(2), np.zeros(2), game_ids=np.array([1, 2])
            ),
            pf.conditional_home_win_prob_vec(mu_h, mu_a),
        )

    def test_uncertainty_pulls_the_probability_toward_a_coin_flip(self):
        plain = pf.conditional_home_win_prob(28.0, 18.0)
        marginalised = pf.marginalized_conditional_home_win_prob(
            28.0,
            18.0,
            lineup_uncertainty_home=0.25,
            lineup_uncertainty_away=0.25,
            n_samples=512,
            rng=pf.rng_for_game(52, salt=2),
        )
        self.assertLess(marginalised, plain)
        self.assertGreater(marginalised, 0.5)


class ScoreMeanReconciliationTests(unittest.TestCase):
    """The calibrated probability and the score distribution are one object."""

    def test_holds_the_total_and_hits_the_target(self):
        mu_h, mu_a = pf.solve_score_means_for_probability(22.0, 19.0, 0.35)
        self.assertAlmostEqual(mu_h + mu_a, 41.0, places=9)
        self.assertAlmostEqual(
            pf.conditional_home_win_prob(mu_h, mu_a), 0.35, places=6
        )

    def test_is_monotone_in_the_target(self):
        totals = []
        for target in (0.2, 0.4, 0.6, 0.8):
            mu_h, mu_a = pf.solve_score_means_for_probability(21.0, 21.0, target)
            totals.append(mu_h - mu_a)
        self.assertEqual(totals, sorted(totals))

    def test_simulated_win_probability_matches_the_calibration(self):
        # The property the old importance-reweighting could not provide: the
        # distribution actually simulated carries the calibrated probability.
        for target in (0.25, 0.5, 0.72):
            probs, _ = pf.simulate_game(
                24.0,
                20.0,
                n_simulations=200_000,
                rng=pf.rng_for_game(31),
                calibrated_cond=target,
            )
            simulated = probs["home_win_prob"] / (
                probs["home_win_prob"] + probs["away_win_prob"]
            )
            self.assertAlmostEqual(simulated, target, delta=0.01)

    def test_unreachable_target_clamps_instead_of_raising(self):
        mu_h, mu_a = pf.solve_score_means_for_probability(20.0, 20.0, 1 - 1e-15)
        self.assertGreater(mu_h, mu_a)
        self.assertTrue(np.isfinite(mu_h) and np.isfinite(mu_a))
        self.assertAlmostEqual(mu_h + mu_a, 40.0, places=9)


class SharedComponentTests(unittest.TestCase):
    def test_dispersion_and_lambda3_apply_together(self):
        # Previously a non-zero lambda3 silently discarded the dispersion. The
        # rescaled k must restore the marginal variance to mu + mu^2/k while
        # the shared component holds the covariance at lambda3.
        mu, k, lambda3 = 24.0, 8.0, 2.0
        home, away = pf.draw_score_samples(
            mu,
            mu,
            400_000,
            lambda3=lambda3,
            dispersion_home=k,
            dispersion_away=k,
            rng=pf.rng_for_game(41),
        )

        expected_variance = mu + mu**2 / k
        self.assertAlmostEqual(float(np.var(home)), expected_variance, delta=1.5)
        self.assertAlmostEqual(
            float(np.cov(home, away)[0, 1]), lambda3, delta=0.3
        )

    def test_pure_poisson_is_unchanged_when_lambda3_is_zero(self):
        mu = 20.0
        home, _ = pf.draw_score_samples(
            mu, mu, 200_000, lambda3=0.0, rng=pf.rng_for_game(43)
        )
        self.assertAlmostEqual(float(np.var(home)), mu, delta=0.5)
        self.assertAlmostEqual(float(np.mean(home)), mu, delta=0.05)


class DispersionTests(unittest.TestCase):
    def test_nb_dispersion_widens_margins_and_stays_deterministic(self):
        def run(disp):
            probs, scoreline = pf.simulate_game(
                24, 20, n_simulations=20000, rng=pf.rng_for_game(5),
                dispersion_home=disp, dispersion_away=disp,
            )
            return probs, scoreline

        poisson_a, line_a = run(None)
        poisson_b, line_b = run(None)
        self.assertEqual(poisson_a, poisson_b)
        self.assertEqual(line_a, line_b)

        nb_a, nb_line_a = run(8.0)
        nb_b, nb_line_b = run(8.0)
        self.assertEqual(nb_a, nb_b)
        self.assertEqual(nb_line_a, nb_line_b)

        # Fatter tails: the favourite wins less often under over-dispersion.
        self.assertLess(nb_a["home_win_prob"], poisson_a["home_win_prob"])


class VectorisedWinProbTests(unittest.TestCase):
    def test_matches_scalar_implementation(self):
        mu_home = np.array([22.0, 18.5, 30.0, 5.0])
        mu_away = np.array([20.0, 24.0, 12.0, 5.0])
        vec = pf.conditional_home_win_prob_vec(mu_home, mu_away)
        scalar = [pf.conditional_home_win_prob(h, a) for h, a in zip(mu_home, mu_away)]
        np.testing.assert_allclose(vec, scalar, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
