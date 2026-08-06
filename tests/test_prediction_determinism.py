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

    @staticmethod
    def _simulated_cond(target, reconcile):
        probs, _ = pf.simulate_game(
            24.0,
            20.0,
            n_simulations=200_000,
            rng=pf.rng_for_game(31),
            calibrated_cond=target,
            reconcile=reconcile,
        )
        return probs, probs["home_win_prob"] / (
            probs["home_win_prob"] + probs["away_win_prob"]
        )

    def test_always_mode_carries_the_calibration_exactly(self):
        # The property the old importance-reweighting could not provide: the
        # distribution actually simulated carries the calibrated probability.
        for target in (0.25, 0.5, 0.72):
            probs, simulated = self._simulated_cond(target, "always")
            self.assertAlmostEqual(simulated, target, delta=0.01)
            self.assertTrue(probs["reconciled"])

    def test_on_conflict_mode_reconciles_only_when_the_side_would_flip(self):
        # Raw means 24-20 favour the home side, so:
        #   a target that also favours home is left alone, and the simulation
        #   keeps the score model's own view;
        #   a target favouring away must be honoured exactly, or the scoreline
        #   would contradict the tip.
        raw = pf.conditional_home_win_prob(24.0, 20.0)
        self.assertGreater(raw, 0.5)

        probs, simulated = self._simulated_cond(0.72, "on_conflict")
        self.assertFalse(probs["reconciled"])
        self.assertAlmostEqual(simulated, raw, delta=0.01)
        self.assertNotAlmostEqual(simulated, 0.72, delta=0.005)

        probs, simulated = self._simulated_cond(0.25, "on_conflict")
        self.assertTrue(probs["reconciled"])
        self.assertAlmostEqual(simulated, 0.25, delta=0.01)

    def test_unreachable_target_clamps_instead_of_raising(self):
        mu_h, mu_a = pf.solve_score_means_for_probability(20.0, 20.0, 1 - 1e-15)
        self.assertGreater(mu_h, mu_a)
        self.assertTrue(np.isfinite(mu_h) and np.isfinite(mu_a))
        self.assertAlmostEqual(mu_h + mu_a, 40.0, places=9)


class DisplayedScorelineTests(unittest.TestCase):
    """The three integers that reach the predictions table.

    Nothing scored these until the margin scorecard went in, and the modal
    scoreline turned out to be a high-variance statistic. These pin the median
    reduction that replaced it.
    """

    def test_median_scoreline_difference_is_the_median_margin(self):
        rng = np.random.default_rng(61)
        home = rng.poisson(24.0, 40_000)
        away = rng.poisson(18.0, 40_000)
        expected = int(round(float(np.median(home - away))))

        scoreline = pf.scoreline_from_samples(home, away, display="median")
        self.assertEqual(scoreline[0] - scoreline[1], expected)

    def test_median_total_is_within_a_point_of_the_simulated_median(self):
        # The total gets nudged at most one point, and only for parity.
        rng = np.random.default_rng(62)
        home = rng.poisson(21.0, 40_000)
        away = rng.poisson(19.0, 40_000)
        median_total = float(np.median(home + away))

        scoreline = pf.scoreline_from_samples(home, away, display="median")
        self.assertLessEqual(abs(sum(scoreline) - median_total), 1.0)

    def test_median_reduction_is_far_less_noisy_than_the_mode(self):
        # The reason for the change: re-drawing the same match must not move the
        # displayed margin around. The mode does, the median barely.
        def spread(display):
            margins = []
            for seed in range(12):
                home, away = pf.draw_score_samples(
                    22.0, 20.0, 20_000, dispersion_home=5.19, dispersion_away=4.34,
                    rng=np.random.default_rng(seed),
                )
                scoreline = pf.scoreline_from_samples(
                    home, away, tipped_home=True, display=display
                )
                margins.append(scoreline[0] - scoreline[1])
            return max(margins) - min(margins)

        self.assertLess(spread("median"), spread("mode"))

    def test_never_contradicts_the_tip_in_the_near_tie_band(self):
        # A median margin of zero, or one landing on the wrong side, still has
        # to show the tipped team in front.
        for cond in (0.5001, 0.4999, 0.502, 0.498, 0.51, 0.49):
            _, scoreline = pf.simulate_game(
                21.0,
                21.0,
                n_simulations=20_000,
                rng=pf.rng_for_game(63),
                calibrated_cond=cond,
                dispersion_home=5.19,
                dispersion_away=4.34,
            )
            margin = scoreline[0] - scoreline[1]
            self.assertNotEqual(margin, 0)
            self.assertEqual(margin > 0, cond > 0.5)

    def test_scores_are_never_negative(self):
        # A lopsided negative margin against a small total must not push a score
        # below zero, and must not silently change the margin either.
        scoreline = pf.scoreline_from_samples(
            np.array([0, 0, 1]), np.array([30, 31, 30]), tipped_home=False,
            display="median",
        )
        self.assertGreaterEqual(scoreline[0], 0)
        self.assertGreaterEqual(scoreline[1], 0)
        self.assertLess(scoreline[0] - scoreline[1], 0)

    def test_rejects_an_unknown_display_mode(self):
        with self.assertRaises(ValueError):
            pf.scoreline_from_samples([1, 2], [3, 4], display="mean")


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


class GoldenPointTests(unittest.TestCase):
    """A tie after eighty minutes is not a drawn game."""

    def _probabilities(self, mu_home=22.0, mu_away=20.0, seed=5):
        probabilities, _ = pf.simulate_game(
            mu_home, mu_away, n_simulations=40000, rng=np.random.default_rng(seed)
        )
        return probabilities

    def test_draw_probability_is_a_small_residual(self):
        probabilities = self._probabilities()
        # The unadjusted simulation puts 4% to 6% on a tie; the realised NRL
        # rate is 0.32% since 2016.
        self.assertLess(probabilities["draw_prob"], 0.01)
        self.assertGreater(probabilities["draw_prob"], 0.0)

    def test_the_triple_still_sums_to_one(self):
        probabilities = self._probabilities()
        total = (
            probabilities["home_win_prob"]
            + probabilities["away_win_prob"]
            + probabilities["draw_prob"]
        )
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_extra_time_does_not_move_the_conditional_probability(self):
        """The invariant that matters.

        Everything upstream is fitted and calibrated on the non-draw
        conditional, so resolving ties must redistribute the tied mass without
        changing which side is favoured or by how much.
        """
        def conditional(share):
            original = pf.GOLDEN_POINT_UNRESOLVED_SHARE
            pf.GOLDEN_POINT_UNRESOLVED_SHARE = share
            try:
                probabilities = self._probabilities()
            finally:
                pf.GOLDEN_POINT_UNRESOLVED_SHARE = original
            return probabilities["home_win_prob"] / (
                probabilities["home_win_prob"] + probabilities["away_win_prob"]
            )

        # share=1.0 leaves every tie unresolved, i.e. the old behaviour.
        self.assertAlmostEqual(conditional(0.10), conditional(1.0), places=9)

    def test_a_lopsided_game_sends_extra_time_to_the_stronger_side(self):
        lopsided = self._probabilities(mu_home=30.0, mu_away=14.0)
        even = self._probabilities(mu_home=20.0, mu_away=20.0)
        self.assertGreater(lopsided["home_win_prob"], even["home_win_prob"])


if __name__ == "__main__":
    unittest.main()
