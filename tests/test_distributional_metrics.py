import unittest

import numpy as np

from pipeline.common.model_training.distributional_metrics import (
    crps_ensemble,
    crps_normal,
    crps_weighted_ensemble,
    interval_coverage,
    pit_histogram,
    predictive_interval,
    randomised_pit,
    score_sample_forecasts,
)


class CrpsTests(unittest.TestCase):
    def test_matches_hand_computed_two_sample_case(self):
        # samples {0, 2}, y = 1:
        #   mean|x - y|            = 1.0
        #   (1/2S^2) sum sum |x-x'| = (1/8) * 4 = 0.5
        self.assertAlmostEqual(crps_ensemble([0.0, 2.0], 1.0), 0.5, places=12)

    def test_reduces_to_absolute_error_for_a_point_forecast(self):
        # A degenerate ensemble has no spread, so CRPS collapses to MAE. This
        # is the property that lets deterministic and probabilistic methods be
        # compared on one scale.
        samples = np.full(500, 7.0)
        self.assertAlmostEqual(crps_ensemble(samples, 3.0), 4.0, places=10)

    def test_closed_form_gaussian_matches_the_known_value_at_the_mean(self):
        # CRPS(N(0, 1), 0) = (sqrt(2) - 1) / sqrt(pi). Pins the closed form
        # exactly, with no sampling noise in the way.
        self.assertAlmostEqual(
            crps_normal(0.0, 1.0, 0.0), (np.sqrt(2.0) - 1.0) / np.sqrt(np.pi), places=12
        )

    def test_ensemble_converges_to_the_closed_form_gaussian(self):
        rng = np.random.default_rng(20100308)
        mu, sigma, y = 2.5, 4.0, -1.0
        samples = rng.normal(mu, sigma, size=400_000)
        # Tolerance is set by Monte Carlo error (se of mean|X - y| is about
        # sigma/sqrt(S) ~= 0.006), not by the estimator. Any real mistake in
        # the normalisation shifts the result by order sigma, far beyond this.
        self.assertAlmostEqual(
            crps_ensemble(samples, y), crps_normal(mu, sigma, y), delta=0.03
        )

    def test_is_minimised_by_the_true_distribution(self):
        # Propriety: the correctly specified forecast must not be beaten by a
        # too-narrow or too-wide one. If this fails the score is not usable as
        # a gate.
        rng = np.random.default_rng(11)
        truth = rng.normal(0.0, 3.0, size=4000)

        def mean_crps(sigma):
            draws = rng.normal(0.0, sigma, size=20_000)
            return float(np.mean([crps_ensemble(draws, y) for y in truth[:400]]))

        honest = mean_crps(3.0)
        self.assertLess(honest, mean_crps(0.75))
        self.assertLess(honest, mean_crps(9.0))

    def test_returns_nan_for_empty_or_missing_input(self):
        self.assertTrue(np.isnan(crps_ensemble([], 1.0)))
        self.assertTrue(np.isnan(crps_ensemble([1.0, 2.0], np.nan)))


class WeightedCrpsTests(unittest.TestCase):
    """Scores the old importance-reweighted ensemble on the same scale."""

    def test_reduces_to_the_unweighted_estimator_when_weights_are_equal(self):
        rng = np.random.default_rng(19)
        samples = rng.normal(3.0, 5.0, size=2000)
        for weight in (1.0, 0.25, 7.5):
            self.assertAlmostEqual(
                crps_weighted_ensemble(samples, np.full(samples.size, weight), 1.5),
                crps_ensemble(samples, 1.5),
                places=10,
            )

    def test_duplicating_a_sample_is_the_same_as_doubling_its_weight(self):
        # The property the legacy comparison relies on: reweighting a draw and
        # replicating it must score identically, so a weighted ensemble and the
        # ensemble it stands for cannot disagree.
        base = np.array([-6.0, 0.0, 4.0, 11.0])
        weights = np.array([2.0, 1.0, 3.0, 1.0])
        expanded = np.repeat(base, weights.astype(int))
        self.assertAlmostEqual(
            crps_weighted_ensemble(base, weights, 2.0),
            crps_ensemble(expanded, 2.0),
            places=10,
        )

    def test_shifting_weight_toward_the_outcome_improves_the_score(self):
        samples = np.array([-10.0, 10.0])
        self.assertLess(
            crps_weighted_ensemble(samples, [0.1, 0.9], 10.0),
            crps_weighted_ensemble(samples, [0.9, 0.1], 10.0),
        )

    def test_returns_nan_for_degenerate_weights(self):
        self.assertTrue(np.isnan(crps_weighted_ensemble([1.0, 2.0], [0.0, 0.0], 1.0)))
        self.assertTrue(np.isnan(crps_weighted_ensemble([1.0, 2.0], [1.0], 1.0)))


class RandomisedPitTests(unittest.TestCase):
    """The discrete adaptation is the whole point, so it is tested directly."""

    @staticmethod
    def _uniformity(pit_values):
        return pit_histogram(pit_values)["uniformity_mae"]

    def test_uniform_when_the_discrete_model_is_correctly_specified(self):
        rng = np.random.default_rng(7)
        draws = rng.poisson(6.0, size=40_000)
        truth = rng.poisson(6.0, size=3000)

        pit_values = [randomised_pit(draws, y, rng=rng) for y in truth]
        # Perfectly specified: the histogram should sit close to flat.
        self.assertLess(self._uniformity(pit_values), 0.01)

    def test_naive_pit_on_the_same_discrete_forecast_is_not_uniform(self):
        # Guard against "simplifying" randomised_pit back to F(y). On integer
        # counts the plain transform is lumpy even though the model is right,
        # which would be misread as miscalibration.
        rng = np.random.default_rng(7)
        draws = rng.poisson(6.0, size=40_000)
        truth = rng.poisson(6.0, size=3000)

        naive = [float(np.mean(draws <= y)) for y in truth]
        randomised = [randomised_pit(draws, y, rng=rng) for y in truth]

        self.assertGreater(self._uniformity(naive), 3.0 * self._uniformity(randomised))

    def test_u_shaped_when_the_forecast_is_too_narrow(self):
        rng = np.random.default_rng(3)
        draws = rng.normal(0.0, 1.0, size=40_000)
        truth = rng.normal(0.0, 4.0, size=3000)

        pit_values = np.array([randomised_pit(draws, y, rng=rng) for y in truth])
        frequencies = pit_histogram(pit_values)["frequencies"]
        edges = frequencies[0] + frequencies[-1]
        middle = sum(frequencies[4:6])
        self.assertGreater(edges, 3.0 * middle)

    def test_hump_shaped_when_the_forecast_is_too_wide(self):
        rng = np.random.default_rng(5)
        draws = rng.normal(0.0, 6.0, size=40_000)
        truth = rng.normal(0.0, 1.0, size=3000)

        frequencies = pit_histogram(
            [randomised_pit(draws, y, rng=rng) for y in truth]
        )["frequencies"]
        self.assertGreater(sum(frequencies[4:6]), frequencies[0] + frequencies[-1])


class IntervalTests(unittest.TestCase):
    def test_coverage_matches_the_nominal_level_when_calibrated(self):
        rng = np.random.default_rng(13)
        draws = rng.normal(0.0, 2.0, size=60_000)
        low, high = predictive_interval(draws, level=0.9)
        truth = rng.normal(0.0, 2.0, size=4000)

        scored = interval_coverage(
            np.full(truth.size, low), np.full(truth.size, high), truth, level=0.9
        )
        self.assertAlmostEqual(scored["coverage"], 0.9, delta=0.02)
        self.assertGreater(scored["width"], 0.0)

    def test_coverage_and_width_are_reported_together(self):
        scored = interval_coverage([0.0], [10.0], [5.0], level=0.9)
        self.assertIn("coverage", scored)
        self.assertIn("width", scored)
        self.assertEqual(scored["coverage"], 1.0)
        self.assertEqual(scored["width"], 10.0)

    def test_handles_no_valid_rows(self):
        scored = interval_coverage([np.nan], [np.nan], [np.nan])
        self.assertEqual(scored["games"], 0)
        self.assertIsNone(scored["coverage"])
        self.assertIsNone(scored["width"])


class ScoreSampleForecastsTests(unittest.TestCase):
    def test_scores_a_set_of_per_game_draws(self):
        rng = np.random.default_rng(2)
        actuals = rng.normal(0.0, 5.0, size=200)
        sample_sets = [rng.normal(0.0, 5.0, size=5000) for _ in actuals]

        scored = score_sample_forecasts(sample_sets, actuals, rng=rng)

        self.assertEqual(scored["games"], 200)
        self.assertGreater(scored["crps"], 0.0)
        self.assertEqual(len(scored["intervals"]), 2)
        for interval in scored["intervals"]:
            self.assertAlmostEqual(
                interval["coverage"], interval["level"], delta=0.08
            )

    def test_a_sharper_honest_forecast_scores_better_than_a_vague_one(self):
        rng = np.random.default_rng(4)
        actuals = rng.normal(0.0, 2.0, size=300)
        honest = [rng.normal(0.0, 2.0, size=4000) for _ in actuals]
        vague = [rng.normal(0.0, 20.0, size=4000) for _ in actuals]

        self.assertLess(
            score_sample_forecasts(honest, actuals, rng=rng)["crps"],
            score_sample_forecasts(vague, actuals, rng=rng)["crps"],
        )


if __name__ == "__main__":
    unittest.main()
