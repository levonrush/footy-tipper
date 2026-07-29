import unittest

import numpy as np

from pipeline import evaluate as ev


def _synthetic_season(n_prior=400, n_test=120, seed=0):
    """Prior and held-out seasons drawn from one honest generating process."""
    rng = np.random.default_rng(seed)
    n = n_prior + n_test

    blended_h = rng.uniform(16.0, 30.0, size=n)
    blended_a = rng.uniform(16.0, 30.0, size=n)
    actual_home = rng.poisson(blended_h).astype(float)
    actual_away = rng.poisson(blended_a).astype(float)
    actual_margin = actual_home - actual_away

    # A line that knows the truth up to noise, on three quarters of the games.
    market_spread = (blended_h - blended_a) + rng.normal(0.0, 2.0, size=n)
    market_spread[rng.random(n) < 0.25] = np.nan

    prior_mask = np.zeros(n, dtype=bool)
    prior_mask[:n_prior] = True
    test_mask = ~prior_mask

    calibrated_cond = np.clip(
        rng.normal(0.55, 0.12, size=int(test_mask.sum())), 0.05, 0.95
    )

    return dict(
        blended_h=blended_h,
        blended_a=blended_a,
        actual_home=actual_home,
        actual_away=actual_away,
        actual_margin=actual_margin,
        market_spread=market_spread,
        calibrated_cond=calibrated_cond,
        prior_mask=prior_mask,
        test_mask=test_mask,
    )


def _score(n_samples=3000, seed=0, **overrides):
    args = _synthetic_season(seed=seed)
    args.update(overrides)
    return ev._score_margin_distributions(n_samples=n_samples, seed=seed, **args)


class ScoreMarginDistributionsTests(unittest.TestCase):
    def test_scores_every_method_and_the_market_comparison(self):
        scored = _score()

        self.assertEqual(scored["games"], 120)
        for name in (
            "model",
            "model_reconciled",
            "normal_approximation",
            "empirical_replay",
        ):
            self.assertIn(name, scored["methods"])
            self.assertIsNotNone(scored["methods"][name]["crps"])
            self.assertGreater(scored["methods"][name]["crps"], 0.0)

        market = scored["market_comparison"]
        self.assertGreater(market["games"], 0)
        self.assertIsNotNone(market["market_line_crps"])
        self.assertIsNotNone(market["model_crps"])

    def test_a_correctly_specified_model_is_close_to_calibrated(self):
        # The synthetic scores really are Poisson around the blended means, so
        # the model's own 90% interval should cover near 90%. If this drifts,
        # the sampling or interval code is wrong rather than the model.
        scored = _score(n_samples=6000)
        intervals = {
            interval["level"]: interval
            for interval in scored["methods"]["model"]["intervals"]
        }
        self.assertAlmostEqual(intervals[0.9]["coverage"], 0.9, delta=0.06)
        self.assertGreater(intervals[0.9]["width"], 0.0)

    def test_coverage_is_always_paired_with_width(self):
        scored = _score()
        for method in scored["methods"].values():
            for interval in method.get("intervals", []):
                self.assertIn("coverage", interval)
                self.assertIn("width", interval)

    def test_distribution_parameters_come_from_prior_seasons_only(self):
        # Changing only held-out rows must not move the fitted parameters.
        base = _synthetic_season(seed=1)
        tampered = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
        tampered["actual_home"][tampered["test_mask"]] += 40.0
        tampered["actual_margin"] = tampered["actual_home"] - tampered["actual_away"]

        first = ev._score_margin_distributions(n_samples=1000, seed=1, **base)
        second = ev._score_margin_distributions(n_samples=1000, seed=1, **tampered)

        self.assertEqual(first["lambda3"], second["lambda3"])
        self.assertEqual(first["dispersion_home"], second["dispersion_home"])
        self.assertEqual(first["dispersion_away"], second["dispersion_away"])

    def test_returns_none_without_enough_rows(self):
        args = _synthetic_season()
        args["test_mask"] = np.zeros(args["test_mask"].shape, dtype=bool)
        self.assertIsNone(
            ev._score_margin_distributions(n_samples=500, seed=0, **args)
        )


class PoolMarginDistributionsTests(unittest.TestCase):
    def test_pools_by_game_count_across_seasons(self):
        seasons = [
            {"margin_distribution": _score(n_samples=1000, seed=s)} for s in (2, 3)
        ]
        pooled = ev._pool_margin_distributions(seasons)

        self.assertEqual(pooled["games"], 240)
        self.assertIn("model", pooled["methods"])

        # Equal game counts per season, so the pool is the plain mean.
        expected = np.mean(
            [s["margin_distribution"]["methods"]["model"]["crps"] for s in seasons]
        )
        self.assertAlmostEqual(pooled["methods"]["model"]["crps"], expected, places=9)

    def test_returns_none_when_no_season_scored(self):
        self.assertIsNone(ev._pool_margin_distributions([{"margin_distribution": None}]))

    def test_printing_a_pooled_scorecard_does_not_raise(self):
        pooled = ev._pool_margin_distributions(
            [{"margin_distribution": _score(n_samples=800, seed=4)}]
        )
        ev._print_margin_distribution(pooled)
        ev._print_margin_distribution(None)


if __name__ == "__main__":
    unittest.main()
