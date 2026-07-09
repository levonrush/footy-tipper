import unittest

import numpy as np

from pipeline.common.model_training import comp_sim


def _season(rng, n=180, edge=0.62):
    """A realized season: market favourites win `edge` of the time."""
    market_p = rng.uniform(0.2, 0.8, n)
    market_p = np.where(np.isclose(market_p, 0.5), 0.55, market_p)
    fav_wins = rng.random(n) < edge
    outcomes = np.where(fav_wins, market_p > 0.5, market_p <= 0.5).astype(int)
    return market_p, outcomes


class CompSimTests(unittest.TestCase):
    def test_empty_input_returns_none(self):
        self.assertIsNone(comp_sim.simulate_comp_placement([], [], []))

    def test_all_nan_returns_none(self):
        nan = np.full(5, np.nan)
        self.assertIsNone(comp_sim.simulate_comp_placement(nan, nan, nan))

    def test_deterministic_under_seed(self):
        rng = np.random.default_rng(1)
        market_p, outcomes = _season(rng)
        model_p = np.clip(market_p + rng.normal(0, 0.05, market_p.size), 0.01, 0.99)
        a = comp_sim.simulate_comp_placement(model_p, market_p, outcomes, n_sims=500)
        b = comp_sim.simulate_comp_placement(model_p, market_p, outcomes, n_sims=500)
        self.assertEqual(a, b)

    def test_better_tipper_places_higher(self):
        rng = np.random.default_rng(2)
        market_p, outcomes = _season(rng)
        # Oracle tips every game right; mediocre tipper is wrong often.
        oracle_p = np.where(outcomes == 1, 0.9, 0.1)
        coin_p = np.full(outcomes.size, 0.51)
        oracle = comp_sim.simulate_comp_placement(oracle_p, market_p, outcomes, n_sims=500)
        coin = comp_sim.simulate_comp_placement(coin_p, market_p, outcomes, n_sims=500)
        self.assertGreater(oracle["p_first"], coin["p_first"])
        self.assertLess(oracle["expected_rank"], coin["expected_rank"])
        self.assertGreater(oracle["p_first"], 0.95)

    def test_explicit_tips_override_model_probs(self):
        rng = np.random.default_rng(3)
        market_p, outcomes = _season(rng)
        model_p = np.full(outcomes.size, 0.5)
        perfect = comp_sim.simulate_comp_placement(
            model_p, market_p, outcomes, tips=outcomes.astype(bool), n_sims=200
        )
        self.assertEqual(perfect["user_score"], outcomes.size)

    def test_user_score_matches_manual_count(self):
        model_p = np.array([0.7, 0.3, 0.6, 0.2])
        market_p = np.array([0.6, 0.4, 0.55, 0.45])
        outcomes = np.array([1, 0, 0, 0])
        res = comp_sim.simulate_comp_placement(model_p, market_p, outcomes, n_sims=100)
        # Tips: H, A, H, A -> correct on games 0, 1, 3.
        self.assertEqual(res["user_score"], 3)
        self.assertEqual(res["market_favourite_score"], 3)


if __name__ == "__main__":
    unittest.main()
