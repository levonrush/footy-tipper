import unittest

import numpy as np

from pipeline.common.model_prediciton import prediction_functions as pf

try:
    from pipeline import evaluate
except ModuleNotFoundError as exc:
    if exc.name not in {"lightgbm", "skopt"}:
        raise
    evaluate = None


@unittest.skipIf(evaluate is None, "model-training dependencies are not installed")
class ProbabilityEvaluationTests(unittest.TestCase):
    def _evaluate_synthetic_season(self, *, all_market=False):
        rng = np.random.default_rng(73)
        seasons = np.arange(2019, 2025, dtype=float)
        per_season = 60
        year_col = np.repeat(seasons, per_season)
        n = len(year_col)

        strength = rng.normal(0.0, 3.0, n)
        baseline_mu_home = np.clip(22.0 + 0.4 * strength, 10.0, 35.0)
        baseline_mu_away = np.clip(20.0 - 0.4 * strength, 10.0, 35.0)
        home_mu_oof = np.clip(22.0 + strength, 8.0, 40.0)
        away_mu_oof = np.clip(20.0 - strength, 8.0, 40.0)
        tier_b = pf.conditional_home_win_prob_vec(home_mu_oof, away_mu_oof)
        tier_a = np.clip(tier_b + rng.normal(0.0, 0.07, n), 0.02, 0.98)
        tier_c = np.clip(tier_b + rng.normal(0.0, 0.05, n), 0.02, 0.98)
        market = np.clip(tier_b + rng.normal(0.0, 0.04, n), 0.05, 0.95)
        y = (rng.uniform(0.0, 1.0, n) < tier_b).astype(int)

        valid_market = np.ones(n, dtype=bool) if all_market else np.arange(n) % 3 != 0
        home_odds = np.where(valid_market, 1.0 / (0.95 * market), np.nan)
        away_odds = np.where(valid_market, 1.0 / (0.95 * (1.0 - market)), np.nan)
        market_for_model = np.where(valid_market, market, 0.5)

        return evaluate._evaluate_season(
            2024,
            year_col,
            np.ones(n, dtype=bool),
            np.ones(n, dtype=bool),
            y,
            baseline_mu_home,
            baseline_mu_away,
            home_mu_oof,
            away_mu_oof,
            tier_a,
            tier_c,
            market_for_model,
            valid_market,
            home_odds,
            away_odds,
            home_mu_oof - away_mu_oof,
            np.full(n, np.nan),
            actual_home_score=home_mu_oof,
            actual_away_score=away_mu_oof,
            sim_samples=500,
        ), per_season

    def test_nested_evaluation_reports_market_and_model_only_regimes(self):
        result, per_season = self._evaluate_synthetic_season()

        self.assertIsNotNone(result)
        self.assertEqual(result["games"], per_season)
        self.assertEqual(
            result["market_regime"]["games"] + result["no_market_regime"]["games"],
            per_season,
        )
        self.assertTrue(np.isfinite(result["model_p"]).all())
        self.assertEqual(
            result["no_market_counterfactual"]["games"],
            per_season,
        )
        self.assertTrue(np.isfinite(result["no_market_counterfactual_p"]).all())
        self.assertLessEqual(result["bets"], result["market_regime"]["games"])
        self.assertIn(result["no_market_strategy"], {"simplex", "tier_b"})

        # The margin scorecard rides along with the season, and must survive
        # this fixture's degenerate zero-residual scores without crashing.
        margin_distribution = result["margin_distribution"]
        self.assertIsNotNone(margin_distribution)
        self.assertEqual(margin_distribution["games"], per_season)
        self.assertIn("model", margin_distribution["methods"])
        self.assertGreater(margin_distribution["methods"]["model"]["crps"], 0.0)

        pool_loss = result["no_market_selection_pool_log_loss"]
        tier_b_loss = result["no_market_selection_tier_b_log_loss"]
        self.assertIsNotNone(pool_loss)
        self.assertIsNotNone(tier_b_loss)
        expected_strategy = "simplex" if pool_loss < tier_b_loss else "tier_b"
        self.assertEqual(result["no_market_strategy"], expected_strategy)
        self.assertEqual(
            result["no_market_eligibility"]["passed"],
            expected_strategy == "simplex",
        )
        self.assertEqual(
            result["no_market_selection"]["strategy"],
            expected_strategy,
        )
        self.assertEqual(
            result["no_market_strategy"],
            ("simplex" if result["no_market_selection"]["eligible"] else "tier_b"),
        )
        if expected_strategy == "tier_b":
            self.assertEqual(
                result["no_market_selection"]["selected"],
                "tier_b",
            )
            self.assertEqual(
                result["no_market_selected_weights"]["tier_b"],
                1.0,
            )
        else:
            self.assertIn(
                result["no_market_selection"]["selected"],
                {"learned", "tier_a", "tier_b", "tier_c"},
            )

    def test_all_market_rows_still_receive_counterfactual_no_market_evaluation(
        self,
    ):
        result, per_season = self._evaluate_synthetic_season(all_market=True)

        self.assertEqual(result["market_regime"]["games"], per_season)
        self.assertEqual(result["no_market_regime"]["games"], 0)
        self.assertEqual(
            result["no_market_counterfactual"]["games"],
            per_season,
        )
        self.assertEqual(
            len(result["no_market_counterfactual_p"]),
            per_season,
        )
        self.assertTrue(np.isfinite(result["no_market_counterfactual_p"]).all())
        routes = result["no_market_counterfactual_routes"]
        self.assertEqual(routes["market"], 0)
        self.assertEqual(
            routes["no_market_pool"] + routes["tier_b"],
            per_season,
        )

    def test_poor_counterfactual_no_market_output_fails_hard_acceptance(self):
        y = np.array([0, 1] * 40)
        good = np.where(y == 1, 0.8, 0.2)
        poor = 1.0 - good

        pooled = evaluate._pool_probability_results(
            [
                {
                    "model_p": good,
                    "no_market_counterfactual_p": poor,
                    "y_test": y,
                    "tier_a_p": good,
                    "tier_b_p": good,
                    "tier_c_p": good,
                    "market_p": good,
                    "valid_market": np.ones(len(y), dtype=bool),
                }
            ]
        )

        self.assertEqual(pooled["market_regime"]["games"], len(y))
        self.assertEqual(pooled["no_market_regime"]["games"], 0)
        self.assertEqual(
            pooled["no_market_counterfactual"]["games"],
            len(y),
        )
        gate = pooled["acceptance"]["no_market_counterfactual"]
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["accuracy_pass"])
        self.assertFalse(gate["log_loss_pass"])
        self.assertFalse(gate["brier_pass"])
        self.assertTrue(pooled["acceptance"]["global"]["passed"])
        self.assertTrue(pooled["acceptance"]["market_regime"]["passed"])
        self.assertFalse(pooled["acceptance"]["passed"])
        self.assertNotEqual(gate.get("reason"), "no_games")


if __name__ == "__main__":
    unittest.main()
