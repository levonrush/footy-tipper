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

        # And the reconciliation scorecard, which prices every way of producing
        # the displayed scoreline in points rather than in CRPS.
        reconciliation = margin_distribution["reconciliation"]
        self.assertEqual(reconciliation["games"], per_season)
        for variant in evaluate.RECONCILIATION_VARIANTS:
            self.assertEqual(reconciliation[variant]["games"], per_season)
            self.assertGreater(reconciliation[variant]["margin_mae"], 0.0)
        # The deployed arrangement must be one of the scored variants, so the
        # report can never describe a configuration it did not measure.
        self.assertIn(reconciliation["deployed"], evaluate.RECONCILIATION_VARIANTS)
        self.assertEqual(
            reconciliation["shipped"], reconciliation[reconciliation["deployed"]]
        )

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

    def test_reweighting_cannot_move_the_displayed_scoreline_but_solving_can(self):
        """Pins the mechanism the reconciliation scorecard replays.

        Importance weights are constant within a side, so the old path could not
        move the displayed scoreline at all for a given side: a 55% home tip and
        an 85% home tip printed the same score. Moving the means can. The two
        mechanisms have to stay distinguishable or the `legacy` column in the
        report is not reproducing the code it claims to.

        `on_conflict`, which is what ships, deliberately declines to move it here
        too. The raw means already put the home side in front, so there is no
        contradiction to fix, and the held-out seasons say the score model is the
        better margin estimator when left alone.
        """
        kwargs = dict(
            n_simulations=40_000, lambda3=0.0, dispersion_home=5.19, dispersion_away=4.34
        )
        home_side = (0.55, 0.70, 0.85)

        def legacy(cond):
            scoreline, _, _, _ = evaluate._legacy_reconciled_prediction(
                22.0, 20.0, cond, rng=pf.rng_for_game(1, salt=1), **kwargs
            )
            return scoreline[0] - scoreline[1]

        def solved(cond, reconcile):
            _, scoreline = pf.simulate_game(
                22.0, 20.0, rng=pf.rng_for_game(1, salt=1), calibrated_cond=cond,
                reconcile=reconcile, **kwargs
            )
            return scoreline[0] - scoreline[1]

        self.assertEqual(len({legacy(c) for c in home_side}), 1)
        self.assertGreater(len({solved(c, "always") for c in home_side}), 1)
        self.assertEqual(len({solved(c, "on_conflict") for c in home_side}), 1)

        # Whatever else moves, no version may contradict the tip.
        for cond in home_side:
            self.assertGreater(legacy(cond), 0)
            self.assertGreater(solved(cond, "always"), 0)
            self.assertGreater(solved(cond, "on_conflict"), 0)

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
