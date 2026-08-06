import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import dill
import numpy as np
import pandas as pd

from pipeline.common.model_training import calibration as calib
from pipeline.common.model_prediciton import prediction_functions as pf
from pipeline.ops import model_release, state_sync


class _PredictModel:
    def predict(self, frame):
        return np.zeros(len(frame))


class _BinaryModel:
    def predict_proba(self, frame):
        return np.column_stack([np.full(len(frame), 0.5), np.full(len(frame), 0.5)])


class _ExplodingLegacyStacker:
    def predict(self, *args, **kwargs):
        raise AssertionError("invalid market rows must not reach the market stacker")


class _SideReversingLegacyStacker:
    def predict(self, *args, **kwargs):
        return np.full(len(args[0]), 0.9)


class SimplexLogitPoolTests(unittest.TestCase):
    def _training_data(self, n=600, seed=31):
        rng = np.random.default_rng(seed)
        tier_a = rng.uniform(0.1, 0.9, n)
        tier_b = rng.uniform(0.1, 0.9, n)
        tier_c = rng.uniform(0.1, 0.9, n)
        market = rng.uniform(0.1, 0.9, n)
        latent = (
            0.15 * calib._safe_logit(tier_a)
            + 0.35 * calib._safe_logit(tier_b)
            + 0.20 * calib._safe_logit(tier_c)
            + 0.30 * calib._safe_logit(market)
        )
        probability = 1.0 / (1.0 + np.exp(-latent))
        y = (rng.uniform(0.0, 1.0, n) < probability).astype(int)
        return tier_a, tier_b, tier_c, market, y

    def test_weights_are_a_nonnegative_simplex(self):
        tier_a, tier_b, tier_c, market, y = self._training_data()
        pool = calib.SimplexLogitPool(include_market=True).fit(
            tier_a, tier_b, y, tier_c=tier_c, market=market
        )

        self.assertTrue(pool._is_fitted)
        self.assertTrue((pool.weights_ >= 0.0).all())
        self.assertAlmostEqual(float(pool.weights_.sum()), 1.0, places=10)
        self.assertEqual(
            tuple(pool.expert_names_), ("tier_a", "tier_b", "tier_c", "market")
        )

    def test_pool_is_neutral_symmetric_and_monotone(self):
        tier_a, tier_b, tier_c, market, y = self._training_data()
        pool = calib.SimplexLogitPool(include_market=True).fit(
            tier_a, tier_b, y, tier_c=tier_c, market=market
        )

        neutral = pool.predict([0.5], [0.5], tier_c=[0.5], market=[0.5])
        self.assertAlmostEqual(float(neutral[0]), 0.5, places=12)

        home = pool.predict([0.62], [0.71], tier_c=[0.58], market=[0.66])
        away = pool.predict([0.38], [0.29], tier_c=[0.42], market=[0.34])
        self.assertAlmostEqual(float(home[0] + away[0]), 1.0, places=12)

        low = pool.predict([0.45], [0.55], tier_c=[0.52], market=[0.60])
        high = pool.predict([0.65], [0.55], tier_c=[0.52], market=[0.60])
        self.assertGreaterEqual(float(high[0]), float(low[0]))

    def test_nested_gate_uses_one_hot_best_expert_when_pool_regresses(self):
        pool = calib.SimplexLogitPool(include_market=True)
        pool.expert_names_ = ("tier_a", "tier_b", "tier_c", "market")
        pool.weights_ = np.full(4, 0.25)
        pool._is_fitted = True
        y = np.array([0, 0, 1, 1] * 40)
        tier_c = np.where(y == 1, 0.8, 0.2)
        experts = {
            "tier_a": np.full(len(y), 0.5),
            "tier_b": np.where(y == 1, 0.65, 0.35),
            "tier_c": tier_c,
            "market": np.where(y == 1, 0.6, 0.4),
        }
        regressing_pool = np.where(y == 1, 0.55, 0.45)

        selection = calib.select_market_pool(pool, regressing_pool, y, experts)

        self.assertEqual(selection["selected"], "tier_c")
        self.assertEqual(pool.weight_map["tier_c"], 1.0)
        self.assertAlmostEqual(sum(pool.weight_map.values()), 1.0)
        learned_calibrator = calib.TemperatureCalibrator().fit(regressing_pool, y)
        selected_calibrator = calib.fit_selected_market_calibrator(
            selection, experts, y, learned_calibrator
        )
        self.assertIsNot(selected_calibrator, learned_calibrator)
        self.assertTrue(selected_calibrator._is_fitted)
        self.assertGreater(selected_calibrator.temperature_, 0.0)
        selected_predictions = selected_calibrator.predict(tier_c)
        self.assertTrue((selected_predictions[y == 0] < 0.5).all())
        self.assertTrue((selected_predictions[y == 1] > 0.5).all())

    def test_rejected_market_pool_uses_strongest_non_market_fallback(self):
        pool = calib.SimplexLogitPool(include_market=True)
        pool.expert_names_ = ("tier_a", "tier_b", "tier_c", "market")
        pool.weights_ = np.full(4, 0.25)
        pool._is_fitted = True
        y = np.array([0, 1] * 80)
        experts = {
            "tier_a": np.where(y == 1, 0.55, 0.45),
            "tier_b": np.where(y == 1, 0.65, 0.35),
            "tier_c": np.where(y == 1, 0.80, 0.20),
            "market": np.where(y == 1, 0.90, 0.10),
        }
        regressing_pool = np.where(y == 1, 0.60, 0.40)

        selection = calib.select_market_pool(
            pool,
            regressing_pool,
            y,
            experts,
        )

        self.assertEqual(selection["best_expert"]["name"], "market")
        self.assertTrue(selection["fallback_applied"])
        self.assertEqual(selection["fallback_expert"]["name"], "tier_c")
        self.assertEqual(
            selection["fallback_reason"],
            "learned_pool_rejected_use_strongest_non_market_expert",
        )
        self.assertEqual(selection["selected"], "tier_c")
        self.assertEqual(pool.weight_map["tier_c"], 1.0)
        self.assertEqual(pool.weight_map["market"], 0.0)

    def test_passing_learned_market_pool_is_not_replaced_by_fallback(self):
        pool = calib.SimplexLogitPool(include_market=True)
        pool.expert_names_ = ("tier_a", "tier_b", "tier_c", "market")
        original_weights = np.array([0.05, 0.10, 0.15, 0.70])
        pool.weights_ = original_weights.copy()
        pool._is_fitted = True
        y = np.array([0, 1] * 80)
        experts = {
            "tier_a": np.where(y == 1, 0.55, 0.45),
            "tier_b": np.where(y == 1, 0.65, 0.35),
            "tier_c": np.where(y == 1, 0.70, 0.30),
            "market": np.where(y == 1, 0.80, 0.20),
        }
        robust_pool = np.where(y == 1, 0.95, 0.05)

        selection = calib.select_market_pool(pool, robust_pool, y, experts)

        self.assertEqual(selection["best_expert"]["name"], "market")
        self.assertFalse(selection["fallback_applied"])
        self.assertEqual(selection["fallback_expert"]["name"], "tier_c")
        self.assertIsNone(selection["fallback_reason"])
        self.assertEqual(selection["selected"], "learned")
        np.testing.assert_allclose(pool.weights_, original_weights)

    def test_eligible_no_market_pool_can_select_one_hot_tier_c(self):
        pool = calib.SimplexLogitPool(include_market=False)
        pool.expert_names_ = ("tier_a", "tier_b", "tier_c")
        pool.weights_ = np.full(3, 1.0 / 3.0)
        pool._is_fitted = True
        y = np.array([0, 1] * 120)
        groups = np.repeat(np.arange(2019, 2025), 40)
        experts = {
            "tier_a": np.where(y == 1, 0.55, 0.45),
            "tier_b": np.where(y == 1, 0.65, 0.35),
            "tier_c": np.where(y == 1, 0.85, 0.15),
        }
        eligible_but_weaker_pool = np.where(y == 1, 0.70, 0.30)

        selection = calib.select_no_market_pool(
            pool,
            eligible_but_weaker_pool,
            y,
            experts,
            groups=groups,
        )

        self.assertTrue(selection["eligible"])
        self.assertTrue(selection["eligibility"]["passed"])
        self.assertLess(
            selection["eligibility"]["pool_log_loss"],
            selection["eligibility"]["tier_b_log_loss"],
        )
        self.assertEqual(selection["strategy"], "simplex")
        self.assertEqual(selection["selected"], "tier_c")
        self.assertEqual(pool.weight_map["tier_c"], 1.0)
        self.assertEqual(len(selection["recent_group_stability"]), 3)
        self.assertFalse(
            all(item["passed"] for item in selection["recent_group_stability"])
        )

        learned_calibrator = calib.TemperatureCalibrator().fit(
            eligible_but_weaker_pool,
            y,
        )
        selected_calibrator = calib.fit_selected_pool_calibrator(
            selection,
            experts,
            y,
            learned_calibrator,
        )
        self.assertIsNot(selected_calibrator, learned_calibrator)
        selected_predictions = selected_calibrator.predict(experts["tier_c"])
        self.assertTrue((selected_predictions[y == 0] < 0.5).all())
        self.assertTrue((selected_predictions[y == 1] > 0.5).all())

    def test_equal_tier_b_loss_is_ineligible_before_stronger_expert_selection(
        self,
    ):
        pool = calib.SimplexLogitPool(include_market=False)
        pool.expert_names_ = ("tier_a", "tier_b", "tier_c")
        pool.weights_ = np.full(3, 1.0 / 3.0)
        pool._is_fitted = True
        y = np.array([0, 1] * 80)
        experts = {
            "tier_a": np.where(y == 1, 0.55, 0.45),
            "tier_b": np.where(y == 1, 0.70, 0.30),
            # Tier C is strongest, but is not eligible for selection until the
            # learned pool first clears the explicit Tier-B eligibility rule.
            "tier_c": np.where(y == 1, 0.90, 0.10),
        }
        equal_tier_b_pool = experts["tier_b"].copy()

        selection = calib.select_no_market_pool(
            pool,
            equal_tier_b_pool,
            y,
            experts,
        )

        self.assertFalse(selection["eligible"])
        self.assertFalse(selection["eligibility"]["passed"])
        self.assertEqual(
            selection["eligibility"]["pool_log_loss"],
            selection["eligibility"]["tier_b_log_loss"],
        )
        self.assertEqual(selection["strategy"], "tier_b")
        self.assertEqual(selection["selected"], "tier_b")
        self.assertEqual(selection["reason"], "pool_did_not_beat_tier_b")
        self.assertEqual(pool.weight_map["tier_b"], 1.0)
        self.assertEqual(selection["recent_group_stability"], [])

    def test_recent_instability_replaces_eligible_learned_pool_with_tier_c(
        self,
    ):
        pool = calib.SimplexLogitPool(include_market=False)
        pool.expert_names_ = ("tier_a", "tier_b", "tier_c")
        pool.weights_ = np.array([0.2, 0.5, 0.3])
        pool._is_fitted = True
        y = np.array([0, 1] * 80)
        groups = np.repeat(
            np.array([2021.0, 2022.0, 2023.0, 2024.0]),
            40,
        )
        confidence = np.where(groups == 2024.0, 0.60, 0.80)
        nested = np.where(y == 1, confidence, 1.0 - confidence)
        experts = {
            "tier_a": np.where(y == 1, 0.55, 0.45),
            "tier_b": np.where(y == 1, 0.60, 0.40),
            "tier_c": np.where(y == 1, 0.65, 0.35),
        }

        selection = calib.select_no_market_pool(
            pool,
            nested,
            y,
            experts,
            groups=groups,
        )

        self.assertTrue(selection["eligible"])
        self.assertLess(
            selection["pool"]["log_loss"],
            selection["best_expert"]["log_loss"]
            - selection["min_log_loss_improvement"],
        )
        self.assertTrue(selection["recent_group_stability"])
        self.assertFalse(selection["recent_group_stability"][-1]["passed"])
        self.assertEqual(
            selection["recent_group_stability"][-1]["group"],
            2024.0,
        )
        self.assertEqual(selection["strategy"], "simplex")
        self.assertEqual(selection["selected"], "tier_c")
        self.assertEqual(pool.weight_map["tier_c"], 1.0)


class ShrinkagePathTests(unittest.TestCase):
    """The mixture path between a learned pool and its one-hot fallback."""

    def _pool(self):
        pool = calib.SimplexLogitPool(include_market=True)
        pool.expert_names_ = ("tier_a", "tier_b", "tier_c", "market")
        pool.weights_ = np.array([0.1, 0.2, 0.4, 0.3])
        pool._is_fitted = True
        return pool

    def test_blend_stays_on_the_simplex(self):
        pool = self._pool()
        learned = pool.weights_.copy()
        for shrinkage in (0.0, 0.25, 0.5, 0.75, 1.0):
            pool.blend_toward_expert("tier_c", learned, shrinkage)
            self.assertTrue((pool.weights_ >= 0.0).all())
            self.assertAlmostEqual(float(pool.weights_.sum()), 1.0, places=12)
            self.assertEqual(pool.expert_names_, ("tier_a", "tier_b", "tier_c", "market"))
            self.assertTrue(pool._is_fitted)

    def test_zero_shrinkage_matches_select_expert_exactly(self):
        pool = self._pool()
        learned = pool.weights_.copy()
        pool.blend_toward_expert("tier_c", learned, 0.0)
        shrunk = pool.weights_.copy()

        reference = self._pool()
        reference.select_expert("tier_c")

        np.testing.assert_array_equal(shrunk, reference.weights_)

    def test_full_shrinkage_returns_the_learned_weights(self):
        pool = self._pool()
        learned = pool.weights_.copy()
        pool.blend_toward_expert("tier_c", learned, 1.0)
        np.testing.assert_allclose(pool.weights_, learned, atol=1e-12)

    def test_intermediate_shrinkage_lies_between_the_endpoints(self):
        n = 200
        rng = np.random.default_rng(7)
        tier_a = rng.uniform(0.2, 0.8, n)
        tier_b = rng.uniform(0.2, 0.8, n)
        tier_c = np.full(n, 0.3)
        market = np.full(n, 0.8)

        pool = self._pool()
        learned = pool.weights_.copy()

        pool.blend_toward_expert("tier_c", learned, 1.0)
        full = pool.predict(tier_a, tier_b, tier_c=tier_c, market=market)
        pool.blend_toward_expert("tier_c", learned, 0.0)
        none_ = pool.predict(tier_a, tier_b, tier_c=tier_c, market=market)
        pool.blend_toward_expert("tier_c", learned, 0.5)
        half = pool.predict(tier_a, tier_b, tier_c=tier_c, market=market)

        between = ((half > np.minimum(full, none_)) & (half < np.maximum(full, none_)))
        self.assertTrue(between.all())

    def test_invalid_shrinkage_is_rejected(self):
        pool = self._pool()
        learned = pool.weights_.copy()
        with self.assertRaises(ValueError):
            pool.blend_toward_expert("tier_c", learned, 1.5)
        with self.assertRaises(ValueError):
            pool.blend_toward_expert("nope", learned, 0.5)

    def test_path_at_one_matches_the_unpathed_nested_predictions(self):
        rng = np.random.default_rng(11)
        groups = np.repeat(np.arange(2020, 2026, dtype=float), 70)
        n = len(groups)
        tier_a = rng.uniform(0.15, 0.85, n)
        tier_b = np.clip(tier_a + rng.normal(0, 0.08, n), 0.01, 0.99)
        tier_c = np.clip(tier_a + rng.normal(0, 0.10, n), 0.01, 0.99)
        y = (rng.uniform(0, 1, n) < tier_a).astype(int)

        plain = calib.nested_loso_simplex_predictions(
            tier_a, tier_b, y, groups, tier_c=tier_c, include_market=False
        )
        path = calib.nested_loso_simplex_predictions(
            tier_a,
            tier_b,
            y,
            groups,
            tier_c=tier_c,
            include_market=False,
            shrinkage_grid=calib.DEFAULT_SHRINKAGE_GRID,
            fallback_expert="tier_c",
        )

        self.assertIsNotNone(plain)
        self.assertIsNotNone(path)
        np.testing.assert_allclose(path[1.0], plain, atol=1e-12)

    def test_path_predictions_ignore_held_season_labels(self):
        rng = np.random.default_rng(13)
        groups = np.repeat(np.arange(2020, 2026, dtype=float), 70)
        n = len(groups)
        tier_a = rng.uniform(0.15, 0.85, n)
        tier_b = np.clip(tier_a + rng.normal(0, 0.08, n), 0.01, 0.99)
        tier_c = np.clip(tier_a + rng.normal(0, 0.10, n), 0.01, 0.99)
        y = (rng.uniform(0, 1, n) < tier_a).astype(int)
        hold = groups == 2025

        def run(labels):
            return calib.nested_loso_simplex_predictions(
                tier_a,
                tier_b,
                labels,
                groups,
                tier_c=tier_c,
                include_market=False,
                shrinkage_grid=calib.DEFAULT_SHRINKAGE_GRID,
                fallback_expert="tier_c",
            )

        flipped = y.copy()
        flipped[hold] = 1 - flipped[hold]
        original, altered = run(y), run(flipped)

        for shrinkage in original:
            np.testing.assert_allclose(
                original[shrinkage][hold], altered[shrinkage][hold], atol=1e-12
            )


class CompObjectiveTests(unittest.TestCase):
    """P(finish first) as the selection objective."""

    def _season(self, seed=3, n=180):
        rng = np.random.default_rng(seed)
        market = rng.uniform(0.2, 0.8, n)
        market = np.where(np.isclose(market, 0.5), 0.55, market)
        y = (rng.uniform(0, 1, n) < market).astype(int)
        return market, y

    def test_p_first_is_monotone_in_tips_correct_within_a_season(self):
        """Rival scores do not depend on our tips, so more correct is always better.

        This is why differentiating for its own sake buys nothing at selection
        time, and why the in-season points gap logic belongs in comp_strategy.
        """
        market, y = self._season()
        groups = np.full(len(y), 2025.0)

        scores = []
        for wrong in (0, 10, 25, 45):
            probabilities = np.where(y == 1, 0.8, 0.2)
            probabilities[:wrong] = 1.0 - probabilities[:wrong]
            placement = calib.comp_placement_metrics(probabilities, market, y, groups)
            scores.append((placement["tips_correct"], placement["mean_p_first"]))

        for (tips_a, p_a), (tips_b, p_b) in zip(scores, scores[1:]):
            self.assertGreater(tips_a, tips_b)
            self.assertGreaterEqual(p_a, p_b)

    def test_mean_p_first_is_not_the_same_as_pooled_accuracy(self):
        """A spiky candidate can beat a flat one that gets the same tips right.

        P(first) saturates, so clearing the field once is worth more than being
        mediocre twice. That is the whole reason seasons are scored separately
        and then averaged rather than pooled.
        """
        market, y = self._season(seed=5, n=200)
        groups = np.where(np.arange(len(y)) < 100, 2024.0, 2025.0)
        correct = np.where(y == 1, 0.8, 0.2)
        wrong = 1.0 - correct

        # Both candidates get exactly the same number of tips right overall.
        spiky = correct.copy()
        spiky[:40] = wrong[:40]
        flat = correct.copy()
        flat[:20] = wrong[:20]
        flat[100:120] = wrong[100:120]

        spiky_metrics = calib.comp_placement_metrics(spiky, market, y, groups)
        flat_metrics = calib.comp_placement_metrics(flat, market, y, groups)

        self.assertEqual(spiky_metrics["tips_correct"], flat_metrics["tips_correct"])
        self.assertNotAlmostEqual(
            spiky_metrics["mean_p_first"], flat_metrics["mean_p_first"], places=6
        )

    def test_games_without_market_are_excluded_not_fabricated(self):
        market, y = self._season(seed=9)
        groups = np.full(len(y), 2025.0)
        probabilities = np.where(y == 1, 0.8, 0.2)
        holed = market.copy()
        holed[:30] = np.nan

        full = calib.comp_placement_metrics(probabilities, market, y, groups)
        partial = calib.comp_placement_metrics(probabilities, holed, y, groups)

        self.assertEqual(partial["excluded_no_market"], 30)
        self.assertEqual(partial["games"], full["games"] - 30)


class ShrinkageSelectionTests(unittest.TestCase):
    """The gate now chooses how much of the pool to keep, on comp evidence."""

    def _pool(self):
        pool = calib.SimplexLogitPool(include_market=True)
        pool.expert_names_ = ("tier_a", "tier_b", "tier_c", "market")
        pool.weights_ = np.array([0.05, 0.10, 0.55, 0.30])
        pool._is_fitted = True
        return pool

    def _fixture(self, n=400, seed=17):
        """Four 100-game seasons where candidates differ in which tips they miss.

        Misses are spread evenly across seasons (by position within the season)
        so no single season carries all of a candidate's errors.
        """
        rng = np.random.default_rng(seed)
        groups = np.repeat(np.array([2023.0, 2024.0, 2025.0, 2026.0]), n // 4)
        y = rng.integers(0, 2, n)
        within_season = np.arange(n) % (n // 4)

        def forecaster(miss_below, confidence=0.65):
            probabilities = np.where(y == 1, confidence, 1.0 - confidence)
            missed = within_season < miss_below
            probabilities[missed] = 1.0 - probabilities[missed]
            return probabilities

        experts = {
            "tier_a": forecaster(44, 0.54),
            "tier_b": forecaster(40, 0.58),
            "tier_c": forecaster(36, 0.64),
            # Wrong on a mostly disjoint set, as a real market would be.
            "market": np.where(
                within_season >= 62,
                np.where(y == 1, 0.38, 0.62),
                np.where(y == 1, 0.62, 0.38),
            ),
        }
        return groups, y, experts["market"], experts, forecaster

    def test_a_shrunk_pool_can_be_deployed(self):
        """The defect this change exists for.

        The learned pool is strong but not by the old margin; previously that
        collapsed the artifact onto one expert. Now an intermediate rung can be
        deployed, keeping some market weight instead of discarding all of it.
        """
        groups, y, market, experts, forecaster = self._fixture()
        pool = self._pool()
        # Each rung up the path misses fewer games than the one below it.
        path = {
            0.0: experts["tier_c"],
            0.5: forecaster(33, 0.66),
            1.0: forecaster(30, 0.68),
        }

        selection = calib.select_market_pool(
            pool, path, y, experts, groups=groups, market_probabilities=market
        )

        self.assertIn(selection["selected_shrinkage"], (0.0, 0.5, 1.0))
        self.assertGreater(selection["selected_shrinkage"], 0.0)
        self.assertEqual(selection["objective"], "p_first")
        self.assertEqual(selection["shrinkage_target"], "tier_c")
        # Policy from #38 survives: market is never the deployed model alone.
        self.assertNotEqual(selection["selected"], "market")
        self.assertLess(pool.weight_map["market"], 1.0)

    def test_shrinkage_target_is_chosen_on_log_loss_not_p_first(self):
        """The safety default must not ride on a noisy statistic.

        A walk-forward fold once promoted Tier A as the target on a 0.024
        P(first) edge while it was 0.15 worse on log loss, and the model
        regressed. Choosing how far to move from the default is a different
        question from choosing the default.
        """
        groups, y, market, experts, forecaster = self._fixture()
        # Tier A clears the field in one season and is dreadful in the rest:
        # P(first) 0.25 against tier_c's 0.02, log loss 0.72 against 0.65.
        experts["tier_a"] = np.where(
            groups == 2026.0, forecaster(20, 0.66), forecaster(65, 0.60)
        )
        pool = self._pool()
        path = {0.0: experts["tier_c"], 1.0: forecaster(30, 0.68)}

        selection = calib.select_market_pool(
            pool, path, y, experts, groups=groups, market_probabilities=market
        )

        self.assertEqual(selection["shrinkage_target"], "tier_c")
        self.assertEqual(
            calib.strongest_deployable_expert(experts, y, groups, market=market),
            "tier_c",
        )

    def test_comparison_bar_is_the_best_deployable_expert(self):
        groups, y, market, experts, forecaster = self._fixture()
        # Make the market the strongest single expert overall.
        experts["market"] = forecaster(20, 0.80)
        pool = self._pool()
        path = {0.0: experts["tier_c"], 1.0: forecaster(30, 0.68)}

        selection = calib.select_market_pool(
            pool, path, y, experts, groups=groups, market_probabilities=experts["market"]
        )

        self.assertEqual(selection["best_expert"]["name"], "market")
        self.assertEqual(selection["best_deployable_expert"]["name"], "tier_c")
        self.assertNotEqual(selection["selected"], "market")

    def test_parsimony_keeps_the_status_quo_when_the_edge_is_trivial(self):
        groups, y, market, experts, _ = self._fixture()
        pool = self._pool()
        # Every rung is effectively the fallback, so the smallest wins.
        path = {
            0.0: experts["tier_c"],
            0.5: experts["tier_c"].copy(),
            1.0: experts["tier_c"].copy(),
        }

        selection = calib.select_market_pool(
            pool, path, y, experts, groups=groups, market_probabilities=market
        )

        self.assertEqual(selection["selected_shrinkage"], 0.0)
        self.assertEqual(selection["selected"], "tier_c")
        self.assertEqual(pool.weight_map["tier_c"], 1.0)

    def test_calibration_guard_blocks_a_badly_calibrated_winner(self):
        groups, y, market, experts, forecaster = self._fixture()
        pool = self._pool()
        # More tips right than tier_c, but ruinously overconfident on the ones
        # it misses, so the log-loss and Brier guard should refuse it.
        reckless = forecaster(25, 0.99)
        path = {0.0: experts["tier_c"], 1.0: reckless}

        selection = calib.select_market_pool(
            pool, path, y, experts, groups=groups, market_probabilities=market
        )

        blocked = [row for row in selection["path"] if row["shrinkage"] == 1.0][0]
        self.assertFalse(blocked["admissible"])
        self.assertFalse(blocked["calibration_guard"]["passed"])
        self.assertEqual(selection["selected_shrinkage"], 0.0)

    def test_legacy_array_still_produces_a_binary_decision(self):
        groups, y, market, experts, _ = self._fixture()
        pool = self._pool()
        nested = np.where(y == 1, 0.66, 0.34)

        selection = calib.select_market_pool(
            pool, nested, y, experts, groups=groups, objective="log_loss"
        )

        self.assertEqual(selection["objective"], "log_loss")
        self.assertIn(selection["selected_shrinkage"], (0.0, 1.0))
        self.assertIn(selection["selected"], ("learned", "tier_c"))

    def test_selected_calibrator_uses_the_path_predictions(self):
        groups, y, market, experts, _ = self._fixture()
        pool = self._pool()
        path = {0.0: experts["tier_c"], 1.0: np.where(y == 1, 0.70, 0.30)}
        selection = calib.select_market_pool(
            pool, path, y, experts, groups=groups, market_probabilities=market
        )

        learned_calibrator = calib.TemperatureCalibrator().fit(path[1.0], y)
        chosen = calib.fit_selected_pool_calibrator(
            selection, experts, y, learned_calibrator, loso_path_predictions=path
        )
        expected = calib.TemperatureCalibrator().fit(
            path[selection["selected_shrinkage"]], y
        )
        self.assertAlmostEqual(chosen.temperature_, expected.temperature_, places=9)

    def test_zero_rung_calibration_matches_the_in_sample_expert_fit(self):
        """The unification is behaviour-preserving at the status quo.

        A one-hot pool has no fitted parameters, so its held-out prediction is
        the expert's own probability and the two fits coincide.
        """
        groups, y, market, experts, _ = self._fixture()
        pool = self._pool()
        path = {0.0: experts["tier_c"], 1.0: experts["tier_c"].copy()}
        selection = calib.select_market_pool(
            pool, path, y, experts, groups=groups, market_probabilities=market
        )
        self.assertEqual(selection["selected_shrinkage"], 0.0)

        with_path = calib.fit_selected_pool_calibrator(
            selection, experts, y, calib.TemperatureCalibrator(), loso_path_predictions=path
        )
        without_path = calib.fit_selected_pool_calibrator(
            selection, experts, y, calib.TemperatureCalibrator()
        )
        self.assertAlmostEqual(
            with_path.temperature_, without_path.temperature_, places=9
        )


class TemperatureCalibratorTests(unittest.TestCase):
    def test_temperature_is_positive_neutral_and_side_preserving(self):
        probs = np.array([0.05, 0.15, 0.3, 0.45, 0.55, 0.7, 0.85, 0.95])
        y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
        calibrator = calib.TemperatureCalibrator().fit(probs, y)
        predictions = calibrator.predict(np.array([0.2, 0.5, 0.8]))

        self.assertGreater(calibrator.temperature_, 0.0)
        self.assertLess(predictions[0], 0.5)
        self.assertAlmostEqual(float(predictions[1]), 0.5, places=12)
        self.assertGreater(predictions[2], 0.5)
        self.assertAlmostEqual(float(predictions[0] + predictions[2]), 1.0, places=12)


class ReleaseAcceptanceTests(unittest.TestCase):
    def test_regime_fails_when_expert_exceeds_release_tolerances(self):
        result = calib.acceptance_against_experts(
            {"accuracy": 0.62, "log_loss": 0.651, "brier": 0.231},
            {
                "tier_c": {
                    "accuracy": 0.64,
                    "log_loss": 0.640,
                    "brier": 0.220,
                },
                "market": {
                    "accuracy": 0.63,
                    "log_loss": 0.645,
                    "brier": 0.224,
                },
            },
        )

        self.assertFalse(result["passed"])
        self.assertFalse(result["accuracy_pass"])
        self.assertFalse(result["log_loss_pass"])
        self.assertFalse(result["brier_pass"])

    def test_regime_passes_at_configured_tolerance_boundary(self):
        result = calib.acceptance_against_experts(
            {"accuracy": 0.63, "log_loss": 0.645, "brier": 0.225},
            {
                "tier_c": {
                    "accuracy": 0.64,
                    "log_loss": 0.640,
                    "brier": 0.220,
                }
            },
        )
        self.assertTrue(result["passed"])


class NestedSelectionTests(unittest.TestCase):
    def test_held_season_labels_cannot_change_nested_predictions(self):
        rng = np.random.default_rng(41)
        groups = np.repeat(np.arange(2021, 2026, dtype=float), 70)
        n = len(groups)
        tier_a = rng.uniform(0.15, 0.85, n)
        tier_b = np.clip(tier_a + rng.normal(0, 0.08, n), 0.01, 0.99)
        tier_c = np.clip(tier_a + rng.normal(0, 0.1, n), 0.01, 0.99)
        y = (rng.uniform(0, 1, n) < tier_a).astype(int)
        hold = groups == 2025

        predictions = calib.nested_loso_simplex_predictions(
            tier_a,
            tier_b,
            y,
            groups,
            tier_c=tier_c,
            include_market=False,
        )
        flipped_y = y.copy()
        flipped_y[hold] = 1 - flipped_y[hold]
        flipped_predictions = calib.nested_loso_simplex_predictions(
            tier_a,
            tier_b,
            flipped_y,
            groups,
            tier_c=tier_c,
            include_market=False,
        )

        self.assertIsNotNone(predictions)
        np.testing.assert_allclose(
            predictions[hold], flipped_predictions[hold], atol=1e-12
        )


class ProbabilityRoutingTests(unittest.TestCase):
    def test_missing_market_probability_is_nan_not_fabricated_neutral(self):
        frame = pd.DataFrame(
            {
                "team_head_to_head_odds_home": [1.91, 0.0, np.nan],
                "team_head_to_head_odds_away": [1.91, 0.0, 2.10],
                # Derived placeholders must not override invalid raw prices.
                "home_market_prob_basic": [0.5, 0.5, 0.5],
            }
        )

        probabilities = pf.derive_market_home_probability(frame)

        self.assertAlmostEqual(float(probabilities[0]), 0.5)
        self.assertTrue(np.isnan(probabilities[1]))
        self.assertTrue(np.isnan(probabilities[2]))

    def test_missing_market_never_calls_legacy_market_stacker(self):
        result, routes = calib.predict_probability_regimes(
            tier_a=np.array([0.2, 0.7]),
            tier_b=np.array([0.25, 0.65]),
            tier_c=np.array([0.3, 0.6]),
            market=np.array([0.5, 0.5]),
            valid_market=np.array([False, False]),
            market_stacker=_ExplodingLegacyStacker(),
            market_calibrator=None,
            no_market_strategy="tier_b",
        )

        np.testing.assert_allclose(result, [0.25, 0.65])
        self.assertEqual(routes["market"], 0)
        self.assertEqual(routes["tier_b"], 2)

    def test_legacy_side_reversal_is_replaced_by_tier_b(self):
        result, routes = calib.predict_probability_regimes(
            tier_a=np.array([0.2]),
            tier_b=np.array([0.25]),
            tier_c=np.array([0.3]),
            market=np.array([0.35]),
            valid_market=np.array([True]),
            market_stacker=_SideReversingLegacyStacker(),
            market_calibrator=None,
            no_market_strategy="tier_b",
        )

        np.testing.assert_allclose(result, [0.25])
        self.assertEqual(routes["consensus_guarded"], 1)

    def test_eels_panthers_and_bulldogs_warriors_cannot_flip_home(self):
        fixtures = (
            "Parramatta Eels vs Penrith Panthers",
            "Canterbury-Bankstown Bulldogs vs New Zealand Warriors",
        )
        tier_a = np.array([0.021, 0.087])
        tier_b = np.array([0.170, 0.307])
        tier_c = np.array([0.356, 0.397])
        market = np.array([0.167, 0.397])
        valid_market = np.array([True, True])

        # Legacy compatibility path: even a pathological old artifact that
        # returns 90% home is stopped by the unanimous-away consensus guard.
        legacy, legacy_routes = calib.predict_probability_regimes(
            tier_a=tier_a,
            tier_b=tier_b,
            tier_c=tier_c,
            market=market,
            valid_market=valid_market,
            market_stacker=_SideReversingLegacyStacker(),
            market_calibrator=None,
            no_market_strategy="tier_b",
        )

        # V3 path: every nonnegative simplex weighting of unanimous-away
        # experts remains on the away side.
        market_pool = calib.SimplexLogitPool(include_market=True)
        market_pool.expert_names_ = ("tier_a", "tier_b", "tier_c", "market")
        market_pool.weights_ = np.full(4, 0.25)
        market_pool._is_fitted = True
        v3, _ = calib.predict_probability_regimes(
            tier_a=tier_a,
            tier_b=tier_b,
            tier_c=tier_c,
            market=market,
            valid_market=valid_market,
            market_stacker=market_pool,
            market_calibrator=calib.TemperatureCalibrator(),
            no_market_strategy="tier_b",
        )

        for fixture, legacy_probability, v3_probability in zip(fixtures, legacy, v3):
            with self.subTest(fixture=fixture):
                self.assertLess(legacy_probability, 0.5)
                self.assertLess(v3_probability, 0.5)
        self.assertEqual(legacy_routes["consensus_guarded"], 2)

    def test_valid_h2h_requires_raw_two_sided_prices_and_missing_flag(self):
        frame = pd.DataFrame(
            {
                "game_id": [101, 102, 103, 104],
                "team_head_to_head_odds_home": [1.91, 0.0, 1.91, 1.91],
                "team_head_to_head_odds_away": [1.91, 1.91, np.nan, 1.91],
                "odds_missing": [0, 0, 0, 1],
                "home_market_prob_basic": [0.5, 0.5, 0.5, 0.5],
            }
        )
        np.testing.assert_array_equal(
            calib.valid_h2h_mask(frame),
            np.array([True, False, False, False]),
        )
        np.testing.assert_array_equal(
            calib.valid_fresh_h2h_mask(frame, fresh_game_ids=(102,)),
            np.array([False, False, False, False]),
        )
        np.testing.assert_array_equal(
            calib.valid_fresh_h2h_mask(frame, fresh_game_ids=(101,)),
            np.array([True, False, False, False]),
        )


class ProbabilityArtifactContractTests(unittest.TestCase):
    def _write_base_models(self, root):
        with open(root / "home_model.pkl", "wb") as handle:
            dill.dump(_PredictModel(), handle)
        with open(root / "away_model.pkl", "wb") as handle:
            dill.dump(_PredictModel(), handle)
        with open(root / "binary_model.pkl", "wb") as handle:
            dill.dump(_BinaryModel(), handle)

    def _manifest(self):
        return {
            "predictors": ["round_id"],
            "probability_stack": {
                "schema_version": 3,
                "market": {
                    "strategy": "simplex",
                    "stacker_file": "stacker.pkl",
                    "calibrator_file": "win_prob_calibrator.pkl",
                    "experts": ["tier_a", "tier_b", "tier_c", "market"],
                    "weights": {
                        "tier_a": 0.25,
                        "tier_b": 0.25,
                        "tier_c": 0.25,
                        "market": 0.25,
                    },
                    "temperature": 1.0,
                },
                "no_market": {
                    "strategy": "tier_b",
                    "stacker_file": "stacker_no_market.pkl",
                    "calibrator_file": "win_prob_calibrator_no_market.pkl",
                    "selection": {
                        "strategy": "tier_b",
                        "selected": "tier_b",
                        "eligible": False,
                        "reason": "pool_did_not_beat_tier_b",
                        "eligibility": {
                            "criterion": "pool_log_loss < tier_b_log_loss",
                            "passed": False,
                            "pool_log_loss": 0.66,
                            "tier_b_log_loss": 0.65,
                        },
                    },
                    "experts": ["tier_a", "tier_b", "tier_c"],
                    "weights": {
                        "tier_a": 0.0,
                        "tier_b": 1.0,
                        "tier_c": 0.0,
                    },
                    "temperature": 1.0,
                },
            },
        }

    def _write_probability_artifacts(self, root):
        market_pool = calib.SimplexLogitPool(include_market=True)
        market_pool.expert_names_ = ("tier_a", "tier_b", "tier_c", "market")
        market_pool.weights_ = np.full(4, 0.25)
        market_pool._is_fitted = True
        no_market_pool = calib.SimplexLogitPool(include_market=False)
        no_market_pool.expert_names_ = ("tier_a", "tier_b", "tier_c")
        no_market_pool.weights_ = np.array([0.0, 1.0, 0.0])
        no_market_pool._is_fitted = True
        market_calibrator = calib.TemperatureCalibrator()
        market_calibrator.temperature_ = 1.0
        market_calibrator._is_fitted = True
        no_market_calibrator = calib.TemperatureCalibrator()
        no_market_calibrator.temperature_ = 1.0
        no_market_calibrator._is_fitted = True
        for name, artifact in (
            ("stacker.pkl", market_pool),
            ("win_prob_calibrator.pkl", market_calibrator),
            ("stacker_no_market.pkl", no_market_pool),
            (
                "win_prob_calibrator_no_market.pkl",
                no_market_calibrator,
            ),
        ):
            with open(root / name, "wb") as handle:
                dill.dump(artifact, handle)

    def test_v3_manifest_rejects_missing_no_market_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_base_models(root)
            (root / "model_manifest.json").write_text(
                json.dumps(self._manifest()), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "missing required artifacts"):
                state_sync._validate_model_artifacts(root)

    def test_v3_manifest_accepts_complete_probability_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_base_models(root)
            self._write_probability_artifacts(root)
            (root / "model_manifest.json").write_text(
                json.dumps(self._manifest()), encoding="utf-8"
            )

            state_sync._validate_model_artifacts(root)

    def test_v3_manifest_accepts_a_shrunk_market_pool(self):
        """A partially shrunk pool is still a fitted simplex, so v3 stands.

        Shrinkage adds manifest keys only; it does not change the artifact
        contract, which is why no schema bump is needed and why every already
        published release stays activatable.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_base_models(root)
            self._write_probability_artifacts(root)

            pool = calib.SimplexLogitPool(include_market=True)
            pool.expert_names_ = ("tier_a", "tier_b", "tier_c", "market")
            pool.weights_ = np.array([0.05, 0.10, 0.55, 0.30])
            pool._is_fitted = True
            learned = pool.weights_.copy()
            pool.blend_toward_expert("tier_c", learned, 0.5)
            with open(root / "stacker.pkl", "wb") as handle:
                dill.dump(pool, handle)

            manifest = self._manifest()
            market = manifest["probability_stack"]["market"]
            market["weights"] = pool.weight_map
            market["learned_weights"] = {
                name: float(weight)
                for name, weight in zip(pool.expert_names_, learned)
            }
            market["selected_shrinkage"] = 0.5
            market["shrinkage_target"] = "tier_c"
            (root / "model_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            state_sync._validate_model_artifacts(root)

    def test_v3_manifest_rejects_non_simplex_artifact_weights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_base_models(root)
            self._write_probability_artifacts(root)
            with open(root / "stacker.pkl", "rb") as handle:
                pool = dill.load(handle)
            pool.weights_ = np.array([0.7, 0.4, -0.1, 0.0])
            with open(root / "stacker.pkl", "wb") as handle:
                dill.dump(pool, handle)
            (root / "model_manifest.json").write_text(
                json.dumps(self._manifest()), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "simplex weights"):
                state_sync._validate_model_artifacts(root)

    def test_v3_manifest_rejects_wrong_simplex_regime_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_base_models(root)
            self._write_probability_artifacts(root)
            with open(root / "stacker_no_market.pkl", "rb") as handle:
                pool = dill.load(handle)
            pool.include_market = True
            pool.expert_names_ = ("tier_a", "tier_b", "market")
            with open(root / "stacker_no_market.pkl", "wb") as handle:
                dill.dump(pool, handle)
            (root / "model_manifest.json").write_text(
                json.dumps(self._manifest()), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "fitted simplex contract"):
                state_sync._validate_model_artifacts(root)

    def test_v3_manifest_rejects_invalid_temperature_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_base_models(root)
            self._write_probability_artifacts(root)
            with open(root / "win_prob_calibrator.pkl", "rb") as handle:
                calibrator = dill.load(handle)
            calibrator.temperature_ = 0.0
            with open(root / "win_prob_calibrator.pkl", "wb") as handle:
                dill.dump(calibrator, handle)
            (root / "model_manifest.json").write_text(
                json.dumps(self._manifest()), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "temperature"):
                state_sync._validate_model_artifacts(root)

    def test_training_receipt_hashes_new_probability_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            models = root / "models"
            models.mkdir()
            self._write_base_models(models)
            self._write_probability_artifacts(models)
            (models / "model_manifest.json").write_text(
                json.dumps(self._manifest()), encoding="utf-8"
            )
            db_path = root / "training.sqlite"
            with sqlite3.connect(db_path) as con:
                con.execute(
                    "CREATE TABLE footy_tipping_data "
                    "(competition_year INTEGER, game_state_name TEXT)"
                )
                con.execute(
                    "INSERT INTO footy_tipping_data VALUES (?, ?)",
                    (2026, "Final"),
                )

            receipt = model_release._build_receipt(
                models,
                db_path,
                release_id="test-release",
                git_sha="abc123",
                tuning_candidates=1,
                source="test",
            )

            for name in (
                "stacker_no_market.pkl",
                "win_prob_calibrator_no_market.pkl",
            ):
                self.assertIn(name, receipt["artifacts"])
                self.assertEqual(len(receipt["artifacts"][name]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
