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


class CompPlacementMetricsTests(unittest.TestCase):
    """The competition scoreboard. Reported, never used to select."""

    def _season(self, seed=3, n=180):
        rng = np.random.default_rng(seed)
        market = rng.uniform(0.2, 0.8, n)
        market = np.where(np.isclose(market, 0.5), 0.55, market)
        y = (rng.uniform(0, 1, n) < market).astype(int)
        return market, y

    def test_p_first_is_monotone_in_tips_correct_within_a_season(self):
        """Rival scores do not depend on our tips, so more correct is better.

        Worth pinning: it is the reason differentiating for its own sake buys
        nothing at selection time, and why the in-season case for it belongs in
        comp_strategy, which knows the live points gap.
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
        """P(first) saturates, so seasons are averaged rather than pooled."""
        market, y = self._season(seed=5, n=200)
        groups = np.where(np.arange(len(y)) < 100, 2024.0, 2025.0)
        correct = np.where(y == 1, 0.8, 0.2)
        wrong = 1.0 - correct

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
