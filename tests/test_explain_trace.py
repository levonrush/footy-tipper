import unittest

import numpy as np

from pipeline.common.explain import game as xgame
from pipeline.common.explain import trace as xt
from pipeline.common.model_training import calibration as calib


def _stack(weights=("tier_a", "tier_b", "tier_c", "market"), chosen="tier_c",
           temperature=0.9277):
    pool = calib.SimplexLogitPool(include_market="market" in weights)
    pool.expert_names_ = tuple(weights)
    pool.weights_ = np.zeros(len(weights))
    pool.weights_[list(weights).index(chosen)] = 1.0
    pool._is_fitted = True

    calibrator = calib.TemperatureCalibrator()
    calibrator.temperature_ = temperature
    calibrator._is_fitted = True
    return xt.ProbabilityStack(stacker=pool, calibrator=calibrator)


class StackReadingTests(unittest.TestCase):
    def test_weight_map_and_temperature_are_read_from_the_artifacts(self):
        stack = _stack()
        self.assertEqual(stack.weight_map["tier_c"], 1.0)
        self.assertEqual(stack.weight_map["market"], 0.0)
        self.assertAlmostEqual(stack.temperature, 0.9277)
        self.assertEqual(stack.describe()["dominant_expert"], "tier_c")

    def test_chain_multiplier_is_weight_over_temperature(self):
        stack = _stack()
        self.assertAlmostEqual(stack.chain_multiplier, 1.0 / 0.9277, places=9)

    def test_zero_weight_expert_has_a_zero_multiplier(self):
        stack = _stack(chosen="market")
        # The honest answer: no feature of the classifier moves this tip.
        self.assertEqual(xt.chain_multiplier(stack.stacker, stack.calibrator), 0.0)

    def test_simplex_plus_temperature_is_recognised_as_exactly_linear(self):
        stack = _stack()
        self.assertTrue(xt.is_logit_linear(stack.stacker, stack.calibrator))
        self.assertFalse(xt.is_logit_linear(None, None))


class ProbabilityChainTests(unittest.TestCase):
    """The reconstructed chain must reproduce the deployed code's own output."""

    def setUp(self):
        self.stack = _stack()
        self.tier_a = np.array([0.62, 0.31, 0.55])
        self.tier_b = np.array([0.58, 0.40, 0.51])
        self.tier_c = np.array([0.70, 0.28, 0.49])
        self.market = np.array([0.66, 0.35, 0.52])
        self.valid_market = np.array([True, True, False])

    def _regimes(self):
        return calib.predict_probability_regimes(
            tier_a=self.tier_a,
            tier_b=self.tier_b,
            tier_c=self.tier_c,
            market=self.market,
            valid_market=self.valid_market,
            market_stacker=self.stack.stacker,
            market_calibrator=self.stack.calibrator,
        )

    def test_reconstructed_probability_matches_the_deployed_regime_code(self):
        published, routes = self._regimes()
        traces = xt.build_probability_traces(
            game_ids=[1, 2, 3],
            stack=self.stack,
            tier_a=self.tier_a,
            tier_b=self.tier_b,
            tier_c=self.tier_c,
            market=self.market,
            valid_market=self.valid_market,
            routes=routes,
            published_cond=published,
            mu_home=np.array([22.0, 18.0, 20.0]),
            mu_away=np.array([18.0, 22.0, 20.0]),
        )
        for i, trace in enumerate(traces):
            if trace.route != "market":
                continue
            rebuilt = 1.0 / (1.0 + np.exp(-trace.calibrated_logit))
            self.assertAlmostEqual(rebuilt, published[i], places=12)

    def test_expert_terms_sum_to_the_pooled_logit(self):
        published, routes = self._regimes()
        traces = xt.build_probability_traces(
            game_ids=[1, 2, 3],
            stack=self.stack,
            tier_a=self.tier_a,
            tier_b=self.tier_b,
            tier_c=self.tier_c,
            market=self.market,
            valid_market=self.valid_market,
            routes=routes,
            published_cond=published,
        )
        for trace in traces:
            self.assertAlmostEqual(
                sum(trace.expert_logit_terms.values()), trace.pooled_logit, places=12
            )

    def test_per_row_routes_agree_with_the_scalar_counts(self):
        _, routes = self._regimes()
        self.assertEqual(len(routes["row_route"]), 3)
        self.assertEqual(len(routes["row_guarded"]), 3)
        self.assertEqual(routes["row_route"].count("market"), routes["market"])
        self.assertEqual(sum(routes["row_guarded"]), routes["consensus_guarded"])
        # JSON-safe: evaluate serialises this dict into its report.
        self.assertTrue(all(isinstance(r, str) for r in routes["row_route"]))
        self.assertTrue(all(isinstance(g, bool) for g in routes["row_guarded"]))


class AttributionSourceTests(unittest.TestCase):
    """The rule that decides WHICH model a game's drivers come from."""

    def _trace(self, *, route, guarded, chosen="tier_c"):
        stack = _stack(chosen=chosen)
        return xt.build_probability_traces(
            game_ids=[1],
            stack=stack,
            tier_a=np.array([0.6]),
            tier_b=np.array([0.6]),
            tier_c=np.array([0.6]),
            market=np.array([0.6]),
            valid_market=np.array([True]),
            routes={"row_route": [route], "row_guarded": [guarded]},
            published_cond=np.array([0.6]),
            mu_home=np.array([22.0]),
            mu_away=np.array([18.0]),
        )[0]

    def test_normal_market_row_is_attributed_to_the_classifier(self):
        trace = self._trace(route="market", guarded=False)
        self.assertEqual(trace.attribution_source, xt.ATTRIBUTION_BINARY)
        self.assertGreater(trace.feature_multiplier, 0)

    def test_guarded_row_is_attributed_to_the_score_models(self):
        # The guard replaced the pooled result with Tier B, so the classifier
        # did not decide this game and must not be credited with it.
        trace = self._trace(route="market", guarded=True)
        self.assertEqual(trace.attribution_source, xt.ATTRIBUTION_SCORE)

    def test_tier_b_fallback_row_is_attributed_to_the_score_models(self):
        trace = self._trace(route="tier_b", guarded=False)
        self.assertEqual(trace.attribution_source, xt.ATTRIBUTION_SCORE)

    def test_zero_weight_classifier_yields_no_feature_drivers(self):
        trace = self._trace(route="market", guarded=False, chosen="market")
        self.assertEqual(trace.attribution_source, xt.ATTRIBUTION_EXPERTS)
        self.assertEqual(trace.feature_multiplier, 0.0)

    def test_no_market_row_excludes_the_market_term_from_its_logit(self):
        stack = _stack()
        trace = xt.build_probability_traces(
            game_ids=[1],
            stack=stack,
            tier_a=np.array([0.6]),
            tier_b=np.array([0.6]),
            tier_c=np.array([0.6]),
            market=np.array([0.9]),
            valid_market=np.array([False]),
            routes={"row_route": ["no_market_pool"], "row_guarded": [False]},
            published_cond=np.array([0.6]),
        )[0]
        self.assertNotIn("market", trace.expert_logit_terms)


class ProbPerMarginPointTests(unittest.TestCase):
    def test_derivative_is_positive_and_deterministic(self):
        first = xt.prob_per_margin_point(22.0, 18.0)
        second = xt.prob_per_margin_point(22.0, 18.0)
        self.assertEqual(first, second)
        self.assertGreater(first, 0.0)

    def test_derivative_shrinks_as_the_tip_becomes_lopsided(self):
        even = xt.prob_per_margin_point(20.0, 20.0)
        lopsided = xt.prob_per_margin_point(40.0, 6.0)
        self.assertGreater(even, lopsided)


class WhyLineTests(unittest.TestCase):
    def _explanation(self, families, *, tipped_home=True, guard=False, source=None):
        probability = xt.ProbabilityTrace(
            game_id=1,
            tier_a=0.5,
            tier_b=0.5,
            published_cond=0.7 if tipped_home else 0.3,
            guard_fired=guard,
            attribution_source=source or xt.ATTRIBUTION_BINARY,
        )
        drivers = tuple(
            xgame.Driver(key=key, label=label, family=key, points=points, share=0.0)
            for key, label, points in families
        )
        return xgame.GameExplanation(
            game_id=1,
            team_home="Storm",
            team_away="Titans",
            probability=probability,
            score=xt.ScoreTrace(game_id=1),
            prob_families=drivers,
        )

    def test_positive_lead_driver_reads_as_favoured(self):
        why = xgame.one_line_why(
            self._explanation(
                [("ladder", "Ladder and season totals", 9.0), ("weather", "Weather", -2.0)]
            )
        )
        self.assertTrue(why.startswith("Storm favoured on ladder and season totals (+9 pts)"))
        self.assertIn("weather (-2 pts) works against them", why)

    def test_negative_lead_driver_reads_as_tipped_despite(self):
        # A tip carried by the base rate against its own biggest feature must
        # not claim the small positive is what favours it.
        why = xgame.one_line_why(
            self._explanation(
                [("player_form", "Player recent form", -5.0), ("ladder", "Ladder", 2.0)]
            )
        )
        self.assertTrue(why.startswith("Storm tipped despite player recent form (-5 pts)"))
        self.assertIn("ladder (+2 pts) in their favour", why)

    def test_away_tip_names_the_away_team(self):
        why = xgame.one_line_why(
            self._explanation([("ladder", "Ladder", 6.0)], tipped_home=False)
        )
        self.assertTrue(why.startswith("Titans favoured on"))

    def test_guard_override_is_announced(self):
        why = xgame.one_line_why(
            self._explanation([("ladder", "Ladder", 6.0)], guard=True)
        )
        self.assertTrue(why.startswith("Guard override: "))

    def test_market_only_tip_says_so(self):
        why = xgame.one_line_why(
            self._explanation([], source=xt.ATTRIBUTION_EXPERTS)
        )
        self.assertEqual(why, xgame.MARKET_ONLY_WHY)

    def test_nothing_above_the_floor_falls_back(self):
        why = xgame.one_line_why(self._explanation([("ladder", "Ladder", 0.2)]))
        self.assertEqual(why, xgame.NO_DOMINANT_DRIVER)

    def test_clause_count_is_capped(self):
        families = [(f"f{i}", f"Family {i}", 9.0 - i) for i in range(6)]
        why = xgame.one_line_why(self._explanation(families))
        self.assertEqual(why.count("pts)"), 3)


if __name__ == "__main__":
    unittest.main()
