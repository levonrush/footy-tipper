import dataclasses
import unittest

import numpy as np
import pandas as pd

from pipeline.common.explain import cohort as xco
from pipeline.common.explain import units


def _inputs(n=1200, seed=7):
    """A cohort with one honest family, one loud liar, and one mute family.

    * elo              carries the real signal
    * team_match_stats is loud but random, so it must not rank as skill
    * broadcast        contributes nothing at all
    """
    rng = np.random.default_rng(seed)
    feature_names = ("home_elo", "away_elo", "points_home_performance", "broadcast_channel1")

    signal = rng.normal(0, 1.2, n)
    y = (signal + rng.normal(0, 0.6, n) > 0).astype(int)

    prob_logit = np.column_stack(
        [
            0.5 * signal,  # home_elo: points at the winner
            0.5 * signal,  # away_elo
            rng.normal(0, 1.5, n),  # loud, but unrelated to y
            np.zeros(n),  # never speaks
        ]
    )
    base = np.zeros(n)
    p_model = units.sigmoid(base + prob_logit.sum(axis=1))

    home_log_mu = np.column_stack(
        [0.02 * signal, np.zeros(n), rng.normal(0, 0.02, n), np.zeros(n)]
    )
    away_log_mu = np.zeros_like(home_log_mu)
    home_base = np.full(n, np.log(22.0))
    away_base = np.full(n, np.log(20.0))
    mu_home = np.exp(home_base + home_log_mu.sum(axis=1))
    mu_away = np.exp(away_base + away_log_mu.sum(axis=1))

    actual_margin = mu_home - mu_away + rng.normal(0, 8.0, n)

    return xco.CohortInputs(
        source=xco.SOURCE_IN_SAMPLE,
        feature_names=feature_names,
        prob_logit=prob_logit,
        prob_base=base,
        home_log_mu=home_log_mu,
        away_log_mu=away_log_mu,
        home_base=home_base,
        away_base=away_base,
        p_model=p_model,
        y=y,
        non_draw=np.ones(n, dtype=bool),
        mu_home=mu_home,
        mu_away=mu_away,
        actual_margin=actual_margin,
        split_counts={
            "home_elo": 120,
            "away_elo": 90,
            "points_home_performance": 45,
            "broadcast_channel1": 0,
        },
        frame=pd.DataFrame(
            {
                "performance_features_missing": rng.integers(0, 2, n).astype(float),
                "weather_missing": np.zeros(n),
            }
        ),
    )


class WilsonIntervalTests(unittest.TestCase):
    def test_interval_brackets_the_point_estimate(self):
        lo, hi = xco.wilson_interval(60, 100)
        self.assertLess(lo, 0.6)
        self.assertGreater(hi, 0.6)

    def test_empty_sample_is_nan_not_a_crash(self):
        lo, hi = xco.wilson_interval(0, 0)
        self.assertTrue(np.isnan(lo) and np.isnan(hi))

    def test_interval_stays_inside_the_unit_range(self):
        lo, hi = xco.wilson_interval(5, 5)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)


class FamilySignalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = _inputs()
        cls.signal = xco.family_signal(cls.inputs)

    def _row(self, family):
        return self.signal.set_index("family").loc[family]

    def test_planted_signal_family_ranks_first_on_skill(self):
        self.assertEqual(self.signal.iloc[0]["family"], "elo")
        self.assertGreater(self._row("elo")["lift_log_loss"], 0.05)
        self.assertEqual(self._row("elo")["direction"], "helps")

    def test_loud_but_random_family_shows_no_demonstrated_direction(self):
        loud = self._row("team_match_stats")
        # It speaks louder than the family that actually carries the signal.
        self.assertGreater(
            loud["mean_abs_prob_points"], self._row("elo")["mean_abs_prob_points"]
        )
        # Volume is not skill, and the CI is what says so.
        self.assertEqual(loud["direction"], "unclear")
        # Stronger than "no skill": loud noise is confidently wrong half the
        # time, so the lift is actively negative. This is what separates a
        # family worth removing from one that is merely quiet.
        self.assertLess(loud["lift_log_loss"], -0.05)

    def test_silent_family_is_labelled_silent(self):
        silent = self._row("broadcast")
        self.assertEqual(silent["mean_abs_prob_points"], 0.0)
        self.assertEqual(silent["direction"], "silent")

    def test_results_are_sorted_by_skill_not_volume(self):
        lifts = self.signal["lift_log_loss"].tolist()
        self.assertEqual(lifts, sorted(lifts, reverse=True))

    def test_every_family_reports_its_feature_count(self):
        self.assertEqual(int(self._row("elo")["n_features"]), 2)
        self.assertEqual(int(self._row("broadcast")["n_features"]), 1)


class DeadFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = xco.dead_features(_inputs())

    def test_never_split_features_are_found_from_split_counts(self):
        self.assertIn("broadcast_channel1", self.result["never_split"])
        self.assertNotIn("home_elo", self.result["never_split"])

    def test_tiers_are_mutually_exclusive(self):
        tiers = ["never_split", "soft_dead", "rare_but_strong"]
        seen = [name for tier in tiers for name in self.result[tier]]
        self.assertEqual(len(seen), len(set(seen)))

    def test_per_feature_table_covers_every_predictor(self):
        self.assertEqual(self.result["n_features"], 4)
        self.assertEqual(len(self.result["per_feature"]), 4)
        self.assertIn("tier", self.result["per_feature"].columns)

    def test_by_family_rollup_counts_the_dead(self):
        by_family = self.result["by_family"].set_index("family")
        self.assertEqual(int(by_family.loc["broadcast", "never_split"]), 1)


class CoverageGapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = xco.coverage_gaps(_inputs())

    def test_residual_buckets_partition_the_cohort(self):
        buckets = self.result["residual_buckets"]
        self.assertEqual(int(buckets["games"].sum()), 1200)
        residuals = buckets["mean_residual"].tolist()
        self.assertEqual(residuals, sorted(residuals))

    def test_missingness_crosstab_only_covers_flags_that_vary(self):
        missing = self.result["missingness"]
        families = set(missing["family"]) if not missing.empty else set()
        self.assertIn("team_match_stats", families)
        # weather_missing is constant here, so there is nothing to compare.
        self.assertNotIn("weather", families)

    def test_side_balance_covers_present_sides(self):
        sides = set(self.result["side_balance"]["side"])
        self.assertEqual(sides, {"home", "away", "neutral"})


class MarketDisagreementTests(unittest.TestCase):
    def test_reports_unavailable_without_market_prices(self):
        result = xco.market_disagreement(_inputs())
        self.assertFalse(result["available"])

    def test_scores_the_model_against_the_market_where_they_diverge(self):
        base = _inputs()
        rng = np.random.default_rng(3)
        # A market that is right more often than the model.
        market = np.where(
            base.y > 0.5,
            rng.uniform(0.55, 0.9, len(base.y)),
            rng.uniform(0.1, 0.45, len(base.y)),
        )
        inputs = dataclasses.replace(base, market_prob=market)

        result = xco.market_disagreement(inputs)

        self.assertTrue(result["available"])
        self.assertGreater(result["disagreement_games"], 0)
        self.assertGreater(result["overall_market_accuracy"], result["overall_model_accuracy"])
        self.assertIn("edge_when_family_leads", result["families"].columns)

    def test_families_with_too_few_games_are_not_scored(self):
        base = _inputs()
        market = np.full(len(base.y), 0.5)
        result = xco.market_disagreement(dataclasses.replace(base, market_prob=market))
        silent = result["families"].set_index("family").loc["broadcast"]
        self.assertEqual(int(silent["games_leading"]), 0)
        self.assertTrue(np.isnan(silent["edge_when_family_leads"]))


class ConfidentlyWrongTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = xco.confidently_wrong(_inputs())

    def test_counts_confident_games_and_their_accuracy(self):
        self.assertGreater(self.result["confident_games"], 0)
        self.assertLessEqual(self.result["confident_wrong"], self.result["confident_games"])
        self.assertTrue(0.0 <= self.result["confident_accuracy"] <= 1.0)

    def test_worst_games_are_ordered_by_confidence_and_carry_drivers(self):
        worst = self.result["worst_games"]
        self.assertTrue(worst)
        confidences = [g["confidence"] for g in worst]
        self.assertEqual(confidences, sorted(confidences, reverse=True))
        self.assertEqual(len(worst[0]["top_drivers"]), 3)
        self.assertIn("family", worst[0]["top_drivers"][0])

    def test_loud_random_family_shows_up_when_confidently_wrong(self):
        # The planted noise family is what makes the model confidently wrong,
        # so it must rank above the family carrying real signal.
        families = self.result["families"].set_index("family")
        self.assertGreater(
            families.loc["team_match_stats", "standardized"],
            families.loc["elo", "standardized"],
        )


class FoldCollectorTests(unittest.TestCase):
    @unittest.skipUnless(
        __import__("importlib").util.find_spec("lightgbm"), "lightgbm not available"
    )
    def test_collector_captures_only_the_rows_its_fold_predicted(self):
        import lightgbm as lgb
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

        from pipeline.common.model_training.modelling_functions import (
            sanitize_feature_names,
        )

        rng = np.random.default_rng(0)
        frame = pd.DataFrame(
            {
                "round_name": rng.choice(["Round 1", "Finals Week 1"], 60),
                "home_elo": rng.normal(1500, 50, 60),
                "away_elo": rng.normal(1500, 50, 60),
            }
        )
        preprocessor = ColumnTransformer(
            transformers=[("encoder", OneHotEncoder(handle_unknown="ignore"), ["round_name"])],
            remainder="passthrough",
        )

        def to_df(X_array):
            cols = preprocessor.get_feature_names_out(preprocessor.feature_names_in_)
            df = pd.DataFrame(X_array, columns=sanitize_feature_names(cols))
            return df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        steps = Pipeline(
            [
                ("one_hot", preprocessor),
                ("to_df", FunctionTransformer(func=to_df, validate=False)),
            ]
        )
        y = (frame["home_elo"] > frame["away_elo"]).astype(int)
        transformed = steps.fit_transform(frame)
        model = lgb.LGBMClassifier(n_estimators=15, num_leaves=4, verbose=-1).fit(
            transformed, y
        )

        collector = xco.FoldCollector(steps, len(frame))
        mask = np.zeros(len(frame), dtype=bool)
        mask[10:25] = True
        collector(model, transformed.iloc[10:25], mask, 2024)

        self.assertEqual(collector.feature_names, ("round_name", "home_elo", "away_elo"))
        self.assertEqual(int(collector.captured.sum()), 15)
        self.assertTrue(collector.captured[10:25].all())
        # Untouched rows stay zero rather than picking up another fold's values.
        self.assertEqual(collector.values[0].sum(), 0.0)
        # Contributions still reconstruct the fold model's raw score.
        raw = model.booster_.predict(transformed.iloc[10:25], raw_score=True)
        np.testing.assert_allclose(
            collector.base_value[10:25] + collector.values[10:25].sum(axis=1),
            raw,
            atol=1e-9,
        )
        self.assertGreater(sum(collector.split_counts.values()), 0)

    def test_mismatched_mask_and_predictions_are_rejected(self):
        class _Stub:
            class booster_:
                @staticmethod
                def predict(X, pred_contrib=False):
                    return np.zeros((3, 4))

                @staticmethod
                def feature_importance(kind):
                    return np.zeros(3)

        collector = xco.FoldCollector.__new__(xco.FoldCollector)
        collector.feature_names = ("a", "b", "c")
        collector._starts = np.array([0, 1, 2])
        collector.values = np.zeros((5, 3))
        collector.base_value = np.zeros(5)
        collector.captured = np.zeros(5, dtype=bool)
        collector.split_counts = {}

        mask = np.array([True, True, False, False, False])
        with self.assertRaises(ValueError):
            collector(_Stub(), None, mask, 2024)


class RunAnalysesTests(unittest.TestCase):
    def test_all_returns_every_analysis_with_its_source(self):
        results = xco.run_analyses(_inputs(), "all")
        for name in ("families", "dead", "coverage"):
            self.assertIn(name, results)
        self.assertEqual(results["source"], xco.SOURCE_IN_SAMPLE)
        self.assertEqual(results["n_games"], 1200)

    def test_unknown_analysis_is_rejected(self):
        with self.assertRaises(ValueError):
            xco.run_analyses(_inputs(), "not-an-analysis")


if __name__ == "__main__":
    unittest.main()
