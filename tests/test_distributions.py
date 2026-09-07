import os
import sqlite3
import tempfile
import unittest

import numpy as np
import pandas as pd

from pipeline.common.model_prediciton import distributions as pdist
from pipeline.common.model_prediciton import prediction_functions as pf


class SummariseTests(unittest.TestCase):
    def test_bands_and_draw_partition_the_cloud(self):
        rng = np.random.default_rng(5)
        home = rng.poisson(24, 40000)
        away = rng.poisson(18, 40000)
        summary = pf.summarise_score_distribution(home, away)
        total = summary["p_draw_full_time"]
        for side in ("home", "away"):
            for band in ("1_6", "7_12", "13_plus"):
                total += summary[f"p_{side}_by_{band}"]
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_one_score_game_matches_the_six_point_bands(self):
        rng = np.random.default_rng(7)
        home, away = rng.poisson(22, 20000), rng.poisson(21, 20000)
        summary = pf.summarise_score_distribution(home, away)
        self.assertAlmostEqual(
            summary["p_one_score_game"],
            summary["p_draw_full_time"]
            + summary["p_home_by_1_6"]
            + summary["p_away_by_1_6"],
            places=9,
        )

    def test_the_stronger_side_wins_the_bigger_bands(self):
        rng = np.random.default_rng(9)
        home, away = rng.poisson(30, 20000), rng.poisson(14, 20000)
        summary = pf.summarise_score_distribution(home, away)
        self.assertGreater(summary["p_home_by_13_plus"], summary["p_away_by_13_plus"])

    def test_line_cover_excludes_pushes_from_both_sides(self):
        # Home wins by exactly 6 every time, against a posted line of -6.
        home = np.full(1000, 26)
        away = np.full(1000, 20)
        summary = pf.summarise_score_distribution(home, away, line_home=-6.0)
        self.assertEqual(summary["p_line_push"], 1.0)
        self.assertIsNone(summary["p_home_covers_line"])

    def test_line_cover_reads_the_handicap_from_the_home_side(self):
        rng = np.random.default_rng(3)
        home, away = rng.poisson(24, 20000), rng.poisson(18, 20000)
        generous = pf.summarise_score_distribution(home, away, line_home=6.5)
        harsh = pf.summarise_score_distribution(home, away, line_home=-12.5)
        # Being given 6.5 points is easier to cover than giving away 12.5.
        self.assertGreater(generous["p_home_covers_line"], harsh["p_home_covers_line"])

    def test_totals_over_probability(self):
        rng = np.random.default_rng(4)
        home, away = rng.poisson(22, 20000), rng.poisson(20, 20000)
        summary = pf.summarise_score_distribution(home, away, total_line=20.5)
        self.assertGreater(summary["p_total_over"], 0.95)
        summary = pf.summarise_score_distribution(home, away, total_line=80.5)
        self.assertLess(summary["p_total_over"], 0.05)

    def test_absent_markets_leave_the_keys_empty(self):
        summary = pf.summarise_score_distribution([20, 30], [10, 12])
        self.assertIsNone(summary["p_home_covers_line"])
        self.assertIsNone(summary["p_total_over"])
        self.assertNotIn("p_line_push", summary)

    def test_empty_cloud(self):
        self.assertEqual(pf.summarise_score_distribution([], []), {})


class WrapperTests(unittest.TestCase):
    def _frame(self, **extra):
        base = {
            "game_id": [1, 2],
            "home_goals_avg": [24.0, 18.0],
            "away_goals_avg": [18.0, 22.0],
        }
        base.update(extra)
        return pd.DataFrame(base)

    def test_distributions_are_opt_in_and_come_last(self):
        frame = self._frame()
        pair = pf.predict_match_outcome_and_scoreline_with_bayes(
            inference_data=frame,
            mu_home=frame["home_goals_avg"],
            mu_away=frame["away_goals_avg"],
            n_simulations=2000,
        )
        self.assertEqual(len(pair), 2)
        quad = pf.predict_match_outcome_and_scoreline_with_bayes(
            inference_data=frame,
            mu_home=frame["home_goals_avg"],
            mu_away=frame["away_goals_avg"],
            n_simulations=2000,
            return_diagnostics=True,
            return_distributions=True,
        )
        self.assertEqual(len(quad), 4)
        self.assertEqual(list(quad[3]["game_id"]), [1, 2])

    def test_asking_for_distributions_does_not_move_a_tip(self):
        """They ride out of the same simulation, so the outputs must be identical."""
        frame = self._frame()
        kwargs = dict(
            inference_data=frame,
            mu_home=frame["home_goals_avg"],
            mu_away=frame["away_goals_avg"],
            n_simulations=5000,
        )
        outcomes, margins = pf.predict_match_outcome_and_scoreline_with_bayes(**kwargs)
        with_dist = pf.predict_match_outcome_and_scoreline_with_bayes(
            **kwargs, return_distributions=True
        )
        pd.testing.assert_frame_equal(outcomes, with_dist[0])
        pd.testing.assert_frame_equal(margins, with_dist[1])

    def test_market_columns_are_picked_up_when_present(self):
        frame = self._frame(
            team_line_amount_home=[-6.5, 4.5], total_line=[42.5, 38.5]
        )
        *_, dist = pf.predict_match_outcome_and_scoreline_with_bayes(
            inference_data=frame,
            mu_home=frame["home_goals_avg"],
            mu_away=frame["away_goals_avg"],
            n_simulations=4000,
            return_distributions=True,
        )
        self.assertTrue(dist["p_home_covers_line"].notna().all())
        self.assertTrue(dist["p_total_over"].notna().all())

    def test_unusable_market_values_are_treated_as_absent(self):
        frame = self._frame(
            team_line_amount_home=[None, float("nan")], total_line=["", None]
        )
        *_, dist = pf.predict_match_outcome_and_scoreline_with_bayes(
            inference_data=frame,
            mu_home=frame["home_goals_avg"],
            mu_away=frame["away_goals_avg"],
            n_simulations=2000,
            return_distributions=True,
        )
        self.assertTrue(dist["p_home_covers_line"].isna().all())

    def test_empty_inference_data_returns_the_requested_arity(self):
        result = pf.predict_match_outcome_and_scoreline_with_bayes(
            inference_data=pd.DataFrame(),
            return_diagnostics=True,
            return_distributions=True,
        )
        self.assertEqual(len(result), 4)
        self.assertTrue(result[3].empty)


class StoreTests(unittest.TestCase):
    def setUp(self):
        handle, self.db_path = tempfile.mkstemp(suffix=".sqlite")
        os.close(handle)

    def tearDown(self):
        os.remove(self.db_path)

    def _frame(self, p_one_score=0.4):
        return pd.DataFrame(
            [
                {
                    "game_id": 1,
                    "p_one_score_game": p_one_score,
                    "p_home_covers_line": 0.55,
                    "p_total_over": None,
                }
            ]
        )

    def test_round_trip(self):
        self.assertEqual(pdist.save_distributions(self._frame(), self.db_path), 1)
        loaded = pdist.load_distributions(self.db_path)
        self.assertEqual(len(loaded), 1)
        self.assertAlmostEqual(float(loaded.iloc[0]["p_one_score_game"]), 0.4)
        self.assertAlmostEqual(float(loaded.iloc[0]["p_home_covers_line"]), 0.55)
        self.assertIsNone(loaded.iloc[0]["p_total_over"])

    def test_rewriting_a_game_replaces_it(self):
        pdist.save_distributions(self._frame(0.4), self.db_path)
        pdist.save_distributions(self._frame(0.9), self.db_path)
        loaded = pdist.load_distributions(self.db_path)
        self.assertEqual(len(loaded), 1)
        self.assertAlmostEqual(float(loaded.iloc[0]["p_one_score_game"]), 0.9)

    def test_filtering_by_game_id(self):
        pdist.save_distributions(self._frame(), self.db_path)
        self.assertEqual(len(pdist.load_distributions(self.db_path, [1])), 1)
        self.assertTrue(pdist.load_distributions(self.db_path, [2]).empty)
        self.assertTrue(pdist.load_distributions(self.db_path, []).empty)

    def test_a_missing_table_is_not_an_error(self):
        self.assertTrue(pdist.load_distributions(self.db_path).empty)

    def test_nothing_to_save_is_a_no_op(self):
        self.assertEqual(pdist.save_distributions(pd.DataFrame(), self.db_path), 0)
        self.assertEqual(pdist.save_distributions(None, self.db_path), 0)
        self.assertEqual(
            pdist.save_distributions(pd.DataFrame([{"x": 1}]), self.db_path), 0
        )

    def test_a_new_column_migrates_onto_an_existing_table(self):
        pdist.save_distributions(self._frame(), self.db_path)
        con = sqlite3.connect(self.db_path)
        con.execute(f"ALTER TABLE {pdist.TABLE_NAME} DROP COLUMN p_total_push")
        con.commit()
        con.close()
        self.assertEqual(pdist.save_distributions(self._frame(), self.db_path), 1)
        self.assertIn("p_total_push", pdist.load_distributions(self.db_path).columns)


if __name__ == "__main__":
    unittest.main()
