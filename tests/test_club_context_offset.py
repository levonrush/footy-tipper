import unittest

import numpy as np
import pandas as pd

from pipeline.common.club_context import materiality as mat


def _frame(*, games=240, effect=-1.1, seed=7):
    """A paired frame where the affected side really does underperform.

    Twelve event clusters of five games each, on top of ordinary fixtures whose
    outcomes match their stated probabilities.
    """

    rng = np.random.default_rng(seed)
    baseline = rng.uniform(0.25, 0.75, size=games)
    side = np.zeros(games)
    clusters = np.array([""] * games, dtype=object)
    for index in range(12):
        start = index * 5
        side[start : start + 5] = 1.0 if index % 2 == 0 else -1.0
        clusters[start : start + 5] = f"event-{index}"
    logit = np.log(baseline / (1 - baseline))
    truth = 1 / (1 + np.exp(-(logit + effect * side)))
    outcome = (rng.uniform(size=games) < truth).astype(float)
    return pd.DataFrame(
        {
            "game_id": np.arange(games) + 1,
            "competition_year": 2019 + (np.arange(games) // 40),
            "round_id": (np.arange(games) % 24) + 1,
            "team_home": [f"Home {i % 16}" for i in range(games)],
            "team_away": [f"Away {i % 16}" for i in range(games)],
            "baseline_home_win_prob": baseline,
            "actual_home_win": outcome,
            "club_context_affected_side": side,
            "club_context_affected_exposure": side * 0.6,
            "event_key": clusters,
        }
    )


def _season_frame(*, seasons=6, teams=16, rounds=22, seed=19):
    """A plausible league: every club plays most rounds, several seasons deep."""

    rng = np.random.default_rng(seed)
    rows = []
    game_id = 0
    for season in range(seasons):
        for round_id in range(1, rounds + 1):
            order = rng.permutation(teams)
            for pair in range(teams // 2):
                home = int(order[2 * pair])
                away = int(order[2 * pair + 1])
                game_id += 1
                probability = float(np.clip(rng.normal(0.5, 0.15), 0.1, 0.9))
                rows.append(
                    {
                        "game_id": game_id,
                        "competition_year": 2019 + season,
                        "round_id": round_id,
                        "team_home": f"Club {home}",
                        "team_away": f"Club {away}",
                        "baseline_home_win_prob": probability,
                        "actual_home_win": float(rng.uniform() < probability),
                    }
                )
    return pd.DataFrame(rows)


class OffsetContractTests(unittest.TestCase):
    def setUp(self):
        self.report = mat.evaluate_context_offset(
            _frame(), bootstrap_reps=200, seed=11
        )

    def test_it_scores(self):
        self.assertEqual(self.report["status"], "ok")

    def test_no_event_means_no_change(self):
        # The whole point of the cohort-restricted design: a game with no event
        # must come out byte-identical to the shipped prediction, so a reported
        # flip cannot be refit noise.
        for result in self.report["specifications"].values():
            self.assertEqual(result["unexposed_rows_changed"], 0)
            self.assertEqual(result["tip_flips_outside_event_cohort"], 0)

    def test_the_offset_recovers_the_planted_direction(self):
        primary = self.report["specifications"]["side"]
        self.assertLess(primary["fit"]["full_sample_coefficients"][0], 0.0)
        self.assertLess(primary["cohorts"]["exposed"]["delta"]["log_loss"], 0.0)

    def test_estimation_is_leave_one_cluster_out(self):
        self.assertIn("cluster", self.report["estimation"]["method"])
        self.assertEqual(self.report["specifications"]["side"]["fit"]["clusters"], 12)

    def test_calibration_detects_the_planted_bias(self):
        calibration = self.report["affected_side_calibration"]
        self.assertTrue(calibration["available"])
        self.assertLess(calibration["mean"], 0.0)

    def test_prevalence_and_season_impact_are_reported(self):
        self.assertEqual(self.report["prevalence"]["exposed_games"], 60)
        season = self.report["specifications"]["side"]["season_impact"]
        self.assertIn("extra_correct_tips_per_season", season)

    def test_season_impact_signs_a_worse_cohort_as_lost_tips(self):
        # Accuracy delta is candidate minus baseline, so better must read
        # positive and worse must read negative. Getting this backwards would
        # turn a harmful offset into a headline gain.
        prevalence = {"exposed_games_per_season": 10.0, "games_per_season": 200.0}
        worse = mat._season_impact(prevalence, {"accuracy": -0.05, "log_loss": 0.01})
        better = mat._season_impact(prevalence, {"accuracy": 0.05, "log_loss": -0.01})
        self.assertAlmostEqual(worse["extra_correct_tips_per_season"], -0.5)
        self.assertAlmostEqual(better["extra_correct_tips_per_season"], 0.5)

    def test_a_null_cohort_produces_no_material_gain(self):
        report = mat.evaluate_context_offset(
            _frame(effect=0.0, seed=3), bootstrap_reps=200, seed=11
        )
        interval = report["specifications"]["side"]["paired_log_loss_delta"]
        self.assertLessEqual(interval["ci95_low"], 0.0)
        self.assertGreaterEqual(interval["ci95_high"], 0.0)

    def test_a_frame_without_orientation_is_not_ready(self):
        frame = _frame().drop(
            columns=["club_context_affected_side", "club_context_affected_exposure"]
        )
        self.assertEqual(
            mat.evaluate_context_offset(frame, bootstrap_reps=10)["status"], "not_ready"
        )

    def test_shadow_only_is_asserted(self):
        self.assertTrue(self.report["shadow_only"])


class FitTests(unittest.TestCase):
    def test_a_separated_cohort_does_not_diverge(self):
        # Every affected game lost, so the unpenalised optimum is minus
        # infinity; the ridge and trust region must still return a finite fit.
        design = np.vstack([np.ones((20, 1)), np.zeros((80, 1))])
        outcome = np.concatenate([np.zeros(20), np.ones(80)])
        offset = np.zeros(100)
        beta = mat.fit_offset_coefficients(design, outcome, offset, ridge=1.0)
        self.assertTrue(np.all(np.isfinite(beta)))
        self.assertLess(beta[0], 0.0)

    def test_no_intercept_is_fitted(self):
        # A design of all zeros can carry no information, so the coefficient
        # must stay at zero rather than absorbing the base rate.
        design = np.zeros((50, 1))
        outcome = np.ones(50)
        beta = mat.fit_offset_coefficients(design, outcome, np.zeros(50), ridge=1.0)
        self.assertAlmostEqual(float(beta[0]), 0.0, places=9)


class CliFrameTests(unittest.TestCase):
    def test_an_empty_paired_file_is_not_ready_rather_than_a_win(self):
        from pipeline import club_context_evaluate as cce

        report = cce.evaluate_offset_frame(pd.DataFrame({"game_id": []}))
        self.assertEqual(report["status"], "not_ready")
        self.assertIn("not_ready", cce._offset_markdown(report))

    def test_the_offset_does_not_require_a_candidate_column(self):
        # It holds the baseline fixed, so a paired frame from the refit
        # ablation is not a precondition for running it.
        from pipeline import club_context_evaluate as cce

        frame = _frame(games=120).drop(columns=["club_context_affected_exposure"])
        report = cce.evaluate_offset_frame(frame, bootstrap_reps=50, seed=3)
        self.assertEqual(report["status"], "ok")
        self.assertIn("side", report["specifications"])
        self.assertNotIn("exposure", report["specifications"])


class ShortfallTests(unittest.TestCase):
    def test_shortfall_uses_only_earlier_games(self):
        frame = _frame(games=120)
        stacked = mat.season_shortfall(frame)
        first = stacked[stacked["prior_games"] == 0]
        self.assertTrue(first["shortfall_rate"].isna().all())

    def test_comparator_covers_far_more_games_than_the_event_cohort(self):
        result = mat.evaluate_form_shortfall(
            _season_frame(), bootstrap_reps=50, seed=5
        )
        self.assertTrue(result["available"], result.get("reason"))
        # The whole argument for the comparator is reach: it must be defined on
        # far more than the 2.8% of fixtures Club Context touches.
        self.assertGreater(result["prevalence"]["coverage"], 0.4)
        self.assertTrue(result["shadow_only"])

    def test_missing_columns_fail_soft(self):
        result = mat.evaluate_form_shortfall(pd.DataFrame({"game_id": [1, 2]}))
        self.assertFalse(result["available"])


if __name__ == "__main__":
    unittest.main()
