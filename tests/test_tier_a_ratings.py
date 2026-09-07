import unittest

import numpy as np
import pandas as pd

from pipeline.common.model_training import tier_a_baseline as ta


def _season(rows):
    """Minimal fixture frame in the shape the baseline requires."""
    return pd.DataFrame(
        [
            {
                "game_id": index + 1,
                "competition_year": 2025,
                "round_id": row[0],
                "start_time": index,
                "game_number": 1,
                "team_home": row[1],
                "team_away": row[2],
                "game_state_name": "Final",
                "team_final_score_home": row[3],
                "team_final_score_away": row[4],
            }
            for index, row in enumerate(rows)
        ]
    )


class RatingsTests(unittest.TestCase):
    def setUp(self):
        # Strong beats Weak repeatedly, and every host scores six more than they
        # manage away, so the base rates carry a real home advantage.
        self.frame = _season(
            [(n, "Strong", "Weak", 32, 12) for n in range(1, 11)]
            + [(n, "Weak", "Strong", 16, 26) for n in range(11, 21)]
        )

    def test_ratings_agree_with_the_feature_walk(self):
        """Both entry points must share one accumulation, not two."""
        features = ta.compute_tier_a_baseline_features(self.frame)
        ratings = ta.compute_tier_a_ratings(self.frame)
        self.assertEqual(len(features), len(self.frame))
        self.assertEqual(set(ratings.attack), {"Strong", "Weak"})
        self.assertGreater(ratings.attack["Strong"], ratings.attack["Weak"])

    def test_empty_frame_returns_neutral_ratings(self):
        ratings = ta.compute_tier_a_ratings(self.frame.iloc[0:0])
        self.assertEqual(ratings.attack, {})
        self.assertFalse(ratings.calibrated)
        self.assertFalse(ratings.knows("Strong"))

    def test_missing_columns_still_raise(self):
        with self.assertRaises(ValueError):
            ta.compute_tier_a_ratings(self.frame.drop(columns=["team_away"]))

    def test_home_advantage_beats_neutral_for_the_same_matchup(self):
        ratings = ta.compute_tier_a_ratings(self.frame)
        home = ratings.home_win_probability("Strong", "Weak")
        neutral = ratings.home_win_probability("Strong", "Weak", neutral=True)
        self.assertGreater(home, neutral)

    def test_a_neutral_matchup_is_exactly_symmetric(self):
        """The two sides of a grand final have to sum to one."""
        ratings = ta.compute_tier_a_ratings(self.frame)
        forward = ratings.home_win_probability("Strong", "Weak", neutral=True)
        reverse = ratings.home_win_probability("Weak", "Strong", neutral=True)
        self.assertAlmostEqual(forward + reverse, 1.0, places=12)


class CalibrationTests(unittest.TestCase):
    def test_fit_recovers_a_known_distortion(self):
        rng = np.random.default_rng(11)
        truth = rng.uniform(0.05, 0.95, size=6000)
        outcomes = rng.random(6000) < truth
        # Overconfident inputs: push the true probability away from a half.
        raw = ta._sigmoid(3.0 * ta._logit(truth))
        calibration = ta.fit_logit_calibration(raw, outcomes)
        self.assertIsNotNone(calibration)
        recovered = np.array(
            [ta.apply_logit_calibration(value, calibration) for value in raw]
        )
        self.assertLess(
            np.mean((recovered - outcomes) ** 2), np.mean((raw - outcomes) ** 2)
        )
        # The slope should undo roughly the distortion that was applied.
        self.assertAlmostEqual(calibration[1], 1 / 3, delta=0.08)

    def test_calibration_preserves_ordering(self):
        calibration = (0.15, 0.17)
        values = [ta.apply_logit_calibration(p, calibration) for p in (0.1, 0.4, 0.6, 0.9)]
        self.assertEqual(values, sorted(values))

    def test_too_little_history_declines_to_fit(self):
        self.assertIsNone(ta.fit_logit_calibration([0.6, 0.4], [1, 0]))

    def test_single_class_history_declines_to_fit(self):
        rng = np.random.default_rng(3)
        self.assertIsNone(
            ta.fit_logit_calibration(rng.uniform(0.2, 0.8, 500), np.ones(500))
        )

    def test_no_calibration_is_a_pass_through(self):
        self.assertEqual(ta.apply_logit_calibration(0.73, None), 0.73)

    def test_dropping_the_intercept_makes_the_transform_antisymmetric(self):
        calibration = (0.5, 0.3)
        forward = ta.apply_logit_calibration(0.7, calibration, include_intercept=False)
        reverse = ta.apply_logit_calibration(0.3, calibration, include_intercept=False)
        self.assertAlmostEqual(forward + reverse, 1.0, places=12)
        # With the intercept it is deliberately not symmetric: that is the home bias.
        biased = ta.apply_logit_calibration(0.7, calibration)
        self.assertGreater(biased, forward)


if __name__ == "__main__":
    unittest.main()
