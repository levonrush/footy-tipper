import unittest

from pipeline.common.nrl_data.performance import (
    build_game_performance,
    build_season_performance,
)


class GamePerformanceTests(unittest.TestCase):
    def test_stat_sources(self):
        row = build_game_performance(
            team_stats={"possession_pct": 52.0, "completion_rate": 81.0,
                        "effective_tackle_pct": 90.1},
            player_sums={"tackles_made": 310.0, "kick_metres": 420.5, "kicks": 18.0,
                         "goals": 4.0, "goal_attempts": 5.0, "tries": 4.0},
            fixture_values={"points_for": 24.0, "win": 1.0, "draw": 0.0, "loss": 0.0},
        )
        self.assertEqual(row["tackle_made"], 310.0)
        # feed quirk: bare kicks column = metres, kicks_occur = count
        self.assertEqual(row["kicks"], 420.5)
        self.assertEqual(row["kicks_occur"], 18.0)
        self.assertEqual(row["possession"], 52.0)
        self.assertEqual(row["set_completion_rate"], 81.0)
        self.assertEqual(row["points"], 24.0)
        self.assertEqual(row["wins"], 1.0)
        self.assertEqual(row["goal_conversion_rate"], 80.0)
        self.assertEqual(row["try"], 4.0)

    def test_missing_sources_are_absent_not_zero(self):
        row = build_game_performance({}, {}, {"points_for": 10.0, "win": 0.0,
                                              "draw": 0.0, "loss": 1.0})
        self.assertNotIn("tackle_made", row)
        self.assertNotIn("possession", row)
        self.assertEqual(row["points"], 10.0)
        self.assertEqual(row["losses"], 1.0)


class SeasonPerformanceTests(unittest.TestCase):
    def _fixtures(self):
        return [
            {
                "game_id": 20251110110.0,
                "round_id": 1.0,
                "round_name": "Round 1",
                "game_state_name": "Final",
                "team_home": "A",
                "team_away": "B",
                "team_final_score_home": 20.0,
                "team_final_score_away": 10.0,
            },
            {
                "game_id": 20251112810.0,
                "round_id": 28.0,
                "round_name": "Finals Week 1",
                "game_state_name": "Final",
                "team_home": "A",
                "team_away": "B",
                "team_final_score_home": 12.0,
                "team_final_score_away": 4.0,
            },
        ]

    def test_per_round_rows_and_bye_zeros(self):
        byes = [{"competition_year": 2025, "round_id": 1, "team": "C"}]
        team_stats = {20251110110: {"home": {"possession_pct": 55.0}, "away": {}}}
        player_sums = {20251110110: {"home": {"tackles_made": 280.0}, "away": {}}}

        rows = build_season_performance(
            self._fixtures(), byes, 2025, team_stats, player_sums
        )
        by_key = {(row["round_id"], row["team"]): row for row in rows}

        a1 = by_key[(1, "A")]
        self.assertEqual(a1["possession"], 55.0)
        self.assertEqual(a1["tackle_made"], 280.0)
        self.assertEqual(a1["wins"], 1.0)
        self.assertEqual(a1["points"], 20.0)

        # bye teams get all-zero rows (feed behaviour)
        c1 = by_key[(1, "C")]
        self.assertEqual(c1["tackle_made"], 0.0)
        self.assertEqual(c1["wins"], 0.0)

        # Finals are finalized matches too. Even with sparse match-centre
        # stats, score/result fields advance the latest-prior team record.
        a28 = by_key[(28, "A")]
        b28 = by_key[(28, "B")]
        self.assertEqual(a28["points"], 12.0)
        self.assertEqual(a28["wins"], 1.0)
        self.assertEqual(b28["points"], 4.0)
        self.assertEqual(b28["losses"], 1.0)


if __name__ == "__main__":
    unittest.main()
