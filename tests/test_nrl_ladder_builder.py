import unittest

from pipeline.common.nrl_data.ladder import build_season_ladder


def _fixture(round_id, game_number, home, away, home_score, away_score,
             start_hour=19, round_name=None, state="Final"):
    # start_time is venue-local wall clock serialised as-if-UTC; only the hour
    # matters for day/night records
    base_epoch = 1735689600  # 2025-01-01T00:00Z
    return {
        "game_id": float(f"2025111{round_id:02d}{game_number}0"),
        "round_id": float(round_id),
        "round_name": round_name or f"Round {round_id}",
        "game_state_name": state,
        "start_time": float((base_epoch + round_id * 7 * 86400) // 86400 * 86400 + start_hour * 3600),
        "team_home": home,
        "team_away": away,
        "team_final_score_home": float(home_score),
        "team_final_score_away": float(away_score),
    }


class LadderBuilderTests(unittest.TestCase):
    def setUp(self):
        self.fixtures = [
            _fixture(1, 1, "A", "B", 20, 10, start_hour=15),  # day game
            _fixture(1, 2, "C", "D", 12, 12, start_hour=20),  # night draw
            _fixture(2, 1, "B", "C", 30, 6, start_hour=20),
            _fixture(2, 2, "A", "D", 14, 12, start_hour=20),  # close game
        ]
        self.byes = []

    def test_round_one_table(self):
        rows = build_season_ladder(self.fixtures, self.byes, 2025)
        round1 = {row["team"]: row for row in rows if row["round_id"] == 1}
        self.assertEqual(len(round1), 4)

        a = round1["A"]
        self.assertEqual(a["wins"], 1.0)
        self.assertEqual(a["competition_points"], 2.0)
        self.assertEqual(a["points_for"], 20.0)
        self.assertEqual(a["position"], 1.0)
        self.assertEqual(a["season_form"], "W")
        self.assertEqual(a["day_record"], "1-0-0")
        self.assertEqual(a["current_streak"], "1W")

        c = round1["C"]
        self.assertEqual(c["draws"], 1.0)
        self.assertEqual(c["competition_points"], 1.0)
        self.assertEqual(c["night_record"], "0-1-0")
        self.assertIsNone(c["current_streak"])

    def test_cumulative_and_form_order(self):
        rows = build_season_ladder(self.fixtures, self.byes, 2025)
        round2 = {row["team"]: row for row in rows if row["round_id"] == 2}

        a = round2["A"]
        self.assertEqual(a["wins"], 2.0)
        self.assertEqual(a["close_games"], 1.0)  # 14-12
        self.assertEqual(a["current_streak"], "2W")
        # most-recent-first form string
        self.assertEqual(a["season_form"], "WW")

        b = round2["B"]
        self.assertEqual(b["season_form"], "WL")  # round 2 win first, round 1 loss second
        self.assertEqual(b["current_streak"], "1W")

    def test_bye_credits_points_and_form(self):
        byes = [{"competition_year": 2025, "round_id": 2, "team": "E"}]
        fixtures = self.fixtures + [_fixture(1, 3, "E", "F", 8, 30)]
        rows = build_season_ladder(fixtures, byes, 2025)
        e_round2 = next(
            row for row in rows if row["team"] == "E" and row["round_id"] == 2
        )
        self.assertEqual(e_round2["byes"], 1.0)
        self.assertEqual(e_round2["competition_points"], 2.0)  # bye = 2 points
        self.assertEqual(e_round2["season_form"], "BL")
        self.assertEqual(e_round2["current_streak"], "1L")  # byes skipped in streak

    def test_finals_rounds_carry_frozen_table(self):
        fixtures = self.fixtures + [
            _fixture(3, 1, "A", "B", 10, 20, round_name="Finals Week 1"),
        ]
        rows = build_season_ladder(fixtures, self.byes, 2025)
        finals = {row["team"]: row for row in rows if row["round_id"] == 3}
        regular = {row["team"]: row for row in rows if row["round_id"] == 2}
        self.assertEqual(len(finals), 4)
        # A lost the final but the ladder freezes at the regular-season table
        self.assertEqual(finals["A"]["wins"], regular["A"]["wins"])
        self.assertEqual(finals["A"]["season_form"], regular["A"]["season_form"])

    def test_contiguous_rows_for_every_round(self):
        fixtures = self.fixtures + [
            _fixture(4, 1, "A", "B", 10, 8, round_name="Round 4"),
        ]
        rows = build_season_ladder(fixtures, self.byes, 2025)
        # round 3 has no games but every team still gets a carried-forward row
        round3 = [row for row in rows if row["round_id"] == 3]
        self.assertEqual(len(round3), 4)

    def test_scoring_aggregates_from_player_stats(self):
        scoring = {
            20251110110: {
                "home": {"tries": 4, "goals": 2, "field_goals": 0, "players": {1, 2, 3}},
                "away": {"tries": 2, "goals": 1, "field_goals": 0, "players": {9, 10}},
            }
        }
        rows = build_season_ladder(self.fixtures, self.byes, 2025, scoring)
        a1 = next(row for row in rows if row["team"] == "A" and row["round_id"] == 1)
        self.assertEqual(a1["tries_for"], 4.0)
        self.assertEqual(a1["tries_conceded"], 2.0)
        self.assertEqual(a1["goals_for"], 2.0)
        self.assertEqual(a1["players_used"], 3.0)


if __name__ == "__main__":
    unittest.main()
