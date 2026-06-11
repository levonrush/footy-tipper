import os
import sqlite3
import tempfile
import unittest

from pipeline.common.use_predictions.scoreboard import (
    get_season_results,
    get_season_scoreboard,
    scoreboard_summary_line,
)


def _build_db(db_path, rows, predictions):
    con = sqlite3.connect(db_path)
    con.executescript(
        """
        CREATE TABLE footy_tipping_data (
            game_id INTEGER PRIMARY KEY,
            game_state_name TEXT,
            competition_year INTEGER,
            round_id INTEGER,
            round_name TEXT,
            team_home TEXT,
            team_away TEXT,
            team_final_score_home REAL,
            team_final_score_away REAL,
            team_head_to_head_odds_home REAL,
            team_head_to_head_odds_away REAL
        );
        CREATE TABLE predictions_table (
            game_id INTEGER PRIMARY KEY,
            home_team_result TEXT,
            home_team_win_prob REAL
        );
        """
    )
    con.executemany(
        "INSERT INTO footy_tipping_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
    )
    con.executemany("INSERT INTO predictions_table VALUES (?, ?, ?)", predictions)
    con.commit()
    con.close()


class ScoreboardTests(unittest.TestCase):
    def setUp(self):
        handle, self.db_path = tempfile.mkstemp(suffix=".sqlite")
        os.close(handle)

    def tearDown(self):
        os.remove(self.db_path)

    def test_scoreboard_math(self):
        rows = [
            # Round 1: model tips home (Win) and home wins -> hit; market favours home -> hit.
            (1, "Final", 2026, 1, "Round 1", "A", "B", 24, 12, 1.50, 2.60),
            # Round 1: model tips away (Loss) but home wins -> miss; market favours home -> hit.
            (2, "Final", 2026, 1, "Round 1", "C", "D", 20, 18, 1.70, 2.20),
            # Round 2: model tips home, home wins -> hit; market favours away -> miss.
            (3, "Final", 2026, 2, "Round 2", "E", "F", 30, 10, 2.40, 1.55),
            # Round 2: a draw is excluded entirely.
            (4, "Final", 2026, 2, "Round 2", "G", "H", 14, 14, 1.90, 1.90),
            # Pre-game rows never count.
            (5, "Pre Game", 2026, 3, "Round 3", "I", "J", None, None, 1.80, 2.00),
        ]
        predictions = [
            (1, "Win", 0.70),
            (2, "Loss", 0.40),
            (3, "Win", 0.60),
            (4, "Win", 0.55),
            (5, "Win", 0.65),
        ]
        _build_db(self.db_path, rows, predictions)

        scoreboard = get_season_scoreboard(self.db_path)
        self.assertEqual(scoreboard["competition_year"], 2026)
        self.assertEqual(scoreboard["season_games"], 3)
        self.assertEqual(scoreboard["season_correct"], 2)
        self.assertEqual(scoreboard["market_games"], 3)
        self.assertEqual(scoreboard["market_correct"], 2)
        self.assertEqual(scoreboard["last_round_id"], 2)
        self.assertEqual(scoreboard["last_round_name"], "Round 2")
        self.assertEqual(scoreboard["last_round_games"], 1)
        self.assertEqual(scoreboard["last_round_correct"], 1)

        line = scoreboard_summary_line(scoreboard)
        self.assertIn("Round 2: 1/1", line)
        self.assertIn("Season: 2/3", line)

    def test_no_settled_games_returns_none(self):
        rows = [(1, "Pre Game", 2026, 1, "Round 1", "A", "B", None, None, 1.5, 2.6)]
        predictions = [(1, "Win", 0.7)]
        _build_db(self.db_path, rows, predictions)
        self.assertIsNone(get_season_scoreboard(self.db_path))
        self.assertTrue(get_season_results(self.db_path).empty)
        self.assertIsNone(scoreboard_summary_line(None))

    def test_season_results_rows(self):
        rows = [
            (1, "Final", 2026, 1, "Round 1", "A", "B", 24, 12, 1.50, 2.60),
            (2, "Final", 2026, 2, "Round 2", "C", "D", 10, 20, 1.70, 2.20),
        ]
        predictions = [(1, "Win", 0.70), (2, "Loss", 0.40)]
        _build_db(self.db_path, rows, predictions)

        results = get_season_results(self.db_path)
        self.assertEqual(len(results), 2)
        # Newest round first.
        self.assertEqual(int(results.iloc[0]["round_id"]), 2)
        self.assertEqual(results.iloc[0]["tipped_team"], "D")
        self.assertTrue(bool(results.iloc[0]["tip_correct"]))
        self.assertAlmostEqual(float(results.iloc[0]["tip_prob"]), 0.60, places=6)
        self.assertEqual(results.iloc[1]["tipped_team"], "A")
        self.assertTrue(bool(results.iloc[1]["tip_correct"]))


if __name__ == "__main__":
    unittest.main()
