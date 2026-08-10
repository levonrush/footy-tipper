import sqlite3
import pathlib
import unittest


SQL_PATH = pathlib.Path(__file__).resolve().parents[1] / "pipeline" / "common" / "sql" / "prediction_table.sql"


def _create_schema(con):
    con.executescript(
        """
        CREATE TABLE footy_tipping_data (
            game_id INTEGER PRIMARY KEY,
            game_state_name TEXT,
            competition_year INTEGER,
            round_id INTEGER,
            round_name TEXT,
            team_home TEXT,
            position_home_ladder INTEGER,
            team_head_to_head_odds_home REAL,
            team_away TEXT,
            position_away_ladder INTEGER,
            team_head_to_head_odds_away REAL,
            start_time REAL,
            game_number INTEGER
        );

        CREATE TABLE predictions_table (
            game_id INTEGER PRIMARY KEY,
            home_team_result TEXT,
            home_team_win_prob REAL,
            home_team_lose_prob REAL,
            draw_prob REAL,
            bayes_factor REAL,
            evidence_strength TEXT,
            predicted_home_score INTEGER,
            predicted_away_score INTEGER,
            predicted_margin INTEGER
        );
        """
    )


def _read_prediction_query():
    return SQL_PATH.read_text(encoding="utf-8")


class PredictionQueryContractTests(unittest.TestCase):
    def test_query_targets_latest_pregame_year_and_min_round(self):
        con = sqlite3.connect(":memory:")
        con.row_factory = sqlite3.Row
        _create_schema(con)

        footy_rows = [
            (101, "Pre Game", 2024, 22, "Round 22", "A", 1, 1.5, "B", 2, 2.5),
            (201, "Pre Game", 2025, 2, "Round 2", "C", 3, 1.8, "D", 4, 2.1),
            (202, "Pre Game", 2025, 1, "Round 1", "E", 5, 1.7, "F", 6, 2.2),
            (203, "Pre Game", 2025, 1, "Round 1", "G", 7, 1.9, "H", 8, 2.0),
            (301, "Final", 2026, 1, "Round 1", "I", 9, 1.6, "J", 10, 2.3)
        ]
        prediction_rows = [
            (101, "Win", 0.60, 0.40),
            (201, "Win", 0.55, 0.45),
            (202, "Loss", 0.49, 0.51),
            (203, "Win", 0.58, 0.42),
            (301, "Win", 0.52, 0.48)
        ]

        con.executemany(
            """
            INSERT INTO footy_tipping_data (
                game_id, game_state_name, competition_year, round_id, round_name,
                team_home, position_home_ladder, team_head_to_head_odds_home,
                team_away, position_away_ladder, team_head_to_head_odds_away
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            footy_rows,
        )
        con.executemany(
            """
            INSERT INTO predictions_table (
                game_id, home_team_result, home_team_win_prob, home_team_lose_prob
            ) VALUES (?, ?, ?, ?)
            """,
            prediction_rows,
        )

        rows = con.execute(_read_prediction_query()).fetchall()
        con.close()

        self.assertEqual({row["game_id"] for row in rows}, {202, 203})
        self.assertTrue(all(row["competition_year"] == 2025 for row in rows))
        self.assertTrue(all(row["round_id"] == 1 for row in rows))

    def test_query_returns_no_rows_without_pregame_data(self):
        con = sqlite3.connect(":memory:")
        con.row_factory = sqlite3.Row
        _create_schema(con)

        con.execute(
            """
            INSERT INTO footy_tipping_data (
                game_id, game_state_name, competition_year, round_id, round_name,
                team_home, position_home_ladder, team_head_to_head_odds_home,
                team_away, position_away_ladder, team_head_to_head_odds_away
            ) VALUES (1, 'Final', 2025, 1, 'Round 1', 'A', 1, 1.6, 'B', 2, 2.2)
            """
        )
        con.execute(
            """
            INSERT INTO predictions_table (
                game_id, home_team_result, home_team_win_prob, home_team_lose_prob
            ) VALUES (1, 'Win', 0.5, 0.5)
            """
        )

        rows = con.execute(_read_prediction_query()).fetchall()
        con.close()

        self.assertEqual(rows, [])

    def test_published_view_stays_the_predictions_contract(self):
        """The view must not grow to depend on the explanations table.

        Explanations are diagnostics: they are left-joined in pandas by
        distribution.get_predictions so a missing or broken explanations table
        costs the email a sentence rather than breaking the send. Pulling them
        into this query would make the published view able to fail.
        """
        con = sqlite3.connect(":memory:")
        _create_schema(con)
        con.execute(
            """
            INSERT INTO footy_tipping_data (
                game_id, game_state_name, competition_year, round_id, round_name,
                team_home, position_home_ladder, team_head_to_head_odds_home,
                team_away, position_away_ladder, team_head_to_head_odds_away,
                start_time, game_number
            ) VALUES (1, 'Pre Game', 2026, 5, 'Round 5', 'A', 1, 1.6, 'B', 2, 2.2, 0, 1)
            """
        )
        con.execute(
            """
            INSERT INTO predictions_table (
                game_id, home_team_result, home_team_win_prob, home_team_lose_prob
            ) VALUES (1, 'Win', 0.6, 0.4)
            """
        )

        cursor = con.execute(_read_prediction_query())
        columns = {description[0] for description in cursor.description}
        cursor.fetchall()
        con.close()

        query = _read_prediction_query().lower()
        self.assertNotIn("prediction_explanations", query)
        self.assertNotIn("why_line", columns)
        self.assertEqual(
            columns,
            {
                "game_id",
                "home_team_result",
                "team_home",
                "position_home",
                "team_head_to_head_odds_home",
                "team_away",
                "position_away",
                "team_head_to_head_odds_away",
                "home_team_win_prob",
                "home_team_lose_prob",
                "draw_prob",
                "predicted_home_score",
                "predicted_away_score",
                "predicted_margin",
                "bayes_factor",
                "evidence_strength",
                "round_id",
                "competition_year",
                "round_name",
            },
        )


if __name__ == "__main__":
    unittest.main()
