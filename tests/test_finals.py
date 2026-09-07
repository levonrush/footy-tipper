import os
import sqlite3
import tempfile
import unittest
from unittest import mock

import pandas as pd

from pipeline.common import rounds
from pipeline.common.use_predictions import finals


def _build_db(db_path, rows):
    con = sqlite3.connect(db_path)
    con.executescript(
        """
        CREATE TABLE footy_tipping_data (
            game_id INTEGER PRIMARY KEY, game_state_name TEXT, competition_year INTEGER,
            round_id INTEGER, round_name TEXT, team_home TEXT, team_away TEXT,
            team_final_score_home REAL, team_final_score_away REAL,
            team_head_to_head_odds_home REAL, team_head_to_head_odds_away REAL
        );
        CREATE TABLE predictions_table (
            game_id INTEGER PRIMARY KEY, home_team_result TEXT,
            home_team_win_prob REAL, home_team_lose_prob REAL
        );
        """
    )
    con.executemany(
        "INSERT INTO footy_tipping_data VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        [row[0] for row in rows],
    )
    con.executemany(
        "INSERT INTO predictions_table VALUES (?,?,?,?)", [row[1] for row in rows]
    )
    con.commit()
    con.close()


def _game(game_id, round_id, round_name, home, away, hs, as_, odds_home, odds_away,
          tip="Win", win=0.70, lose=0.27):
    return (
        (game_id, "Final", 2026, round_id, round_name, home, away, hs, as_, odds_home, odds_away),
        (game_id, tip, win, lose),
    )


class ThemeTests(unittest.TestCase):
    def test_a_regular_round_keeps_the_original_palette(self):
        """The regular email must render exactly as it did before finals support."""
        theme = finals.theme(rounds.REGULAR)
        self.assertEqual(theme["accent"], "#0f766e")
        self.assertEqual(theme["value_accent"], "#16a34a")
        self.assertEqual(theme["feature_accent"], "#f59e0b")
        self.assertEqual(
            theme["header_gradient"],
            "linear-gradient(135deg, #115e59 0%, #0369a1 100%)",
        )
        self.assertIsNone(theme["ribbon_label"])

    def test_every_finals_stage_has_its_own_ribbon(self):
        labels = {finals.theme(stage)["ribbon_label"] for stage in rounds.FINALS_STAGES}
        self.assertEqual(len(labels), len(rounds.FINALS_STAGES))
        self.assertTrue(all("SPECIAL EDITION" in label for label in labels))

    def test_an_unknown_stage_falls_back_to_the_regular_palette(self):
        self.assertIsNone(finals.theme("something_else")["ribbon_label"])


class KnockoutStakesTests(unittest.TestCase):
    def test_week_one_distinguishes_qualifying_from_elimination(self):
        for seeds in ((1, 4), (2, 3)):
            with self.subTest(seeds=seeds):
                text = finals.knockout_stakes(rounds.FINALS_WEEK_1, *seeds)
                self.assertIn("Qualifying final", text)
                self.assertIn("week off", text)
        for seeds in ((5, 8), (6, 7)):
            with self.subTest(seeds=seeds):
                text = finals.knockout_stakes(rounds.FINALS_WEEK_1, *seeds)
                self.assertIn("Elimination final", text)
                self.assertIn("season ends", text)

    def test_seed_order_does_not_matter(self):
        self.assertEqual(
            finals.knockout_stakes(rounds.FINALS_WEEK_1, 4, 1),
            finals.knockout_stakes(rounds.FINALS_WEEK_1, 1, 4),
        )

    def test_later_rounds_do_not_need_seeds(self):
        self.assertIn("season ends", finals.knockout_stakes(rounds.FINALS_WEEK_2))
        self.assertIn("Grand Final", finals.knockout_stakes(rounds.PRELIMINARY))
        self.assertIn("premiership", finals.knockout_stakes(rounds.GRAND_FINAL))

    def test_missing_seeds_still_produce_a_finals_line(self):
        text = finals.knockout_stakes(rounds.FINALS_WEEK_1, None, None)
        self.assertIn("Finals football", text)

    def test_a_regular_round_has_no_stakes_line(self):
        self.assertIsNone(finals.knockout_stakes(rounds.REGULAR, 1, 2))


class LedgerTests(unittest.TestCase):
    def setUp(self):
        handle, self.db_path = tempfile.mkstemp(suffix=".sqlite")
        os.close(handle)

    def tearDown(self):
        os.remove(self.db_path)

    def test_regular_season_and_finals_are_counted_separately(self):
        _build_db(
            self.db_path,
            [
                _game(1, 1, "Round 1", "A", "B", 24, 12, 1.5, 2.6),
                _game(2, 2, "Round 2", "C", "D", 10, 20, 1.5, 2.6),
                _game(3, 28, "Finals Week 1", "E", "F", 30, 10, 1.5, 2.6),
            ],
        )
        ledger = finals.stage_ledger(self.db_path)
        self.assertEqual(ledger["regular_season"]["games"], 2)
        self.assertEqual(ledger["regular_season"]["correct"], 1)
        self.assertEqual(ledger["finals"]["games"], 1)
        self.assertEqual(ledger["finals"]["correct"], 1)

    def test_finals_is_none_before_any_finals_game(self):
        _build_db(self.db_path, [_game(1, 1, "Round 1", "A", "B", 24, 12, 1.5, 2.6)])
        self.assertIsNone(finals.stage_ledger(self.db_path)["finals"])

    def test_best_call_is_the_biggest_upset_that_landed(self):
        _build_db(
            self.db_path,
            [
                # Heavy outsider, tipped, and it won.
                _game(1, 1, "Round 1", "A", "B", 24, 12, 6.00, 1.15),
                # Short favourite, tipped, and it won: correct but not an upset.
                _game(2, 2, "Round 2", "C", "D", 30, 10, 1.10, 7.00),
            ],
        )
        best = finals.stage_ledger(self.db_path)["best_call"]
        self.assertEqual(best["tipped_team"], "A")
        self.assertEqual(best["round_name"], "Round 1")
        self.assertTrue(best["correct"])

    def test_worst_call_is_the_most_confident_miss(self):
        _build_db(
            self.db_path,
            [
                _game(1, 1, "Round 1", "A", "B", 10, 24, 1.5, 2.6, win=0.90, lose=0.08),
                _game(2, 2, "Round 2", "C", "D", 10, 24, 1.5, 2.6, win=0.55, lose=0.42),
            ],
        )
        worst = finals.stage_ledger(self.db_path)["worst_call"]
        self.assertEqual(worst["tipped_team"], "A")
        self.assertFalse(worst["correct"])

    def test_nothing_settled_returns_none(self):
        _build_db(self.db_path, [])
        self.assertIsNone(finals.stage_ledger(self.db_path))


class HeadToHeadTests(unittest.TestCase):
    def setUp(self):
        handle, self.db_path = tempfile.mkstemp(suffix=".sqlite")
        os.close(handle)
        _build_db(
            self.db_path,
            [
                _game(1, 5, "Round 5", "A", "B", 20, 10, 1.5, 2.6),
                _game(2, 15, "Round 15", "B", "A", 30, 12, 1.5, 2.6),
                _game(3, 28, "Finals Week 1", "A", "B", 18, 28, 1.5, 2.6),
            ],
        )

    def tearDown(self):
        os.remove(self.db_path)

    def test_counts_split_the_season_from_the_finals(self):
        history = finals.head_to_head(self.db_path, "A", "B", 2026)
        self.assertEqual(history["season_meetings"], 3)
        self.assertEqual(history["season_home_wins"], 1)
        self.assertEqual(history["season_away_wins"], 2)
        self.assertEqual(history["finals_meetings"], 1)
        self.assertEqual(history["finals_away_wins"], 1)

    def test_the_last_meeting_score_reads_from_the_winner(self):
        """`B won 18-28` is a home-away scoreline and reads as a typo."""
        last = finals.head_to_head(self.db_path, "A", "B", 2026)["last_meeting"]
        self.assertEqual(last["winner"], "B")
        self.assertEqual(last["score"], "28-18")

    def test_history_never_quotes_a_future_season(self):
        history = finals.head_to_head(self.db_path, "A", "B", 2025)
        self.assertIsNone(history)

    def test_teams_that_have_never_met(self):
        self.assertIsNone(finals.head_to_head(self.db_path, "A", "Z", 2026))

    def test_a_broken_database_fails_soft(self):
        self.assertIsNone(finals.head_to_head("/nope/x.sqlite", "A", "B", 2026))

    def test_reader_line_summarises_both_splits(self):
        history = finals.head_to_head(self.db_path, "A", "B", 2026)
        line = finals.head_to_head_line(history, "A", "B")
        self.assertIn("B won 2 of 3 this season", line)
        self.assertIn("In finals", line)
        self.assertIn("Last time", line)

    def test_reader_line_handles_no_history(self):
        self.assertIsNone(finals.head_to_head_line(None, "A", "B"))


class ContextTests(unittest.TestCase):
    def _predictions(self, round_name, round_id):
        return pd.DataFrame(
            [
                {
                    "game_id": 1,
                    "round_name": round_name,
                    "round_id": round_id,
                    "competition_year": 2026,
                    "team_home": "A",
                    "team_away": "B",
                }
            ]
        )

    def test_a_regular_round_is_not_finals(self):
        context = finals.finals_context(self._predictions("Round 24", 24))
        self.assertFalse(context["is_finals"])
        self.assertEqual(context["stage"], rounds.REGULAR)
        self.assertIsNone(context["ledger"])

    def test_a_finals_round_is_detected_and_named(self):
        context = finals.finals_context(self._predictions("Finals Week 3", 30))
        self.assertTrue(context["is_finals"])
        self.assertEqual(context["display_name"], "Preliminary Finals")
        self.assertEqual(context["week"], 3)

    def test_mode_off_restores_the_regular_email(self):
        with mock.patch.dict(os.environ, {finals.FINALS_MODE_ENV: "off"}):
            context = finals.finals_context(self._predictions("Grand Final", 31))
        self.assertFalse(context["is_finals"])
        # The stage still reports honestly so logs do not lie.
        self.assertEqual(context["stage"], rounds.GRAND_FINAL)

    def test_mode_on_forces_the_finals_path_out_of_season(self):
        with mock.patch.dict(os.environ, {finals.FINALS_MODE_ENV: "on"}):
            context = finals.finals_context(self._predictions("Round 12", 12))
        self.assertTrue(context["is_finals"])
        self.assertEqual(context["stage"], rounds.FINALS_WEEK_1)

    def test_an_unknown_mode_falls_back_to_auto(self):
        with mock.patch.dict(os.environ, {finals.FINALS_MODE_ENV: "banana"}):
            self.assertEqual(finals.resolve_finals_mode(), "auto")

    def test_empty_predictions_are_safe(self):
        context = finals.finals_context(pd.DataFrame())
        self.assertFalse(context["is_finals"])


class SubjectTests(unittest.TestCase):
    def test_each_stage_gets_its_own_subject(self):
        subjects = {finals.finals_subject(s, 2026) for s in rounds.FINALS_STAGES}
        self.assertEqual(len(subjects), 4)
        self.assertTrue(all("2026" in subject for subject in subjects))

    def test_a_regular_round_has_no_finals_subject(self):
        self.assertIsNone(finals.finals_subject(rounds.REGULAR, 2026))


if __name__ == "__main__":
    unittest.main()
