import json
import os
import sqlite3
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import pandas as pd

from pipeline.common.use_predictions import sending_functions as sf


class JokerRoundTests(unittest.TestCase):
    def setUp(self):
        self.fixtures = pd.DataFrame(
            [
                {
                    "game_id": 1,
                    "round_id": 1,
                    "competition_year": 2026,
                    "round_name": "Round 1",
                    "team_home": "Alpha",
                    "team_away": "Bravo",
                    "team_head_to_head_odds_home": 1.30,
                    "team_head_to_head_odds_away": 3.60,
                },
                {
                    "game_id": 2,
                    "round_id": 1,
                    "competition_year": 2026,
                    "round_name": "Round 1",
                    "team_home": "Charlie",
                    "team_away": "Delta",
                    "team_head_to_head_odds_home": 1.45,
                    "team_head_to_head_odds_away": 2.90,
                },
                {
                    "game_id": 3,
                    "round_id": 2,
                    "competition_year": 2026,
                    "round_name": "Round 2",
                    "team_home": "Echo",
                    "team_away": "Foxtrot",
                    "team_head_to_head_odds_home": 1.95,
                    "team_head_to_head_odds_away": 1.95,
                },
                {
                    "game_id": 4,
                    "round_id": 2,
                    "competition_year": 2026,
                    "round_name": "Round 2",
                    "team_home": "Golf",
                    "team_away": "Hotel",
                    "team_head_to_head_odds_home": 1.90,
                    "team_head_to_head_odds_away": 1.90,
                },
            ]
        )

    def _write_fixture_rows(self, db_path):
        con = sqlite3.connect(str(db_path))
        try:
            con.execute(
                """
                CREATE TABLE footy_tipping_data (
                    game_id INTEGER,
                    round_id INTEGER,
                    competition_year INTEGER,
                    round_name TEXT,
                    team_home TEXT,
                    team_away TEXT,
                    team_head_to_head_odds_home REAL,
                    team_head_to_head_odds_away REAL,
                    game_state_name TEXT
                )
                """
            )
            rows = []
            for row in self.fixtures.to_dict(orient="records"):
                rows.append(
                    (
                        int(row["game_id"]),
                        int(row["round_id"]),
                        int(row["competition_year"]),
                        row["round_name"],
                        row["team_home"],
                        row["team_away"],
                        float(row["team_head_to_head_odds_home"]),
                        float(row["team_head_to_head_odds_away"]),
                        "Pre Game",
                    )
                )
            con.executemany(
                """
                INSERT INTO footy_tipping_data (
                    game_id,
                    round_id,
                    competition_year,
                    round_name,
                    team_home,
                    team_away,
                    team_head_to_head_odds_home,
                    team_head_to_head_odds_away,
                    game_state_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            con.commit()
        finally:
            con.close()

    def test_points_strategy_prefers_high_expected_correct_tips(self):
        with mock.patch.dict(
            os.environ,
            {"FOOTY_TIPPER_JOKER_STRATEGY": "points"},
            clear=False,
        ):
            recommendation = sf.recommend_joker_round(
                self.fixtures,
                current_round_id=1,
                current_round_name="Round 1",
            )

        self.assertTrue(recommendation["available"])
        self.assertEqual(recommendation["recommended_round_id"], 1)
        self.assertTrue(recommendation["should_use_this_round"])
        self.assertIn("PLAY", recommendation["headline"])

    def test_chase_strategy_prefers_higher_variance_round(self):
        with mock.patch.dict(
            os.environ,
            {"FOOTY_TIPPER_JOKER_STRATEGY": "chase"},
            clear=False,
        ):
            recommendation = sf.recommend_joker_round(
                self.fixtures,
                current_round_id=1,
                current_round_name="Round 1",
            )

        self.assertTrue(recommendation["available"])
        self.assertEqual(recommendation["recommended_round_id"], 2)
        self.assertFalse(recommendation["should_use_this_round"])
        self.assertIn("HOLD", recommendation["headline"])

    def test_email_payload_includes_joker_call_section(self):
        predictions = pd.DataFrame(
            [
                {
                    "game_id": 99,
                    "home_team_result": "Win",
                    "team_home": "Knights",
                    "team_away": "Sea Eagles",
                    "team_head_to_head_odds_home": 1.80,
                    "team_head_to_head_odds_away": 2.05,
                    "home_team_win_prob": 0.57,
                    "home_team_lose_prob": 0.43,
                    "round_id": 1,
                    "competition_year": 2026,
                    "round_name": "Round 1",
                }
            ]
        )
        tipper_picks = sf.get_tipper_picks(pd.DataFrame())

        with mock.patch.dict(
            os.environ,
            {"FOOTY_TIPPER_JOKER_STRATEGY": "points"},
            clear=False,
        ):
            recommendation = sf.recommend_joker_round(
                self.fixtures,
                current_round_id=1,
                current_round_name="Round 1",
            )

        payload = sf.generate_reg_regan_email_payload(
            predictions,
            tipper_picks,
            api_key=None,
            folder_url=None,
            temperature=0.9,
            use_openai=False,
            joker_recommendation=recommendation,
        )

        self.assertIn("Joker round call:", payload["plain_text"])
        self.assertIn(recommendation["headline"], payload["plain_text"])
        self.assertIn("Joker round call", payload["html_text"])

    def test_holds_when_not_enough_future_rounds_have_odds(self):
        sparse_fixtures = self.fixtures.copy()
        sparse_fixtures.loc[sparse_fixtures["round_id"] == 2, "team_head_to_head_odds_home"] = pd.NA
        sparse_fixtures.loc[sparse_fixtures["round_id"] == 2, "team_head_to_head_odds_away"] = pd.NA

        with mock.patch.dict(
            os.environ,
            {
                "FOOTY_TIPPER_JOKER_STRATEGY": "points",
                "FOOTY_TIPPER_JOKER_MIN_ROUNDS_WITH_ODDS": "2",
            },
            clear=False,
        ):
            recommendation = sf.recommend_joker_round(
                sparse_fixtures,
                current_round_id=1,
                current_round_name="Round 1",
            )

        self.assertTrue(recommendation["available"])
        self.assertEqual(recommendation["recommended_round_id"], 1)
        self.assertFalse(recommendation["should_use_this_round"])
        self.assertIn("HOLD", recommendation["headline"])
        self.assertIn("min 2", recommendation["detail"])

    def test_auto_strategy_uses_learned_policy_for_chasing_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "models").mkdir(parents=True, exist_ok=True)
            (root / "models" / "joker_policy.json").write_text(
                json.dumps(
                    {
                        "default_strategy": "points",
                        "recommended_strategy_by_scenario": {
                            "lead": "protect",
                            "neutral": "points",
                            "chase": "chase",
                        },
                        "state_thresholds": {
                            "lead_max_gap": -3.0,
                            "chase_min_gap": 3.0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.dict(
                os.environ,
                {
                    "FOOTY_TIPPER_JOKER_STRATEGY": "auto",
                    "FOOTY_TIPPER_JOKER_POINTS_GAP": "6",
                },
                clear=False,
            ):
                context = sf._resolve_joker_strategy_context(root)

        self.assertEqual(context["strategy"], "chase")
        self.assertEqual(context["source"], "policy_auto")
        self.assertEqual(context["scenario"], "chase")
        self.assertTrue(context["policy_used"])

    def test_explicit_strategy_overrides_policy_auto(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "models").mkdir(parents=True, exist_ok=True)
            (root / "models" / "joker_policy.json").write_text(
                json.dumps(
                    {
                        "default_strategy": "points",
                        "recommended_strategy_by_scenario": {
                            "lead": "protect",
                            "neutral": "points",
                            "chase": "chase",
                        },
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.dict(
                os.environ,
                {
                    "FOOTY_TIPPER_JOKER_STRATEGY": "protect",
                    "FOOTY_TIPPER_JOKER_POINTS_GAP": "8",
                },
                clear=False,
            ):
                context = sf._resolve_joker_strategy_context(root)

        self.assertEqual(context["strategy"], "protect")
        self.assertEqual(context["source"], "explicit_env")
        self.assertFalse(context["policy_used"])

    def test_usage_write_gate_blocks_test_mode_persistence(self):
        with mock.patch.dict(
            os.environ,
            {
                "FOOTY_TIPPER_JOKER_STRATEGY": "points",
                "FOOTY_TIPPER_JOKER_MIN_MARGIN_RATIO": "0.0",
            },
            clear=False,
        ):
            recommendation = sf.recommend_joker_round(
                self.fixtures,
                current_round_id=1,
                current_round_name="Round 1",
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            outcome = sf.persist_joker_usage_if_applicable(
                db_path,
                recommendation,
                allow_write=False,
                source="unit_test",
            )
            usage = sf.get_joker_usage_for_year(db_path, 2026)

        self.assertFalse(outcome["recorded"])
        self.assertEqual(outcome["reason"], "write_disabled")
        self.assertIsNone(usage)

    def test_recommendation_respects_existing_usage_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            self._write_fixture_rows(db_path)

            predictions = pd.DataFrame(
                [
                    {
                        "round_id": 1,
                        "competition_year": 2026,
                        "round_name": "Round 1",
                    }
                ]
            )
            project_root = Path(__file__).resolve().parents[1]

            with mock.patch.dict(
                os.environ,
                {
                    "FOOTY_TIPPER_JOKER_STRATEGY": "points",
                    "FOOTY_TIPPER_JOKER_MIN_MARGIN_RATIO": "0.0",
                },
                clear=False,
            ):
                initial = sf.get_joker_round_recommendation(db_path, project_root, predictions)
            self.assertTrue(initial["should_use_this_round"])

            write_outcome = sf.persist_joker_usage_if_applicable(
                db_path,
                initial,
                allow_write=True,
                source="unit_test",
            )
            self.assertTrue(write_outcome["recorded"])

            with mock.patch.dict(
                os.environ,
                {
                    "FOOTY_TIPPER_JOKER_STRATEGY": "points",
                    "FOOTY_TIPPER_JOKER_MIN_MARGIN_RATIO": "0.0",
                },
                clear=False,
            ):
                after_usage = sf.get_joker_round_recommendation(db_path, project_root, predictions)

        self.assertTrue(after_usage["joker_already_used"])
        self.assertTrue(after_usage["should_use_this_round"])
        self.assertIn("PLAY", after_usage["headline"])
        self.assertEqual(after_usage["status"], "already_used_current_round")

    def test_recommendation_holds_in_later_round_after_usage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            self._write_fixture_rows(db_path)

            project_root = Path(__file__).resolve().parents[1]
            round_one_predictions = pd.DataFrame(
                [
                    {
                        "round_id": 1,
                        "competition_year": 2026,
                        "round_name": "Round 1",
                    }
                ]
            )
            round_two_predictions = pd.DataFrame(
                [
                    {
                        "round_id": 2,
                        "competition_year": 2026,
                        "round_name": "Round 2",
                    }
                ]
            )

            with mock.patch.dict(
                os.environ,
                {
                    "FOOTY_TIPPER_JOKER_STRATEGY": "points",
                    "FOOTY_TIPPER_JOKER_MIN_MARGIN_RATIO": "0.0",
                },
                clear=False,
            ):
                initial = sf.get_joker_round_recommendation(db_path, project_root, round_one_predictions)
            write_outcome = sf.persist_joker_usage_if_applicable(
                db_path,
                initial,
                allow_write=True,
                source="unit_test",
            )
            self.assertTrue(write_outcome["recorded"])

            with mock.patch.dict(
                os.environ,
                {
                    "FOOTY_TIPPER_JOKER_STRATEGY": "points",
                    "FOOTY_TIPPER_JOKER_MIN_MARGIN_RATIO": "0.0",
                },
                clear=False,
            ):
                later_round = sf.get_joker_round_recommendation(db_path, project_root, round_two_predictions)

        self.assertTrue(later_round["joker_already_used"])
        self.assertFalse(later_round["should_use_this_round"])
        self.assertEqual(later_round["status"], "already_used")
        self.assertIn("ALREADY USED", later_round["headline"])


if __name__ == "__main__":
    unittest.main()
