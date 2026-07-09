import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from pipeline.common.model_training import joker_policy as jp


def _build_training_rows():
    rows = []
    game_id = 1000
    for season in [2024, 2025]:
        for round_id in [1, 2, 3, 4]:
            # Two matches per round keeps tests fast while still giving signal.
            odds_set = [
                (1.45 + (round_id * 0.03), 2.75 - (round_id * 0.02)),
                (1.65 + (round_id * 0.01), 2.20 - (round_id * 0.01)),
            ]
            for home_odds, away_odds in odds_set:
                rows.append(
                    {
                        "game_id": game_id,
                        "competition_year": season,
                        "round_id": round_id,
                        "team_head_to_head_odds_home": home_odds,
                        "team_head_to_head_odds_away": away_odds,
                    }
                )
                game_id += 1
    return pd.DataFrame(rows)


class JokerPolicyTrainingTests(unittest.TestCase):
    def test_backtest_returns_policy_payload(self):
        training_data = _build_training_rows()
        env = {
            "FOOTY_TIPPER_JOKER_MIN_ROUND_COVERAGE": "1.0",
            "FOOTY_TIPPER_JOKER_MIN_MATCHES_PER_ROUND": "2",
            "FOOTY_TIPPER_JOKER_MIN_ROUNDS_PER_SEASON": "3",
            "FOOTY_TIPPER_JOKER_BACKTEST_SIMULATIONS": "600",
            "FOOTY_TIPPER_JOKER_BACKTEST_FIELD_SIZE": "30",
            "FOOTY_TIPPER_JOKER_BACKTEST_SEED": "123",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            policy = jp.run_joker_policy_backtest(training_data)

        self.assertEqual(policy.get("status"), "ok")
        self.assertGreaterEqual(int(policy.get("seasons_evaluated", 0)), 1)
        recommended = policy.get("recommended_strategy_by_scenario", {})
        self.assertIn("lead", recommended)
        self.assertIn("neutral", recommended)
        self.assertIn("chase", recommended)
        self.assertIn(recommended["lead"], jp.VALID_STRATEGIES)
        self.assertIn(recommended["neutral"], jp.VALID_STRATEGIES)
        self.assertIn(recommended["chase"], jp.VALID_STRATEGIES)

    def test_backtest_reports_no_joker_baseline_and_lift(self):
        training_data = _build_training_rows()
        env = {
            "FOOTY_TIPPER_JOKER_MIN_ROUND_COVERAGE": "1.0",
            "FOOTY_TIPPER_JOKER_MIN_MATCHES_PER_ROUND": "2",
            "FOOTY_TIPPER_JOKER_MIN_ROUNDS_PER_SEASON": "3",
            "FOOTY_TIPPER_JOKER_BACKTEST_SIMULATIONS": "600",
            "FOOTY_TIPPER_JOKER_BACKTEST_FIELD_SIZE": "30",
            "FOOTY_TIPPER_JOKER_BACKTEST_SEED": "123",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            policy = jp.run_joker_policy_backtest(training_data)

        self.assertEqual(policy.get("status"), "ok")
        self.assertEqual(policy.get("version"), 2)
        for record in policy.get("scenario_results", []):
            self.assertIn("mean_win_prob_no_joker", record)
            self.assertIn("mean_joker_lift", record)
            # Doubling a round can only help against the same draws.
            self.assertGreaterEqual(
                record["mean_win_prob"] + 1e-9, record["mean_win_prob_no_joker"]
            )

    def test_strategy_ties_within_epsilon_prefer_points(self):
        training_data = _build_training_rows()
        env = {
            "FOOTY_TIPPER_JOKER_MIN_ROUND_COVERAGE": "1.0",
            "FOOTY_TIPPER_JOKER_MIN_MATCHES_PER_ROUND": "2",
            "FOOTY_TIPPER_JOKER_MIN_ROUNDS_PER_SEASON": "3",
            "FOOTY_TIPPER_JOKER_BACKTEST_SIMULATIONS": "600",
            "FOOTY_TIPPER_JOKER_BACKTEST_FIELD_SIZE": "30",
            "FOOTY_TIPPER_JOKER_BACKTEST_SEED": "123",
            # Huge epsilon: every strategy ties, so points must win everywhere.
            "FOOTY_TIPPER_JOKER_TIE_EPSILON": "1.0",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            policy = jp.run_joker_policy_backtest(training_data)

        recommended = policy.get("recommended_strategy_by_scenario", {})
        self.assertEqual(set(recommended.values()), {"points"})

    def test_save_policy_writes_json_file(self):
        training_data = _build_training_rows()
        env = {
            "FOOTY_TIPPER_JOKER_MIN_ROUND_COVERAGE": "1.0",
            "FOOTY_TIPPER_JOKER_MIN_MATCHES_PER_ROUND": "2",
            "FOOTY_TIPPER_JOKER_MIN_ROUNDS_PER_SEASON": "3",
            "FOOTY_TIPPER_JOKER_BACKTEST_SIMULATIONS": "500",
            "FOOTY_TIPPER_JOKER_BACKTEST_FIELD_SIZE": "20",
            "FOOTY_TIPPER_JOKER_BACKTEST_SEED": "321",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "models" / "joker_policy.json"
            with mock.patch.dict(os.environ, env, clear=False):
                policy = jp.save_joker_policy(training_data, output_path)

            self.assertTrue(output_path.exists())
            loaded = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded.get("default_strategy"), policy.get("default_strategy"))
            self.assertEqual(loaded.get("status"), policy.get("status"))


if __name__ == "__main__":
    unittest.main()
