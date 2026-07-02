import os
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from pipeline.common.use_predictions import comp_strategy as cs


def _predictions():
    """One near-coin-flip game where the model disagrees with the market,
    plus three games where model and market agree comfortably."""
    rows = [
        # Model narrowly tips HOME, market strongly favours AWAY.
        {"game_id": 1, "home_p": 0.54, "odds_h": 3.20, "odds_a": 1.35, "home": "Alpha", "away": "Bravo"},
        {"game_id": 2, "home_p": 0.78, "odds_h": 1.30, "odds_a": 3.50, "home": "Charlie", "away": "Delta"},
        {"game_id": 3, "home_p": 0.25, "odds_h": 3.40, "odds_a": 1.32, "home": "Echo", "away": "Foxtrot"},
        {"game_id": 4, "home_p": 0.70, "odds_h": 1.45, "odds_a": 2.80, "home": "Golf", "away": "Hotel"},
    ]
    return pd.DataFrame(
        [
            {
                "game_id": r["game_id"],
                "round_id": 20,
                "competition_year": 2026,
                "team_home": r["home"],
                "team_away": r["away"],
                "home_team_win_prob": r["home_p"] * 0.97,
                "home_team_lose_prob": (1 - r["home_p"]) * 0.97,
                "team_head_to_head_odds_home": r["odds_h"],
                "team_head_to_head_odds_away": r["odds_a"],
                "home_team_result": "Win" if r["home_p"] > 0.5 else "Loss",
                "predicted_margin": 4 if r["home_p"] > 0.5 else -4,
                "predicted_home_score": 24 if r["home_p"] > 0.5 else 16,
                "predicted_away_score": 16 if r["home_p"] > 0.5 else 24,
            }
            for r in rows
        ]
    )


def _future_rounds(n=6):
    """A realistic remaining schedule: 8-game rounds priced by the market."""
    return pd.DataFrame(
        {
            "round_id": [21 + i for i in range(n)],
            "mu": [5.2] * n,
            "sigma": [1.15] * n,
            "matches_considered": [8] * n,
        }
    )


def _recommend(env, future_rounds=6):
    base_env = {"FOOTY_TIPPER_COMP_SIMULATIONS": "20000", "FOOTY_TIPPER_COMP_FIELD_SIZE": "40"}
    with mock.patch.dict(os.environ, {**base_env, **env}, clear=False):
        with mock.patch.object(cs, "_future_round_metrics", return_value=_future_rounds(future_rounds)):
            return cs.get_comp_strategy_recommendation(
                Path("/nonexistent/db.sqlite"), Path("/nonexistent"), _predictions()
            )


class CompStrategyTests(unittest.TestCase):
    def test_off_mode_returns_off_status(self):
        rec = _recommend({"FOOTY_TIPPER_COMP_STRATEGY": "off"})
        self.assertEqual(rec["status"], "off")
        self.assertFalse(rec["available"])

    def test_deterministic(self):
        env = {"FOOTY_TIPPER_COMP_STRATEGY": "advisory", "FOOTY_TIPPER_COMP_GAP": "-8"}
        self.assertEqual(_recommend(env), _recommend(env))

    def test_adjusted_never_worse_than_baseline(self):
        for gap in ("-8", "0", "8"):
            rec = _recommend({"FOOTY_TIPPER_COMP_STRATEGY": "advisory", "FOOTY_TIPPER_COMP_GAP": gap})
            self.assertTrue(rec["available"])
            self.assertGreaterEqual(rec["p_win_adjusted"], rec["p_win_baseline"])

    def test_big_lead_shadows_the_field(self):
        # Leading by 8 with no future rounds: mirroring the field's tip on the
        # model-vs-market disagreement game removes relative variance.
        rec = _recommend({"FOOTY_TIPPER_COMP_STRATEGY": "advisory", "FOOTY_TIPPER_COMP_GAP": "-8"})
        self.assertTrue(rec["available"])
        self.assertEqual(rec["scenario"], "lead")
        flipped_ids = {d["game_id"] for d in rec["deviations"]}
        self.assertIn(1, flipped_ids)
        deviation = next(d for d in rec["deviations"] if d["game_id"] == 1)
        # Strategy tip must move to the market favourite (the away side).
        self.assertEqual(deviation["strategy_tip"], "Bravo")

    def test_auto_mode_applies_flips_consistently(self):
        rec = _recommend({"FOOTY_TIPPER_COMP_STRATEGY": "auto", "FOOTY_TIPPER_COMP_GAP": "-8"})
        self.assertEqual(rec["mode"], "auto")
        self.assertGreaterEqual(rec["tips_changed"], 1)
        adjusted = cs.apply_comp_strategy_to_predictions(_predictions(), rec)
        row = adjusted[adjusted["game_id"] == 1].iloc[0]
        self.assertEqual(row["home_team_result"], "Loss")
        self.assertLess(row["predicted_margin"], 0)
        self.assertLess(row["predicted_home_score"], row["predicted_away_score"])
        # Untouched games keep their tips.
        row2 = adjusted[adjusted["game_id"] == 2].iloc[0]
        self.assertEqual(row2["home_team_result"], "Win")

    def test_no_flip_outside_band(self):
        # Game 2 (p=0.78) and 3 (p=0.25) are outside the flip band and must
        # never be deviated even when trailing badly.
        rec = _recommend({"FOOTY_TIPPER_COMP_STRATEGY": "advisory", "FOOTY_TIPPER_COMP_GAP": "10"})
        flipped_ids = {d["game_id"] for d in rec["deviations"]}
        self.assertNotIn(2, flipped_ids)
        self.assertNotIn(3, flipped_ids)
        self.assertNotIn(4, flipped_ids)

    def test_persist_decision_writes_rows(self):
        import sqlite3
        import tempfile

        rec = _recommend({"FOOTY_TIPPER_COMP_STRATEGY": "advisory", "FOOTY_TIPPER_COMP_GAP": "-8"})
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            ok = cs.persist_comp_strategy_decision(db_path, rec, _predictions())
            self.assertTrue(ok)
            con = sqlite3.connect(str(db_path))
            try:
                rows = con.execute("SELECT COUNT(*) FROM comp_strategy_decisions").fetchone()[0]
            finally:
                con.close()
            self.assertEqual(rows, 4)


if __name__ == "__main__":
    unittest.main()
