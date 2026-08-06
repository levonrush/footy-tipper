import os
import unittest
from unittest import mock

import pandas as pd

from pipeline.common.use_predictions import sending_functions as sf


class ValuePickTests(unittest.TestCase):
    def test_selects_tipped_side_with_positive_edge_per_match(self):
        predictions = pd.DataFrame(
            [
                {
                    "game_id": 1,
                    "team_home": "Home A",
                    "team_away": "Away A",
                    "home_team_result": "Loss",
                    "home_team_win_prob": 0.40,
                    "home_team_lose_prob": 0.55,
                    "team_head_to_head_odds_home": 1.70,
                    "team_head_to_head_odds_away": 2.20,
                },
                {
                    "game_id": 2,
                    "team_home": "Home B",
                    "team_away": "Away B",
                    "home_team_result": "Win",
                    "home_team_win_prob": 0.50,
                    "home_team_lose_prob": 0.48,
                    "team_head_to_head_odds_home": 1.80,
                    "team_head_to_head_odds_away": 2.00,
                },
            ]
        )

        picks = sf.get_tipper_picks(predictions)

        # Game 1's tipped (away) side has positive EV; game 2's tipped side does not.
        # Only sides the model tips to win are eligible.
        self.assertEqual(len(picks), 1)
        self.assertEqual(picks.iloc[0]["game_id"], 1)
        self.assertEqual(picks.iloc[0]["team"], "Away A")
        self.assertEqual(picks.iloc[0]["side"], "away")
        # Fair price comes from the two-way probability, matching the two-way
        # decimal price it is compared against: 0.55 / (0.40 + 0.55).
        self.assertAlmostEqual(float(picks.iloc[0]["price_min"]), 0.95 / 0.55, places=6)

    def test_untipped_side_never_picked_even_with_edge(self):
        predictions = pd.DataFrame(
            [
                {
                    "game_id": 3,
                    "team_home": "Home X",
                    "team_away": "Away X",
                    "home_team_result": "Win",
                    "home_team_win_prob": 0.55,
                    "home_team_lose_prob": 0.42,
                    "team_head_to_head_odds_home": 1.70,
                    "team_head_to_head_odds_away": 3.20,
                }
            ]
        )

        # Two-way: home 0.567 at $1.70 has no edge; away 0.433 at $3.20 has a
        # large one, but the model tips home, so nothing is eligible.
        picks = sf.get_tipper_picks(predictions)
        self.assertEqual(len(picks), 0)

    def test_kelly_fraction_and_cap_with_bankroll(self):
        predictions = pd.DataFrame(
            [
                {
                    "game_id": 9,
                    "team_home": "Home C",
                    "team_away": "Away C",
                    "home_team_result": "Win",
                    # Coherent and draw-free, so the arithmetic below is exact.
                    "home_team_win_prob": 0.60,
                    "home_team_lose_prob": 0.40,
                    "team_head_to_head_odds_home": 2.50,
                    "team_head_to_head_odds_away": 3.50,
                }
            ]
        )

        env = {
            "FOOTY_TIPPER_KELLY_FRACTION": "1.0",
            "FOOTY_TIPPER_MAX_STAKE_FRACTION": "0.02",
            "FOOTY_TIPPER_MIN_VALUE_EDGE": "0.01",
            "FOOTY_TIPPER_STAKE_MODE": "bankroll",
            "FOOTY_TIPPER_BANKROLL": "1000",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            picks = sf.get_tipper_picks(predictions)

        self.assertEqual(len(picks), 1)
        row = picks.iloc[0]

        # Home edge = 0.60*2.50 - 1 = 0.50
        # Full Kelly = edge/(odds-1) = 0.50/1.50 = 0.333...
        # Capped at 2%.
        self.assertAlmostEqual(float(row["kelly_full"]), 1 / 3, places=6)
        self.assertAlmostEqual(float(row["kelly_capped_fraction"]), 0.02, places=8)
        self.assertAlmostEqual(float(row["stake_fraction"]), 0.02, places=8)
        self.assertAlmostEqual(float(row["stake_amount"]), 20.0, places=8)

    def test_draw_deflated_probability_no_longer_hides_an_edge(self):
        """A pick that the stored (draw-deflated) probability would suppress.

        Stored 0.485 / 0.475 with a 4% draw is a conditional of 0.505. At $2.10
        the deflated number gives an edge of 0.0185, under the 2% floor, while
        the two-way number gives 0.0605 -- the units error was worth roughly the
        whole edge threshold.
        """
        predictions = pd.DataFrame(
            [
                {
                    "game_id": 21,
                    "team_home": "Home D",
                    "team_away": "Away D",
                    "home_team_result": "Win",
                    "home_team_win_prob": 0.485,
                    "home_team_lose_prob": 0.475,
                    "team_head_to_head_odds_home": 2.10,
                    "team_head_to_head_odds_away": 1.80,
                }
            ]
        )

        picks = sf.get_tipper_picks(predictions)

        self.assertEqual(len(picks), 1)
        self.assertEqual(picks.iloc[0]["team"], "Home D")
        two_way = 0.485 / 0.960
        self.assertAlmostEqual(float(picks.iloc[0]["model_prob"]), two_way, places=9)
        self.assertAlmostEqual(float(picks.iloc[0]["edge"]), two_way * 2.10 - 1.0, places=9)
        # The deflated probability would have fallen under the default floor.
        self.assertLess(0.485 * 2.10 - 1.0, 0.02)

    def test_normalized_mode_sums_to_one(self):
        predictions = pd.DataFrame(
            [
                {
                    "game_id": 11,
                    "team_home": "A",
                    "team_away": "B",
                    "home_team_result": "Win",
                    "home_team_win_prob": 0.58,
                    "home_team_lose_prob": 0.34,
                    "team_head_to_head_odds_home": 2.10,
                    "team_head_to_head_odds_away": 3.30,
                },
                {
                    "game_id": 12,
                    "team_home": "C",
                    "team_away": "D",
                    "home_team_result": "Loss",
                    "home_team_win_prob": 0.44,
                    "home_team_lose_prob": 0.52,
                    "team_head_to_head_odds_home": 2.40,
                    "team_head_to_head_odds_away": 2.30,
                },
            ]
        )

        env = {
            "FOOTY_TIPPER_MIN_VALUE_EDGE": "0.005",
            "FOOTY_TIPPER_STAKE_MODE": "normalized",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            picks = sf.get_tipper_picks(predictions)

        self.assertGreaterEqual(len(picks), 1)
        self.assertAlmostEqual(float(picks["stake_fraction"].sum()), 1.0, places=6)

    def test_empty_predictions_returns_expected_schema(self):
        picks = sf.get_tipper_picks(pd.DataFrame())
        self.assertEqual(
            list(picks.columns),
            [
                "game_id",
                "team",
                "opponent",
                "side",
                "price",
                "price_min",
                "model_prob",
                "edge",
                "kelly_full",
                "kelly_fraction",
                "kelly_capped_fraction",
                "stake_fraction",
                "stake_amount",
            ],
        )


if __name__ == "__main__":
    unittest.main()
