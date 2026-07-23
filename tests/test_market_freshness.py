import unittest
from unittest import mock

import pandas as pd

from pipeline.common.use_predictions import distribution
from pipeline.common.use_predictions.staking import get_tipper_picks
from pipeline.ops.odds_gate import OddsCoverage


class MarketFreshnessTests(unittest.TestCase):
    def test_sanitizer_masks_stale_numeric_markets_and_preserves_fresh_rows(self):
        predictions = pd.DataFrame(
            [
                {
                    "game_id": 1,
                    "team_home": "Eels",
                    "team_away": "Panthers",
                    "home_team_win_prob": 0.40,
                    "home_team_lose_prob": 0.60,
                    "team_head_to_head_odds_home": 2.80,
                    "team_head_to_head_odds_away": 1.45,
                    "team_line_amount_home": 6.5,
                    "team_line_amount_away": -6.5,
                    "team_line_odds_home": 1.91,
                    "team_line_odds_away": 1.91,
                    "total_line": 42.5,
                    "total_over_odds": 1.91,
                    "total_under_odds": 1.91,
                },
                {
                    "game_id": 2,
                    "team_home": "Knights",
                    "team_away": "Roosters",
                    "home_team_win_prob": 0.30,
                    "home_team_lose_prob": 0.70,
                    "team_head_to_head_odds_home": 3.40,
                    "team_head_to_head_odds_away": 1.32,
                    "team_line_amount_home": 9.5,
                    "team_line_amount_away": -9.5,
                    "team_line_odds_home": 1.91,
                    "team_line_odds_away": 1.91,
                    "total_line": 44.5,
                    "total_over_odds": 1.91,
                    "total_under_odds": 1.91,
                },
            ]
        )
        coverage = OddsCoverage(
            competition_year=2026,
            round_id=21,
            round_name="Round 21",
            total_games=2,
            covered_games=1,
            stale_game_ids=(1,),
            fresh_game_ids=(2,),
            fresh_line_game_ids=(2,),
            fresh_total_game_ids=(2,),
        )

        with mock.patch.object(
            distribution,
            "current_round_odds_coverage",
            return_value=coverage,
        ):
            sanitized = distribution._sanitize_market_freshness(
                predictions, "runtime.sqlite"
            )

        self.assertFalse(bool(sanitized.loc[0, "market_odds_fresh"]))
        self.assertTrue(bool(sanitized.loc[1, "market_odds_fresh"]))
        for column in (
            "team_head_to_head_odds_home",
            "team_head_to_head_odds_away",
            "team_line_amount_home",
            "team_line_odds_home",
            "total_line",
            "total_over_odds",
        ):
            self.assertTrue(pd.isna(sanitized.loc[0, column]))
            self.assertFalse(pd.isna(sanitized.loc[1, column]))

    def test_invalid_or_infinite_prices_never_create_value_picks(self):
        predictions = pd.DataFrame(
            [
                {
                    "game_id": 1,
                    "team_home": "Eels",
                    "team_away": "Panthers",
                    "home_team_result": "Win",
                    "home_team_win_prob": 0.80,
                    "home_team_lose_prob": 0.20,
                    "team_head_to_head_odds_home": float("inf"),
                    "team_head_to_head_odds_away": 1.25,
                },
                {
                    "game_id": 2,
                    "team_home": "Bulldogs",
                    "team_away": "Warriors",
                    "home_team_result": "Loss",
                    "home_team_win_prob": 0.20,
                    "home_team_lose_prob": 0.80,
                    "team_head_to_head_odds_home": 5.0,
                    "team_head_to_head_odds_away": 0.0,
                },
            ]
        )

        picks = get_tipper_picks(predictions)

        self.assertTrue(picks.empty)

    def test_fresh_h2h_only_masks_older_line_and_total_families(self):
        predictions = pd.DataFrame(
            [
                {
                    "game_id": 1,
                    "team_head_to_head_odds_home": 1.90,
                    "team_head_to_head_odds_away": 2.10,
                    "team_line_amount_home": -6.5,
                    "team_line_amount_away": 6.5,
                    "team_line_odds_home": 1.91,
                    "team_line_odds_away": 1.91,
                    "total_line": 42.5,
                    "total_over_odds": 1.91,
                    "total_under_odds": 1.91,
                }
            ]
        )
        coverage = OddsCoverage(
            competition_year=2026,
            round_id=21,
            round_name="Round 21",
            total_games=1,
            covered_games=1,
            fresh_game_ids=(1,),
        )

        with mock.patch.object(
            distribution,
            "current_round_odds_coverage",
            return_value=coverage,
        ):
            sanitized = distribution._sanitize_market_freshness(
                predictions, "runtime.sqlite"
            )

        self.assertTrue(bool(sanitized.loc[0, "market_odds_fresh"]))
        self.assertEqual(sanitized.loc[0, "team_head_to_head_odds_home"], 1.90)
        self.assertFalse(bool(sanitized.loc[0, "line_odds_fresh"]))
        self.assertFalse(bool(sanitized.loc[0, "total_odds_fresh"]))
        self.assertTrue(pd.isna(sanitized.loc[0, "team_line_amount_home"]))
        self.assertTrue(pd.isna(sanitized.loc[0, "total_line"]))


if __name__ == "__main__":
    unittest.main()
