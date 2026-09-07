import unittest

import pandas as pd

from pipeline.common.use_predictions.staking import MARKET_PICK_COLUMNS, get_market_picks


def _predictions(**overrides):
    row = {
        "game_id": 1,
        "team_home": "Storm",
        "team_away": "Broncos",
        "team_line_amount_home": -6.5,
        "team_line_odds_home": 1.90,
        "team_line_odds_away": 1.90,
        "total_line": 42.5,
        "total_over_odds": 1.90,
        "total_under_odds": 1.90,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _distributions(**overrides):
    row = {"game_id": 1, "p_home_covers_line": 0.5, "p_total_over": 0.5}
    row.update(overrides)
    return pd.DataFrame([row])


class MarketPickTests(unittest.TestCase):
    def test_a_fair_market_produces_no_picks(self):
        """Even money against a coin flip is a 5% loss, not an edge."""
        picks = get_market_picks(_predictions(), _distributions())
        self.assertTrue(picks.empty)
        self.assertEqual(list(picks.columns), MARKET_PICK_COLUMNS)

    def test_a_line_edge_is_found_and_labelled_from_the_home_side(self):
        picks = get_market_picks(
            _predictions(), _distributions(p_home_covers_line=0.70)
        )
        line = picks[picks["market"] == "Line"]
        self.assertEqual(len(line), 1)
        self.assertEqual(line.iloc[0]["selection"], "Storm -6.5")
        self.assertAlmostEqual(float(line.iloc[0]["model_prob"]), 0.70)
        self.assertAlmostEqual(float(line.iloc[0]["edge"]), 0.70 * 1.90 - 1.0)

    def test_the_other_side_of_the_line_is_the_complement(self):
        picks = get_market_picks(
            _predictions(), _distributions(p_home_covers_line=0.20)
        )
        line = picks[picks["market"] == "Line"]
        self.assertEqual(len(line), 1)
        self.assertEqual(line.iloc[0]["selection"], "Broncos +6.5")
        self.assertAlmostEqual(float(line.iloc[0]["model_prob"]), 0.80)

    def test_totals_are_priced_both_ways(self):
        over = get_market_picks(_predictions(), _distributions(p_total_over=0.72))
        self.assertEqual(over[over["market"] == "Total"].iloc[0]["selection"], "Over 42.5")
        under = get_market_picks(_predictions(), _distributions(p_total_over=0.25))
        self.assertEqual(
            under[under["market"] == "Total"].iloc[0]["selection"], "Under 42.5"
        )

    def test_stakes_are_normalised_across_every_selected_pick(self):
        picks = get_market_picks(
            _predictions(), _distributions(p_home_covers_line=0.70, p_total_over=0.72)
        )
        self.assertEqual(len(picks), 2)
        self.assertAlmostEqual(float(picks["stake_fraction"].sum()), 1.0, places=9)

    def test_a_stale_or_missing_line_is_skipped(self):
        picks = get_market_picks(
            _predictions(team_line_amount_home=None),
            _distributions(p_home_covers_line=0.70, p_total_over=0.72),
        )
        self.assertEqual(set(picks["market"]), {"Total"})

    def test_an_unpriced_market_is_skipped(self):
        picks = get_market_picks(
            _predictions(team_line_odds_home=None, team_line_odds_away=None),
            _distributions(p_home_covers_line=0.70),
        )
        self.assertTrue(picks.empty)

    def test_missing_distributions_yield_nothing_rather_than_failing(self):
        self.assertTrue(get_market_picks(_predictions(), None).empty)
        self.assertTrue(get_market_picks(_predictions(), pd.DataFrame()).empty)
        self.assertTrue(get_market_picks(pd.DataFrame(), _distributions()).empty)

    def test_a_game_without_a_stored_distribution_is_skipped(self):
        picks = get_market_picks(
            _predictions(game_id=99), _distributions(p_home_covers_line=0.70)
        )
        self.assertTrue(picks.empty)

    def test_a_null_cover_probability_is_skipped(self):
        picks = get_market_picks(
            _predictions(), _distributions(p_home_covers_line=None, p_total_over=0.72)
        )
        self.assertEqual(set(picks["market"]), {"Total"})


if __name__ == "__main__":
    unittest.main()
