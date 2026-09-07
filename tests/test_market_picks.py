import os
import unittest

import pandas as pd

from pipeline.common.use_predictions import staking
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


class ContextValueGuardTests(unittest.TestCase):
    """The guard exists, is off, and stays out of the way while it is off."""

    def setUp(self):
        self._previous = os.environ.pop("FOOTY_TIPPER_CONTEXT_VALUE_GUARD", None)

    def tearDown(self):
        os.environ.pop("FOOTY_TIPPER_CONTEXT_VALUE_GUARD", None)
        if self._previous is not None:
            os.environ["FOOTY_TIPPER_CONTEXT_VALUE_GUARD"] = self._previous

    def test_the_guard_is_off_by_default(self):
        self.assertFalse(staking.context_value_guard_enabled())

    def test_no_teams_are_guarded_while_it_is_off(self):
        self.assertEqual(
            staking.load_context_guarded_teams(predictions=_predictions()), set()
        )

    def test_picks_are_identical_with_the_guard_off(self):
        # update-model and the next live send must be untouched by this work, so
        # the unset flag has to reproduce the shipped frame exactly.
        picks = get_market_picks(
            _predictions(), _distributions(p_home_covers_line=0.70)
        )
        expected = pd.DataFrame(
            [
                {
                    "game_id": 1,
                    "fixture": "Storm v Broncos",
                    "market": "Line",
                    "selection": "Storm -6.5",
                }
            ]
        )
        self.assertEqual(len(picks), 1)
        for column, value in expected.iloc[0].items():
            self.assertEqual(picks.iloc[0][column], value)

    def test_an_enabled_guard_withholds_the_affected_fixture(self):
        os.environ["FOOTY_TIPPER_CONTEXT_VALUE_GUARD"] = "true"
        original = staking.load_context_guarded_teams
        staking.load_context_guarded_teams = lambda **_kwargs: {"storm"}
        try:
            picks = get_market_picks(
                _predictions(), _distributions(p_home_covers_line=0.70)
            )
        finally:
            staking.load_context_guarded_teams = original
        self.assertTrue(picks.empty)

    def test_an_unreadable_registry_leaves_picks_alone(self):
        os.environ["FOOTY_TIPPER_CONTEXT_VALUE_GUARD"] = "true"
        guarded = staking.load_context_guarded_teams(
            db_path="/nonexistent/path/to.sqlite", predictions=_predictions()
        )
        self.assertEqual(guarded, set())
