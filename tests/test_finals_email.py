import unittest

import pandas as pd

from pipeline.common import rounds
from pipeline.common.use_predictions import finals as fin
from pipeline.common.use_predictions.email_render import (
    _default_subject,
    _render_html_email,
    _render_plain_email,
)


def _predictions(round_name="Round 24", round_id=24):
    return pd.DataFrame(
        [
            {
                "game_id": 1,
                "round_name": round_name,
                "round_id": round_id,
                "competition_year": 2026,
                "team_home": "Storm",
                "team_away": "Broncos",
                "position_home": 1,
                "position_away": 4,
                "home_team_result": "Win",
                "home_team_win_prob": 0.62,
                "home_team_lose_prob": 0.34,
                "draw_prob": 0.04,
                "predicted_home_score": 24,
                "predicted_away_score": 18,
                "predicted_margin": 6,
                "team_head_to_head_odds_home": 1.70,
                "team_head_to_head_odds_away": 2.20,
                "market_odds_fresh": True,
            }
        ]
    )


def _finals_payload(stage=rounds.FINALS_WEEK_1, **overrides):
    payload = {
        "mode": "auto",
        "stage": stage,
        "is_finals": True,
        "week": rounds.stage_week(stage),
        "display_name": rounds.stage_display_name(stage),
        "round_name": rounds.stage_feed_label(stage),
        "competition_year": 2026,
        "theme": fin.theme(stage),
        "ledger": None,
        "stakes": {1: "Qualifying final. Winner gets a week off."},
        "head_to_head": {1: "Storm won 2 of 3 this season."},
        "distributions": {
            1: {
                "p_one_score_game": 0.34,
                "p_home_by_13_plus": 0.21,
                "p_away_by_13_plus": 0.18,
                "median_total": 42.0,
            }
        },
        "market_picks": pd.DataFrame(
            [
                {
                    "game_id": 1,
                    "fixture": "Storm v Broncos",
                    "market": "Line",
                    "selection": "Storm -6.5",
                    "price": 1.90,
                    "price_min": 1.55,
                    "model_prob": 0.645,
                    "edge": 0.2255,
                    "stake_fraction": 1.0,
                    "stake_amount": pd.NA,
                }
            ]
        ),
        "premiership": {
            "available": True,
            "stage": stage,
            "simulations": 20000,
            "method": "Ratings price undrawn matchups.",
            "teams": [
                {"team": "Storm", "seed": 1, "p_premiership": 0.31,
                 "p_grand_final": 0.52, "alive": True},
                {"team": "Roosters", "seed": 8, "p_premiership": 0.0,
                 "p_grand_final": 0.0, "alive": False},
            ],
        },
    }
    payload.update(overrides)
    return payload


def _render(finals=None, predictions=None):
    predictions = _predictions() if predictions is None else predictions
    empty = pd.DataFrame(columns=["team", "opponent", "price", "price_min", "edge",
                                  "stake_fraction", "stake_amount"])
    html = _render_html_email(
        predictions, empty, None, "Opening.", "Closing.", banner_available=False,
        joker_recommendation={"headline": "HOLD JOKER THIS ROUND"}, finals=finals,
    )
    text = _render_plain_email(
        predictions, empty, None, "Subject", "Opening.", "Closing.",
        joker_recommendation={"headline": "HOLD JOKER THIS ROUND"}, finals=finals,
    )
    return html, text


class RegularRoundIsUnchangedTests(unittest.TestCase):
    """Twenty-seven weeks a year must render exactly as they always have."""

    def test_no_payload_and_a_non_finals_payload_render_identically(self):
        baseline_html, baseline_text = _render(finals=None)
        context = fin.finals_context(_predictions())
        self.assertFalse(context["is_finals"])
        html, text = _render(finals=context)
        self.assertEqual(html, baseline_html)
        self.assertEqual(text, baseline_text)

    def test_a_regular_round_keeps_the_joker_and_gains_nothing_else(self):
        html, text = _render(finals=None)
        self.assertIn("Joker round call", html)
        self.assertIn("Joker round call:", text)
        for absent in ("SPECIAL EDITION", "Road to the big dance", "Line and totals"):
            self.assertNotIn(absent, html)
            self.assertNotIn(absent, text)

    def test_the_original_accents_survive(self):
        html, _ = _render(finals=None)
        self.assertIn("border-left:4px solid #0f766e", html)
        self.assertIn("border-left:4px solid #16a34a", html)
        self.assertIn("border-left:4px solid #f59e0b", html)


class FinalsRenderTests(unittest.TestCase):
    def test_the_joker_slot_becomes_the_premiership_race(self):
        html, text = _render(finals=_finals_payload())
        self.assertNotIn("Joker round call", html)
        self.assertNotIn("Joker round call:", text)
        self.assertIn("Road to the big dance", html)
        self.assertIn("Road to the big dance:", text)

    def test_eliminated_teams_are_left_out_of_the_race_table(self):
        html, text = _render(finals=_finals_payload())
        self.assertIn("Storm", html)
        self.assertNotIn("Roosters", html)
        self.assertNotIn("Roosters", text)

    def test_the_ribbon_names_the_stage(self):
        for stage in rounds.FINALS_STAGES:
            with self.subTest(stage=stage):
                html, _ = _render(finals=_finals_payload(stage))
                self.assertIn(rounds.stage_display_name(stage).upper(), html)
                self.assertIn("SPECIAL EDITION", html)

    def test_the_heading_uses_the_supporter_name_not_the_feed_name(self):
        html, _ = _render(finals=_finals_payload(rounds.PRELIMINARY))
        self.assertIn("Preliminary Finals 2026", html)
        self.assertNotIn("Finals Week 3 2026", html)

    def test_stakes_shape_and_history_appear_under_the_fixture(self):
        html, text = _render(finals=_finals_payload())
        for expected in ("Qualifying final", "one-score game", "won 2 of 3 this season"):
            self.assertIn(expected, html)
            self.assertIn(expected, text)

    def test_line_and_totals_picks_are_rendered(self):
        html, text = _render(finals=_finals_payload())
        self.assertIn("Line and totals", html)
        self.assertIn("Storm -6.5", html)
        self.assertIn("Storm -6.5", text)

    def test_no_market_picks_means_no_section(self):
        html, text = _render(finals=_finals_payload(market_picks=pd.DataFrame()))
        self.assertNotIn("Line and totals", html)
        self.assertNotIn("Line and totals:", text)

    def test_an_unavailable_race_says_so_rather_than_inventing_numbers(self):
        payload = _finals_payload(premiership={"available": False, "reason": "no bracket"})
        html, text = _render(finals=payload)
        self.assertIn("unavailable", html)
        self.assertIn("unavailable", text)
        self.assertNotIn("%", html.split("Road to the big dance")[1][:400])

    def test_a_missing_race_entirely_still_renders(self):
        html, _ = _render(finals=_finals_payload(premiership=None))
        self.assertIn("Road to the big dance", html)

    def test_fixtures_without_extras_do_not_emit_an_empty_detail_row(self):
        payload = _finals_payload(stakes={}, head_to_head={}, distributions={})
        html, _ = _render(finals=payload)
        self.assertNotIn('colspan="4"', html)

    def test_the_finals_theme_replaces_the_regular_accents(self):
        html, _ = _render(finals=_finals_payload(rounds.GRAND_FINAL))
        self.assertIn("border-left:4px solid #b45309", html)
        self.assertNotIn("border-left:4px solid #0f766e", html)


class SubjectTests(unittest.TestCase):
    def test_a_regular_round_keeps_the_standard_subject(self):
        self.assertEqual(
            _default_subject(_predictions()),
            "Footy Tipper Predictions for Round 24 2026",
        )

    def test_finals_get_a_branded_subject(self):
        subject = _default_subject(_predictions(), finals=_finals_payload())
        self.assertIn("FINALS WEEK 1", subject)
        self.assertIn("2026", subject)

    def test_an_empty_round_is_unaffected(self):
        self.assertEqual(
            _default_subject(pd.DataFrame()), "Footy Tipper Predictions Update"
        )


if __name__ == "__main__":
    unittest.main()
