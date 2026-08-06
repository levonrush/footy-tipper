import unittest

import numpy as np
import pandas as pd

from pipeline.common.use_predictions import sending_functions as sf
from pipeline.common.use_predictions.email_copy import _build_prompt_input


def _predictions():
    return pd.DataFrame(
        [
            {
                "game_id": 1,
                "round_id": 15,
                "competition_year": 2026,
                "round_name": "Round 15",
                "team_home": "Knights",
                "team_away": "Storm",
                "home_team_result": "Loss",
                "home_team_win_prob": 0.22,
                "home_team_lose_prob": 0.78,
                "team_head_to_head_odds_home": 3.60,
                "team_head_to_head_odds_away": 1.30,
                "predicted_home_score": 12,
                "predicted_away_score": 28,
                "predicted_margin": -16,
            }
        ]
    )


def _scoreboard():
    return {
        "competition_year": 2026,
        "season_games": 92,
        "season_correct": 61,
        "season_accuracy": 61 / 92,
        "market_games": 92,
        "market_correct": 60,
        "market_accuracy": 60 / 92,
        "last_round_id": 14,
        "last_round_name": "Round 14",
        "last_round_games": 8,
        "last_round_correct": 7,
    }


def _joker():
    return {
        "headline": "HOLD JOKER THIS ROUND",
        "detail": "Round 15 is ranked #6/12 on expected correct tips (mu).",
        "available": True,
        "should_use_this_round": False,
        "recommended_round_name": "Round 22",
        "strategy_label": "Max expected points",
        "current_mu": 5.91,
        "current_sigma": 1.32,
    }


def _render_html(**overrides):
    kwargs = dict(
        predictions=_predictions(),
        tipper_picks=pd.DataFrame(
            columns=[
                "team", "opponent", "price", "price_min", "edge",
                "stake_fraction", "stake_amount",
            ]
        ),
        folder_url=None,
        opening="Opening paragraph.",
        closing="Closing paragraph.",
        banner_available=False,
        joker_recommendation=_joker(),
        news_hit=None,
        scoreboard=_scoreboard(),
    )
    kwargs.update(overrides)
    return sf._render_html_email(**kwargs)


class HtmlRenderTests(unittest.TestCase):
    def test_margin_is_derived_from_displayed_scoreline(self):
        predictions = _predictions()
        predictions.loc[0, "team_home"] = "Parramatta Eels"
        predictions.loc[0, "team_away"] = "Penrith Panthers"
        predictions.loc[0, "home_team_result"] = "Win"
        predictions.loc[0, "predicted_home_score"] = 17
        predictions.loc[0, "predicted_away_score"] = 14
        predictions.loc[0, "predicted_margin"] = 1

        html_out = _render_html(predictions=predictions)

        self.assertIn("Parramatta Eels by 3", html_out)
        self.assertNotIn("Parramatta Eels by 1", html_out)

    def test_missing_market_is_labelled_model_only(self):
        predictions = _predictions()
        predictions.loc[0, "team_head_to_head_odds_home"] = pd.NA
        predictions.loc[0, "team_head_to_head_odds_away"] = pd.NA

        html_out = _render_html(predictions=predictions)

        self.assertIn("Market data notice", html_out)
        self.assertIn("model-only", html_out)
        self.assertIn("no market edge or staking claim", html_out)

    def test_invalid_market_prices_render_as_unavailable(self):
        predictions = _predictions()
        predictions.loc[0, "team_head_to_head_odds_home"] = 0.0
        predictions.loc[0, "team_head_to_head_odds_away"] = np.inf

        html_out = _render_html(predictions=predictions)

        self.assertIn("Market data notice", html_out)
        self.assertIn("H n/a", html_out)
        self.assertIn("A n/a", html_out)
        self.assertNotIn("$0.00", html_out)
        self.assertNotIn("$inf", html_out)

    def test_stale_numeric_market_is_hidden_from_email_and_copy_prompt(self):
        predictions = _predictions()
        predictions["market_odds_fresh"] = False

        html_out = _render_html(predictions=predictions)
        fixture_lines, _, _ = _build_prompt_input(
            predictions,
            pd.DataFrame(),
            joker_recommendation=_joker(),
        )

        self.assertIn("Market data notice", html_out)
        self.assertIn("H n/a", html_out)
        self.assertIn("A n/a", html_out)
        self.assertIn("market Knights n/a, Storm n/a", fixture_lines)

    def test_badge_uses_tip_confidence_not_home_prob(self):
        html_out = _render_html()
        # Strong away favourite: green badge showing the tip probability.
        self.assertIn(">78%<", html_out)
        self.assertIn("#dcfce7", html_out)
        self.assertNotIn("#fee2e2", html_out)
        # Only the tipped team's probability is shown — no H/A split.
        self.assertNotIn("H 22%", html_out)
        self.assertNotIn("A 78%", html_out)
        self.assertIn(">Confidence</th>", html_out)

    def test_confidence_band_uses_the_draw_excluded_probability(self):
        """A 0.71 conditional with a 4% draw stores as 0.6816 and must stay green.

        Reading the stored probability raw would drop this tip out of the
        >= 70% band purely because of draw mass the model is not scored on.
        """
        predictions = _predictions()
        predictions.loc[0, "home_team_result"] = "Win"
        predictions.loc[0, "home_team_win_prob"] = 0.71 * 0.96
        predictions.loc[0, "home_team_lose_prob"] = 0.29 * 0.96

        html_out = _render_html(predictions=predictions)

        self.assertIn(">71%<", html_out)
        self.assertNotIn(">68%<", html_out)
        self.assertIn("#dcfce7", html_out)

    def test_joker_section_is_reader_friendly(self):
        html_out = _render_html()
        self.assertIn("HOLD JOKER THIS ROUND", html_out)
        self.assertIn("Best round left for it looks like Round 22.", html_out)
        # Diagnostics live only in the small footnote line.
        self.assertIn("Strategy: Max expected points", html_out)
        self.assertNotIn("Strategy source", html_out)

    def test_scoreboard_strip_renders(self):
        html_out = _render_html()
        self.assertIn("7/8", html_out)
        self.assertIn("Round 14", html_out)
        self.assertIn("66%", html_out)

    def test_scoreboard_omitted_when_missing(self):
        html_out = _render_html(scoreboard=None)
        self.assertNotIn("Round 14", html_out)

    def test_unsubscribe_footer(self):
        html_out = _render_html()
        self.assertIn("unsubscribe", html_out.lower())


class PlainRenderTests(unittest.TestCase):
    def test_plain_email_has_scoreboard_and_unsubscribe(self):
        text = sf._render_plain_email(
            _predictions(),
            pd.DataFrame(
                columns=[
                    "team", "opponent", "price", "price_min", "edge",
                    "stake_fraction", "stake_amount",
                ]
            ),
            None,
            "Subject",
            "Opening.",
            "Closing.",
            joker_recommendation=_joker(),
            scoreboard=_scoreboard(),
        )
        self.assertIn("The ledger — Round 14: 7/8 tips.", text)
        self.assertIn("Reply 'unsubscribe' to stop getting these.", text)


if __name__ == "__main__":
    unittest.main()
