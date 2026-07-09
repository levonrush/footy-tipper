import unittest

import pandas as pd

from pipeline.common.use_predictions import sending_functions as sf


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
