import unittest
from unittest import mock

import pandas as pd

from pipeline.common.club_context.product import (
    SHADOW_DISCLAIMER,
    context_copy_is_safe,
    normalize_context_cards,
    safe_context_copy,
)
from pipeline.common.use_predictions import sending_functions as sf
from pipeline.common.use_predictions import email_copy
from pipeline.common.use_predictions.site import _context_watch_card


def _predictions():
    return pd.DataFrame(
        [
            {
                "game_id": 101,
                "round_id": 15,
                "competition_year": 2026,
                "round_name": "Round 15",
                "team_home": "South Sydney Rabbitohs",
                "team_away": "Brisbane Broncos",
                "home_team_result": "Win",
                "home_team_win_prob": 0.61,
                "home_team_lose_prob": 0.37,
                "team_head_to_head_odds_home": 1.85,
                "team_head_to_head_odds_away": 2.05,
                "predicted_home_score": 24,
                "predicted_away_score": 20,
                "predicted_margin": 4,
            }
        ]
    )


def _empty_picks():
    return pd.DataFrame(
        columns=[
            "team",
            "opponent",
            "price",
            "price_min",
            "edge",
            "stake_fraction",
            "stake_amount",
        ]
    )


def _card(**overrides):
    card = {
        "event_id": "tribute-rabbitohs-arrow-whiteout-match-2026",
        "game_id": 101,
        "team_name": "South Sydney Rabbitohs",
        "category": "tribute_milestone",
        "phase": "tribute",
        "factual_summary": (
            "South Sydney confirmed a one-match whiteout jersey, guard of honour "
            "and pre-kick-off tribute for Jai Arrow."
        ),
        "source_url": "https://www.rabbitohs.com.au/news/arrow-whiteout",
        "source_title": "Rabbitohs confirm Round 15 tribute",
        "sensitive": True,
        "confidence": 0.98,
        "eligible": True,
    }
    card.update(overrides)
    return card


def _render_html(**overrides):
    kwargs = {
        "predictions": _predictions(),
        "tipper_picks": _empty_picks(),
        "folder_url": None,
        "opening": "Opening paragraph.",
        "closing": "Closing paragraph.",
        "banner_available": False,
        "joker_recommendation": None,
        "news_hit": None,
        "scoreboard": None,
    }
    kwargs.update(overrides)
    return sf._render_html_email(**kwargs)


def _render_plain(**overrides):
    kwargs = {
        "predictions": _predictions(),
        "tipper_picks": _empty_picks(),
        "folder_url": None,
        "subject": "Subject",
        "opening": "Opening paragraph.",
        "closing": "Closing paragraph.",
        "joker_recommendation": None,
        "news_hit": None,
        "scoreboard": None,
    }
    kwargs.update(overrides)
    return sf._render_plain_email(**kwargs)


class ClubContextCopySafetyTests(unittest.TestCase):
    def test_prohibited_causal_and_betting_language_falls_back_to_locked_fact(self):
        card = _card()
        for proposed in (
            "Playing for him will guarantee a win.",
            "The illness is a betting edge this round.",
            "Bet on the tragedy because the side cannot lose.",
        ):
            with self.subTest(proposed=proposed):
                self.assertFalse(context_copy_is_safe(proposed, sensitive=True))
                self.assertEqual(
                    safe_context_copy(card, proposed),
                    card["factual_summary"],
                )

    def test_sensitive_event_banter_falls_back_but_neutral_copy_can_pass(self):
        card = _card()
        self.assertEqual(
            safe_context_copy(card, "A bit of Reg banter about the tribute."),
            card["factual_summary"],
        )
        neutral = "A verified off-field event is on Context Watch this round."
        self.assertEqual(safe_context_copy(card, neutral), neutral)

    def test_card_requires_a_locked_fact_event_id_and_valid_source_link(self):
        self.assertEqual(normalize_context_cards([_card(event_id="")]), [])
        self.assertEqual(normalize_context_cards([_card(factual_summary="")]), [])
        self.assertEqual(normalize_context_cards([_card(source_url="javascript:x")]), [])
        self.assertEqual(normalize_context_cards([_card(eligible=False)]), [])

    def test_standard_context_uses_a_deterministic_reg_voice(self):
        card = _card(sensitive=False, category="leadership_change")
        normalized = normalize_context_cards([card])

        self.assertEqual(len(normalized), 1)
        self.assertTrue(normalized[0]["display_copy"].startswith("Reg's"))
        self.assertIn(card["factual_summary"], normalized[0]["display_copy"])


class ClubContextRenderTests(unittest.TestCase):
    def test_no_context_and_invalid_context_leave_email_byte_identical(self):
        baseline_html = _render_html()
        baseline_plain = _render_plain()

        self.assertEqual(baseline_html, _render_html(context_cards=[]))
        self.assertEqual(baseline_plain, _render_plain(context_cards=[]))
        self.assertEqual(
            baseline_html,
            _render_html(context_cards=[_card(source_url="not-a-link")]),
        )
        self.assertEqual(
            baseline_plain,
            _render_plain(context_cards=[_card(source_url="not-a-link")]),
        )

    def test_context_watch_is_sourced_and_explicitly_shadow_only(self):
        card = _card(editorial_copy="Playing for him will guarantee a win.")
        html_out = _render_html(context_cards=[card])
        plain_out = _render_plain(context_cards=[card])

        for output in (html_out, plain_out):
            self.assertIn("Context Watch", output.title())
            self.assertIn(SHADOW_DISCLAIMER, output)
            self.assertIn(card["factual_summary"], output)
            self.assertIn(card["source_url"], output)
            self.assertNotIn(card["editorial_copy"], output)

    def test_site_card_is_empty_without_eligible_context(self):
        self.assertEqual(_context_watch_card(None), "")
        self.assertEqual(_context_watch_card([]), "")
        self.assertEqual(
            _context_watch_card([_card(source_url="not-a-link")]),
            "",
        )

    def test_site_context_watch_uses_the_same_sourced_safe_copy(self):
        card = _card(editorial_copy="Playing for him will guarantee a win.")
        output = _context_watch_card([card])

        self.assertIn("Context Watch", output)
        self.assertIn(SHADOW_DISCLAIMER, output)
        self.assertIn(card["factual_summary"], output)
        self.assertIn(card["source_url"], output)
        self.assertNotIn(card["editorial_copy"], output)

    def test_sensitive_context_cannot_trigger_dynamic_banner_generation(self):
        with mock.patch.object(email_copy, "_generate_dynamic_banner") as generated, \
             mock.patch.object(email_copy, "_resolve_banner_path", return_value=None):
            payload = email_copy.generate_reg_regan_email_payload(
                _predictions(),
                _empty_picks(),
                api_key=None,
                folder_url=None,
                temperature=0.2,
                use_llm=False,
                context_cards=[_card(sensitive=True)],
            )

        generated.assert_not_called()
        self.assertEqual(payload["inline_images"], [])


if __name__ == "__main__":
    unittest.main()
