import datetime as dt
from email.utils import format_datetime
from html import escape
import io
import json
import os
from types import SimpleNamespace
import unittest
from unittest import mock

import pandas as pd

from pipeline.common.use_predictions import banner, email_copy, news
from pipeline.common.use_predictions.email_render import _render_html_email, _render_plain_email


NOW = dt.datetime(2026, 9, 25, 3, tzinfo=dt.timezone.utc)
FINALS = {"is_finals": True, "stage": "preliminary_final", "premiership": {
    "teams": [{"team": "Melbourne Storm", "alive": True}],
}}


def predictions():
    return pd.DataFrame([{
        "game_id": 1, "competition_year": 2026, "round_id": 30,
        "round_name": "Finals Week 3", "team_home": "Newcastle Knights",
        "team_away": "Penrith Panthers", "home_team_result": "Win",
        "home_team_win_prob": 0.6, "home_team_lose_prob": 0.4,
        "team_head_to_head_odds_home": 1.8, "team_head_to_head_odds_away": 2.1,
        "predicted_home_score": 24, "predicted_away_score": 18, "predicted_margin": 6,
    }])


def item(title, *, age=1, date=None, link=None, description="", publisher="NRL.com"):
    pub = date if date is not None else format_datetime(NOW - dt.timedelta(days=age))
    link = link or "https://example.com/" + title.replace(" ", "-")
    return (
        f"<item><title>{escape(title)} - {escape(publisher)}</title>"
        f"<source>{escape(publisher)}</source><pubDate>{escape(pub)}</pubDate>"
        f"<link>{escape(link)}</link><description>{escape(description)}</description></item>"
    )


def rss(*items):
    return io.BytesIO(("<rss><channel>" + "".join(items) + "</channel></rss>").encode())


class FinalsNewsTests(unittest.TestCase):
    def setUp(self):
        self.environment = mock.patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_recent_deduplicated_finalist_first_reporting_retains_sources(self):
        stories = [
            item("League announces finals entertainment", age=0),
            item("Storm train for finals", age=0),
            item("Knights captain ready for finals", age=2, description="<b>Captain</b> trains &amp; prepares"),
            item("Panthers selection news", age=1),
            item("Old Knights story", age=8),
            item("Future Knights story", age=-1),
            item("Undated Knights story", date="invalid"),
            item("No source Knights story", publisher=""),
            item("Invalid link Knights story", link="javascript:alert(1)"),
        ]
        with mock.patch.object(news.urllib.request, "urlopen", side_effect=[rss(*stories), rss(*stories)]):
            result = news._fetch_finals_news_context(predictions(), FINALS, now=NOW)
        self.assertEqual(result.editorial.count("Source:"), 4)
        self.assertLess(result.editorial.index("Panthers selection"), result.editorial.index("Storm train"))
        self.assertLess(result.editorial.index("Storm train"), result.editorial.index("League announces"))
        self.assertIn("NRL.com, 2026-09-23", result.editorial)
        self.assertIn("Captain trains & prepares", result.editorial)
        self.assertIn("https://example.com/Knights", result.editorial)
        for absent in ("Old Knights", "Future Knights", "Undated", "No source", "Invalid link", "<b>"):
            self.assertNotIn(absent, result.editorial)

    def test_sensitive_stories_excluded_and_injuries_only_in_prose(self):
        stories = [item("Knights finals injury update"), item("Panthers captain wins milestone"),
                   item("Club mourns tragic death before finals"),
                   item("Doing it for a friend: Knights fighting spirit"),
                   item("Knights v Panthers: Smith recalled; Jones returns"),
                   item("Knights finals betting tips"),
                   item("Panthers Grand Final preview", publisher="NSWRL")]
        with mock.patch.object(news.urllib.request, "urlopen", side_effect=[rss(*stories), rss()]):
            result = news._fetch_finals_news_context(predictions(), FINALS, now=NOW)
        self.assertIn("injury", result.editorial)
        self.assertNotIn("injury", result.banner)
        self.assertIn("milestone", result.banner)
        self.assertNotIn("death", result.editorial + result.banner)
        self.assertNotIn("fighting spirit", result.editorial)
        self.assertNotIn("betting tips", result.editorial)
        self.assertNotIn("NSWRL", result.editorial)
        self.assertNotIn("Smith recalled", result.editorial)

    def test_one_failed_feed_preserves_other_feed_and_total_failure_is_empty(self):
        with mock.patch.object(news.urllib.request, "urlopen", side_effect=[TimeoutError(), rss(item("Knights finals training"))]):
            self.assertIn("Knights", news._fetch_finals_news_context(predictions(), FINALS, now=NOW).editorial)
        with mock.patch.object(news.urllib.request, "urlopen", side_effect=TimeoutError()):
            self.assertEqual(news._fetch_finals_news_context(predictions(), FINALS, now=NOW), news.FinalsNews())

    def test_finals_default_is_enabled_independent_of_legacy_flag_and_can_opt_out(self):
        with mock.patch.object(news.urllib.request, "urlopen", side_effect=[rss(item("Knights finals training")), rss()]) as fetch:
            self.assertTrue(news._fetch_finals_news_context(predictions(), FINALS, now=NOW).editorial)
            self.assertEqual(fetch.call_count, 2)
        with mock.patch.dict(os.environ, {"FOOTY_TIPPER_FINALS_NEWS_ENABLED": "false", "FOOTY_TIPPER_LEGACY_NEWS_ENABLED": "true"}), mock.patch.object(news.urllib.request, "urlopen") as fetch:
            self.assertEqual(news._fetch_finals_news_context(predictions(), FINALS), news.FinalsNews())
            fetch.assert_not_called()

    def test_legacy_still_disabled_by_default(self):
        with mock.patch.object(news, "_fetch_rss_headlines") as fetch:
            self.assertIsNone(news._fetch_nrl_news_context(mock.Mock()))
            fetch.assert_not_called()


class FinalsNewsCopyTests(unittest.TestCase):
    def test_prompt_weaves_news_and_discards_model_returned_highlight(self):
        payload = {"subject": "Finals", "opening": "NRL.com reports training news.",
                   "closing": "Bring back the biff.", "news_hit": "Unwanted highlight"}
        client = mock.Mock()
        client.messages.create.return_value = SimpleNamespace(content=[SimpleNamespace(text=json.dumps(payload))])
        finals = dict(FINALS, market_picks=pd.DataFrame([{
            "market": "Total", "selection": "Under 45.5", "price": 1.9, "edge": 0.04,
        }]))
        with mock.patch.object(email_copy, "Anthropic", return_value=client):
            result = email_copy._generate_claude_copy(predictions(), pd.DataFrame(), "key", None, 0.9,
                                                     news_context="Sourced training story", finals=finals)
        prompt = client.messages.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("Sourced training story", prompt)
        self.assertIn("Weave 2-4", prompt)
        self.assertIn('"news_hit": null', prompt)
        self.assertNotIn("you MUST write news_hit", prompt)
        self.assertIn("Additional Total value pick: Under 45.5", prompt)
        self.assertIsNone(result["news_hit"])

    def test_payload_routes_finals_briefs_separately_without_legacy_fetch(self):
        copy = {"subject": "Finals", "opening": "Opening", "closing": "Closing", "news_hit": "Stray highlight"}
        with mock.patch.object(email_copy, "Anthropic"), \
             mock.patch.object(email_copy, "_fetch_finals_news_context", return_value=news.FinalsNews("Editorial injuries", "Safe training")) as fetch, \
             mock.patch.object(email_copy, "_fetch_nrl_news_context") as legacy, \
             mock.patch.object(email_copy, "_generate_claude_copy", return_value=copy) as generate, \
             mock.patch.object(email_copy, "_generate_dynamic_banner", return_value=None) as generate_banner, \
             mock.patch.object(email_copy, "_resolve_banner_path", return_value=None):
            output = email_copy.generate_reg_regan_email_payload(predictions(), pd.DataFrame(), "key", None, 0.9, finals=FINALS)
        fetch.assert_called_once()
        legacy.assert_not_called()
        self.assertEqual(generate.call_args.kwargs["news_context"], "Editorial injuries")
        self.assertEqual(generate_banner.call_args.kwargs["finals_news_context"], "Safe training")
        self.assertNotIn("Stray highlight", output["html_text"] + output["plain_text"])

    def test_renderers_suppress_direct_finals_highlight_but_keep_regular_highlight(self):
        for finals in (None, FINALS):
            kwargs = dict(predictions=predictions(), tipper_picks=pd.DataFrame(), folder_url=None,
                          opening="Opening", closing="Closing", news_hit="Distinct news highlight", finals=finals)
            html = _render_html_email(**kwargs, banner_available=False)
            plain = _render_plain_email(**kwargs, subject="Subject")
            for rendered in (html, plain):
                self.assertEqual("Distinct news highlight" in rendered, finals is None)

    def test_banner_only_uses_safe_brief_never_news_bearing_copy(self):
        for safe in ("Knights finals training", None):
            client = mock.Mock()
            client.messages.create.return_value = SimpleNamespace(content=[SimpleNamespace(text="Reg holds a trophy.")])
            banner._build_banner_edit_instruction(
                {"subject": "Injury subject", "opening": "Injury opening"}, client,
                news_context="Injury reporting", news_hit="Injury highlight", finals=FINALS,
                finals_news_context=safe,
            )
            prompt = client.messages.create.call_args.kwargs["messages"][0]["content"]
            self.assertNotIn("Injury", prompt)
            self.assertIn("preliminary final", prompt)
            if safe:
                self.assertIn(safe, prompt)

    def test_disabling_llm_avoids_all_news_fetches(self):
        with mock.patch.object(email_copy, "_fetch_finals_news_context") as finals_fetch, \
             mock.patch.object(email_copy, "_fetch_nrl_news_context") as legacy_fetch, \
             mock.patch.object(email_copy, "_generate_dynamic_banner", return_value=None), \
             mock.patch.object(email_copy, "_resolve_banner_path", return_value=None):
            email_copy.generate_reg_regan_email_payload(predictions(), pd.DataFrame(), None, None, 0.9,
                                                        finals=FINALS, use_llm=False)
        finals_fetch.assert_not_called()
        legacy_fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
