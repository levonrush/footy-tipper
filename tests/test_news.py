import contextlib
import io
import os
import urllib.error
import unittest
from unittest import mock

import pandas as pd

from pipeline.common.use_predictions import banner, email_copy, news


class _UrlResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self.body


def _rss(*items):
    rows = []
    for title, published, description in items:
        rows.append(
            "<item>"
            f"<title><![CDATA[{title}]]></title>"
            f"<pubDate>{published}</pubDate>"
            f"<description><![CDATA[{description}]]></description>"
            "</item>"
        )
    return f"<rss><channel>{''.join(rows)}</channel></rss>".encode()


def _predictions():
    return pd.DataFrame(
        [
            {
                "game_id": 1,
                "round_id": 22,
                "competition_year": 2026,
                "round_name": "Round 22",
                "team_home": "Broncos",
                "team_away": "Knights",
                "home_team_result": "Loss",
                "home_team_win_prob": 0.41,
                "home_team_lose_prob": 0.59,
                "team_head_to_head_odds_home": 2.10,
                "team_head_to_head_odds_away": 1.75,
                "predicted_home_score": 12,
                "predicted_away_score": 13,
                "predicted_margin": -1,
            }
        ]
    )


class NewsRetrievalTests(unittest.TestCase):
    def test_google_headlines_are_cleaned_deduplicated_and_capped(self):
        first = _rss(
            (
                "Player signs shock deal",
                "Mon, 27 Jul 2026 01:00:00 GMT",
                '<a href="https://example.test">Full story</a> &amp; reaction',
            ),
            ("Coach faces axe", "Tue, 28 Jul 2026 01:00:00 GMT", "Details"),
        )
        second = _rss(
            (
                "Player signs shock deal",
                "Mon, 27 Jul 2026 01:00:00 GMT",
                "Duplicate syndication",
            ),
            ("Third headline", "Wed, 29 Jul 2026 01:00:00 GMT", "More"),
        )

        with mock.patch.object(
            news.urllib.request,
            "urlopen",
            side_effect=[_UrlResponse(first), _UrlResponse(second)],
        ):
            headlines = news._fetch_rss_headlines(max_items=2)

        self.assertEqual(headlines.count("Player signs shock deal"), 1)
        self.assertIn("Coach faces axe", headlines)
        self.assertNotIn("Third headline", headlines)
        self.assertIn("Full story & reaction", headlines)
        self.assertNotIn("<a", headlines)

    def test_one_google_feed_can_fail_while_the_other_succeeds(self):
        working = _rss(
            ("Origin star changes clubs", "Tue, 28 Jul 2026", "Confirmed"),
        )
        output = io.StringIO()

        with mock.patch.object(
            news.urllib.request,
            "urlopen",
            side_effect=[
                urllib.error.URLError("cloud IP blocked"),
                urllib.error.URLError("cloud IP blocked"),
                _UrlResponse(working),
            ],
        ), mock.patch.object(news.time, "sleep"), contextlib.redirect_stdout(output):
            headlines = news._fetch_rss_headlines()

        self.assertIn("Origin star changes clubs", headlines)
        self.assertIn("Google News general attempt 2/2", output.getvalue())
        self.assertIn("Google News drama returned 1 headline", output.getvalue())

    def test_google_feeds_are_interleaved_before_the_cap(self):
        general = _rss(
            ("General one", "Mon, 27 Jul 2026", ""),
            ("General two", "Tue, 28 Jul 2026", ""),
        )
        drama = _rss(
            ("Drama one", "Mon, 27 Jul 2026", ""),
            ("Drama two", "Tue, 28 Jul 2026", ""),
        )

        with mock.patch.object(
            news.urllib.request,
            "urlopen",
            side_effect=[_UrlResponse(general), _UrlResponse(drama)],
        ):
            headlines = news._fetch_rss_headlines(max_items=2)

        self.assertIn("General one", headlines)
        self.assertIn("Drama one", headlines)
        self.assertNotIn("General two", headlines)
        self.assertNotIn("Drama two", headlines)

    def test_malformed_xml_is_retried(self):
        working = _rss(("Recovered headline", "Wed, 29 Jul 2026", "Recovered"))

        with mock.patch.object(
            news,
            "_GOOGLE_NEWS_FEEDS",
            [("Google News test", "https://example.test/rss")],
        ), mock.patch.object(
            news.urllib.request,
            "urlopen",
            side_effect=[_UrlResponse(b"<not-rss"), _UrlResponse(working)],
        ), mock.patch.object(news.time, "sleep") as sleep:
            headlines = news._fetch_rss_headlines()

        self.assertIn("Recovered headline", headlines)
        sleep.assert_called_once_with(news._NEWS_RETRY_DELAY_SECONDS)

    def test_nrl_parser_uses_articles_and_skips_video_cards(self):
        html_body = b"""
        <html><body>
          <a href="/news/2026/07/29/big-signing/"
             aria-label="Signings Article - Star joins Knights. 4 minute read. Published 1 hour ago">
            <p class="card-content__text">Star joins Knights</p>
            <h3 class="card-content__topic">Signings</h3>
            <time datetime="2026-07-29T10:00:00Z">1 hour ago</time>
          </a>
          <a href="/news/2026/07/29/match-highlights/"
             aria-label="Highlights Video - Match highlights. 03:00 Min duration. Published 2 hours ago">
            <p class="card-content__text">Match highlights</p>
            <h3 class="card-content__topic">Highlights</h3>
          </a>
        </body></html>
        """

        records = news._parse_nrl_news_records(html_body, 20)

        self.assertEqual(
            records,
            [
                (
                    "Star joins Knights",
                    "2026-07-29T10:00:00Z",
                    "Signings",
                )
            ],
        )

    def test_nrl_dot_com_is_used_when_google_has_no_headlines(self):
        selected = "A star player has been stood down after a dramatic week."
        anthropic_client = mock.Mock()
        anthropic_client.messages.create.return_value = mock.Mock(
            content=[mock.Mock(text=selected)]
        )

        with mock.patch.object(
            news,
            "_fetch_google_news_records",
            return_value=([], ["Google failed"], []),
        ), mock.patch.object(
            news,
            "_fetch_nrl_dot_com_records",
            return_value=(
                [("Star stood down", "2026-07-29T10:00:00Z", "Breaking")],
                [],
            ),
        ):
            result = news._fetch_nrl_news_context(anthropic_client)

        self.assertEqual(result, selected)
        prompt = anthropic_client.messages.create.call_args.kwargs["messages"][0][
            "content"
        ]
        self.assertIn("Star stood down", prompt)

    def test_total_outage_emits_github_warning_and_remains_fail_soft(self):
        output = io.StringIO()
        anthropic_client = mock.Mock()

        with mock.patch.object(
            news,
            "_fetch_google_news_records",
            return_value=([], ["Google News general: HTTPError 429"], []),
        ), mock.patch.object(
            news,
            "_fetch_nrl_dot_com_records",
            return_value=([], ["NRL.com news: TimeoutError"],),
        ), mock.patch.dict(
            os.environ,
            {"GITHUB_ACTIONS": "true"},
            clear=False,
        ), contextlib.redirect_stdout(output):
            result = news._fetch_nrl_news_context(anthropic_client)

        self.assertIsNone(result)
        self.assertIn("::warning title=NRL news unavailable::", output.getvalue())
        self.assertIn("HTTPError 429", output.getvalue())
        anthropic_client.messages.create.assert_not_called()


class NewsContentPropagationTests(unittest.TestCase):
    def test_selected_news_is_rendered_and_sent_to_banner(self):
        selected = "The Knights signed a representative forward after a wild week."
        generated_copy = {
            "subject": "Round 22",
            "news_hit": None,
            "opening": "Round opening.",
            "closing": "Bring back the biff.",
        }

        with mock.patch.object(
            email_copy,
            "Anthropic",
            return_value=mock.sentinel.anthropic_client,
        ), mock.patch.object(
            email_copy,
            "_fetch_nrl_news_context",
            return_value=selected,
        ), mock.patch.object(
            email_copy,
            "_generate_claude_copy",
            return_value=generated_copy,
        ), mock.patch.object(
            email_copy,
            "_generate_dynamic_banner",
            return_value=None,
        ) as dynamic_banner, mock.patch.object(
            email_copy,
            "_resolve_banner_path",
            return_value=None,
        ):
            payload = email_copy.generate_reg_regan_email_payload(
                _predictions(),
                pd.DataFrame(),
                api_key="anthropic-key",
                folder_url=None,
                temperature=0.9,
                use_llm=True,
                openai_api_key="openai-key",
            )

        self.assertIn(selected, payload["plain_text"])
        self.assertIn(selected, payload["html_text"])
        self.assertEqual(dynamic_banner.call_args.kwargs["news_context"], selected)
        self.assertEqual(dynamic_banner.call_args.kwargs["news_hit"], selected)

    def test_total_news_outage_still_builds_round_based_payload(self):
        generated_copy = {
            "subject": "Round 22",
            "news_hit": None,
            "opening": "The Knights can win this round.",
            "closing": "Bring back the biff.",
        }

        with mock.patch.object(
            email_copy,
            "Anthropic",
            return_value=mock.sentinel.anthropic_client,
        ), mock.patch.object(
            email_copy,
            "_fetch_nrl_news_context",
            return_value=None,
        ), mock.patch.object(
            email_copy,
            "_generate_claude_copy",
            return_value=generated_copy,
        ), mock.patch.object(
            email_copy,
            "_generate_dynamic_banner",
            return_value=None,
        ), mock.patch.object(
            email_copy,
            "_resolve_banner_path",
            return_value=None,
        ):
            payload = email_copy.generate_reg_regan_email_payload(
                _predictions(),
                pd.DataFrame(),
                api_key="anthropic-key",
                folder_url=None,
                temperature=0.9,
                use_llm=True,
                openai_api_key="openai-key",
            )

        self.assertEqual(payload["subject"], "Round 22")
        self.assertIn("The Knights can win this round.", payload["plain_text"])
        self.assertNotIn("THIS WEEK IN LEAGUE", payload["plain_text"])

    def test_banner_prefers_news_hit_over_general_news_context(self):
        client = mock.Mock()
        client.messages.create.return_value = mock.Mock(
            content=[mock.Mock(text="Reg celebrates the breaking story.")]
        )

        banner._build_banner_edit_instruction(
            {"subject": "Round 22", "opening": "Round context"},
            client,
            news_context="General news context",
            news_hit="The primary breaking story",
        )

        prompt = client.messages.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("PRIMARY inspiration", prompt)
        self.assertIn("The primary breaking story", prompt)
        self.assertNotIn("General news context", prompt)


if __name__ == "__main__":
    unittest.main()
