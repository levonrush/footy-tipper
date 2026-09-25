"""Editorial NRL news for email copy, never model or registry evidence.

Structured, sourced Club Context cards now own model-adjacent weekly context.
Regular-season colour retains its explicit legacy flag. Finals use a separate,
default-on brief; neither path persists evidence or supplies model features.
"""

import datetime as dt
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from html import unescape
from html.parser import HTMLParser
import os
import re
from urllib.parse import urlencode, urlparse
import urllib.request
import xml.etree.ElementTree as ET

from pipeline.common.nrl_data.teams import NICKNAME_TO_NAME
from pipeline.common.use_predictions.llm import resolve_claude_model


_NRL_NEWS_FEEDS = [
    "https://news.google.com/rss/search?q=NRL+rugby+league&hl=en-AU&gl=AU&ceid=AU:en",
    "https://news.google.com/rss/search?q=NRL+rugby+league+scandal+drama&hl=en-AU&gl=AU&ceid=AU:en",
]


@dataclass(frozen=True)
class FinalsNews:
    editorial: str = ""
    banner: str = ""


class _PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def _plain_text(value):
    parser = _PlainText()
    parser.feed(value or "")
    return " ".join(unescape(" ".join(parser.parts)).split())


# Serious human events belong in the reviewed Context Watch, not in an
# unreviewed headline brief for Reg's banter. Routine injury/selection reports
# can inform prose, but must not become cartoon imagery.
_SENSITIVE_NEWS = re.compile(
    r"\b(?:death|dead|died|dies|dying|mourning|mourns?|grief|funeral|tragedy|tragic|"
    r"suicid\w*|cancer|diagnosis|illness|abuse|assault\w*|domestic|arrest\w*|"
    r"criminal|court|rape|sexual|racis\w*|homophob\w*|bereave\w*|tribute|"
    r"doing it for|fighting spirit|great wish|in memory|in honour)\b", re.I
)
_UNSAFE_BANNER_NEWS = re.compile(
    r"\b(?:injur\w*|concuss\w*|surg\w*|hospital\w*|suspend\w*|suspension|"
    r"banned|sack\w*|scandal\w*|alleg\w*|investigat\w*|mental|addict\w*)\b", re.I
)
_FOOTBALL_BANNER_NEWS = re.compile(
    r"\b(?:finals?|premiership|trophy|training|selection|captain|milestone|"
    r"comeback|wins?|victory|coach|homecoming)\b", re.I
)
_OTHER_COMPETITION_OR_BETTING = re.compile(
    r"\b(?:NRLW|NSWRL|NSW Cup|Queensland Cup|Jersey Flegg|Super League|"
    r"women's|womens|betting tips|best bets|score centre|live scores)\b", re.I
)


def _banner_safe(text):
    return bool(
        _FOOTBALL_BANNER_NEWS.search(text)
        and not _SENSITIVE_NEWS.search(text)
        and not _UNSAFE_BANNER_NEWS.search(text)
    )


def _team_terms(teams):
    names = {str(team).strip().lower() for team in teams if str(team).strip()}
    return names | {
        nickname for nickname, name in NICKNAME_TO_NAME.items()
        if name.lower() in names or nickname in names
    }


def _mentions_team(text, terms):
    return any(re.search(r"\b" + re.escape(term) + r"\b", text, re.I) for term in terms)


def _fetch_finals_news_context(predictions, finals, *, now=None, max_items=20):
    """A dated finalist-first brief and a separate safe subset for imagery.

    RSS metadata only; no article scraping, persistence or extra LLM call.
    Every failure costs editorial colour, never the email.
    """
    empty = FinalsNews()
    enabled = os.getenv("FOOTY_TIPPER_FINALS_NEWS_ENABLED", "true").strip().lower()
    if enabled not in {"1", "true", "yes", "y", "on"} or predictions.empty:
        return empty
    try:
        now = now or dt.datetime.now(dt.timezone.utc)
        cutoff = now - dt.timedelta(days=7)
        fixture_teams = {
            str(team) for column in ("team_home", "team_away")
            for team in predictions[column].dropna()
        }
        surviving = {
            team["team"] for team in ((finals or {}).get("premiership") or {}).get("teams", [])
            if team.get("alive") and team.get("team")
        }
        fixture_terms = _team_terms(fixture_teams)
        survivor_terms = _team_terms(surviving)
        team_query = " OR ".join(f'"{team}"' for team in sorted(fixture_teams | surviving))
        queries = [f"NRL ({team_query}) when:7d", "NRL finals rugby league when:7d"]
        items = []
        seen_urls, seen_titles = set(), set()
        for query in queries:
            url = "https://news.google.com/rss/search?" + urlencode({
                "q": query, "hl": "en-AU", "gl": "AU", "ceid": "AU:en",
            })
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(request, timeout=10) as response:
                    root = ET.fromstring(response.read())
                for item in root.findall(".//item"):
                    try:
                        published = parsedate_to_datetime(item.findtext("pubDate") or "")
                        if published.tzinfo is None:
                            published = published.replace(tzinfo=dt.timezone.utc)
                        if not cutoff <= published <= now:
                            continue
                    except (ValueError, TypeError, OverflowError):
                        continue
                    title = _plain_text(item.findtext("title"))
                    publisher = _plain_text(item.findtext("source"))
                    link = (item.findtext("link") or "").strip()
                    parsed = urlparse(link)
                    if not title or not publisher or parsed.scheme not in {"http", "https"} or not parsed.netloc:
                        continue
                    # Google appends the publisher to the title; removing it
                    # also deduplicates the same story across our two queries.
                    title = title.removesuffix(f" - {publisher}")
                    key = re.sub(r"\W+", " ", title).strip().lower()
                    if link in seen_urls or key in seen_titles:
                        continue
                    snippet = _plain_text(item.findtext("description"))[:300]
                    text = f"{title} {snippet}"
                    if _SENSITIVE_NEWS.search(text) or _OTHER_COMPETITION_OR_BETTING.search(f"{text} {publisher}"):
                        continue
                    seen_urls.add(link)
                    seen_titles.add(key)
                    priority = (0 if _mentions_team(text, fixture_terms) else
                                1 if _mentions_team(text, survivor_terms) else 2)
                    brief = (
                        f"- {title} — {publisher}, {published.isoformat()}. "
                        f"{snippet}\n  Source: {link}"
                    )
                    items.append((priority, -published.timestamp(), brief, _banner_safe(text)))
            except Exception as exc:
                print(f"Finals news: RSS unavailable ({type(exc).__name__}); trying remaining sources.")
        selected = sorted(items, key=lambda item: (item[0], item[1], item[2]))[:max_items]
        print(f"Finals news: {len(selected)} recent stories; {sum(item[3] for item in selected)} suitable for banner inspiration.")
        return FinalsNews(
            editorial="\n".join(item[2] for item in selected),
            banner="\n".join(item[2] for item in selected if item[3]),
        )
    except Exception as exc:
        print(f"Finals news unavailable ({type(exc).__name__}); continuing without news.")
        return empty


def _fetch_rss_headlines(max_items=20):
    """Fetch recent NRL headlines from Google News RSS. Returns plain text list or empty string."""
    import xml.etree.ElementTree as ET
    headlines = []
    for url in _NRL_NEWS_FEEDS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                xml_bytes = resp.read()
            root = ET.fromstring(xml_bytes)
            for item in root.findall(".//item")[:max_items]:
                title = (item.findtext("title") or "").strip()
                desc = (item.findtext("description") or "").strip()
                pub = (item.findtext("pubDate") or "").strip()
                if title:
                    headlines.append(f"- {title} ({pub}): {desc[:120]}")
        except Exception:
            continue
    return "\n".join(headlines[:max_items])


def _fetch_nrl_news_context(anthropic_client):
    """Return legacy editorial colour only when explicitly enabled."""
    enabled = os.getenv("FOOTY_TIPPER_LEGACY_NEWS_ENABLED", "false").strip().lower()
    if enabled not in {"1", "true", "yes", "y", "on"}:
        return None
    try:
        headlines = _fetch_rss_headlines()
        if not headlines:
            print("NRL news: RSS fetch returned nothing.")
            return None

        response = anthropic_client.messages.create(
            model=resolve_claude_model(),
            system=(
                "You are a news editor. Given a list of NRL rugby league headlines, "
                "pick the single most interesting, scandalous, or dramatic story from the past 7 days and summarise it in 2-3 sentences. "
                "Be specific — name the player, club, or incident. "
                "It could be anything: a scandal, a big signing, a code switch, a surprise result, a feud, a sacking, a comeback — whatever people in NRL circles are talking about most this week. "
                "Return only the summary. No preamble."
            ),
            messages=[{"role": "user", "content": f"Headlines:\n{headlines}"}],
            max_tokens=300,
        )
        text = response.content[0].text.strip() if response.content else None
        if text:
            print(f"NRL news: {text[:100]}...")
            return text
        return None
    except Exception as exc:
        print(f"NRL news fetch failed ({exc}). Skipping.")
        return None
