"""Shared HTTP helpers for nrl.com ingestion.

Mirrors the session conventions used by pipeline/common/lineups/ingest.py
(browser-like headers, robots.txt awareness, polite request delay) so both
scrapers present the same footprint to nrl.com.
"""

from __future__ import annotations

import html
import json
import time
from dataclasses import dataclass

import requests

ROBOTS_URL = "https://www.nrl.com/robots.txt"
DRAW_DATA_URL_TEMPLATE = (
    "https://www.nrl.com/draw/data?competition=111&round={round_id}&season={season}"
)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass
class FetchConfig:
    request_timeout_seconds: int = 30
    request_sleep_seconds: float = 0.15


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    return session


def _looks_like_bot_challenge(text: str) -> bool:
    # nrl.com fronts /draw/data with a small cookie-setting interstitial on a
    # session's first request; the retry (with cookies) passes.
    return len(text) < 2048 and "<html" in text[:200].lower()


def fetch_text(
    session: requests.Session,
    url: str,
    config: FetchConfig,
) -> str:
    response = session.get(url, timeout=config.request_timeout_seconds)
    response.raise_for_status()
    if config.request_sleep_seconds > 0:
        time.sleep(config.request_sleep_seconds)
    if _looks_like_bot_challenge(response.text):
        response = session.get(url, timeout=config.request_timeout_seconds)
        response.raise_for_status()
        if config.request_sleep_seconds > 0:
            time.sleep(config.request_sleep_seconds)
    return response.text


def robots_disallows(session: requests.Session, config: FetchConfig, path: str) -> bool:
    """Conservative robots.txt check for the wildcard user agent.

    Fails open (returns False) when robots.txt cannot be fetched, matching the
    behaviour of the existing lineup scraper.
    """
    try:
        robots = fetch_text(session, ROBOTS_URL, config)
    except Exception:
        return False

    applies = False
    for line in robots.splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("user-agent:"):
            applies = lower.split(":", 1)[1].strip() == "*"
        elif applies and lower.startswith("disallow:"):
            rule = stripped.split(":", 1)[1].strip()
            if rule and path.startswith(rule):
                return True
    return False


def parse_json_or_q_data(raw: str, element_id: str) -> dict | None:
    """Parse either a raw JSON response or an HTML page with a q-data payload."""
    text = raw.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except Exception:
            return None
    return extract_q_data(raw, element_id)


def extract_q_data(page_html: str, element_id: str) -> dict | None:
    """Extract the JSON payload from `<div id="{element_id}" q-data="...">`.

    Uses a targeted scan rather than a full BeautifulSoup parse: the pages are
    ~300 KB and this is called thousands of times during backfill.
    """
    marker = f'id="{element_id}"'
    start = page_html.find(marker)
    if start == -1:
        return None
    tag_end = page_html.find(">", start)
    if tag_end == -1:
        return None
    tag = page_html[start:tag_end]
    attr_start = tag.find('q-data="')
    if attr_start == -1:
        # attribute may precede the id within the same tag; rescan from tag open
        tag_open = page_html.rfind("<", 0, start)
        tag = page_html[tag_open:tag_end]
        attr_start = tag.find('q-data="')
        if attr_start == -1:
            return None
    attr_start += len('q-data="')
    attr_end = tag.find('"', attr_start)
    if attr_end == -1:
        return None
    try:
        return json.loads(html.unescape(tag[attr_start:attr_end]))
    except Exception:
        return None
