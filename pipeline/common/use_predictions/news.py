"""NRL news context fetching for the weekly email."""

import html
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from pipeline.common.use_predictions.llm import resolve_claude_model


_GOOGLE_NEWS_FEEDS = [
    (
        "Google News general",
        "https://news.google.com/rss/search?q=NRL+rugby+league&hl=en-AU&gl=AU&ceid=AU:en",
    ),
    (
        "Google News drama",
        "https://news.google.com/rss/search?q=NRL+rugby+league+scandal+drama&hl=en-AU&gl=AU&ceid=AU:en",
    ),
]
# Backward-compatible URL list for callers that imported the old module constant.
_NRL_NEWS_FEEDS = [url for _, url in _GOOGLE_NEWS_FEEDS]
_NRL_NEWS_URL = "https://www.nrl.com/news/"
_NRL_ARTICLE_PATH = re.compile(r"^/news/\d{4}/\d{2}/\d{2}/", re.IGNORECASE)
_HTML_TAG = re.compile(r"<[^>]+>")
_NEWS_FETCH_ATTEMPTS = 2
_NEWS_FETCH_TIMEOUT_SECONDS = 8
_NEWS_RETRY_DELAY_SECONDS = 0.5


def _clean_text(value):
    """Collapse whitespace and remove the simple HTML found in RSS descriptions."""
    unescaped = html.unescape(str(value or ""))
    without_tags = _HTML_TAG.sub(" ", unescaped)
    return " ".join(without_tags.split())


def _exception_summary(exc):
    message = " ".join(str(exc).split())
    if len(message) > 240:
        message = f"{message[:237]}..."
    return f"{type(exc).__name__}: {message or 'no detail'}"


def _with_retries(source_name, loader):
    """Run a news loader with bounded retries and return records plus diagnostics."""
    diagnostics = []
    for attempt in range(1, _NEWS_FETCH_ATTEMPTS + 1):
        try:
            records = loader()
            if not records:
                raise ValueError("response contained no usable headlines")
            print(
                f"NRL news source: {source_name} returned "
                f"{len(records)} headline(s) on attempt {attempt}."
            )
            return records, diagnostics
        except Exception as exc:
            detail = (
                f"{source_name} attempt {attempt}/{_NEWS_FETCH_ATTEMPTS}: "
                f"{_exception_summary(exc)}"
            )
            diagnostics.append(detail)
            print(f"NRL news source failed: {detail}")
            if attempt < _NEWS_FETCH_ATTEMPTS:
                time.sleep(_NEWS_RETRY_DELAY_SECONDS)
    return [], diagnostics


def _request_bytes(url):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 Chrome/126.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(
        req,
        timeout=_NEWS_FETCH_TIMEOUT_SECONDS,
    ) as response:
        return response.read()


def _parse_rss_records(xml_bytes, max_items):
    root = ET.fromstring(xml_bytes)
    records = []
    for item in root.findall(".//item")[:max_items]:
        title = _clean_text(item.findtext("title"))
        if not title:
            continue
        records.append(
            (
                title,
                _clean_text(item.findtext("pubDate")),
                _clean_text(item.findtext("description"))[:240],
            )
        )
    return records


def _deduplicate_records(records, max_items):
    unique = []
    seen = set()
    for title, published, description in records:
        key = re.sub(r"\W+", "", title, flags=re.UNICODE).casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append((title, published, description))
        if len(unique) >= max_items:
            break
    return unique


def _format_records(records):
    lines = []
    for title, published, description in records:
        suffix = f" ({published})" if published else ""
        if description:
            suffix += f": {description}"
        lines.append(f"- {title}{suffix}")
    return "\n".join(lines)


def _fetch_google_news_records(max_items):
    records_by_source = []
    diagnostics = []
    successful_sources = []
    for source_name, url in _GOOGLE_NEWS_FEEDS:
        source_records, source_diagnostics = _with_retries(
            source_name,
            lambda url=url: _parse_rss_records(_request_bytes(url), max_items),
        )
        diagnostics.extend(source_diagnostics)
        if source_records:
            successful_sources.append(source_name)
            records_by_source.append(source_records)

    # Round-robin the feeds so the broad query cannot consume the entire cap
    # before the more targeted drama query contributes any candidates.
    records = []
    row_count = max((len(rows) for rows in records_by_source), default=0)
    for row_index in range(row_count):
        for source_records in records_by_source:
            if row_index < len(source_records):
                records.append(source_records[row_index])
    return (
        _deduplicate_records(records, max_items),
        diagnostics,
        successful_sources,
    )


def _parse_nrl_news_records(html_bytes, max_items):
    try:
        from bs4 import BeautifulSoup
    except Exception as exc:
        raise RuntimeError("BeautifulSoup is unavailable") from exc

    soup = BeautifulSoup(html_bytes, "lxml")
    records = []
    for anchor in soup.select("a[href]"):
        href = str(anchor.get("href") or "")
        if not _NRL_ARTICLE_PATH.match(urllib.parse.urlparse(href).path):
            continue
        aria_label = str(anchor.get("aria-label") or "")
        if " Video - " in aria_label:
            continue
        title_node = anchor.select_one(".card-content__text")
        title = _clean_text(title_node.get_text(" ", strip=True) if title_node else "")
        if not title:
            continue
        published_node = anchor.select_one("time[datetime]")
        published = _clean_text(
            published_node.get("datetime") if published_node else ""
        )
        topic_node = anchor.select_one(".card-content__topic")
        topic = _clean_text(
            topic_node.get_text(" ", strip=True) if topic_node else ""
        )
        records.append((title, published, topic))
        if len(records) >= max_items:
            break
    return _deduplicate_records(records, max_items)


def _fetch_nrl_dot_com_records(max_items):
    return _with_retries(
        "NRL.com news",
        lambda: _parse_nrl_news_records(_request_bytes(_NRL_NEWS_URL), max_items),
    )


def _fetch_rss_headlines(max_items=20):
    """Fetch clean, deduplicated Google News RSS headlines."""
    records, _, _ = _fetch_google_news_records(max_items)
    return _format_records(records)


def _github_warning(message):
    clean = " ".join(str(message).split())
    if os.getenv("GITHUB_ACTIONS", "").lower() == "true":
        escaped = clean.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print(f"::warning title=NRL news unavailable::{escaped}")
    else:
        print(f"NRL news warning: {clean}")


def _fetch_nrl_news_context(anthropic_client):
    """Fetch NRL headlines and ask Claude to select the top current story."""
    records, diagnostics, successful_sources = _fetch_google_news_records(20)
    if not records:
        print("NRL news: Google News returned no usable headlines; trying NRL.com.")
        records, fallback_diagnostics = _fetch_nrl_dot_com_records(20)
        diagnostics.extend(fallback_diagnostics)
        successful_sources = ["NRL.com news"] if records else []

    if not records:
        recent_errors = "; ".join(diagnostics[-3:])
        detail = "All Google News and NRL.com sources returned no usable headlines."
        if recent_errors:
            detail = f"{detail} Last errors: {recent_errors}"
        _github_warning(detail)
        return None

    print(
        f"NRL news: using {len(records)} unique headline(s) from "
        f"{', '.join(successful_sources)}."
    )
    headlines = _format_records(records)
    try:
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
            print(f"NRL news selected story: {text[:100]}...")
            return text
        print("NRL news selection returned an empty response. Skipping.")
        return None
    except Exception as exc:
        print(f"NRL news selection failed ({_exception_summary(exc)}). Skipping.")
        return None
