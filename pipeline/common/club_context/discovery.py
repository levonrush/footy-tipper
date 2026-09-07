"""Rights-aware candidate discovery for the Club Context registry.

ABC RSS and GDELT are observation channels, never event truth.  Their output is
stored as discovery evidence and remains ineligible until an official source or
two independent reputable confirmations are linked to one reviewed event.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

from .sources import DiscoveryCandidate, SourcePolicy
from .taxonomy import RightsStatus, SourceClass
from .time import parse_datetime, utc_iso


ABC_NRL_RSS = "https://www.abc.net.au/news/feed/2486/rss.xml"
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
USER_AGENT = "footy-tipper-club-context/1.0 (+facts-and-links-only)"


def _request_bytes(url: str, *, timeout: int = 15) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/rss+xml, application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _publisher_key(url: str) -> str:
    host = (urlparse(url).hostname or "unknown").lower()
    return host[4:] if host.startswith("www.") else host


def _strip_html(value: str | None) -> str | None:
    if not value:
        return None
    text = re.sub(r"<[^>]+>", " ", value)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:500] or None


class ABCRssAdapter:
    policy = SourcePolicy(
        source_key="abc_nrl_rss",
        source_class=SourceClass.PUBLIC_BROADCASTER,
        rights_status=RightsStatus.DISCOVERY_ONLY,
        automated_discovery_allowed=True,
        automated_extraction_allowed=False,
        terms_url="https://www.abc.net.au/conditions.htm",
    )

    def __init__(self, feed_url: str = ABC_NRL_RSS, *, timeout: int = 15):
        self.feed_url = feed_url
        self.timeout = timeout

    def discover(self, *, query: str, start_at_utc=None, end_at_utc=None, limit=100):
        root = ET.fromstring(_request_bytes(self.feed_url, timeout=self.timeout))
        tokens = [token for token in re.findall(r"[a-z0-9]+", query.lower()) if len(token) > 2]
        start = parse_datetime(start_at_utc) if start_at_utc else None
        end = parse_datetime(end_at_utc) if end_at_utc else None
        rows = []
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            if not title or not link:
                continue
            description = _strip_html(item.findtext("description"))
            searchable = f"{title} {description or ''}".lower()
            if tokens and not any(token in searchable for token in tokens):
                continue
            published = None
            raw_date = (item.findtext("pubDate") or "").strip()
            if raw_date:
                try:
                    published_dt = parsedate_to_datetime(raw_date)
                    published = utc_iso(published_dt)
                    if start and parse_datetime(published) < start:
                        continue
                    if end and parse_datetime(published) > end:
                        continue
                except Exception:
                    published = None
            rows.append(
                DiscoveryCandidate(
                    url=link,
                    title=title,
                    publisher_key=_publisher_key(link),
                    source_class=SourceClass.PUBLIC_BROADCASTER,
                    published_at_utc=published,
                    snippet=description,
                )
            )
            if len(rows) >= max(1, int(limit)):
                break
        return rows


class GDELTDocAdapter:
    policy = SourcePolicy(
        source_key="gdelt_doc",
        source_class=SourceClass.DISCOVERY_INDEX,
        rights_status=RightsStatus.DISCOVERY_ONLY,
        automated_discovery_allowed=True,
        automated_extraction_allowed=False,
        terms_url="https://www.gdeltproject.org/about.html",
    )

    def __init__(self, endpoint: str = GDELT_DOC_API, *, timeout: int = 20):
        self.endpoint = endpoint
        self.timeout = timeout

    @staticmethod
    def _gdelt_time(value) -> str | None:
        if value is None:
            return None
        raw = str(value).strip()
        for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S"):
            try:
                return utc_iso(dt.datetime.strptime(raw, fmt).replace(tzinfo=dt.timezone.utc))
            except ValueError:
                continue
        try:
            return utc_iso(raw)
        except Exception:
            return None

    def discover(self, *, query: str, start_at_utc=None, end_at_utc=None, limit=100):
        params = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": str(min(250, max(1, int(limit)))),
            "sort": "datedesc",
        }
        if start_at_utc:
            params["startdatetime"] = parse_datetime(start_at_utc).strftime("%Y%m%d%H%M%S")
        if end_at_utc:
            params["enddatetime"] = parse_datetime(end_at_utc).strftime("%Y%m%d%H%M%S")
        payload = json.loads(
            _request_bytes(
                f"{self.endpoint}?{urllib.parse.urlencode(params)}",
                timeout=self.timeout,
            ).decode("utf-8")
        )
        rows = []
        for article in payload.get("articles") or []:
            url = str(article.get("url") or "").strip()
            title = str(article.get("title") or "").strip()
            if not url or not title:
                continue
            rows.append(
                DiscoveryCandidate(
                    url=url,
                    title=title,
                    publisher_key=_publisher_key(url),
                    source_class=SourceClass.DISCOVERY_INDEX,
                    # GDELT's seendate is first observation, not a trustworthy
                    # publisher timestamp. The registry preserves that basis.
                    published_at_utc=self._gdelt_time(article.get("seendate")),
                    snippet=None,
                )
            )
        return rows[: max(1, int(limit))]


def discover_all(*, query: str, start_at_utc=None, end_at_utc=None, limit=100, adapters=None):
    """Run permitted adapters independently and return candidates + errors."""

    adapters = list(adapters or (ABCRssAdapter(), GDELTDocAdapter()))
    candidates: list[tuple[SourcePolicy, DiscoveryCandidate]] = []
    errors: list[str] = []
    seen: set[str] = set()
    for adapter in adapters:
        if not adapter.policy.automated_discovery_allowed:
            continue
        try:
            found = adapter.discover(
                query=query,
                start_at_utc=start_at_utc,
                end_at_utc=end_at_utc,
                limit=limit,
            )
        except Exception as exc:
            errors.append(f"{adapter.policy.source_key}: {exc}")
            continue
        for item in found:
            key = item.url.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append((adapter.policy, item))
    return candidates, errors


__all__ = [
    "ABC_NRL_RSS",
    "GDELT_DOC_API",
    "ABCRssAdapter",
    "GDELTDocAdapter",
    "discover_all",
]
