"""NRL news context fetching for the weekly email."""

import urllib.request


_NRL_NEWS_FEEDS = [
    "https://news.google.com/rss/search?q=NRL+rugby+league&hl=en-AU&gl=AU&ceid=AU:en",
    "https://news.google.com/rss/search?q=NRL+rugby+league+scandal+drama&hl=en-AU&gl=AU&ceid=AU:en",
]


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
    """Fetch NRL headlines then ask Claude to pick the top story. Always returns something."""
    try:
        headlines = _fetch_rss_headlines()
        if not headlines:
            print("NRL news: RSS fetch returned nothing.")
            return None

        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
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
