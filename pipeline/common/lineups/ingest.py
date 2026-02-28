from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import html
import json
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, Tag

from .normalization import normalize_player_name, normalize_team_name


TEAM_LISTS_TOPIC_URL = "https://www.nrl.com/news/topic/team-lists/"
ROBOTS_URL = "https://www.nrl.com/robots.txt"
DEFAULT_SITEMAP_URL = "https://www.nrl.com/sitemap/sitemap.xml"
DRAW_URL_TEMPLATE = "https://www.nrl.com/draw/?competition=111&round={round_id}&season={season}"
DRAW_DATA_URL_TEMPLATE = "https://www.nrl.com/draw/data?competition=111&round={round_id}&season={season}"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

PROFILE_PATTERN = re.compile(
    r"^(?P<position>.+?)\s+for\s+(?P<team>.+?)\s+is\s+number\s+(?P<number>\d+)\s+(?P<player>.+)$",
    re.IGNORECASE,
)

YEAR_PATTERN = re.compile(r"/news/(?P<year>\d{4})/")
ROUND_PATTERN = re.compile(r"\bround\s+(?P<round>\d{1,2})\b", re.IGNORECASE)
FINALS_WEEK_PATTERN = re.compile(r"\bfinals\s+week\s+(?P<week>\d{1,2})\b", re.IGNORECASE)
MATCHUP_PATTERN = re.compile(r"(?P<home>.+?)\s+v\s+(?P<away>.+)", re.IGNORECASE)
HOME_AWAY_PATTERN = re.compile(
    r"home\s+team\s+(?P<home>.+?)\s+away\s+team\s+(?P<away>.+?)(?:\s+scored|\s*$)",
    re.IGNORECASE,
)
LEGACY_MATCH_HEADER_PATTERN = re.compile(
    r"(?P<home>.+?)\s+v(?:s)?\s+(?P<away>.+?)(?:\s+[–-]\s+|,|\sat\b|\||$)",
    re.IGNORECASE,
)
LEGACY_TEAM_LABEL_PATTERN = re.compile(r"^(?P<label>[A-Za-z][A-Za-z'& .-]+?)\s*:\s*(?P<rest>.+)$")
LEGACY_NUMBERED_TEAM_PATTERN = re.compile(
    r"^(?P<label>[A-Za-z][A-Za-z'& .-]+?)\s+(?P<rest>\d{1,2}\.?\s+.+)$"
)
LEGACY_SECTION_MARKER_PATTERN = re.compile(
    r"\b(?P<label>Interchange(?:\s*\(from\))?|Reserves?)\b\s*:?",
    re.IGNORECASE,
)
NUMBERED_PLAYER_PATTERN = re.compile(
    r"(?<!\d)(?P<number>\d{1,2})\.?\s+(?P<player>.*?)(?=(?:(?<!\d)\d{1,2}\.?\s+)|$)",
    re.DOTALL,
)
PROGRESS_EVERY_N_URLS = 25
PROGRESS_EVERY_N_SITEMAPS = 10
PROGRESS_EVERY_N_DRAWS = 10
MAX_DRAW_ROUNDS = 35
POSITION_BY_SLOT = {
    1: "Fullback",
    2: "Wing",
    3: "Centre",
    4: "Centre",
    5: "Wing",
    6: "Five-Eighth",
    7: "Halfback",
    8: "Prop",
    9: "Hooker",
    10: "Prop",
    11: "Second Row",
    12: "Second Row",
    13: "Lock",
    14: "Interchange",
    15: "Interchange",
    16: "Interchange",
    17: "Interchange",
}

FINAL_MATCH_STATES = {"fulltime"}


@dataclass
class IngestionConfig:
    mode: str = "recent"  # recent | backfill
    start_year: int | None = None
    end_year: int | None = None
    max_articles: int = 80
    request_timeout_seconds: int = 30
    request_sleep_seconds: float = 0.15
    include_sitemap_in_recent: bool = False


ProgressCallback = Callable[[str], None]


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _emit_progress(progress_callback: ProgressCallback | None, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)


def _clean_text(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def parse_iso_datetime(value: str | None) -> str | None:
    raw = _clean_text(value)
    if not raw:
        return None

    try:
        parsed = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat()
    except Exception:
        return None


def parse_year_from_url(url: str) -> int | None:
    match = YEAR_PATTERN.search(url)
    if not match:
        return None
    return int(match.group("year"))


def parse_round_id(*candidates: str) -> int | None:
    for text in candidates:
        match = ROUND_PATTERN.search(_clean_text(text))
        if match:
            return int(match.group("round"))
    return None


def parse_round_name(*candidates: str) -> str | None:
    for text in candidates:
        clean = _clean_text(text).lower()
        if not clean:
            continue

        round_match = ROUND_PATTERN.search(clean)
        if round_match:
            return f"Round {int(round_match.group('round'))}"

        finals_match = FINALS_WEEK_PATTERN.search(clean)
        if finals_match:
            return f"Finals Week {int(finals_match.group('week'))}"

        if "grand final" in clean:
            return "Grand Final"
        if "preliminary final" in clean:
            return "Preliminary Final"
        if "elimination final" in clean:
            return "Elimination Final"
        if "qualifying final" in clean:
            return "Qualifying Final"
    return None


def is_nrl_mens_article(article_url: str, article_title: str) -> bool:
    slug = article_url.lower()
    title = article_title.lower()

    # Hard excludes: non-NRL men's competitions.
    blocked_tokens = (
        "nrlw",
        "harvey norman",
        "all stars",
        "state championship",
        "pacific championships",
        "ashes",
        "origin",
    )
    if any(token in slug or token in title for token in blocked_tokens):
        return False

    # Strong includes.
    include_tokens = (
        "nrl-team-lists",
        "nrl-late-mail",
        "nrl team lists",
        "nrl late mail",
        "pre-season challenge team lists",
    )
    return any(token in slug or token in title for token in include_tokens)


def _fetch(session: requests.Session, url: str, timeout_seconds: int) -> str:
    response = session.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    return response.text


def _load_q_data(tag: Tag | None) -> dict | None:
    if tag is None:
        return None
    raw = tag.get("q-data")
    if not raw:
        return None
    try:
        return json.loads(html.unescape(raw))
    except Exception:
        return None


def extract_topic_urls(topic_html: str, max_urls: int | None = None) -> list[str]:
    soup = BeautifulSoup(topic_html, "lxml")
    urls: list[str] = []
    seen: set[str] = set()

    for anchor in soup.select("a[href]"):
        href = _clean_text(anchor.get("href"))
        if not href:
            continue
        if "/news/" not in href:
            continue
        lower_href = href.lower()
        if "team-lists" not in lower_href and "late-mail" not in lower_href:
            continue

        full_url = urljoin("https://www.nrl.com", href)
        if full_url in seen:
            continue
        seen.add(full_url)
        urls.append(full_url)

        if max_urls is not None and len(urls) >= max_urls:
            break

    return urls


def extract_match_centre_urls(draw_html: str) -> list[str]:
    payload: dict | None = None
    raw = draw_html.strip()
    if raw.startswith("{"):
        try:
            payload = json.loads(raw)
        except Exception:
            payload = None

    if payload is None:
        soup = BeautifulSoup(draw_html, "lxml")
        payload = _load_q_data(soup.select_one("#vue-draw[q-data]"))
    if not payload:
        return []

    urls: list[str] = []
    seen: set[str] = set()
    for fixture in payload.get("fixtures", []):
        match_centre_url = _clean_text(fixture.get("matchCentreUrl"))
        if not match_centre_url:
            continue
        full_url = urljoin("https://www.nrl.com", match_centre_url)
        if full_url in seen:
            continue
        seen.add(full_url)
        urls.append(full_url)
    return urls


def discover_sitemap_index_url(session: requests.Session, timeout_seconds: int) -> str:
    try:
        robots = _fetch(session, ROBOTS_URL, timeout_seconds=timeout_seconds)
    except Exception:
        return DEFAULT_SITEMAP_URL

    for line in robots.splitlines():
        if line.lower().startswith("sitemap:"):
            sitemap_url = _clean_text(line.split(":", 1)[1])
            if sitemap_url:
                return sitemap_url
    return DEFAULT_SITEMAP_URL


def _parse_xml_with_bom(value: bytes) -> ET.Element:
    return ET.fromstring(value.decode("utf-8-sig"))


def iter_sitemap_urls(
    session: requests.Session,
    sitemap_index_url: str,
    timeout_seconds: int,
    progress_callback: ProgressCallback | None = None,
) -> list[str]:
    _emit_progress(progress_callback, f"Fetching sitemap index: {sitemap_index_url}")
    xml_bytes = session.get(sitemap_index_url, timeout=timeout_seconds).content
    root = _parse_xml_with_bom(xml_bytes)
    sitemap_urls = [loc.text for loc in root.findall(".//sm:loc", NS) if loc.text]
    _emit_progress(progress_callback, f"Discovered {len(sitemap_urls)} sitemap files from index")
    return sitemap_urls


def iter_news_urls_from_sitemap(
    session: requests.Session,
    sitemap_urls: Iterable[str],
    timeout_seconds: int,
    start_year: int | None = None,
    end_year: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> list[str]:
    sitemap_urls = list(sitemap_urls)
    collected: list[str] = []
    seen: set[str] = set()

    total = len(sitemap_urls)
    for idx, sitemap_url in enumerate(sitemap_urls, start=1):
        raw_bytes = session.get(sitemap_url, timeout=timeout_seconds).content
        if sitemap_url.endswith(".gz"):
            raw_bytes = gzip.decompress(raw_bytes)

        root = _parse_xml_with_bom(raw_bytes)
        for loc in root.findall(".//sm:loc", NS):
            url = _clean_text(loc.text)
            if "/news/" not in url:
                continue

            lower_url = url.lower()
            if "team-lists" not in lower_url and "late-mail" not in lower_url:
                continue

            year = parse_year_from_url(url)
            if start_year is not None and year is not None and year < start_year:
                continue
            if end_year is not None and year is not None and year > end_year:
                continue

            if url in seen:
                continue
            seen.add(url)
            collected.append(url)

        if idx == 1 or idx == total or idx % PROGRESS_EVERY_N_SITEMAPS == 0:
            _emit_progress(
                progress_callback,
                f"Sitemap scan progress: {idx}/{total} files, collected {len(collected)} lineup article URLs",
            )

    return collected


def _load_round_hints_from_db(
    db_path: Path | str,
    start_year: int | None,
    end_year: int | None,
    recent_only: bool,
) -> list[tuple[int, int]]:
    db_file = Path(db_path)
    if not db_file.exists():
        return []

    try:
        with sqlite3.connect(str(db_file)) as con:
            tables = {
                row[0]
                for row in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('footy_tipping_data', 'training_data', 'inference_data')"
                ).fetchall()
            }
            if not tables:
                return []

            if recent_only:
                source_tables = []
                if "inference_data" in tables:
                    source_tables.append("inference_data")
                if "footy_tipping_data" in tables:
                    source_tables.append("footy_tipping_data")
                if "training_data" in tables:
                    source_tables.append("training_data")
            else:
                source_tables = [
                    "footy_tipping_data" if "footy_tipping_data" in tables else ("training_data" if "training_data" in tables else "inference_data")
                ]

            clauses = [
                "competition_year IS NOT NULL",
                "round_id IS NOT NULL",
            ]
            params: list[int | str] = []
            if start_year is not None:
                clauses.append("competition_year >= ?")
                params.append(int(start_year))
            if end_year is not None:
                clauses.append("competition_year <= ?")
                params.append(int(end_year))
            if recent_only:
                recent_clauses = list(clauses)
                recent_clauses.append("game_state_name = 'Pre Game'")
                for source_table in source_tables:
                    query = (
                        "SELECT DISTINCT CAST(competition_year AS INTEGER), CAST(round_id AS INTEGER) "
                        f"FROM {source_table} "
                        f"WHERE {' AND '.join(recent_clauses)} "
                        "ORDER BY 1, 2"
                    )
                    rows = [(int(year), int(round_id)) for year, round_id in con.execute(query, params).fetchall()]
                    if not rows:
                        continue
                    if source_table == "footy_tipping_data":
                        min_round_by_year: dict[int, int] = {}
                        for year, round_id in rows:
                            min_round_by_year[year] = min(round_id, min_round_by_year.get(year, round_id))
                        rows = sorted((year, round_id) for year, round_id in rows if min_round_by_year.get(year) == round_id)
                    return rows
                return []

            source_table = source_tables[0]
            query = (
                "SELECT DISTINCT CAST(competition_year AS INTEGER), CAST(round_id AS INTEGER) "
                f"FROM {source_table} "
                f"WHERE {' AND '.join(clauses)} "
                "ORDER BY 1, 2"
            )
            rows = con.execute(query, params).fetchall()
            return [(int(year), int(round_id)) for year, round_id in rows]
    except Exception:
        return []


def _collect_match_centre_urls(
    session: requests.Session,
    db_path: Path | str,
    cfg: IngestionConfig,
    recent_round_hints: set[tuple[int, int]],
    progress_callback: ProgressCallback | None = None,
) -> list[str]:
    if cfg.mode == "recent":
        round_hints = set(recent_round_hints)
        round_hints.update(_load_round_hints_from_db(db_path, cfg.end_year, cfg.end_year, recent_only=True))
        round_hints = sorted(round_hints)
        if not round_hints:
            _emit_progress(progress_callback, "No recent round hints available for match-centre fetch. Skipping.")
            return []
    else:
        round_hints = _load_round_hints_from_db(db_path, cfg.start_year, cfg.end_year, recent_only=False)
        if not round_hints:
            years = range(int(cfg.start_year or dt.datetime.now(dt.timezone.utc).year), int(cfg.end_year or dt.datetime.now(dt.timezone.utc).year) + 1)
            round_hints = [(year, round_id) for year in years for round_id in range(1, MAX_DRAW_ROUNDS + 1)]

    urls: list[str] = []
    seen: set[str] = set()
    total = len(round_hints)
    for idx, (season, round_id) in enumerate(round_hints, start=1):
        draw_url = DRAW_DATA_URL_TEMPLATE.format(season=season, round_id=round_id)
        try:
            draw_payload = _fetch(session, draw_url, timeout_seconds=cfg.request_timeout_seconds)
            discovered = extract_match_centre_urls(draw_payload)
            for url in discovered:
                if url in seen:
                    continue
                seen.add(url)
                urls.append(url)
        except Exception:
            continue

        if idx == 1 or idx == total or idx % PROGRESS_EVERY_N_DRAWS == 0:
            _emit_progress(
                progress_callback,
                f"Draw scan progress: {idx}/{total} rounds, collected {len(urls)} match-centre URLs",
            )

    if cfg.mode == "recent":
        _emit_progress(progress_callback, f"Discovered {len(urls)} match-centre URLs from recent draw windows")
    else:
        _emit_progress(progress_callback, f"Discovered {len(urls)} match-centre URLs from draw backfill")
    return urls


def extract_article_title(soup: BeautifulSoup) -> str:
    header = soup.select_one("h1")
    if header:
        return _clean_text(header.get_text(" ", strip=True))
    title_tag = soup.find("title")
    if title_tag:
        return _clean_text(title_tag.get_text(" ", strip=True))
    return ""


def extract_published_at_utc(soup: BeautifulSoup) -> str | None:
    # Prefer machine-readable datetime attributes.
    for node in soup.select("time[datetime], [datetime]"):
        value = parse_iso_datetime(node.get("datetime"))
        if value:
            return value

    # Fallback: look for textual "Timestamp ..." if present.
    for node in soup.select("*"):
        text = _clean_text(node.get_text(" ", strip=True))
        if "timestamp" not in text.lower():
            continue
        iso_match = re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})", text)
        if iso_match:
            return parse_iso_datetime(iso_match.group(0))

    return None


def parse_profile_text(profile_node: Tag) -> dict | None:
    text = _clean_text(profile_node.get_text(" ", strip=True))
    match = PROFILE_PATTERN.match(text)
    if not match:
        return None

    payload = match.groupdict()
    return {
        "listed_position": _clean_text(payload.get("position")),
        "team_name": _clean_text(payload.get("team")),
        "jersey_number": int(payload.get("number")),
        "player_name": _clean_text(payload.get("player")),
    }


def normalize_group_label(raw_label: str) -> str:
    lower = _clean_text(raw_label).lower()
    if "back" in lower:
        return "backs"
    if "forward" in lower:
        return "forwards"
    if "interchange" in lower:
        return "interchange"
    if "reserve" in lower:
        return "reserves"
    return lower or "unknown"


def parse_match_header_teams(header_text: str) -> tuple[str | None, str | None]:
    clean = _clean_text(header_text)
    if not clean:
        return None, None

    home_away_match = HOME_AWAY_PATTERN.search(clean)
    if home_away_match:
        return _clean_text(home_away_match.group("home")), _clean_text(home_away_match.group("away"))

    matchup_match = MATCHUP_PATTERN.search(clean)
    if matchup_match:
        return _clean_text(matchup_match.group("home")), _clean_text(matchup_match.group("away"))

    return None, None


def _teams_match(left: str | None, right: str | None) -> bool:
    left_key = normalize_team_name(left)
    right_key = normalize_team_name(right)
    if "unknown_team" in {left_key, right_key}:
        return False
    return left_key == right_key or left_key in right_key or right_key in left_key


def _infer_position(slot_number: int, printed_number: int | None = None, squad_group: str | None = None) -> str:
    if squad_group in {"interchange", "reserves"} and printed_number in POSITION_BY_SLOT and printed_number <= 13:
        return POSITION_BY_SLOT[printed_number]
    if slot_number in POSITION_BY_SLOT:
        return POSITION_BY_SLOT[slot_number]
    return "Reserve"


def _infer_group(slot_number: int) -> str:
    if slot_number <= 7:
        return "backs"
    if slot_number <= 13:
        return "forwards"
    if slot_number <= 17:
        return "interchange"
    return "reserves"


def _clean_legacy_player_name(value: str) -> str:
    text = _clean_text(value)
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\bTop Point Scorer\b.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bToyota NRL Dream Team\b.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bLatest odds\b.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bCoach\s*:.*$", "", text, flags=re.IGNORECASE)
    text = text.strip(" ,.;:-")
    if normalize_player_name(text) in {"tba", "from", "coach", "squad"}:
        return ""
    return text


def _extract_legacy_blocks(soup: BeautifulSoup) -> list[str]:
    root = soup.select_one("article .s-cms-content") or soup.find("article")
    if root is None:
        return []

    blocks: list[str] = []
    seen: set[str] = set()
    for node in root.find_all(["div", "p", "h2", "h3", "h4", "li"], recursive=True):
        text = _clean_text(node.get_text(" ", strip=True))
        if not text or text in seen:
            continue
        seen.add(text)
        blocks.append(text)
    return blocks


def _parse_legacy_match_header(header_text: str) -> tuple[str | None, str | None]:
    clean = _clean_text(header_text)
    if not clean:
        return None, None

    match = LEGACY_MATCH_HEADER_PATTERN.search(clean)
    if not match:
        return parse_match_header_teams(clean)

    home = _clean_text(match.group("home"))
    away = _clean_text(match.group("away"))
    if not home or not away:
        return None, None

    if len(home.split()) > 6 or len(away.split()) > 6:
        return None, None
    if any(token in home.lower() for token in ("preview", "debut", "debuts", "will", "with")):
        return None, None
    if any(token in away.lower() for token in ("preview", "debut", "debuts", "will", "with")):
        return None, None

    return home, away


def _split_legacy_sections(roster_text: str) -> list[tuple[str, str]]:
    clean = _clean_text(roster_text)
    if not clean:
        return []

    clean = re.sub(r"^\(squad\)\s*:\s*", "", clean, flags=re.IGNORECASE)
    clean = re.split(r"\b(?:Late Mail|Coach)\b\s*:", clean, maxsplit=1, flags=re.IGNORECASE)[0]
    clean = re.split(r"\b(?:Latest odds|Toyota NRL Dream Team|Top Point Scorer)\b", clean, maxsplit=1, flags=re.IGNORECASE)[0]
    clean = clean.strip(" .;|")
    if not clean:
        return []

    markers = list(LEGACY_SECTION_MARKER_PATTERN.finditer(clean))
    if not markers:
        return [("starting", clean)]

    sections: list[tuple[str, str]] = []
    start_idx = 0
    current_label = "starting"
    for marker in markers:
        segment = clean[start_idx : marker.start()].strip(" .;|")
        if segment:
            sections.append((current_label, segment))

        label = marker.group("label").lower()
        current_label = "interchange" if "interchange" in label else "reserves"
        start_idx = marker.end()

    tail = clean[start_idx:].strip(" .;|")
    if tail:
        tail = re.sub(r"^\(from\)\s*", "", tail, flags=re.IGNORECASE).strip(" .;|")
        sections.append((current_label, tail))

    return sections


def _parse_legacy_numbered_roster(roster_text: str) -> list[dict]:
    parsed_entries: list[dict] = []
    for section_name, section_text in _split_legacy_sections(roster_text):
        matches = list(NUMBERED_PLAYER_PATTERN.finditer(section_text))
        if not matches:
            continue

        if section_name == "starting":
            next_slot = 1
        elif section_name == "interchange":
            next_slot = 14
        else:
            next_slot = 18

        for match in matches:
            printed_number = int(match.group("number"))
            player_name = _clean_legacy_player_name(match.group("player"))
            if not player_name:
                continue

            if section_name == "starting":
                squad_group = _infer_group(next_slot)
            elif section_name == "interchange":
                squad_group = "interchange" if next_slot <= 17 else "reserves"
            else:
                squad_group = "reserves"
            parsed_entries.append(
                {
                    "player_name": player_name,
                    "jersey_number": next_slot,
                    "listed_position": _infer_position(next_slot, printed_number=printed_number, squad_group=squad_group),
                    "squad_group": squad_group,
                }
            )
            next_slot += 1

    return parsed_entries


def _parse_legacy_name_list_roster(roster_text: str) -> list[dict]:
    parsed_entries: list[dict] = []
    for section_name, section_text in _split_legacy_sections(roster_text):
        names = [_clean_legacy_player_name(part) for part in section_text.split(",")]
        names = [name for name in names if name]
        if not names:
            continue

        if section_name == "starting":
            next_slot = 1
        elif section_name == "interchange":
            next_slot = 14
        else:
            next_slot = 18

        for player_name in names:
            if section_name == "starting":
                squad_group = _infer_group(next_slot)
            elif section_name == "interchange":
                squad_group = "interchange" if next_slot <= 17 else "reserves"
            else:
                squad_group = "reserves"
            parsed_entries.append(
                {
                    "player_name": player_name,
                    "jersey_number": next_slot,
                    "listed_position": _infer_position(next_slot, printed_number=None, squad_group=squad_group),
                    "squad_group": squad_group,
                }
            )
            next_slot += 1

    return parsed_entries


def _match_legacy_team_label(label: str, home_team: str | None, away_team: str | None) -> tuple[str | None, str | None]:
    clean_label = _clean_text(label)
    if _teams_match(clean_label, home_team):
        return _clean_text(home_team), "home"
    if _teams_match(clean_label, away_team):
        return _clean_text(away_team), "away"
    return None, None


def _parse_legacy_team_hint(
    text: str,
    home_team: str | None,
    away_team: str | None,
) -> tuple[str | None, str | None, str | None]:
    clean = _clean_text(text)
    if not clean:
        return None, None, None

    label_match = LEGACY_TEAM_LABEL_PATTERN.match(clean)
    if label_match:
        team_name, side = _match_legacy_team_label(label_match.group("label"), home_team, away_team)
        if team_name is None:
            return None, None, None
        return team_name, side, label_match.group("rest")

    numbered_match = LEGACY_NUMBERED_TEAM_PATTERN.match(clean)
    if numbered_match:
        team_name, side = _match_legacy_team_label(numbered_match.group("label"), home_team, away_team)
        if team_name is None:
            return None, None, None
        return team_name, side, numbered_match.group("rest")

    return None, None, None


def _parse_legacy_team_block(
    text: str,
    home_team: str | None,
    away_team: str | None,
) -> tuple[str | None, str | None, list[dict]] | None:
    team_name, side, roster_text = _parse_legacy_team_hint(text, home_team, away_team)
    if team_name is None or roster_text is None:
        return None

    if re.search(r"\d{1,2}\.?\s+[A-Za-z]", roster_text):
        entries = _parse_legacy_numbered_roster(roster_text)
    elif re.search(r"\bInterchange\b", roster_text, flags=re.IGNORECASE):
        entries = _parse_legacy_name_list_roster(roster_text)
    else:
        return None

    if len(entries) < 8:
        return None

    return team_name, side, entries


def _parse_legacy_continuation_block(text: str) -> list[dict]:
    clean = _clean_text(text)
    if not clean or not re.match(r"^(Interchange|Reserves?)\b", clean, flags=re.IGNORECASE):
        return []

    if re.search(r"\d{1,2}\.?\s+[A-Za-z]", clean):
        return _parse_legacy_numbered_roster(clean)
    return _parse_legacy_name_list_roster(clean)


def _parse_legacy_standalone_numbered_block(text: str) -> list[dict]:
    clean = _clean_text(text)
    if not re.match(r"^\d{1,2}\.?\s+[A-Za-z]", clean):
        return []
    entries = _parse_legacy_numbered_roster(clean)
    return entries if len(entries) >= 8 else []


def _parse_legacy_team_list_article(article_url: str, soup: BeautifulSoup) -> dict:
    article_title = extract_article_title(soup)
    published_at_utc = extract_published_at_utc(soup)
    year_from_url = parse_year_from_url(article_url)
    year = year_from_url
    if year is None and published_at_utc:
        try:
            year = dt.datetime.fromisoformat(published_at_utc.replace("Z", "+00:00")).year
        except Exception:
            year = None

    all_entries: list[dict] = []
    detected_round_id = parse_round_id(article_title, article_url)
    detected_round_name = parse_round_name(article_title, article_url)
    blocks = _extract_legacy_blocks(soup)
    current_match_index = 0
    current_home_team: str | None = None
    current_away_team: str | None = None
    pending_team: dict | None = None
    pending_team_hint: dict | None = None

    def flush_pending_team() -> None:
        nonlocal pending_team
        if pending_team is None:
            return
        all_entries.extend(pending_team["entries"])
        pending_team = None

    for block in blocks:
        header_home, header_away = _parse_legacy_match_header(block)
        if header_home and header_away:
            flush_pending_team()
            pending_team_hint = None
            current_match_index += 1
            current_home_team = header_home
            current_away_team = header_away
            if detected_round_id is None:
                detected_round_id = parse_round_id(block)
            if detected_round_name is None:
                detected_round_name = parse_round_name(block)
            continue

        if current_home_team is None or current_away_team is None:
            continue

        if pending_team_hint is not None:
            hinted_entries = _parse_legacy_standalone_numbered_block(block)
            if hinted_entries:
                flush_pending_team()
                pending_team = {
                    "entries": [
                        {
                            "match_index": current_match_index,
                            "match_home_team": _clean_text(current_home_team),
                            "match_away_team": _clean_text(current_away_team),
                            "side": pending_team_hint["side"],
                            "team_name": pending_team_hint["team_name"],
                            "opponent_team_name": pending_team_hint["opponent_team_name"],
                            **entry,
                        }
                        for entry in hinted_entries
                    ]
                }
                pending_team_hint = None
                continue

        team_block = _parse_legacy_team_block(block, current_home_team, current_away_team)
        if team_block is not None:
            flush_pending_team()
            pending_team_hint = None
            team_name, side, parsed_entries = team_block
            opponent_name = current_away_team if side == "home" else current_home_team
            pending_team = {
                "entries": [
                    {
                        "match_index": current_match_index,
                        "match_home_team": _clean_text(current_home_team),
                        "match_away_team": _clean_text(current_away_team),
                        "side": side,
                        "team_name": team_name,
                        "opponent_team_name": _clean_text(opponent_name),
                        **entry,
                    }
                    for entry in parsed_entries
                ]
            }
            continue

        hinted_team_name, hinted_side, _ = _parse_legacy_team_hint(block, current_home_team, current_away_team)
        if hinted_team_name is not None and hinted_side is not None:
            pending_team_hint = {
                "team_name": hinted_team_name,
                "side": hinted_side,
                "opponent_team_name": _clean_text(current_away_team if hinted_side == "home" else current_home_team),
            }
            continue

        continuation_entries = _parse_legacy_continuation_block(block)
        if pending_team is not None and continuation_entries:
            opponent_name = pending_team["entries"][0]["opponent_team_name"]
            team_name = pending_team["entries"][0]["team_name"]
            side = pending_team["entries"][0]["side"]
            pending_team["entries"].extend(
                {
                    "match_index": current_match_index,
                    "match_home_team": _clean_text(current_home_team),
                    "match_away_team": _clean_text(current_away_team),
                    "side": side,
                    "team_name": team_name,
                    "opponent_team_name": _clean_text(opponent_name),
                    **entry,
                }
                for entry in continuation_entries
            )
            continue

        pending_team_hint = None
        flush_pending_team()

    flush_pending_team()

    return {
        "article_url": article_url,
        "article_title": article_title,
        "article_type": "late_mail" if "late mail" in article_title.lower() else "team_list",
        "competition_year": year,
        "round_id": detected_round_id,
        "round_name": detected_round_name,
        "source_published_at_utc": published_at_utc,
        "match_id": None,
        "match_state": None,
        "entries": all_entries,
    }


def parse_team_list_article(article_url: str, article_html: str) -> dict:
    soup = BeautifulSoup(article_html, "lxml")
    article_title = extract_article_title(soup)
    published_at_utc = extract_published_at_utc(soup)
    year_from_url = parse_year_from_url(article_url)
    year = year_from_url
    if year is None and published_at_utc:
        try:
            year = dt.datetime.fromisoformat(published_at_utc.replace("Z", "+00:00")).year
        except Exception:
            year = None

    match_headers = soup.select(".match-header")
    if not match_headers:
        legacy_payload = _parse_legacy_team_list_article(article_url, soup)
        if legacy_payload.get("entries"):
            return legacy_payload

        return {
            "article_url": article_url,
            "article_title": article_title,
            "article_type": "late_mail" if "late mail" in article_title.lower() else "team_list",
            "competition_year": year,
            "round_id": parse_round_id(article_title, article_url),
            "round_name": parse_round_name(article_title, article_url),
            "source_published_at_utc": published_at_utc,
            "match_id": None,
            "match_state": None,
            "entries": [],
        }

    all_entries: list[dict] = []
    detected_round_id = parse_round_id(article_title, article_url)
    detected_round_name = parse_round_name(article_title, article_url)

    for idx, header in enumerate(match_headers):
        next_header = match_headers[idx + 1] if idx + 1 < len(match_headers) else None
        header_text = _clean_text(header.get_text(" ", strip=True))
        header_home, header_away = parse_match_header_teams(header_text)

        if detected_round_id is None:
            detected_round_id = parse_round_id(header_text)
        if detected_round_name is None:
            detected_round_name = parse_round_name(header_text)

        current_group = "unknown"
        for node in header.next_elements:
            if node is next_header:
                break
            if not isinstance(node, Tag):
                continue

            node_classes = node.get("class", [])
            if node.name == "h4" and "teamsheet-group__title" in node_classes:
                current_group = normalize_group_label(node.get_text(" ", strip=True))
                continue

            if node.name != "li":
                continue
            if "team-list" not in node_classes:
                continue

            for side in ("home", "away"):
                profile_node = node.select_one(f".team-list-profile--{side} .team-list-profile__name")
                if profile_node is None:
                    continue

                parsed = parse_profile_text(profile_node)
                if parsed is None:
                    continue

                team_name = parsed["team_name"]
                opponent_name = header_away if side == "home" else header_home
                if side == "home" and header_home is None:
                    header_home = team_name
                if side == "away" and header_away is None:
                    header_away = team_name

                all_entries.append(
                    {
                        "match_index": idx + 1,
                        "match_home_team": _clean_text(header_home),
                        "match_away_team": _clean_text(header_away),
                        "side": side,
                        "team_name": team_name,
                        "opponent_team_name": _clean_text(opponent_name),
                        "player_name": parsed["player_name"],
                        "jersey_number": parsed["jersey_number"],
                        "listed_position": parsed["listed_position"],
                        "squad_group": current_group,
                    }
                )

    return {
        "article_url": article_url,
        "article_title": article_title,
        "article_type": "late_mail" if "late mail" in article_title.lower() else "team_list",
        "competition_year": year,
        "round_id": detected_round_id,
        "round_name": detected_round_name,
        "source_published_at_utc": published_at_utc,
        "match_id": None,
        "match_state": None,
        "entries": all_entries,
    }


def _normalize_match_centre_group(number: int | None, position: str) -> str:
    lower = _clean_text(position).lower()
    if "replacement" in lower or "reserve" in lower:
        return "reserves"
    if number is None:
        return "unknown"
    if number <= 7:
        return "backs"
    if number <= 13:
        return "forwards"
    if number <= 17:
        return "interchange"
    return "reserves"


def parse_match_centre_page(match_centre_url: str, match_centre_html: str) -> dict:
    payload: dict | None = None
    article_title = ""
    raw = match_centre_html.strip()
    if raw.startswith("{"):
        try:
            payload = json.loads(raw)
        except Exception:
            payload = None

    if payload is None:
        soup = BeautifulSoup(match_centre_html, "lxml")
        payload = _load_q_data(soup.select_one("#vue-match-centre[q-data]"))
        article_title = _clean_text(soup.select_one("h1").get_text(" ", strip=True) if soup.select_one("h1") else "")
    if payload is None:
        raise ValueError("match-centre payload not found")

    match = payload.get("match", payload)
    if not article_title:
        home_title = _clean_text((match.get("homeTeam") or {}).get("nickName") or (match.get("homeTeam") or {}).get("name"))
        away_title = _clean_text((match.get("awayTeam") or {}).get("nickName") or (match.get("awayTeam") or {}).get("name"))
        article_title = _clean_text(f"{home_title} v {away_title}")
    match_state = _clean_text(match.get("matchState"))
    start_time = parse_iso_datetime(match.get("startTime"))
    updated_time = parse_iso_datetime(match.get("updated")) or start_time
    round_id = match.get("roundNumber")
    round_name = _clean_text(match.get("roundTitle")) or parse_round_name(match_centre_url)

    competition_year = None
    if start_time:
        try:
            competition_year = dt.datetime.fromisoformat(start_time.replace("Z", "+00:00")).year
        except Exception:
            competition_year = None
    if competition_year is None:
        competition_year = parse_year_from_url(match_centre_url)

    home_team = match.get("homeTeam", {}) or {}
    away_team = match.get("awayTeam", {}) or {}
    final_snapshot = match_state.lower() in FINAL_MATCH_STATES

    def _team_entries(team: dict, opponent: dict, side: str) -> list[dict]:
        rows: list[dict] = []
        for player in team.get("players", []) or []:
            number = player.get("number")
            try:
                number = int(number) if number is not None else None
            except Exception:
                number = None

            position = _clean_text(player.get("position"))
            if final_snapshot and (number is None or number > 17 or position.lower() == "replacement"):
                continue

            player_name = _clean_text(
                " ".join(
                    part for part in [player.get("firstName"), player.get("lastName")] if _clean_text(part)
                )
            )
            if not player_name:
                continue

            squad_group = _normalize_match_centre_group(number, position)
            rows.append(
                {
                    "match_index": 1,
                    "match_home_team": _clean_text(home_team.get("name") or home_team.get("nickName")),
                    "match_away_team": _clean_text(away_team.get("name") or away_team.get("nickName")),
                    "side": side,
                    "team_name": _clean_text(team.get("name") or team.get("nickName")),
                    "opponent_team_name": _clean_text(opponent.get("name") or opponent.get("nickName")),
                    "player_name": player_name,
                    "player_external_id": str(player.get("playerId") or "").strip() or None,
                    "jersey_number": number,
                    "listed_position": position or _infer_position(number or 18, squad_group=squad_group),
                    "squad_group": squad_group,
                }
            )
        return rows

    entries = _team_entries(home_team, away_team, "home") + _team_entries(away_team, home_team, "away")

    return {
        "article_url": match_centre_url,
        "article_title": article_title or _clean_text(f"{home_team.get('nickName', '')} v {away_team.get('nickName', '')}"),
        "article_type": "match_centre",
        "competition_year": competition_year,
        "round_id": int(round_id) if round_id is not None else None,
        "round_name": round_name or None,
        "source_published_at_utc": updated_time,
        "match_id": str(match.get("matchId") or "").strip() or None,
        "match_state": match_state or None,
        "entries": entries,
    }


def ensure_lineup_tables(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS lineup_article_snapshots (
            snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_url TEXT NOT NULL,
            article_title TEXT,
            article_type TEXT,
            competition_year INTEGER,
            round_id INTEGER,
            round_name TEXT,
            source_published_at_utc TEXT,
            match_id TEXT,
            match_state TEXT,
            scraped_at_utc TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            parse_status TEXT NOT NULL,
            parse_error TEXT,
            entry_count INTEGER NOT NULL DEFAULT 0
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_lineup_article_url_hash
          ON lineup_article_snapshots (article_url, content_hash);

        CREATE INDEX IF NOT EXISTS idx_lineup_article_year_round
          ON lineup_article_snapshots (competition_year, round_id);

        CREATE TABLE IF NOT EXISTS lineup_entries (
            entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id INTEGER NOT NULL,
            article_url TEXT NOT NULL,
            competition_year INTEGER,
            round_id INTEGER,
            round_name TEXT,
            article_type TEXT,
            source_published_at_utc TEXT,
            match_index INTEGER,
            match_home_team TEXT,
            match_home_team_key TEXT,
            match_away_team TEXT,
            match_away_team_key TEXT,
            side TEXT NOT NULL,
            team_name TEXT NOT NULL,
            team_key TEXT NOT NULL,
            opponent_team_name TEXT,
            opponent_team_key TEXT,
            player_name TEXT NOT NULL,
            player_key TEXT NOT NULL,
            player_external_id TEXT,
            jersey_number INTEGER,
            listed_position TEXT,
            squad_group TEXT,
            inserted_at_utc TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_lineup_entries_year_round_team
          ON lineup_entries (competition_year, round_id, team_key);

        CREATE INDEX IF NOT EXISTS idx_lineup_entries_snapshot
          ON lineup_entries (snapshot_id);

        CREATE TABLE IF NOT EXISTS lineup_ingestion_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            mode TEXT NOT NULL,
            requested_start_year INTEGER,
            requested_end_year INTEGER,
            max_articles INTEGER,
            include_sitemap_in_recent INTEGER NOT NULL DEFAULT 0,
            started_at_utc TEXT NOT NULL,
            completed_at_utc TEXT NOT NULL,
            status TEXT NOT NULL,
            url_candidates INTEGER NOT NULL DEFAULT 0,
            urls_processed INTEGER NOT NULL DEFAULT 0,
            snapshots_inserted INTEGER NOT NULL DEFAULT 0,
            snapshots_skipped_existing_hash INTEGER NOT NULL DEFAULT 0,
            articles_skipped_not_nrl INTEGER NOT NULL DEFAULT 0,
            entries_inserted INTEGER NOT NULL DEFAULT 0,
            parse_failures INTEGER NOT NULL DEFAULT 0,
            error_count INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_lineup_ingestion_runs_mode_window
          ON lineup_ingestion_runs (mode, requested_start_year, requested_end_year, completed_at_utc);
        """
    )
    _ensure_column(con, "lineup_article_snapshots", "match_id", "TEXT")
    _ensure_column(con, "lineup_article_snapshots", "match_state", "TEXT")
    _ensure_column(con, "lineup_entries", "player_external_id", "TEXT")


def _ensure_column(con: sqlite3.Connection, table_name: str, column_name: str, column_ddl: str) -> None:
    columns = {row[1] for row in con.execute(f"PRAGMA table_info({table_name})").fetchall()}
    if column_name in columns:
        return
    con.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_ddl}")


def _record_ingestion_run(
    con: sqlite3.Connection,
    cfg: IngestionConfig,
    stats: dict,
    started_at_utc: str,
    completed_at_utc: str,
) -> None:
    con.execute(
        """
        INSERT INTO lineup_ingestion_runs (
            mode,
            requested_start_year,
            requested_end_year,
            max_articles,
            include_sitemap_in_recent,
            started_at_utc,
            completed_at_utc,
            status,
            url_candidates,
            urls_processed,
            snapshots_inserted,
            snapshots_skipped_existing_hash,
            articles_skipped_not_nrl,
            entries_inserted,
            parse_failures,
            error_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            cfg.mode,
            cfg.start_year,
            cfg.end_year,
            cfg.max_articles,
            1 if cfg.include_sitemap_in_recent else 0,
            started_at_utc,
            completed_at_utc,
            "ok" if not stats.get("errors") else "completed_with_errors",
            int(stats.get("url_candidates", 0)),
            int(stats.get("urls_processed", 0)),
            int(stats.get("snapshots_inserted", 0)),
            int(stats.get("snapshots_skipped_existing_hash", 0)),
            int(stats.get("articles_skipped_not_nrl", 0)),
            int(stats.get("entries_inserted", 0)),
            int(stats.get("parse_failures", 0)),
            len(stats.get("errors", [])),
        ),
    )


def _insert_snapshot(
    con: sqlite3.Connection,
    parsed_article: dict,
    content_hash: str,
    scraped_at_utc: str,
    parse_status: str,
    parse_error: str | None = None,
) -> int | None:
    cursor = con.execute(
        """
        INSERT OR IGNORE INTO lineup_article_snapshots (
            article_url,
            article_title,
            article_type,
            competition_year,
            round_id,
            round_name,
            source_published_at_utc,
            match_id,
            match_state,
            scraped_at_utc,
            content_hash,
            parse_status,
            parse_error,
            entry_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        """,
        (
            parsed_article.get("article_url"),
            parsed_article.get("article_title"),
            parsed_article.get("article_type"),
            parsed_article.get("competition_year"),
            parsed_article.get("round_id"),
            parsed_article.get("round_name"),
            parsed_article.get("source_published_at_utc"),
            parsed_article.get("match_id"),
            parsed_article.get("match_state"),
            scraped_at_utc,
            content_hash,
            parse_status,
            parse_error,
        ),
    )
    if cursor.rowcount == 0:
        return None
    return int(cursor.lastrowid)


def _insert_entries(con: sqlite3.Connection, snapshot_id: int, parsed_article: dict, inserted_at_utc: str) -> int:
    entries = parsed_article.get("entries", [])
    if not entries:
        return 0

    rows = []
    for row in entries:
        team_name = _clean_text(row.get("team_name"))
        opponent_name = _clean_text(row.get("opponent_team_name"))
        match_home_team = _clean_text(row.get("match_home_team"))
        match_away_team = _clean_text(row.get("match_away_team"))
        player_name = _clean_text(row.get("player_name"))

        rows.append(
            (
                snapshot_id,
                parsed_article.get("article_url"),
                parsed_article.get("competition_year"),
                parsed_article.get("round_id"),
                parsed_article.get("round_name"),
                parsed_article.get("article_type"),
                parsed_article.get("source_published_at_utc"),
                row.get("match_index"),
                match_home_team,
                normalize_team_name(match_home_team),
                match_away_team,
                normalize_team_name(match_away_team),
                row.get("side"),
                team_name,
                normalize_team_name(team_name),
                opponent_name,
                normalize_team_name(opponent_name),
                player_name,
                normalize_player_name(player_name),
                row.get("player_external_id"),
                row.get("jersey_number"),
                row.get("listed_position"),
                row.get("squad_group"),
                inserted_at_utc,
            )
        )

    con.executemany(
        """
        INSERT INTO lineup_entries (
            snapshot_id,
            article_url,
            competition_year,
            round_id,
            round_name,
            article_type,
            source_published_at_utc,
            match_index,
            match_home_team,
            match_home_team_key,
            match_away_team,
            match_away_team_key,
            side,
            team_name,
            team_key,
            opponent_team_name,
            opponent_team_key,
            player_name,
            player_key,
            player_external_id,
            jersey_number,
            listed_position,
            squad_group,
            inserted_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )

    con.execute(
        "UPDATE lineup_article_snapshots SET entry_count = ? WHERE snapshot_id = ?",
        (len(rows), snapshot_id),
    )
    return len(rows)


def _repair_existing_snapshot_if_needed(
    con: sqlite3.Connection,
    parsed_article: dict,
    content_hash: str,
    scraped_at_utc: str,
) -> tuple[int | None, int]:
    cursor = con.execute(
        """
        SELECT snapshot_id, parse_status, entry_count
        FROM lineup_article_snapshots
        WHERE article_url = ? AND content_hash = ?
        ORDER BY snapshot_id DESC
        LIMIT 1
        """,
        (parsed_article.get("article_url"), content_hash),
    )
    row = cursor.fetchone()
    if row is None:
        cursor = con.execute(
            """
            SELECT snapshot_id, parse_status, entry_count
            FROM lineup_article_snapshots
            WHERE article_url = ?
            ORDER BY snapshot_id DESC
            LIMIT 1
            """,
            (parsed_article.get("article_url"),),
        )
        row = cursor.fetchone()
        if row is None:
            return None, 0

    snapshot_id = int(row[0])
    entry_count = int(row[2] or 0)
    parsed_entries = parsed_article.get("entries", [])
    if entry_count > 0 or not parsed_entries:
        return snapshot_id, 0

    con.execute("DELETE FROM lineup_entries WHERE snapshot_id = ?", (snapshot_id,))
    con.execute(
        """
        UPDATE lineup_article_snapshots
        SET article_title = ?,
            article_type = ?,
            competition_year = ?,
            round_id = ?,
            round_name = ?,
            source_published_at_utc = ?,
            match_id = ?,
            match_state = ?,
            content_hash = ?,
            scraped_at_utc = ?,
            parse_status = 'ok',
            parse_error = NULL,
            entry_count = 0
        WHERE snapshot_id = ?
        """,
        (
            parsed_article.get("article_title"),
            parsed_article.get("article_type"),
            parsed_article.get("competition_year"),
            parsed_article.get("round_id"),
            parsed_article.get("round_name"),
            parsed_article.get("source_published_at_utc"),
            parsed_article.get("match_id"),
            parsed_article.get("match_state"),
            content_hash,
            scraped_at_utc,
            snapshot_id,
        ),
    )
    inserted_count = _insert_entries(con, snapshot_id=snapshot_id, parsed_article=parsed_article, inserted_at_utc=scraped_at_utc)
    return snapshot_id, inserted_count


def _persist_parsed_snapshot(
    con: sqlite3.Connection,
    parsed: dict,
    content_hash: str,
    scraped_at_utc: str,
    progress_callback: ProgressCallback | None = None,
) -> tuple[str, int]:
    _existing_snapshot_id, repaired_entries = _repair_existing_snapshot_if_needed(
        con,
        parsed_article=parsed,
        content_hash=content_hash,
        scraped_at_utc=scraped_at_utc,
    )
    if repaired_entries > 0:
        _emit_progress(
            progress_callback,
            f"Repaired zero-entry snapshot for {parsed.get('article_url')} with {repaired_entries} lineup rows",
        )
        return "repaired", repaired_entries
    if parsed.get("article_type") == "match_centre" and not parsed.get("entries"):
        return "skipped", 0
    if _existing_snapshot_id is not None and not parsed.get("entries"):
        return "skipped", 0

    snapshot_id = _insert_snapshot(
        con,
        parsed_article=parsed,
        content_hash=content_hash,
        scraped_at_utc=scraped_at_utc,
        parse_status="ok",
        parse_error=None,
    )
    if snapshot_id is None:
        return "skipped", 0

    inserted = _insert_entries(
        con,
        snapshot_id=snapshot_id,
        parsed_article=parsed,
        inserted_at_utc=scraped_at_utc,
    )
    return "inserted", inserted


def _collect_urls(
    session: requests.Session,
    cfg: IngestionConfig,
    progress_callback: ProgressCallback | None = None,
) -> list[str]:
    _emit_progress(progress_callback, f"Fetching team-lists topic page: {TEAM_LISTS_TOPIC_URL}")
    topic_html = _fetch(session, TEAM_LISTS_TOPIC_URL, timeout_seconds=cfg.request_timeout_seconds)
    topic_urls = extract_topic_urls(topic_html, max_urls=cfg.max_articles if cfg.mode == "recent" else None)
    _emit_progress(progress_callback, f"Discovered {len(topic_urls)} candidate URLs from team-lists topic page")

    if cfg.mode == "recent" and not cfg.include_sitemap_in_recent:
        _emit_progress(progress_callback, f"Using recent mode only. Candidate URL count={min(len(topic_urls), cfg.max_articles)}")
        return topic_urls[: cfg.max_articles]

    sitemap_index = discover_sitemap_index_url(session, timeout_seconds=cfg.request_timeout_seconds)
    sitemap_urls = iter_sitemap_urls(
        session,
        sitemap_index,
        timeout_seconds=cfg.request_timeout_seconds,
        progress_callback=progress_callback,
    )
    sitemap_news_urls = iter_news_urls_from_sitemap(
        session,
        sitemap_urls,
        timeout_seconds=cfg.request_timeout_seconds,
        start_year=cfg.start_year,
        end_year=cfg.end_year,
        progress_callback=progress_callback,
    )
    _emit_progress(progress_callback, f"Discovered {len(sitemap_news_urls)} candidate URLs from sitemap archives")

    merged: list[str] = []
    seen: set[str] = set()
    for url in topic_urls + sitemap_news_urls:
        if url in seen:
            continue
        seen.add(url)
        merged.append(url)

    if cfg.max_articles > 0:
        final_urls = merged[: cfg.max_articles]
        _emit_progress(progress_callback, f"Candidate URL pool after merge/cap: {len(final_urls)}")
        return final_urls
    _emit_progress(progress_callback, f"Candidate URL pool after merge: {len(merged)}")
    return merged


def run_lineup_ingestion(
    db_path: Path | str,
    cfg: IngestionConfig,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    started_at_utc = utc_now_iso()

    stats = {
        "mode": cfg.mode,
        "start_year": cfg.start_year,
        "end_year": cfg.end_year,
        "url_candidates": 0,
        "urls_processed": 0,
        "snapshots_inserted": 0,
        "snapshots_skipped_existing_hash": 0,
        "articles_skipped_not_nrl": 0,
        "entries_inserted": 0,
        "parse_failures": 0,
        "errors": [],
    }

    with requests.Session() as session:
        session.headers.update(DEFAULT_HEADERS)
        candidate_urls = _collect_urls(session, cfg, progress_callback=progress_callback)
        recent_round_hints: set[tuple[int, int]] = set()
        match_centre_urls: list[str] = []
        stats["url_candidates"] = len(candidate_urls)
        _emit_progress(progress_callback, f"Processing {len(candidate_urls)} lineup article candidates")

        with sqlite3.connect(str(db_file)) as con:
            ensure_lineup_tables(con)

            total_urls = len(candidate_urls)
            for idx, url in enumerate(candidate_urls, start=1):
                stats["urls_processed"] += 1
                if idx == 1 or idx == total_urls or idx % PROGRESS_EVERY_N_URLS == 0:
                    pct = (100.0 * idx / total_urls) if total_urls else 100.0
                    _emit_progress(
                        progress_callback,
                        "Article progress: "
                        f"{idx}/{total_urls} ({pct:.1f}%), "
                        f"inserted={stats['snapshots_inserted']}, "
                        f"hash_skips={stats['snapshots_skipped_existing_hash']}, "
                        f"nrl_skips={stats['articles_skipped_not_nrl']}, "
                        f"parse_failures={stats['parse_failures']}",
                    )
                scraped_at_utc = utc_now_iso()
                try:
                    html = _fetch(session, url, timeout_seconds=cfg.request_timeout_seconds)
                    content_hash = hashlib.sha256(html.encode("utf-8", errors="ignore")).hexdigest()
                    parsed = parse_team_list_article(url, html)

                    if not is_nrl_mens_article(url, parsed.get("article_title", "")):
                        stats["articles_skipped_not_nrl"] += 1
                        time.sleep(cfg.request_sleep_seconds)
                        continue

                    persist_status, inserted_entries = _persist_parsed_snapshot(
                        con,
                        parsed=parsed,
                        content_hash=content_hash,
                        scraped_at_utc=scraped_at_utc,
                        progress_callback=progress_callback,
                    )
                    if persist_status == "skipped":
                        stats["snapshots_skipped_existing_hash"] += 1
                        time.sleep(cfg.request_sleep_seconds)
                        continue

                    if persist_status == "inserted":
                        stats["snapshots_inserted"] += 1
                    stats["entries_inserted"] += inserted_entries
                    if parsed.get("competition_year") is not None and parsed.get("round_id") is not None:
                        recent_round_hints.add((int(parsed["competition_year"]), int(parsed["round_id"])))

                except Exception as exc:
                    stats["parse_failures"] += 1
                    stats["errors"].append({"url": url, "error": str(exc)})
                    _emit_progress(
                        progress_callback,
                        f"Parse failure {stats['parse_failures']} at {idx}/{total_urls}: {url} ({exc})",
                    )
                    try:
                        failed_payload = {
                            "article_url": url,
                            "article_title": "",
                            "article_type": None,
                            "competition_year": parse_year_from_url(url),
                            "round_id": None,
                            "round_name": None,
                            "source_published_at_utc": None,
                            "match_id": None,
                            "match_state": None,
                            "entries": [],
                        }
                        _insert_snapshot(
                            con,
                            parsed_article=failed_payload,
                            content_hash=hashlib.sha256((url + scraped_at_utc).encode("utf-8")).hexdigest(),
                            scraped_at_utc=scraped_at_utc,
                            parse_status="error",
                            parse_error=str(exc),
                        )
                    except Exception:
                        pass

                time.sleep(cfg.request_sleep_seconds)

            match_centre_urls = _collect_match_centre_urls(
                session,
                db_path=db_file,
                cfg=cfg,
                recent_round_hints=recent_round_hints,
                progress_callback=progress_callback,
            )
            if match_centre_urls:
                total_match_centre_urls = len(match_centre_urls)
                _emit_progress(
                    progress_callback,
                    f"Processing {total_match_centre_urls} match-centre candidates",
                )
                stats["url_candidates"] += total_match_centre_urls
                for idx, url in enumerate(match_centre_urls, start=1):
                    stats["urls_processed"] += 1
                    if idx == 1 or idx == total_match_centre_urls or idx % PROGRESS_EVERY_N_URLS == 0:
                        pct = (100.0 * idx / total_match_centre_urls) if total_match_centre_urls else 100.0
                        _emit_progress(
                            progress_callback,
                            "Match-centre progress: "
                            f"{idx}/{total_match_centre_urls} ({pct:.1f}%), "
                            f"inserted={stats['snapshots_inserted']}, "
                            f"hash_skips={stats['snapshots_skipped_existing_hash']}, "
                            f"parse_failures={stats['parse_failures']}",
                        )

                    scraped_at_utc = utc_now_iso()
                    try:
                        payload_text = _fetch(
                            session,
                            f"{url.rstrip('/')}/data",
                            timeout_seconds=cfg.request_timeout_seconds,
                        )
                        content_hash = hashlib.sha256(payload_text.encode("utf-8", errors="ignore")).hexdigest()
                        parsed = parse_match_centre_page(url, payload_text)
                        persist_status, inserted_entries = _persist_parsed_snapshot(
                            con,
                            parsed=parsed,
                            content_hash=content_hash,
                            scraped_at_utc=scraped_at_utc,
                            progress_callback=progress_callback,
                        )
                        if persist_status == "skipped":
                            stats["snapshots_skipped_existing_hash"] += 1
                            time.sleep(cfg.request_sleep_seconds)
                            continue
                        if persist_status == "inserted":
                            stats["snapshots_inserted"] += 1
                        stats["entries_inserted"] += inserted_entries
                    except Exception as exc:
                        try:
                            html = _fetch(session, url, timeout_seconds=cfg.request_timeout_seconds)
                            content_hash = hashlib.sha256(html.encode("utf-8", errors="ignore")).hexdigest()
                            parsed = parse_match_centre_page(url, html)
                            persist_status, inserted_entries = _persist_parsed_snapshot(
                                con,
                                parsed=parsed,
                                content_hash=content_hash,
                                scraped_at_utc=scraped_at_utc,
                                progress_callback=progress_callback,
                            )
                            if persist_status == "skipped":
                                stats["snapshots_skipped_existing_hash"] += 1
                                time.sleep(cfg.request_sleep_seconds)
                                continue
                            if persist_status == "inserted":
                                stats["snapshots_inserted"] += 1
                            stats["entries_inserted"] += inserted_entries
                        except Exception as inner_exc:
                            stats["parse_failures"] += 1
                            stats["errors"].append({"url": url, "error": str(inner_exc)})
                            _emit_progress(
                                progress_callback,
                                f"Match-centre parse failure {stats['parse_failures']} at {idx}/{total_match_centre_urls}: {url} ({inner_exc})",
                            )
                            try:
                                failed_payload = {
                                    "article_url": url,
                                    "article_title": "",
                                    "article_type": "match_centre",
                                    "competition_year": parse_year_from_url(url),
                                    "round_id": None,
                                    "round_name": None,
                                    "source_published_at_utc": None,
                                    "match_id": None,
                                    "match_state": None,
                                    "entries": [],
                                }
                                _insert_snapshot(
                                    con,
                                    parsed_article=failed_payload,
                                    content_hash=hashlib.sha256((url + scraped_at_utc).encode("utf-8")).hexdigest(),
                                    scraped_at_utc=scraped_at_utc,
                                    parse_status="error",
                                    parse_error=str(inner_exc),
                                )
                            except Exception:
                                pass

                    time.sleep(cfg.request_sleep_seconds)

            _record_ingestion_run(
                con,
                cfg=cfg,
                stats=stats,
                started_at_utc=started_at_utc,
                completed_at_utc=utc_now_iso(),
            )
        _emit_progress(progress_callback, "Lineup ingestion run recorded in lineup_ingestion_runs")

    return stats
