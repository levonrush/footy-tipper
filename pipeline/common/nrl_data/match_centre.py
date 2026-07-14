"""Match centre ingestion: per-match team stats, player stats, context, officials.

The match centre page embeds JSON in `#vue-match-centre[q-data]`. Its
`match.matchId` equals the feed `game_id` (verified across seasons), which
makes it the authoritative join key. Team stats are stored long (stat titles
vary slightly by era); player stats are stored wide (the ~55 stat keys are
identical from 2012 through 2026).
"""

from __future__ import annotations

import re

import requests

from .web import FetchConfig, fetch_text, parse_json_or_q_data

_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def to_snake(name: str) -> str:
    return _CAMEL_RE.sub("_", str(name)).lower()


def slug_stat_title(title: str) -> str:
    slug = _NON_ALNUM_RE.sub("_", str(title).strip().lower().replace("%", " pct "))
    return slug.strip("_")


def coerce_stat_value(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    if ":" in text:
        # "mm:ss" durations (e.g. Time In Possession) -> seconds
        parts = text.split(":")
        try:
            return float(int(parts[0]) * 60 + int(parts[1]))
        except (ValueError, IndexError):
            return None
    try:
        return float(text.rstrip("%s"))
    except ValueError:
        return None


def fetch_match_centre(
    session: requests.Session,
    config: FetchConfig,
    url: str,
) -> dict | None:
    raw = fetch_text(session, url, config)
    return parse_json_or_q_data(raw, "vue-match-centre")


def _player_name(roster_entry: dict) -> str:
    first = str(roster_entry.get("firstName") or "").strip()
    last = str(roster_entry.get("lastName") or "").strip()
    return " ".join(part for part in (first, last) if part)


def parse_match_centre(payload: dict, source_url: str = "") -> dict | None:
    match = payload.get("match") if isinstance(payload, dict) else None
    if not isinstance(match, dict):
        return None

    game_id = match.get("matchId")
    try:
        game_id = int(game_id)
    except (TypeError, ValueError):
        return None

    stats = match.get("stats") or {}

    team_stats: list[dict] = []
    for group in stats.get("groups") or []:
        for stat in group.get("stats") or []:
            stat_name = slug_stat_title(stat.get("title") or "")
            if not stat_name:
                continue
            for side, key in (("home", "homeValue"), ("away", "awayValue")):
                holder = stat.get(key) or {}
                value = coerce_stat_value(
                    holder.get("value") if isinstance(holder, dict) else holder
                )
                if value is not None:
                    team_stats.append(
                        {"side": side, "stat_name": stat_name, "value": value}
                    )

    rosters: dict[str, dict[int, dict]] = {}
    for side, key in (("home", "homeTeam"), ("away", "awayTeam")):
        side_roster = {}
        for entry in (match.get(key) or {}).get("players") or []:
            player_id = entry.get("playerId")
            if player_id is not None:
                side_roster[int(player_id)] = entry
        rosters[side] = side_roster

    player_stats: list[dict] = []
    players_payload = stats.get("players") or {}
    for side, key in (("home", "homeTeam"), ("away", "awayTeam")):
        for stat_row in players_payload.get(key) or []:
            player_id = stat_row.get("playerId")
            if player_id is None:
                continue
            player_id = int(player_id)
            roster_entry = rosters[side].get(player_id, {})
            row = {
                "side": side,
                "player_id": player_id,
                "player_name": _player_name(roster_entry) or None,
                "jersey_number": roster_entry.get("number"),
                "position": roster_entry.get("position"),
            }
            for stat_key, value in stat_row.items():
                if stat_key == "playerId":
                    continue
                row[to_snake(stat_key)] = coerce_stat_value(value)
            player_stats.append(row)

    officials = []
    for official in match.get("officials") or []:
        name = " ".join(
            part
            for part in (
                str(official.get("firstName") or "").strip(),
                str(official.get("lastName") or "").strip(),
            )
            if part
        )
        if name:
            officials.append(
                {
                    "role": official.get("position") or "Unknown",
                    "official_name": name,
                    "profile_id": official.get("profileId"),
                }
            )

    home_team = (match.get("homeTeam") or {}).get("nickName")
    away_team = (match.get("awayTeam") or {}).get("nickName")

    return {
        "game_id": game_id,
        "match_state": match.get("matchState"),
        "home_nickname": home_team,
        "away_nickname": away_team,
        "team_stats": team_stats,
        "player_stats": player_stats,
        "context": {
            "weather_label": match.get("weather"),
            "ground_condition": match.get("groundConditions"),
            "attendance": match.get("attendance"),
        },
        "officials": officials,
        "source_url": source_url,
    }
