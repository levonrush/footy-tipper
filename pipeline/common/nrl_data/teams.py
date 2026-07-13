"""Canonical team naming for nrl.com payloads.

The feed cache stores full names ("Canterbury-Bankstown Bulldogs"); nrl.com
draw JSON exposes teamId + nickname only. Both mappings below target the
canonical feed spelling so new rows join seamlessly with 2008-2026 history.
"""

from __future__ import annotations

TEAM_ID_TO_NAME = {
    500011: "Brisbane Broncos",
    500013: "Canberra Raiders",
    500010: "Canterbury-Bankstown Bulldogs",
    500028: "Cronulla-Sutherland Sharks",
    500723: "Dolphins",
    500004: "Gold Coast Titans",
    500002: "Manly-Warringah Sea Eagles",
    500021: "Melbourne Storm",
    500003: "Newcastle Knights",
    500032: "New Zealand Warriors",
    500012: "North Queensland Cowboys",
    500031: "Parramatta Eels",
    500014: "Penrith Panthers",
    500005: "South Sydney Rabbitohs",
    500022: "St. George Illawarra Dragons",
    500001: "Sydney Roosters",
    500023: "Wests Tigers",
}

NICKNAME_TO_NAME = {
    "broncos": "Brisbane Broncos",
    "raiders": "Canberra Raiders",
    "bulldogs": "Canterbury-Bankstown Bulldogs",
    "sharks": "Cronulla-Sutherland Sharks",
    "dolphins": "Dolphins",
    "titans": "Gold Coast Titans",
    "sea eagles": "Manly-Warringah Sea Eagles",
    "storm": "Melbourne Storm",
    "knights": "Newcastle Knights",
    "warriors": "New Zealand Warriors",
    "cowboys": "North Queensland Cowboys",
    "eels": "Parramatta Eels",
    "panthers": "Penrith Panthers",
    "rabbitohs": "South Sydney Rabbitohs",
    "dragons": "St. George Illawarra Dragons",
    "roosters": "Sydney Roosters",
    "wests tigers": "Wests Tigers",
    "tigers": "Wests Tigers",
}


def canonical_team_name(team_id: int | None, nickname: str | None) -> str | None:
    if team_id is not None:
        name = TEAM_ID_TO_NAME.get(int(team_id))
        if name:
            return name
    if nickname:
        return NICKNAME_TO_NAME.get(" ".join(str(nickname).lower().split()))
    return None
