"""Canonicalise team names from odds sources to the fixture-cache spelling.

Delegates fuzzy matching to the lineup normaliser (which already handles
"Canterbury Bulldogs", "Manly Sea Eagles", etc.), then maps the normalised key
to the canonical feed name.
"""

from __future__ import annotations

from ..lineups.normalization import normalize_team_name

KEY_TO_CANONICAL = {
    "broncos": "Brisbane Broncos",
    "raiders": "Canberra Raiders",
    "bulldogs": "Canterbury-Bankstown Bulldogs",
    "sharks": "Cronulla-Sutherland Sharks",
    "dolphins": "Dolphins",
    "titans": "Gold Coast Titans",
    "sea_eagles": "Manly-Warringah Sea Eagles",
    "storm": "Melbourne Storm",
    "knights": "Newcastle Knights",
    "warriors": "New Zealand Warriors",
    "cowboys": "North Queensland Cowboys",
    "eels": "Parramatta Eels",
    "panthers": "Penrith Panthers",
    "rabbitohs": "South Sydney Rabbitohs",
    "dragons": "St. George Illawarra Dragons",
    "roosters": "Sydney Roosters",
    "tigers": "Wests Tigers",
}


def canonical_team(name: str | None) -> str | None:
    if not name:
        return None
    return KEY_TO_CANONICAL.get(normalize_team_name(name))
