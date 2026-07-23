"""Canonicalise team names from odds sources to the fixture-cache spelling.

Delegates fuzzy matching to the lineup normaliser (which already handles
"Canterbury Bulldogs", "Manly Sea Eagles", etc.), then maps the normalised key
to the canonical feed name.
"""

from __future__ import annotations

import re

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

# Betfair's NRL catalogue uses short runner labels rather than the public
# fixture spellings.  Keep these provider-specific: in particular, "Sydney"
# is too ambiguous to make a global alias, but Betfair consistently uses it
# for the Roosters while spelling South Sydney in full.
BETFAIR_SHORT_ALIASES = {
    "brisbane": "Brisbane Broncos",
    "canberra": "Canberra Raiders",
    "canterbury": "Canterbury-Bankstown Bulldogs",
    "cronulla": "Cronulla-Sutherland Sharks",
    "dolphins": "Dolphins",
    "gold coast": "Gold Coast Titans",
    "manly": "Manly-Warringah Sea Eagles",
    "melbourne": "Melbourne Storm",
    "newcastle": "Newcastle Knights",
    "north qld": "North Queensland Cowboys",
    "north queensland": "North Queensland Cowboys",
    "nz warriors": "New Zealand Warriors",
    "new zealand": "New Zealand Warriors",
    "parramatta": "Parramatta Eels",
    "penrith": "Penrith Panthers",
    "south sydney": "South Sydney Rabbitohs",
    "st george": "St. George Illawarra Dragons",
    "sydney": "Sydney Roosters",
    "wests tigers": "Wests Tigers",
}


def _plain_name(name: str | None) -> str:
    if not name:
        return ""
    clean = re.sub(r"[^a-z0-9 ]+", " ", str(name).lower())
    return re.sub(r"\s+", " ", clean).strip()


def canonical_team(name: str | None) -> str | None:
    if not name:
        return None
    return KEY_TO_CANONICAL.get(normalize_team_name(name))


def canonical_betfair_team(
    name: str | None,
    fixture_teams: tuple[str, str] | None = None,
) -> str | None:
    """Resolve a Betfair event/runner label, optionally in fixture context."""
    resolved = canonical_team(name)
    if resolved and (fixture_teams is None or resolved in fixture_teams):
        return resolved

    short = BETFAIR_SHORT_ALIASES.get(_plain_name(name))
    if short and (fixture_teams is None or short in fixture_teams):
        return short
    return None
