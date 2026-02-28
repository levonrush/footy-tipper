from __future__ import annotations

import re


TEAM_ALIASES = {
    "broncos": ("brisbane broncos", "broncos"),
    "raiders": ("canberra raiders", "raiders"),
    "bulldogs": ("canterbury bankstown bulldogs", "canterbury bulldogs", "bulldogs"),
    "sharks": ("cronulla sutherland sharks", "cronulla sharks", "sharks"),
    "dolphins": ("dolphins", "the dolphins"),
    "titans": ("gold coast titans", "titans"),
    "sea_eagles": ("manly warringah sea eagles", "manly sea eagles", "sea eagles", "manly"),
    "storm": ("melbourne storm", "storm"),
    "knights": ("newcastle knights", "knights"),
    "cowboys": ("north queensland cowboys", "cowboys"),
    "eels": ("parramatta eels", "eels"),
    "panthers": ("penrith panthers", "panthers"),
    "rabbitohs": ("south sydney rabbitohs", "rabbitohs", "south sydney"),
    "dragons": ("st george illawarra dragons", "st george dragons", "dragons"),
    "roosters": ("sydney roosters", "roosters"),
    "warriors": ("new zealand warriors", "warriors"),
    "tigers": ("wests tigers", "tigers"),
}


def _clean_text(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def normalize_team_name(team_name: str | None) -> str:
    clean = re.sub(r"[^a-z0-9 ]+", " ", _clean_text(team_name).lower()).strip()
    clean = re.sub(r"\s+", " ", clean)
    if not clean:
        return "unknown_team"

    for key, aliases in TEAM_ALIASES.items():
        if clean == key:
            return key
        for alias in aliases:
            alias_clean = re.sub(r"[^a-z0-9 ]+", " ", alias.lower()).strip()
            if alias_clean and alias_clean in clean:
                return key

    return clean.replace(" ", "_")


def normalize_player_name(player_name: str | None) -> str:
    clean = re.sub(r"[^a-z0-9 ]+", " ", _clean_text(player_name).lower()).strip()
    clean = re.sub(r"\s+", " ", clean)
    return clean.replace(" ", "_") if clean else "unknown_player"

