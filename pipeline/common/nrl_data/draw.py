"""Fixture ingestion from the nrl.com draw JSON endpoint.

Produces rows in the exact `feed_cache_fixtures` schema so the R data prep
consumes them unchanged. Two timestamp columns are reproduced from the feed's
conventions:

- `start_time_utc`: true UTC kickoff epoch.
- `start_time`: venue-local wall-clock kickoff serialised as-if-UTC (the feed
  stored local time this way; R derives start_hour/game_day/day-night from it).

`game_id` follows nrl.com's own scheme (verified identical to the feed across
all cached seasons, and equal to the match centre `matchId`):
`int(f"{year}111{round_id:02d}{game_number}0")` with game_number = kickoff
order within the round.
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from .teams import canonical_team_name
from .web import DRAW_DATA_URL_TEMPLATE, FetchConfig, fetch_text, parse_json_or_q_data

MAX_DRAW_ROUNDS = 35
COMPETITION_ID = 111

MATCH_STATE_TO_GAME_STATE = {
    "fulltime": "Final",
    "postgame": "Final",
    "upcoming": "Pre Game",
    "pregame": "Pre Game",
}

CITY_TIMEZONES = {
    "sydney": "Australia/Sydney",
    "newcastle": "Australia/Sydney",
    "wollongong": "Australia/Sydney",
    "canberra": "Australia/Sydney",
    "gosford": "Australia/Sydney",
    "central coast": "Australia/Sydney",
    "kogarah": "Australia/Sydney",
    "penrith": "Australia/Sydney",
    "bathurst": "Australia/Sydney",
    "dubbo": "Australia/Sydney",
    "mudgee": "Australia/Sydney",
    "tamworth": "Australia/Sydney",
    "albury": "Australia/Sydney",
    "wagga wagga": "Australia/Sydney",
    "coffs harbour": "Australia/Sydney",
    "brisbane": "Australia/Brisbane",
    "redcliffe": "Australia/Brisbane",
    "gold coast": "Australia/Brisbane",
    "townsville": "Australia/Brisbane",
    "cairns": "Australia/Brisbane",
    "mackay": "Australia/Brisbane",
    "rockhampton": "Australia/Brisbane",
    "gladstone": "Australia/Brisbane",
    "bundaberg": "Australia/Brisbane",
    "toowoomba": "Australia/Brisbane",
    "sunshine coast": "Australia/Brisbane",
    "melbourne": "Australia/Melbourne",
    "adelaide": "Australia/Adelaide",
    "perth": "Australia/Perth",
    "darwin": "Australia/Darwin",
    "auckland": "Pacific/Auckland",
    "hamilton": "Pacific/Auckland",
    "wellington": "Pacific/Auckland",
    "napier": "Pacific/Auckland",
    "new plymouth": "Pacific/Auckland",
    "christchurch": "Pacific/Auckland",
    "dunedin": "Pacific/Auckland",
    "las vegas": "America/Los_Angeles",
}
DEFAULT_TIMEZONE = "Australia/Sydney"

FIXTURE_CACHE_COLUMNS = [
    "game_id",
    "round_id",
    "round_name",
    "game_number",
    "game_state_name",
    "start_time",
    "start_time_utc",
    "venue_name",
    "city",
    "crowd",
    "broadcast_channel1",
    "broadcast_channel2",
    "broadcast_channel3",
    "team_home",
    "team_away",
    "team_final_score_home",
    "team_head_to_head_odds_home",
    "team_line_odds_home",
    "team_line_amount_home",
    "team_final_score_away",
    "team_head_to_head_odds_away",
    "team_line_odds_away",
    "team_line_amount_away",
    "competition_year",
]


def build_game_id(year: int, round_id: int, game_number: int) -> int:
    return int(f"{year}111{round_id:02d}{game_number}0")


def load_venue_timezones(csv_path: str | Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    path = Path(csv_path)
    if not path.exists():
        return lookup
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            venue = (row.get("venue_name") or "").strip()
            timezone = (row.get("timezone") or "").strip()
            if venue and timezone:
                lookup[venue.lower()] = timezone
    return lookup


def venue_timezone(
    venue_name: str | None,
    city: str | None,
    venue_tz_lookup: dict[str, str],
) -> str:
    if venue_name:
        tz = venue_tz_lookup.get(" ".join(str(venue_name).lower().split()))
        if tz:
            return tz
    if city:
        tz = CITY_TIMEZONES.get(" ".join(str(city).lower().split()))
        if tz:
            return tz
    return DEFAULT_TIMEZONE


def parse_kickoff_utc(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    raw = str(value).strip()
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        parsed = dt.datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def local_wallclock_epoch(kickoff_utc: dt.datetime, timezone_name: str) -> float:
    """Venue-local wall clock serialised as-if-UTC (the feed's start_time)."""
    local = kickoff_utc.astimezone(ZoneInfo(timezone_name))
    return local.replace(tzinfo=dt.timezone.utc).timestamp()


def fetch_round_draw(
    session: requests.Session,
    config: FetchConfig,
    season: int,
    round_id: int,
) -> dict | None:
    url = DRAW_DATA_URL_TEMPLATE.format(season=season, round_id=round_id)
    raw = fetch_text(session, url, config)
    return parse_json_or_q_data(raw, "vue-draw")


def _team_name(team_payload: dict | None) -> str | None:
    if not isinstance(team_payload, dict):
        return None
    return canonical_team_name(team_payload.get("teamId"), team_payload.get("nickName"))


def _score(team_payload: dict | None) -> float:
    if isinstance(team_payload, dict) and team_payload.get("score") is not None:
        try:
            return float(team_payload["score"])
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def draw_to_fixture_rows(
    payload: dict,
    season: int,
    venue_tz_lookup: dict[str, str],
) -> list[dict]:
    """Fixture rows in feed_cache_fixtures schema.

    Each row carries an extra non-schema key `match_centre_url` for downstream
    match-centre ingestion; cache writers select FIXTURE_CACHE_COLUMNS only.
    """
    round_id = payload.get("selectedRoundId")
    if round_id is None:
        return []
    round_id = int(round_id)

    matches = []
    for fixture in payload.get("fixtures", []):
        if fixture.get("type") not in (None, "Match"):
            continue
        clock = fixture.get("clock") or {}
        kickoff_utc = parse_kickoff_utc(clock.get("kickOffTimeLong"))
        matches.append((kickoff_utc, fixture))

    # game_number is kickoff order within the round; the payload is already
    # chronological, so the sort is a stable no-op safeguard.
    matches.sort(key=lambda item: (item[0] is None, item[0]))

    rows: list[dict] = []
    for game_number, (kickoff_utc, fixture) in enumerate(matches, start=1):
        team_home = _team_name(fixture.get("homeTeam"))
        team_away = _team_name(fixture.get("awayTeam"))
        if not team_home or not team_away:
            continue

        venue_name = fixture.get("venue")
        city = fixture.get("venueCity")
        state_raw = str(fixture.get("matchState") or "").strip()
        game_state = MATCH_STATE_TO_GAME_STATE.get(state_raw.lower(), state_raw)

        start_time = None
        start_time_utc = None
        if kickoff_utc is not None:
            start_time_utc = kickoff_utc.timestamp()
            tz_name = venue_timezone(venue_name, city, venue_tz_lookup)
            start_time = local_wallclock_epoch(kickoff_utc, tz_name)

        rows.append(
            {
                "game_id": float(build_game_id(season, round_id, game_number)),
                "round_id": float(round_id),
                "round_name": fixture.get("roundTitle") or f"Round {round_id}",
                "game_number": float(game_number),
                "game_state_name": game_state,
                "start_time": start_time,
                "start_time_utc": start_time_utc,
                "venue_name": venue_name,
                "city": city,
                "crowd": None,
                "broadcast_channel1": None,
                "broadcast_channel2": None,
                "broadcast_channel3": None,
                "team_home": team_home,
                "team_away": team_away,
                "team_final_score_home": _score(fixture.get("homeTeam")),
                "team_final_score_away": _score(fixture.get("awayTeam")),
                "team_head_to_head_odds_home": None,
                "team_head_to_head_odds_away": None,
                "team_line_odds_home": None,
                "team_line_odds_away": None,
                "team_line_amount_home": None,
                "team_line_amount_away": None,
                "competition_year": season,
                "match_centre_url": fixture.get("matchCentreUrl"),
            }
        )
    return rows


def draw_to_bye_rows(payload: dict, season: int) -> list[dict]:
    round_id = payload.get("selectedRoundId")
    if round_id is None:
        return []
    rows = []
    for bye in payload.get("byes", []):
        team = canonical_team_name(None, bye.get("teamNickName"))
        if team:
            rows.append(
                {
                    "competition_year": season,
                    "round_id": int(round_id),
                    "team": team,
                }
            )
    return rows


def fetch_season_draw(
    session: requests.Session,
    config: FetchConfig,
    season: int,
    venue_tz_lookup: dict[str, str],
    max_rounds: int = MAX_DRAW_ROUNDS,
) -> tuple[list[dict], list[dict]]:
    """All fixture + bye rows for a season, iterating rounds until exhausted."""
    fixture_rows: list[dict] = []
    bye_rows: list[dict] = []
    consecutive_empty = 0
    for round_id in range(1, max_rounds + 1):
        try:
            payload = fetch_round_draw(session, config, season, round_id)
        except requests.HTTPError:
            payload = None
        if not payload:
            consecutive_empty += 1
            if consecutive_empty >= 2:
                break
            continue
        # The endpoint clamps out-of-range rounds to the nearest valid round;
        # skip payloads that answer for a different round than requested.
        answered_round = payload.get("selectedRoundId")
        if answered_round is not None and int(answered_round) != round_id:
            break
        round_fixtures = draw_to_fixture_rows(payload, season, venue_tz_lookup)
        if not round_fixtures:
            consecutive_empty += 1
            if consecutive_empty >= 2:
                break
            continue
        consecutive_empty = 0
        fixture_rows.extend(round_fixtures)
        bye_rows.extend(draw_to_bye_rows(payload, season))
    return fixture_rows, bye_rows
