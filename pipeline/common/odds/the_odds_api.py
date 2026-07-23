"""The Odds API v4 client for CI-safe live NRL markets.

The hosted workflow cannot rely on Betfair being reachable from the runner's
region.  This provider fetches the Australian NRL market and persists one real
bookmaker's paired prices per fixture.  It never averages prices from different
books into a synthetic quote.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import statistics
from pathlib import Path

import requests

from .live import parse_utc_datetime, persist_live_snapshots
from .team_names import canonical_team
from .validity import finite_number, valid_price_pair

SPORT_KEY = "rugbyleague_nrl"
API_URL_DEFAULT = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds"
REQUESTED_MARKETS = ("h2h", "spreads", "totals")
PREFERRED_BOOKMAKER = "betfair_ex_au"
MAX_BOOK_AGE = dt.timedelta(hours=6)
MAX_CLOCK_SKEW = dt.timedelta(minutes=5)


class OddsApiError(RuntimeError):
    pass


class OddsApiClient:
    def __init__(
        self,
        api_key: str | None = None,
        api_url: str | None = None,
        timeout_seconds: int = 30,
    ) -> None:
        self.api_key = api_key or os.environ.get("ODDS_API_KEY", "")
        self.api_url = api_url or API_URL_DEFAULT
        self.timeout_seconds = timeout_seconds

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def fetch_odds(self) -> tuple[list[dict], dict]:
        response = requests.get(
            self.api_url,
            params={
                "apiKey": self.api_key,
                "regions": "au",
                "markets": ",".join(REQUESTED_MARKETS),
                "oddsFormat": "decimal",
                "dateFormat": "iso",
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise OddsApiError("The Odds API response was not an event list")
        quota = {
            "requests_remaining": response.headers.get("x-requests-remaining"),
            "requests_used": response.headers.get("x-requests-used"),
            "requests_last": response.headers.get("x-requests-last"),
        }
        return payload, {key: value for key, value in quota.items() if value is not None}


def _safe_error(exc: Exception, api_key: str) -> str:
    """Render provider failures without leaking the query-string credential."""
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return f"HTTP {exc.response.status_code} from The Odds API"
    message = str(exc)
    if api_key:
        message = message.replace(api_key, "<redacted>")
    return re.sub(r"(?i)(apiKey=)[^&\s]+", r"\1<redacted>", message)


def _markets_by_key(bookmaker: dict) -> dict[str, dict]:
    return {
        str(market.get("key")): market
        for market in bookmaker.get("markets") or []
        if market.get("key")
    }


def _team_outcomes(
    market: dict | None,
    home: str,
    away: str,
) -> dict[str, dict]:
    outcomes: dict[str, dict] = {}
    for outcome in (market or {}).get("outcomes") or []:
        team = canonical_team(outcome.get("name"))
        if team in {home, away}:
            outcomes[team] = outcome
    return outcomes


def _h2h_pair(bookmaker: dict, home: str, away: str) -> tuple[float, float] | None:
    outcomes = _team_outcomes(_markets_by_key(bookmaker).get("h2h"), home, away)
    home_price = outcomes.get(home, {}).get("price")
    away_price = outcomes.get(away, {}).get("price")
    if not valid_price_pair(home_price, away_price):
        return None
    return float(home_price), float(away_price)


def _devig_home_probability(pair: tuple[float, float]) -> float:
    home_price, away_price = pair
    home_raw = 1.0 / home_price
    away_raw = 1.0 / away_price
    return home_raw / (home_raw + away_raw)


def _freshness(bookmaker: dict) -> float:
    parsed = parse_utc_datetime(bookmaker.get("last_update"))
    return parsed.timestamp() if parsed is not None else 0.0


def _fresh_bookmaker(bookmaker: dict, now: dt.datetime) -> bool:
    updated = parse_utc_datetime(bookmaker.get("last_update"))
    return bool(
        updated is not None
        and now - MAX_BOOK_AGE <= updated <= now + MAX_CLOCK_SKEW
    )


def select_bookmaker(
    event: dict,
    home: str,
    away: str,
    now: dt.datetime | None = None,
) -> tuple[dict, tuple[float, float], list[tuple[dict, tuple[float, float]]]] | None:
    """Pick one complete real book deterministically.

    Betfair Exchange AU wins when it has a paired H2H market.  Otherwise the
    chosen bookmaker is closest to the median de-vigged home probability;
    equally close candidates prefer the freshest update, then bookmaker key.
    """
    observed_at = now or dt.datetime.now(dt.timezone.utc)
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=dt.timezone.utc)
    observed_at = observed_at.astimezone(dt.timezone.utc)

    candidates: list[tuple[dict, tuple[float, float]]] = []
    for bookmaker in event.get("bookmakers") or []:
        if not bookmaker.get("key") or not _fresh_bookmaker(bookmaker, observed_at):
            continue
        pair = _h2h_pair(bookmaker, home, away)
        if pair is not None:
            candidates.append((bookmaker, pair))
    if not candidates:
        return None

    complete_candidates = [
        candidate
        for candidate in candidates
        if _spread_values(candidate[0], home, away)
        and _totals_values(candidate[0])
    ]
    preferred = [
        candidate
        for candidate in complete_candidates
        if candidate[0].get("key") == PREFERRED_BOOKMAKER
    ]
    if preferred:
        bookmaker, pair = min(
            preferred,
            key=lambda candidate: (
                -_freshness(candidate[0]),
                str(candidate[0].get("key") or ""),
            ),
        )
        return bookmaker, pair, candidates

    median_probability = statistics.median(
        _devig_home_probability(pair) for _, pair in candidates
    )
    selection_pool = complete_candidates or candidates
    bookmaker, pair = min(
        selection_pool,
        key=lambda candidate: (
            abs(_devig_home_probability(candidate[1]) - median_probability),
            -_freshness(candidate[0]),
            str(candidate[0].get("key") or ""),
        ),
    )
    return bookmaker, pair, candidates


def _spread_values(
    bookmaker: dict,
    home: str,
    away: str,
) -> dict:
    outcomes = _team_outcomes(_markets_by_key(bookmaker).get("spreads"), home, away)
    home_outcome = outcomes.get(home) or {}
    away_outcome = outcomes.get(away) or {}
    home_point = home_outcome.get("point")
    away_point = away_outcome.get("point")
    home_price = home_outcome.get("price")
    away_price = away_outcome.get("price")
    if (
        not finite_number(home_point)
        or not finite_number(away_point)
        or abs(float(home_point) + float(away_point)) > 1e-6
        or not valid_price_pair(home_price, away_price)
    ):
        return {}
    return {
        "line_amount_home": float(home_point),
        "line_odds_home": float(home_price),
        "line_odds_away": float(away_price),
    }


def _totals_values(bookmaker: dict) -> dict:
    market = _markets_by_key(bookmaker).get("totals") or {}
    by_side: dict[str, list[dict]] = {"over": [], "under": []}
    for outcome in market.get("outcomes") or []:
        side = str(outcome.get("name") or "").strip().lower()
        if side in by_side:
            by_side[side].append(outcome)

    pairs: list[tuple[float, float, float]] = []
    for over in by_side["over"]:
        for under in by_side["under"]:
            if (
                not finite_number(over.get("point"))
                or not finite_number(under.get("point"))
                or float(over["point"]) <= 0
                or abs(float(over["point"]) - float(under["point"])) > 1e-6
                or not valid_price_pair(over.get("price"), under.get("price"))
            ):
                continue
            pairs.append(
                (
                    float(over["point"]),
                    float(over["price"]),
                    float(under["price"]),
                )
            )
    if not pairs:
        return {}
    total, over_price, under_price = min(
        pairs,
        key=lambda pair: (abs(pair[1] - pair[2]), pair[0]),
    )
    return {
        "total_line": total,
        "total_over_odds": over_price,
        "total_under_odds": under_price,
    }


def parse_events(
    events: list[dict],
    quota: dict | None = None,
    now: dt.datetime | None = None,
) -> list[dict]:
    observed_at = now or dt.datetime.now(dt.timezone.utc)
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=dt.timezone.utc)
    observed_at = observed_at.astimezone(dt.timezone.utc)

    snapshots: list[dict] = []
    for event in events:
        if event.get("sport_key") not in {None, SPORT_KEY}:
            continue
        home = canonical_team(event.get("home_team"))
        away = canonical_team(event.get("away_team"))
        commence = parse_utc_datetime(event.get("commence_time"))
        if not home or not away or home == away or commence is None:
            continue

        selected = select_bookmaker(event, home, away, now=observed_at)
        if selected is None:
            continue
        bookmaker, h2h, candidates = selected
        bookmaker_updated = parse_utc_datetime(bookmaker.get("last_update"))
        values = {
            "h2h_odds_home": h2h[0],
            "h2h_odds_away": h2h[1],
            "h2h_odds_home_min": min(pair[0] for _, pair in candidates),
            "h2h_odds_home_max": max(pair[0] for _, pair in candidates),
            "h2h_odds_away_min": min(pair[1] for _, pair in candidates),
            "h2h_odds_away_max": max(pair[1] for _, pair in candidates),
        }
        values.update(_spread_values(bookmaker, home, away))
        values.update(_totals_values(bookmaker))

        raw_meta = {
            "event_id": event.get("id"),
            "bookmaker_key": bookmaker.get("key"),
            "bookmaker_title": bookmaker.get("title"),
            "bookmaker_last_update": bookmaker.get("last_update"),
            "consensus_book_count": len(candidates),
        }
        raw_meta.update(quota or {})
        snapshots.append(
            {
                "home": home,
                "away": away,
                "commence_time": commence,
                # Freshness follows the quote source timestamp rather than
                # the later fetch time. Otherwise a six-hour-old bookmaker
                # update could remain "fresh" for another six hours locally.
                "snapshot_time_utc": bookmaker_updated.replace(
                    microsecond=0
                ).isoformat(),
                "values": values,
                "raw_meta": {
                    key: value for key, value in raw_meta.items() if value is not None
                },
            }
        )
    return snapshots


def snapshot_live_odds(
    db_path: str | Path,
    client: OddsApiClient | None = None,
    now: dt.datetime | None = None,
) -> dict:
    client = client or OddsApiClient()
    if not client.configured:
        print("[odds] The Odds API key not configured; skipping primary provider.")
        return {"status": "skipped", "reason": "not_configured", "provider": "the_odds_api"}

    try:
        events, quota = client.fetch_odds()
        observed_at = now or dt.datetime.now(dt.timezone.utc)
        if observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=dt.timezone.utc)
        observed_at = observed_at.astimezone(dt.timezone.utc).replace(microsecond=0)
        snapshots = parse_events(events, quota, now=observed_at)
    except Exception as exc:
        reason = _safe_error(exc, getattr(client, "api_key", ""))
        print(f"[odds] The Odds API snapshot failed softly: {reason}")
        return {
            "status": "failed",
            "reason": reason,
            "provider": "the_odds_api",
        }

    if not snapshots:
        print("[odds] The Odds API returned no complete NRL H2H markets.")
        return {"status": "no_markets", "provider": "the_odds_api"}

    summary = persist_live_snapshots(
        db_path,
        source="the_odds_api",
        snapshots=snapshots,
        observed_at=observed_at.isoformat(),
    )
    status = "completed" if summary["games_updated"] else "no_matches"
    quota_note = ", ".join(f"{key}={value}" for key, value in sorted(quota.items()))
    suffix = f" ({quota_note})" if quota_note else ""
    print(
        "[odds] The Odds API snapshot: "
        f"{summary['games_updated']}/{summary['fixture_count']} upcoming games matched; "
        f"H2H={summary['h2h_games']}, line={summary['line_games']}, "
        f"totals={summary['totals_games']}{suffix}."
    )
    return {"status": status, "provider": "the_odds_api", **summary, "quota": quota}
