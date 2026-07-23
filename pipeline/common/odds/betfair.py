"""Betfair Exchange client for live pre-game NRL odds.

Uses the interactive login + JSON-RPC betting API directly (no
betfairlightweight dependency). Designed for the free delayed app key: prices
are 1-180 s stale, which is immaterial ~6-24 h before kickoff.

Everything fails soft: any auth/network/parse problem leaves fixture odds
untouched, and the existing odds_missing=1 path downstream handles the gap.

Env vars (secrets.env): BETFAIR_APP_KEY, BETFAIR_USERNAME, BETFAIR_PASSWORD,
optional BETFAIR_IDENTITY_URL for jurisdiction overrides.
"""

from __future__ import annotations

import datetime as dt
import os
import re
from pathlib import Path

import requests

from .live import persist_live_snapshots
from .team_names import canonical_betfair_team
from .validity import valid_decimal_odds

# The operator account is Australian. Betfair assigns login endpoints by
# account jurisdiction (not the laptop's current location), while the env
# override keeps the fallback usable for a differently domiciled account.
IDENTITY_URL_DEFAULT = "https://identitysso.betfair.com.au/api/login"
BETTING_URL = "https://api.betfair.com/exchange/betting/json-rpc/v1"
RUGBY_LEAGUE_EVENT_TYPE_ID = "1477"
NRL_COMPETITION_ID = "10564377"

_OVER_UNDER_RE = re.compile(r"^(over|under)\s+([\d.]+)", re.IGNORECASE)
_HANDICAP_RE = re.compile(r"([+-]\d+(?:\.\d+)?)\s*$")


class BetfairError(RuntimeError):
    pass


class BetfairClient:
    def __init__(
        self,
        app_key: str | None = None,
        username: str | None = None,
        password: str | None = None,
        identity_url: str | None = None,
        timeout_seconds: int = 30,
    ) -> None:
        self.app_key = app_key or os.environ.get("BETFAIR_APP_KEY", "")
        self.username = username or os.environ.get("BETFAIR_USERNAME", "")
        self.password = password or os.environ.get("BETFAIR_PASSWORD", "")
        self.identity_url = (
            identity_url
            or os.environ.get("BETFAIR_IDENTITY_URL")
            or IDENTITY_URL_DEFAULT
        )
        self.timeout_seconds = timeout_seconds
        self.session_token: str | None = None

    @property
    def configured(self) -> bool:
        return bool(self.app_key and self.username and self.password)

    def login(self) -> None:
        response = requests.post(
            self.identity_url,
            data={"username": self.username, "password": self.password},
            headers={
                "X-Application": self.app_key,
                "Accept": "application/json",
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        token = payload.get("token") or payload.get("sessionToken")
        status = payload.get("status") or payload.get("loginStatus")
        if not token or str(status).upper() not in {"SUCCESS", "SUCCESSFUL"}:
            raise BetfairError(f"Betfair login failed: {status}")
        self.session_token = token

    def _rpc(self, method: str, params: dict) -> list | dict:
        if not self.session_token:
            raise BetfairError("Not logged in")
        response = requests.post(
            BETTING_URL,
            json=[
                {
                    "jsonrpc": "2.0",
                    "method": f"SportsAPING/v1.0/{method}",
                    "params": params,
                    "id": 1,
                }
            ],
            headers={
                "X-Application": self.app_key,
                "X-Authentication": self.session_token,
                "Content-Type": "application/json",
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()[0]
        if "error" in body:
            raise BetfairError(f"{method}: {body['error']}")
        return body.get("result", [])

    def list_nrl_markets(self, days_ahead: int = 8) -> list[dict]:
        now = dt.datetime.now(dt.timezone.utc)
        market_filter = {
            "eventTypeIds": [RUGBY_LEAGUE_EVENT_TYPE_ID],
            "competitionIds": [NRL_COMPETITION_ID],
            "marketStartTime": {
                "from": now.isoformat().replace("+00:00", "Z"),
                "to": (now + dt.timedelta(days=days_ahead))
                .isoformat()
                .replace("+00:00", "Z"),
            },
        }
        return self._rpc(
            "listMarketCatalogue",
            {
                "filter": market_filter,
                "maxResults": 200,
                "marketProjection": [
                    "EVENT",
                    "COMPETITION",
                    "MARKET_START_TIME",
                    "RUNNER_DESCRIPTION",
                ],
            },
        )

    def market_books(self, market_ids: list[str]) -> list[dict]:
        books: list[dict] = []
        for start in range(0, len(market_ids), 25):
            books.extend(
                self._rpc(
                    "listMarketBook",
                    {
                        "marketIds": market_ids[start : start + 25],
                        "priceProjection": {"priceData": ["EX_BEST_OFFERS"]},
                    },
                )
            )
        return books


# A tradeable quote needs both sides of the book and a tight spread;
# one-sided 1.01/1000 placeholder books are noise, not prices.
MAX_BACK_LAY_SPREAD_RATIO = 1.35
H2H_OVERROUND_RANGE = (0.90, 1.18)
BALANCED_ODDS_RANGE = (1.3, 3.2)


def _mid_price(runner_book: dict) -> float | None:
    """Two-way midpoint only; one-sided or wide books return None."""
    exchange = runner_book.get("ex") or {}
    backs = exchange.get("availableToBack") or []
    lays = exchange.get("availableToLay") or []
    if not backs or not lays:
        return None
    best_back = backs[0].get("price")
    best_lay = lays[0].get("price")
    if not valid_decimal_odds(best_back) or not valid_decimal_odds(best_lay):
        return None
    if float(best_lay) / float(best_back) > MAX_BACK_LAY_SPREAD_RATIO:
        return None
    return round((float(best_back) + float(best_lay)) / 2, 3)


def _classify_market(market_name: str) -> str | None:
    name = market_name.strip().lower()
    if name in {"match odds", "head to head"}:
        return "h2h"
    if name in {"handicap", "line"} or "handicap" in name:
        return "line"
    if name in {"total points", "total match points"} or name.startswith("over/under"):
        return "totals"
    return None


def _pick_balanced_line(lines: dict[float, tuple[float, float]]) -> float | None:
    """Choose the active line from an Asian-style multi-line book.

    `lines` maps line -> (mid_a, mid_b). The main line is where both sides
    are priced near even money; unpriced/one-sided lines never get here.
    """
    candidates = []
    for line, (mid_a, mid_b) in lines.items():
        low, high = BALANCED_ODDS_RANGE
        if not (low <= mid_a <= high and low <= mid_b <= high):
            continue
        gap = abs(mid_a - mid_b)
        candidates.append((gap, abs(float(line)), float(line)))
    return min(candidates)[2] if candidates else None


def collect_snapshots(client: BetfairClient) -> dict[tuple, dict]:
    """(home, away, kickoff ISO time) -> odds values from current NRL markets."""
    catalogue = client.list_nrl_markets()
    if not catalogue:
        return {}

    markets_by_id: dict[str, dict] = {}
    for market in catalogue:
        market_kind = _classify_market(market.get("marketName", ""))
        event = market.get("event") or {}
        event_name = event.get("name") or ""
        competition = market.get("competition") or {}
        if competition.get("id") and str(competition["id"]) != NRL_COMPETITION_ID:
            continue
        if market_kind is None:
            continue
        parts = re.split(r"\s+v(?:s)?\s+", event_name, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) != 2:
            continue
        home = canonical_betfair_team(parts[0])
        away = canonical_betfair_team(parts[1])
        if not home or not away:
            continue
        markets_by_id[market["marketId"]] = {
            "kind": market_kind,
            "home": home,
            "away": away,
            "open_date": event.get("openDate"),
            "runners": {
                runner["selectionId"]: runner
                for runner in market.get("runners", [])
            },
        }

    if not markets_by_id:
        return {}

    snapshots: dict[tuple, dict] = {}
    books = client.market_books(list(markets_by_id))
    for book in books:
        meta = markets_by_id.get(book.get("marketId"))
        if meta is None:
            continue
        open_date = meta.get("open_date") or ""
        key = (meta["home"], meta["away"], str(open_date))
        values = snapshots.setdefault(key, {})

        if meta["kind"] == "h2h":
            mids: dict[str, float] = {}
            for runner_book in book.get("runners", []):
                runner_meta = meta["runners"].get(runner_book.get("selectionId"), {})
                team = canonical_betfair_team(
                    runner_meta.get("runnerName", ""),
                    (meta["home"], meta["away"]),
                )
                price = _mid_price(runner_book)
                if team and price:
                    mids[team] = price
            home_mid = mids.get(meta["home"])
            away_mid = mids.get(meta["away"])
            if home_mid and away_mid:
                overround = 1.0 / home_mid + 1.0 / away_mid
                if H2H_OVERROUND_RANGE[0] <= overround <= H2H_OVERROUND_RANGE[1]:
                    values["h2h_odds_home"] = home_mid
                    values["h2h_odds_away"] = away_mid
            continue

        # line/totals are Asian-style multi-line books: pair the two sides
        # per handicap value, then pick the balanced (active) line
        per_line: dict[float, dict[str, float]] = {}
        for runner_book in book.get("runners", []):
            runner_meta = meta["runners"].get(runner_book.get("selectionId"), {})
            runner_name = str(runner_meta.get("runnerName", "")).strip()
            handicap = runner_book.get("handicap")
            price = _mid_price(runner_book)
            if handicap is None or price is None:
                continue
            handicap = float(handicap)
            if meta["kind"] == "line":
                team = canonical_betfair_team(
                    _HANDICAP_RE.sub("", runner_name).strip(),
                    (meta["home"], meta["away"]),
                )
                if team == meta["home"]:
                    per_line.setdefault(handicap, {})["home"] = price
                elif team == meta["away"]:
                    # away runner is listed at the mirrored handicap
                    per_line.setdefault(-handicap, {})["away"] = price
            else:
                lowered = runner_name.lower()
                side = "over" if lowered.startswith("over") else (
                    "under" if lowered.startswith("under") else None
                )
                if side is None:
                    match = _OVER_UNDER_RE.match(runner_name)
                    side = match.group(1).lower() if match else None
                if side:
                    per_line.setdefault(handicap, {})[side] = price

        if meta["kind"] == "line":
            complete = {
                line: (sides["home"], sides["away"])
                for line, sides in per_line.items()
                if "home" in sides and "away" in sides
            }
            line = _pick_balanced_line(complete)
            if line is not None:
                values["line_amount_home"] = line
                values["line_odds_home"] = complete[line][0]
                values["line_odds_away"] = complete[line][1]
        else:
            complete = {
                line: (sides["over"], sides["under"])
                for line, sides in per_line.items()
                if "over" in sides and "under" in sides
            }
            line = _pick_balanced_line(complete)
            if line is not None:
                values["total_line"] = line
                values["total_over_odds"] = complete[line][0]
                values["total_under_odds"] = complete[line][1]

    return {key: values for key, values in snapshots.items() if values}


def snapshot_live_odds(
    db_path: str | Path,
    client: BetfairClient | None = None,
    exclude_game_ids: set[int] | None = None,
) -> dict:
    """Snapshot Betfair prices for upcoming games and update the fixture cache."""
    client = client or BetfairClient()
    if not client.configured:
        print("[odds] Betfair credentials not configured; skipping live snapshot.")
        return {
            "status": "skipped",
            "reason": "not_configured",
            "provider": "betfair",
        }

    try:
        client.login()
        snapshots = collect_snapshots(client)
    except Exception as exc:
        print(f"[odds] Betfair snapshot failed softly: {exc}")
        return {"status": "failed", "reason": str(exc), "provider": "betfair"}

    if not snapshots:
        print("[odds] Betfair returned no NRL markets in the window.")
        return {"status": "no_markets", "provider": "betfair"}

    provider_snapshots = [
        {
            "home": home,
            "away": away,
            "commence_time": commence_time,
            "values": values,
        }
        for (home, away, commence_time), values in snapshots.items()
    ]
    summary = persist_live_snapshots(
        db_path,
        source="betfair",
        snapshots=provider_snapshots,
        exclude_game_ids=exclude_game_ids,
    )
    status = "completed" if summary["games_updated"] else "no_matches"
    print(
        "[odds] Betfair snapshot: "
        f"{summary['games_updated']}/{summary['fixture_count']} upcoming games matched; "
        f"H2H={summary['h2h_games']}, line={summary['line_games']}, "
        f"totals={summary['totals_games']}."
    )
    return {"status": status, "provider": "betfair", **summary}
