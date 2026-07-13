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
import sqlite3
from pathlib import Path

import requests

from ..nrl_data.cache_writer import update_fixture_odds
from . import store
from .team_names import canonical_team

IDENTITY_URL_DEFAULT = "https://identitysso.betfair.com.au/api/login"
BETTING_URL = "https://api.betfair.com/exchange/betting/json-rpc/v1"
RUGBY_LEAGUE_EVENT_TYPE_ID = "1477"

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
    if not best_back or not best_lay or best_back <= 1.0:
        return None
    if best_lay / best_back > MAX_BACK_LAY_SPREAD_RATIO:
        return None
    return round((best_back + best_lay) / 2, 3)


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
    best_line = None
    best_gap = None
    for line, (mid_a, mid_b) in lines.items():
        low, high = BALANCED_ODDS_RANGE
        if not (low <= mid_a <= high and low <= mid_b <= high):
            continue
        gap = abs(mid_a - mid_b)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_line = line
    return best_line


def collect_snapshots(client: BetfairClient) -> dict[tuple, dict]:
    """(home, away, kickoff_date) -> odds values from current NRL markets."""
    catalogue = client.list_nrl_markets()
    if not catalogue:
        return {}

    markets_by_id: dict[str, dict] = {}
    for market in catalogue:
        market_kind = _classify_market(market.get("marketName", ""))
        event = market.get("event") or {}
        event_name = event.get("name") or ""
        if market_kind is None:
            continue
        parts = re.split(r"\s+v(?:s)?\s+", event_name, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) != 2:
            continue
        home = canonical_team(parts[0])
        away = canonical_team(parts[1])
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
        kickoff_date = str(open_date)[:10]
        key = (meta["home"], meta["away"], kickoff_date)
        values = snapshots.setdefault(key, {})

        if meta["kind"] == "h2h":
            mids: dict[str, float] = {}
            for runner_book in book.get("runners", []):
                runner_meta = meta["runners"].get(runner_book.get("selectionId"), {})
                team = canonical_team(runner_meta.get("runnerName", ""))
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
                team = canonical_team(_HANDICAP_RE.sub("", runner_name).strip())
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


def _pre_game_fixtures(con: sqlite3.Connection) -> list[dict]:
    rows = []
    for row in con.execute(
        "SELECT game_id, competition_year, round_id, team_home, team_away, "
        "start_time_utc FROM feed_cache_fixtures "
        "WHERE game_state_name = 'Pre Game'"
    ):
        rows.append(
            {
                "game_id": int(float(row[0])),
                "competition_year": int(float(row[1])),
                "round_id": int(float(row[2])),
                "team_home": row[3],
                "team_away": row[4],
                "start_time_utc": row[5],
            }
        )
    return rows


def snapshot_live_odds(db_path: str | Path, client: BetfairClient | None = None) -> dict:
    """Snapshot Betfair prices for upcoming games and update the fixture cache."""
    client = client or BetfairClient()
    if not client.configured:
        print("[odds] Betfair credentials not configured; skipping live snapshot.")
        return {"status": "skipped", "reason": "not_configured"}

    try:
        client.login()
        snapshots = collect_snapshots(client)
    except Exception as exc:
        print(f"[odds] Betfair snapshot failed softly: {exc}")
        return {"status": "failed", "reason": str(exc)}

    if not snapshots:
        print("[odds] Betfair returned no NRL markets in the window.")
        return {"status": "no_markets"}

    con = sqlite3.connect(str(db_path))
    try:
        store.ensure_tables(con)
        now_iso = store.utc_now_iso()
        matched = 0
        for fixture in _pre_game_fixtures(con):
            kickoff = fixture.get("start_time_utc")
            kickoff_dates = set()
            if kickoff is not None:
                base = dt.datetime.fromtimestamp(
                    float(kickoff), tz=dt.timezone.utc
                ).date()
                kickoff_dates = {
                    str(base + dt.timedelta(days=offset)) for offset in (-1, 0, 1)
                }
            values = None
            for (home, away, date), snap in snapshots.items():
                if home == fixture["team_home"] and away == fixture["team_away"]:
                    if not kickoff_dates or date in kickoff_dates:
                        values = snap
                        break
            if not values:
                continue
            matched += 1
            store.insert_snapshot(
                con,
                fixture["game_id"],
                fixture["competition_year"],
                fixture["round_id"],
                source="betfair",
                snapshot_kind="live",
                snapshot_time_utc=now_iso,
                values=values,
            )
            odds_update = {
                "team_head_to_head_odds_home": values.get("h2h_odds_home"),
                "team_head_to_head_odds_away": values.get("h2h_odds_away"),
                "team_line_odds_home": values.get("line_odds_home"),
                "team_line_odds_away": values.get("line_odds_away"),
                "team_line_amount_home": values.get("line_amount_home"),
                "total_line": values.get("total_line"),
                "total_over_odds": values.get("total_over_odds"),
                "total_under_odds": values.get("total_under_odds"),
            }
            if values.get("line_amount_home") is not None:
                odds_update["team_line_amount_away"] = -values["line_amount_home"]
            odds_update = {
                key: value for key, value in odds_update.items() if value is not None
            }
            if odds_update:
                update_fixture_odds(con, fixture["game_id"], odds_update)
        con.commit()
        print(f"[odds] Betfair snapshot: {matched} upcoming games updated.")
        return {"status": "completed", "games_updated": matched}
    finally:
        con.close()
