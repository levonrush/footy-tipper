import datetime as dt
import os
import pathlib
import sqlite3
import tempfile
import unittest
from contextlib import closing
from unittest import mock

from pipeline.common.nrl_data.cache_writer import update_fixture_odds
from pipeline.common.odds import store as odds_store
from pipeline.common.odds.aussportsbetting import (
    _match_game,
    _fixture_index,
    _snapshot_values,
)
from pipeline.common.odds.betfair import (
    IDENTITY_URL_DEFAULT,
    NRL_COMPETITION_ID,
    BetfairClient,
    _classify_market,
    _mid_price,
    _pick_balanced_line,
    collect_snapshots,
)
from pipeline.common.odds.live import persist_live_snapshots
from pipeline.common.odds.team_names import canonical_betfair_team, canonical_team
from pipeline.common.odds.the_odds_api import (
    OddsApiClient,
    _safe_error,
    parse_events,
    select_bookmaker,
    snapshot_live_odds as snapshot_the_odds_api,
)
from pipeline.common.odds.validity import (
    valid_decimal_odds,
    validated_market_values,
)


def _make_db():
    con = sqlite3.connect(":memory:")
    con.execute(
        """
        CREATE TABLE feed_cache_fixtures (
            game_id REAL, competition_year REAL, round_id REAL,
            game_state_name TEXT, start_time REAL, start_time_utc REAL,
            team_home TEXT, team_away TEXT,
            team_head_to_head_odds_home REAL, team_head_to_head_odds_away REAL,
            team_line_odds_home REAL, team_line_odds_away REAL,
            team_line_amount_home REAL, team_line_amount_away REAL
        )
        """
    )
    # start_time = local wall clock as-if-UTC for 2026-07-12 18:15 AEST
    local = dt.datetime(2026, 7, 12, 18, 15, tzinfo=dt.timezone.utc).timestamp()
    con.execute(
        "INSERT INTO feed_cache_fixtures VALUES (20261111950.0, 2026, 19, 'Final', ?, ?, "
        "'Melbourne Storm', 'Gold Coast Titans', 1.40, NULL, NULL, NULL, NULL, NULL)",
        (local, local - 10 * 3600),
    )
    return con


class TeamNameTests(unittest.TestCase):
    def test_aliases_map_to_canonical(self):
        self.assertEqual(canonical_team("Canterbury Bulldogs"), "Canterbury-Bankstown Bulldogs")
        self.assertEqual(canonical_team("Manly Sea Eagles"), "Manly-Warringah Sea Eagles")
        self.assertEqual(canonical_team("St George Illawarra Dragons"), "St. George Illawarra Dragons")
        self.assertEqual(canonical_team("Melbourne Storm"), "Melbourne Storm")
        self.assertIsNone(canonical_team("Not A Team"))


class AussportsbettingTests(unittest.TestCase):
    def test_date_tolerant_match(self):
        con = _make_db()
        self.addCleanup(con.close)
        index = _fixture_index(con)
        record = {
            "home_team": "Melbourne Storm",
            "away_team": "Gold Coast Titans",
            "date": dt.datetime(2026, 7, 12),
        }
        self.assertEqual(_match_game(record, index)[0], 20261111950)
        record["date"] = dt.datetime(2026, 7, 13)  # +-1 day tolerance
        self.assertEqual(_match_game(record, index)[0], 20261111950)
        record["date"] = dt.datetime(2026, 7, 15)
        self.assertIsNone(_match_game(record, index))

    def test_snapshot_values_open_vs_close(self):
        record = {
            "h2h_home_open": 1.5, "h2h_home_close": 1.4,
            "h2h_away_open": 2.6, "h2h_away_close": 2.9,
            "line_home_close": -3.5, "line_odds_home_close": 1.9,
            "line_odds_away_close": 1.9,
            "total_close": 41.5, "total_over_close": 1.9, "total_under_close": 1.9,
        }
        open_values = _snapshot_values(record, "open")
        close_values = _snapshot_values(record, "close")
        self.assertEqual(open_values["h2h_odds_home"], 1.5)
        self.assertEqual(close_values["h2h_odds_home"], 1.4)
        self.assertEqual(close_values["line_amount_home"], -3.5)
        self.assertEqual(close_values["total_line"], 41.5)
        self.assertNotIn("line_amount_home", open_values)

    def test_fill_only_when_null(self):
        con = _make_db()
        self.addCleanup(con.close)
        updated = update_fixture_odds(
            con,
            20261111950.0,
            {
                "team_head_to_head_odds_home": 9.99,  # existing 1.40 must survive
                "team_head_to_head_odds_away": 2.85,
                "total_line": 41.5,
            },
            only_when_null=True,
        )
        self.assertTrue(updated)
        row = con.execute(
            "SELECT team_head_to_head_odds_home, team_head_to_head_odds_away, total_line "
            "FROM feed_cache_fixtures"
        ).fetchone()
        self.assertEqual(row[0], 1.40)
        self.assertEqual(row[1], 2.85)
        self.assertEqual(row[2], 41.5)


class OddsHistoryStoreTests(unittest.TestCase):
    def test_insert_dedup(self):
        con = sqlite3.connect(":memory:")
        self.addCleanup(con.close)
        odds_store.ensure_tables(con)
        values = {"h2h_odds_home": 1.5, "h2h_odds_away": 2.6}
        first = odds_store.insert_snapshot(
            con, 1, 2026, 19, "aussportsbetting", "close", None, values
        )
        second = odds_store.insert_snapshot(
            con, 1, 2026, 19, "aussportsbetting", "close", None, values
        )
        self.assertTrue(first)
        self.assertFalse(second)


def _two_way(back, lay):
    return {"availableToBack": [{"price": back}], "availableToLay": [{"price": lay}]}


class BetfairTests(unittest.TestCase):
    def test_australian_identity_default_and_nrl_competition_filter(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            client = BetfairClient("key", "user", "password")
        self.assertEqual(client.identity_url, IDENTITY_URL_DEFAULT)
        self.assertEqual(
            IDENTITY_URL_DEFAULT,
            "https://identitysso.betfair.com.au/api/login",
        )
        with mock.patch.object(client, "_rpc", return_value=[]) as rpc:
            client.list_nrl_markets()
        params = rpc.call_args.args[1]
        self.assertEqual(params["filter"]["competitionIds"], [NRL_COMPETITION_ID])
        self.assertIn("COMPETITION", params["marketProjection"])

    def test_classify_market(self):
        self.assertEqual(_classify_market("Match Odds"), "h2h")
        self.assertEqual(_classify_market("Head To Head"), "h2h")
        self.assertEqual(_classify_market("Handicap"), "line")
        self.assertEqual(_classify_market("Total Points"), "totals")
        self.assertIsNone(_classify_market("First Try Scorer"))
        self.assertIsNone(_classify_market("Half Time/Full Time"))
        self.assertIsNone(_classify_market("Regular Time Match Odds"))

    def test_mid_price_requires_two_way_tight_book(self):
        self.assertEqual(_mid_price({"ex": _two_way(1.90, 1.94)}), 1.92)
        # one-sided books are placeholder noise, not prices
        self.assertIsNone(_mid_price({"ex": {"availableToBack": [{"price": 1.01}]}}))
        self.assertIsNone(_mid_price({"ex": {}}))
        # wide spreads rejected
        self.assertIsNone(_mid_price({"ex": _two_way(1.5, 2.5)}))

    def test_balanced_line_tie_break_is_order_independent(self):
        first = {
            8.5: (1.90, 2.00),
            -4.5: (2.00, 1.90),
            4.5: (1.90, 2.00),
        }
        second = dict(reversed(list(first.items())))

        self.assertEqual(_pick_balanced_line(first), -4.5)
        self.assertEqual(_pick_balanced_line(second), -4.5)

    def test_collect_snapshots_multi_line_markets(self):
        class FakeClient:
            def list_nrl_markets(self):
                return [
                    {
                        "marketId": "1.1",
                        "marketName": "Head To Head",
                        "event": {"name": "Melbourne Storm v Gold Coast Titans",
                                  "openDate": "2026-07-12T08:15:00.000Z"},
                        "runners": [
                            {"selectionId": 1, "runnerName": "Melbourne Storm"},
                            {"selectionId": 2, "runnerName": "Gold Coast Titans"},
                        ],
                    },
                    {
                        "marketId": "1.2",
                        "marketName": "Handicap",
                        "event": {"name": "Melbourne Storm v Gold Coast Titans",
                                  "openDate": "2026-07-12T08:15:00.000Z"},
                        "runners": [
                            {"selectionId": 3, "runnerName": "Melbourne Storm"},
                            {"selectionId": 4, "runnerName": "Gold Coast Titans"},
                        ],
                    },
                    {
                        "marketId": "1.3",
                        "marketName": "Total Points",
                        "event": {"name": "Melbourne Storm v Gold Coast Titans",
                                  "openDate": "2026-07-12T08:15:00.000Z"},
                        "runners": [
                            {"selectionId": 5, "runnerName": "Over"},
                            {"selectionId": 6, "runnerName": "Under"},
                        ],
                    },
                ]

            def market_books(self, market_ids):
                return [
                    {
                        "marketId": "1.1",
                        "runners": [
                            {"selectionId": 1, "handicap": 0.0, "ex": _two_way(1.38, 1.42)},
                            {"selectionId": 2, "handicap": 0.0, "ex": _two_way(3.2, 3.4)},
                        ],
                    },
                    {
                        "marketId": "1.2",
                        "runners": [
                            # far line: one-sided placeholder, must be ignored
                            {"selectionId": 3, "handicap": -20.5,
                             "ex": {"availableToBack": [{"price": 1.01}]}},
                            {"selectionId": 4, "handicap": 20.5, "ex": {}},
                            # active line: balanced two-way prices
                            {"selectionId": 3, "handicap": -6.5, "ex": _two_way(1.90, 1.94)},
                            {"selectionId": 4, "handicap": 6.5, "ex": _two_way(1.92, 1.96)},
                        ],
                    },
                    {
                        "marketId": "1.3",
                        "runners": [
                            {"selectionId": 5, "handicap": 30.5,
                             "ex": {"availableToBack": [{"price": 1.01}]}},
                            {"selectionId": 6, "handicap": 30.5, "ex": {}},
                            {"selectionId": 5, "handicap": 41.5, "ex": _two_way(1.9, 1.94)},
                            {"selectionId": 6, "handicap": 41.5, "ex": _two_way(1.88, 1.92)},
                        ],
                    },
                ]

        snapshots = collect_snapshots(FakeClient())
        key = (
            "Melbourne Storm",
            "Gold Coast Titans",
            "2026-07-12T08:15:00.000Z",
        )
        self.assertIn(key, snapshots)
        values = snapshots[key]
        self.assertEqual(values["h2h_odds_home"], 1.4)
        self.assertEqual(values["h2h_odds_away"], 3.3)
        self.assertEqual(values["line_amount_home"], -6.5)
        self.assertEqual(values["line_odds_home"], 1.92)
        self.assertEqual(values["line_odds_away"], 1.94)
        self.assertEqual(values["total_line"], 41.5)
        self.assertEqual(values["total_over_odds"], 1.92)
        self.assertEqual(values["total_under_odds"], 1.9)

    def test_empty_books_yield_no_snapshot(self):
        class FakeClient:
            def list_nrl_markets(self):
                return [
                    {
                        "marketId": "1.1",
                        "marketName": "Head To Head",
                        "event": {"name": "Melbourne Storm v Gold Coast Titans",
                                  "openDate": "2026-07-12T08:15:00.000Z"},
                        "runners": [
                            {"selectionId": 1, "runnerName": "Melbourne Storm"},
                            {"selectionId": 2, "runnerName": "Gold Coast Titans"},
                        ],
                    },
                ]

            def market_books(self, market_ids):
                return [
                    {
                        "marketId": "1.1",
                        "runners": [
                            {"selectionId": 1, "handicap": 0.0,
                             "ex": {"availableToBack": [{"price": 1.23}]}},
                            {"selectionId": 2, "handicap": 0.0,
                             "ex": {"availableToBack": [{"price": 1.01}]}},
                        ],
                    },
                ]

        self.assertEqual(collect_snapshots(FakeClient()), {})


    def test_real_short_runner_names_and_non_nrl_filter(self):
        class FakeClient:
            def list_nrl_markets(self):
                base = {
                    "marketName": "Head To Head",
                    "event": {
                        "name": "Parramatta v Penrith",
                        "openDate": "2026-07-23T09:50:00Z",
                    },
                    "runners": [
                        {"selectionId": 1, "runnerName": "Parramatta"},
                        {"selectionId": 2, "runnerName": "Penrith"},
                    ],
                }
                return [
                    {
                        **base,
                        "marketId": "nrl",
                        "competition": {"id": NRL_COMPETITION_ID},
                    },
                    {
                        **base,
                        "marketId": "nrlw",
                        "competition": {"id": "other"},
                    },
                ]

            def market_books(self, market_ids):
                self.market_ids = market_ids
                return [
                    {
                        "marketId": "nrl",
                        "runners": [
                            {"selectionId": 1, "ex": _two_way(5.9, 6.1)},
                            {"selectionId": 2, "ex": _two_way(1.19, 1.20)},
                        ],
                    }
                ]

        client = FakeClient()
        snapshots = collect_snapshots(client)
        self.assertEqual(client.market_ids, ["nrl"])
        values = snapshots[
            ("Parramatta Eels", "Penrith Panthers", "2026-07-23T09:50:00Z")
        ]
        self.assertEqual(values["h2h_odds_home"], 6.0)
        self.assertEqual(values["h2h_odds_away"], 1.195)


class MarketValidityTests(unittest.TestCase):
    def test_decimal_prices_and_pickem_line(self):
        for value in (None, 0, 1, float("nan"), float("inf"), "1.91"):
            self.assertFalse(valid_decimal_odds(value))
        self.assertTrue(valid_decimal_odds(1.91))

        missing = validated_market_values(
            {"line_amount_home": 0, "line_odds_home": 0, "line_odds_away": 0}
        )
        self.assertNotIn("line_amount_home", missing)
        pickem = validated_market_values(
            {
                "line_amount_home": 0,
                "line_odds_home": 1.91,
                "line_odds_away": 1.91,
            }
        )
        self.assertEqual(pickem["line_amount_home"], 0.0)

    def test_betfair_short_names_are_contextual(self):
        aliases = {
            "Parramatta": "Parramatta Eels",
            "Penrith": "Penrith Panthers",
            "Newcastle": "Newcastle Knights",
            "Sydney": "Sydney Roosters",
            "South Sydney": "South Sydney Rabbitohs",
            "Melbourne": "Melbourne Storm",
            "Canberra": "Canberra Raiders",
            "Wests Tigers": "Wests Tigers",
            "Canterbury": "Canterbury-Bankstown Bulldogs",
            "NZ Warriors": "New Zealand Warriors",
            "North Qld": "North Queensland Cowboys",
            "Brisbane": "Brisbane Broncos",
            "St George": "St. George Illawarra Dragons",
            "Gold Coast": "Gold Coast Titans",
            "Manly": "Manly-Warringah Sea Eagles",
            "Cronulla": "Cronulla-Sutherland Sharks",
        }
        for label, expected in aliases.items():
            with self.subTest(label=label):
                self.assertEqual(canonical_betfair_team(label), expected)
        self.assertIsNone(
            canonical_betfair_team(
                "Sydney",
                ("South Sydney Rabbitohs", "Melbourne Storm"),
            )
        )


def _bookmaker(
    key,
    home_price,
    away_price,
    updated="2026-07-23T07:00:00Z",
    include_extras=False,
):
    markets = [
        {
            "key": "h2h",
            "outcomes": [
                {"name": "Parramatta Eels", "price": home_price},
                {"name": "Penrith Panthers", "price": away_price},
            ],
        }
    ]
    if include_extras:
        markets.extend(
            [
                {
                    "key": "spreads",
                    "outcomes": [
                        {"name": "Parramatta Eels", "price": 2.02, "point": 14.5},
                        {"name": "Penrith Panthers", "price": 1.99, "point": -14.5},
                    ],
                },
                {
                    "key": "totals",
                    "outcomes": [
                        {"name": "Over", "price": 1.97, "point": 44.5},
                        {"name": "Under", "price": 2.04, "point": 44.5},
                    ],
                },
            ]
        )
    return {
        "key": key,
        "title": key,
        "last_update": updated,
        "markets": markets,
    }


def _odds_api_event(bookmakers):
    return {
        "id": "event-1",
        "sport_key": "rugbyleague_nrl",
        "commence_time": "2026-07-23T09:50:00Z",
        "home_team": "Parramatta Eels",
        "away_team": "Penrith Panthers",
        "bookmakers": bookmakers,
    }


ODDS_API_NOW = dt.datetime(2026, 7, 23, 8, 0, tzinfo=dt.timezone.utc)


class TheOddsApiTests(unittest.TestCase):
    def test_prefers_complete_betfair_exchange_book(self):
        event = _odds_api_event(
            [
                _bookmaker("sportsbet", 2.0, 2.0),
                _bookmaker("betfair_ex_au", 6.0, 1.195, include_extras=True),
                _bookmaker("tab", 1.9, 2.1),
            ]
        )
        selected, pair, _ = select_bookmaker(
            event, "Parramatta Eels", "Penrith Panthers", now=ODDS_API_NOW
        )
        self.assertEqual(selected["key"], "betfair_ex_au")
        self.assertEqual(pair, (6.0, 1.195))

        snapshot = parse_events(
            [event],
            {"requests_remaining": "498"},
            now=ODDS_API_NOW,
        )[0]
        self.assertEqual(snapshot["values"]["line_amount_home"], 14.5)
        self.assertEqual(snapshot["values"]["total_line"], 44.5)
        self.assertEqual(snapshot["raw_meta"]["bookmaker_key"], "betfair_ex_au")
        self.assertEqual(snapshot["raw_meta"]["requests_remaining"], "498")

    def test_incomplete_betfair_does_not_beat_a_complete_real_book(self):
        event = _odds_api_event(
            [
                _bookmaker("betfair_ex_au", 1.99, 2.01),
                _bookmaker("sportsbet", 2.02, 1.98, include_extras=True),
                _bookmaker("tab", 1.90, 2.10),
            ]
        )

        selected, pair, _ = select_bookmaker(
            event,
            "Parramatta Eels",
            "Penrith Panthers",
            now=ODDS_API_NOW,
        )

        self.assertEqual(selected["key"], "sportsbet")
        self.assertEqual(pair, (2.02, 1.98))

    def test_consensus_nearest_book_with_freshness_tie_break(self):
        event = _odds_api_event(
            [
                _bookmaker("old", 2.2, 1.8, "2026-07-23T06:00:00Z"),
                _bookmaker("fresh", 1.8, 2.2, "2026-07-23T08:00:00Z"),
            ]
        )
        selected, _, _ = select_bookmaker(
            event, "Parramatta Eels", "Penrith Panthers", now=ODDS_API_NOW
        )
        self.assertEqual(selected["key"], "fresh")

        event["bookmakers"].append(
            _bookmaker("median", 2.0, 2.0, "2026-07-23T05:00:00Z")
        )
        selected, _, _ = select_bookmaker(
            event, "Parramatta Eels", "Penrith Panthers", now=ODDS_API_NOW
        )
        self.assertEqual(selected["key"], "median")

        lexical_tie = _odds_api_event(
            [
                _bookmaker("z_book", 2.0, 2.0),
                _bookmaker("a_book", 2.0, 2.0),
            ]
        )
        selected, _, _ = select_bookmaker(
            lexical_tie,
            "Parramatta Eels",
            "Penrith Panthers",
            now=ODDS_API_NOW,
        )
        self.assertEqual(selected["key"], "a_book")

    def test_partial_markets_are_rejected(self):
        partial = _bookmaker("partial", 2.0, None, include_extras=True)
        self.assertIsNone(
            select_bookmaker(
                _odds_api_event([partial]),
                "Parramatta Eels",
                "Penrith Panthers",
                now=ODDS_API_NOW,
            )
        )
        bookmaker = _bookmaker("sportsbet", 6.0, 1.2, include_extras=True)
        bookmaker["markets"][1]["outcomes"][1]["price"] = 0
        snapshot = parse_events(
            [_odds_api_event([bookmaker])],
            now=ODDS_API_NOW,
        )[0]
        self.assertNotIn("line_amount_home", snapshot["values"])
        self.assertEqual(snapshot["values"]["total_line"], 44.5)

    def test_stale_and_implausibly_future_books_are_rejected(self):
        event = _odds_api_event(
            [
                _bookmaker("stale", 6.0, 1.2, "2026-07-23T01:59:59Z"),
                _bookmaker("future", 5.9, 1.21, "2026-07-23T08:05:01Z"),
                _bookmaker("fresh", 5.8, 1.22, "2026-07-23T02:00:00Z"),
            ]
        )
        selected, _, candidates = select_bookmaker(
            event,
            "Parramatta Eels",
            "Penrith Panthers",
            now=ODDS_API_NOW,
        )
        self.assertEqual(selected["key"], "fresh")
        self.assertEqual([book["key"] for book, _ in candidates], ["fresh"])

        event["bookmakers"] = event["bookmakers"][:2]
        self.assertIsNone(
            select_bookmaker(
                event,
                "Parramatta Eels",
                "Penrith Panthers",
                now=ODDS_API_NOW,
            )
        )

    def test_client_requests_exact_v4_market_and_returns_quota(self):
        response = mock.Mock()
        response.json.return_value = []
        response.headers = {
            "x-requests-remaining": "497",
            "x-requests-used": "3",
        }
        with mock.patch(
            "pipeline.common.odds.the_odds_api.requests.get",
            return_value=response,
        ) as get:
            events, quota = OddsApiClient(api_key="secret").fetch_odds()
        response.raise_for_status.assert_called_once_with()
        self.assertEqual(events, [])
        self.assertEqual(quota["requests_remaining"], "497")
        params = get.call_args.kwargs["params"]
        self.assertEqual(params["regions"], "au")
        self.assertEqual(params["markets"], "h2h,spreads,totals")
        self.assertEqual(params["apiKey"], "secret")

    def test_provider_errors_redact_api_key(self):
        reason = _safe_error(
            RuntimeError(
                "request failed at https://example.test/odds?apiKey=very-secret&x=1"
            ),
            "very-secret",
        )
        self.assertNotIn("very-secret", reason)
        self.assertIn("apiKey=<redacted>", reason)


class LiveOddsPersistenceTests(unittest.TestCase):
    def _db_path(self, directory):
        path = pathlib.Path(directory) / "odds.sqlite"
        with closing(sqlite3.connect(path)) as con:
            con.execute(
                """
                CREATE TABLE feed_cache_fixtures (
                    game_id REAL, competition_year REAL, round_id REAL,
                    game_state_name TEXT, start_time_utc REAL,
                    team_home TEXT, team_away TEXT,
                    team_head_to_head_odds_home REAL,
                    team_head_to_head_odds_away REAL,
                    team_line_odds_home REAL, team_line_odds_away REAL,
                    team_line_amount_home REAL, team_line_amount_away REAL
                )
                """
            )
            kickoff = dt.datetime(
                2026, 7, 23, 9, 50, tzinfo=dt.timezone.utc
            ).timestamp()
            con.execute(
                "INSERT INTO feed_cache_fixtures VALUES "
                "(1, 2026, 21, 'Pre Game', ?, 'Parramatta Eels', "
                "'Penrith Panthers', 1.5, 2.6, 0, 0, 0, 0)",
                (kickoff,),
            )
            con.commit()
        return path

    def test_six_hour_match_and_partial_values_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._db_path(directory)
            summary = persist_live_snapshots(
                path,
                "the_odds_api",
                [
                    {
                        "home": "Parramatta Eels",
                        "away": "Penrith Panthers",
                        "commence_time": "2026-07-23T15:49:00Z",
                        "values": {
                            # Partial H2H must not overwrite the existing pair.
                            "h2h_odds_home": 9.99,
                            "line_amount_home": 0,
                            "line_odds_home": 1.91,
                            "line_odds_away": 1.91,
                        },
                        "raw_meta": {"bookmaker_key": "sportsbet"},
                    }
                ],
                observed_at="2026-07-23T07:00:00+00:00",
            )
            self.assertEqual(summary["games_updated"], 1)
            self.assertEqual(summary["h2h_games"], 0)
            self.assertEqual(summary["line_games"], 1)
            with closing(sqlite3.connect(path)) as con:
                fixture = con.execute(
                    "SELECT team_head_to_head_odds_home, "
                    "team_head_to_head_odds_away, team_line_amount_home, "
                    "team_line_odds_home, team_line_odds_away "
                    "FROM feed_cache_fixtures"
                ).fetchone()
                # A complete line family without a complete H2H observation is
                # retained in history but cannot become the current atomic
                # fixture snapshot.
                self.assertEqual(fixture, (1.5, 2.6, 0.0, 0.0, 0.0))
                history = con.execute(
                    "SELECT source, h2h_odds_home, h2h_odds_away, "
                    "line_amount_home FROM odds_history"
                ).fetchone()
                self.assertEqual(history, ("the_odds_api", None, None, 0.0))

    def test_more_than_six_hours_does_not_match(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._db_path(directory)
            summary = persist_live_snapshots(
                path,
                "the_odds_api",
                [
                    {
                        "home": "Parramatta Eels",
                        "away": "Penrith Panthers",
                        "commence_time": "2026-07-23T15:51:00Z",
                        "values": {
                            "h2h_odds_home": 6.0,
                            "h2h_odds_away": 1.195,
                        },
                    }
                ],
            )
            self.assertEqual(summary["games_updated"], 0)

    def test_fixture_aliases_are_canonicalized_before_matching(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._db_path(directory)
            with closing(sqlite3.connect(path)) as con:
                con.execute(
                    "UPDATE feed_cache_fixtures "
                    "SET team_home = 'Canterbury Bulldogs', "
                    "team_away = 'New Zealand Warriors'"
                )
                con.commit()
            summary = persist_live_snapshots(
                path,
                "the_odds_api",
                [
                    {
                        "home": "Canterbury-Bankstown Bulldogs",
                        "away": "New Zealand Warriors",
                        "commence_time": "2026-07-23T09:50:00Z",
                        "values": {
                            "h2h_odds_home": 2.10,
                            "h2h_odds_away": 1.80,
                        },
                    }
                ],
            )

            self.assertEqual(summary["games_updated"], 1)

    def test_provider_persists_the_bookmaker_quote_time_for_freshness(self):
        class FakeClient:
            configured = True
            api_key = ""

            def fetch_odds(self):
                return (
                    [
                        _odds_api_event(
                            [
                                _bookmaker(
                                    "sportsbet",
                                    6.0,
                                    1.195,
                                    "2026-07-23T07:00:00Z",
                                )
                            ]
                        )
                    ],
                    {"requests_remaining": "498"},
                )

        with tempfile.TemporaryDirectory() as directory:
            path = self._db_path(directory)
            result = snapshot_the_odds_api(
                path,
                client=FakeClient(),
                now=ODDS_API_NOW,
            )
            self.assertEqual(result["status"], "completed")
            with closing(sqlite3.connect(path)) as con:
                snapshot_time = con.execute(
                    "SELECT snapshot_time_utc FROM odds_history "
                    "WHERE source = 'the_odds_api'"
                ).fetchone()[0]
            self.assertEqual(snapshot_time, "2026-07-23T07:00:00+00:00")

    def test_same_quote_time_updates_ledger_and_cache_together(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._db_path(directory)
            base = {
                "home": "Parramatta Eels",
                "away": "Penrith Panthers",
                "commence_time": "2026-07-23T09:50:00Z",
                "snapshot_time_utc": "2026-07-23T07:00:00+00:00",
            }
            persist_live_snapshots(
                path,
                "the_odds_api",
                [
                    {
                        **base,
                        "values": {
                            "h2h_odds_home": 2.0,
                            "h2h_odds_away": 2.0,
                            "line_amount_home": -6.5,
                            "line_odds_home": 1.91,
                            "line_odds_away": 1.91,
                        },
                    }
                ],
                observed_at="2026-07-23T07:01:00+00:00",
            )
            second = persist_live_snapshots(
                path,
                "the_odds_api",
                [
                    {
                        **base,
                        "values": {
                            "h2h_odds_home": 3.0,
                            "h2h_odds_away": 1.5,
                            "line_amount_home": 4.5,
                            "line_odds_home": 1.92,
                            "line_odds_away": 1.90,
                        },
                    }
                ],
                observed_at="2026-07-23T07:02:00+00:00",
            )

            self.assertEqual(second["snapshots_inserted"], 0)
            with closing(sqlite3.connect(path)) as con:
                history = con.execute(
                    "SELECT COUNT(*), h2h_odds_home, h2h_odds_away, "
                    "line_amount_home FROM odds_history"
                ).fetchone()
                fixture = con.execute(
                    "SELECT team_head_to_head_odds_home, "
                    "team_head_to_head_odds_away, team_line_amount_home "
                    "FROM feed_cache_fixtures"
                ).fetchone()
            self.assertEqual(history, (1, 3.0, 1.5, 4.5))
            self.assertEqual(fixture, (3.0, 1.5, 4.5))

    def test_out_of_order_quote_cannot_replace_newer_fixture_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._db_path(directory)

            def snapshot(quote_time, home_price, away_price):
                return {
                    "home": "Parramatta Eels",
                    "away": "Penrith Panthers",
                    "commence_time": "2026-07-23T09:50:00Z",
                    "snapshot_time_utc": quote_time,
                    "values": {
                        "h2h_odds_home": home_price,
                        "h2h_odds_away": away_price,
                    },
                }

            persist_live_snapshots(
                path,
                "the_odds_api",
                [
                    snapshot(
                        "2026-07-23T07:00:00+00:00",
                        2.0,
                        2.0,
                    )
                ],
            )
            persist_live_snapshots(
                path,
                "the_odds_api",
                [
                    snapshot(
                        "2026-07-23T06:00:00+00:00",
                        5.0,
                        1.2,
                    )
                ],
            )

            with closing(sqlite3.connect(path)) as con:
                fixture = con.execute(
                    "SELECT team_head_to_head_odds_home, "
                    "team_head_to_head_odds_away "
                    "FROM feed_cache_fixtures"
                ).fetchone()
                history_count = con.execute(
                    "SELECT COUNT(*) FROM odds_history"
                ).fetchone()[0]
            self.assertEqual(history_count, 2)
            self.assertEqual(fixture, (2.0, 2.0))


class OddsProviderSelectionTests(unittest.TestCase):
    def test_default_provider_falls_back_to_betfair(self):
        from pipeline import odds

        with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
            "pipeline.common.odds.the_odds_api.snapshot_live_odds",
            return_value={"status": "failed", "provider": "the_odds_api"},
        ) as primary, mock.patch(
            "pipeline.common.odds.betfair.snapshot_live_odds",
            return_value={
                "status": "completed",
                "provider": "betfair",
                "games_updated": 8,
            },
        ) as fallback:
            result = odds._snapshot_live_odds(pathlib.Path("/tmp/odds.sqlite"))
        primary.assert_called_once()
        fallback.assert_called_once()
        self.assertEqual(result["provider"], "betfair")
        self.assertEqual(len(result["attempts"]), 2)

    def test_partial_primary_uses_betfair_only_for_gaps(self):
        from pipeline import odds

        coverage = mock.Mock(
            complete=False,
            error=None,
            covered_games=4,
            total_games=8,
        )
        primary_result = {
            "status": "completed",
            "provider": "the_odds_api",
            "games_updated": 4,
            "h2h_games": 4,
            "line_games": 4,
            "totals_games": 4,
            "snapshots_inserted": 4,
            "fixture_count": 8,
            "game_ids_updated": (1, 2, 3, 4),
        }
        fallback_result = {
            "status": "completed",
            "provider": "betfair",
            "games_updated": 3,
            "h2h_games": 3,
            "line_games": 0,
            "totals_games": 0,
            "snapshots_inserted": 3,
            "fixture_count": 8,
            "game_ids_updated": (5, 6, 7),
        }
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
            "pipeline.common.odds.the_odds_api.snapshot_live_odds",
            return_value=primary_result,
        ), mock.patch(
            "pipeline.ops.odds_gate.current_round_odds_coverage",
            return_value=coverage,
        ), mock.patch(
            "pipeline.common.odds.betfair.snapshot_live_odds",
            return_value=fallback_result,
        ) as fallback:
            result = odds._snapshot_live_odds(pathlib.Path("/tmp/odds.sqlite"))

        fallback.assert_called_once_with(
            db_path=pathlib.Path("/tmp/odds.sqlite"),
            exclude_game_ids={1, 2, 3, 4},
        )
        self.assertEqual(result["provider"], "the_odds_api")
        self.assertEqual(result["fallback_provider"], "betfair")
        self.assertEqual(result["games_updated"], 7)
        self.assertEqual(result["h2h_games"], 7)

    def test_partial_primary_success_survives_failed_fallback(self):
        from pipeline import odds

        coverage = mock.Mock(
            complete=False,
            error=None,
            covered_games=4,
            total_games=8,
        )
        primary_result = {
            "status": "completed",
            "provider": "the_odds_api",
            "games_updated": 4,
            "h2h_games": 4,
            "fixture_count": 8,
            "game_ids_updated": (1, 2, 3, 4),
        }
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
            "pipeline.common.odds.the_odds_api.snapshot_live_odds",
            return_value=primary_result,
        ), mock.patch(
            "pipeline.ops.odds_gate.current_round_odds_coverage",
            return_value=coverage,
        ), mock.patch(
            "pipeline.common.odds.betfair.snapshot_live_odds",
            return_value={"status": "failed", "provider": "betfair"},
        ):
            result = odds._snapshot_live_odds(pathlib.Path("/tmp/odds.sqlite"))

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["games_updated"], 4)
        self.assertEqual(len(result["attempts"]), 2)

    def test_complete_current_round_does_not_fallback_for_future_cache_gaps(self):
        from pipeline import odds

        coverage = mock.Mock(
            complete=True,
            error=None,
            covered_games=8,
            total_games=8,
        )
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
            "pipeline.common.odds.the_odds_api.snapshot_live_odds",
            return_value={
                "status": "completed",
                "provider": "the_odds_api",
                "games_updated": 14,
                "h2h_games": 14,
                "fixture_count": 56,
                "game_ids_updated": tuple(range(1, 15)),
            },
        ), mock.patch(
            "pipeline.ops.odds_gate.current_round_odds_coverage",
            return_value=coverage,
        ) as current_round_coverage, mock.patch(
            "pipeline.common.odds.betfair.snapshot_live_odds"
        ) as fallback:
            db_path = pathlib.Path("/tmp/odds.sqlite")
            result = odds._snapshot_live_odds(db_path)

        current_round_coverage.assert_called_once_with(db_path)
        fallback.assert_not_called()
        self.assertEqual(result["provider"], "the_odds_api")

    def test_partial_current_round_falls_back_despite_future_matches(self):
        from pipeline import odds

        coverage = mock.Mock(
            complete=False,
            error=None,
            covered_games=7,
            total_games=8,
        )
        primary_result = {
            "status": "completed",
            "provider": "the_odds_api",
            "games_updated": 14,
            "h2h_games": 14,
            "line_games": 14,
            "totals_games": 14,
            "snapshots_inserted": 14,
            "fixture_count": 56,
            "game_ids_updated": tuple(range(1, 15)),
        }
        fallback_result = {
            "status": "completed",
            "provider": "betfair",
            "games_updated": 1,
            "h2h_games": 1,
            "line_games": 0,
            "totals_games": 0,
            "snapshots_inserted": 1,
            "fixture_count": 56,
            "game_ids_updated": (15,),
        }
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
            "pipeline.common.odds.the_odds_api.snapshot_live_odds",
            return_value=primary_result,
        ), mock.patch(
            "pipeline.ops.odds_gate.current_round_odds_coverage",
            return_value=coverage,
        ), mock.patch(
            "pipeline.common.odds.betfair.snapshot_live_odds",
            return_value=fallback_result,
        ) as fallback:
            result = odds._snapshot_live_odds(pathlib.Path("/tmp/odds.sqlite"))

        fallback.assert_called_once_with(
            db_path=pathlib.Path("/tmp/odds.sqlite"),
            exclude_game_ids=set(range(1, 15)),
        )
        self.assertEqual(result["provider"], "the_odds_api")
        self.assertEqual(result["fallback_provider"], "betfair")

    def test_explicit_betfair_does_not_call_primary(self):
        from pipeline import odds

        with mock.patch.dict(
            os.environ,
            {"FOOTY_TIPPER_LIVE_ODDS_PROVIDER": "betfair"},
            clear=True,
        ), mock.patch(
            "pipeline.common.odds.the_odds_api.snapshot_live_odds"
        ) as primary, mock.patch(
            "pipeline.common.odds.betfair.snapshot_live_odds",
            return_value={
                "status": "completed",
                "provider": "betfair",
                "games_updated": 8,
            },
        ):
            result = odds._snapshot_live_odds(pathlib.Path("/tmp/odds.sqlite"))
        primary.assert_not_called()
        self.assertEqual(result["provider"], "betfair")


if __name__ == "__main__":
    unittest.main()
