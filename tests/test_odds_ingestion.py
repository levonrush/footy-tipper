import datetime as dt
import sqlite3
import unittest

from pipeline.common.nrl_data.cache_writer import update_fixture_odds
from pipeline.common.odds import store as odds_store
from pipeline.common.odds.aussportsbetting import (
    _match_game,
    _fixture_index,
    _snapshot_values,
)
from pipeline.common.odds.betfair import (
    _classify_market,
    _mid_price,
    collect_snapshots,
)
from pipeline.common.odds.team_names import canonical_team


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
        key = ("Melbourne Storm", "Gold Coast Titans", "2026-07-12")
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


if __name__ == "__main__":
    unittest.main()
