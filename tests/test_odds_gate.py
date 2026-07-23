import datetime as dt
import sqlite3
import tempfile
import unittest
from pathlib import Path

from pipeline.ops.odds_gate import current_round_odds_coverage


class OddsCoverageGateTests(unittest.TestCase):
    def _database(self, root: Path, *, games: int = 8) -> Path:
        path = root / "runtime.sqlite"
        with sqlite3.connect(path) as con:
            con.executescript(
                """
                CREATE TABLE feed_cache_fixtures (
                    game_id INTEGER,
                    competition_year INTEGER,
                    round_id INTEGER,
                    round_name TEXT,
                    game_number INTEGER,
                    game_state_name TEXT,
                    start_time REAL
                );
                CREATE TABLE odds_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id INTEGER,
                    source TEXT,
                    snapshot_kind TEXT,
                    snapshot_time_utc TEXT,
                    h2h_odds_home REAL,
                    h2h_odds_away REAL,
                    line_amount_home REAL,
                    line_odds_home REAL,
                    line_odds_away REAL,
                    total_line REAL,
                    total_over_odds REAL,
                    total_under_odds REAL
                );
                """
            )
            con.executemany(
                """
                INSERT INTO feed_cache_fixtures
                    (game_id, competition_year, round_id, round_name,
                     game_number, game_state_name, start_time)
                VALUES (?, 2026, 21, 'Round 21', ?, 'Pre Game', ?)
                """,
                [(game_id, game_id, 1_784_836_200 + game_id) for game_id in range(1, games + 1)],
            )
        return path

    @staticmethod
    def _insert_prices(path: Path, game_ids, snapshot_time, *, home=1.90, away=2.10):
        with sqlite3.connect(path) as con:
            con.executemany(
                """
                INSERT INTO odds_history
                    (game_id, source, snapshot_kind, snapshot_time_utc,
                     h2h_odds_home, h2h_odds_away)
                VALUES (?, 'the_odds_api', 'live', ?, ?, ?)
                """,
                [(game_id, snapshot_time.isoformat(), home, away) for game_id in game_ids],
            )

    def test_seven_of_eight_is_incomplete_and_eight_of_eight_passes(self):
        now = dt.datetime(2026, 7, 23, 8, 0, tzinfo=dt.timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            path = self._database(Path(tmp))
            self._insert_prices(path, range(1, 8), now)

            partial = current_round_odds_coverage(path, now=now)

            self.assertFalse(partial.complete)
            self.assertEqual(partial.covered_games, 7)
            self.assertEqual(partial.total_games, 8)
            self.assertEqual(partial.missing_game_ids, (8,))

            self._insert_prices(path, [8], now)
            complete = current_round_odds_coverage(path, now=now)

            self.assertTrue(complete.complete)
            self.assertEqual(complete.covered_games, 8)

    def test_invalid_or_stale_prices_do_not_count(self):
        now = dt.datetime(2026, 7, 23, 8, 0, tzinfo=dt.timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            path = self._database(Path(tmp), games=2)
            self._insert_prices(path, [1], now, home=1.0, away=2.10)
            self._insert_prices(path, [2], now - dt.timedelta(hours=7))

            coverage = current_round_odds_coverage(path, now=now)

            self.assertFalse(coverage.complete)
            self.assertEqual(coverage.covered_games, 0)
            self.assertEqual(coverage.missing_game_ids, (1,))
            self.assertEqual(coverage.stale_game_ids, (2,))

    def test_fresh_h2h_only_suppresses_other_snapshot_line_and_total(self):
        now = dt.datetime(2026, 7, 23, 8, 0, tzinfo=dt.timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            path = self._database(Path(tmp), games=1)
            with sqlite3.connect(path) as con:
                con.execute(
                    """
                    INSERT INTO odds_history
                        (game_id, source, snapshot_kind, snapshot_time_utc,
                         h2h_odds_home, h2h_odds_away,
                         line_amount_home, line_odds_home, line_odds_away,
                         total_line, total_over_odds, total_under_odds)
                    VALUES
                        (1, 'the_odds_api', 'live', ?, 1.90, 2.10,
                         -6.5, 1.91, 1.91, 42.5, 1.91, 1.91),
                        (1, 'the_odds_api', 'live', ?, 1.95, 2.05,
                         NULL, NULL, NULL, NULL, NULL, NULL)
                    """,
                    (
                        # Still fresh, but from a different atomic bookmaker
                        # snapshot than the latest H2H-only observation.
                        (now - dt.timedelta(hours=1)).isoformat(),
                        now.isoformat(),
                    ),
                )

            coverage = current_round_odds_coverage(path, now=now)

            self.assertTrue(coverage.complete)
            self.assertEqual(coverage.fresh_game_ids, (1,))
            self.assertEqual(coverage.fresh_line_game_ids, ())
            self.assertEqual(coverage.fresh_total_game_ids, ())

    def test_offseason_without_pregame_fixtures_passes_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._database(Path(tmp), games=0)

            coverage = current_round_odds_coverage(path)

            self.assertTrue(coverage.complete)
            self.assertTrue(coverage.no_fixtures)


if __name__ == "__main__":
    unittest.main()
