"""Rebuilding feed_cache_ladders from cached data, without the network.

The historical XML feed stored END-OF-SEASON values in the form columns on
every round row, so a round-2 row leaked how the season finished. Those columns
are declared predictors and refresh_season only ever rewrites the current
season, so the leaked values survived into training_data. rebuild_ladder_cache
re-derives them as-of-round from data already in SQLite.

The draw JSON carries byes explicitly but never persisted them, so the rebuild
has to infer them. These tests pin that inference and the leak fix.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from pipeline.common.nrl_data.refresh import (
    derive_bye_rows,
    load_cached_fixture_rows,
    rebuild_ladder_cache,
)


def _fixture(round_id, game_number, home, away, home_score, away_score,
             round_name=None, state="Final"):
    base_epoch = 1735689600  # 2025-01-01T00:00Z
    return {
        "game_id": float(f"2025111{round_id:02d}{game_number}0"),
        "competition_year": 2025.0,
        "round_id": float(round_id),
        "round_name": round_name or f"Round {round_id}",
        "game_number": float(game_number),
        "game_state_name": state,
        "start_time": float(base_epoch + round_id * 7 * 86400 + 19 * 3600),
        "team_home": home,
        "team_away": away,
        "team_final_score_home": float(home_score),
        "team_final_score_away": float(away_score),
    }


class DeriveByeRowsTests(unittest.TestCase):
    def test_team_absent_from_a_regular_round_has_a_bye(self):
        fixtures = [
            _fixture(1, 1, "A", "B", 20, 10),
            _fixture(1, 2, "C", "D", 12, 6),
            # round 2: D does not play
            _fixture(2, 1, "A", "C", 18, 14),
        ]
        byes = derive_bye_rows(fixtures)
        self.assertEqual(byes, [{"round_id": 2, "team": "B"},
                                {"round_id": 2, "team": "D"}])

    def test_finals_rounds_never_produce_byes(self):
        # A team missing from a finals round has been eliminated, not byed.
        fixtures = [
            _fixture(1, 1, "A", "B", 20, 10),
            _fixture(1, 2, "C", "D", 12, 6),
            _fixture(2, 1, "A", "C", 18, 14, round_name="Finals Week 1"),
        ]
        self.assertEqual(derive_bye_rows(fixtures), [])

    def test_full_round_produces_no_byes(self):
        fixtures = [_fixture(1, 1, "A", "B", 20, 10), _fixture(1, 2, "C", "D", 12, 6)]
        self.assertEqual(derive_bye_rows(fixtures), [])


class RebuildLadderCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "test.sqlite"
        self.fixtures = [
            _fixture(1, 1, "A", "B", 20, 10),
            _fixture(1, 2, "C", "D", 12, 6),
            _fixture(2, 1, "A", "C", 30, 6),
            _fixture(2, 2, "B", "D", 14, 12),
            _fixture(3, 1, "A", "D", 24, 8),
            _fixture(3, 2, "B", "C", 10, 22),
        ]
        con = sqlite3.connect(self.db)
        cols = list(self.fixtures[0])
        con.execute(
            f"CREATE TABLE feed_cache_fixtures ({', '.join(f'{c} REAL' if c not in {'round_name', 'game_state_name', 'team_home', 'team_away'} else f'{c} TEXT' for c in cols)})"
        )
        con.executemany(
            f"INSERT INTO feed_cache_fixtures ({', '.join(cols)}) "
            f"VALUES ({', '.join('?' for _ in cols)})",
            [tuple(f[c] for c in cols) for f in self.fixtures],
        )
        # A frozen ladder carrying end-of-season values on every round: the
        # exact shape of the leak this rebuild exists to remove.
        con.execute(
            "CREATE TABLE feed_cache_ladders (competition_year REAL, round_id REAL, "
            "team TEXT, season_form TEXT, current_streak TEXT, players_used REAL)"
        )
        con.executemany(
            "INSERT INTO feed_cache_ladders VALUES (?, ?, ?, ?, ?, ?)",
            [(2025.0, float(r), t, "WWW", "3W", 29.0)
             for r in (1, 2, 3) for t in ("A", "B", "C", "D")],
        )
        con.commit()
        con.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_cached_fixture_rows_round_trips(self):
        con = sqlite3.connect(self.db)
        try:
            rows = load_cached_fixture_rows(con, 2025)
        finally:
            con.close()
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]["team_home"], "A")

    def test_rebuild_replaces_frozen_values_with_as_of_round_values(self):
        result = rebuild_ladder_cache(self.db, 2025, 2025)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["seasons"], {2025: 12})

        con = sqlite3.connect(self.db)
        try:
            rows = dict(
                con.execute(
                    "SELECT round_id, season_form FROM feed_cache_ladders "
                    "WHERE team = 'A' ORDER BY round_id"
                ).fetchall()
            )
        finally:
            con.close()

        # A wins every game, so the form string grows one character per round
        # instead of holding the end-of-season value from round 1.
        self.assertEqual(rows[1.0], "W")
        self.assertEqual(rows[2.0], "WW")
        self.assertEqual(rows[3.0], "WWW")

    def test_rebuild_is_idempotent(self):
        first = rebuild_ladder_cache(self.db, 2025, 2025)
        second = rebuild_ladder_cache(self.db, 2025, 2025)
        self.assertEqual(first["seasons"], second["seasons"])

        con = sqlite3.connect(self.db)
        try:
            count = con.execute("SELECT COUNT(*) FROM feed_cache_ladders").fetchone()[0]
        finally:
            con.close()
        self.assertEqual(count, 12)

    def test_missing_season_is_reported_not_raised(self):
        result = rebuild_ladder_cache(self.db, 2019, 2019)
        self.assertEqual(result["status"], "completed_with_errors")
        self.assertEqual(result["seasons"], {})
        self.assertIn("no cached fixtures", result["errors"][0])


if __name__ == "__main__":
    unittest.main()
