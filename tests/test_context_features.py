import sqlite3
import unittest
from pathlib import Path

import pandas as pd

from pipeline.common.nrl_data import features as ctx
from pipeline.common.lineups.player_form import compute_lineup_player_form_features


def _seed_db(path: Path) -> None:
    con = sqlite3.connect(str(path))
    con.executescript(
        """
        CREATE TABLE feed_cache_fixtures (
            game_id REAL, competition_year REAL, round_id REAL,
            game_state_name TEXT, start_time REAL, start_time_utc REAL,
            venue_name TEXT, team_home TEXT, team_away TEXT
        );
        CREATE TABLE match_team_stats (
            game_id INTEGER, competition_year INTEGER, round_id INTEGER,
            team TEXT, side TEXT, stat_name TEXT, value REAL,
            source_url TEXT, ingested_at_utc TEXT
        );
        CREATE TABLE match_officials (
            game_id INTEGER, competition_year INTEGER, round_id INTEGER,
            role TEXT, official_name TEXT, profile_id INTEGER, ingested_at_utc TEXT
        );
        CREATE TABLE match_context (
            game_id INTEGER PRIMARY KEY, competition_year INTEGER, round_id INTEGER,
            weather_label TEXT, ground_condition TEXT, attendance INTEGER,
            source_url TEXT, ingested_at_utc TEXT
        );
        CREATE TABLE venue_locations (
            venue_name TEXT PRIMARY KEY, city TEXT, latitude REAL,
            longitude REAL, timezone TEXT
        );
        CREATE TABLE match_player_stats (
            game_id INTEGER, competition_year INTEGER, round_id INTEGER,
            team TEXT, side TEXT, player_id INTEGER, player_key TEXT,
            player_name TEXT, jersey_number INTEGER, position TEXT,
            fantasy_points_total REAL, all_run_metres REAL, tackles_made REAL,
            errors REAL, tries REAL, try_assists REAL, line_breaks REAL,
            line_break_assists REAL, minutes_played REAL, ingested_at_utc TEXT
        );
        """
    )
    base = 1750000000
    games = [
        (101, 1, base, "Suncorp Stadium", "A", "B", "Final"),
        (102, 2, base + 7 * 86400, "AAMI Park", "A", "B", "Final"),
        (103, 3, base + 14 * 86400, "Suncorp Stadium", "A", "B", "Pre Game"),
    ]
    for gid, rnd, start, venue, home, away, state in games:
        con.execute(
            "INSERT INTO feed_cache_fixtures VALUES (?, 2026, ?, ?, ?, ?, ?, ?, ?)",
            (float(gid), float(rnd), state, float(start + 10 * 3600), float(start), venue, home, away),
        )

    stats = [
        (101, "A", "home", "all_run_metres", 1500.0),
        (101, "B", "away", "all_run_metres", 1300.0),
        (101, "A", "home", "penalties_conceded", 4.0),
        (101, "B", "away", "penalties_conceded", 6.0),
        (101, "A", "home", "sin_bins", 1.0),
        (101, "B", "away", "sin_bins", 0.0),
        (102, "A", "home", "all_run_metres", 1700.0),
        (102, "B", "away", "all_run_metres", 1400.0),
    ]
    for gid, team, side, stat, value in stats:
        con.execute(
            "INSERT INTO match_team_stats VALUES (?, 2026, 1, ?, ?, ?, ?, '', '')",
            (gid, team, side, stat, value),
        )

    con.execute(
        "INSERT INTO match_officials VALUES (101, 2026, 1, 'Referee', 'Ref One', 1, '')"
    )
    con.execute(
        "INSERT INTO match_officials VALUES (102, 2026, 2, 'Referee', 'Ref One', 1, '')"
    )
    con.execute(
        "INSERT INTO match_context VALUES (101, 2026, 1, 'Rain', 'Wet', 15000, '', '')"
    )
    con.execute(
        "INSERT INTO venue_locations VALUES "
        "('Suncorp Stadium', 'Brisbane', -27.4648, 153.0095, 'Australia/Brisbane'), "
        "('AAMI Park', 'Melbourne', -37.8250, 144.9840, 'Australia/Melbourne')"
    )

    players = [
        # (game, side, pid, key, name, jersey, fantasy, metres)
        (101, "home", 11, "alpha_one", "Alpha One", 1, 30.0, 150.0),
        (101, "home", 12, "alpha_two", "Alpha Two", 7, 20.0, 90.0),
        (101, "away", 21, "beta_one", "Beta One", 1, 10.0, 80.0),
        (102, "home", 11, "alpha_one", "Alpha One", 1, 50.0, 200.0),
        (102, "home", 12, "alpha_two", "Alpha Two", 7, 24.0, 100.0),
        (102, "away", 21, "beta_one", "Beta One", 1, 14.0, 90.0),
    ]
    for gid, side, pid, key, name, jersey, fantasy, metres in players:
        con.execute(
            "INSERT INTO match_player_stats VALUES "
            "(?, 2026, 1, ?, ?, ?, ?, ?, ?, 'X', ?, ?, 30.0, 1.0, 0.0, 0.0, 0.0, 0.0, 80.0, '')",
            (gid, "A" if side == "home" else "B", side, pid, key, name, jersey,
             fantasy, metres),
        )
    con.commit()
    con.close()


class ContextFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile

        cls._dir = tempfile.TemporaryDirectory()
        cls.db_path = Path(cls._dir.name) / "test.sqlite"
        _seed_db(cls.db_path)
        cls.matches = pd.DataFrame({"game_id": [101.0, 102.0, 103.0]})
        cls.frame = ctx.build_match_context_features(
            cls.db_path,
            cls.matches,
            team_bases_csv=Path("data/reference/team_home_venues.csv"),
        )

    @classmethod
    def tearDownClass(cls):
        cls._dir.cleanup()

    def _row(self, game_id):
        return self.frame[self.frame["game_id"] == game_id].iloc[0]

    def test_team_form_is_leak_safe(self):
        # game 101 is each team's first game: no prior form
        first = self._row(101.0)
        self.assertEqual(first["form_features_missing_home"], 1.0)
        # game 102 sees exactly game 101's value (single prior observation)
        second = self._row(102.0)
        self.assertAlmostEqual(second["form_all_run_metres_home"], 1500.0)
        self.assertAlmostEqual(second["form_all_run_metres_away"], 1300.0)
        self.assertAlmostEqual(second["form_all_run_metres_delta"], 200.0)
        # upcoming game 103 blends games 101 and 102 with halflife 5
        third = self._row(103.0)
        self.assertGreater(third["form_all_run_metres_home"], 1500.0)
        self.assertLess(third["form_all_run_metres_home"], 1700.0)

    def test_referee_rates_shifted(self):
        first = self._row(101.0)
        self.assertEqual(first["referee_name"], "Ref One")
        self.assertEqual(first["ref_games_officiated"], 0.0)
        self.assertTrue(pd.isna(first["ref_penalty_rate_ewma"]))
        second = self._row(102.0)
        self.assertEqual(second["ref_games_officiated"], 1.0)
        self.assertAlmostEqual(second["ref_penalty_rate_ewma"], 10.0)  # 4+6 in game 101
        self.assertAlmostEqual(second["ref_sin_bin_rate_ewma"], 1.0)

    def test_weather_labels(self):
        first = self._row(101.0)
        self.assertEqual(first["wx_wet"], 1.0)  # 'Rain' label
        self.assertEqual(first["ground_condition"], "Wet")

    def test_travel_distances(self):
        # game 102 at AAMI Park: Storm-adjacent teams not in fixture, so use
        # whatever bases exist; A/B are not real teams -> travel missing
        second = self._row(102.0)
        self.assertEqual(second["travel_missing"], 1.0)


class PlayerFormTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile

        cls._dir = tempfile.TemporaryDirectory()
        cls.db_path = Path(cls._dir.name) / "test.sqlite"
        _seed_db(cls.db_path)
        cls.frame = compute_lineup_player_form_features(
            cls.db_path, pd.DataFrame({"game_id": [101.0, 102.0]})
        )

    @classmethod
    def tearDownClass(cls):
        cls._dir.cleanup()

    def _row(self, game_id):
        return self.frame[self.frame["game_id"] == game_id].iloc[0]

    def test_first_game_has_no_form(self):
        first = self._row(101.0)
        self.assertEqual(first["lineup_form_missing_home"], 1.0)
        self.assertEqual(first["lineup_form_coverage_home"], 0.0)

    def test_second_game_uses_first_game_form(self):
        second = self._row(102.0)
        self.assertEqual(second["lineup_form_missing_home"], 0.0)
        self.assertEqual(second["lineup_form_coverage_home"], 1.0)
        # both players' game-101 fantasy: (30 + 20) / 2
        self.assertAlmostEqual(second["lineup_form_fantasy_home"], 25.0)
        # spine = jerseys 1 and 7 here, same two players
        self.assertAlmostEqual(second["lineup_spine_form_fantasy_home"], 25.0)
        self.assertAlmostEqual(second["lineup_form_fantasy_away"], 10.0)
        self.assertAlmostEqual(second["lineup_form_fantasy_delta"], 15.0)


if __name__ == "__main__":
    unittest.main()
