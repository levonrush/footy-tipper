import sqlite3
import unittest
from unittest.mock import patch

from pipeline.common.nrl_data import refresh, store
from pipeline.common.nrl_data.web import FetchConfig


class FinalMatchCentreRefreshTests(unittest.TestCase):
    def setUp(self):
        self.con = sqlite3.connect(":memory:")
        store.ensure_tables(self.con)
        self.game_id = 20261112110
        self.url = "/draw/nrl-premiership/2026/round-21/eels-v-panthers/"
        store.upsert_match_bundle(
            self.con,
            {
                "game_id": self.game_id,
                "team_stats": [
                    {
                        "side": "home",
                        "stat_name": "completion_rate",
                        "value": 75.0,
                    }
                ],
                "player_stats": [],
                "context": {},
                "officials": [],
                "source_url": f"https://www.nrl.com{self.url}",
            },
            competition_year=2026,
            round_id=21,
            team_home="Parramatta Eels",
            team_away="Penrith Panthers",
        )

    def tearDown(self):
        self.con.close()

    def _fixture(self):
        return {
            "game_id": self.game_id,
            "round_id": 21,
            "game_state_name": "Final",
            "match_centre_url": self.url,
            "team_home": "Parramatta Eels",
            "team_away": "Penrith Panthers",
        }

    def _final_bundle(self):
        return {
            "game_id": self.game_id,
            "team_stats": [
                {"side": "home", "stat_name": "possession_pct", "value": 48.0},
                {"side": "away", "stat_name": "possession_pct", "value": 52.0},
            ],
            "player_stats": [
                {
                    "side": "home",
                    "player_id": 1,
                    "player_name": "Home Player",
                    "jersey_number": 1,
                    "position": "Fullback",
                    "tackles_made": 8.0,
                },
                {
                    "side": "away",
                    "player_id": 2,
                    "player_name": "Away Player",
                    "jersey_number": 1,
                    "position": "Fullback",
                    "tackles_made": 6.0,
                },
            ],
            "context": {},
            "officials": [],
            "source_url": f"https://www.nrl.com{self.url}",
        }

    def test_final_refetches_when_only_pregame_team_summary_exists(self):
        self.assertIn(self.game_id, store.games_with_team_stats(self.con))
        self.assertNotIn(self.game_id, store.games_with_player_stats(self.con))

        with (
            patch.object(refresh, "fetch_match_centre", return_value={}),
            patch.object(
                refresh, "parse_match_centre", return_value=self._final_bundle()
            ) as parse,
        ):
            pages, errors, _ = refresh._fetch_match_centres(
                session=None,
                config=FetchConfig(),
                con=self.con,
                fixture_rows=[self._fixture()],
                season=2026,
                only_missing=True,
            )

        self.assertEqual(pages, 1)
        self.assertEqual(errors, [])
        parse.assert_called_once()
        self.assertIn(self.game_id, store.games_with_player_stats(self.con))
        self.assertEqual(
            self.con.execute(
                "SELECT COUNT(*) FROM match_team_stats "
                "WHERE game_id = ? AND stat_name = 'completion_rate'",
                (self.game_id,),
            ).fetchone()[0],
            0,
        )

    def test_final_with_player_stats_is_not_fetched_again(self):
        store.upsert_match_bundle(
            self.con,
            self._final_bundle(),
            competition_year=2026,
            round_id=21,
            team_home="Parramatta Eels",
            team_away="Penrith Panthers",
        )

        with patch.object(refresh, "fetch_match_centre") as fetch:
            pages, errors, _ = refresh._fetch_match_centres(
                session=None,
                config=FetchConfig(),
                con=self.con,
                fixture_rows=[self._fixture()],
                season=2026,
                only_missing=True,
            )

        self.assertEqual(pages, 0)
        self.assertEqual(errors, [])
        fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
