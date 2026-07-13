import unittest

from pipeline.common.nrl_data.match_centre import (
    coerce_stat_value,
    parse_match_centre,
    slug_stat_title,
    to_snake,
)


def _payload():
    return {
        "match": {
            "matchId": 20261111910,
            "matchState": "FullTime",
            "weather": "Fine",
            "groundConditions": "Good",
            "attendance": 10445,
            "officials": [
                {"firstName": "Wyatt", "lastName": "Raymond", "profileId": 500262, "position": "Referee"},
                {"firstName": "Kasey", "lastName": "Badger", "profileId": 500072, "position": "Touch Judge"},
            ],
            "homeTeam": {
                "nickName": "Wests Tigers",
                "players": [
                    {"playerId": 111, "firstName": "Jahream", "lastName": "Bula",
                     "number": 1, "position": "Fullback"},
                ],
            },
            "awayTeam": {"nickName": "Warriors", "players": []},
            "stats": {
                "groups": [
                    {
                        "title": "Possession & Completions",
                        "stats": [
                            {
                                "title": "Possession %",
                                "homeValue": {"value": 51.0},
                                "awayValue": {"value": 49.0},
                            },
                            {
                                "title": "Time In Possession",
                                "homeValue": {"value": "31:32"},
                                "awayValue": {"value": "30:11"},
                            },
                        ],
                    },
                    {
                        "title": "Attack",
                        "stats": [
                            {
                                "title": "All Run Metres",
                                "homeValue": {"value": 1500},
                                "awayValue": {"value": 1655},
                            }
                        ],
                    },
                ],
                "players": {
                    "homeTeam": [
                        {"playerId": 111, "allRunMetres": 154, "tackleBreaks": 7,
                         "fantasyPointsTotal": 27, "minutesPlayed": 80},
                    ],
                    "awayTeam": [],
                },
            },
        }
    }


class HelperTests(unittest.TestCase):
    def test_to_snake(self):
        self.assertEqual(to_snake("allRunMetres"), "all_run_metres")
        self.assertEqual(to_snake("playTheBallAverageSpeed"), "play_the_ball_average_speed")

    def test_slug_stat_title(self):
        self.assertEqual(slug_stat_title("Possession %"), "possession_pct")
        self.assertEqual(slug_stat_title("Effective Tackle %"), "effective_tackle_pct")
        self.assertEqual(slug_stat_title("All Run Metres"), "all_run_metres")

    def test_coerce_stat_value(self):
        self.assertEqual(coerce_stat_value(51.0), 51.0)
        self.assertEqual(coerce_stat_value("31:32"), 31 * 60 + 32)
        self.assertEqual(coerce_stat_value("87.6%"), 87.6)
        self.assertIsNone(coerce_stat_value("N/A"))
        self.assertIsNone(coerce_stat_value(None))


class ParseMatchCentreTests(unittest.TestCase):
    def test_full_bundle(self):
        bundle = parse_match_centre(_payload(), source_url="https://example/mc")
        self.assertEqual(bundle["game_id"], 20261111910)

        stats = {(s["side"], s["stat_name"]): s["value"] for s in bundle["team_stats"]}
        self.assertEqual(stats[("home", "possession_pct")], 51.0)
        self.assertEqual(stats[("away", "all_run_metres")], 1655.0)
        self.assertEqual(stats[("home", "time_in_possession")], 1892.0)

        self.assertEqual(len(bundle["player_stats"]), 1)
        player = bundle["player_stats"][0]
        self.assertEqual(player["player_id"], 111)
        self.assertEqual(player["player_name"], "Jahream Bula")
        self.assertEqual(player["position"], "Fullback")
        self.assertEqual(player["all_run_metres"], 154.0)
        self.assertEqual(player["fantasy_points_total"], 27.0)

        self.assertEqual(bundle["context"]["weather_label"], "Fine")
        self.assertEqual(bundle["context"]["attendance"], 10445)

        refs = [o for o in bundle["officials"] if o["role"] == "Referee"]
        self.assertEqual(refs[0]["official_name"], "Wyatt Raymond")

    def test_missing_match_returns_none(self):
        self.assertIsNone(parse_match_centre({}))
        self.assertIsNone(parse_match_centre({"match": {"matchId": "not-a-number"}}))


if __name__ == "__main__":
    unittest.main()
