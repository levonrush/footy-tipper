import unittest

from pipeline.common.nrl_data.draw import (
    FIXTURE_CACHE_COLUMNS,
    build_game_id,
    draw_to_bye_rows,
    draw_to_fixture_rows,
    local_wallclock_epoch,
    parse_kickoff_utc,
    venue_timezone,
)


def _fixture(home_id, away_id, home_nick, away_nick, state, kickoff, venue, city,
             home_score=None, away_score=None):
    return {
        "type": "Match",
        "roundTitle": "Round 19",
        "matchState": state,
        "venue": venue,
        "venueCity": city,
        "matchCentreUrl": "/draw/nrl-premiership/2026/round-19/a-v-b/",
        "homeTeam": {"teamId": home_id, "nickName": home_nick, "score": home_score},
        "awayTeam": {"teamId": away_id, "nickName": away_nick, "score": away_score},
        "clock": {"kickOffTimeLong": kickoff},
    }


def _payload():
    return {
        "selectedRoundId": 19,
        "fixtures": [
            _fixture(500023, 500032, "Wests Tigers", "Warriors", "FullTime",
                     "2026-07-10T10:00:00Z", "Campbelltown Sports Stadium", "Sydney",
                     home_score=6, away_score=34),
            _fixture(500021, 500004, "Storm", "Titans", "Upcoming",
                     "2026-07-12T06:00:00Z", "AAMI Park", "Melbourne"),
        ],
        "byes": [
            {"roundTitle": "Round 19", "type": "Bye", "teamNickName": "Broncos"},
        ],
    }


class GameIdTests(unittest.TestCase):
    def test_matches_feed_formula(self):
        # verified against all ~3,800 cached feed fixtures (0 mismatches)
        self.assertEqual(build_game_id(2026, 19, 1), 20261111910)
        self.assertEqual(build_game_id(2012, 5, 1), 20121110510)
        self.assertEqual(build_game_id(2008, 1, 6), 20081110160)


class DrawParsingTests(unittest.TestCase):
    def test_fixture_rows_schema_and_values(self):
        rows = draw_to_fixture_rows(_payload(), 2026, {})
        self.assertEqual(len(rows), 2)

        first = rows[0]
        for column in FIXTURE_CACHE_COLUMNS:
            self.assertIn(column, first)

        self.assertEqual(first["game_id"], 20261111910.0)
        self.assertEqual(first["game_state_name"], "Final")
        self.assertEqual(first["team_home"], "Wests Tigers")
        self.assertEqual(first["team_away"], "New Zealand Warriors")
        self.assertEqual(first["team_final_score_home"], 6.0)
        self.assertEqual(first["team_final_score_away"], 34.0)
        self.assertIsNone(first["team_head_to_head_odds_home"])

        second = rows[1]
        self.assertEqual(second["game_state_name"], "Pre Game")
        self.assertEqual(second["team_home"], "Melbourne Storm")
        self.assertEqual(second["team_final_score_home"], 0.0)
        self.assertEqual(second["game_number"], 2.0)

    def test_game_number_is_kickoff_order(self):
        payload = _payload()
        payload["fixtures"].reverse()  # payload order no longer chronological
        rows = draw_to_fixture_rows(payload, 2026, {})
        self.assertEqual(rows[0]["team_home"], "Wests Tigers")
        self.assertEqual(rows[0]["game_number"], 1.0)

    def test_unknown_match_state_passes_through(self):
        payload = _payload()
        payload["fixtures"][0]["matchState"] = "SecondHalf"
        rows = draw_to_fixture_rows(payload, 2026, {})
        self.assertEqual(rows[0]["game_state_name"], "SecondHalf")

    def test_bye_rows(self):
        byes = draw_to_bye_rows(_payload(), 2026)
        self.assertEqual(
            byes,
            [{"competition_year": 2026, "round_id": 19, "team": "Brisbane Broncos"}],
        )


class KickoffEpochTests(unittest.TestCase):
    def test_utc_epoch(self):
        kickoff = parse_kickoff_utc("2026-07-10T10:00:00Z")
        self.assertEqual(kickoff.timestamp(), 1783677600.0)

    def test_local_wallclock_epoch_matches_feed_convention(self):
        kickoff = parse_kickoff_utc("2026-07-10T10:00:00Z")
        # feed stored venue-local wall clock as-if-UTC: Sydney in July = +10h
        self.assertEqual(
            local_wallclock_epoch(kickoff, "Australia/Sydney"), 1783713600.0
        )

    def test_dst_offsets(self):
        # March: Sydney observes DST (+11), Brisbane does not (+10)
        kickoff = parse_kickoff_utc("2026-03-06T08:00:00Z")
        sydney = local_wallclock_epoch(kickoff, "Australia/Sydney")
        brisbane = local_wallclock_epoch(kickoff, "Australia/Brisbane")
        self.assertEqual(sydney - kickoff.timestamp(), 11 * 3600)
        self.assertEqual(brisbane - kickoff.timestamp(), 10 * 3600)

    def test_venue_timezone_fallbacks(self):
        lookup = {"aami park": "Australia/Melbourne"}
        self.assertEqual(venue_timezone("AAMI Park", None, lookup), "Australia/Melbourne")
        self.assertEqual(venue_timezone("Unknown Oval", "Las Vegas", lookup), "America/Los_Angeles")
        self.assertEqual(venue_timezone("Unknown Oval", None, lookup), "Australia/Sydney")


if __name__ == "__main__":
    unittest.main()
