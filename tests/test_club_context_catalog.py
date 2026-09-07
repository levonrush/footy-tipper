import datetime as dt
import json
import unittest
from pathlib import Path


CATALOG_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "reference"
    / "club_context_events.json"
)
FORBIDDEN_POST_MATCH_ARROW_URL = (
    "https://www.nrl.com/news/2026/06/12/"
    "the-ultimate-team-mate-rabbitohs-pay-tribute-to-arrow-after-emotion-charged-win/"
)


def _as_utc(value):
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        dt.timezone.utc
    )


class ClubContextCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

    def test_leadership_census_declared_count_years_and_keys_are_complete(self):
        census = self.catalog["leadership_change_census"]
        coverage = self.catalog["coverage"]

        self.assertEqual(len(census), coverage["leadership_handover_count"])
        self.assertEqual(len(census), 32)
        self.assertEqual(len({row["event_key"] for row in census}), len(census))
        self.assertTrue(
            all(
                coverage["census_start_year"]
                <= row["competition_year"]
                <= coverage["census_end_year"]
                for row in census
            )
        )

        represented = {row["competition_year"] for row in census}
        expected_missing = set(
            coverage["seasons_with_no_effective_midseason_handover"]
        )
        all_years = set(
            range(coverage["census_start_year"], coverage["census_end_year"] + 1)
        )
        self.assertEqual(all_years - represented, expected_missing)
        self.assertEqual(expected_missing, {2016, 2025})

    def test_arrow_announcement_tribute_and_milestones_are_separate_pulses(self):
        arrow = {
            row["event_key"]: row
            for row in self.catalog["events"]
            if "arrow" in row["event_key"]
        }
        expected = {
            "human-rabbitohs-arrow-retirement-announcement-2026",
            "tribute-rabbitohs-arrow-whiteout-match-2026",
            "milestone-rabbitohs-arrow-game-99-2026",
            "milestone-rabbitohs-arrow-game-100-2026",
        }
        self.assertEqual(set(arrow), expected)
        self.assertEqual(
            arrow["human-rabbitohs-arrow-retirement-announcement-2026"]["phase"],
            "announcement",
        )
        self.assertEqual(
            arrow["tribute-rabbitohs-arrow-whiteout-match-2026"]["phase"],
            "tribute",
        )
        self.assertEqual(
            arrow["milestone-rabbitohs-arrow-game-99-2026"]["phase"],
            "milestone",
        )
        self.assertEqual(
            arrow["milestone-rabbitohs-arrow-game-100-2026"]["phase"],
            "milestone",
        )

        announcement = arrow[
            "human-rabbitohs-arrow-retirement-announcement-2026"
        ]
        tribute = arrow["tribute-rabbitohs-arrow-whiteout-match-2026"]
        game_99 = arrow["milestone-rabbitohs-arrow-game-99-2026"]
        game_100 = arrow["milestone-rabbitohs-arrow-game-100-2026"]
        self.assertLess(
            _as_utc(announcement["expires_at_utc"]),
            _as_utc(tribute["effective_from_utc"]),
        )
        self.assertLess(
            _as_utc(tribute["expires_at_utc"]),
            _as_utc(game_99["effective_from_utc"]),
        )
        self.assertLess(
            _as_utc(game_99["expires_at_utc"]),
            _as_utc(game_100["effective_from_utc"]),
        )

    def test_arrow_sources_are_pre_match_and_exclude_post_match_claims(self):
        arrow = [
            row for row in self.catalog["events"] if "arrow" in row["event_key"]
        ]
        source_urls = {
            source["url"]
            for event in arrow
            for source in event.get("sources", [])
        }

        self.assertNotIn(FORBIDDEN_POST_MATCH_ARROW_URL, source_urls)
        for event in arrow:
            effective = _as_utc(event["effective_from_utc"])
            self.assertLessEqual(_as_utc(event["known_at_utc"]), effective)
            for source in event.get("sources", []):
                observed = source.get("published_at_utc") or source.get(
                    "first_observed_at_utc"
                )
                self.assertIsNotNone(observed)
                self.assertLessEqual(_as_utc(observed), effective)

    def test_catalogue_stores_facts_and_links_not_article_content(self):
        serialized = json.dumps(self.catalog).lower()
        for forbidden_key in ('"article_body"', '"body"', '"snippet"'):
            self.assertNotIn(forbidden_key, serialized)
        self.assertTrue(self.catalog["scope"]["facts_and_links_only"])
        self.assertFalse(self.catalog["scope"]["article_bodies_stored"])


if __name__ == "__main__":
    unittest.main()
