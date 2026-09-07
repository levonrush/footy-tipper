import unittest

from pipeline.common import rounds


class RoundStageTests(unittest.TestCase):
    def test_every_round_name_the_feed_actually_emits(self):
        """These five shapes are the entire 2008-2026 vocabulary in the DB."""
        self.assertEqual(rounds.round_stage("Round 1"), rounds.REGULAR)
        self.assertEqual(rounds.round_stage("Round 27"), rounds.REGULAR)
        self.assertEqual(rounds.round_stage("Finals Week 1"), rounds.FINALS_WEEK_1)
        self.assertEqual(rounds.round_stage("Finals Week 2"), rounds.FINALS_WEEK_2)
        self.assertEqual(rounds.round_stage("Finals Week 3"), rounds.PRELIMINARY)
        self.assertEqual(rounds.round_stage("Grand Final"), rounds.GRAND_FINAL)

    def test_lineup_article_vocabulary(self):
        """`lineups/ingest.parse_round_name` emits these, and they must agree."""
        self.assertEqual(rounds.round_stage("Qualifying Final"), rounds.FINALS_WEEK_1)
        self.assertEqual(rounds.round_stage("Elimination Final"), rounds.FINALS_WEEK_1)
        self.assertEqual(rounds.round_stage("Semi Final"), rounds.FINALS_WEEK_2)
        self.assertEqual(rounds.round_stage("Preliminary Final"), rounds.PRELIMINARY)

    def test_matching_is_case_and_whitespace_insensitive(self):
        self.assertEqual(rounds.round_stage("  GRAND   FINAL "), rounds.GRAND_FINAL)
        self.assertEqual(rounds.round_stage("finals week 2"), rounds.FINALS_WEEK_2)

    def test_unrecognised_finals_name_resolves_to_finals_not_regular(self):
        """Under-hyping an email is recoverable; accumulating a ladder is not."""
        self.assertTrue(rounds.is_finals("Wildcard Final"))

    def test_bare_round_name_stays_regular_without_a_season_length(self):
        """The draw synthesises `Round N` when `roundTitle` is missing.

        With no `last_regular_round` there is nothing to contradict the name, so
        the classifier must not start inventing finals from round numbers alone.
        """
        self.assertEqual(rounds.round_stage("Round 28", round_id=28), rounds.REGULAR)

    def test_round_id_overrides_a_synthesised_name_when_season_length_is_known(self):
        self.assertEqual(
            rounds.round_stage("Round 28", round_id=28, last_regular_round=27),
            rounds.FINALS_WEEK_1,
        )
        self.assertEqual(
            rounds.round_stage("Round 31", round_id=31, last_regular_round=27),
            rounds.GRAND_FINAL,
        )
        self.assertEqual(
            rounds.round_stage("Round 27", round_id=27, last_regular_round=27),
            rounds.REGULAR,
        )

    def test_shortened_seasons_are_not_misclassified_by_number(self):
        """2020 played finals at rounds 21-24; a fixed `> 27` rule would miss them."""
        self.assertEqual(
            rounds.round_stage("Round 21", round_id=21, last_regular_round=20),
            rounds.FINALS_WEEK_1,
        )
        self.assertEqual(
            rounds.round_stage("Round 24", round_id=24, last_regular_round=20),
            rounds.GRAND_FINAL,
        )

    def test_missing_and_blank_names(self):
        self.assertEqual(rounds.round_stage(None), rounds.REGULAR)
        self.assertEqual(rounds.round_stage(""), rounds.REGULAR)
        self.assertEqual(rounds.round_stage("   "), rounds.REGULAR)
        self.assertEqual(
            rounds.round_stage(None, round_id=29, last_regular_round=27),
            rounds.FINALS_WEEK_2,
        )


class StageMetadataTests(unittest.TestCase):
    def test_week_numbers(self):
        self.assertEqual(rounds.stage_week(rounds.FINALS_WEEK_1), 1)
        self.assertEqual(rounds.stage_week(rounds.FINALS_WEEK_2), 2)
        self.assertEqual(rounds.stage_week(rounds.PRELIMINARY), 3)
        self.assertEqual(rounds.stage_week(rounds.GRAND_FINAL), 4)
        self.assertIsNone(rounds.stage_week(rounds.REGULAR))

    def test_labels_round_trip_through_the_classifier(self):
        for stage in rounds.FINALS_STAGES:
            self.assertEqual(rounds.round_stage(rounds.stage_feed_label(stage)), stage)
            self.assertEqual(rounds.round_stage(rounds.stage_display_name(stage)), stage)

    def test_display_names_use_the_footy_vocabulary(self):
        self.assertEqual(rounds.stage_display_name(rounds.FINALS_WEEK_2), "Semi Finals")
        self.assertEqual(
            rounds.stage_display_name(rounds.PRELIMINARY), "Preliminary Finals"
        )
        self.assertIsNone(rounds.stage_display_name(rounds.REGULAR))

    def test_slugs(self):
        self.assertEqual(rounds.round_slug("Round 24", 24), "round-24")
        self.assertEqual(rounds.round_slug("Finals Week 1", 28), "finals-week-1")
        self.assertEqual(rounds.round_slug("Finals Week 3", 30), "preliminary-finals")
        self.assertEqual(rounds.round_slug("Grand Final", 31), "grand-final")
        # A missing name still produces a usable filename.
        self.assertEqual(rounds.round_slug(None, 24), "round-24")
        self.assertEqual(rounds.round_slug(None, None), "round")


class LastRegularRoundTests(unittest.TestCase):
    def test_modern_season(self):
        season = [(n, f"Round {n}") for n in range(1, 28)] + [
            (28, "Finals Week 1"),
            (29, "Finals Week 2"),
            (30, "Finals Week 3"),
            (31, "Grand Final"),
        ]
        self.assertEqual(rounds.last_regular_round(season), 27)

    def test_shortened_season(self):
        season = [(n, f"Round {n}") for n in range(1, 21)] + [(21, "Finals Week 1")]
        self.assertEqual(rounds.last_regular_round(season), 20)

    def test_no_regular_rounds(self):
        self.assertIsNone(rounds.last_regular_round([(28, "Finals Week 1")]))
        self.assertIsNone(rounds.last_regular_round([]))

    def test_unparseable_round_ids_are_skipped(self):
        self.assertEqual(
            rounds.last_regular_round([(1, "Round 1"), (None, "Round 2"), ("x", "R3")]),
            1,
        )


if __name__ == "__main__":
    unittest.main()
