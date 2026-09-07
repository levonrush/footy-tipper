import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import pandas as pd

from pipeline.common.club_context import (
    ArticleSnapshot,
    ConfirmationStatus,
    ContextEvent,
    EntityType,
    EventCategory,
    EventEntity,
    EventPhase,
    EventSource,
    PredictionMode,
    ReviewStatus,
    RightsStatus,
    Sensitivity,
    SourceClass,
    build_context_match_features,
    create_prediction_context_snapshot,
    ensure_context_tables,
    insert_context_event,
    load_prediction_context,
    round_target_at_utc,
    upsert_article_snapshot,
)


FIXTURES = pd.DataFrame(
    [
        {
            "game_id": 101,
            "competition_year": 2026,
            "round_id": 15,
            "team_home": "South Sydney Rabbitohs",
            "team_away": "Brisbane Broncos",
            "start_time_utc": "2026-06-11T09:50:00Z",
        },
        {
            "game_id": 102,
            "competition_year": 2026,
            "round_id": 15,
            "team_home": "Sydney Roosters",
            "team_away": "Melbourne Storm",
            "start_time_utc": "2026-06-14T06:05:00Z",
        },
    ]
)


def _official_article(con, slug, published):
    return upsert_article_snapshot(
        con,
        ArticleSnapshot(
            source_url=f"https://www.nrl.com/news/2026/06/10/{slug}/?utm_source=rss",
            publisher_key="nrl",
            source_name="NRL",
            source_class=SourceClass.OFFICIAL_NRL,
            title=slug.replace("-", " ").title(),
            source_published_at_utc=published,
            first_observed_at_utc=published,
            fetched_at_utc=published,
            rights_status=RightsStatus.FACTS_AND_LINKS,
        ),
    )


def _seed_event(
    con,
    *,
    key,
    team,
    source_id,
    category,
    phase,
    known,
    effective,
    expires,
    sensitive=False,
):
    return insert_context_event(
        con,
        ContextEvent(
            event_key=key,
            category=category,
            phase=phase,
            known_at_utc=known,
            effective_from_utc=effective,
            expires_at_utc=expires,
            factual_summary=f"Confirmed factual context for {team}.",
            confidence=0.9,
            salience=0.8,
            sensitivity=Sensitivity.SENSITIVE if sensitive else Sensitivity.STANDARD,
            confirmation_status=ConfirmationStatus.CONFIRMED,
            review_status=ReviewStatus.APPROVED,
        ),
        sources=[EventSource(source_id)],
        entities=[
            EventEntity(
                entity_type=EntityType.TEAM,
                entity_key=team,
                display_name=team,
                team_key=team,
            )
        ],
    )


def _seed_db(path):
    with closing(sqlite3.connect(path)) as con, con:
        ensure_context_tables(con)
        announcement = _official_article(
            con, "arrow-retirement-announcement", "2026-05-20T03:00:00Z"
        )
        early = _official_article(con, "arrow-tribute", "2026-06-10T03:00:00Z")
        late = _official_article(con, "late-club-crisis", "2026-06-12T02:00:00Z")
        post = _official_article(con, "post-match-quotes", "2026-06-11T13:00:00Z")
        _seed_event(
            con,
            key="arrow-retirement-announcement",
            team="South Sydney Rabbitohs",
            source_id=announcement,
            category=EventCategory.SERIOUS_HUMAN_EVENT,
            phase=EventPhase.ANNOUNCEMENT,
            known="2026-05-20T03:00:00Z",
            effective="2026-05-20T03:00:00Z",
            expires="2026-05-25T00:00:00Z",
            sensitive=True,
        )
        _seed_event(
            con,
            key="arrow-tribute-match",
            team="South Sydney Rabbitohs",
            source_id=early,
            category=EventCategory.TRIBUTE_MILESTONE,
            phase=EventPhase.TRIBUTE,
            known="2026-06-10T03:00:00Z",
            effective="2026-06-11T09:50:00Z",
            expires="2026-06-11T13:00:00Z",
            sensitive=True,
        )
        _seed_event(
            con,
            key="roosters-late-crisis",
            team="Sydney Roosters",
            source_id=late,
            category=EventCategory.CLUB_CRISIS,
            phase=EventPhase.ANNOUNCEMENT,
            known="2026-06-12T02:00:00Z",
            effective="2026-06-12T02:00:00Z",
            expires="2026-06-15T00:00:00Z",
        )
        _seed_event(
            con,
            key="arrow-post-match-comment",
            team="South Sydney Rabbitohs",
            source_id=post,
            category=EventCategory.SERIOUS_HUMAN_EVENT,
            phase=EventPhase.ONGOING,
            known="2026-06-11T13:00:00Z",
            effective="2026-06-11T13:00:00Z",
            expires="2026-06-20T00:00:00Z",
            sensitive=True,
        )


class ClubContextFeatureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "db.sqlite"
        _seed_db(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_target_is_11am_sydney_with_dst(self):
        winter = round_target_at_utc(FIXTURES.to_dict("records"))
        self.assertEqual(winter.isoformat(), "2026-06-11T01:00:00+00:00")
        summer = round_target_at_utc(
            [
                {
                    "competition_year": 2026,
                    "round_id": 1,
                    "start_time_utc": "2026-03-05T09:00:00Z",
                }
            ]
        )
        self.assertEqual(summer.isoformat(), "2026-03-05T00:00:00+00:00")

    def test_historical_cutoff_is_round_wide_and_blocks_late_week_news(self):
        features = build_context_match_features(self.path, FIXTURES).set_index("game_id")
        self.assertEqual(features.loc[101, "club_context_event_count_home"], 1.0)
        self.assertEqual(
            features.loc[101, "club_context_tribute_milestone_count_home"], 1.0
        )
        self.assertEqual(features.loc[101, "club_context_match_pulse_count_home"], 1.0)
        self.assertEqual(features.loc[101, "club_context_sensitive_count_home"], 1.0)
        self.assertEqual(features.loc[101, "club_context_event_count_delta"], 1.0)

        # Friday's story is known before Sunday's kickoff, but after the shared
        # Thursday 11am round decision.  It cannot enter Sunday's row.
        self.assertEqual(features.loc[102, "club_context_event_count_home"], 0.0)
        # Post-match comments cannot flow backwards into the Thursday match.
        self.assertEqual(features.loc[101, "club_context_human_event_count_home"], 0.0)

    def test_arrow_announcement_and_tribute_are_distinct_bounded_phases(self):
        announcement_match = pd.DataFrame(
            [
                {
                    "game_id": 90,
                    "competition_year": 2026,
                    "round_id": 12,
                    "team_home": "North Queensland Cowboys",
                    "team_away": "South Sydney Rabbitohs",
                    "start_time_utc": "2026-05-24T06:05:00Z",
                }
            ]
        )
        announcement = build_context_match_features(
            self.path, announcement_match
        ).iloc[0]
        tribute = build_context_match_features(self.path, FIXTURES).set_index("game_id").loc[101]
        self.assertEqual(announcement["club_context_human_event_count_away"], 1.0)
        self.assertEqual(announcement["club_context_announcement_count_away"], 1.0)
        self.assertEqual(announcement["club_context_tribute_milestone_count_away"], 0.0)
        self.assertEqual(tribute["club_context_human_event_count_home"], 0.0)
        self.assertEqual(tribute["club_context_tribute_milestone_count_home"], 1.0)

    def test_explicit_live_decision_time_can_see_new_context_on_a_new_run(self):
        features = build_context_match_features(
            self.path,
            FIXTURES,
            decision_at_utc="2026-06-12T03:00:00Z",
        ).set_index("game_id")
        self.assertEqual(features.loc[102, "club_context_event_count_home"], 1.0)
        self.assertEqual(features.loc[102, "club_context_club_crisis_count_home"], 1.0)

    def test_missing_database_fails_soft_with_explicit_missing_flag(self):
        missing = build_context_match_features(
            Path(self.tmp.name) / "absent.sqlite", FIXTURES[["game_id"]]
        )
        self.assertEqual(len(missing), 2)
        self.assertTrue((missing["club_context_features_missing"] == 1.0).all())
        self.assertTrue((missing["club_context_event_count_home"] == 0.0).all())

    def test_features_are_exposure_only_not_signed_sentiment(self):
        features = build_context_match_features(self.path, FIXTURES)
        self.assertFalse(any("sentiment" in column for column in features.columns))
        self.assertFalse(any("boost" in column for column in features.columns))
        non_delta = [
            column
            for column in features.columns
            if column != "game_id" and not column.endswith("_delta")
        ]
        self.assertTrue((features[non_delta] >= 0.0).all().all())


class ClubContextSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "db.sqlite"
        _seed_db(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_snapshot_is_append_only_and_contains_product_facts(self):
        run_id = create_prediction_context_snapshot(
            self.path,
            FIXTURES,
            mode=PredictionMode.LIVE,
            decision_at_utc="2026-06-11T01:07:00Z",
            model_release={"release_id": "release-1"},
            prediction_run_id="run-one",
            strict=True,
        )
        self.assertEqual(run_id, "run-one")
        loaded = load_prediction_context(self.path, [101], prediction_run_id=run_id)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded.iloc[0]["decision_at_utc"], "2026-06-11T01:07:00+00:00")
        self.assertEqual(loaded.iloc[0]["round_target_at_utc"], "2026-06-11T01:00:00+00:00")
        self.assertEqual(loaded.iloc[0]["model_release"], "release-1")
        cards = json.loads(loaded.iloc[0]["observed_context_json"])
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["category"], "tribute_milestone")
        self.assertEqual(cards[0]["phase"], "tribute")
        self.assertTrue(cards[0]["sensitive"])
        self.assertTrue(cards[0]["eligible"])
        self.assertTrue(cards[0]["shadow_mode"])
        self.assertEqual(cards[0]["game_ids"], [101])
        self.assertEqual(cards[0]["source_url"], "https://nrl.com/news/2026/06/10/arrow-tribute")

        with closing(sqlite3.connect(self.path)) as con, con:
            with self.assertRaisesRegex(sqlite3.IntegrityError, "immutable"):
                con.execute(
                    "UPDATE prediction_context SET features_json='{}' "
                    "WHERE prediction_run_id='run-one'"
                )
            with self.assertRaisesRegex(sqlite3.IntegrityError, "immutable"):
                con.execute(
                    "DELETE FROM context_prediction_runs "
                    "WHERE prediction_run_id='run-one'"
                )

    def test_later_refresh_appends_and_does_not_overwrite_live_snapshot(self):
        first = create_prediction_context_snapshot(
            self.path,
            FIXTURES,
            mode=PredictionMode.REFRESH,
            decision_at_utc="2026-06-11T01:07:00Z",
            prediction_run_id="first",
            strict=True,
        )
        second = create_prediction_context_snapshot(
            self.path,
            FIXTURES,
            mode=PredictionMode.REFRESH,
            decision_at_utc="2026-06-12T03:00:00Z",
            prediction_run_id="second",
            strict=True,
        )
        self.assertNotEqual(first, second)
        with closing(sqlite3.connect(self.path)) as con:
            self.assertEqual(
                con.execute("SELECT COUNT(*) FROM context_prediction_runs").fetchone()[0],
                2,
            )
            first_events = con.execute(
                "SELECT eligible_event_ids_json FROM prediction_context "
                "WHERE prediction_run_id='first' AND game_id=102"
            ).fetchone()[0]
            second_events = con.execute(
                "SELECT eligible_event_ids_json FROM prediction_context "
                "WHERE prediction_run_id='second' AND game_id=102"
            ).fetchone()[0]
        self.assertEqual(json.loads(first_events), [])
        self.assertEqual(len(json.loads(second_events)), 1)

    def test_same_frozen_context_has_deterministic_hash(self):
        for run_id in ("one", "two"):
            create_prediction_context_snapshot(
                self.path,
                FIXTURES,
                mode=PredictionMode.HISTORICAL,
                decision_at_utc="2026-06-11T01:00:00Z",
                prediction_run_id=run_id,
                strict=True,
            )
        with closing(sqlite3.connect(self.path)) as con:
            hashes = [
                row[0]
                for row in con.execute(
                    "SELECT context_hash FROM context_prediction_runs ORDER BY prediction_run_id"
                )
            ]
        self.assertEqual(hashes[0], hashes[1])

    def test_operational_snapshot_requires_the_pre_feature_decision_time(self):
        self.assertIsNone(
            create_prediction_context_snapshot(
                self.path,
                FIXTURES,
                mode=PredictionMode.LIVE,
            )
        )
        with self.assertRaisesRegex(ValueError, "captured before feature construction"):
            create_prediction_context_snapshot(
                self.path,
                FIXTURES,
                mode=PredictionMode.LIVE,
                strict=True,
            )
        # Honest historical replay has a deterministic default.
        self.assertEqual(
            create_prediction_context_snapshot(
                self.path,
                FIXTURES,
                mode=PredictionMode.HISTORICAL,
                prediction_run_id="historical-default",
                strict=True,
            ),
            "historical-default",
        )

    def test_load_missing_table_is_fail_soft(self):
        empty = Path(self.tmp.name) / "empty.sqlite"
        sqlite3.connect(empty).close()
        self.assertTrue(load_prediction_context(empty).empty)


if __name__ == "__main__":
    unittest.main()


class ContextFeatureV2Tests(unittest.TestCase):
    """The v2 block: continuous shape, dense regime state, signed orientation."""

    def _database(self):
        folder = tempfile.mkdtemp()
        path = Path(folder) / "context.sqlite"
        with closing(sqlite3.connect(str(path))) as con:
            ensure_context_tables(con)
            con.commit()
        return path

    def test_v1_columns_are_all_preserved(self):
        from pipeline.common.club_context.features import (
            CONTEXT_FEATURE_COLUMNS,
            V1_SIDE_METRICS,
        )

        # The shipped 7 September materiality report must stay reproducible
        # against an unchanged v1 column contract.
        for metric in V1_SIDE_METRICS:
            for side in ("home", "away", "delta"):
                self.assertIn(f"club_context_{metric}_{side}", CONTEXT_FEATURE_COLUMNS)

    def test_orientation_columns_exist_and_are_signed(self):
        from pipeline.common.club_context.features import _orientation

        self.assertEqual(
            _orientation(home_exposure=0.8, away_exposure=0.0)[
                "club_context_affected_side"
            ],
            1.0,
        )
        self.assertEqual(
            _orientation(home_exposure=0.0, away_exposure=0.8)[
                "club_context_affected_side"
            ],
            -1.0,
        )
        # Equal exposure on both sides carries no orientation at all.
        self.assertEqual(
            _orientation(home_exposure=0.4, away_exposure=0.4)[
                "club_context_affected_side"
            ],
            0.0,
        )

    def test_no_registry_reports_missing_rather_than_quiet_zeroes(self):
        from pipeline.common.club_context.features import build_context_match_features

        features = build_context_match_features(
            Path(tempfile.mkdtemp()) / "absent.sqlite", FIXTURES
        )
        self.assertTrue((features["club_context_features_missing"] == 1.0).all())
        self.assertTrue((features["club_context_attention_missing_home"] == 1.0).all())
        self.assertTrue((features["club_context_data_available"] == 0.0).all())

    def test_attention_missing_defaults_to_one_after_a_merge(self):
        from pipeline.common.club_context.features import fill_context_feature_columns

        frame = pd.DataFrame(
            {
                "game_id": [1],
                "club_context_attention_missing_home": [None],
                "club_context_exposure_index_home": [None],
            }
        )
        filled = fill_context_feature_columns(frame)
        self.assertEqual(filled["club_context_attention_missing_home"].iloc[0], 1.0)
        self.assertEqual(filled["club_context_exposure_index_home"].iloc[0], 0.0)

    def test_regime_state_is_censored_without_an_in_season_handover(self):
        import datetime as dt

        from pipeline.common.club_context.features import _regime_matches

        match_at = dt.datetime(2024, 6, 1, tzinfo=dt.timezone.utc)
        kickoffs = {
            "eels": [
                dt.datetime(2024, 3, 10, tzinfo=dt.timezone.utc),
                dt.datetime(2024, 4, 10, tzinfo=dt.timezone.utc),
                dt.datetime(2024, 5, 10, tzinfo=dt.timezone.utc),
            ]
        }
        matches, censored = _regime_matches(
            {}, kickoffs, team_key="eels", match_at=match_at
        )
        self.assertEqual(censored, 1.0)
        self.assertEqual(matches, 3.0)

        handovers = {"eels": [dt.datetime(2024, 4, 1, tzinfo=dt.timezone.utc)]}
        matches, censored = _regime_matches(
            handovers, kickoffs, team_key="eels", match_at=match_at
        )
        self.assertEqual(censored, 0.0)
        self.assertEqual(matches, 2.0)

    def test_a_previous_season_handover_does_not_leak_into_this_one(self):
        import datetime as dt

        from pipeline.common.club_context.features import _regime_matches

        # The census records in-season handovers only, so a 2019 change says
        # nothing about who is coaching in 2024.
        handovers = {"eels": [dt.datetime(2019, 5, 1, tzinfo=dt.timezone.utc)]}
        kickoffs = {"eels": [dt.datetime(2024, 3, 10, tzinfo=dt.timezone.utc)]}
        _, censored = _regime_matches(
            handovers,
            kickoffs,
            team_key="eels",
            match_at=dt.datetime(2024, 6, 1, tzinfo=dt.timezone.utc),
        )
        self.assertEqual(censored, 1.0)

    def test_context_columns_stay_out_of_production_predictors(self):
        from pipeline.common.model_training import training_config as tc

        offenders = [
            name
            for name in tc.predictors
            if "club_context" in str(name) or "attention" in str(name)
        ]
        self.assertEqual(offenders, [])
        self.assertTrue(tc.shadow_context_predictors)
