import hashlib
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from pipeline.common.club_context import (
    AcquisitionMethod,
    ArticleSnapshot,
    ConfirmationStatus,
    ContextEvent,
    EntityType,
    EventCategory,
    EventEntity,
    EventPhase,
    EventSource,
    EvidenceRole,
    IngestionMode,
    IngestionStatus,
    ReviewStatus,
    RightsStatus,
    SourceClass,
    canonicalize_url,
    ensure_context_tables,
    event_is_eligible,
    finish_ingestion_run,
    insert_context_event,
    safe_insert_context_event,
    start_ingestion_run,
    upsert_article_snapshot,
    validate_registry,
)


def _article(
    con,
    *,
    url,
    publisher,
    source_class=SourceClass.REPUTABLE_MEDIA,
    published="2026-06-10T00:00:00Z",
    independence_key=None,
    rights=RightsStatus.FACTS_AND_LINKS,
):
    return upsert_article_snapshot(
        con,
        ArticleSnapshot(
            source_url=url,
            publisher_key=publisher,
            independence_key=independence_key,
            source_name=publisher.title(),
            source_class=source_class,
            title=f"Verified report from {publisher}",
            source_published_at_utc=published,
            first_observed_at_utc=published,
            fetched_at_utc=published,
            rights_status=rights,
            acquisition_method=AcquisitionMethod.MANUAL,
        ),
    )


def _event(
    con,
    *,
    key,
    source_ids,
    confidence=0.9,
    review=ReviewStatus.APPROVED,
    confirmation=ConfirmationStatus.CONFIRMED,
    known="2026-06-10T00:00:00Z",
):
    return insert_context_event(
        con,
        ContextEvent(
            event_key=key,
            category=EventCategory.LEADERSHIP_CHANGE,
            phase=EventPhase.FIRST_MATCH,
            known_at_utc=known,
            effective_from_utc="2026-06-11T09:50:00Z",
            expires_at_utc="2026-06-15T00:00:00Z",
            factual_summary="The club confirmed an interim coach for this match.",
            confidence=confidence,
            salience=0.8,
            confirmation_status=confirmation,
            review_status=review,
        ),
        sources=[EventSource(value, EvidenceRole.CONFIRMATION) for value in source_ids],
        entities=[
            EventEntity(
                entity_type=EntityType.TEAM,
                entity_key="rabbitohs",
                display_name="South Sydney Rabbitohs",
                team_key="South Sydney Rabbitohs",
            )
        ],
    )


class ClubContextSchemaTests(unittest.TestCase):
    def test_schema_is_additive_and_preserves_an_old_row(self):
        con = sqlite3.connect(":memory:")
        self.addCleanup(con.close)
        con.execute(
            "CREATE TABLE context_article_snapshots "
            "(snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT)"
        )
        con.execute("INSERT INTO context_article_snapshots (title) VALUES ('kept')")

        ensure_context_tables(con)
        ensure_context_tables(con)

        tables = {
            row[0]
            for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        self.assertTrue(
            {
                "context_article_snapshots",
                "context_events",
                "context_event_sources",
                "context_event_entities",
                "context_ingestion_runs",
                "context_prediction_runs",
                "prediction_context",
            }.issubset(tables)
        )
        columns = {
            row[1] for row in con.execute("PRAGMA table_info(context_article_snapshots)")
        }
        self.assertIn("first_observed_at_utc", columns)
        self.assertNotIn("article_body", columns)
        self.assertEqual(
            con.execute(
                "SELECT title FROM context_article_snapshots WHERE snapshot_id=1"
            ).fetchone()[0],
            "kept",
        )

    def test_url_canonicalisation_and_versioned_snapshot_deduplication(self):
        con = sqlite3.connect(":memory:")
        self.addCleanup(con.close)
        ensure_context_tables(con)
        article = ArticleSnapshot(
            source_url="http://www.NRL.com/news/story/?utm_source=x&b=2&a=1#top",
            publisher_key="nrl",
            source_name="NRL",
            source_class=SourceClass.OFFICIAL_NRL,
            title="  Club confirms   a fact ",
            first_observed_at_utc="2026-05-20T03:00:00Z",
            fetched_at_utc="2026-05-20T03:00:01Z",
            source_published_at_utc="2026-05-20T02:59:00Z",
            rights_status=RightsStatus.FACTS_AND_LINKS,
        )
        first = upsert_article_snapshot(con, article)
        second = upsert_article_snapshot(
            con,
            ArticleSnapshot(
                **{
                    **article.__dict__,
                    "source_url": "https://nrl.com/news/story?a=1&b=2",
                }
            ),
        )
        self.assertEqual(first, second)
        self.assertEqual(
            canonicalize_url(article.source_url),
            "https://nrl.com/news/story?a=1&b=2",
        )
        self.assertEqual(
            con.execute("SELECT COUNT(*) FROM context_article_snapshots").fetchone()[0],
            1,
        )

        changed_hash = hashlib.sha256(b"changed response").hexdigest()
        third = upsert_article_snapshot(
            con,
            ArticleSnapshot(**{**article.__dict__, "content_hash": changed_hash}),
        )
        self.assertNotEqual(first, third)

    def test_evidence_gate_requires_official_or_two_independent_sources(self):
        con = sqlite3.connect(":memory:")
        self.addCleanup(con.close)
        ensure_context_tables(con)
        official = _article(
            con,
            url="https://nrl.com/news/official",
            publisher="nrl",
            source_class=SourceClass.OFFICIAL_NRL,
        )
        official_event = _event(con, key="official-event", source_ids=[official])
        self.assertTrue(event_is_eligible(con, official_event))

        abc = _article(con, url="https://abc.net.au/a", publisher="abc")
        aap_copy_one = _article(
            con,
            url="https://paper.example/aap-copy-1",
            publisher="paper-one",
            independence_key="aap-wire",
        )
        aap_copy_two = _article(
            con,
            url="https://other.example/aap-copy-2",
            publisher="paper-two",
            independence_key="aap-wire",
        )
        same_wire = _event(
            con,
            key="syndicated-is-one-source",
            source_ids=[aap_copy_one, aap_copy_two],
        )
        self.assertFalse(event_is_eligible(con, same_wire))

        independent = _event(
            con,
            key="two-independent-sources",
            source_ids=[abc, aap_copy_one],
        )
        self.assertTrue(event_is_eligible(con, independent))

    def test_rumours_low_confidence_and_unapproved_events_are_ineligible(self):
        con = sqlite3.connect(":memory:")
        self.addCleanup(con.close)
        ensure_context_tables(con)
        official = _article(
            con,
            url="https://nrl.com/news/statuses",
            publisher="nrl",
            source_class=SourceClass.OFFICIAL_NRL,
        )
        low = _event(con, key="low", source_ids=[official], confidence=0.69)
        pending = _event(
            con, key="pending", source_ids=[official], review=ReviewStatus.PENDING
        )
        rumour = _event(
            con,
            key="rumour",
            source_ids=[official],
            confirmation=ConfirmationStatus.RUMOUR,
        )
        self.assertFalse(event_is_eligible(con, low))
        self.assertFalse(event_is_eligible(con, pending))
        self.assertFalse(event_is_eligible(con, rumour))

        invalid = ContextEvent(
            event_key="invalid-confidence",
            category=EventCategory.CLUB_CRISIS,
            phase=EventPhase.ANNOUNCEMENT,
            known_at_utc="2026-01-01T00:00:00Z",
            effective_from_utc="2026-01-01T00:00:00Z",
            factual_summary="A confirmed event.",
            confidence=1.2,
            salience=0.5,
        )
        self.assertIsNone(
            safe_insert_context_event(
                con,
                invalid,
                sources=[EventSource(official)],
                entities=[
                    EventEntity(
                        EntityType.TEAM,
                        "rabbitohs",
                        "Rabbitohs",
                        "Rabbitohs",
                    )
                ],
            )
        )

    def test_rights_blocked_snapshot_does_not_satisfy_evidence_gate(self):
        con = sqlite3.connect(":memory:")
        self.addCleanup(con.close)
        ensure_context_tables(con)
        blocked = _article(
            con,
            url="https://blocked.example/story",
            publisher="blocked",
            source_class=SourceClass.OFFICIAL_CLUB,
            rights=RightsStatus.PROHIBITED,
        )
        event_id = _event(con, key="blocked-rights", source_ids=[blocked])
        self.assertFalse(event_is_eligible(con, event_id))

    def test_validator_is_read_only_and_flags_rights_disabled_adapter_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "db.sqlite"
            with closing(sqlite3.connect(path)) as con, con:
                ensure_context_tables(con)
                con.execute(
                    """
                    INSERT INTO context_article_snapshots (
                        canonical_url, source_url, publisher_key, independence_key,
                        source_name, source_class, title, first_observed_at_utc,
                        fetched_at_utc, content_hash, rights_status,
                        automated_use_allowed, acquisition_method
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                    """,
                    (
                        "https://example.com/a",
                        "https://example.com/a",
                        "example",
                        "example",
                        "Example",
                        SourceClass.REPUTABLE_MEDIA.value,
                        "Story",
                        "2026-01-01T00:00:00+00:00",
                        "2026-01-01T00:00:00+00:00",
                        "a" * 64,
                        RightsStatus.FACTS_AND_LINKS.value,
                        AcquisitionMethod.ADAPTER.value,
                    ),
                )
            report = validate_registry(path)
            self.assertFalse(report.valid)
            self.assertTrue(any("rights-disabled" in error for error in report.errors))

    def test_ingestion_run_records_started_state_before_completion(self):
        con = sqlite3.connect(":memory:")
        self.addCleanup(con.close)
        run_id = start_ingestion_run(
            con,
            IngestionMode.REFRESH,
            run_id="refresh-1",
            started_at_utc="2026-06-11T00:55:00Z",
            config={"source": "official-first"},
        )
        self.assertEqual(run_id, "refresh-1")
        self.assertEqual(
            con.execute(
                "SELECT status FROM context_ingestion_runs WHERE run_id=?", (run_id,)
            ).fetchone()[0],
            "started",
        )
        self.assertTrue(
            finish_ingestion_run(
                con,
                run_id,
                IngestionStatus.COMPLETED_WITH_ERRORS,
                candidate_count=3,
                snapshot_count=2,
                event_count=1,
                eligible_event_count=1,
                errors=["one source unavailable"],
                completed_at_utc="2026-06-11T01:00:00Z",
            )
        )
        self.assertEqual(
            con.execute(
                "SELECT status, error_count FROM context_ingestion_runs WHERE run_id=?",
                (run_id,),
            ).fetchone(),
            ("completed_with_errors", 1),
        )


if __name__ == "__main__":
    unittest.main()
