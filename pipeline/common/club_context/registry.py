"""Validated persistence and eligibility rules for Club Context events."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import sqlite3
import uuid
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from ..lineups.normalization import TEAM_ALIASES, normalize_player_name, normalize_team_name
from .schema import context_schema_current, context_tables_present, ensure_context_tables
from .taxonomy import (
    CONTEXT_TAXONOMY_VERSION,
    EVIDENCE_RIGHTS_STATUSES,
    MIN_ELIGIBLE_CONFIDENCE,
    OFFICIAL_SOURCE_CLASSES,
    REPUTABLE_SOURCE_CLASSES,
    AcquisitionMethod,
    ConfirmationStatus,
    EntityRelationship,
    EntityType,
    EventCategory,
    EventPhase,
    EvidenceRole,
    IngestionMode,
    IngestionStatus,
    KnownAtBasis,
    ReviewStatus,
    RightsStatus,
    Sensitivity,
    SourceClass,
    enum_value,
    unit_interval,
)
from .time import parse_datetime, utc_iso, utc_now


_TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "oc",
    "ref",
    "referrer",
    "source",
}
_SAFE_KEY_PATTERN = re.compile(r"[^a-z0-9_.:-]+")
_PARSE_STATUSES = frozenset({"ok", "error", "skipped"})


def _clean_text(value: object | None) -> str:
    return " ".join(str(value or "").split())


def _stable_key(value: object, field_name: str) -> str:
    clean = _SAFE_KEY_PATTERN.sub("-", _clean_text(value).casefold()).strip("-._:")
    if not clean:
        raise ValueError(f"{field_name} is required")
    return clean


def canonicalize_url(url: str) -> str:
    """Canonicalise an HTTP(S) article URL for snapshot deduplication."""

    raw = _clean_text(url)
    parts = urlsplit(raw)
    if parts.scheme.casefold() not in {"http", "https"} or not parts.hostname:
        raise ValueError("article URL must be absolute HTTP(S)")

    host = parts.hostname.casefold().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    try:
        port = parts.port
    except ValueError as exc:
        raise ValueError("article URL contains an invalid port") from exc
    if port and port not in {80, 443}:
        host = f"{host}:{port}"

    path = re.sub(r"/{2,}", "/", parts.path or "/")
    if path != "/":
        path = path.rstrip("/")

    query = []
    for key, value in parse_qsl(parts.query, keep_blank_values=False):
        folded = key.casefold()
        if folded.startswith("utm_") or folded in _TRACKING_QUERY_KEYS:
            continue
        query.append((key, value))
    query.sort(key=lambda item: (item[0].casefold(), item[1]))
    return urlunsplit(("https", host, path, urlencode(query, doseq=True), ""))


def _article_hash(
    *,
    title: str,
    snippet: str | None,
    source_published_at_utc: str | None,
    supplied: str | None,
) -> str:
    if supplied:
        clean = _clean_text(supplied).casefold()
        if not re.fullmatch(r"[0-9a-f]{64}", clean):
            raise ValueError("content_hash must be a SHA-256 hex digest")
        return clean
    payload = json.dumps(
        {
            "title": _clean_text(title),
            "snippet": _clean_text(snippet),
            "published": source_published_at_utc,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ArticleSnapshot:
    source_url: str
    publisher_key: str
    source_name: str
    source_class: SourceClass
    title: str
    first_observed_at_utc: Any
    fetched_at_utc: Any
    rights_status: RightsStatus
    acquisition_method: AcquisitionMethod = AcquisitionMethod.MANUAL
    automated_use_allowed: bool = False
    independence_key: str | None = None
    snippet: str | None = None
    source_published_at_utc: Any | None = None
    source_modified_at_utc: Any | None = None
    extraction_version: str = "1"
    content_hash: str | None = None
    parse_status: str = "ok"
    parse_error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EventSource:
    snapshot_id: int
    evidence_role: EvidenceRole = EvidenceRole.CORROBORATION


@dataclass(frozen=True)
class EventEntity:
    entity_type: EntityType
    entity_key: str
    display_name: str
    team_key: str
    relationship: EntityRelationship = EntityRelationship.AFFECTED
    external_id: str = ""


@dataclass(frozen=True)
class ContextEvent:
    event_key: str
    category: EventCategory
    phase: EventPhase
    known_at_utc: Any
    effective_from_utc: Any
    factual_summary: str
    confidence: float
    salience: float
    confirmation_status: ConfirmationStatus = ConfirmationStatus.UNCONFIRMED
    review_status: ReviewStatus = ReviewStatus.PENDING
    sensitivity: Sensitivity = Sensitivity.STANDARD
    expires_at_utc: Any | None = None
    known_at_basis: KnownAtBasis = KnownAtBasis.SOURCE_PUBLISHED_AT
    taxonomy_version: int = CONTEXT_TAXONOMY_VERSION
    extractor_version: str = "manual-v1"


@dataclass(frozen=True)
class RegistryValidationReport:
    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    counts: Mapping[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _normalise_article(article: ArticleSnapshot) -> dict[str, Any]:
    source_class = enum_value(SourceClass, article.source_class, "source_class")
    rights = enum_value(RightsStatus, article.rights_status, "rights_status")
    acquisition = enum_value(
        AcquisitionMethod, article.acquisition_method, "acquisition_method"
    )
    title = _clean_text(article.title)
    if not title:
        raise ValueError("title is required")
    publisher_key = _stable_key(article.publisher_key, "publisher_key")
    independence_key = _stable_key(
        article.independence_key or publisher_key, "independence_key"
    )
    parse_status = _clean_text(article.parse_status).casefold()
    if parse_status not in _PARSE_STATUSES:
        raise ValueError("parse_status must be ok, error, or skipped")
    published = (
        utc_iso(article.source_published_at_utc)
        if article.source_published_at_utc is not None
        else None
    )
    modified = (
        utc_iso(article.source_modified_at_utc)
        if article.source_modified_at_utc is not None
        else None
    )
    first_observed = utc_iso(article.first_observed_at_utc)
    fetched = utc_iso(article.fetched_at_utc)
    if parse_datetime(fetched) < parse_datetime(first_observed):
        raise ValueError("fetched_at_utc cannot precede first_observed_at_utc")
    try:
        metadata_json = json.dumps(
            dict(article.metadata), sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must be JSON serialisable") from exc
    return {
        "canonical_url": canonicalize_url(article.source_url),
        "source_url": _clean_text(article.source_url),
        "publisher_key": publisher_key,
        "independence_key": independence_key,
        "source_name": _clean_text(article.source_name) or publisher_key,
        "source_class": source_class,
        "title": title,
        "snippet": _clean_text(article.snippet) or None,
        "source_published_at_utc": published,
        "source_modified_at_utc": modified,
        "first_observed_at_utc": first_observed,
        "fetched_at_utc": fetched,
        "content_hash": _article_hash(
            title=title,
            snippet=article.snippet,
            source_published_at_utc=published,
            supplied=article.content_hash,
        ),
        "rights_status": rights,
        "automated_use_allowed": int(bool(article.automated_use_allowed)),
        "acquisition_method": acquisition,
        "extraction_version": _clean_text(article.extraction_version) or "1",
        "parse_status": parse_status,
        "parse_error": _clean_text(article.parse_error) or None,
        "metadata_json": metadata_json,
    }


def upsert_article_snapshot(
    con: sqlite3.Connection, article: ArticleSnapshot
) -> int:
    """Insert one immutable article version or return its existing id."""

    if not context_schema_current(con):
        ensure_context_tables(con)
    row = _normalise_article(article)
    columns = tuple(row)
    con.execute(
        f"INSERT OR IGNORE INTO context_article_snapshots "
        f"({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
        tuple(row[column] for column in columns),
    )
    found = con.execute(
        """
        SELECT snapshot_id FROM context_article_snapshots
        WHERE canonical_url = ? AND content_hash = ? AND extraction_version = ?
        """,
        (row["canonical_url"], row["content_hash"], row["extraction_version"]),
    ).fetchone()
    if found is None:
        raise sqlite3.IntegrityError("article snapshot could not be stored")
    return int(found[0])


def safe_upsert_article_snapshot(
    con: sqlite3.Connection, article: ArticleSnapshot
) -> int | None:
    try:
        return upsert_article_snapshot(con, article)
    except Exception:
        return None


def _normalise_entity(entity: EventEntity) -> dict[str, Any]:
    entity_type = enum_value(EntityType, entity.entity_type, "entity_type")
    relationship = enum_value(
        EntityRelationship, entity.relationship, "relationship"
    )
    team_key = normalize_team_name(entity.team_key)
    if team_key not in TEAM_ALIASES:
        raise ValueError(f"unknown NRL team: {entity.team_key!r}")
    display_name = _clean_text(entity.display_name)
    if not display_name:
        raise ValueError("entity display_name is required")
    if entity_type == EntityType.TEAM.value:
        entity_key = team_key
    elif entity_type == EntityType.PLAYER.value:
        entity_key = normalize_player_name(entity.entity_key)
    else:
        entity_key = _stable_key(entity.entity_key, "entity_key")
    return {
        "entity_type": entity_type,
        "entity_key": entity_key,
        "display_name": display_name,
        "team_key": team_key,
        "relationship": relationship,
        "external_id": _clean_text(entity.external_id),
    }


def _normalise_event(event: ContextEvent) -> dict[str, Any]:
    event_key = _stable_key(event.event_key, "event_key")
    category = enum_value(EventCategory, event.category, "category")
    phase = enum_value(EventPhase, event.phase, "phase")
    known_at = utc_iso(event.known_at_utc)
    effective = utc_iso(event.effective_from_utc)
    expires = utc_iso(event.expires_at_utc) if event.expires_at_utc is not None else None
    if expires is not None and parse_datetime(expires) < parse_datetime(effective):
        raise ValueError("expires_at_utc cannot precede effective_from_utc")
    summary = _clean_text(event.factual_summary)
    if not summary:
        raise ValueError("factual_summary is required")
    taxonomy_version = int(event.taxonomy_version)
    if taxonomy_version < 1:
        raise ValueError("taxonomy_version must be positive")
    now = utc_iso(utc_now())
    return {
        "event_key": event_key,
        "category": category,
        "phase": phase,
        "known_at_utc": known_at,
        "known_at_basis": enum_value(
            KnownAtBasis, event.known_at_basis, "known_at_basis"
        ),
        "effective_from_utc": effective,
        "expires_at_utc": expires,
        "confidence": unit_interval(event.confidence, "confidence"),
        "salience": unit_interval(event.salience, "salience"),
        "sensitivity": enum_value(Sensitivity, event.sensitivity, "sensitivity"),
        "factual_summary": summary,
        "confirmation_status": enum_value(
            ConfirmationStatus, event.confirmation_status, "confirmation_status"
        ),
        "review_status": enum_value(ReviewStatus, event.review_status, "review_status"),
        "taxonomy_version": taxonomy_version,
        "extractor_version": _clean_text(event.extractor_version) or "manual-v1",
        "created_at_utc": now,
        "updated_at_utc": now,
    }


def insert_context_event(
    con: sqlite3.Connection,
    event: ContextEvent,
    *,
    sources: Sequence[EventSource],
    entities: Sequence[EventEntity],
) -> int:
    """Idempotently insert an event and additive evidence/entity links.

    Existing event facts are not silently rewritten.  A corrected phase or
    fact should use a new event key; prediction-run snapshots preserve the
    exact earlier interpretation.
    """

    if not context_schema_current(con):
        ensure_context_tables(con)
    if not sources:
        raise ValueError("at least one event source is required")
    if not entities:
        raise ValueError("at least one affected entity is required")
    row = _normalise_event(event)

    # Validate every foreign row before writing anything.  The savepoint also
    # keeps a plain long-lived caller connection from retaining a partial
    # event if an unexpected SQLite error occurs midway through the links.
    source_rows = []
    for source in sources:
        role = enum_value(EvidenceRole, source.evidence_role, "evidence_role")
        article = con.execute(
            """
            SELECT publisher_key, independence_key, source_class
            FROM context_article_snapshots WHERE snapshot_id = ?
            """,
            (int(source.snapshot_id),),
        ).fetchone()
        if article is None:
            raise ValueError(f"unknown article snapshot: {source.snapshot_id}")
        source_class = SourceClass(article[2])
        source_rows.append(
            (
                int(source.snapshot_id),
                role,
                article[0],
                article[1],
                int(source_class in OFFICIAL_SOURCE_CLASSES),
                int(source_class in REPUTABLE_SOURCE_CLASSES),
            )
        )
    entity_rows = [_normalise_entity(entity) for entity in entities]

    con.execute("SAVEPOINT club_context_event_insert")
    try:
        columns = tuple(row)
        con.execute(
            f"INSERT OR IGNORE INTO context_events "
            f"({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
            tuple(row[column] for column in columns),
        )
        found = con.execute(
            "SELECT event_id FROM context_events WHERE event_key = ?",
            (row["event_key"],),
        ).fetchone()
        if found is None:
            raise sqlite3.IntegrityError("context event could not be stored")
        event_id = int(found[0])
        linked_at = utc_iso(utc_now())

        for snapshot_id, role, publisher, independence, official, reputable in source_rows:
            con.execute(
                """
                INSERT OR IGNORE INTO context_event_sources (
                    event_id, snapshot_id, evidence_role, publisher_key,
                    independence_key, is_official, is_reputable, linked_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    snapshot_id,
                    role,
                    publisher,
                    independence,
                    official,
                    reputable,
                    linked_at,
                ),
            )

        for item in entity_rows:
            con.execute(
                """
                INSERT OR IGNORE INTO context_event_entities (
                    event_id, entity_type, entity_key, display_name, team_key,
                    relationship, external_id, linked_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    item["entity_type"],
                    item["entity_key"],
                    item["display_name"],
                    item["team_key"],
                    item["relationship"],
                    item["external_id"],
                    linked_at,
                ),
            )
        con.execute("RELEASE SAVEPOINT club_context_event_insert")
        return event_id
    except Exception:
        con.execute("ROLLBACK TO SAVEPOINT club_context_event_insert")
        con.execute("RELEASE SAVEPOINT club_context_event_insert")
        raise


def safe_insert_context_event(
    con: sqlite3.Connection,
    event: ContextEvent,
    *,
    sources: Sequence[EventSource],
    entities: Sequence[EventEntity],
) -> int | None:
    try:
        return insert_context_event(con, event, sources=sources, entities=entities)
    except Exception:
        return None


def set_event_review_status(
    con: sqlite3.Connection,
    event_id: int,
    review_status: ReviewStatus,
    *,
    confirmation_status: ConfirmationStatus | None = None,
) -> bool:
    """Apply a human review decision to the mutable research registry."""

    review = enum_value(ReviewStatus, review_status, "review_status")
    values: list[Any] = [review]
    assignments = ["review_status = ?"]
    if confirmation_status is not None:
        assignments.append("confirmation_status = ?")
        values.append(
            enum_value(
                ConfirmationStatus, confirmation_status, "confirmation_status"
            )
        )
    assignments.append("updated_at_utc = ?")
    values.append(utc_iso(utc_now()))
    values.append(int(event_id))
    cursor = con.execute(
        f"UPDATE context_events SET {', '.join(assignments)} WHERE event_id = ?",
        tuple(values),
    )
    return cursor.rowcount == 1


def _source_rows(
    con: sqlite3.Connection,
    event_id: int,
    decision_at_utc: Any | None = None,
) -> list[dict[str, Any]]:
    rows = con.execute(
        """
        SELECT es.snapshot_id, es.evidence_role, es.publisher_key,
               es.independence_key, es.is_official, es.is_reputable,
               a.canonical_url, a.source_url, a.source_name, a.source_class,
               a.title, a.source_published_at_utc, a.first_observed_at_utc,
               a.rights_status, a.parse_status
        FROM context_event_sources es
        JOIN context_article_snapshots a ON a.snapshot_id = es.snapshot_id
        WHERE es.event_id = ?
        ORDER BY es.is_official DESC, es.is_reputable DESC,
                 COALESCE(a.source_published_at_utc, a.first_observed_at_utc),
                 es.snapshot_id
        """,
        (int(event_id),),
    ).fetchall()
    columns = (
        "snapshot_id",
        "evidence_role",
        "publisher_key",
        "independence_key",
        "is_official",
        "is_reputable",
        "canonical_url",
        "source_url",
        "source_name",
        "source_class",
        "title",
        "source_published_at_utc",
        "first_observed_at_utc",
        "rights_status",
        "parse_status",
    )
    cutoff = parse_datetime(decision_at_utc) if decision_at_utc is not None else None
    result = []
    for raw in rows:
        item = dict(zip(columns, raw))
        source_known = item["source_published_at_utc"] or item["first_observed_at_utc"]
        try:
            if cutoff is not None and parse_datetime(source_known) > cutoff:
                continue
        except ValueError:
            continue
        result.append(item)
    return result


def _event_row_eligible(
    row: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    *,
    decision_at_utc: Any | None,
    match_at_utc: Any | None,
) -> bool:
    try:
        if row["review_status"] != ReviewStatus.APPROVED.value:
            return False
        if row["confirmation_status"] != ConfirmationStatus.CONFIRMED.value:
            return False
        if float(row["confidence"]) < MIN_ELIGIBLE_CONFIDENCE:
            return False
        if int(row["taxonomy_version"]) != CONTEXT_TAXONOMY_VERSION:
            return False

        if decision_at_utc is not None:
            decision = parse_datetime(decision_at_utc)
            if parse_datetime(row["known_at_utc"]) > decision:
                return False
        if match_at_utc is not None:
            match_at = parse_datetime(match_at_utc)
            if parse_datetime(row["effective_from_utc"]) > match_at:
                return False
            if row.get("expires_at_utc") and parse_datetime(row["expires_at_utc"]) < match_at:
                return False
    except (KeyError, TypeError, ValueError):
        return False

    qualifying = [
        source
        for source in sources
        if source["parse_status"] == "ok"
        and source["evidence_role"]
        in {EvidenceRole.CONFIRMATION.value, EvidenceRole.CORROBORATION.value}
        and source["rights_status"]
        in {status.value for status in EVIDENCE_RIGHTS_STATUSES}
    ]
    if any(bool(source["is_official"]) for source in qualifying):
        return True
    independent = {
        source["independence_key"]
        for source in qualifying
        if bool(source["is_reputable"])
    }
    return len(independent) >= 2


def event_is_eligible(
    con: sqlite3.Connection,
    event_id: int,
    *,
    decision_at_utc: Any | None = None,
    match_at_utc: Any | None = None,
) -> bool:
    """Apply the evidence gate; malformed/missing registry data is neutral."""

    previous_factory = con.row_factory
    try:
        con.row_factory = sqlite3.Row
        row = con.execute(
            "SELECT * FROM context_events WHERE event_id = ?", (int(event_id),)
        ).fetchone()
        if row is None:
            return False
        sources = _source_rows(con, int(event_id), decision_at_utc)
        return _event_row_eligible(
            dict(row),
            sources,
            decision_at_utc=decision_at_utc,
            match_at_utc=match_at_utc,
        )
    except Exception:
        return False
    finally:
        con.row_factory = previous_factory


def load_eligible_events(
    con: sqlite3.Connection,
    team_keys: Sequence[str],
    *,
    decision_at_utc: Any,
    match_at_utc: Any,
) -> list[dict[str, Any]]:
    """Eligible event facts for explicit affected teams at one decision time."""

    previous_factory = con.row_factory
    try:
        keys = sorted({normalize_team_name(value) for value in team_keys})
        if not keys or not context_tables_present(con):
            return []
        placeholders = ",".join("?" for _ in keys)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            f"""
            SELECT DISTINCT e.*
            FROM context_events e
            JOIN context_event_entities ee ON ee.event_id = e.event_id
            WHERE ee.team_key IN ({placeholders})
              AND ee.relationship IN ('affected', 'subject')
            ORDER BY e.effective_from_utc, e.event_id
            """,
            tuple(keys),
        ).fetchall()
        eligible = []
        for raw in rows:
            row = dict(raw)
            sources = _source_rows(con, int(row["event_id"]), decision_at_utc)
            if not _event_row_eligible(
                row,
                sources,
                decision_at_utc=decision_at_utc,
                match_at_utc=match_at_utc,
            ):
                continue
            entity_rows = con.execute(
                """
                SELECT entity_type, entity_key, display_name, team_key,
                       relationship, external_id
                FROM context_event_entities WHERE event_id = ?
                ORDER BY team_key, entity_type, entity_key
                """,
                (int(row["event_id"]),),
            ).fetchall()
            row["entities"] = [dict(entity) for entity in entity_rows]
            row["sources"] = sources
            eligible.append(row)
        return eligible
    except Exception:
        return []
    finally:
        con.row_factory = previous_factory


def start_ingestion_run(
    con: sqlite3.Connection,
    mode: IngestionMode,
    *,
    config: Mapping[str, Any] | None = None,
    started_at_utc: Any | None = None,
    run_id: str | None = None,
) -> str:
    if not context_schema_current(con):
        ensure_context_tables(con)
    identifier = run_id or str(uuid.uuid4())
    config_json = json.dumps(dict(config or {}), sort_keys=True, separators=(",", ":"))
    con.execute(
        """
        INSERT INTO context_ingestion_runs (
            run_id, mode, started_at_utc, status, config_json
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            identifier,
            enum_value(IngestionMode, mode, "mode"),
            utc_iso(started_at_utc or utc_now()),
            IngestionStatus.STARTED.value,
            config_json,
        ),
    )
    return identifier


def finish_ingestion_run(
    con: sqlite3.Connection,
    run_id: str,
    status: IngestionStatus,
    *,
    candidate_count: int = 0,
    snapshot_count: int = 0,
    event_count: int = 0,
    eligible_event_count: int = 0,
    errors: Sequence[str] = (),
    completed_at_utc: Any | None = None,
) -> bool:
    status_value = enum_value(IngestionStatus, status, "status")
    if status_value == IngestionStatus.STARTED.value:
        raise ValueError("a completed run cannot retain started status")
    errors_json = json.dumps([_clean_text(error) for error in errors])
    cursor = con.execute(
        """
        UPDATE context_ingestion_runs SET
            completed_at_utc = ?, status = ?, candidate_count = ?,
            snapshot_count = ?, event_count = ?, eligible_event_count = ?,
            error_count = ?, errors_json = ?
        WHERE run_id = ? AND status = 'started'
        """,
        (
            utc_iso(completed_at_utc or utc_now()),
            status_value,
            max(0, int(candidate_count)),
            max(0, int(snapshot_count)),
            max(0, int(event_count)),
            max(0, int(eligible_event_count)),
            len(errors),
            errors_json,
            run_id,
        ),
    )
    return cursor.rowcount == 1


def validate_registry(db_path: str | Path) -> RegistryValidationReport:
    """Audit the persisted registry without creating or changing it."""

    path = Path(db_path)
    if not path.exists():
        return RegistryValidationReport(False, ("database does not exist",), (), {})
    errors: list[str] = []
    warnings: list[str] = []
    counts: dict[str, int] = {}
    try:
        with closing(sqlite3.connect(str(path))) as con:
            con.row_factory = sqlite3.Row
            if not context_tables_present(con):
                return RegistryValidationReport(
                    False, ("Club Context tables are incomplete",), (), {}
                )
            for table in (
                "context_article_snapshots",
                "context_events",
                "context_event_sources",
                "context_event_entities",
                "context_ingestion_runs",
                "context_prediction_runs",
                "prediction_context",
            ):
                counts[table] = int(
                    con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                )

            articles = con.execute("SELECT * FROM context_article_snapshots").fetchall()
            for article in articles:
                prefix = f"snapshot {article['snapshot_id']}"
                try:
                    SourceClass(article["source_class"])
                    RightsStatus(article["rights_status"])
                    AcquisitionMethod(article["acquisition_method"])
                    canonical = canonicalize_url(article["canonical_url"])
                    if canonical != article["canonical_url"]:
                        raise ValueError("canonical_url is not canonical")
                    parse_datetime(article["first_observed_at_utc"])
                    parse_datetime(article["fetched_at_utc"])
                    if article["source_published_at_utc"]:
                        parse_datetime(article["source_published_at_utc"])
                    if not re.fullmatch(r"[0-9a-f]{64}", article["content_hash"] or ""):
                        raise ValueError("content_hash is not SHA-256")
                except Exception as exc:
                    errors.append(f"{prefix}: {exc}")
                if (
                    article["acquisition_method"] == AcquisitionMethod.ADAPTER.value
                    and not bool(article["automated_use_allowed"])
                ):
                    errors.append(f"{prefix}: rights-disabled adapter output was stored")
                if article["rights_status"] in {
                    RightsStatus.PROHIBITED.value,
                    RightsStatus.UNKNOWN.value,
                } and article["parse_status"] == "ok":
                    warnings.append(f"{prefix}: source is not eligible evidence")

            events = con.execute("SELECT * FROM context_events").fetchall()
            eligible_count = 0
            for event in events:
                prefix = f"event {event['event_id']}"
                try:
                    EventCategory(event["category"])
                    EventPhase(event["phase"])
                    ConfirmationStatus(event["confirmation_status"])
                    ReviewStatus(event["review_status"])
                    Sensitivity(event["sensitivity"])
                    KnownAtBasis(event["known_at_basis"])
                    unit_interval(event["confidence"], "confidence")
                    unit_interval(event["salience"], "salience")
                    parse_datetime(event["known_at_utc"])
                    effective = parse_datetime(event["effective_from_utc"])
                    if event["expires_at_utc"] and parse_datetime(event["expires_at_utc"]) < effective:
                        raise ValueError("expiry precedes effective time")
                except Exception as exc:
                    errors.append(f"{prefix}: {exc}")
                    continue
                source_count = con.execute(
                    "SELECT COUNT(*) FROM context_event_sources WHERE event_id = ?",
                    (event["event_id"],),
                ).fetchone()[0]
                entity_count = con.execute(
                    "SELECT COUNT(*) FROM context_event_entities WHERE event_id = ?",
                    (event["event_id"],),
                ).fetchone()[0]
                if not source_count:
                    errors.append(f"{prefix}: has no evidence sources")
                if not entity_count:
                    errors.append(f"{prefix}: has no affected entities")
                if event_is_eligible(con, event["event_id"]):
                    eligible_count += 1
                elif event["review_status"] == ReviewStatus.APPROVED.value:
                    warnings.append(f"{prefix}: approved but fails the evidence gate")
            counts["eligible_events"] = eligible_count

            source_links = con.execute("SELECT * FROM context_event_sources").fetchall()
            for link in source_links:
                prefix = f"event source {link['event_id']}/{link['snapshot_id']}"
                try:
                    EvidenceRole(link["evidence_role"])
                    if not _clean_text(link["independence_key"]):
                        raise ValueError("independence_key is missing")
                    if not con.execute(
                        "SELECT 1 FROM context_events WHERE event_id = ?",
                        (link["event_id"],),
                    ).fetchone():
                        raise ValueError("event does not exist")
                    if not con.execute(
                        "SELECT 1 FROM context_article_snapshots WHERE snapshot_id = ?",
                        (link["snapshot_id"],),
                    ).fetchone():
                        raise ValueError("article snapshot does not exist")
                except Exception as exc:
                    errors.append(f"{prefix}: {exc}")

            entity_links = con.execute("SELECT * FROM context_event_entities").fetchall()
            for entity in entity_links:
                prefix = f"event entity {entity['event_id']}/{entity['entity_key']}"
                try:
                    EntityType(entity["entity_type"])
                    EntityRelationship(entity["relationship"])
                    if entity["team_key"] not in TEAM_ALIASES:
                        raise ValueError(f"unknown NRL team key {entity['team_key']!r}")
                    if not con.execute(
                        "SELECT 1 FROM context_events WHERE event_id = ?",
                        (entity["event_id"],),
                    ).fetchone():
                        raise ValueError("event does not exist")
                except Exception as exc:
                    errors.append(f"{prefix}: {exc}")

            for run in con.execute("SELECT run_id, mode, status FROM context_ingestion_runs"):
                try:
                    IngestionMode(run["mode"])
                    IngestionStatus(run["status"])
                except Exception as exc:
                    errors.append(f"ingestion run {run['run_id']}: {exc}")
    except Exception as exc:
        errors.append(f"registry validation failed: {exc}")
    return RegistryValidationReport(not errors, tuple(errors), tuple(warnings), counts)
