"""Maintain the evidence-gated Club Context registry.

``refresh`` records discovery metadata only.  It never turns a headline into
an eligible event.  ``backfill`` imports the reviewed facts-and-links research
catalogue, and ``validate`` audits the resulting registry without mutation.
All three commands are fail-soft unless ``--strict`` is explicit.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import sqlite3
import sys
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional in stripped environments
    def load_dotenv(*_args, **_kwargs):
        return False

from pipeline.common import console
from pipeline.common.club_context import (
    AcquisitionMethod,
    ArticleSnapshot,
    ConfirmationStatus,
    ContextEvent,
    EntityRelationship,
    EntityType,
    EventCategory,
    EventEntity,
    EventPhase,
    EventSource,
    EvidenceRole,
    IngestionMode,
    IngestionStatus,
    KnownAtBasis,
    ReviewStatus,
    RightsStatus,
    Sensitivity,
    SourceClass,
    event_is_eligible,
    finish_ingestion_run,
    insert_context_event,
    start_ingestion_run,
    upsert_article_snapshot,
    validate_registry,
)
from pipeline.common.club_context.discovery import discover_all
from pipeline.common.club_context.time import utc_iso, utc_now


DEFAULT_QUERY = "NRL rugby league coach club player tribute"
DEFAULT_LOOKBACK_DAYS = 10
DEFAULT_MAX_ITEMS = 100


def _log(message: str) -> None:
    console.emit_progress(message)


def _to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline/club_context.py",
        description="Refresh, backfill, or validate the Club Context registry.",
    )
    parser.add_argument("action", choices=("refresh", "backfill", "validate"))
    parser.add_argument("--db-path", help="SQLite path override.")
    parser.add_argument(
        "--input",
        help="Reviewed JSON catalogue for backfill (defaults to data/reference/club_context_events.json).",
    )
    parser.add_argument("--report-path", help="Optional JSON validation-report destination.")
    parser.add_argument("--start-year", type=int, help="Backfill lower year bound.")
    parser.add_argument("--end-year", type=int, help="Backfill upper year bound.")
    parser.add_argument("--query", default=None, help="Refresh discovery query.")
    parser.add_argument("--lookback-days", type=int, default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--strict", action="store_true")
    return parser


def _write_report(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _refresh(
    db_path: pathlib.Path,
    *,
    query: str,
    lookback_days: int,
    max_items: int,
) -> dict[str, Any]:
    observed = utc_now()
    start = observed - dt.timedelta(days=max(1, int(lookback_days)))
    _log("Discovering recent NRL context candidates (discovery only)")
    candidates, discovery_errors = discover_all(
        query=query,
        start_at_utc=utc_iso(start),
        end_at_utc=utc_iso(observed),
        limit=max(1, int(max_items)),
    )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    errors = list(discovery_errors)
    stored = 0
    with sqlite3.connect(str(db_path)) as con:
        run_id = start_ingestion_run(
            con,
            IngestionMode.REFRESH,
            config={
                "query": query,
                "lookback_days": int(lookback_days),
                "max_items": int(max_items),
                "facts_and_links_only": True,
                "auto_classification": False,
            },
            started_at_utc=observed,
        )
        con.commit()
        for policy, candidate in candidates:
            try:
                # Discovery feeds may expose snippets, but the registry keeps
                # only metadata, hashes and links.  No article body is stored.
                upsert_article_snapshot(
                    con,
                    ArticleSnapshot(
                        source_url=candidate.url,
                        publisher_key=candidate.publisher_key,
                        independence_key=candidate.publisher_key,
                        source_name=candidate.publisher_key,
                        source_class=candidate.source_class,
                        title=candidate.title,
                        snippet=None,
                        source_published_at_utc=candidate.published_at_utc,
                        first_observed_at_utc=observed,
                        fetched_at_utc=observed,
                        rights_status=policy.rights_status,
                        acquisition_method=AcquisitionMethod.ADAPTER,
                        automated_use_allowed=policy.automated_discovery_allowed,
                        extraction_version="discovery-v1",
                        metadata={
                            "adapter": policy.source_key,
                            "evidence_role": EvidenceRole.DISCOVERY.value,
                            "terms_url": policy.terms_url,
                        },
                    ),
                )
                stored += 1
            except Exception as exc:
                errors.append(f"{policy.source_key}: {candidate.url}: {exc}")
        status = (
            IngestionStatus.COMPLETED_WITH_ERRORS if errors else IngestionStatus.OK
        )
        finish_ingestion_run(
            con,
            run_id,
            status,
            candidate_count=len(candidates),
            snapshot_count=stored,
            errors=errors,
        )
        con.commit()
    return {
        "action": "refresh",
        "run_id": run_id,
        "candidates": len(candidates),
        "snapshots": stored,
        "events": 0,
        "eligible_events": 0,
        "errors": errors,
        "shadow_only": True,
    }


def _enum(enum_type, value: Any, field: str):
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ValueError(f"{field} must be one of: {allowed}") from exc


def _leadership_event(item: dict[str, Any]) -> dict[str, Any]:
    """Expand one compact, audited census row into the registry contract."""

    team_name = item["team_name"]
    outgoing = item["outgoing_coach"]
    incoming = item["incoming_coach"]
    raw_sources = item.get("sources") or [item["source"]]
    sources = []
    for raw_source in raw_sources:
        source = dict(raw_source)
        source.setdefault("publisher_key", "nrl.com")
        source.setdefault("source_name", "NRL.com")
        source.setdefault("source_class", SourceClass.OFFICIAL_NRL.value)
        source.setdefault("rights_status", RightsStatus.FACTS_AND_LINKS.value)
        source.setdefault("evidence_role", EvidenceRole.CONFIRMATION.value)
        source.setdefault("published_at_utc", item["known_at_utc"])
        sources.append(source)
    effective = item.get("effective_from_utc") or item["known_at_utc"]
    expiry = item.get("expires_at_utc")
    if expiry is None:
        expiry = utc_iso(
            dt.datetime.fromisoformat(effective.replace("Z", "+00:00"))
            + dt.timedelta(days=21)
        )
    return {
        "event_key": item["event_key"],
        "competition_year": item["competition_year"],
        "category": EventCategory.LEADERSHIP_CHANGE.value,
        "phase": item.get("phase", EventPhase.EFFECTIVE.value),
        "known_at_utc": item["known_at_utc"],
        "known_at_basis": item.get(
            "known_at_basis", KnownAtBasis.SOURCE_PUBLISHED_AT.value
        ),
        "effective_from_utc": effective,
        "expires_at_utc": expiry,
        "confidence": item.get("confidence", 0.95),
        "salience": item.get("salience", 0.75),
        "sensitivity": Sensitivity.STANDARD.value,
        "factual_summary": item.get("factual_summary")
        or f"{team_name} changed head coach from {outgoing} to {incoming} during the season.",
        "confirmation_status": ConfirmationStatus.CONFIRMED.value,
        "review_status": ReviewStatus.APPROVED.value,
        "extractor_version": "manual-leadership-census-v1",
        "sources": sources,
        "entities": [
            {
                "entity_type": EntityType.TEAM.value,
                "entity_key": item["team_key"],
                "display_name": team_name,
                "team_key": item["team_key"],
                "relationship": EntityRelationship.AFFECTED.value,
            },
            {
                "entity_type": EntityType.COACH.value,
                "entity_key": outgoing,
                "display_name": outgoing,
                "team_key": item["team_key"],
                "relationship": EntityRelationship.SUBJECT.value,
            },
            {
                "entity_type": EntityType.COACH.value,
                "entity_key": incoming,
                "display_name": incoming,
                "team_key": item["team_key"],
                "relationship": EntityRelationship.CONTEXT.value,
            },
        ],
    }


def _catalog(path: pathlib.Path) -> dict[str, Any]:
    body = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ValueError("catalogue must be a JSON object")
    events = body.get("events") or []
    census = body.get("leadership_change_census") or []
    if not isinstance(events, list) or not isinstance(census, list):
        raise ValueError("events and leadership_change_census must be arrays")
    body["events"] = [*(_leadership_event(item) for item in census), *events]
    return body


def _event_year(item: dict[str, Any]) -> int:
    if item.get("competition_year") is not None:
        return int(item["competition_year"])
    return int(str(item["effective_from_utc"])[:4])


def _import_source(
    con: sqlite3.Connection,
    source: dict[str, Any],
    *,
    observed_at: dt.datetime,
) -> tuple[int, EvidenceRole]:
    published = source.get("published_at_utc")
    first_observed = source.get("first_observed_at_utc") or published or observed_at
    snapshot_id = upsert_article_snapshot(
        con,
        ArticleSnapshot(
            source_url=source["url"],
            publisher_key=source["publisher_key"],
            independence_key=source.get("independence_key") or source["publisher_key"],
            source_name=source.get("source_name") or source["publisher_key"],
            source_class=_enum(SourceClass, source["source_class"], "source_class"),
            title=source["title"],
            snippet=None,
            source_published_at_utc=published,
            first_observed_at_utc=first_observed,
            fetched_at_utc=observed_at,
            rights_status=_enum(
                RightsStatus,
                source.get("rights_status", RightsStatus.FACTS_AND_LINKS.value),
                "rights_status",
            ),
            acquisition_method=AcquisitionMethod.IMPORT,
            automated_use_allowed=False,
            extraction_version=str(source.get("extraction_version") or "catalog-v1"),
            metadata={
                "catalogued_fact_only": True,
                "terms_url": source.get("terms_url"),
            },
        ),
    )
    return snapshot_id, _enum(
        EvidenceRole,
        source.get("evidence_role", EvidenceRole.CONFIRMATION.value),
        "evidence_role",
    )


def _import_event(
    con: sqlite3.Connection,
    item: dict[str, Any],
    *,
    observed_at: dt.datetime,
) -> int:
    source_links = []
    for source in item.get("sources") or []:
        snapshot_id, role = _import_source(con, source, observed_at=observed_at)
        source_links.append(EventSource(snapshot_id=snapshot_id, evidence_role=role))
    entities = [
        EventEntity(
            entity_type=_enum(EntityType, entity["entity_type"], "entity_type"),
            entity_key=entity["entity_key"],
            display_name=entity["display_name"],
            team_key=entity["team_key"],
            relationship=_enum(
                EntityRelationship,
                entity.get("relationship", EntityRelationship.AFFECTED.value),
                "relationship",
            ),
            external_id=str(entity.get("external_id") or ""),
        )
        for entity in item.get("entities") or []
    ]
    event = ContextEvent(
        event_key=item["event_key"],
        category=_enum(EventCategory, item["category"], "category"),
        phase=_enum(EventPhase, item["phase"], "phase"),
        known_at_utc=item["known_at_utc"],
        known_at_basis=_enum(
            KnownAtBasis,
            item.get("known_at_basis", KnownAtBasis.SOURCE_PUBLISHED_AT.value),
            "known_at_basis",
        ),
        effective_from_utc=item["effective_from_utc"],
        expires_at_utc=item.get("expires_at_utc"),
        confidence=float(item["confidence"]),
        salience=float(item["salience"]),
        sensitivity=_enum(
            Sensitivity,
            item.get("sensitivity", Sensitivity.STANDARD.value),
            "sensitivity",
        ),
        factual_summary=item["factual_summary"],
        confirmation_status=_enum(
            ConfirmationStatus,
            item.get("confirmation_status", ConfirmationStatus.CONFIRMED.value),
            "confirmation_status",
        ),
        review_status=_enum(
            ReviewStatus,
            item.get("review_status", ReviewStatus.APPROVED.value),
            "review_status",
        ),
        extractor_version=str(item.get("extractor_version") or "manual-catalog-v1"),
    )
    return insert_context_event(con, event, sources=source_links, entities=entities)


def _backfill(
    db_path: pathlib.Path,
    catalog_path: pathlib.Path,
    *,
    start_year: int | None,
    end_year: int | None,
) -> dict[str, Any]:
    body = _catalog(catalog_path)
    observed = utc_now()
    selected = [
        item
        for item in body["events"]
        if (start_year is None or _event_year(item) >= start_year)
        and (end_year is None or _event_year(item) <= end_year)
    ]
    errors: list[str] = []
    event_ids: list[int] = []
    _log(f"Importing {len(selected)} reviewed Club Context events")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as con:
        snapshots_before = int(
            con.execute(
                "SELECT COUNT(*) FROM sqlite_master "
                "WHERE type = 'table' AND name = 'context_article_snapshots'"
            ).fetchone()[0]
        )
        if snapshots_before:
            snapshots_before = int(
                con.execute("SELECT COUNT(*) FROM context_article_snapshots").fetchone()[0]
            )
        run_id = start_ingestion_run(
            con,
            IngestionMode.BACKFILL,
            config={
                "catalog": str(catalog_path),
                "catalog_version": body.get("catalog_version"),
                "start_year": start_year,
                "end_year": end_year,
            },
            started_at_utc=observed,
        )
        con.commit()
        for item in selected:
            try:
                with con:
                    event_ids.append(_import_event(con, item, observed_at=observed))
            except Exception as exc:
                errors.append(f"{item.get('event_key', '<unknown>')}: {exc}")
        eligible = sum(event_is_eligible(con, event_id) for event_id in set(event_ids))
        status = (
            IngestionStatus.COMPLETED_WITH_ERRORS if errors else IngestionStatus.OK
        )
        finish_ingestion_run(
            con,
            run_id,
            status,
            candidate_count=len(selected),
            snapshot_count=(
                int(
                    con.execute(
                        "SELECT COUNT(*) FROM context_article_snapshots"
                    ).fetchone()[0]
                )
                - snapshots_before
            ),
            event_count=len(set(event_ids)),
            eligible_event_count=eligible,
            errors=errors,
        )
        con.commit()
    return {
        "action": "backfill",
        "run_id": run_id,
        "catalog": str(catalog_path),
        "candidates": len(selected),
        "events": len(set(event_ids)),
        "eligible_events": eligible,
        "errors": errors,
        "shadow_only": True,
    }


def _validate(db_path: pathlib.Path) -> dict[str, Any]:
    report = validate_registry(db_path)
    return {"action": "validate", **report.as_dict(), "shadow_only": True}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = pathlib.Path(__file__).resolve().parents[1]
    load_dotenv(dotenv_path=project_root / "secrets.env")
    strict = bool(args.strict) or _to_bool(
        os.getenv("FOOTY_TIPPER_CONTEXT_STRICT"), False
    )
    enabled = _to_bool(os.getenv("FOOTY_TIPPER_CONTEXT_ENABLED"), True)
    db_path = pathlib.Path(args.db_path) if args.db_path else (
        project_root / "data" / "footy-tipper-db.sqlite"
    )
    catalog_path = pathlib.Path(args.input) if args.input else (
        project_root / "data" / "reference" / "club_context_events.json"
    )

    if args.action == "refresh" and not enabled:
        _log("Club Context refresh disabled; predictions remain unchanged")
        return 0

    try:
        if args.action == "refresh":
            lookback = args.lookback_days or int(
                os.getenv("FOOTY_TIPPER_CONTEXT_LOOKBACK_DAYS", DEFAULT_LOOKBACK_DAYS)
            )
            max_items = args.max_items or int(
                os.getenv("FOOTY_TIPPER_CONTEXT_MAX_ITEMS", DEFAULT_MAX_ITEMS)
            )
            result = _refresh(
                db_path,
                query=args.query or os.getenv("FOOTY_TIPPER_CONTEXT_QUERY", DEFAULT_QUERY),
                lookback_days=lookback,
                max_items=max_items,
            )
        elif args.action == "backfill":
            result = _backfill(
                db_path,
                catalog_path,
                start_year=args.start_year,
                end_year=args.end_year,
            )
        else:
            result = _validate(db_path)
    except Exception as exc:
        _log(f"Club Context {args.action} failed: {exc}")
        if strict:
            return 1
        _log("Fail-soft mode enabled; prediction and delivery remain unchanged")
        return 0

    if args.report_path:
        _write_report(pathlib.Path(args.report_path), result)
    console.emit_result(
        "freshness",
        source="club context",
        detail=(
            f"{result.get('eligible_events', result.get('counts', {}).get('eligible_events', 0))} "
            "eligible reviewed events (shadow only)"
        ),
    )
    if result.get("errors"):
        _log(f"Club Context completed with {len(result['errors'])} validation/ingestion errors")
        return 1 if strict else 0
    if args.action == "validate" and not result.get("valid", False):
        return 1 if strict else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
