"""Controlled vocabulary and validation for Club Context.

The product calls this feature family "Club Context".  The corresponding
research construct is psychosocial match context: sparse, acute events rather
than a generic media-sentiment score.  Values stored in SQLite come from these
enums so historical backfills and live inference use the same vocabulary.
"""

from __future__ import annotations

from enum import StrEnum


CONTEXT_TAXONOMY_VERSION = 1
MIN_ELIGIBLE_CONFIDENCE = 0.70


class EventCategory(StrEnum):
    LEADERSHIP_CHANGE = "leadership_change"
    SERIOUS_HUMAN_EVENT = "serious_human_event"
    TRIBUTE_MILESTONE = "tribute_milestone"
    CLUB_CRISIS = "club_crisis"


class EventPhase(StrEnum):
    ANNOUNCEMENT = "announcement"
    EFFECTIVE = "effective"
    FIRST_MATCH = "first_match"
    TRIBUTE = "tribute"
    FAREWELL = "farewell"
    MILESTONE = "milestone"
    ONGOING = "ongoing"
    RESOLUTION = "resolution"


class ConfirmationStatus(StrEnum):
    CONFIRMED = "confirmed"
    UNCONFIRMED = "unconfirmed"
    RUMOUR = "rumour"


class ReviewStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class Sensitivity(StrEnum):
    STANDARD = "standard"
    SENSITIVE = "sensitive"
    HIGHLY_SENSITIVE = "highly_sensitive"


class EntityType(StrEnum):
    TEAM = "team"
    PLAYER = "player"
    COACH = "coach"
    CLUB = "club"


class EntityRelationship(StrEnum):
    AFFECTED = "affected"
    SUBJECT = "subject"
    CONTEXT = "context"


class EvidenceRole(StrEnum):
    CONFIRMATION = "confirmation"
    CORROBORATION = "corroboration"
    DISCOVERY = "discovery"


class SourceClass(StrEnum):
    OFFICIAL_NRL = "official_nrl"
    OFFICIAL_CLUB = "official_club"
    PUBLIC_BROADCASTER = "public_broadcaster"
    REPUTABLE_MEDIA = "reputable_media"
    WIRE_SERVICE = "wire_service"
    DISCOVERY_INDEX = "discovery_index"
    MANUAL_RESEARCH = "manual_research"


class RightsStatus(StrEnum):
    FACTS_AND_LINKS = "facts_and_links"
    LICENSED = "licensed"
    DISCOVERY_ONLY = "discovery_only"
    PROHIBITED = "prohibited"
    UNKNOWN = "unknown"


class AcquisitionMethod(StrEnum):
    ADAPTER = "adapter"
    MANUAL = "manual"
    IMPORT = "import"


class KnownAtBasis(StrEnum):
    SOURCE_PUBLISHED_AT = "source_published_at"
    FIRST_OBSERVED_AT = "first_observed_at"
    MANUALLY_VERIFIED_AT = "manually_verified_at"


class IngestionMode(StrEnum):
    REFRESH = "refresh"
    BACKFILL = "backfill"
    VALIDATE = "validate"
    MANUAL = "manual"


class IngestionStatus(StrEnum):
    STARTED = "started"
    OK = "ok"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    FAILED = "failed"


class PredictionMode(StrEnum):
    LIVE = "live"
    TEST = "test"
    REFRESH = "refresh"
    PREVIEW = "preview"
    HISTORICAL = "historical"
    EVALUATION = "evaluation"


OFFICIAL_SOURCE_CLASSES = frozenset(
    {SourceClass.OFFICIAL_NRL, SourceClass.OFFICIAL_CLUB}
)
REPUTABLE_SOURCE_CLASSES = frozenset(
    {
        SourceClass.PUBLIC_BROADCASTER,
        SourceClass.REPUTABLE_MEDIA,
        SourceClass.WIRE_SERVICE,
    }
)
EVIDENCE_RIGHTS_STATUSES = frozenset(
    {RightsStatus.FACTS_AND_LINKS, RightsStatus.LICENSED}
)


def enum_value(enum_type: type[StrEnum], value: object, field_name: str) -> str:
    """Return the canonical stored value or raise a useful validation error."""

    try:
        return enum_type(value).value
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(member.value for member in enum_type)
        raise ValueError(f"{field_name} must be one of: {allowed}") from exc


def unit_interval(value: object, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number between 0 and 1") from exc
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return number
