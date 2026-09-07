"""Rights-aware boundaries for optional Club Context discovery adapters.

This module deliberately performs no network calls.  Discovery providers are
pluggable candidate generators; registry eligibility is decided later from
persisted evidence, never from an adapter's sentiment or ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .taxonomy import RightsStatus, SourceClass, enum_value


@dataclass(frozen=True)
class SourcePolicy:
    source_key: str
    source_class: SourceClass
    rights_status: RightsStatus
    automated_discovery_allowed: bool
    automated_extraction_allowed: bool
    attribution_required: bool = True
    terms_url: str | None = None

    def __post_init__(self) -> None:
        if not self.source_key.strip():
            raise ValueError("source_key is required")
        enum_value(SourceClass, self.source_class, "source_class")
        enum_value(RightsStatus, self.rights_status, "rights_status")
        if self.rights_status in {RightsStatus.PROHIBITED, RightsStatus.UNKNOWN}:
            if self.automated_discovery_allowed or self.automated_extraction_allowed:
                raise ValueError("prohibited/unknown sources cannot enable automation")
        if (
            self.rights_status == RightsStatus.DISCOVERY_ONLY
            and self.automated_extraction_allowed
        ):
            raise ValueError("discovery-only sources cannot enable extraction")


@dataclass(frozen=True)
class DiscoveryCandidate:
    url: str
    title: str
    publisher_key: str
    source_class: SourceClass
    published_at_utc: str | None = None
    snippet: str | None = None


@runtime_checkable
class DiscoveryAdapter(Protocol):
    """A source-specific, side-effect-free interface from the core's view."""

    policy: SourcePolicy

    def discover(
        self,
        *,
        query: str,
        start_at_utc: str | None = None,
        end_at_utc: str | None = None,
        limit: int = 100,
    ) -> list[DiscoveryCandidate]: ...


def adapter_can_run(adapter: DiscoveryAdapter, *, extract: bool = False) -> bool:
    """Fail-closed permission check before orchestration invokes an adapter."""

    try:
        policy = adapter.policy
        if extract:
            return bool(policy.automated_extraction_allowed)
        return bool(policy.automated_discovery_allowed)
    except Exception:
        return False
