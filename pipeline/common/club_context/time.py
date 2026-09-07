"""Leakage-safe timestamp helpers for Club Context."""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping, Sequence
from typing import Any
from zoneinfo import ZoneInfo


UTC = dt.timezone.utc
SYDNEY = ZoneInfo("Australia/Sydney")
ROUND_TARGET_HOUR = 11


def parse_datetime(value: Any) -> dt.datetime:
    """Parse common pipeline timestamp shapes into a UTC-aware datetime.

    Numeric inputs are true Unix timestamps.  Naive textual/datetime inputs
    are interpreted as UTC, matching the rest of the Python ingestion code.
    Callers calculating a round cutoff should prefer ``start_time_utc``: the
    legacy ``start_time`` field is a venue-local wall clock encoded as UTC.
    """

    if isinstance(value, dt.datetime):
        parsed = value
    elif isinstance(value, dt.date):
        parsed = dt.datetime.combine(value, dt.time(), tzinfo=UTC)
    elif isinstance(value, bool) or value is None:
        raise ValueError("timestamp is missing")
    elif isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("timestamp must be finite")
        # Be tolerant of millisecond epochs from an external discovery feed.
        if abs(number) >= 100_000_000_000:
            number /= 1000.0
        parsed = dt.datetime.fromtimestamp(number, tz=UTC)
    else:
        raw = str(value).strip()
        if not raw:
            raise ValueError("timestamp is missing")
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = dt.datetime.fromisoformat(raw)
        except ValueError:
            try:
                number = float(raw)
            except ValueError as exc:
                raise ValueError(f"invalid timestamp: {value!r}") from exc
            return parse_datetime(number)

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def utc_iso(value: Any) -> str:
    """Canonical UTC ISO text, second precision, suitable for SQLite."""

    return parse_datetime(value).replace(microsecond=0).isoformat()


def utc_now() -> dt.datetime:
    return dt.datetime.now(UTC).replace(microsecond=0)


def capture_live_decision_at_utc(now: Any | None = None) -> dt.datetime:
    """Capture one feature-freeze instant to share across the whole round."""

    return (utc_now() if now is None else parse_datetime(now)).replace(microsecond=0)


def _records(fixtures: Any) -> list[Mapping[str, Any]]:
    if fixtures is None:
        return []
    if hasattr(fixtures, "to_dict"):
        return list(fixtures.to_dict("records"))
    if isinstance(fixtures, Mapping):
        return [fixtures]
    if isinstance(fixtures, Sequence) and not isinstance(fixtures, (str, bytes)):
        return list(fixtures)
    return list(fixtures)


def _kickoff(row: Mapping[str, Any]) -> dt.datetime:
    value = row.get("start_time_utc")
    if value is None or str(value).strip() == "":
        value = row.get("start_time")
    return parse_datetime(value)


def round_target_at_utc(fixtures: Any) -> dt.datetime:
    """11:00 Sydney on the local date of the round's earliest fixture.

    The input must describe exactly one competition-year/round when those
    columns are present.  Every match in that round therefore shares one
    historical information cutoff, including games played later in the week.
    """

    rows = _records(fixtures)
    if not rows:
        raise ValueError("at least one fixture is required")

    identities = {
        (row.get("competition_year"), row.get("round_id"))
        for row in rows
        if row.get("competition_year") is not None or row.get("round_id") is not None
    }
    if len(identities) > 1:
        raise ValueError("round_target_at_utc requires fixtures from one round")

    earliest = min(_kickoff(row) for row in rows)
    local = earliest.astimezone(SYDNEY)
    target = local.replace(hour=ROUND_TARGET_HOUR, minute=0, second=0, microsecond=0)
    return target.astimezone(UTC)


def round_target_map(fixtures: Any) -> dict[tuple[int, int], dt.datetime]:
    """Return a deterministic historical decision cutoff for each round."""

    rows = _records(fixtures)
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        try:
            key = (int(row["competition_year"]), int(row["round_id"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("fixtures require competition_year and round_id") from exc
        grouped.setdefault(key, []).append(row)
    return {key: round_target_at_utc(group) for key, group in sorted(grouped.items())}
