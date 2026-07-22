"""Drive-backed production-email idempotency markers.

The SQLite ledger is useful after a successful runtime push, but it cannot
protect against the dangerous window where SMTP succeeds and the runner dies
before the DB reaches Drive. After deterministic credentials and recipient
preparation, a per-round marker is created immediately before SMTP. A
``pending`` marker is deliberately treated as uncertain and blocks automatic
resends until a human reconciles it; this includes a partially refused send.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import re
import tempfile
import uuid

from pipeline.ops import state_sync


DELIVERIES_FOLDER = "deliveries"
SCHEMA_VERSION = 1
MARKER_STATUSES = frozenset({"pending", "sent"})
MARKER_NAME_PATTERN = re.compile(r"([1-9][0-9]*)-round-([1-9][0-9]*)\.json")


def _marker_name(competition_year, round_id) -> str:
    competition_year = int(competition_year)
    round_id = int(round_id)
    if competition_year < 1 or round_id < 1:
        raise ValueError("delivery marker year and round must be positive integers")
    return f"{competition_year}-round-{round_id}.json"


def _validate_marker(name, payload) -> dict:
    match = MARKER_NAME_PATTERN.fullmatch(str(name))
    if not match:
        raise RuntimeError(f"delivery marker filename is invalid: {name}")
    expected_year, expected_round = (int(value) for value in match.groups())
    if not isinstance(payload, dict):
        raise RuntimeError(f"delivery marker {name} must be a JSON object")
    schema_version = payload.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != SCHEMA_VERSION
    ):
        raise RuntimeError(f"delivery marker {name} has an unsupported schema")

    year = payload.get("competition_year")
    round_id = payload.get("round_id")
    if (
        isinstance(year, bool)
        or not isinstance(year, int)
        or year != expected_year
        or isinstance(round_id, bool)
        or not isinstance(round_id, int)
        or round_id != expected_round
    ):
        raise RuntimeError(
            f"delivery marker {name} year/round does not match its filename"
        )

    status = payload.get("status")
    if status not in MARKER_STATUSES:
        raise RuntimeError(
            f"delivery marker {name} has invalid status {status!r}"
        )
    attempt_id = payload.get("attempt_id")
    if not isinstance(attempt_id, str) or not attempt_id.strip():
        raise RuntimeError(f"delivery marker {name} has an invalid attempt_id")
    return payload


def _context(root, *, create=False):
    root = pathlib.Path(root)
    service = state_sync.drive_service(root / "service-account-token.json")
    state_id = (
        state_sync._state_folder(service, root)
        if create
        else state_sync._existing_state_folder(service, root)
    )
    folder_id = (
        state_sync.get_or_create_folder(service, DELIVERIES_FOLDER, state_id)
        if create
        else state_sync.find_folder_id(service, DELIVERIES_FOLDER, state_id)
    )
    return root, service, folder_id


def _download_marker(service, folder_id, name):
    # Validate before using a Drive filename as a temporary local path.
    if not MARKER_NAME_PATTERN.fullmatch(str(name)):
        raise RuntimeError(f"delivery marker filename is invalid: {name}")
    if not folder_id:
        return None
    file_id = state_sync.find_file_id(service, folder_id, name)
    if not file_id:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / name
        state_sync.download_to(service, file_id, path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"delivery marker {name} is invalid JSON") from exc
    return _validate_marker(name, payload)


def get_delivery(root, competition_year, round_id):
    _, service, folder_id = _context(root, create=False)
    return _download_marker(
        service, folder_id, _marker_name(competition_year, round_id)
    )


def begin_delivery(root, competition_year, round_id, source="actions_live") -> dict:
    """Create a pending marker, returning ``allowed=False`` if one exists."""
    _, service, folder_id = _context(root, create=True)
    name = _marker_name(competition_year, round_id)
    existing = _download_marker(service, folder_id, name)
    if existing is not None:
        return {
            "allowed": False,
            "reason": f"delivery is already {existing.get('status', 'unknown')}",
            "marker": existing,
        }

    now = dt.datetime.now(dt.timezone.utc).isoformat()
    marker = {
        "schema_version": SCHEMA_VERSION,
        "competition_year": int(competition_year),
        "round_id": int(round_id),
        "status": "pending",
        "attempt_id": str(uuid.uuid4()),
        "source": str(source),
        "created_at_utc": now,
        "updated_at_utc": now,
        "note": "Pending means SMTP outcome may be uncertain; never resend automatically.",
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / name
        path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        try:
            state_sync.upload_create_only(
                service, folder_id, name, path, "application/json"
            )
        except FileExistsError:
            existing = _download_marker(service, folder_id, name)
            return {
                "allowed": False,
                "reason": "another runner claimed this round",
                "marker": existing,
            }
    return {"allowed": True, "reason": "delivery claimed", "marker": marker}


def mark_sent(root, competition_year, round_id, attempt_id, recipients_count=None) -> dict:
    """Turn the caller's pending marker into sent after confirmed SMTP success."""
    _, service, folder_id = _context(root, create=False)
    name = _marker_name(competition_year, round_id)
    marker = _download_marker(service, folder_id, name)
    if marker is None:
        raise RuntimeError("delivery marker disappeared before it could be marked sent")
    if marker.get("attempt_id") != attempt_id:
        raise RuntimeError("delivery marker belongs to a different attempt")
    if marker.get("status") == "sent":
        return marker
    if marker.get("status") != "pending":
        raise RuntimeError(f"delivery marker has unexpected status {marker.get('status')!r}")
    marker["status"] = "sent"
    marker["sent_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    marker["updated_at_utc"] = marker["sent_at_utc"]
    marker["recipients_count"] = recipients_count
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / name
        path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        state_sync.upload_or_update(
            service, folder_id, name, path, "application/json"
        )
    return marker


def list_deliveries(root) -> list:
    """Return marker summaries for status output, newest first."""
    _, service, folder_id = _context(root, create=False)
    if not folder_id:
        return []
    markers = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        for item in state_sync.list_files(service, folder_id):
            name = item.get("name", "")
            if not name.endswith(".json"):
                continue
            marker = _download_marker(service, folder_id, name)
            if marker is not None:
                markers.append(marker)
    return sorted(
        markers,
        key=lambda item: str(item.get("updated_at_utc", item.get("created_at_utc", ""))),
        reverse=True,
    )
