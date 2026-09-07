"""Fail-soft reader-product helpers for the Club Context shadow layer.

Only verified facts and source links cross this boundary.  The prediction
stack never consumes prose produced for the email or site, and an unavailable
registry is indistinguishable from an ordinary round with no context cards.
"""

from __future__ import annotations

import math
import json
import re
from pathlib import Path
from urllib.parse import urlparse


SHADOW_DISCLAIMER = (
    "Experimental context only — it does not change the model probability."
)

_CATEGORY_LABELS = {
    "leadership_change": "Leadership change",
    "serious_human_event": "Serious human event",
    "tribute_milestone": "Tribute or milestone",
    "club_crisis": "Club disruption",
}

# These checks are deliberately conservative.  Generated prose that fails one
# of them should fall back to the locked factual summary rather than be repaired
# creatively around a sensitive event.
_PROHIBITED_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:guarantee(?:d)?|certain(?:ly)?|surely)\s+(?:a\s+)?(?:win|victory)\b",
        r"\b(?:tragedy|grief|diagnosis|illness|death)\b.{0,32}\b(?:betting\s+)?(?:boost|edge|angle)\b",
        r"\b(?:bet|wager|stake|punt)\s+(?:on|because of)\s+(?:the\s+)?(?:tragedy|grief|diagnosis|illness|death)\b",
        r"\bplaying for (?:him|her|them) (?:will|means|guarantees)\b",
        r"\b(?:diagnosis|condition|illness)\b.{0,40}\b(?:probably|likely|must|suggests?|worsen(?:ing)?|improv(?:e|ing))\b",
        r"\b(?:tribute|sacking|dismissal|tragedy|grief|death|illness)\b.{0,40}\b(?:will|must|is going to)\b.{0,24}\b(?:lift|inspire|motivate|fire up|distract|hurt|help|win|lose)\b",
    )
)


def category_label(value: object) -> str:
    raw = str(value or "").strip()
    return _CATEGORY_LABELS.get(raw, raw.replace("_", " ").strip().title() or "Club context")


def context_copy_is_safe(text: object, *, sensitive: bool = False) -> bool:
    """Return whether optional editorial copy passes the non-causal guard."""

    value = str(text or "").strip()
    if not value:
        return False
    if any(pattern.search(value) for pattern in _PROHIBITED_PATTERNS):
        return False
    if sensitive and re.search(r"\b(?:hilarious|laugh|joke|banter|silver lining)\b", value, re.I):
        return False
    return True


def deterministic_context_copy(card: dict) -> str:
    """Reg-flavoured copy assembled only from the locked factual summary."""

    factual = str(card.get("factual_summary") or card.get("summary") or "").strip()
    if not factual or bool(card.get("sensitive")):
        return factual
    prefix = {
        "leadership_change": "Reg's keeping this change on the radar: ",
        "tribute_milestone": "One for Reg's Context Watch: ",
        "club_crisis": "Reg's watching the off-field disruption: ",
    }.get(str(card.get("category") or ""), "On Reg's Context Watch: ")
    return prefix + factual


def safe_context_copy(card: dict, proposed: object | None = None) -> str:
    """Use Reg-style copy only when safe; otherwise return the locked fact."""

    factual = str(card.get("factual_summary") or card.get("summary") or "").strip()
    candidate = str(proposed or card.get("editorial_copy") or "").strip()
    if candidate and context_copy_is_safe(candidate, sensitive=bool(card.get("sensitive"))):
        return candidate
    if candidate:
        return factual
    return deterministic_context_copy(card)


def _valid_source_url(value: object) -> bool:
    try:
        parsed = urlparse(str(value).strip())
    except Exception:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _normal_game_ids(card: dict) -> list[int]:
    raw = card.get("game_ids")
    if raw is None and card.get("game_id") is not None:
        raw = [card.get("game_id")]
    if isinstance(raw, (str, bytes)):
        raw = [part.strip() for part in str(raw).split(",") if part.strip()]
    values: list[int] = []
    for item in raw or []:
        try:
            values.append(int(item))
        except (TypeError, ValueError):
            continue
    return sorted(set(values))


def normalize_context_cards(cards) -> list[dict]:
    """Validate, filter and deterministically order product-safe cards."""

    normalized: list[dict] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for raw in cards or []:
        if not isinstance(raw, dict) or raw.get("eligible") is False:
            continue
        factual = str(raw.get("factual_summary") or raw.get("summary") or "").strip()
        source_url = str(raw.get("source_url") or "").strip()
        source_title = str(raw.get("source_title") or raw.get("publisher") or "Source").strip()
        event_id = str(raw.get("event_id") or "").strip()
        if not event_id or not factual or not _valid_source_url(source_url):
            continue
        confidence_raw = raw.get("confidence", 0.0)
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.0
        if not math.isfinite(confidence):
            confidence = 0.0
        game_ids = _normal_game_ids(raw)
        key = (event_id, tuple(game_ids))
        if key in seen:
            continue
        seen.add(key)
        card = {
            "event_id": event_id,
            "game_ids": game_ids,
            "team_name": str(raw.get("team_name") or "").strip() or None,
            "category": str(raw.get("category") or "").strip(),
            "phase": str(raw.get("phase") or "").strip(),
            "factual_summary": factual,
            "editorial_copy": str(raw.get("editorial_copy") or "").strip() or None,
            "source_url": source_url,
            "source_title": source_title,
            "sensitive": bool(raw.get("sensitive")),
            "confidence": min(1.0, max(0.0, confidence)),
            "eligible": True,
            "shadow_mode": True,
        }
        card["display_copy"] = safe_context_copy(card)
        normalized.append(card)
    return sorted(
        normalized,
        key=lambda card: (
            min(card["game_ids"]) if card["game_ids"] else 2**63 - 1,
            card["team_name"] or "",
            card["event_id"],
        ),
    )


def load_context_cards(
    db_path: str | Path,
    game_ids=None,
    *,
    prediction_run_id: str | None = None,
) -> list[dict]:
    """Load the latest immutable context snapshot.  Never raises.

    The core schema owns eligibility and the as-of cutoff.  This function only
    turns its locked snapshot JSON into the small public card contract.
    """

    try:
        from .snapshots import load_prediction_context

        rows = load_prediction_context(
            db_path,
            game_ids=game_ids,
            prediction_run_id=prediction_run_id,
        )
        cards = []
        for row in rows.to_dict("records"):
            try:
                observed = json.loads(row.get("observed_context_json") or "[]")
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(observed, list):
                continue
            for raw in observed:
                if not isinstance(raw, dict):
                    continue
                card = dict(raw)
                card.setdefault("game_id", row.get("game_id"))
                card.setdefault("shadow_mode", bool(row.get("shadow_mode", True)))
                cards.append(card)
        return normalize_context_cards(cards)
    except Exception:
        # During an additive rollout (or against an old runtime DB), the table
        # may not exist yet.  Product delivery must remain unaffected.
        return []


__all__ = [
    "SHADOW_DISCLAIMER",
    "category_label",
    "context_copy_is_safe",
    "deterministic_context_copy",
    "load_context_cards",
    "normalize_context_cards",
    "safe_context_copy",
]
