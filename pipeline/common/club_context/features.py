"""Leakage-safe, sign-neutral Club Context features.

This transformer intentionally does not infer emotional valence.  It records
only exposure, timing, evidence strength and uncertainty for each side.  In
shadow mode callers can evaluate these columns without adding them to the
production predictor list.
"""

from __future__ import annotations

import datetime as dt
import math
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from ..lineups.normalization import normalize_team_name
from .registry import load_eligible_events
from .schema import context_tables_present
from .taxonomy import EventCategory, EventPhase, RightsStatus, Sensitivity
from .time import parse_datetime, round_target_map


CONTEXT_DECAY_HALFLIFE_DAYS = 7.0

CATEGORY_FEATURES = {
    EventCategory.LEADERSHIP_CHANGE.value: "leadership_count",
    EventCategory.SERIOUS_HUMAN_EVENT.value: "human_event_count",
    EventCategory.TRIBUTE_MILESTONE.value: "tribute_milestone_count",
    EventCategory.CLUB_CRISIS.value: "club_crisis_count",
}

PHASE_GROUPS = {
    "announcement_count": {EventPhase.ANNOUNCEMENT.value},
    "transition_count": {
        EventPhase.EFFECTIVE.value,
        EventPhase.FIRST_MATCH.value,
    },
    "match_pulse_count": {
        EventPhase.TRIBUTE.value,
        EventPhase.FAREWELL.value,
        EventPhase.MILESTONE.value,
    },
    "ongoing_count": {
        EventPhase.ONGOING.value,
        EventPhase.RESOLUTION.value,
    },
}

SIDE_METRICS = (
    "event_count",
    "max_salience",
    "max_confidence",
    "uncertainty",
    "source_diversity",
    "recency_weight",
    "official_count",
    "sensitive_count",
    "games_since_event",
    *CATEGORY_FEATURES.values(),
    *PHASE_GROUPS.keys(),
)

CONTEXT_FEATURE_COLUMNS = ["game_id"]
for _metric in SIDE_METRICS:
    CONTEXT_FEATURE_COLUMNS.extend(
        [
            f"club_context_{_metric}_home",
            f"club_context_{_metric}_away",
            f"club_context_{_metric}_delta",
        ]
    )
CONTEXT_FEATURE_COLUMNS.extend(
    ["club_context_data_available", "club_context_features_missing"]
)


def _empty_features(game_ids: Sequence[Any], *, available: bool) -> pd.DataFrame:
    data: dict[str, Any] = {"game_id": list(game_ids)}
    for column in CONTEXT_FEATURE_COLUMNS:
        if column == "game_id":
            continue
        if column == "club_context_data_available":
            data[column] = [float(available)] * len(game_ids)
        elif column == "club_context_features_missing":
            data[column] = [float(not available)] * len(game_ids)
        else:
            data[column] = [0.0] * len(game_ids)
    return pd.DataFrame(data, columns=CONTEXT_FEATURE_COLUMNS)


def fill_context_feature_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill post-merge defaults without claiming absent registry data exists."""

    for column in CONTEXT_FEATURE_COLUMNS:
        if column == "game_id" or column not in frame.columns:
            continue
        default = 1.0 if column == "club_context_features_missing" else 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return frame


def _table_names(con: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def _fixture_history(con: sqlite3.Connection) -> pd.DataFrame:
    """Best available true-UTC fixture history for games-since-event."""

    tables = _table_names(con)
    for table in ("feed_cache_fixtures", "footy_tipping_data"):
        if table not in tables:
            continue
        columns = {row[1] for row in con.execute(f"PRAGMA table_info({table})")}
        required = {"game_id", "team_home", "team_away"}
        if not required.issubset(columns):
            continue
        time_column = "start_time_utc" if "start_time_utc" in columns else "start_time"
        if time_column not in columns:
            continue
        try:
            history = pd.read_sql_query(
                f"""
                SELECT game_id, competition_year, round_id, team_home, team_away,
                       {time_column} AS start_time_utc
                FROM {table}
                WHERE {time_column} IS NOT NULL
                """,
                con,
            )
            if not history.empty:
                return history
        except Exception:
            continue
    return pd.DataFrame()


def _matches_frame(con: sqlite3.Connection, matches: Any) -> pd.DataFrame:
    if isinstance(matches, pd.DataFrame):
        frame = matches.copy()
    else:
        frame = pd.DataFrame(list(matches or []))
    if frame.empty or "game_id" not in frame.columns:
        return frame

    required = {
        "competition_year",
        "round_id",
        "team_home",
        "team_away",
        "start_time_utc",
    }
    if required.issubset(frame.columns) and frame[list(required)].notna().all().all():
        return frame

    history = _fixture_history(con)
    if history.empty:
        # A prepared frame often calls the real UTC column start_time.  Use it
        # only as a final compatibility fallback; callers should pass the true
        # UTC value whenever available.
        if "start_time_utc" not in frame.columns and "start_time" in frame.columns:
            frame["start_time_utc"] = frame["start_time"]
        return frame

    frame["game_id"] = pd.to_numeric(frame["game_id"], errors="coerce")
    history["game_id"] = pd.to_numeric(history["game_id"], errors="coerce")
    history = history.dropna(subset=["game_id"]).drop_duplicates("game_id", keep="last")
    hydrate_columns = [
        column for column in sorted(required - {"game_id"}) if column in history.columns
    ]
    hydrated = frame.merge(
        history[["game_id", *hydrate_columns]],
        on="game_id",
        how="left",
        suffixes=("", "__fixture"),
    )
    for column in hydrate_columns:
        fallback = f"{column}__fixture"
        if column not in hydrated.columns:
            hydrated[column] = hydrated[fallback]
        elif fallback in hydrated.columns:
            hydrated[column] = hydrated[column].where(
                hydrated[column].notna(), hydrated[fallback]
            )
        if fallback in hydrated.columns:
            hydrated = hydrated.drop(columns=[fallback])
    return hydrated


def _event_team_keys(event: Mapping[str, Any]) -> set[str]:
    return {
        str(entity.get("team_key"))
        for entity in event.get("entities", [])
        if entity.get("relationship") in {"affected", "subject"}
    }


def _games_since_event(
    history: pd.DataFrame,
    *,
    team_key: str,
    effective_at: dt.datetime,
    match_at: dt.datetime,
) -> int:
    if history.empty:
        return 0
    count = 0
    for row in history.to_dict("records"):
        if team_key not in {
            normalize_team_name(row.get("team_home")),
            normalize_team_name(row.get("team_away")),
        }:
            continue
        try:
            kickoff = parse_datetime(row.get("start_time_utc"))
        except ValueError:
            continue
        if effective_at <= kickoff < match_at:
            count += 1
    return count


def _side_values(
    events: Sequence[Mapping[str, Any]],
    *,
    team_key: str,
    match_at: dt.datetime,
    history: pd.DataFrame,
) -> dict[str, float]:
    relevant = [event for event in events if team_key in _event_team_keys(event)]
    values = {metric: 0.0 for metric in SIDE_METRICS}
    if not relevant:
        return values

    values["event_count"] = float(len(relevant))
    values["max_salience"] = max(float(event["salience"]) for event in relevant)
    values["max_confidence"] = max(float(event["confidence"]) for event in relevant)
    values["uncertainty"] = sum(
        1.0 - float(event["confidence"]) for event in relevant
    ) / len(relevant)

    publisher_keys: set[str] = set()
    official_event_ids: set[int] = set()
    recency_weights: list[float] = []
    games_since: list[int] = []
    for event in relevant:
        for source in _qualifying_sources(event):
            if source.get("evidence_role") not in {"confirmation", "corroboration"}:
                continue
            publisher_keys.add(str(source.get("independence_key") or ""))
            if bool(source.get("is_official")):
                official_event_ids.add(int(event["event_id"]))
        effective = parse_datetime(event["effective_from_utc"])
        age_days = max(0.0, (match_at - effective).total_seconds() / 86400.0)
        recency_weights.append(math.exp(-math.log(2.0) * age_days / CONTEXT_DECAY_HALFLIFE_DAYS))
        games_since.append(
            _games_since_event(
                history,
                team_key=team_key,
                effective_at=effective,
                match_at=match_at,
            )
        )

    values["source_diversity"] = float(len(publisher_keys - {""}))
    values["recency_weight"] = max(recency_weights, default=0.0)
    values["official_count"] = float(len(official_event_ids))
    values["sensitive_count"] = float(
        sum(event["sensitivity"] != Sensitivity.STANDARD.value for event in relevant)
    )
    values["games_since_event"] = float(min(games_since, default=0))

    for category, metric in CATEGORY_FEATURES.items():
        values[metric] = float(sum(event["category"] == category for event in relevant))
    for metric, phases in PHASE_GROUPS.items():
        values[metric] = float(sum(event["phase"] in phases for event in relevant))
    return values


def _primary_source(event: Mapping[str, Any]) -> Mapping[str, Any]:
    for source in _qualifying_sources(event):
        if source.get("evidence_role") in {"confirmation", "corroboration"}:
            return source
    return {}


def _qualifying_sources(event: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        source
        for source in event.get("sources", [])
        if source.get("parse_status") == "ok"
        and source.get("rights_status")
        in {RightsStatus.FACTS_AND_LINKS.value, RightsStatus.LICENSED.value}
    ]


def _observed_cards(
    events: Sequence[Mapping[str, Any]],
    *,
    game_id: int,
    team_names: Mapping[str, str],
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for event in events:
        source = _primary_source(event)
        for team_key in sorted(_event_team_keys(event) & set(team_names)):
            cards.append(
                {
                    "event_id": int(event["event_id"]),
                    "event_key": str(event["event_key"]),
                    "game_ids": [int(game_id)],
                    "team_key": team_key,
                    "team_name": team_names[team_key],
                    "category": str(event["category"]),
                    "phase": str(event["phase"]),
                    "factual_summary": str(event["factual_summary"]),
                    "source_url": str(source.get("canonical_url") or ""),
                    "source_title": str(source.get("title") or ""),
                    "sensitive": event["sensitivity"] != Sensitivity.STANDARD.value,
                    "sensitivity": str(event["sensitivity"]),
                    "confidence": float(event["confidence"]),
                    "salience": float(event["salience"]),
                    "known_at_utc": str(event["known_at_utc"]),
                    "effective_from_utc": str(event["effective_from_utc"]),
                    "expires_at_utc": event.get("expires_at_utc"),
                    "eligible": True,
                    "shadow_mode": True,
                }
            )
    return sorted(cards, key=lambda card: (card["team_key"], card["event_id"]))


def resolve_context_for_matches(
    con: sqlite3.Connection,
    matches: Any,
    *,
    decision_at_utc: Any | None = None,
) -> tuple[pd.DataFrame, dict[int, list[dict[str, Any]]]]:
    """Resolve feature rows and the exact observed facts from one DB view.

    With no explicit decision time, each historical round uses 11:00 Sydney on
    its earliest-fixture date.  Live/test/refresh callers must capture and pass
    one actual decision time after ingestion and before inference.
    """

    frame = _matches_frame(con, matches)
    game_ids = frame.get("game_id", pd.Series(dtype=float)).tolist()
    if frame.empty:
        return _empty_features([], available=context_tables_present(con)), {}
    if not context_tables_present(con):
        return _empty_features(game_ids, available=False), {}

    required = {
        "game_id",
        "competition_year",
        "round_id",
        "team_home",
        "team_away",
        "start_time_utc",
    }
    if not required.issubset(frame.columns):
        return _empty_features(game_ids, available=False), {}

    try:
        targets = round_target_map(frame.to_dict("records"))
    except Exception:
        return _empty_features(game_ids, available=False), {}
    history = _fixture_history(con)
    if history.empty:
        history = frame[
            ["game_id", "competition_year", "round_id", "team_home", "team_away", "start_time_utc"]
        ].copy()

    rows: list[dict[str, Any]] = []
    observed: dict[int, list[dict[str, Any]]] = {}
    for match in frame.to_dict("records"):
        game_id = int(float(match["game_id"]))
        try:
            year = int(float(match["competition_year"]))
            round_id = int(float(match["round_id"]))
            match_at = parse_datetime(match["start_time_utc"])
            decision = (
                parse_datetime(decision_at_utc)
                if decision_at_utc is not None
                else targets[(year, round_id)]
            )
            home_key = normalize_team_name(match["team_home"])
            away_key = normalize_team_name(match["team_away"])
            events = load_eligible_events(
                con,
                [home_key, away_key],
                decision_at_utc=decision,
                match_at_utc=match_at,
            )
            home_values = _side_values(
                events,
                team_key=home_key,
                match_at=match_at,
                history=history,
            )
            away_values = _side_values(
                events,
                team_key=away_key,
                match_at=match_at,
                history=history,
            )
            row: dict[str, Any] = {"game_id": match["game_id"]}
            for metric in SIDE_METRICS:
                row[f"club_context_{metric}_home"] = home_values[metric]
                row[f"club_context_{metric}_away"] = away_values[metric]
                row[f"club_context_{metric}_delta"] = (
                    home_values[metric] - away_values[metric]
                )
            row["club_context_data_available"] = 1.0
            row["club_context_features_missing"] = 0.0
            observed[game_id] = _observed_cards(
                events,
                game_id=game_id,
                team_names={
                    home_key: str(match["team_home"]),
                    away_key: str(match["team_away"]),
                },
            )
            rows.append(row)
        except Exception:
            missing = _empty_features([match["game_id"]], available=False).iloc[0].to_dict()
            rows.append(missing)
            observed[game_id] = []
    return pd.DataFrame(rows, columns=CONTEXT_FEATURE_COLUMNS), observed


def build_context_match_features(
    db_path: str | Path,
    matches: Any,
    *,
    decision_at_utc: Any | None = None,
) -> pd.DataFrame:
    """Public fail-soft transformer; it never creates registry tables."""

    try:
        game_ids = (
            matches["game_id"].tolist()
            if isinstance(matches, pd.DataFrame) and "game_id" in matches
            else [row.get("game_id") for row in (matches or [])]
        )
    except Exception:
        game_ids = []
    path = Path(db_path)
    if not path.exists():
        return _empty_features(game_ids, available=False)
    try:
        with closing(sqlite3.connect(str(path))) as con:
            features, _ = resolve_context_for_matches(
                con, matches, decision_at_utc=decision_at_utc
            )
            return features
    except Exception:
        return _empty_features(game_ids, available=False)
