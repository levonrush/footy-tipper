"""Leakage-safe Club Context features.

This transformer records exposure, timing, evidence strength, uncertainty and
attention volume for each side.  It still infers no emotional valence: nothing
here reads the tone of any article, and no article text, embedding or generated
summary reaches a feature.

Direction is a different matter from tone.  Every ``*_delta`` column is home
minus away, so which club an event happened to has always been expressible; the
v2 block simply names that orientation explicitly, because the offset estimator
in :mod:`pipeline.common.club_context.materiality` needs one signed exposure
rather than fifty-one unsigned ones.  In shadow mode callers can evaluate these
columns without adding them to the production predictor list.
"""

from __future__ import annotations

import bisect
import datetime as dt
import math
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from ..lineups.normalization import normalize_team_name
from .attention import ATTENTION_METRICS, AttentionIndex, load_attention_index
from .registry import load_eligible_events
from .schema import context_tables_present
from .taxonomy import (
    EventCategory,
    EventDisposition,
    EventPhase,
    RightsStatus,
    Sensitivity,
)
from .time import parse_datetime, round_target_map


CONTEXT_DECAY_HALFLIFE_DAYS = 7.0

CATEGORY_FEATURES = {
    EventCategory.LEADERSHIP_CHANGE.value: "leadership_count",
    EventCategory.SERIOUS_HUMAN_EVENT.value: "human_event_count",
    EventCategory.TRIBUTE_MILESTONE.value: "tribute_milestone_count",
    EventCategory.CLUB_CRISIS.value: "club_crisis_count",
}

CATEGORY_EXPOSURE = {
    EventCategory.LEADERSHIP_CHANGE.value: "leadership",
    EventCategory.SERIOUS_HUMAN_EVENT.value: "human_event",
    EventCategory.TRIBUTE_MILESTONE.value: "tribute_milestone",
    EventCategory.CLUB_CRISIS.value: "club_crisis",
    EventCategory.JUDICIARY_SANCTION.value: "judiciary_sanction",
    EventCategory.CONTRACT_EXIT.value: "contract_exit",
    EventCategory.OWNERSHIP_GOVERNANCE.value: "ownership_governance",
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

# The v1 block is frozen so the 7 September 2026 materiality report stays
# reproducible against an unchanged column contract.
V1_SIDE_METRICS = (
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

# v2 adds continuous shape where v1 used integer counts, the timing an event
# carried, the reviewed factual attributes, dense regime state, and attention
# volume.  All of it remains tone-free.
V2_SIDE_METRICS = (
    "exposure_index",
    "peak_exposure",
    *(f"{name}_exposure" for name in CATEGORY_EXPOSURE.values()),
    "days_since_effective",
    "notice_days",
    "window_fraction",
    "evidence_strength",
    "involuntary_exposure",
    "voluntary_exposure",
    "commemorative_exposure",
    "availability_impact",
    "magnitude",
    "regime_matches",
    "regime_censored",
    *ATTENTION_METRICS,
)

SIDE_METRICS = (*V1_SIDE_METRICS, *V2_SIDE_METRICS)

# The one explicitly signed pair.  Orientation is a fact about which club the
# event happened to, and it was already implicit in every ``*_delta`` column;
# naming it lets a one-parameter estimator pool both sides into a single
# coefficient instead of asking a 300-predictor GBM to rediscover the split.
ORIENTATION_COLUMNS = (
    "club_context_affected_side",
    "club_context_affected_exposure",
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
    [
        *ORIENTATION_COLUMNS,
        # Per-side, because GDELT coverage of one club can be present while the
        # other's is not; a shared flag would hide that.
        "club_context_attention_missing_home",
        "club_context_attention_missing_away",
        "club_context_attention_available",
        "club_context_data_available",
        "club_context_features_missing",
    ]
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
        elif column in {
            "club_context_attention_missing_home",
            "club_context_attention_missing_away",
        }:
            # No features were built, so no attention was measured either.
            data[column] = [1.0] * len(game_ids)
        else:
            data[column] = [0.0] * len(game_ids)
    return pd.DataFrame(data, columns=CONTEXT_FEATURE_COLUMNS)


def fill_context_feature_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill post-merge defaults without claiming absent registry data exists."""

    for column in CONTEXT_FEATURE_COLUMNS:
        if column == "game_id" or column not in frame.columns:
            continue
        default = (
            1.0
            if column
            in {
                "club_context_features_missing",
                "club_context_attention_missing_home",
                "club_context_attention_missing_away",
            }
            else 0.0
        )
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


def _team_kickoffs(history: pd.DataFrame) -> dict[str, list[dt.datetime]]:
    """One pass over the fixture history, indexed by normalized team key.

    ``_games_since_event`` is called twice per match and again for regime state,
    so re-scanning the whole history each time is quadratic on a full-corpus
    evaluation.  Building the index once keeps the transformer usable across
    thousands of rows.
    """

    index: dict[str, list[dt.datetime]] = {}
    if history is None or history.empty:
        return index
    for row in history.itertuples(index=False):
        try:
            kickoff = parse_datetime(getattr(row, "start_time_utc", None))
        except (TypeError, ValueError):
            continue
        for name in (getattr(row, "team_home", None), getattr(row, "team_away", None)):
            key = normalize_team_name(name)
            if key:
                index.setdefault(key, []).append(kickoff)
    for values in index.values():
        values.sort()
    return index


def _games_since_event(
    kickoffs: Mapping[str, Sequence[dt.datetime]],
    *,
    team_key: str,
    effective_at: dt.datetime,
    match_at: dt.datetime,
) -> int:
    values = kickoffs.get(team_key)
    if not values:
        return 0
    return bisect.bisect_left(values, match_at) - bisect.bisect_left(values, effective_at)


def _decay(effective: dt.datetime, match_at: dt.datetime) -> float:
    age_days = max(0.0, (match_at - effective).total_seconds() / 86400.0)
    return math.exp(-math.log(2.0) * age_days / CONTEXT_DECAY_HALFLIFE_DAYS)


def _regime_matches(
    handovers: Mapping[str, list[dt.datetime]],
    kickoffs: Mapping[str, Sequence[dt.datetime]],
    *,
    team_key: str,
    match_at: dt.datetime,
) -> tuple[float, float]:
    """Matches played under the current in-season regime, and whether censored.

    The audited census covers in-season handovers only, so a club whose coach
    was appointed between seasons has no handover to count from.  Those rows are
    censored at the season opener and flagged, rather than being given a
    tenure number the registry cannot support.
    """

    season_start = dt.datetime(match_at.year, 1, 1, tzinfo=dt.timezone.utc)
    effective_dates = [
        value
        for value in handovers.get(team_key, [])
        if season_start <= value <= match_at
    ]
    if effective_dates:
        anchor = max(effective_dates)
        censored = 0.0
    else:
        # No in-season handover this season, so the regime began at or before
        # the opener and its true age is unknown.  Count from the season start
        # and say so, rather than reaching back to a handover several seasons
        # old that a later off-season appointment may have superseded.
        anchor = season_start
        censored = 1.0
    return (
        float(
            _games_since_event(
                kickoffs, team_key=team_key, effective_at=anchor, match_at=match_at
            )
        ),
        censored,
    )


def _side_values(
    events: Sequence[Mapping[str, Any]],
    *,
    team_key: str,
    match_at: dt.datetime,
    kickoffs: Mapping[str, Sequence[dt.datetime]],
    handovers: Mapping[str, list[dt.datetime]] | None = None,
    attention: AttentionIndex | None = None,
    decision_at: dt.datetime | None = None,
) -> dict[str, float]:
    relevant = [event for event in events if team_key in _event_team_keys(event)]
    values = {metric: 0.0 for metric in SIDE_METRICS}

    # Regime state and attention are dense: they are defined for every club in
    # every round, event or no event, so they are filled before the early exit.
    if handovers is not None:
        regime, censored = _regime_matches(
            handovers, kickoffs, team_key=team_key, match_at=match_at
        )
        values["regime_matches"] = regime
        values["regime_censored"] = censored
    else:
        values["regime_censored"] = 1.0
    if attention is not None and decision_at is not None:
        measured = attention.values(team_key, decision_at)
        for metric in ATTENTION_METRICS:
            values[metric] = measured[metric]
        values["attention_missing"] = measured["attention_missing"]
    else:
        values["attention_missing"] = 1.0

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
                kickoffs,
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

    # --- v2 continuous shape -------------------------------------------------
    # An integer count forces a tree to split on 0/1/2.  Weighting each event by
    # how strongly it was evidenced and how recent it is gives the same
    # information a usable gradient, which matters far more at 89 exposed rows
    # than at 3,180.
    exposures = []
    for event in relevant:
        effective = parse_datetime(event["effective_from_utc"])
        weight = (
            float(event["salience"])
            * float(event["confidence"])
            * _decay(effective, match_at)
        )
        exposures.append((event, effective, weight))

    values["exposure_index"] = float(sum(weight for _, _, weight in exposures))
    values["peak_exposure"] = float(max((weight for _, _, weight in exposures), default=0.0))
    for category, name in CATEGORY_EXPOSURE.items():
        values[f"{name}_exposure"] = float(
            sum(weight for event, _, weight in exposures if event["category"] == category)
        )

    ages = [
        max(0.0, (match_at - effective).total_seconds() / 86400.0)
        for _, effective, _ in exposures
    ]
    values["days_since_effective"] = float(min(ages, default=0.0))

    notices = []
    windows = []
    for event, effective, _ in exposures:
        try:
            known = parse_datetime(event["known_at_utc"])
        except (KeyError, TypeError, ValueError):
            continue
        notices.append(max(0.0, (effective - known).total_seconds() / 86400.0))
        expiry = event.get("expires_at_utc")
        if not expiry:
            continue
        try:
            expires = parse_datetime(expiry)
        except ValueError:
            continue
        span = (expires - effective).total_seconds()
        if span > 0:
            elapsed = (match_at - effective).total_seconds()
            windows.append(min(1.0, max(0.0, elapsed / span)))
    values["notice_days"] = float(max(notices, default=0.0))
    values["window_fraction"] = float(max(windows, default=0.0))

    values["evidence_strength"] = float(
        values["official_count"] + 0.5 * values["source_diversity"]
    )

    # Reviewed factual attributes.  Disposition is what the source said about
    # how the event came about, never a reading of its tone.
    for disposition, metric in (
        (EventDisposition.INVOLUNTARY.value, "involuntary_exposure"),
        (EventDisposition.VOLUNTARY.value, "voluntary_exposure"),
        (EventDisposition.COMMEMORATIVE.value, "commemorative_exposure"),
    ):
        values[metric] = float(
            sum(
                weight
                for event, _, weight in exposures
                if str(event.get("disposition") or "") == disposition
            )
        )
    values["availability_impact"] = float(
        max((float(event.get("availability_impact") or 0.0) for event in relevant), default=0.0)
    )
    values["magnitude"] = float(
        max((float(event.get("magnitude") or 0.0) for event in relevant), default=0.0)
    )
    return values


def _load_regime_handovers(
    con: sqlite3.Connection, decision_at: dt.datetime | None
) -> dict[str, list[dt.datetime]]:
    """Effective in-season leadership handovers known by the decision cutoff.

    Unlike :func:`load_eligible_events` this deliberately ignores the event
    expiry window: a handover keeps defining the regime long after its news
    window closes.  The evidence and cutoff rules still apply.
    """

    handovers: dict[str, list[dt.datetime]] = {}
    try:
        rows = con.execute(
            """
            SELECT ee.team_key, e.effective_from_utc, e.known_at_utc
            FROM context_events e
            JOIN context_event_entities ee ON ee.event_id = e.event_id
            WHERE e.category = ?
              AND e.phase = ?
              AND e.review_status = 'approved'
              AND e.confirmation_status = 'confirmed'
              AND ee.relationship = 'affected'
            """,
            (EventCategory.LEADERSHIP_CHANGE.value, EventPhase.EFFECTIVE.value),
        ).fetchall()
    except Exception:
        return handovers
    for team_key, effective, known in rows:
        try:
            if decision_at is not None and parse_datetime(known) > decision_at:
                continue
            handovers.setdefault(str(team_key), []).append(parse_datetime(effective))
        except (TypeError, ValueError):
            continue
    for values in handovers.values():
        values.sort()
    return handovers


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


def _orientation(*, home_exposure: float, away_exposure: float) -> dict[str, float]:
    """Which side carries the event, and how strongly.

    ``affected_side`` is +1 when the home club is the affected one, -1 for the
    away club, and 0 when neither or both sides carry equal exposure. This is a
    fact about the fixture, not an estimate of the effect: the sign of the
    effect is still fitted from held-out data.
    """

    difference = float(home_exposure) - float(away_exposure)
    if abs(difference) < 1e-12:
        return {
            "club_context_affected_side": 0.0,
            "club_context_affected_exposure": 0.0,
        }
    side = 1.0 if difference > 0 else -1.0
    return {
        "club_context_affected_side": side,
        "club_context_affected_exposure": difference,
    }


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

    attention = load_attention_index(con)
    kickoffs = _team_kickoffs(history)
    rows: list[dict[str, Any]] = []
    observed: dict[int, list[dict[str, Any]]] = {}
    handover_cache: dict[Any, dict[str, list[dt.datetime]]] = {}
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
            if decision not in handover_cache:
                handover_cache[decision] = _load_regime_handovers(con, decision)
            handovers = handover_cache[decision]
            side_kwargs = dict(
                match_at=match_at,
                kickoffs=kickoffs,
                handovers=handovers,
                attention=attention,
                decision_at=decision,
            )
            home_values = _side_values(events, team_key=home_key, **side_kwargs)
            away_values = _side_values(events, team_key=away_key, **side_kwargs)
            row: dict[str, Any] = {"game_id": match["game_id"]}
            for metric in SIDE_METRICS:
                row[f"club_context_{metric}_home"] = home_values[metric]
                row[f"club_context_{metric}_away"] = away_values[metric]
                row[f"club_context_{metric}_delta"] = (
                    home_values[metric] - away_values[metric]
                )
            row.update(
                _orientation(
                    home_exposure=home_values["exposure_index"],
                    away_exposure=away_values["exposure_index"],
                )
            )
            row["club_context_attention_missing_home"] = home_values["attention_missing"]
            row["club_context_attention_missing_away"] = away_values["attention_missing"]
            row["club_context_attention_available"] = float(attention.available)
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
