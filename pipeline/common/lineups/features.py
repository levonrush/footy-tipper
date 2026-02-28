from __future__ import annotations

import os
import sqlite3
from collections import defaultdict
from itertools import combinations
import math
from pathlib import Path

import pandas as pd

from .normalization import normalize_team_name


LINEUP_FEATURE_COLUMNS = [
    "game_id",
    "lineup_data_available_home",
    "lineup_data_available_away",
    "lineup_named_count_home",
    "lineup_named_count_away",
    "lineup_interchange_count_home",
    "lineup_interchange_count_away",
    "lineup_reserve_count_home",
    "lineup_reserve_count_away",
    "lineup_spine_count_home",
    "lineup_spine_count_away",
    "lineup_spine_complete_home",
    "lineup_spine_complete_away",
    "lineup_bench_hooker_count_home",
    "lineup_bench_hooker_count_away",
    "lineup_bench_spine_cover_count_home",
    "lineup_bench_spine_cover_count_away",
    "lineup_source_age_hours_home",
    "lineup_source_age_hours_away",
    "lineup_retained_ratio_home",
    "lineup_retained_ratio_away",
    "lineup_starters_retained_ratio_home",
    "lineup_starters_retained_ratio_away",
    "lineup_spine_retained_ratio_home",
    "lineup_spine_retained_ratio_away",
    "lineup_spine_same_as_prev_home",
    "lineup_spine_same_as_prev_away",
    "lineup_halves_pair_same_as_prev_home",
    "lineup_halves_pair_same_as_prev_away",
    "lineup_avg_named_experience_home",
    "lineup_avg_named_experience_away",
    "lineup_avg_spine_experience_home",
    "lineup_avg_spine_experience_away",
    "lineup_avg_halves_experience_home",
    "lineup_avg_halves_experience_away",
    "lineup_avg_middles_experience_home",
    "lineup_avg_middles_experience_away",
    "lineup_avg_edges_experience_home",
    "lineup_avg_edges_experience_away",
    "lineup_avg_outside_backs_experience_home",
    "lineup_avg_outside_backs_experience_away",
    "lineup_avg_interchange_experience_home",
    "lineup_avg_interchange_experience_away",
    "lineup_avg_named_margin_rating_home",
    "lineup_avg_named_margin_rating_away",
    "lineup_avg_spine_margin_rating_home",
    "lineup_avg_spine_margin_rating_away",
    "lineup_avg_halves_margin_rating_home",
    "lineup_avg_halves_margin_rating_away",
    "lineup_avg_middles_margin_rating_home",
    "lineup_avg_middles_margin_rating_away",
    "lineup_avg_edges_margin_rating_home",
    "lineup_avg_edges_margin_rating_away",
    "lineup_avg_outside_backs_margin_rating_home",
    "lineup_avg_outside_backs_margin_rating_away",
    "lineup_avg_interchange_margin_rating_home",
    "lineup_avg_interchange_margin_rating_away",
    "lineup_debutant_count_home",
    "lineup_debutant_count_away",
    "lineup_named_cohesion_home",
    "lineup_named_cohesion_away",
    "lineup_spine_cohesion_home",
    "lineup_spine_cohesion_away",
    "lineup_halves_pair_cohesion_home",
    "lineup_halves_pair_cohesion_away",
    "lineup_recent_named_stability_home",
    "lineup_recent_named_stability_away",
    "lineup_recent_spine_stability_home",
    "lineup_recent_spine_stability_away",
    "lineup_snapshot_count_home",
    "lineup_snapshot_count_away",
    "lineup_snapshot_window_hours_home",
    "lineup_snapshot_window_hours_away",
    "lineup_named_change_count_home",
    "lineup_named_change_count_away",
    "lineup_named_change_rate_home",
    "lineup_named_change_rate_away",
    "lineup_spine_change_count_home",
    "lineup_spine_change_count_away",
    "lineup_spine_change_rate_home",
    "lineup_spine_change_rate_away",
    "lineup_expected_named_count_home",
    "lineup_expected_named_count_away",
    "lineup_expected_interchange_count_home",
    "lineup_expected_interchange_count_away",
    "lineup_expected_spine_count_home",
    "lineup_expected_spine_count_away",
    "lineup_selection_uncertainty_home",
    "lineup_selection_uncertainty_away",
    "lineup_named_count_delta",
    "lineup_interchange_count_delta",
    "lineup_reserve_count_delta",
    "lineup_spine_count_delta",
    "lineup_spine_complete_delta",
    "lineup_bench_hooker_count_delta",
    "lineup_bench_spine_cover_count_delta",
    "lineup_source_age_hours_delta",
    "lineup_retained_ratio_delta",
    "lineup_starters_retained_ratio_delta",
    "lineup_spine_retained_ratio_delta",
    "lineup_spine_same_as_prev_delta",
    "lineup_halves_pair_same_as_prev_delta",
    "lineup_avg_named_experience_delta",
    "lineup_avg_spine_experience_delta",
    "lineup_avg_halves_experience_delta",
    "lineup_avg_middles_experience_delta",
    "lineup_avg_edges_experience_delta",
    "lineup_avg_outside_backs_experience_delta",
    "lineup_avg_interchange_experience_delta",
    "lineup_avg_named_margin_rating_delta",
    "lineup_avg_spine_margin_rating_delta",
    "lineup_avg_halves_margin_rating_delta",
    "lineup_avg_middles_margin_rating_delta",
    "lineup_avg_edges_margin_rating_delta",
    "lineup_avg_outside_backs_margin_rating_delta",
    "lineup_avg_interchange_margin_rating_delta",
    "lineup_debutant_count_delta",
    "lineup_named_cohesion_delta",
    "lineup_spine_cohesion_delta",
    "lineup_halves_pair_cohesion_delta",
    "lineup_recent_named_stability_delta",
    "lineup_recent_spine_stability_delta",
    "lineup_snapshot_count_delta",
    "lineup_snapshot_window_hours_delta",
    "lineup_named_change_count_delta",
    "lineup_named_change_rate_delta",
    "lineup_spine_change_count_delta",
    "lineup_spine_change_rate_delta",
    "lineup_expected_named_count_delta",
    "lineup_expected_interchange_count_delta",
    "lineup_expected_spine_count_delta",
    "lineup_selection_uncertainty_delta",
    "lineup_features_missing",
    "lineup_home_players",
    "lineup_away_players",
]


def _empty_lineup_feature_frame(game_ids: pd.Series | list | None = None) -> pd.DataFrame:
    game_ids = [] if game_ids is None else list(game_ids)
    data = {"game_id": game_ids}
    for col in LINEUP_FEATURE_COLUMNS:
        if col == "game_id":
            continue
        if col in {"lineup_home_players", "lineup_away_players"}:
            data[col] = [""] * len(game_ids)
        elif col == "lineup_features_missing":
            data[col] = [1.0] * len(game_ids)
        else:
            data[col] = [0.0] * len(game_ids)
    return pd.DataFrame(data, columns=LINEUP_FEATURE_COLUMNS)


def load_lineup_entries(db_path: Path | str, years: list[int] | None = None) -> pd.DataFrame:
    db_file = Path(db_path)
    if not db_file.exists():
        return pd.DataFrame()

    with sqlite3.connect(str(db_file)) as con:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ('lineup_entries', 'lineup_article_snapshots')",
            con,
        )
        if len(tables) < 2:
            return pd.DataFrame()

        query = """
            SELECT
                e.*,
                s.parse_status,
                s.scraped_at_utc
            FROM lineup_entries e
            INNER JOIN lineup_article_snapshots s
                ON e.snapshot_id = s.snapshot_id
            WHERE s.parse_status = 'ok'
        """
        params = []
        if years:
            placeholders = ",".join("?" for _ in years)
            query += f" AND e.competition_year IN ({placeholders})"
            params.extend(years)

        entries = pd.read_sql_query(query, con, params=params)
    return entries


def _normalize_group(value: str | None) -> str:
    lower = str(value or "").strip().lower()
    if "back" in lower:
        return "backs"
    if "forward" in lower:
        return "forwards"
    if "interchange" in lower:
        return "interchange"
    if "reserve" in lower:
        return "reserves"
    return lower or "unknown"


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _role_group(jersey_number, position_norm: str, is_interchange: bool) -> str:
    if is_interchange:
        return "interchange"

    if pd.notna(jersey_number):
        jersey = int(jersey_number)
        if jersey in {6, 7}:
            return "halves"
        if jersey in {8, 9, 10, 13}:
            return "middles"
        if jersey in {11, 12}:
            return "edges"
        if jersey in {1, 2, 3, 4, 5}:
            return "outside_backs"

    if _contains_any(position_norm, ("five-eighth", "halfback", "half back")):
        return "halves"
    if _contains_any(position_norm, ("prop", "hooker", "lock", "middle")):
        return "middles"
    if _contains_any(position_norm, ("second row", "second-row", "edge", "back row", "back-row")):
        return "edges"
    if _contains_any(position_norm, ("fullback", "wing", "centre", "center", "outside back", "outside-back")):
        return "outside_backs"
    return "other"


def _jersey_bucket(value) -> str:
    jersey = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(jersey):
        return "unknown"
    jersey = int(jersey)
    if jersey <= 13:
        return "starting_13"
    if jersey <= 17:
        return "bench_14_17"
    return "reserve_18_plus"


def _to_event_time(df: pd.DataFrame) -> pd.Series:
    published = pd.to_datetime(df.get("source_published_at_utc"), errors="coerce", utc=True)
    inserted = pd.to_datetime(df.get("inserted_at_utc"), errors="coerce", utc=True)
    return published.fillna(inserted)


def _resolve_player_refs(entries_df: pd.DataFrame) -> pd.DataFrame:
    if entries_df.empty:
        return entries_df.copy()

    entries = entries_df.copy()
    base_ref = entries.get("player_key", pd.Series("", index=entries.index)).fillna("").astype(str)
    external = entries.get("player_external_id", pd.Series(index=entries.index, dtype=object))
    external = external.astype(str).str.strip()
    external = external.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    resolved = base_ref.copy()
    direct_mask = external.notna()
    resolved.loc[direct_mask] = "pid_" + external.loc[direct_mask]

    mapped = entries.loc[direct_mask, ["team_key", "player_key"]].copy()
    mapped["resolved_ref"] = "pid_" + external.loc[direct_mask]

    team_alias_map = {}
    if not mapped.empty:
        team_unique = mapped.drop_duplicates()
        team_counts = team_unique.groupby(["team_key", "player_key"])["resolved_ref"].nunique()
        stable_team_keys = team_counts[team_counts == 1].index.tolist()
        if stable_team_keys:
            stable_rows = team_unique.set_index(["team_key", "player_key"]).loc[stable_team_keys]["resolved_ref"]
            if hasattr(stable_rows, "to_dict"):
                team_alias_map = stable_rows.to_dict()

    global_alias_map = {}
    if not mapped.empty:
        global_unique = mapped[["player_key", "resolved_ref"]].drop_duplicates()
        global_counts = global_unique.groupby("player_key")["resolved_ref"].nunique()
        stable_global_keys = global_counts[global_counts == 1].index.tolist()
        if stable_global_keys:
            stable_rows = global_unique.set_index("player_key").loc[stable_global_keys]["resolved_ref"]
            if hasattr(stable_rows, "to_dict"):
                global_alias_map = stable_rows.to_dict()

    unresolved_mask = ~direct_mask
    if unresolved_mask.any():
        for idx in entries.index[unresolved_mask]:
            team_key = str(entries.at[idx, "team_key"])
            player_key = str(entries.at[idx, "player_key"])
            resolved_ref = team_alias_map.get((team_key, player_key)) or global_alias_map.get(player_key)
            if resolved_ref:
                resolved.at[idx] = resolved_ref

    entries["player_ref"] = resolved.fillna(base_ref).astype(str)
    return entries


def _to_match_time(values) -> pd.Series:
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    parsed = pd.to_datetime(series, errors="coerce", utc=True)

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        seconds_mask = numeric.abs().between(1e8, 1e11, inclusive="left")
        millis_mask = numeric.abs().between(1e11, 1e14, inclusive="left")
        micros_mask = numeric.abs().between(1e14, 1e17, inclusive="left")
        nanos_mask = numeric.abs() >= 1e17

        if seconds_mask.any():
            parsed.loc[seconds_mask] = pd.to_datetime(
                numeric.loc[seconds_mask],
                unit="s",
                errors="coerce",
                utc=True,
            )
        if millis_mask.any():
            parsed.loc[millis_mask] = pd.to_datetime(
                numeric.loc[millis_mask],
                unit="ms",
                errors="coerce",
                utc=True,
            )
        if micros_mask.any():
            parsed.loc[micros_mask] = pd.to_datetime(
                numeric.loc[micros_mask],
                unit="us",
                errors="coerce",
                utc=True,
            )
        if nanos_mask.any():
            parsed.loc[nanos_mask] = pd.to_datetime(
                numeric.loc[nanos_mask],
                unit="ns",
                errors="coerce",
                utc=True,
            )

    return parsed


def _build_selection_probability_model(entries_df: pd.DataFrame) -> dict:
    if entries_df.empty:
        return {
            "global_prob": 0.75,
            "bucket_prob": {
                "starting_13": 0.97,
                "bench_14_17": 0.88,
                "reserve_18_plus": 0.18,
                "unknown": 0.55,
            },
            "group_bucket_prob": {},
        }

    entries = entries_df.copy()
    entries["competition_year"] = pd.to_numeric(entries["competition_year"], errors="coerce")
    entries["round_id"] = pd.to_numeric(entries["round_id"], errors="coerce")
    entries = entries.dropna(subset=["competition_year", "round_id", "team_key", "snapshot_id", "player_ref"])
    if entries.empty:
        return {
            "global_prob": 0.75,
            "bucket_prob": {
                "starting_13": 0.97,
                "bench_14_17": 0.88,
                "reserve_18_plus": 0.18,
                "unknown": 0.55,
            },
            "group_bucket_prob": {},
        }

    entries["competition_year"] = entries["competition_year"].astype(int)
    entries["round_id"] = entries["round_id"].astype(int)
    entries["event_time"] = _to_event_time(entries)

    snapshot_index = (
        entries[["competition_year", "round_id", "team_key", "snapshot_id", "event_time"]]
        .drop_duplicates()
        .sort_values(["competition_year", "round_id", "team_key", "event_time", "snapshot_id"])
    )

    transitions = []
    for key, snapshot_group in snapshot_index.groupby(["competition_year", "round_id", "team_key"], dropna=False):
        ordered = snapshot_group.sort_values(["event_time", "snapshot_id"]).reset_index(drop=True)
        if ordered.empty:
            continue

        target_snapshot_id = int(ordered.iloc[-1]["snapshot_id"])
        target_rows = entries[entries["snapshot_id"] == target_snapshot_id].copy()
        target_rows["jersey_number"] = pd.to_numeric(target_rows["jersey_number"], errors="coerce")
        target_players = set(
            target_rows[target_rows["jersey_number"].between(1, 17, inclusive="both")]["player_ref"].astype(str)
        )
        if not target_players:
            continue

        for snapshot_id in ordered["snapshot_id"].tolist():
            source_rows = entries[entries["snapshot_id"] == snapshot_id].copy()
            source_rows = source_rows.drop_duplicates(subset=["player_ref"])
            source_rows["jersey_number"] = pd.to_numeric(source_rows["jersey_number"], errors="coerce")

            for row in source_rows.itertuples(index=False):
                player_key = str(getattr(row, "player_ref"))
                bucket = _jersey_bucket(getattr(row, "jersey_number", None))
                squad_group = _normalize_group(getattr(row, "squad_group", None))
                selected = 1 if player_key in target_players else 0
                transitions.append(
                    {
                        "squad_group": squad_group,
                        "jersey_bucket": bucket,
                        "selected": selected,
                    }
                )

    default_bucket_prob = {
        "starting_13": 0.97,
        "bench_14_17": 0.88,
        "reserve_18_plus": 0.18,
        "unknown": 0.55,
    }

    if not transitions:
        return {
            "global_prob": 0.75,
            "bucket_prob": default_bucket_prob,
            "group_bucket_prob": {},
        }

    trans_df = pd.DataFrame(transitions)
    global_prob = float(trans_df["selected"].mean())

    bucket_stats = trans_df.groupby("jersey_bucket", as_index=False).agg(
        n=("selected", "count"),
        success=("selected", "sum"),
    )

    bucket_prob: dict[str, float] = default_bucket_prob.copy()
    pseudo_weight_bucket = 10.0
    for row in bucket_stats.itertuples(index=False):
        bucket = str(getattr(row, "jersey_bucket"))
        n = float(getattr(row, "n"))
        success = float(getattr(row, "success"))
        prior = bucket_prob.get(bucket, global_prob)
        bucket_prob[bucket] = (success + (pseudo_weight_bucket * prior)) / (n + pseudo_weight_bucket)

    gb_stats = trans_df.groupby(["squad_group", "jersey_bucket"], as_index=False).agg(
        n=("selected", "count"),
        success=("selected", "sum"),
    )

    group_bucket_prob: dict[tuple[str, str], float] = {}
    pseudo_weight_group = 6.0
    for row in gb_stats.itertuples(index=False):
        group = str(getattr(row, "squad_group"))
        bucket = str(getattr(row, "jersey_bucket"))
        n = float(getattr(row, "n"))
        success = float(getattr(row, "success"))
        prior = bucket_prob.get(bucket, global_prob)
        group_bucket_prob[(group, bucket)] = (success + (pseudo_weight_group * prior)) / (n + pseudo_weight_group)

    return {
        "global_prob": max(0.01, min(0.99, global_prob)),
        "bucket_prob": {k: max(0.01, min(0.99, v)) for k, v in bucket_prob.items()},
        "group_bucket_prob": {
            key: max(0.01, min(0.99, value))
            for key, value in group_bucket_prob.items()
        },
    }


def _selection_probability(jersey_number, squad_group, model: dict) -> float:
    group = _normalize_group(squad_group)
    bucket = _jersey_bucket(jersey_number)

    group_bucket = model.get("group_bucket_prob", {})
    bucket_prob = model.get("bucket_prob", {})
    global_prob = float(model.get("global_prob", 0.75))

    if (group, bucket) in group_bucket:
        return float(group_bucket[(group, bucket)])
    if bucket in bucket_prob:
        return float(bucket_prob[bucket])
    return float(global_prob)


def _build_match_long(matches: pd.DataFrame, now_utc: pd.Timestamp, horizon_hours: float) -> pd.DataFrame:
    frame = matches.copy()
    frame["start_time_utc"] = _to_match_time(frame.get("start_time"))
    if "game_state_name" in frame.columns:
        frame["game_state_name"] = frame["game_state_name"].fillna("").astype(str)
    else:
        frame["game_state_name"] = ""

    home = frame[
        ["game_id", "competition_year", "round_id", "game_state_name", "start_time_utc", "game_number", "team_home_key"]
    ].rename(columns={"team_home_key": "team_key"})
    home["side"] = "home"

    away = frame[
        ["game_id", "competition_year", "round_id", "game_state_name", "start_time_utc", "game_number", "team_away_key"]
    ].rename(columns={"team_away_key": "team_key"})
    away["side"] = "away"

    long_rows = pd.concat([home, away], ignore_index=True)

    def _as_of(row) -> pd.Timestamp:
        start = row.start_time_utc
        if pd.isna(start):
            return now_utc

        state = str(row.game_state_name).strip().lower()
        if state == "final":
            cutoff = start - pd.Timedelta(hours=float(horizon_hours))
            return cutoff

        return min(now_utc, start)

    long_rows["as_of_time_utc"] = long_rows.apply(_as_of, axis=1)
    return long_rows


def _choose_snapshots_for_matches(matches: pd.DataFrame, entries_df: pd.DataFrame) -> pd.DataFrame:
    if matches.empty or entries_df.empty:
        return pd.DataFrame(columns=["game_id", "side", "team_key", "snapshot_id", "snapshot_time_utc"])

    horizon_hours = float(os.getenv("FOOTY_TIPPER_LINEUPS_AS_OF_HOURS_BEFORE_KICKOFF", "24"))
    now_utc = pd.Timestamp.now(tz="UTC")

    long_rows = _build_match_long(matches, now_utc=now_utc, horizon_hours=horizon_hours)

    entries = entries_df.copy()
    entries["competition_year"] = pd.to_numeric(entries["competition_year"], errors="coerce")
    entries["round_id"] = pd.to_numeric(entries["round_id"], errors="coerce")
    entries = entries.dropna(subset=["competition_year", "round_id", "team_key", "snapshot_id"])
    if entries.empty:
        return pd.DataFrame(columns=["game_id", "side", "team_key", "snapshot_id", "snapshot_time_utc"])

    entries["competition_year"] = entries["competition_year"].astype(int)
    entries["round_id"] = entries["round_id"].astype(int)
    entries["event_time"] = _to_event_time(entries)

    snapshot_index = (
        entries[["competition_year", "round_id", "team_key", "snapshot_id", "event_time"]]
        .drop_duplicates()
        .dropna(subset=["event_time"])
        .sort_values(["competition_year", "round_id", "team_key", "event_time", "snapshot_id"])
    )
    if snapshot_index.empty:
        return pd.DataFrame(columns=["game_id", "side", "team_key", "snapshot_id", "snapshot_time_utc"])

    choices = []
    grouped_snapshots = {
        key: group.sort_values(["event_time", "snapshot_id"]).reset_index(drop=True)
        for key, group in snapshot_index.groupby(["competition_year", "round_id", "team_key"], dropna=False)
    }

    for row in long_rows.itertuples(index=False):
        key = (int(row.competition_year), int(row.round_id), str(row.team_key))
        snapshots = grouped_snapshots.get(key)
        if snapshots is None or snapshots.empty:
            continue

        cutoff = row.as_of_time_utc
        if pd.isna(cutoff):
            continue

        eligible = snapshots[snapshots["event_time"] <= cutoff]
        if eligible.empty:
            continue

        chosen = eligible.iloc[-1]
        choices.append(
            {
                "game_id": int(row.game_id),
                "side": row.side,
                "team_key": str(row.team_key),
                "snapshot_id": int(chosen["snapshot_id"]),
                "snapshot_time_utc": chosen["event_time"],
            }
        )

    if not choices:
        return pd.DataFrame(columns=["game_id", "side", "team_key", "snapshot_id", "snapshot_time_utc"])

    return pd.DataFrame(choices).drop_duplicates(subset=["game_id", "side"], keep="last")


def _build_selected_entries(choices: pd.DataFrame, entries_df: pd.DataFrame, model: dict) -> pd.DataFrame:
    if choices.empty:
        return pd.DataFrame()

    selected_entries = choices.merge(
        entries_df,
        on=["snapshot_id", "team_key"],
        how="left",
        suffixes=("", "_entry"),
    )
    selected_entries = selected_entries.dropna(subset=["player_ref"])
    if selected_entries.empty:
        return pd.DataFrame()

    selected_entries = selected_entries.copy()
    selected_entries["jersey_number"] = pd.to_numeric(selected_entries["jersey_number"], errors="coerce")
    selected_entries["group_norm"] = selected_entries["squad_group"].map(_normalize_group)
    selected_entries["position_norm"] = selected_entries["listed_position"].fillna("").astype(str).str.lower()

    selected_entries["is_named"] = selected_entries["jersey_number"].between(1, 17, inclusive="both").astype(int)
    selected_entries["is_interchange"] = (
        selected_entries["group_norm"].eq("interchange")
        | selected_entries["jersey_number"].between(14, 17, inclusive="both")
    ).astype(int)
    selected_entries["is_reserve"] = (
        selected_entries["group_norm"].eq("reserves")
        | (selected_entries["jersey_number"] >= 18)
    ).astype(int)
    selected_entries["is_spine"] = selected_entries["jersey_number"].isin([1, 6, 7, 9]).astype(int)
    selected_entries["is_bench_hooker"] = (
        selected_entries["is_interchange"].astype(bool)
        & selected_entries["position_norm"].str.contains("hooker", regex=False)
    ).astype(int)
    selected_entries["is_bench_spine_cover"] = (
        selected_entries["is_interchange"].astype(bool)
        & selected_entries["position_norm"].apply(
            lambda value: _contains_any(
                str(value),
                ("hooker", "fullback", "halfback", "five-eighth", "half back"),
            )
        )
    ).astype(int)
    selected_entries["role_group"] = selected_entries.apply(
        lambda row: _role_group(
            row.get("jersey_number"),
            row.get("position_norm", ""),
            bool(row.get("is_interchange", 0)),
        ),
        axis=1,
    )

    selected_entries["selection_prob"] = selected_entries.apply(
        lambda row: _selection_probability(
            row.get("jersey_number"),
            row.get("group_norm"),
            model,
        ),
        axis=1,
    )
    selected_entries["selection_var"] = selected_entries["selection_prob"] * (1.0 - selected_entries["selection_prob"])
    selected_entries["w_named"] = selected_entries["selection_prob"] * selected_entries["is_named"]
    selected_entries["w_interchange"] = selected_entries["selection_prob"] * selected_entries["is_interchange"]
    selected_entries["w_spine"] = selected_entries["selection_prob"] * selected_entries["is_spine"]

    # Keep one row per player per side/game snapshot.
    selected_entries = selected_entries.sort_values(["game_id", "side", "player_ref"]).drop_duplicates(
        subset=["game_id", "side", "player_ref"],
        keep="last",
    )
    return selected_entries


def _build_side_features(matches: pd.DataFrame, choices: pd.DataFrame, selected_entries: pd.DataFrame) -> pd.DataFrame:
    if selected_entries.empty:
        return pd.DataFrame()

    metrics = selected_entries.groupby(["game_id", "side"], as_index=False).agg(
        lineup_named_count=("is_named", "sum"),
        lineup_interchange_count=("is_interchange", "sum"),
        lineup_reserve_count=("is_reserve", "sum"),
        lineup_spine_count=("is_spine", "sum"),
        lineup_bench_hooker_count=("is_bench_hooker", "sum"),
        lineup_bench_spine_cover_count=("is_bench_spine_cover", "sum"),
        lineup_expected_named_count=("w_named", "sum"),
        lineup_expected_interchange_count=("w_interchange", "sum"),
        lineup_expected_spine_count=("w_spine", "sum"),
        lineup_selection_uncertainty=("selection_var", "mean"),
        lineup_homeaway_players=("player_name", lambda values: "|".join(sorted({str(v).strip() for v in values if str(v).strip()}))),
    )

    metrics["lineup_data_available"] = 1.0
    metrics["lineup_spine_complete"] = (metrics["lineup_spine_count"] >= 4).astype(float)
    metrics["lineup_selection_uncertainty"] = pd.to_numeric(
        metrics["lineup_selection_uncertainty"],
        errors="coerce",
    ).fillna(0.0)

    snapshot_time = choices[["game_id", "side", "snapshot_time_utc"]].copy()
    metrics = metrics.merge(snapshot_time, on=["game_id", "side"], how="left")

    match_time = matches[["game_id", "start_time"]].copy()
    match_time["start_time"] = _to_match_time(match_time["start_time"])
    metrics = metrics.merge(match_time, on="game_id", how="left")
    metrics["lineup_source_age_hours"] = (
        (metrics["start_time"] - pd.to_datetime(metrics["snapshot_time_utc"], errors="coerce", utc=True))
        .dt.total_seconds()
        .div(3600.0)
        .clip(lower=0)
    )
    metrics["lineup_source_age_hours"] = metrics["lineup_source_age_hours"].fillna(0.0)

    return metrics


def _build_long_team_rows(matches: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["game_id", "competition_year", "round_id", "game_number", "start_time", "team_home_key", "team_away_key"]
    if {"team_final_score_home", "team_final_score_away"}.issubset(matches.columns):
        base_cols += ["team_final_score_home", "team_final_score_away"]

    base = matches[base_cols].copy()
    base["competition_year"] = pd.to_numeric(base["competition_year"], errors="coerce")
    base["round_id"] = pd.to_numeric(base["round_id"], errors="coerce")
    base["game_number"] = pd.to_numeric(base.get("game_number"), errors="coerce")
    base["start_time"] = _to_match_time(base.get("start_time"))

    home_rows = base[
        ["game_id", "competition_year", "round_id", "game_number", "start_time", "team_home_key"]
    ].rename(columns={"team_home_key": "team_key"})
    home_rows["side"] = "home"

    away_rows = base[
        ["game_id", "competition_year", "round_id", "game_number", "start_time", "team_away_key"]
    ].rename(columns={"team_away_key": "team_key"})
    away_rows["side"] = "away"

    if {"team_final_score_home", "team_final_score_away"}.issubset(base.columns):
        base["team_final_score_home"] = pd.to_numeric(base["team_final_score_home"], errors="coerce")
        base["team_final_score_away"] = pd.to_numeric(base["team_final_score_away"], errors="coerce")
        home_rows["team_margin"] = base["team_final_score_home"] - base["team_final_score_away"]
        away_rows["team_margin"] = base["team_final_score_away"] - base["team_final_score_home"]
    else:
        home_rows["team_margin"] = pd.NA
        away_rows["team_margin"] = pd.NA

    long_rows = pd.concat([home_rows, away_rows], ignore_index=True)
    long_rows = long_rows.sort_values(
        ["start_time", "competition_year", "round_id", "game_number", "game_id", "side"]
    ).reset_index(drop=True)
    return long_rows


def _player_sets_by_side(selected_entries: pd.DataFrame) -> dict[tuple[int, str], dict[str, set[str]]]:
    if selected_entries.empty:
        return {}

    grouped: dict[tuple[int, str], dict[str, set[str]]] = {}
    for row in selected_entries.itertuples(index=False):
        key = (int(getattr(row, "game_id")), str(getattr(row, "side")))
        payload = grouped.setdefault(
            key,
            {
                "named": set(),
                "spine": set(),
            },
        )
        player_key = str(getattr(row, "player_ref"))
        if int(getattr(row, "is_named", 0)) > 0:
            payload["named"].add(player_key)
        if int(getattr(row, "is_spine", 0)) > 0:
            payload["spine"].add(player_key)
    return grouped


def _safe_average(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _safe_recent_overlap(current_players: set[str], recent_history: list[set[str]]) -> float:
    if not current_players or not recent_history:
        return 0.0

    overlaps = []
    for previous_players in recent_history:
        union = current_players.union(previous_players)
        if not union:
            continue
        overlaps.append(float(len(current_players.intersection(previous_players))) / float(len(union)))
    return _safe_average(overlaps)


def _log_pairwise_cohesion(players: set[str], pair_counts: dict[tuple[str, str], int]) -> float:
    ordered = sorted(players)
    if len(ordered) < 2:
        return 0.0
    values = [float(pair_counts.get((a, b), 0)) for a, b in combinations(ordered, 2)]
    if not values:
        return 0.0
    return float(math.log1p(sum(values) / len(values)))


def _compute_lineup_history_features(selected_entries: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    if selected_entries.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "lineup_starters_retained_ratio_home",
                "lineup_starters_retained_ratio_away",
                "lineup_spine_retained_ratio_home",
                "lineup_spine_retained_ratio_away",
                "lineup_spine_same_as_prev_home",
                "lineup_spine_same_as_prev_away",
                "lineup_halves_pair_same_as_prev_home",
                "lineup_halves_pair_same_as_prev_away",
                "lineup_avg_named_experience_home",
                "lineup_avg_named_experience_away",
                "lineup_avg_spine_experience_home",
                "lineup_avg_spine_experience_away",
                "lineup_avg_halves_experience_home",
                "lineup_avg_halves_experience_away",
                "lineup_avg_middles_experience_home",
                "lineup_avg_middles_experience_away",
                "lineup_avg_edges_experience_home",
                "lineup_avg_edges_experience_away",
                "lineup_avg_outside_backs_experience_home",
                "lineup_avg_outside_backs_experience_away",
                "lineup_avg_interchange_experience_home",
                "lineup_avg_interchange_experience_away",
                "lineup_avg_named_margin_rating_home",
                "lineup_avg_named_margin_rating_away",
                "lineup_avg_spine_margin_rating_home",
                "lineup_avg_spine_margin_rating_away",
                "lineup_avg_halves_margin_rating_home",
                "lineup_avg_halves_margin_rating_away",
                "lineup_avg_middles_margin_rating_home",
                "lineup_avg_middles_margin_rating_away",
                "lineup_avg_edges_margin_rating_home",
                "lineup_avg_edges_margin_rating_away",
                "lineup_avg_outside_backs_margin_rating_home",
                "lineup_avg_outside_backs_margin_rating_away",
                "lineup_avg_interchange_margin_rating_home",
                "lineup_avg_interchange_margin_rating_away",
                "lineup_debutant_count_home",
                "lineup_debutant_count_away",
                "lineup_named_cohesion_home",
                "lineup_named_cohesion_away",
                "lineup_spine_cohesion_home",
                "lineup_spine_cohesion_away",
                "lineup_halves_pair_cohesion_home",
                "lineup_halves_pair_cohesion_away",
                "lineup_recent_named_stability_home",
                "lineup_recent_named_stability_away",
                "lineup_recent_spine_stability_home",
                "lineup_recent_spine_stability_away",
            ]
        )

    long_rows = _build_long_team_rows(matches)
    player_sets = _player_sets_by_side(selected_entries)

    group_sets: dict[tuple[int, str], dict[str, set[str]]] = {}
    for row in selected_entries.itertuples(index=False):
        key = (int(getattr(row, "game_id")), str(getattr(row, "side")))
        payload = group_sets.setdefault(
            key,
            {
                "halves": set(),
                "middles": set(),
                "edges": set(),
                "outside_backs": set(),
                "interchange": set(),
            },
        )
        player_key = str(getattr(row, "player_ref"))
        role_group = str(getattr(row, "role_group", "other"))
        if role_group in payload and int(getattr(row, "is_named", 0)) > 0:
            payload[role_group].add(player_key)
        if role_group == "interchange":
            payload["interchange"].add(player_key)

    player_appearances: defaultdict[str, int] = defaultdict(int)
    player_margin_sum: defaultdict[str, float] = defaultdict(float)
    player_margin_count: defaultdict[str, int] = defaultdict(int)
    team_pair_counts: defaultdict[str, dict[tuple[str, str], int]] = defaultdict(dict)
    halves_pair_counts: defaultdict[str, dict[tuple[str, str], int]] = defaultdict(dict)
    prev_named_by_team: dict[str, set[str]] = {}
    prev_spine_by_team: dict[str, set[str]] = {}
    prev_halves_by_team: dict[str, set[str]] = {}
    recent_named_by_team: defaultdict[str, list[set[str]]] = defaultdict(list)
    recent_spine_by_team: defaultdict[str, list[set[str]]] = defaultdict(list)
    records: list[dict] = []

    for row in long_rows.itertuples(index=False):
        key = (int(getattr(row, "game_id")), str(getattr(row, "side")))
        sets = player_sets.get(key)
        if not sets:
            continue

        groups = group_sets.get(
            key,
            {
                "halves": set(),
                "middles": set(),
                "edges": set(),
                "outside_backs": set(),
                "interchange": set(),
            },
        )
        team_key = str(getattr(row, "team_key"))
        named_players = set(sets["named"])
        spine_players = set(sets["spine"])
        halves_players = set(groups["halves"])
        prev_named = prev_named_by_team.get(team_key, set())
        prev_spine = prev_spine_by_team.get(team_key, set())
        prev_halves = prev_halves_by_team.get(team_key, set())

        named_experience = [float(player_appearances[player]) for player in sorted(named_players)]
        spine_experience = [float(player_appearances[player]) for player in sorted(spine_players)]
        halves_experience = [float(player_appearances[player]) for player in sorted(groups["halves"])]
        middles_experience = [float(player_appearances[player]) for player in sorted(groups["middles"])]
        edges_experience = [float(player_appearances[player]) for player in sorted(groups["edges"])]
        outside_backs_experience = [
            float(player_appearances[player]) for player in sorted(groups["outside_backs"])
        ]
        interchange_experience = [float(player_appearances[player]) for player in sorted(groups["interchange"])]
        named_margin_ratings = [
            float(player_margin_sum[player] / (player_margin_count[player] + 5.0))
            for player in sorted(named_players)
        ]
        spine_margin_ratings = [
            float(player_margin_sum[player] / (player_margin_count[player] + 5.0))
            for player in sorted(spine_players)
        ]
        halves_margin_ratings = [
            float(player_margin_sum[player] / (player_margin_count[player] + 5.0))
            for player in sorted(groups["halves"])
        ]
        middles_margin_ratings = [
            float(player_margin_sum[player] / (player_margin_count[player] + 5.0))
            for player in sorted(groups["middles"])
        ]
        edges_margin_ratings = [
            float(player_margin_sum[player] / (player_margin_count[player] + 5.0))
            for player in sorted(groups["edges"])
        ]
        outside_backs_margin_ratings = [
            float(player_margin_sum[player] / (player_margin_count[player] + 5.0))
            for player in sorted(groups["outside_backs"])
        ]
        interchange_margin_ratings = [
            float(player_margin_sum[player] / (player_margin_count[player] + 5.0))
            for player in sorted(groups["interchange"])
        ]
        pair_counts = team_pair_counts[team_key]
        team_halves_pair_counts = halves_pair_counts[team_key]

        starters_retained_ratio = (
            float(len(named_players.intersection(prev_named)) / len(named_players))
            if named_players and prev_named
            else 0.0
        )
        spine_retained_ratio = (
            float(len(spine_players.intersection(prev_spine)) / len(spine_players))
            if spine_players and prev_spine
            else 0.0
        )
        spine_same_as_prev = 1.0 if spine_players and prev_spine and spine_players == prev_spine else 0.0
        halves_pair_same_as_prev = 1.0 if len(halves_players) == 2 and halves_players == prev_halves else 0.0
        halves_pair = tuple(sorted(halves_players))
        halves_pair_cohesion = float(math.log1p(team_halves_pair_counts.get(halves_pair, 0))) if len(halves_pair) == 2 else 0.0

        records.append(
            {
                "game_id": int(getattr(row, "game_id")),
                "side": str(getattr(row, "side")),
                "lineup_starters_retained_ratio": starters_retained_ratio,
                "lineup_spine_retained_ratio": spine_retained_ratio,
                "lineup_spine_same_as_prev": spine_same_as_prev,
                "lineup_halves_pair_same_as_prev": halves_pair_same_as_prev,
                "lineup_avg_named_experience": _safe_average(named_experience),
                "lineup_avg_spine_experience": _safe_average(spine_experience),
                "lineup_avg_halves_experience": _safe_average(halves_experience),
                "lineup_avg_middles_experience": _safe_average(middles_experience),
                "lineup_avg_edges_experience": _safe_average(edges_experience),
                "lineup_avg_outside_backs_experience": _safe_average(outside_backs_experience),
                "lineup_avg_interchange_experience": _safe_average(interchange_experience),
                "lineup_avg_named_margin_rating": _safe_average(named_margin_ratings),
                "lineup_avg_spine_margin_rating": _safe_average(spine_margin_ratings),
                "lineup_avg_halves_margin_rating": _safe_average(halves_margin_ratings),
                "lineup_avg_middles_margin_rating": _safe_average(middles_margin_ratings),
                "lineup_avg_edges_margin_rating": _safe_average(edges_margin_ratings),
                "lineup_avg_outside_backs_margin_rating": _safe_average(outside_backs_margin_ratings),
                "lineup_avg_interchange_margin_rating": _safe_average(interchange_margin_ratings),
                "lineup_debutant_count": float(sum(1 for player in named_players if player_appearances[player] <= 0)),
                "lineup_named_cohesion": _log_pairwise_cohesion(named_players, pair_counts),
                "lineup_spine_cohesion": _log_pairwise_cohesion(spine_players, pair_counts),
                "lineup_halves_pair_cohesion": halves_pair_cohesion,
                "lineup_recent_named_stability": _safe_recent_overlap(
                    named_players,
                    recent_named_by_team.get(team_key, []),
                ),
                "lineup_recent_spine_stability": _safe_recent_overlap(
                    spine_players,
                    recent_spine_by_team.get(team_key, []),
                ),
            }
        )

        for player in named_players:
            player_appearances[player] += 1

        ordered_named = sorted(named_players)
        for a, b in combinations(ordered_named, 2):
            pair_counts[(a, b)] = int(pair_counts.get((a, b), 0)) + 1
        if len(halves_pair) == 2:
            team_halves_pair_counts[halves_pair] = int(team_halves_pair_counts.get(halves_pair, 0)) + 1

        team_margin = pd.to_numeric(pd.Series([getattr(row, "team_margin", pd.NA)]), errors="coerce").iloc[0]
        if pd.notna(team_margin):
            for player in named_players:
                player_margin_sum[player] += float(team_margin)
                player_margin_count[player] += 1

        prev_named_by_team[team_key] = named_players
        prev_spine_by_team[team_key] = spine_players
        prev_halves_by_team[team_key] = halves_players
        recent_named_by_team[team_key] = (recent_named_by_team.get(team_key, []) + [named_players])[-4:]
        recent_spine_by_team[team_key] = (recent_spine_by_team.get(team_key, []) + [spine_players])[-4:]

    if not records:
        return pd.DataFrame()

    history_df = pd.DataFrame(records)
    wide = pd.DataFrame({"game_id": sorted(history_df["game_id"].unique())})
    feature_cols = [
        "lineup_starters_retained_ratio",
        "lineup_spine_retained_ratio",
        "lineup_spine_same_as_prev",
        "lineup_halves_pair_same_as_prev",
        "lineup_avg_named_experience",
        "lineup_avg_spine_experience",
        "lineup_avg_halves_experience",
        "lineup_avg_middles_experience",
        "lineup_avg_edges_experience",
        "lineup_avg_outside_backs_experience",
        "lineup_avg_interchange_experience",
        "lineup_avg_named_margin_rating",
        "lineup_avg_spine_margin_rating",
        "lineup_avg_halves_margin_rating",
        "lineup_avg_middles_margin_rating",
        "lineup_avg_edges_margin_rating",
        "lineup_avg_outside_backs_margin_rating",
        "lineup_avg_interchange_margin_rating",
        "lineup_debutant_count",
        "lineup_named_cohesion",
        "lineup_spine_cohesion",
        "lineup_halves_pair_cohesion",
        "lineup_recent_named_stability",
        "lineup_recent_spine_stability",
    ]
    for side in ("home", "away"):
        side_df = history_df[history_df["side"] == side][["game_id", *feature_cols]].copy()
        side_df = side_df.rename(columns={col: f"{col}_{side}" for col in feature_cols})
        wide = wide.merge(side_df, on="game_id", how="left")
    return wide


def _build_snapshot_transition_features(matches: pd.DataFrame, entries_df: pd.DataFrame) -> pd.DataFrame:
    if matches.empty or entries_df.empty:
        return pd.DataFrame(columns=["game_id"])

    horizon_hours = float(os.getenv("FOOTY_TIPPER_LINEUPS_AS_OF_HOURS_BEFORE_KICKOFF", "24"))
    now_utc = pd.Timestamp.now(tz="UTC")
    long_rows = _build_match_long(matches, now_utc=now_utc, horizon_hours=horizon_hours)

    entries = entries_df.copy()
    entries["competition_year"] = pd.to_numeric(entries["competition_year"], errors="coerce")
    entries["round_id"] = pd.to_numeric(entries["round_id"], errors="coerce")
    entries = entries.dropna(subset=["competition_year", "round_id", "team_key", "snapshot_id", "player_ref"])
    if entries.empty:
        return pd.DataFrame(columns=["game_id"])

    entries["competition_year"] = entries["competition_year"].astype(int)
    entries["round_id"] = entries["round_id"].astype(int)
    entries["jersey_number"] = pd.to_numeric(entries["jersey_number"], errors="coerce")
    entries["event_time"] = _to_event_time(entries)
    entries = entries.dropna(subset=["event_time"])
    if entries.empty:
        return pd.DataFrame(columns=["game_id"])

    snapshot_index = (
        entries[["competition_year", "round_id", "team_key", "snapshot_id", "event_time"]]
        .drop_duplicates()
        .sort_values(["competition_year", "round_id", "team_key", "event_time", "snapshot_id"])
    )
    grouped_snapshots = {
        key: group.sort_values(["event_time", "snapshot_id"]).reset_index(drop=True)
        for key, group in snapshot_index.groupby(["competition_year", "round_id", "team_key"], dropna=False)
    }

    player_sets: dict[tuple[int, str], dict[str, set[str]]] = {}
    dedup_entries = entries.sort_values(["snapshot_id", "team_key", "player_ref"]).drop_duplicates(
        subset=["snapshot_id", "team_key", "player_ref"],
        keep="last",
    )
    for row in dedup_entries.itertuples(index=False):
        key = (int(getattr(row, "snapshot_id")), str(getattr(row, "team_key")))
        payload = player_sets.setdefault(key, {"named": set(), "spine": set()})
        player_key = str(getattr(row, "player_ref"))
        jersey_number = getattr(row, "jersey_number")
        if pd.notna(jersey_number) and 1 <= int(jersey_number) <= 17:
            payload["named"].add(player_key)
        if pd.notna(jersey_number) and int(jersey_number) in {1, 6, 7, 9}:
            payload["spine"].add(player_key)

    records: list[dict] = []
    for row in long_rows.itertuples(index=False):
        if pd.isna(getattr(row, "competition_year")) or pd.isna(getattr(row, "round_id")):
            continue
        key = (int(getattr(row, "competition_year")), int(getattr(row, "round_id")), str(getattr(row, "team_key")))
        snapshots = grouped_snapshots.get(key)
        if snapshots is None or snapshots.empty:
            continue

        eligible = snapshots[snapshots["event_time"] <= getattr(row, "as_of_time_utc")]
        if eligible.empty:
            continue

        earliest_snapshot_id = int(eligible.iloc[0]["snapshot_id"])
        latest_snapshot_id = int(eligible.iloc[-1]["snapshot_id"])
        earliest_sets = player_sets.get((earliest_snapshot_id, str(getattr(row, "team_key"))), {"named": set(), "spine": set()})
        latest_sets = player_sets.get((latest_snapshot_id, str(getattr(row, "team_key"))), {"named": set(), "spine": set()})

        records.append(
            {
                "game_id": int(getattr(row, "game_id")),
                "side": str(getattr(row, "side")),
                "lineup_snapshot_count": float(len(eligible)),
                "lineup_snapshot_window_hours": float(
                    (
                        pd.to_datetime(eligible.iloc[-1]["event_time"], utc=True)
                        - pd.to_datetime(eligible.iloc[0]["event_time"], utc=True)
                    ).total_seconds()
                    / 3600.0
                ),
                "lineup_named_change_count": float(len(earliest_sets["named"].symmetric_difference(latest_sets["named"]))),
                "lineup_named_change_rate": float(
                    len(earliest_sets["named"].symmetric_difference(latest_sets["named"])) / max(len(eligible) - 1, 1)
                ),
                "lineup_spine_change_count": float(len(earliest_sets["spine"].symmetric_difference(latest_sets["spine"]))),
                "lineup_spine_change_rate": float(
                    len(earliest_sets["spine"].symmetric_difference(latest_sets["spine"])) / max(len(eligible) - 1, 1)
                ),
            }
        )

    if not records:
        return pd.DataFrame(columns=["game_id"])

    transition_df = pd.DataFrame(records)
    wide = pd.DataFrame({"game_id": sorted(transition_df["game_id"].unique())})
    feature_cols = [
        "lineup_snapshot_count",
        "lineup_snapshot_window_hours",
        "lineup_named_change_count",
        "lineup_named_change_rate",
        "lineup_spine_change_count",
        "lineup_spine_change_rate",
    ]
    for side in ("home", "away"):
        side_df = transition_df[transition_df["side"] == side][["game_id", *feature_cols]].copy()
        side_df = side_df.rename(columns={col: f"{col}_{side}" for col in feature_cols})
        wide = wide.merge(side_df, on="game_id", how="left")
    return wide


def _pivot_side_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame(columns=["game_id"])

    side_cols = [
        "lineup_data_available",
        "lineup_named_count",
        "lineup_interchange_count",
        "lineup_reserve_count",
        "lineup_spine_count",
        "lineup_spine_complete",
        "lineup_bench_hooker_count",
        "lineup_bench_spine_cover_count",
        "lineup_source_age_hours",
        "lineup_expected_named_count",
        "lineup_expected_interchange_count",
        "lineup_expected_spine_count",
        "lineup_selection_uncertainty",
        "lineup_homeaway_players",
    ]

    wide = pd.DataFrame({"game_id": sorted(metrics["game_id"].unique())})
    for side in ("home", "away"):
        side_df = metrics[metrics["side"] == side][["game_id", *side_cols]].copy()
        rename_map = {col: f"{col}_{side}" for col in side_cols}
        side_df = side_df.rename(columns=rename_map)
        wide = wide.merge(side_df, on="game_id", how="left")

    wide["lineup_home_players"] = wide.get("lineup_homeaway_players_home", "").fillna("")
    wide["lineup_away_players"] = wide.get("lineup_homeaway_players_away", "").fillna("")
    return wide


def _compute_retained_ratio(features_df: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    if features_df.empty:
        return pd.DataFrame(columns=["game_id", "lineup_retained_ratio_home", "lineup_retained_ratio_away"])

    base = matches[["game_id", "competition_year", "round_id", "game_number", "start_time", "team_home_key", "team_away_key"]].copy()
    base["competition_year"] = pd.to_numeric(base["competition_year"], errors="coerce")
    base["round_id"] = pd.to_numeric(base["round_id"], errors="coerce")
    base["game_number"] = pd.to_numeric(base.get("game_number"), errors="coerce")
    base["start_time"] = _to_match_time(base.get("start_time"))

    merged = base.merge(
        features_df[["game_id", "lineup_home_players", "lineup_away_players"]],
        on="game_id",
        how="left",
    )

    home_rows = merged[
        ["game_id", "competition_year", "round_id", "game_number", "start_time", "team_home_key", "lineup_home_players"]
    ].rename(columns={"team_home_key": "team_key", "lineup_home_players": "lineup_players"})
    home_rows["side"] = "home"

    away_rows = merged[
        ["game_id", "competition_year", "round_id", "game_number", "start_time", "team_away_key", "lineup_away_players"]
    ].rename(columns={"team_away_key": "team_key", "lineup_away_players": "lineup_players"})
    away_rows["side"] = "away"

    long_rows = pd.concat([home_rows, away_rows], ignore_index=True)
    long_rows = long_rows.sort_values(
        ["team_key", "competition_year", "round_id", "start_time", "game_number", "game_id"]
    )

    output = []
    for team_key, group in long_rows.groupby("team_key", dropna=False):
        if not team_key:
            continue

        prev_players: set[str] = set()
        for row in group.itertuples(index=False):
            players = {
                token.strip()
                for token in str(getattr(row, "lineup_players", "")).split("|")
                if token and token.strip()
            }
            if players and prev_players:
                retained = float(len(players.intersection(prev_players))) / float(len(players))
            else:
                retained = 0.0

            output.append(
                {
                    "game_id": int(getattr(row, "game_id")),
                    "side": str(getattr(row, "side")),
                    "retained_ratio": retained,
                }
            )

            if players:
                prev_players = players

    if not output:
        return pd.DataFrame(columns=["game_id", "lineup_retained_ratio_home", "lineup_retained_ratio_away"])

    ratio_df = pd.DataFrame(output)
    home = ratio_df[ratio_df["side"] == "home"][["game_id", "retained_ratio"]].rename(
        columns={"retained_ratio": "lineup_retained_ratio_home"}
    )
    away = ratio_df[ratio_df["side"] == "away"][["game_id", "retained_ratio"]].rename(
        columns={"retained_ratio": "lineup_retained_ratio_away"}
    )
    return home.merge(away, on="game_id", how="outer").fillna(0.0)


def build_lineup_match_features(matches_df: pd.DataFrame, lineup_entries_df: pd.DataFrame) -> pd.DataFrame:
    if matches_df.empty:
        return _empty_lineup_feature_frame(game_ids=[])

    if lineup_entries_df.empty:
        return _empty_lineup_feature_frame(game_ids=matches_df["game_id"].tolist())

    lineup_entries_df = _resolve_player_refs(lineup_entries_df)

    required_cols = {
        "game_id",
        "competition_year",
        "round_id",
        "team_home",
        "team_away",
        "start_time",
    }
    missing = sorted(required_cols.difference(matches_df.columns))
    if missing:
        raise ValueError("Lineup feature build requires columns: " + ", ".join(missing))

    matches = matches_df.copy()
    matches["competition_year"] = pd.to_numeric(matches["competition_year"], errors="coerce")
    matches["round_id"] = pd.to_numeric(matches["round_id"], errors="coerce")
    matches = matches.dropna(subset=["competition_year", "round_id", "team_home", "team_away"])
    if matches.empty:
        return _empty_lineup_feature_frame(game_ids=matches_df["game_id"].tolist())

    matches["competition_year"] = matches["competition_year"].astype(int)
    matches["round_id"] = matches["round_id"].astype(int)
    matches["team_home_key"] = matches["team_home"].map(normalize_team_name)
    matches["team_away_key"] = matches["team_away"].map(normalize_team_name)

    model = _build_selection_probability_model(lineup_entries_df)
    choices = _choose_snapshots_for_matches(matches, lineup_entries_df)
    if choices.empty:
        return _empty_lineup_feature_frame(game_ids=matches_df["game_id"].tolist())

    selected_entries = _build_selected_entries(choices, lineup_entries_df, model)
    side_metrics = _build_side_features(matches, choices, selected_entries)
    wide = _pivot_side_metrics(side_metrics)

    result = matches[["game_id"]].merge(wide, on="game_id", how="left")

    retained_ratio = _compute_retained_ratio(result, matches)
    result = result.merge(retained_ratio, on="game_id", how="left")
    history_features = _compute_lineup_history_features(selected_entries, matches)
    result = result.merge(history_features, on="game_id", how="left")
    transition_features = _build_snapshot_transition_features(matches, lineup_entries_df)
    result = result.merge(transition_features, on="game_id", how="left")

    for base in [
        "lineup_data_available",
        "lineup_named_count",
        "lineup_interchange_count",
        "lineup_reserve_count",
        "lineup_spine_count",
        "lineup_spine_complete",
        "lineup_bench_hooker_count",
        "lineup_bench_spine_cover_count",
        "lineup_source_age_hours",
        "lineup_expected_named_count",
        "lineup_expected_interchange_count",
        "lineup_expected_spine_count",
        "lineup_selection_uncertainty",
        "lineup_retained_ratio",
        "lineup_starters_retained_ratio",
        "lineup_spine_retained_ratio",
        "lineup_spine_same_as_prev",
        "lineup_halves_pair_same_as_prev",
        "lineup_avg_named_experience",
        "lineup_avg_spine_experience",
        "lineup_avg_halves_experience",
        "lineup_avg_middles_experience",
        "lineup_avg_edges_experience",
        "lineup_avg_outside_backs_experience",
        "lineup_avg_interchange_experience",
        "lineup_avg_named_margin_rating",
        "lineup_avg_spine_margin_rating",
        "lineup_avg_halves_margin_rating",
        "lineup_avg_middles_margin_rating",
        "lineup_avg_edges_margin_rating",
        "lineup_avg_outside_backs_margin_rating",
        "lineup_avg_interchange_margin_rating",
        "lineup_debutant_count",
        "lineup_named_cohesion",
        "lineup_spine_cohesion",
        "lineup_halves_pair_cohesion",
        "lineup_recent_named_stability",
        "lineup_recent_spine_stability",
        "lineup_snapshot_count",
        "lineup_snapshot_window_hours",
        "lineup_named_change_count",
        "lineup_named_change_rate",
        "lineup_spine_change_count",
        "lineup_spine_change_rate",
    ]:
        for side in ("home", "away"):
            col = f"{base}_{side}"
            result[col] = pd.to_numeric(result.get(col), errors="coerce").fillna(0.0)

    delta_frame = pd.DataFrame(
        {
            "lineup_named_count_delta": result["lineup_named_count_home"] - result["lineup_named_count_away"],
            "lineup_interchange_count_delta": (
                result["lineup_interchange_count_home"] - result["lineup_interchange_count_away"]
            ),
            "lineup_reserve_count_delta": result["lineup_reserve_count_home"] - result["lineup_reserve_count_away"],
            "lineup_spine_count_delta": result["lineup_spine_count_home"] - result["lineup_spine_count_away"],
            "lineup_spine_complete_delta": (
                result["lineup_spine_complete_home"] - result["lineup_spine_complete_away"]
            ),
            "lineup_bench_hooker_count_delta": (
                result["lineup_bench_hooker_count_home"] - result["lineup_bench_hooker_count_away"]
            ),
            "lineup_bench_spine_cover_count_delta": (
                result["lineup_bench_spine_cover_count_home"] - result["lineup_bench_spine_cover_count_away"]
            ),
            "lineup_source_age_hours_delta": (
                result["lineup_source_age_hours_home"] - result["lineup_source_age_hours_away"]
            ),
            "lineup_retained_ratio_delta": result["lineup_retained_ratio_home"] - result["lineup_retained_ratio_away"],
            "lineup_starters_retained_ratio_delta": (
                result["lineup_starters_retained_ratio_home"] - result["lineup_starters_retained_ratio_away"]
            ),
            "lineup_spine_retained_ratio_delta": (
                result["lineup_spine_retained_ratio_home"] - result["lineup_spine_retained_ratio_away"]
            ),
            "lineup_spine_same_as_prev_delta": (
                result["lineup_spine_same_as_prev_home"] - result["lineup_spine_same_as_prev_away"]
            ),
            "lineup_halves_pair_same_as_prev_delta": (
                result["lineup_halves_pair_same_as_prev_home"] - result["lineup_halves_pair_same_as_prev_away"]
            ),
            "lineup_avg_named_experience_delta": (
                result["lineup_avg_named_experience_home"] - result["lineup_avg_named_experience_away"]
            ),
            "lineup_avg_spine_experience_delta": (
                result["lineup_avg_spine_experience_home"] - result["lineup_avg_spine_experience_away"]
            ),
            "lineup_avg_halves_experience_delta": (
                result["lineup_avg_halves_experience_home"] - result["lineup_avg_halves_experience_away"]
            ),
            "lineup_avg_middles_experience_delta": (
                result["lineup_avg_middles_experience_home"] - result["lineup_avg_middles_experience_away"]
            ),
            "lineup_avg_edges_experience_delta": (
                result["lineup_avg_edges_experience_home"] - result["lineup_avg_edges_experience_away"]
            ),
            "lineup_avg_outside_backs_experience_delta": (
                result["lineup_avg_outside_backs_experience_home"] - result["lineup_avg_outside_backs_experience_away"]
            ),
            "lineup_avg_interchange_experience_delta": (
                result["lineup_avg_interchange_experience_home"] - result["lineup_avg_interchange_experience_away"]
            ),
            "lineup_avg_named_margin_rating_delta": (
                result["lineup_avg_named_margin_rating_home"] - result["lineup_avg_named_margin_rating_away"]
            ),
            "lineup_avg_spine_margin_rating_delta": (
                result["lineup_avg_spine_margin_rating_home"] - result["lineup_avg_spine_margin_rating_away"]
            ),
            "lineup_avg_halves_margin_rating_delta": (
                result["lineup_avg_halves_margin_rating_home"] - result["lineup_avg_halves_margin_rating_away"]
            ),
            "lineup_avg_middles_margin_rating_delta": (
                result["lineup_avg_middles_margin_rating_home"] - result["lineup_avg_middles_margin_rating_away"]
            ),
            "lineup_avg_edges_margin_rating_delta": (
                result["lineup_avg_edges_margin_rating_home"] - result["lineup_avg_edges_margin_rating_away"]
            ),
            "lineup_avg_outside_backs_margin_rating_delta": (
                result["lineup_avg_outside_backs_margin_rating_home"] - result["lineup_avg_outside_backs_margin_rating_away"]
            ),
            "lineup_avg_interchange_margin_rating_delta": (
                result["lineup_avg_interchange_margin_rating_home"] - result["lineup_avg_interchange_margin_rating_away"]
            ),
            "lineup_debutant_count_delta": (
                result["lineup_debutant_count_home"] - result["lineup_debutant_count_away"]
            ),
            "lineup_named_cohesion_delta": result["lineup_named_cohesion_home"] - result["lineup_named_cohesion_away"],
            "lineup_spine_cohesion_delta": result["lineup_spine_cohesion_home"] - result["lineup_spine_cohesion_away"],
            "lineup_halves_pair_cohesion_delta": (
                result["lineup_halves_pair_cohesion_home"] - result["lineup_halves_pair_cohesion_away"]
            ),
            "lineup_recent_named_stability_delta": (
                result["lineup_recent_named_stability_home"] - result["lineup_recent_named_stability_away"]
            ),
            "lineup_recent_spine_stability_delta": (
                result["lineup_recent_spine_stability_home"] - result["lineup_recent_spine_stability_away"]
            ),
            "lineup_snapshot_count_delta": result["lineup_snapshot_count_home"] - result["lineup_snapshot_count_away"],
            "lineup_snapshot_window_hours_delta": (
                result["lineup_snapshot_window_hours_home"] - result["lineup_snapshot_window_hours_away"]
            ),
            "lineup_named_change_count_delta": (
                result["lineup_named_change_count_home"] - result["lineup_named_change_count_away"]
            ),
            "lineup_named_change_rate_delta": (
                result["lineup_named_change_rate_home"] - result["lineup_named_change_rate_away"]
            ),
            "lineup_spine_change_count_delta": (
                result["lineup_spine_change_count_home"] - result["lineup_spine_change_count_away"]
            ),
            "lineup_spine_change_rate_delta": (
                result["lineup_spine_change_rate_home"] - result["lineup_spine_change_rate_away"]
            ),
            "lineup_expected_named_count_delta": (
                result["lineup_expected_named_count_home"] - result["lineup_expected_named_count_away"]
            ),
            "lineup_expected_interchange_count_delta": (
                result["lineup_expected_interchange_count_home"] - result["lineup_expected_interchange_count_away"]
            ),
            "lineup_expected_spine_count_delta": (
                result["lineup_expected_spine_count_home"] - result["lineup_expected_spine_count_away"]
            ),
            "lineup_selection_uncertainty_delta": (
                result["lineup_selection_uncertainty_home"] - result["lineup_selection_uncertainty_away"]
            ),
        },
        index=result.index,
    )
    result = pd.concat([result, delta_frame], axis=1)

    result["lineup_features_missing"] = (
        (result["lineup_data_available_home"] <= 0.0) | (result["lineup_data_available_away"] <= 0.0)
    ).astype(float)

    result["lineup_home_players"] = result.get("lineup_home_players", "").fillna("").astype(str)
    result["lineup_away_players"] = result.get("lineup_away_players", "").fillna("").astype(str)

    for col in LINEUP_FEATURE_COLUMNS:
        if col not in result.columns:
            if col in {"lineup_home_players", "lineup_away_players"}:
                result[col] = ""
            elif col == "lineup_features_missing":
                result[col] = 1.0
            else:
                result[col] = 0.0

    numeric_cols = [
        col
        for col in LINEUP_FEATURE_COLUMNS
        if col not in {"game_id", "lineup_home_players", "lineup_away_players"}
    ]
    for col in numeric_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0)

    result["lineup_home_players"] = result["lineup_home_players"].fillna("").astype(str)
    result["lineup_away_players"] = result["lineup_away_players"].fillna("").astype(str)

    return result[LINEUP_FEATURE_COLUMNS]
