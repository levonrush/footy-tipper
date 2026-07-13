"""Player form ratings aggregated over the named/actual lineup per side.

Per-player leak-safe EWMA (halflife 5 appearances) over match centre player
stats, aggregated per game and side:

- Final (played) games use the actual roster from match_player_stats with
  each player's form as of the PREVIOUS appearance (shift-by-one).
- Pre Game rows use the latest lineup_entries snapshot for the round, with
  each named player's form after their most recent appearance.

Players are matched by nrl.com player id (match centre playerId equals the
lineup player_external_id space) with normalized-name fallback. Spine = jersey
numbers 1/6/7/9.

Output is keyed by game_id (float64) and merged next to the existing lineup
features in train/inference/evaluate.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

FORM_HALFLIFE_APPEARANCES = 5.0
SPINE_JERSEYS = {1, 6, 7, 9}

PLAYER_FORM_STATS = {
    "fantasy": "fantasy_points_total",
    "run_metres": "all_run_metres",
    "tackles": "tackles_made",
    "errors": "errors",
}

PLAYER_FORM_FEATURE_PREFIX = "lineup_form_"


def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    return (
        con.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
        ).fetchone()
        is not None
    )


def _player_ref(player_id, player_key) -> str | None:
    if player_id is not None and not pd.isna(player_id):
        return f"pid_{int(player_id)}"
    if isinstance(player_key, str) and player_key:
        return player_key
    return None


def _load_player_history(con: sqlite3.Connection) -> pd.DataFrame:
    stats_cols = ", ".join(PLAYER_FORM_STATS.values())
    history = pd.read_sql_query(
        f"""
        SELECT p.game_id, p.side, p.player_id, p.player_key, p.jersey_number,
               p.tries, p.try_assists, p.line_breaks, p.line_break_assists,
               {stats_cols},
               f.start_time_utc
        FROM match_player_stats p
        JOIN feed_cache_fixtures f ON CAST(f.game_id AS INTEGER) = p.game_id
        WHERE f.start_time_utc IS NOT NULL
        """,
        con,
    )
    if history.empty:
        return history
    history["start_time_utc"] = pd.to_numeric(history["start_time_utc"], errors="coerce")
    history["player_ref"] = [
        _player_ref(pid, key)
        for pid, key in zip(history["player_id"], history["player_key"])
    ]
    history = history.dropna(subset=["player_ref", "start_time_utc"])
    history["involvements"] = (
        history[["tries", "try_assists", "line_breaks", "line_break_assists"]]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .sum(axis=1)
    )
    return history.sort_values("start_time_utc").reset_index(drop=True)


def _with_form_columns(history: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Adds form_<name> (as-of previous appearance) per player row."""
    form_cols = []
    grouped = history.groupby("player_ref", sort=False)
    stat_map = dict(PLAYER_FORM_STATS)
    stat_map["involvements"] = "involvements"
    for name, source in stat_map.items():
        values = pd.to_numeric(history[source], errors="coerce")
        ewma_incl = (
            values.groupby(history["player_ref"])
            .transform(lambda s: s.ewm(halflife=FORM_HALFLIFE_APPEARANCES, min_periods=1).mean())
        )
        history[f"_ewma_incl_{name}"] = ewma_incl
        history[f"form_{name}"] = grouped[f"_ewma_incl_{name}"].shift(1)
        form_cols.append(f"form_{name}")
    return history, form_cols


def _latest_form(history: pd.DataFrame, form_names: list[str]) -> pd.DataFrame:
    """Per player: EWMA including their most recent appearance."""
    last_rows = history.groupby("player_ref", sort=False).tail(1)
    out = last_rows[["player_ref"]].copy()
    for name in form_names:
        out[name] = last_rows[f"_ewma_incl_{name.removeprefix('form_')}"].to_numpy()
    return out


def _aggregate_side(roster: pd.DataFrame, form_cols: list[str]) -> dict:
    values: dict = {}
    with_form = roster.dropna(subset=[form_cols[0]]) if len(roster) else roster
    coverage = len(with_form) / len(roster) if len(roster) else 0.0
    for col in form_cols:
        name = col.removeprefix("form_")
        values[name] = (
            float(pd.to_numeric(roster[col], errors="coerce").mean())
            if len(roster)
            else np.nan
        )
    spine = roster[
        pd.to_numeric(roster.get("jersey_number"), errors="coerce").isin(SPINE_JERSEYS)
    ]
    values["spine_fantasy"] = (
        float(pd.to_numeric(spine["form_fantasy"], errors="coerce").mean())
        if len(spine)
        else np.nan
    )
    values["coverage"] = coverage
    return values


def compute_lineup_player_form_features(
    db_path: str | Path, matches_df: pd.DataFrame
) -> pd.DataFrame:
    from .normalization import normalize_team_name

    requested = matches_df[["game_id"]].copy()
    requested["game_id"] = pd.to_numeric(requested["game_id"], errors="coerce")
    requested = requested.dropna().drop_duplicates()
    requested["game_id_int"] = requested["game_id"].astype("int64")

    con = sqlite3.connect(str(db_path))
    try:
        if not _table_exists(con, "match_player_stats"):
            return pd.DataFrame(columns=["game_id"])
        history = _load_player_history(con)
        if history.empty:
            return pd.DataFrame(columns=["game_id"])
        history, form_cols = _with_form_columns(history)
        latest = _latest_form(history, form_cols)

        fixtures = pd.read_sql_query(
            """
            SELECT game_id, competition_year, round_id, game_state_name,
                   team_home, team_away
            FROM feed_cache_fixtures
            """,
            con,
        )
        fixtures["game_id_int"] = pd.to_numeric(
            fixtures["game_id"], errors="coerce"
        ).astype("Int64")
        fixtures = fixtures.dropna(subset=["game_id_int"])
        fixtures = fixtures[fixtures["game_id_int"].isin(requested["game_id_int"])]

        # index played rosters once; per-game boolean scans over the full
        # history are quadratic and dominate training-time feature builds
        roster_index = {
            key: group[["player_ref", "jersey_number", *form_cols]]
            for key, group in history.groupby(["game_id", "side"], sort=False)
        }

        lineup_entries = pd.DataFrame()
        if _table_exists(con, "lineup_entries"):
            lineup_entries = pd.read_sql_query(
                """
                SELECT competition_year, round_id, team_key, snapshot_id,
                       player_external_id, player_key, jersey_number
                FROM lineup_entries
                WHERE round_id IS NOT NULL
                """,
                con,
            )

        records = []
        for fixture in fixtures.itertuples(index=False):
            game_id_int = int(fixture.game_id_int)
            record: dict = {"game_id_int": game_id_int}
            for side, team in (("home", fixture.team_home), ("away", fixture.team_away)):
                roster = pd.DataFrame()
                if fixture.game_state_name == "Final":
                    roster = roster_index.get((game_id_int, side), pd.DataFrame())
                if roster.empty and not lineup_entries.empty:
                    named = lineup_entries[
                        (
                            pd.to_numeric(lineup_entries["competition_year"], errors="coerce")
                            == float(fixture.competition_year)
                        )
                        & (
                            pd.to_numeric(lineup_entries["round_id"], errors="coerce")
                            == float(fixture.round_id)
                        )
                        & (lineup_entries["team_key"] == normalize_team_name(team))
                    ]
                    if not named.empty:
                        named = named[named["snapshot_id"] == named["snapshot_id"].max()].copy()
                        named["player_ref"] = [
                            _player_ref(
                                pd.to_numeric(pid, errors="coerce"), key
                            )
                            for pid, key in zip(
                                named["player_external_id"], named["player_key"]
                            )
                        ]
                        roster = named[["player_ref", "player_key", "jersey_number"]].merge(
                            latest.rename(columns=dict(zip(form_cols, form_cols))),
                            on="player_ref",
                            how="left",
                        )
                        # name-key fallback for ids that differ between sources
                        name_latest = (
                            history.groupby("player_key", sort=False).tail(1)[
                                ["player_key"]
                                + [f"_ewma_incl_{c.removeprefix('form_')}" for c in form_cols]
                            ].rename(
                                columns={
                                    f"_ewma_incl_{c.removeprefix('form_')}": f"_name_{c}"
                                    for c in form_cols
                                }
                            )
                        )
                        roster = roster.merge(name_latest, on="player_key", how="left")
                        for col in form_cols:
                            roster[col] = roster[col].fillna(roster[f"_name_{col}"])

                if roster.empty:
                    record[f"{PLAYER_FORM_FEATURE_PREFIX}missing_{side}"] = 1.0
                    continue

                aggregated = _aggregate_side(roster, form_cols)
                record[f"{PLAYER_FORM_FEATURE_PREFIX}missing_{side}"] = float(
                    aggregated["coverage"] == 0.0
                )
                record[f"{PLAYER_FORM_FEATURE_PREFIX}coverage_{side}"] = aggregated["coverage"]
                for col in form_cols:
                    name = col.removeprefix("form_")
                    record[f"{PLAYER_FORM_FEATURE_PREFIX}{name}_{side}"] = aggregated[name]
                record[f"lineup_spine_form_fantasy_{side}"] = aggregated["spine_fantasy"]
            records.append(record)

        if not records:
            return pd.DataFrame(columns=["game_id"])
        out = pd.DataFrame.from_records(records)
        for delta_col, home_col, away_col in (
            ("lineup_form_fantasy_delta", "lineup_form_fantasy_home", "lineup_form_fantasy_away"),
            (
                "lineup_spine_form_fantasy_delta",
                "lineup_spine_form_fantasy_home",
                "lineup_spine_form_fantasy_away",
            ),
        ):
            if home_col in out.columns and away_col in out.columns:
                out[delta_col] = out[home_col] - out[away_col]

        out = requested.merge(out, on="game_id_int", how="left").drop(
            columns=["game_id_int"]
        )
        return out
    finally:
        con.close()
