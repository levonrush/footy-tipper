"""Match-context feature families from the nrl.com ingestion tables.

Families (each fails soft to missing flags so train/inference never break):
- team form: leak-safe EWMA (halflife 5 matches, shifted one game) over
  per-match team stats -> form_<stat>_home/away/delta
- referee: categorical referee_name + leak-safe rolling penalty/sin-bin rates
- weather: Open-Meteo observations + match centre labels -> wx_* numerics,
  wet flag, ground_condition categorical
- travel: haversine from each team's home base to the venue + timezone shift

Everything is keyed by game_id (float64, matching the prepared tables) and
assembled by build_match_context_features(), the single merge point used by
train.py, inference.py, and evaluate.py.
"""

from __future__ import annotations

import csv
import datetime as dt
import math
import sqlite3
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

FORM_HALFLIFE_GAMES = 5.0
REF_HALFLIFE_GAMES = 20.0

# match_team_stats stat_name slugs used for form (modern match centre groups;
# older seasons lack some, which EWMA handles as missing)
TEAM_FORM_STATS = [
    "possession_pct",
    "completion_rate",
    "all_run_metres",
    "post_contact_metres",
    "line_breaks",
    "tackle_breaks",
    "offloads",
    "kicking_metres",
    "effective_tackle_pct",
    "missed_tackles",
    "errors",
    "penalties_conceded",
]

WET_LABELS = {"rain", "raining", "showers", "wet", "drizzle", "storm", "thunderstorm"}

DEFAULT_TEAM_BASES_CSV = Path("data") / "reference" / "team_home_venues.csv"

CONTEXT_CATEGORICAL_COLUMNS = ["referee_name", "ground_condition"]


def _fixtures_frame(con: sqlite3.Connection) -> pd.DataFrame:
    fixtures = pd.read_sql_query(
        """
        SELECT game_id, competition_year, round_id, game_state_name,
               start_time_utc, venue_name, team_home, team_away
        FROM feed_cache_fixtures
        WHERE start_time_utc IS NOT NULL
        """,
        con,
    )
    fixtures["game_id"] = pd.to_numeric(fixtures["game_id"], errors="coerce")
    fixtures["start_time_utc"] = pd.to_numeric(fixtures["start_time_utc"], errors="coerce")
    fixtures = fixtures.dropna(subset=["game_id", "start_time_utc"])
    fixtures["game_id_int"] = fixtures["game_id"].astype("int64")
    return fixtures.sort_values("start_time_utc").reset_index(drop=True)


# ── team form ─────────────────────────────────────────────────────────────────


def build_team_form_features(con: sqlite3.Connection, fixtures: pd.DataFrame) -> pd.DataFrame:
    stats = pd.read_sql_query(
        "SELECT game_id, side, stat_name, value FROM match_team_stats",
        con,
    )
    if stats.empty:
        return pd.DataFrame(columns=["game_id"])
    stats = stats[stats["stat_name"].isin(TEAM_FORM_STATS)]
    wide = stats.pivot_table(
        index=["game_id", "side"], columns="stat_name", values="value", aggfunc="last"
    ).reset_index()

    sides = []
    for side, team_col in (("home", "team_home"), ("away", "team_away")):
        frame = fixtures[["game_id_int", "start_time_utc", team_col]].rename(
            columns={team_col: "team"}
        )
        frame["side"] = side
        sides.append(frame)
    long_fixtures = pd.concat(sides, ignore_index=True)

    merged = long_fixtures.merge(
        wide, left_on=["game_id_int", "side"], right_on=["game_id", "side"], how="left"
    )
    merged = merged.sort_values("start_time_utc")

    available = [col for col in TEAM_FORM_STATS if col in merged.columns]
    if not available:
        return pd.DataFrame(columns=["game_id"])

    grouped = merged.groupby("team", sort=False)
    for col in available:
        shifted = grouped[col].shift(1)
        merged[f"form_{col}"] = (
            shifted.groupby(merged["team"])
            .transform(lambda s: s.ewm(halflife=FORM_HALFLIFE_GAMES, min_periods=1).mean())
        )
    merged["form_features_missing"] = (
        merged[[f"form_{col}" for col in available]].isna().all(axis=1).astype(float)
    )

    out = None
    for side in ("home", "away"):
        side_frame = merged[merged["side"] == side][
            ["game_id_int"]
            + [f"form_{col}" for col in available]
            + ["form_features_missing"]
        ].rename(
            columns={
                **{f"form_{col}": f"form_{col}_{side}" for col in available},
                "form_features_missing": f"form_features_missing_{side}",
            }
        )
        out = side_frame if out is None else out.merge(side_frame, on="game_id_int", how="outer")

    for col in available:
        out[f"form_{col}_delta"] = out[f"form_{col}_home"] - out[f"form_{col}_away"]
    return out


# ── referee ───────────────────────────────────────────────────────────────────


def build_referee_features(con: sqlite3.Connection, fixtures: pd.DataFrame) -> pd.DataFrame:
    refs = pd.read_sql_query(
        """
        SELECT game_id, MIN(official_name) AS referee_name
        FROM match_officials
        WHERE role = 'Referee'
        GROUP BY game_id
        """,
        con,
    )
    if refs.empty:
        return pd.DataFrame(columns=["game_id"])

    per_game = pd.read_sql_query(
        """
        SELECT game_id,
               SUM(CASE WHEN stat_name = 'penalties_conceded' THEN value END) AS game_penalties,
               SUM(CASE WHEN stat_name = 'sin_bins' THEN value END) AS game_sin_bins
        FROM match_team_stats
        GROUP BY game_id
        """,
        con,
    )

    frame = fixtures[["game_id_int", "start_time_utc", "game_state_name"]].merge(
        refs, left_on="game_id_int", right_on="game_id", how="inner"
    )
    frame = frame.merge(per_game, left_on="game_id_int", right_on="game_id", how="left")
    frame = frame.sort_values("start_time_utc")

    grouped = frame.groupby("referee_name", sort=False)
    frame["ref_games_officiated"] = grouped.cumcount().astype(float)
    for source, target in (
        ("game_penalties", "ref_penalty_rate_ewma"),
        ("game_sin_bins", "ref_sin_bin_rate_ewma"),
    ):
        shifted = grouped[source].shift(1)
        frame[target] = (
            shifted.groupby(frame["referee_name"])
            .transform(lambda s: s.ewm(halflife=REF_HALFLIFE_GAMES, min_periods=1).mean())
        )

    out = frame[
        [
            "game_id_int",
            "referee_name",
            "ref_games_officiated",
            "ref_penalty_rate_ewma",
            "ref_sin_bin_rate_ewma",
        ]
    ].copy()
    out["ref_missing"] = 0.0
    return out


# ── weather ───────────────────────────────────────────────────────────────────


def build_weather_features(con: sqlite3.Connection, fixtures: pd.DataFrame) -> pd.DataFrame:
    tables = {
        row[0]
        for row in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    frames = fixtures[["game_id_int"]].copy()

    if "weather_observations" in tables:
        weather = pd.read_sql_query(
            """
            SELECT game_id, temp_c, precip_mm_3h, precip_mm_24h,
                   wind_speed_kmh, humidity_pct
            FROM weather_observations
            """,
            con,
        )
        frames = frames.merge(weather, left_on="game_id_int", right_on="game_id", how="left")
    else:
        for col in ("temp_c", "precip_mm_3h", "precip_mm_24h", "wind_speed_kmh", "humidity_pct"):
            frames[col] = np.nan

    if "match_context" in tables:
        context = pd.read_sql_query(
            "SELECT game_id, weather_label, ground_condition FROM match_context",
            con,
        )
        frames = frames.merge(
            context, left_on="game_id_int", right_on="game_id", how="left", suffixes=("", "_ctx")
        )
    else:
        frames["weather_label"] = None
        frames["ground_condition"] = None

    label_wet = (
        frames["weather_label"].fillna("").str.strip().str.lower().isin(WET_LABELS)
    )
    condition_wet = (
        frames["ground_condition"].fillna("").str.strip().str.lower().isin({"wet", "heavy", "muddy"})
    )
    rain_wet = (
        pd.to_numeric(frames["precip_mm_3h"], errors="coerce").fillna(0.0) > 0.5
    ) | (pd.to_numeric(frames["precip_mm_24h"], errors="coerce").fillna(0.0) > 5.0)

    out = pd.DataFrame(
        {
            "game_id_int": frames["game_id_int"],
            "wx_temp": pd.to_numeric(frames["temp_c"], errors="coerce"),
            "wx_rain_3h": pd.to_numeric(frames["precip_mm_3h"], errors="coerce"),
            "wx_rain_24h": pd.to_numeric(frames["precip_mm_24h"], errors="coerce"),
            "wx_wind": pd.to_numeric(frames["wind_speed_kmh"], errors="coerce"),
            "wx_humidity": pd.to_numeric(frames["humidity_pct"], errors="coerce"),
            "wx_wet": (label_wet | condition_wet | rain_wet).astype(float),
            "ground_condition": frames["ground_condition"],
        }
    )
    out["weather_missing"] = out["wx_temp"].isna().astype(float)
    return out


# ── travel ────────────────────────────────────────────────────────────────────


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    radius = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def _load_team_bases(csv_path: str | Path) -> dict[str, dict]:
    bases: dict[str, dict] = {}
    path = Path(csv_path)
    if not path.exists():
        return bases
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            bases[row["team"]] = {
                "lat": float(row["latitude"]),
                "lon": float(row["longitude"]),
                "timezone": row["timezone"],
            }
    return bases


def _tz_offset_hours(timezone_name: str, when_utc: dt.datetime) -> float | None:
    try:
        offset = ZoneInfo(timezone_name).utcoffset(when_utc)
    except Exception:
        return None
    return offset.total_seconds() / 3600 if offset is not None else None


def build_travel_features(
    con: sqlite3.Connection,
    fixtures: pd.DataFrame,
    team_bases_csv: str | Path = DEFAULT_TEAM_BASES_CSV,
) -> pd.DataFrame:
    bases = _load_team_bases(team_bases_csv)
    venues = {
        row[0]: {"lat": row[1], "lon": row[2], "timezone": row[3]}
        for row in con.execute(
            "SELECT venue_name, latitude, longitude, timezone FROM venue_locations"
        )
    }
    if not bases or not venues:
        return pd.DataFrame(columns=["game_id"])

    records = []
    for row in fixtures.itertuples(index=False):
        venue = venues.get(row.venue_name)
        kickoff = dt.datetime.fromtimestamp(row.start_time_utc, tz=dt.timezone.utc)
        record = {"game_id_int": row.game_id_int}
        for side, team in (("home", row.team_home), ("away", row.team_away)):
            base = bases.get(team)
            if base is None or venue is None:
                record[f"travel_km_{side}"] = np.nan
                record[f"tz_shift_{side}"] = np.nan
                continue
            record[f"travel_km_{side}"] = round(
                _haversine_km(base["lat"], base["lon"], venue["lat"], venue["lon"]), 1
            )
            base_offset = _tz_offset_hours(base["timezone"], kickoff)
            venue_offset = _tz_offset_hours(venue["timezone"], kickoff)
            record[f"tz_shift_{side}"] = (
                abs(venue_offset - base_offset)
                if base_offset is not None and venue_offset is not None
                else np.nan
            )
        records.append(record)

    out = pd.DataFrame.from_records(records)
    out["travel_km_delta"] = out["travel_km_away"] - out["travel_km_home"]
    out["travel_missing"] = out[["travel_km_home", "travel_km_away"]].isna().any(axis=1).astype(float)
    return out


# ── assembly ──────────────────────────────────────────────────────────────────


def build_match_context_features(
    db_path: str | Path,
    matches_df: pd.DataFrame,
    team_bases_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Context features for the requested games, keyed by game_id (float64)."""
    requested = pd.DataFrame(
        {"game_id": pd.to_numeric(matches_df["game_id"], errors="coerce")}
    ).dropna()
    requested["game_id_int"] = requested["game_id"].astype("int64")

    con = sqlite3.connect(str(db_path))
    try:
        fixtures = _fixtures_frame(con)
        result = requested[["game_id", "game_id_int"]].drop_duplicates()

        family_builders = [
            ("team form", build_team_form_features, {}),
            ("referee", build_referee_features, {}),
            ("weather", build_weather_features, {}),
            (
                "travel",
                build_travel_features,
                {"team_bases_csv": team_bases_csv or DEFAULT_TEAM_BASES_CSV},
            ),
        ]
        for name, builder, kwargs in family_builders:
            try:
                frame = builder(con, fixtures, **kwargs)
            except Exception as exc:
                print(f"[nrl-data] context features: {name} family skipped ({exc}).")
                continue
            if frame is None or frame.empty or "game_id_int" not in frame.columns:
                continue
            keep = [col for col in frame.columns if col not in ("game_id",)]
            result = result.merge(frame[keep], on="game_id_int", how="left")

        return result.drop(columns=["game_id_int"])
    finally:
        con.close()


def context_numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [
        col
        for col in frame.columns
        if col not in ("game_id", *CONTEXT_CATEGORICAL_COLUMNS)
    ]


def merge_match_context_features(data: pd.DataFrame, db_path: str | Path) -> pd.DataFrame:
    """Single merge point for context + player-form features (fail-soft).

    Used identically by train.py, inference.py, and evaluate.py so the three
    scripts never diverge on feature availability.
    """
    if "game_id" not in data.columns or data.empty:
        return data

    context = build_match_context_features(db_path, data)
    if context is not None and not context.empty and len(context.columns) > 1:
        data = data.merge(context, on="game_id", how="left")

    try:
        from ..lineups.player_form import compute_lineup_player_form_features

        player_form = compute_lineup_player_form_features(db_path, data)
        if player_form is not None and not player_form.empty and len(player_form.columns) > 1:
            data = data.merge(player_form, on="game_id", how="left")
    except Exception as exc:
        print(f"[nrl-data] player form features skipped ({exc}).")

    for col in CONTEXT_CATEGORICAL_COLUMNS:
        if col in data.columns:
            data[col] = data[col].fillna("Unknown")
    return data
