"""Weather observations for games via Open-Meteo (free, no key).

Historical rows use the ERA5 archive; upcoming games use the forecast API
(16-day horizon comfortably covers the prediction window). Values are stored
in `weather_observations` keyed by game_id; the feature builder only reads
the table, so fetch failures degrade to missing flags, never errors.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
import time
from pathlib import Path

import requests

from pipeline.common import console

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/era5"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_SLEEP_SECONDS = 0.1

HOURLY_FIELDS = "temperature_2m,precipitation,wind_speed_10m,relative_humidity_2m"


def ensure_table(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS weather_observations (
            game_id INTEGER PRIMARY KEY,
            provider TEXT,
            temp_c REAL,
            precip_mm_3h REAL,
            precip_mm_24h REAL,
            wind_speed_kmh REAL,
            humidity_pct REAL,
            fetched_at_utc TEXT
        );
        """
    )


def _games_needing_weather(con: sqlite3.Connection, refresh_upcoming: bool) -> list[dict]:
    """Games with venue coordinates and either no weather row yet or an
    upcoming kickoff (forecasts refresh until the game is played)."""
    rows = []
    for row in con.execute(
        """
        SELECT f.game_id, f.start_time_utc, f.game_state_name,
               v.latitude, v.longitude
        FROM feed_cache_fixtures f
        JOIN venue_locations v ON v.venue_name = f.venue_name
        LEFT JOIN weather_observations w
               ON w.game_id = CAST(f.game_id AS INTEGER)
        WHERE f.start_time_utc IS NOT NULL
          AND (
                w.game_id IS NULL
                OR (? AND f.game_state_name = 'Pre Game')
              )
        """,
        (1 if refresh_upcoming else 0,),
    ):
        rows.append(
            {
                "game_id": int(float(row[0])),
                "start_time_utc": float(row[1]),
                "state": row[2],
                "lat": row[3],
                "lon": row[4],
            }
        )
    return rows


def _extract_at_kickoff(payload: dict, kickoff_utc: dt.datetime) -> dict | None:
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return None
    target = kickoff_utc.strftime("%Y-%m-%dT%H:00")
    try:
        idx = times.index(target)
    except ValueError:
        return None

    def series(name):
        values = hourly.get(name) or []
        return values[idx] if idx < len(values) else None

    precip = hourly.get("precipitation") or []
    precip_3h = sum(v or 0.0 for v in precip[max(0, idx - 3) : idx])
    precip_24h = sum(v or 0.0 for v in precip[max(0, idx - 24) : idx])
    return {
        "temp_c": series("temperature_2m"),
        "wind_speed_kmh": series("wind_speed_10m"),
        "humidity_pct": series("relative_humidity_2m"),
        "precip_mm_3h": round(precip_3h, 2),
        "precip_mm_24h": round(precip_24h, 2),
    }


def _fetch_game_weather(game: dict) -> tuple[str, dict] | None:
    kickoff = dt.datetime.fromtimestamp(game["start_time_utc"], tz=dt.timezone.utc)
    now = dt.datetime.now(dt.timezone.utc)
    # ERA5 lags realtime by ~5 days; anything newer uses the forecast API
    use_archive = kickoff < now - dt.timedelta(days=6)

    start_date = (kickoff - dt.timedelta(days=1)).date().isoformat()
    end_date = kickoff.date().isoformat()
    params = {
        "latitude": game["lat"],
        "longitude": game["lon"],
        "hourly": HOURLY_FIELDS,
        "timezone": "UTC",
    }
    if use_archive:
        url = ARCHIVE_URL
        params.update({"start_date": start_date, "end_date": end_date})
        provider = "era5"
    else:
        url = FORECAST_URL
        params.update(
            {"start_date": start_date, "end_date": end_date, "past_days": 0}
        )
        provider = "forecast"

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    values = _extract_at_kickoff(response.json(), kickoff)
    if values is None:
        return None
    return provider, values


def fetch_weather_for_games(
    db_path: str | Path,
    refresh_upcoming: bool = True,
    max_requests: int | None = None,
) -> dict:
    con = sqlite3.connect(str(db_path))
    try:
        ensure_table(con)
        games = _games_needing_weather(con, refresh_upcoming)
        fetched = 0
        errors = 0
        now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
        for game in games:
            if max_requests is not None and fetched >= max_requests:
                break
            try:
                result = _fetch_game_weather(game)
            except Exception:
                errors += 1
                continue
            time.sleep(REQUEST_SLEEP_SECONDS)
            if result is None:
                errors += 1
                continue
            provider, values = result
            con.execute(
                """
                INSERT INTO weather_observations
                    (game_id, provider, temp_c, precip_mm_3h, precip_mm_24h,
                     wind_speed_kmh, humidity_pct, fetched_at_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_id) DO UPDATE SET
                    provider = excluded.provider,
                    temp_c = excluded.temp_c,
                    precip_mm_3h = excluded.precip_mm_3h,
                    precip_mm_24h = excluded.precip_mm_24h,
                    wind_speed_kmh = excluded.wind_speed_kmh,
                    humidity_pct = excluded.humidity_pct,
                    fetched_at_utc = excluded.fetched_at_utc
                """,
                (
                    game["game_id"],
                    provider,
                    values["temp_c"],
                    values["precip_mm_3h"],
                    values["precip_mm_24h"],
                    values["wind_speed_kmh"],
                    values["humidity_pct"],
                    now,
                ),
            )
            fetched += 1
            if fetched % 50 == 0:
                con.commit()
        con.commit()
        print(f"[nrl-data] weather: {fetched} games updated, {errors} errors.")
        console.emit_result(
            "freshness",
            source="weather",
            detail=f"{fetched} games updated · {errors} errors",
        )
        return {"fetched": fetched, "errors": errors, "candidates": len(games)}
    finally:
        con.close()
