"""Club attention volume: a sign-neutral, high-prevalence context measure.

This module deliberately measures *how much* a club is being written about, not
*how* it is being written about.  Tone is downstream of results the model
already has, and the source policy forbids retaining article bodies, so no text
is fetched, stored, embedded, or scored here.  Only daily counts are kept.

GDELT's DOC 2.0 API in ``timelinevolraw`` mode returns a whole daily series for
one query, so a club-season costs one request rather than one per round.  Its
coverage begins in 2017; earlier seasons are recorded as missing rather than
zero, because "no coverage" and "no coverage of this club" are different facts.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
import time
import urllib.parse
import urllib.request
import uuid
from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import ensure_context_tables
from .time import utc_iso, utc_now


GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_SOURCE_KEY = "gdelt_doc_timelinevolraw"

# GDELT DOC 2.0 indexes from 2017-01-01.  Asking for earlier dates returns an
# empty series, which must not be read as "nobody wrote about this club".
GDELT_COVERAGE_START_YEAR = 2017

# The published guidance is one request every five seconds.  Exceeded, the API
# returns a plain-text scolding with HTTP 200, so the floor is enforced here
# rather than relied on being enforced upstream.
RATE_LIMIT_SECONDS = 5.5

ATTENTION_WINDOW_DAYS = 7
ATTENTION_BASELINE_WEEKS = 26
ATTENTION_MIN_BASELINE_WEEKS = 8

# Exact phrases per club.  Nicknames alone are ambiguous across codes and
# countries ("Warriors", "Tigers", "Roosters", "Dolphins"), so every phrase
# carries a disambiguating city or full club name.
TEAM_QUERY_PHRASES: dict[str, tuple[str, ...]] = {
    "broncos": ("Brisbane Broncos",),
    "bulldogs": ("Canterbury Bulldogs", "Canterbury-Bankstown Bulldogs"),
    "cowboys": ("North Queensland Cowboys",),
    "dolphins": ("Redcliffe Dolphins", "NRL Dolphins"),
    "dragons": ("St George Illawarra",),
    "eels": ("Parramatta Eels",),
    "knights": ("Newcastle Knights",),
    "panthers": ("Penrith Panthers",),
    "rabbitohs": ("South Sydney Rabbitohs",),
    "raiders": ("Canberra Raiders",),
    "roosters": ("Sydney Roosters",),
    "sea_eagles": ("Manly Sea Eagles", "Manly-Warringah Sea Eagles"),
    "sharks": ("Cronulla Sharks", "Cronulla-Sutherland Sharks"),
    "storm": ("Melbourne Storm",),
    "tigers": ("Wests Tigers",),
    "titans": ("Gold Coast Titans",),
    "warriors": ("New Zealand Warriors",),
}


# Side metrics; ``features.py`` owns the full column contract built from these,
# plus the two per-side missing flags.
ATTENTION_METRICS = ("attention_rate", "attention_index", "attention_z")


def _gdelt_stamp(value: dt.date | dt.datetime) -> str:
    if isinstance(value, dt.datetime):
        return value.strftime("%Y%m%d%H%M%S")
    return value.strftime("%Y%m%d") + "000000"


def _query_expression(team_key: str) -> str:
    phrases = TEAM_QUERY_PHRASES.get(team_key)
    if not phrases:
        raise KeyError(f"no GDELT query phrase for team {team_key!r}")
    if len(phrases) == 1:
        return f'"{phrases[0]}"'
    joined = " OR ".join(f'"{phrase}"' for phrase in phrases)
    return f"({joined})"


def fetch_team_volume(
    team_key: str,
    start: dt.date,
    end: dt.date,
    *,
    timeout: int = 30,
    opener=urllib.request.urlopen,
) -> list[dict[str, Any]]:
    """Daily article counts for one club over one window.

    Returns ``[{"observed_date", "article_count", "corpus_norm"}, ...]``.  A
    non-JSON body means GDELT refused (rate limit, malformed query) and raises,
    so the caller can record the failure instead of storing a silent zero.
    """

    params = {
        "query": _query_expression(team_key),
        "mode": "timelinevolraw",
        "startdatetime": _gdelt_stamp(start),
        "enddatetime": _gdelt_stamp(end),
        "format": "json",
    }
    url = f"{GDELT_DOC_API}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "footy-tipper/1.0"})
    with closing(opener(request, timeout=timeout)) as response:
        raw = response.read()
    text = raw.decode("utf-8", errors="replace").strip()
    if not text.startswith("{"):
        raise ValueError(f"GDELT returned a non-JSON body: {text[:160]}")
    body = json.loads(text)
    timeline = body.get("timeline") or []
    if not timeline:
        return []
    points = timeline[0].get("data") or []
    observations = []
    for point in points:
        stamp = str(point.get("date") or "")
        if len(stamp) < 8:
            continue
        observations.append(
            {
                "observed_date": f"{stamp[0:4]}-{stamp[4:6]}-{stamp[6:8]}",
                "article_count": float(point.get("value") or 0.0),
                "corpus_norm": float(point.get("norm") or 0.0),
            }
        )
    return observations


def store_team_volume(
    con: sqlite3.Connection,
    team_key: str,
    observations: Iterable[Mapping[str, Any]],
    *,
    source_key: str = GDELT_SOURCE_KEY,
    fetched_at_utc: Any | None = None,
) -> int:
    fetched = utc_iso(fetched_at_utc or utc_now())
    rows = [
        (
            team_key,
            str(item["observed_date"]),
            source_key,
            float(item["article_count"]),
            float(item["corpus_norm"]),
            fetched,
        )
        for item in observations
    ]
    if not rows:
        return 0
    con.executemany(
        """
        INSERT INTO context_attention_series (
            team_key, observed_date, source_key, article_count, corpus_norm,
            fetched_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(team_key, observed_date, source_key) DO UPDATE SET
            article_count = excluded.article_count,
            corpus_norm = excluded.corpus_norm,
            fetched_at_utc = excluded.fetched_at_utc
        """,
        rows,
    )
    return len(rows)


def backfill_attention(
    db_path: str | Path,
    *,
    start_year: int,
    end_year: int,
    team_keys: Sequence[str] | None = None,
    fetcher=fetch_team_volume,
    sleep=time.sleep,
    rate_limit_seconds: float = RATE_LIMIT_SECONDS,
    log=print,
) -> dict[str, Any]:
    """Fetch and store one daily series per club per season.

    Fails soft per club-season: a refused request is recorded as an error and
    the run continues, because a partial attention series is still honest data
    while a crashed backfill is not.
    """

    teams = list(team_keys or sorted(TEAM_QUERY_PHRASES))
    first_year = max(int(start_year), GDELT_COVERAGE_START_YEAR)
    last_year = int(end_year)
    run_id = str(uuid.uuid4())
    started = utc_now()
    errors: list[str] = []
    observation_count = 0

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(str(path))) as con:
        with con:
            ensure_context_tables(con)
            con.execute(
                """
                INSERT OR REPLACE INTO context_attention_runs (
                    run_id, source_key, started_at_utc, status, start_year,
                    end_year, team_count
                ) VALUES (?, ?, ?, 'started', ?, ?, ?)
                """,
                (
                    run_id,
                    GDELT_SOURCE_KEY,
                    utc_iso(started),
                    first_year,
                    last_year,
                    len(teams),
                ),
            )
        first_request = True
        for year in range(first_year, last_year + 1):
            # A season plus a lead-in quarter, so the trailing baseline has
            # history before round one.
            window_start = dt.date(year - 1, 11, 1)
            window_end = min(dt.date(year, 12, 31), dt.date.today())
            if window_end <= window_start:
                continue
            for team_key in teams:
                if not first_request and rate_limit_seconds:
                    sleep(rate_limit_seconds)
                first_request = False
                try:
                    observations = fetcher(team_key, window_start, window_end)
                except Exception as exc:
                    errors.append(f"{team_key} {year}: {exc}")
                    continue
                with con:
                    observation_count += store_team_volume(
                        con, team_key, observations, fetched_at_utc=started
                    )
            log(f"Club attention: {year} stored, {observation_count} observations so far")
        with con:
            con.execute(
                """
                UPDATE context_attention_runs
                SET completed_at_utc = ?, status = ?, observation_count = ?,
                    error_count = ?, errors_json = ?
                WHERE run_id = ?
                """,
                (
                    utc_iso(utc_now()),
                    "completed_with_errors" if errors else "ok",
                    observation_count,
                    len(errors),
                    json.dumps(errors[:50]),
                    run_id,
                ),
            )
    return {
        "run_id": run_id,
        "source_key": GDELT_SOURCE_KEY,
        "start_year": first_year,
        "end_year": last_year,
        "teams": len(teams),
        "observations": observation_count,
        "errors": errors,
    }


def load_attention_series(con: sqlite3.Connection) -> pd.DataFrame:
    try:
        frame = pd.read_sql_query(
            """
            SELECT team_key, observed_date, article_count, corpus_norm
            FROM context_attention_series
            ORDER BY team_key, observed_date
            """,
            con,
        )
    except Exception:
        return pd.DataFrame(
            columns=["team_key", "observed_date", "article_count", "corpus_norm"]
        )
    return frame


def _weekly_rate(daily: pd.DataFrame) -> pd.Series:
    """Articles per million indexed documents, so GDELT's own growth cancels."""

    counts = pd.to_numeric(daily["article_count"], errors="coerce").fillna(0.0)
    norms = pd.to_numeric(daily["corpus_norm"], errors="coerce")
    scale = norms.where(norms > 0).fillna(norms[norms > 0].median() if (norms > 0).any() else 1.0)
    return (counts / scale.clip(lower=1.0)) * 1_000_000.0


class AttentionIndex:
    """Trailing-baseline attention lookups for one registry snapshot.

    Both the window and its baseline end strictly before the decision cutoff,
    so nothing published after the round was frozen can reach a feature row.
    """

    def __init__(self, series: pd.DataFrame):
        self._by_team: dict[str, pd.DataFrame] = {}
        if series is None or series.empty:
            return
        frame = series.copy()
        frame["observed_date"] = pd.to_datetime(
            frame["observed_date"], errors="coerce", utc=True
        )
        frame = frame.dropna(subset=["observed_date"])
        if frame.empty:
            return
        frame["rate"] = _weekly_rate(frame)
        for team_key, group in frame.groupby("team_key"):
            ordered = group.sort_values("observed_date").reset_index(drop=True)
            self._by_team[str(team_key)] = ordered

    @property
    def available(self) -> bool:
        return bool(self._by_team)

    def values(self, team_key: str, decision_at: dt.datetime) -> dict[str, float]:
        missing = {
            "attention_rate": 0.0,
            "attention_index": 0.0,
            "attention_z": 0.0,
            "attention_missing": 1.0,
        }
        frame = self._by_team.get(team_key)
        if frame is None or frame.empty:
            return missing
        cutoff = pd.Timestamp(decision_at).tz_convert("UTC")
        history = frame[frame["observed_date"] < cutoff]
        window_start = cutoff - pd.Timedelta(days=ATTENTION_WINDOW_DAYS)
        window = history[history["observed_date"] >= window_start]
        baseline_start = cutoff - pd.Timedelta(weeks=ATTENTION_BASELINE_WEEKS)
        baseline = history[
            (history["observed_date"] >= baseline_start)
            & (history["observed_date"] < window_start)
        ]
        if window.empty or len(baseline) < ATTENTION_MIN_BASELINE_WEEKS * 7:
            return missing

        current = float(window["rate"].mean())
        reference = float(baseline["rate"].median())
        spread = float((baseline["rate"] - reference).abs().median())
        index = current / reference if reference > 1e-9 else 0.0
        # 1.4826 rescales the median absolute deviation to a normal-equivalent
        # standard deviation, so the z stays comparable across clubs and eras.
        scale = spread * 1.4826
        z = (current - reference) / scale if scale > 1e-9 else 0.0
        return {
            "attention_rate": current,
            "attention_index": float(index),
            "attention_z": float(z),
            "attention_missing": 0.0,
        }


def load_attention_index(con: sqlite3.Connection) -> AttentionIndex:
    return AttentionIndex(load_attention_series(con))
