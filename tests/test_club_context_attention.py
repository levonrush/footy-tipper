import datetime as dt
import json
import sqlite3
import unittest
from contextlib import closing

import pandas as pd

from pipeline.common.club_context.attention import (
    ATTENTION_BASELINE_WEEKS,
    GDELT_COVERAGE_START_YEAR,
    AttentionIndex,
    TEAM_QUERY_PHRASES,
    _query_expression,
    backfill_attention,
    fetch_team_volume,
    load_attention_index,
    store_team_volume,
)
from pipeline.common.club_context.schema import ensure_context_tables
from pipeline.common.lineups.normalization import normalize_team_name


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return self._payload

    def close(self):
        return None


def _timeline(points):
    return json.dumps({"timeline": [{"data": points}]}).encode("utf-8")


class QueryTests(unittest.TestCase):
    def test_every_phrase_disambiguates_the_club(self):
        # A bare nickname matches other codes and countries, so each phrase must
        # carry a city or full club name.
        for team_key, phrases in TEAM_QUERY_PHRASES.items():
            for phrase in phrases:
                self.assertGreaterEqual(
                    len(phrase.split()), 2, f"{team_key}: {phrase!r} is ambiguous"
                )

    def test_phrases_cover_the_normalized_team_vocabulary(self):
        for team_key in TEAM_QUERY_PHRASES:
            self.assertEqual(normalize_team_name(team_key), team_key)

    def test_multiple_phrases_are_combined_as_alternatives(self):
        self.assertEqual(_query_expression("broncos"), '"Brisbane Broncos"')
        self.assertIn(" OR ", _query_expression("sharks"))


class FetchTests(unittest.TestCase):
    def test_daily_points_are_parsed(self):
        payload = _timeline(
            [
                {"date": "20240301T000000Z", "value": 3, "norm": 100000},
                {"date": "20240302T000000Z", "value": 0, "norm": 90000},
            ]
        )
        observations = fetch_team_volume(
            "broncos",
            dt.date(2024, 3, 1),
            dt.date(2024, 3, 2),
            opener=lambda *_args, **_kwargs: _Response(payload),
        )
        self.assertEqual(
            [item["observed_date"] for item in observations],
            ["2024-03-01", "2024-03-02"],
        )
        self.assertEqual(observations[0]["article_count"], 3.0)
        self.assertEqual(observations[0]["corpus_norm"], 100000.0)

    def test_a_refusal_raises_rather_than_returning_zeroes(self):
        # GDELT answers a rate-limit with HTTP 200 and a plain-text scolding.
        # Storing that as "no coverage" would invent an absence of news.
        with self.assertRaises(ValueError):
            fetch_team_volume(
                "broncos",
                dt.date(2024, 3, 1),
                dt.date(2024, 3, 2),
                opener=lambda *_a, **_k: _Response(b"Please limit requests"),
            )


class BackfillTests(unittest.TestCase):
    def test_a_failing_club_does_not_stop_the_run(self):
        import tempfile
        from pathlib import Path

        def fetcher(team_key, start, end):
            if team_key == "eels":
                raise RuntimeError("refused")
            return [
                {"observed_date": "2024-03-01", "article_count": 2.0, "corpus_norm": 1000.0}
            ]

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "db.sqlite"
            result = backfill_attention(
                path,
                start_year=2024,
                end_year=2024,
                team_keys=["broncos", "eels"],
                fetcher=fetcher,
                sleep=lambda _seconds: None,
                rate_limit_seconds=0.0,
                log=lambda _message: None,
            )
            self.assertEqual(result["observations"], 1)
            self.assertEqual(len(result["errors"]), 1)
            with closing(sqlite3.connect(str(path))) as con:
                teams = [
                    row[0]
                    for row in con.execute(
                        "SELECT DISTINCT team_key FROM context_attention_series"
                    )
                ]
            self.assertEqual(teams, ["broncos"])

    def test_the_window_never_precedes_gdelt_coverage(self):
        import tempfile
        from pathlib import Path

        seen = []

        def fetcher(team_key, start, end):
            seen.append(start.year)
            return []

        with tempfile.TemporaryDirectory() as folder:
            backfill_attention(
                Path(folder) / "db.sqlite",
                start_year=2010,
                end_year=GDELT_COVERAGE_START_YEAR,
                team_keys=["broncos"],
                fetcher=fetcher,
                sleep=lambda _seconds: None,
                rate_limit_seconds=0.0,
                log=lambda _message: None,
            )
        self.assertTrue(all(year >= GDELT_COVERAGE_START_YEAR - 1 for year in seen))


class IndexTests(unittest.TestCase):
    @staticmethod
    def _series(counts_by_date):
        return pd.DataFrame(
            [
                {
                    "team_key": "broncos",
                    "observed_date": date,
                    "article_count": count,
                    "corpus_norm": 1000.0,
                }
                for date, count in counts_by_date.items()
            ]
        )

    def _flat_history(self, end, *, weeks, per_day):
        counts = {}
        day = end - dt.timedelta(days=weeks * 7)
        while day < end:
            counts[day.isoformat()] = per_day
            day += dt.timedelta(days=1)
        return counts

    def test_missing_coverage_is_flagged_rather_than_zeroed(self):
        index = AttentionIndex(pd.DataFrame())
        values = index.values("broncos", dt.datetime(2024, 6, 1, tzinfo=dt.timezone.utc))
        self.assertEqual(values["attention_missing"], 1.0)
        self.assertEqual(values["attention_index"], 0.0)
        self.assertFalse(index.available)

    def test_a_quiet_week_scores_below_its_own_baseline(self):
        end = dt.date(2024, 6, 1)
        counts = self._flat_history(end, weeks=ATTENTION_BASELINE_WEEKS, per_day=4.0)
        for offset in range(1, 8):
            counts[(end - dt.timedelta(days=offset)).isoformat()] = 1.0
        index = AttentionIndex(self._series(counts))
        values = index.values("broncos", dt.datetime(2024, 6, 1, tzinfo=dt.timezone.utc))
        self.assertEqual(values["attention_missing"], 0.0)
        self.assertLess(values["attention_index"], 1.0)

    def test_a_spike_scores_above_its_own_baseline(self):
        end = dt.date(2024, 6, 1)
        counts = self._flat_history(end, weeks=ATTENTION_BASELINE_WEEKS, per_day=4.0)
        for offset in range(1, 8):
            counts[(end - dt.timedelta(days=offset)).isoformat()] = 40.0
        index = AttentionIndex(self._series(counts))
        values = index.values("broncos", dt.datetime(2024, 6, 1, tzinfo=dt.timezone.utc))
        self.assertGreater(values["attention_index"], 1.0)

    def test_coverage_after_the_cutoff_cannot_reach_the_feature(self):
        end = dt.date(2024, 6, 1)
        counts = self._flat_history(end, weeks=ATTENTION_BASELINE_WEEKS, per_day=4.0)
        quiet = AttentionIndex(self._series(counts)).values(
            "broncos", dt.datetime(2024, 6, 1, tzinfo=dt.timezone.utc)
        )
        # A storm of coverage published after the decision cutoff.
        for offset in range(0, 10):
            counts[(end + dt.timedelta(days=offset)).isoformat()] = 400.0
        later = AttentionIndex(self._series(counts)).values(
            "broncos", dt.datetime(2024, 6, 1, tzinfo=dt.timezone.utc)
        )
        self.assertEqual(quiet, later)


class StorageTests(unittest.TestCase):
    def test_restoring_the_same_day_updates_rather_than_duplicates(self):
        with closing(sqlite3.connect(":memory:")) as con:
            ensure_context_tables(con)
            store_team_volume(
                con,
                "broncos",
                [{"observed_date": "2024-03-01", "article_count": 1.0, "corpus_norm": 10.0}],
            )
            store_team_volume(
                con,
                "broncos",
                [{"observed_date": "2024-03-01", "article_count": 5.0, "corpus_norm": 10.0}],
            )
            rows = con.execute(
                "SELECT article_count FROM context_attention_series"
            ).fetchall()
        self.assertEqual(rows, [(5.0,)])

    def test_an_absent_table_yields_an_unavailable_index(self):
        with closing(sqlite3.connect(":memory:")) as con:
            self.assertFalse(load_attention_index(con).available)


if __name__ == "__main__":
    unittest.main()
