import sqlite3
import tempfile
import unittest
from pathlib import Path

from pipeline.ops import state_sync


HOUR = 3600
DAY = 86400


def _make_db(path, pre_game_rows, sent_rounds=()):
    con = sqlite3.connect(str(path))
    try:
        con.execute(
            """
            CREATE TABLE footy_tipping_data (
                game_id INTEGER,
                competition_year INTEGER,
                round_id INTEGER,
                game_state_name TEXT,
                start_time_utc REAL
            )
            """
        )
        for game_id, year, round_id, state, kickoff in pre_game_rows:
            con.execute(
                "INSERT INTO footy_tipping_data VALUES (?, ?, ?, ?, ?)",
                (game_id, year, round_id, state, kickoff),
            )
        if sent_rounds:
            con.execute(
                """
                CREATE TABLE email_sends (
                    competition_year INTEGER NOT NULL,
                    round_id INTEGER NOT NULL,
                    sent_at_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    recipients_count INTEGER,
                    source TEXT NOT NULL DEFAULT 'unknown',
                    PRIMARY KEY (competition_year, round_id)
                )
                """
            )
            for year, round_id in sent_rounds:
                con.execute(
                    "INSERT INTO email_sends (competition_year, round_id) VALUES (?, ?)",
                    (year, round_id),
                )
        con.commit()
    finally:
        con.close()


class ComputeScheduleTests(unittest.TestCase):
    def test_upcoming_rounds_with_sent_flags(self):
        now = 1_000_000
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "db.sqlite"
            _make_db(
                db,
                pre_game_rows=[
                    # Round 18: two games, first kickoff wins.
                    (1, 2026, 18, "Pre Game", now + 2 * HOUR),
                    (2, 2026, 18, "Pre Game", now + 26 * HOUR),
                    # Round 19 later.
                    (3, 2026, 19, "Pre Game", now + 7 * DAY),
                    # Finished game must be ignored.
                    (4, 2026, 17, "Final", now - 7 * DAY),
                    # Older year Pre Game must be ignored.
                    (5, 2025, 27, "Pre Game", now + HOUR),
                ],
                sent_rounds=[(2026, 18)],
            )
            schedule = state_sync.compute_schedule(db, now=now)

        self.assertEqual(schedule["competition_year"], 2026)
        self.assertEqual(
            schedule["upcoming_rounds"],
            [
                {"round_id": 18, "first_kickoff_utc": now + 2 * HOUR, "sent": True},
                {"round_id": 19, "first_kickoff_utc": now + 7 * DAY, "sent": False},
            ],
        )

    def test_offseason_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "db.sqlite"
            _make_db(db, pre_game_rows=[(1, 2026, 27, "Final", 500)])
            schedule = state_sync.compute_schedule(db, now=1_000)
        self.assertIsNone(schedule["competition_year"])
        self.assertEqual(schedule["upcoming_rounds"], [])

    def test_missing_ledger_table_means_unsent(self):
        now = 1_000_000
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "db.sqlite"
            _make_db(db, pre_game_rows=[(1, 2026, 18, "Pre Game", now + HOUR)])
            schedule = state_sync.compute_schedule(db, now=now)
        self.assertFalse(schedule["upcoming_rounds"][0]["sent"])


class GateDecisionTests(unittest.TestCase):
    def _schedule(self, rounds, generated_at=None, now=1_000_000):
        return {
            "generated_at_utc": now if generated_at is None else generated_at,
            "competition_year": 2026,
            "upcoming_rounds": rounds,
        }

    def test_missing_schedule_skips(self):
        mode, reason = state_sync.gate_decision(None, now=1_000_000)
        self.assertEqual(mode, "skip")
        self.assertIn("not seeded", reason)

    def test_too_early_skips(self):
        now = 1_000_000
        schedule = self._schedule(
            [{"round_id": 18, "first_kickoff_utc": now + 10 * HOUR, "sent": False}]
        )
        mode, reason = state_sync.gate_decision(schedule, now=now)
        self.assertEqual(mode, "skip")
        self.assertIn("too early", reason)

    def test_window_open_sends(self):
        now = 1_000_000
        schedule = self._schedule(
            [{"round_id": 18, "first_kickoff_utc": now + 5 * HOUR, "sent": False}]
        )
        mode, _ = state_sync.gate_decision(schedule, now=now)
        self.assertEqual(mode, "send")

    def test_grace_after_kickoff_still_sends(self):
        now = 1_000_000
        schedule = self._schedule(
            [{"round_id": 18, "first_kickoff_utc": now - 11 * HOUR, "sent": False}]
        )
        mode, _ = state_sync.gate_decision(schedule, now=now)
        self.assertEqual(mode, "send")

    def test_sent_round_is_skipped_next_round_too_early(self):
        now = 1_000_000
        schedule = self._schedule(
            [
                {"round_id": 18, "first_kickoff_utc": now + HOUR, "sent": True},
                {"round_id": 19, "first_kickoff_utc": now + 7 * DAY, "sent": False},
            ]
        )
        mode, reason = state_sync.gate_decision(schedule, now=now)
        self.assertEqual(mode, "skip")
        self.assertIn("19", reason)

    def test_expired_unsent_round_falls_through_to_next(self):
        now = 1_000_000
        schedule = self._schedule(
            [
                {"round_id": 18, "first_kickoff_utc": now - 2 * DAY, "sent": False},
                {"round_id": 19, "first_kickoff_utc": now + 3 * HOUR, "sent": False},
            ]
        )
        mode, reason = state_sync.gate_decision(schedule, now=now)
        self.assertEqual(mode, "send")
        self.assertIn("19", reason)

    def test_stale_schedule_refreshes(self):
        now = 1_000_000
        schedule = self._schedule([], generated_at=now - 10 * DAY, now=now)
        mode, _ = state_sync.gate_decision(schedule, now=now)
        self.assertEqual(mode, "refresh")

    def test_fresh_offseason_skips(self):
        now = 1_000_000
        schedule = self._schedule([], generated_at=now - DAY, now=now)
        mode, _ = state_sync.gate_decision(schedule, now=now)
        self.assertEqual(mode, "skip")


if __name__ == "__main__":
    unittest.main()
