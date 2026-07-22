import datetime as dt
import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

import dill

from pipeline.ops import state_sync


HOUR = 3600
DAY = 86400
SYDNEY = ZoneInfo("Australia/Sydney")


def _epoch(year, month, day, hour, minute=0, tz=dt.timezone.utc):
    return dt.datetime(year, month, day, hour, minute, tzinfo=tz).timestamp()


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


def _make_models_archive(path, files):
    source_dir = path.parent / f"{path.stem}-contents"
    source_dir.mkdir()
    with tarfile.open(path, "w:gz") as archive:
        for name, contents in files.items():
            source = source_dir / name
            source.write_bytes(contents)
            archive.add(source, arcname=name)


class _PredictionModelStub:
    def predict(self, rows):
        return [0.0] * len(rows)


def _valid_model_files(extra=None):
    files = {
        "home_model.pkl": dill.dumps(_PredictionModelStub()),
        "away_model.pkl": dill.dumps(_PredictionModelStub()),
        "model_manifest.json": b'{"predictors": ["round_id"]}',
    }
    files.update(extra or {})
    return files


def _valid_release_files(release_id="release-1", extra=None):
    files = _valid_model_files(extra)
    artifacts = {
        name: {"size": len(contents), "sha256": hashlib.sha256(contents).hexdigest()}
        for name, contents in files.items()
    }
    files[state_sync.TRAINING_RECEIPT_FILE] = json.dumps(
        {
            "schema_version": 1,
            "release_id": release_id,
            "artifacts": artifacts,
        }
    ).encode("utf-8")
    return files


def _mock_drive_downloads(downloads):
    db_archive = Path(downloads["db-id"])
    models_archive = Path(downloads["models-id"])
    archive_hash = hashlib.sha256(models_archive.read_bytes()).hexdigest()
    pointer = models_archive.parent / f"{models_archive.name}.pointer.json"
    pointer.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "release_id": "release-1",
                "archive": "release-1.tar.gz",
                "metadata": "release-1.json",
                "archive_sha256": archive_hash,
            }
        ),
        encoding="utf-8",
    )
    metadata = models_archive.parent / f"{models_archive.name}.metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "release_id": "release-1",
                "archive_sha256": archive_hash,
            }
        ),
        encoding="utf-8",
    )
    files = {
        "db-id": db_archive,
        "pointer-id": pointer,
        "archive-id": models_archive,
        "metadata-id": metadata,
    }

    def fake_download(_service, file_id, local_path):
        shutil.copyfile(files[file_id], local_path)

    def fake_find(_service, folder, name):
        return {
            ("state-folder", state_sync.DB_ARCHIVE): "db-id",
            ("state-folder", state_sync.MODEL_POINTER_FILE): "pointer-id",
            ("release-folder", "release-1.tar.gz"): "archive-id",
            ("release-folder", "release-1.json"): "metadata-id",
        }.get((folder, name))

    return mock.patch.multiple(
        state_sync,
        drive_service=mock.Mock(return_value=object()),
        _existing_state_folder=mock.Mock(return_value="state-folder"),
        _existing_release_folder=mock.Mock(return_value="release-folder"),
        find_file_id=mock.Mock(side_effect=fake_find),
        download_to=mock.Mock(side_effect=fake_download),
    )


class StatePublicationTests(unittest.TestCase):
    def test_runtime_push_does_not_require_or_upload_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            db_path = root / "data" / "footy-tipper-db.sqlite"
            _make_db(
                db_path,
                [(1, 2026, 21, "Pre Game", 1_800_000_000.0)],
            )
            uploaded_names = []

            with mock.patch.multiple(
                state_sync,
                drive_service=mock.Mock(return_value=object()),
                _state_folder=mock.Mock(return_value="state-folder"),
                upload_or_update=mock.Mock(
                    side_effect=lambda _service, _folder, name, _path, _mime: uploaded_names.append(name)
                ),
            ):
                result = state_sync.push_runtime_state(root)

        self.assertEqual(result, 0)
        self.assertEqual(
            uploaded_names,
            [state_sync.DB_ARCHIVE, state_sync.SCHEDULE_FILE],
        )
        self.assertNotIn(state_sync.MODELS_ARCHIVE, uploaded_names)

    def test_runtime_push_rejects_invalid_db_before_drive_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            (root / "data" / "footy-tipper-db.sqlite").touch()

            with mock.patch.object(state_sync, "drive_service") as drive:
                result = state_sync.push_runtime_state(root)

        self.assertEqual(result, 1)
        drive.assert_not_called()

    def test_pull_rejects_incomplete_archive_and_preserves_local_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = root / "models"
            data = root / "data"
            models.mkdir()
            data.mkdir()
            (models / "home_model.pkl").write_bytes(b"old-home")
            (models / "away_model.pkl").write_bytes(b"old-away")
            (models / "model_manifest.json").write_text('{"old": true}')
            (models / "old-only.pkl").write_bytes(b"keep-me")
            (data / "footy-tipper-db.sqlite").write_bytes(b"old-db")

            db_archive = root / "source-db.gz"
            with gzip.open(db_archive, "wb") as archive:
                archive.write(b"new-db")
            models_archive = root / "source-models.tar.gz"
            _make_models_archive(
                models_archive,
                {
                    "home_model.pkl": b"new-home",
                    "model_manifest.json": b"{}",
                },
            )

            with _mock_drive_downloads(
                {"db-id": db_archive, "models-id": models_archive}
            ), mock.patch.object(
                state_sync,
                "_extract_models_archive",
                wraps=state_sync._extract_models_archive,
            ) as extract_models:
                result = state_sync.pull_runtime_state(root)

            self.assertEqual(result, 1)
            extract_models.assert_called_once()
            self.assertEqual((models / "home_model.pkl").read_bytes(), b"old-home")
            self.assertEqual((models / "away_model.pkl").read_bytes(), b"old-away")
            self.assertEqual((models / "old-only.pkl").read_bytes(), b"keep-me")
            self.assertEqual((data / "footy-tipper-db.sqlite").read_bytes(), b"old-db")

    def test_pull_rejects_corrupt_archive_and_preserves_local_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = root / "models"
            models.mkdir()
            (models / "home_model.pkl").write_bytes(b"old-home")
            (models / "away_model.pkl").write_bytes(b"old-away")
            (models / "model_manifest.json").write_text("{}")

            db_archive = root / "source-db.gz"
            with gzip.open(db_archive, "wb") as archive:
                archive.write(b"new-db")
            models_archive = root / "source-models.tar.gz"
            models_archive.write_bytes(b"not a tar archive")

            with _mock_drive_downloads(
                {"db-id": db_archive, "models-id": models_archive}
            ), mock.patch.object(
                state_sync,
                "_extract_models_archive",
                wraps=state_sync._extract_models_archive,
            ) as extract_models:
                result = state_sync.pull_runtime_state(root)

            self.assertEqual(result, 1)
            extract_models.assert_called_once()
            self.assertEqual((models / "home_model.pkl").read_bytes(), b"old-home")
            self.assertEqual((models / "away_model.pkl").read_bytes(), b"old-away")

    def test_pull_rejects_invalid_artifact_contents_and_preserves_local_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = root / "models"
            data = root / "data"
            models.mkdir()
            data.mkdir()
            (models / "home_model.pkl").write_bytes(b"old-home")
            (models / "away_model.pkl").write_bytes(b"old-away")
            (models / "model_manifest.json").write_text('{"old": true}')
            (data / "footy-tipper-db.sqlite").write_bytes(b"old-db")

            db_archive = root / "source-db.gz"
            with gzip.open(db_archive, "wb") as archive:
                archive.write(b"new-db")
            models_archive = root / "source-models.tar.gz"
            _make_models_archive(
                models_archive,
                {
                    "home_model.pkl": b"not-a-model",
                    "away_model.pkl": b"not-a-model",
                    "model_manifest.json": b'{"predictors": ["round_id"]}',
                },
            )

            with _mock_drive_downloads(
                {"db-id": db_archive, "models-id": models_archive}
            ), mock.patch.object(
                state_sync,
                "_validate_model_artifacts",
                wraps=state_sync._validate_model_artifacts,
            ) as validate_models:
                result = state_sync.pull_runtime_state(root)

            self.assertEqual(result, 1)
            validate_models.assert_called_once()
            self.assertEqual((models / "home_model.pkl").read_bytes(), b"old-home")
            self.assertEqual((models / "away_model.pkl").read_bytes(), b"old-away")
            self.assertEqual((data / "footy-tipper-db.sqlite").read_bytes(), b"old-db")

    def test_pull_rejects_corrupt_db_and_preserves_local_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = root / "models"
            data = root / "data"
            models.mkdir()
            data.mkdir()
            (models / "home_model.pkl").write_bytes(b"old-home")
            (models / "away_model.pkl").write_bytes(b"old-away")
            (models / "model_manifest.json").write_text("{}")
            (data / "footy-tipper-db.sqlite").write_bytes(b"old-db")

            db_archive = root / "source-db.gz"
            db_archive.write_bytes(b"not a gzip archive")
            models_archive = root / "source-models.tar.gz"
            _make_models_archive(models_archive, _valid_release_files())

            with _mock_drive_downloads(
                {"db-id": db_archive, "models-id": models_archive}
            ), mock.patch.object(
                state_sync,
                "_extract_db_archive",
                wraps=state_sync._extract_db_archive,
            ) as extract_db:
                result = state_sync.pull_runtime_state(root)

            self.assertEqual(result, 1)
            extract_db.assert_called_once()
            self.assertEqual((models / "home_model.pkl").read_bytes(), b"old-home")
            self.assertEqual((models / "away_model.pkl").read_bytes(), b"old-away")
            self.assertEqual((data / "footy-tipper-db.sqlite").read_bytes(), b"old-db")

    def test_valid_pull_replaces_models_as_a_complete_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = root / "models"
            data = root / "data"
            models.mkdir()
            data.mkdir()
            (models / "old-only.pkl").write_bytes(b"stale")
            (data / "footy-tipper-db.sqlite").write_bytes(b"old-db")

            new_db = root / "new.sqlite"
            _make_db(new_db, [(1, 2026, 1, "Pre Game", 123456.0)])
            db_archive = root / "source-db.gz"
            with open(new_db, "rb") as source, gzip.open(db_archive, "wb") as archive:
                shutil.copyfileobj(source, archive)
            staged_models = root / "new-models"
            staged_models.mkdir()
            for name, contents in _valid_model_files(
                {"binary_model.pkl": dill.dumps(_PredictionModelStub())}
            ).items():
                (staged_models / name).write_bytes(contents)

            with mock.patch.multiple(
                state_sync,
                drive_service=mock.Mock(return_value=object()),
                _existing_state_folder=mock.Mock(return_value="state-folder"),
                find_file_id=mock.Mock(return_value="db-id"),
                _load_remote_pointer=mock.Mock(
                    return_value={
                        "release_id": "release-1",
                        "archive_sha256": "a" * 64,
                    }
                ),
                _download_release=mock.Mock(
                    return_value=(staged_models, root / "release.tar.gz", {}, {})
                ),
                download_to=mock.Mock(
                    side_effect=lambda _service, _file_id, destination: shutil.copyfile(
                        db_archive, destination
                    )
                ),
            ):
                result = state_sync.pull_runtime_state(root)

            self.assertEqual(result, 0)
            self.assertTrue((models / "home_model.pkl").is_file())
            self.assertTrue((models / "away_model.pkl").is_file())
            self.assertTrue((models / "binary_model.pkl").is_file())
            self.assertTrue((models / ".gitkeep").is_file())
            self.assertFalse((models / "old-only.pkl").exists())
            with sqlite3.connect(data / "footy-tipper-db.sqlite") as con:
                self.assertEqual(con.execute("SELECT COUNT(*) FROM footy_tipping_data").fetchone()[0], 1)

    def test_db_commit_failure_restores_last_known_good_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = root / "models"
            data = root / "data"
            models.mkdir()
            data.mkdir()
            (models / "home_model.pkl").write_bytes(b"old-home")
            (models / "away_model.pkl").write_bytes(b"old-away")
            (models / "model_manifest.json").write_text('{"old": true}')
            (data / "footy-tipper-db.sqlite").write_bytes(b"old-db")

            new_db = root / "new.sqlite"
            _make_db(new_db, [(1, 2026, 1, "Pre Game", 123456.0)])
            db_archive = root / "source-db.gz"
            with open(new_db, "rb") as source, gzip.open(db_archive, "wb") as archive:
                shutil.copyfileobj(source, archive)
            staged_models = root / "new-models"
            staged_models.mkdir()
            for name, contents in _valid_model_files().items():
                (staged_models / name).write_bytes(contents)

            real_replace = os.replace

            def fail_db_commit(source, destination):
                if Path(source).name == "db-staged.sqlite":
                    raise OSError("simulated DB commit failure")
                return real_replace(source, destination)

            with mock.patch.multiple(
                state_sync,
                drive_service=mock.Mock(return_value=object()),
                _existing_state_folder=mock.Mock(return_value="state-folder"),
                find_file_id=mock.Mock(return_value="db-id"),
                _load_remote_pointer=mock.Mock(
                    return_value={
                        "release_id": "release-1",
                        "archive_sha256": "a" * 64,
                    }
                ),
                _download_release=mock.Mock(
                    return_value=(staged_models, root / "release.tar.gz", {}, {})
                ),
                download_to=mock.Mock(
                    side_effect=lambda _service, _file_id, destination: shutil.copyfile(
                        db_archive, destination
                    )
                ),
            ), mock.patch.object(state_sync.os, "replace", side_effect=fail_db_commit):
                with self.assertRaisesRegex(OSError, "simulated DB commit failure"):
                    state_sync.pull_runtime_state(root)

            self.assertEqual((models / "home_model.pkl").read_bytes(), b"old-home")
            self.assertEqual((models / "away_model.pkl").read_bytes(), b"old-away")
            self.assertEqual((data / "footy-tipper-db.sqlite").read_bytes(), b"old-db")


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

    def test_aest_target_is_11am_sydney(self):
        kickoff = _epoch(2026, 7, 16, 19, 30, tz=SYDNEY)
        target = state_sync.sydney_send_target_utc(kickoff)
        self.assertEqual(target, _epoch(2026, 7, 16, 11, tz=SYDNEY))
        self.assertEqual(target, _epoch(2026, 7, 16, 1))

    def test_aedt_target_handles_utc_date_boundary(self):
        kickoff = _epoch(2026, 3, 5, 19, 30, tz=SYDNEY)
        target = state_sync.sydney_send_target_utc(kickoff)
        self.assertEqual(target, _epoch(2026, 3, 5, 11, tz=SYDNEY))
        self.assertEqual(target, _epoch(2026, 3, 5, 0))

    def test_before_11am_sydney_skips(self):
        kickoff = _epoch(2026, 7, 16, 19, 30, tz=SYDNEY)
        now = _epoch(2026, 7, 16, 10, 59, tz=SYDNEY)
        schedule = self._schedule(
            [{"round_id": 18, "first_kickoff_utc": kickoff, "sent": False}]
        )
        mode, reason = state_sync.gate_decision(schedule, now=now)
        self.assertEqual(mode, "skip")
        self.assertIn("too early", reason)

    def test_at_11am_sydney_sends(self):
        kickoff = _epoch(2026, 7, 16, 19, 30, tz=SYDNEY)
        now = _epoch(2026, 7, 16, 11, tz=SYDNEY)
        schedule = self._schedule(
            [{"round_id": 18, "first_kickoff_utc": kickoff, "sent": False}]
        )
        mode, _ = state_sync.gate_decision(schedule, now=now)
        self.assertEqual(mode, "live")

    def test_grace_after_kickoff_still_sends(self):
        kickoff = _epoch(2026, 7, 16, 19, 30, tz=SYDNEY)
        now = kickoff + 11 * HOUR
        schedule = self._schedule(
            [{"round_id": 18, "first_kickoff_utc": kickoff, "sent": False}]
        )
        mode, _ = state_sync.gate_decision(schedule, now=now)
        self.assertEqual(mode, "live")

    def test_sent_round_is_skipped_next_round_too_early(self):
        now = _epoch(2026, 7, 16, 12, tz=SYDNEY)
        schedule = self._schedule(
            [
                {"round_id": 18, "first_kickoff_utc": now + 7 * HOUR, "sent": True},
                {"round_id": 19, "first_kickoff_utc": now + 7 * DAY + 7 * HOUR, "sent": False},
            ]
        )
        mode, reason = state_sync.gate_decision(schedule, now=now)
        self.assertEqual(mode, "skip")
        self.assertIn("19", reason)

    def test_expired_unsent_round_falls_through_to_next(self):
        now = _epoch(2026, 7, 23, 12, tz=SYDNEY)
        schedule = self._schedule(
            [
                {"round_id": 18, "first_kickoff_utc": now - 7 * DAY + 7 * HOUR, "sent": False},
                {"round_id": 19, "first_kickoff_utc": now + 7 * HOUR, "sent": False},
            ]
        )
        mode, reason = state_sync.gate_decision(schedule, now=now)
        self.assertEqual(mode, "live")
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
