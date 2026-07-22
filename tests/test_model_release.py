import json
import os
import signal
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import dill

from pipeline.ops import model_release, state_sync


class _PredictionModelStub:
    def predict(self, rows):
        return [0.0] * len(rows)


def _make_db(path):
    with sqlite3.connect(path) as con:
        con.execute(
            """
            CREATE TABLE footy_tipping_data (
                competition_year INTEGER,
                round_id INTEGER,
                game_state_name TEXT,
                start_time_utc REAL
            )
            """
        )
        con.execute(
            "INSERT INTO footy_tipping_data VALUES (2026, 1, 'Final', 1.0)"
        )


def _make_models(path):
    path.mkdir(parents=True)
    (path / "home_model.pkl").write_bytes(dill.dumps(_PredictionModelStub()))
    (path / "away_model.pkl").write_bytes(dill.dumps(_PredictionModelStub()))
    (path / "model_manifest.json").write_text(
        '{"predictors": ["round_id"]}', encoding="utf-8"
    )


class ModelReleaseTests(unittest.TestCase):
    def test_model_update_lock_rejects_a_second_local_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = model_release._acquire_update_lock(root)
            try:
                with self.assertRaises(model_release.ModelUpdateAlreadyRunning):
                    model_release._acquire_update_lock(root)
            finally:
                model_release._release_update_lock(first)

    def test_interrupted_logged_command_terminates_and_reaps_process_group(self):
        class FakeProcess:
            pid = 4321
            returncode = None

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                self.returncode = -signal.SIGTERM
                return self.returncode

            def terminate(self):
                self.returncode = -signal.SIGTERM

            def kill(self):
                self.returncode = -signal.SIGKILL

        fake = FakeProcess()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            model_release.subprocess, "Popen", return_value=fake
        ) as popen, mock.patch.object(
            model_release.time, "sleep", side_effect=KeyboardInterrupt
        ), mock.patch.object(model_release.os, "killpg") as killpg:
            root = Path(tmp)
            with self.assertRaises(KeyboardInterrupt):
                model_release._run_logged(
                    ["trainer"],
                    root=root,
                    env={},
                    log_path=root / "update.log",
                    label="Training candidate models",
                )

        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        killpg.assert_called_once_with(fake.pid, signal.SIGTERM)
        self.assertEqual(fake.returncode, -signal.SIGTERM)

    def test_default_search_budget_is_one_hundred(self):
        self.assertEqual(model_release.DEFAULT_TUNING_CANDIDATES, 100)

    def test_receipt_is_complete_and_detects_artifact_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = root / "models"
            db_path = root / "runtime.sqlite"
            _make_models(models)
            _make_db(db_path)
            receipt = model_release._build_receipt(
                models,
                db_path,
                release_id="release-1",
                git_sha="a" * 40,
                tuning_candidates=100,
                source="test",
            )
            receipt_path = model_release._write_receipt_last(models, receipt)

            self.assertTrue(receipt_path.is_file())
            self.assertEqual(
                state_sync._validate_release_directory(models, "release-1")[
                    "tuning_candidates"
                ],
                100,
            )
            (models / "home_model.pkl").write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                state_sync._validate_release_directory(models, "release-1")

    def test_interrupted_journal_is_resumed(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            model_release, "_git_sha", return_value="b" * 40
        ), mock.patch.object(
            model_release, "_release_id", return_value="release-1"
        ):
            root = Path(tmp)
            journal, resumed = model_release._new_or_resumable_journal(root, 100)
            self.assertFalse(resumed)
            journal["status"] = "interrupted"
            journal["stages"]["trained"] = {"completed": True}
            model_release._save_journal(root, journal)

            loaded, resumed = model_release._new_or_resumable_journal(root, 10)

            self.assertTrue(resumed)
            self.assertEqual(loaded["release_id"], "release-1")
            self.assertEqual(loaded["tuning_candidates"], 100)

    def test_resume_refuses_to_undo_an_explicit_production_rollback(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            state_sync,
            "get_model_pointer",
            return_value={"release_id": "release-previous"},
        ):
            root = Path(tmp)
            journal = {
                "schema_version": 1,
                "release_id": "release-candidate",
                "status": "failed",
                "stages": {"pointer_activated": {"completed": True}},
            }
            with self.assertRaises(model_release.ProductionCodeChanged):
                model_release._revalidate_resume_evidence(
                    root, journal, root / "models"
                )

    def test_advanced_activation_requires_hosted_check_and_allows_pointer_repair(self):
        with mock.patch.object(
            model_release, "_hosted_validate_for_activation", return_value=True
        ) as validate, mock.patch.object(
            state_sync,
            "activate_model_release",
            return_value={"release_id": "release-1"},
        ) as activate:
            result = model_release.activate_release("/repo", "release-1")

        self.assertEqual(result, 0)
        validate.assert_called_once_with("/repo", "release-1")
        activate.assert_called_once_with(
            "/repo", "release-1", repair_broken_pointer=True
        )

    def test_rollback_requires_hosted_check_of_previous_release(self):
        with mock.patch.object(
            state_sync,
            "get_model_pointer",
            return_value={
                "release_id": "release-2",
                "previous_release_id": "release-1",
            },
        ), mock.patch.object(
            model_release, "_hosted_validate_for_activation", return_value=True
        ) as validate, mock.patch.object(
            state_sync,
            "activate_model_release",
            return_value={"release_id": "release-1"},
        ) as activate:
            result = model_release.rollback("/repo")

        self.assertEqual(result, 0)
        validate.assert_called_once_with("/repo", "release-1")
        activate.assert_called_once_with("/repo", "release-1")

    def test_runtime_push_never_uploads_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            _make_db(root / "data" / "footy-tipper-db.sqlite")
            uploaded = []

            def capture(_service, _folder, name, _path, _mime):
                uploaded.append(name)

            with mock.patch.multiple(
                state_sync,
                drive_service=mock.Mock(return_value=object()),
                _state_folder=mock.Mock(return_value="state"),
                upload_or_update=mock.Mock(side_effect=capture),
            ):
                result = state_sync.push_runtime_state(root)

            self.assertEqual(result, 0)
            self.assertEqual(
                uploaded, [state_sync.DB_ARCHIVE, state_sync.SCHEDULE_FILE]
            )
            self.assertNotIn(state_sync.MODELS_ARCHIVE, uploaded)

    def test_malformed_active_pointer_is_rejected_without_legacy_fallback(self):
        def fake_download(_service, _file_id, destination):
            Path(destination).write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "release_id": "release-1",
                        "archive": "wrong.tar.gz",
                        "metadata": "release-1.json",
                        "archive_sha256": "a" * 64,
                    }
                ),
                encoding="utf-8",
            )

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            state_sync, "find_file_id", return_value="pointer-id"
        ), mock.patch.object(state_sync, "download_to", side_effect=fake_download):
            with self.assertRaisesRegex(ValueError, "archive does not match"):
                state_sync._load_remote_pointer(
                    object(), "state", Path(tmp) / state_sync.MODEL_POINTER_FILE
                )

    def test_release_archives_are_reproducible_for_safe_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = root / "models"
            _make_models(models)
            first = root / "first.tar.gz"
            second = root / "second.tar.gz"

            state_sync._write_models_archive(models, first)
            os.utime(models / "home_model.pkl", (999999999, 999999999))
            state_sync._write_models_archive(models, second)

            self.assertEqual(state_sync._sha256(first), state_sync._sha256(second))

    def test_rollback_reactivates_pointer_previous_release(self):
        with mock.patch.object(
            state_sync,
            "get_model_pointer",
            return_value={"release_id": "release-2", "previous_release_id": "release-1"},
        ), mock.patch.object(
            state_sync,
            "activate_model_release",
            return_value={"release_id": "release-1"},
        ) as activate:
            result = state_sync.rollback_model_release("/repo")

        self.assertEqual(result["release_id"], "release-1")
        activate.assert_called_once_with("/repo", "release-1")

    def test_legacy_rollout_wraps_validated_models_before_activation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_models(root / "models")
            (root / "data").mkdir()
            _make_db(root / "data" / "footy-tipper-db.sqlite")
            captured = {}

            def publish(_root, _models, receipt, release_id):
                captured["receipt"] = receipt
                captured["release_id"] = release_id
                return {"archive_sha256": "a" * 64}

            with mock.patch.object(
                model_release, "_git_sha", return_value="f" * 40
            ), mock.patch.object(
                model_release, "_release_id", return_value="rollout-1"
            ), mock.patch.object(
                state_sync, "publish_model_release", side_effect=publish
            ), mock.patch.object(
                model_release, "_container_check"
            ), mock.patch.object(
                state_sync,
                "activate_model_release",
                return_value={"release_id": "legacy-rollout-1"},
            ) as activate:
                result = model_release.import_legacy_models(root)

            self.assertEqual(result["release_id"], "legacy-rollout-1")
            self.assertEqual(captured["receipt"]["source"], "legacy_rollout_import")
            activate.assert_called_once_with(root.resolve(), "legacy-rollout-1")

    def test_material_remote_code_change_cancels_candidate(self):
        completed = [
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="remote-sha\n", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="pipeline/cli.py\n", stderr=""),
        ]
        with mock.patch.object(
            model_release, "_git_sha", return_value="local-sha"
        ), mock.patch.object(
            model_release.subprocess, "run", side_effect=completed
        ):
            with self.assertRaises(model_release.ProductionCodeChanged):
                model_release._verify_production_code(
                    Path("/repo"), {"git_sha": "local-sha"}
                )


if __name__ == "__main__":
    unittest.main()
