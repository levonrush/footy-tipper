import inspect
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest import mock

from pipeline.ops import actions_runner, runtime_prediction


class ActionsRunnerTests(unittest.TestCase):
    def test_runner_and_runtime_share_the_same_exact_modes(self):
        self.assertEqual(actions_runner.PREDICT_MODES, ("test", "refresh", "live"))
        self.assertEqual(actions_runner.PREDICT_MODES, runtime_prediction.VALID_MODES)

    def test_predict_rejects_unknown_mode(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
            actions_runner.main(["predict", "--mode", "surprise-live"])
        self.assertEqual(raised.exception.code, 2)

    def test_unknown_machine_command_is_an_invocation_error(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
            actions_runner.main(["do-everything"])
        self.assertEqual(raised.exception.code, 2)

    def test_predict_routes_only_allowlisted_mode(self):
        with mock.patch.object(actions_runner, "_run_prediction", return_value=17) as run:
            result = actions_runner.main(["predict", "--mode", "refresh"])
        self.assertEqual(result, 17)
        run.assert_called_once_with("refresh", None)

    def test_live_prediction_passes_exact_confirmed_round(self):
        with mock.patch.object(
            actions_runner, "_run_prediction", return_value=0
        ) as run, mock.patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}):
            result = actions_runner.main(
                ["predict", "--mode", "live", "--confirmed-round", "22"]
            )
        self.assertEqual(result, 0)
        run.assert_called_once_with("live", 22)

    def test_machine_live_is_github_actions_only(self):
        with mock.patch.dict(os.environ, {}, clear=True), redirect_stderr(
            StringIO()
        ), self.assertRaises(SystemExit) as raised:
            actions_runner.main(
                ["predict", "--mode", "live", "--confirmed-round", "22"]
            )
        self.assertEqual(raised.exception.code, 2)

    def test_non_live_prediction_rejects_confirmed_round(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
            actions_runner.main(
                ["predict", "--mode", "refresh", "--confirmed-round", "22"]
            )
        self.assertEqual(raised.exception.code, 2)

    def test_confirmed_round_is_available_only_during_prediction(self):
        variable = "FOOTY_TIPPER_CONFIRMED_LIVE_ROUND"
        with mock.patch.object(runtime_prediction, "run", return_value=0) as run:
            with mock.patch.dict(os.environ, {variable: "old"}):
                result = actions_runner._run_prediction("live", 22)
                self.assertEqual(os.environ[variable], "old")
        self.assertEqual(result, 0)
        self.assertEqual(run.call_args.args, ("live",))

    def test_scheduled_live_masks_stale_confirmed_round(self):
        variable = "FOOTY_TIPPER_CONFIRMED_LIVE_ROUND"

        def inspect_environment(_mode):
            self.assertEqual(os.environ.get(variable), "")
            return 0

        with mock.patch.object(runtime_prediction, "run", side_effect=inspect_environment):
            with mock.patch.dict(os.environ, {variable: "old"}):
                result = actions_runner._run_prediction("live")
                self.assertEqual(os.environ[variable], "old")
        self.assertEqual(result, 0)

    def test_runtime_push_uses_runtime_only_api(self):
        push = mock.Mock(return_value=0)
        with mock.patch.object(
            actions_runner.state_sync, "push_runtime_state", push, create=True
        ):
            result = actions_runner.main(["runtime-push"])
        self.assertEqual(result, 0)
        push.assert_called_once_with(actions_runner._project_root())

    def test_runtime_pull_uses_active_model_runtime_api(self):
        pull = mock.Mock(return_value=0)
        with mock.patch.object(
            actions_runner.state_sync, "pull_runtime_state", pull, create=True
        ):
            result = actions_runner.main(["runtime-pull"])
        self.assertEqual(result, 0)
        pull.assert_called_once_with(actions_runner._project_root())

    def test_model_check_requires_explicit_release(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
            actions_runner.main(["model-check"])
        self.assertEqual(raised.exception.code, 2)

    def test_model_check_routes_release_id(self):
        check = mock.Mock(return_value=0)
        with mock.patch.object(actions_runner, "_required_state_api", return_value=check) as api:
            result = actions_runner.main(
                ["model-check", "--release", "2026-07-22T120000Z-deadbee"]
            )
        self.assertEqual(result, 0)
        api.assert_called_once_with("check_model_release")
        check.assert_called_once_with(
            actions_runner._project_root(), "2026-07-22T120000Z-deadbee"
        )

    def test_model_check_operation_is_download_and_validation_only(self):
        source = inspect.getsource(actions_runner.state_sync.check_model_release)
        self.assertIn("_download_release", source)
        self.assertIn('"inference.py"', source)
        for forbidden in (
            "upload_or_update(",
            "upload_create_only(",
            "publish_model_release(",
            "activate_model_release(",
            "push_runtime_state(",
            "pipeline/train.py",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_gate_emits_live_vocabulary(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "github-output"
            patches = (
                mock.patch.object(
                    actions_runner.state_sync, "drive_service", return_value=object()
                ),
                mock.patch.object(
                    actions_runner.state_sync, "_existing_state_folder", return_value="state"
                ),
                mock.patch.object(
                    actions_runner.state_sync, "find_file_id", return_value=None
                ),
                mock.patch.object(
                    actions_runner.state_sync,
                    "gate_decision",
                    return_value=("live", "target reached"),
                ),
                mock.patch.dict(os.environ, {"GITHUB_OUTPUT": str(output)}),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                result = actions_runner.main(["gate"])

            self.assertEqual(result, 0)
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                "mode=live\nreason=target reached\n",
            )

    def test_gate_rejects_old_send_vocabulary(self):
        with mock.patch.object(
            actions_runner.state_sync, "drive_service", return_value=object()
        ), mock.patch.object(
            actions_runner.state_sync, "_existing_state_folder", return_value="state"
        ), mock.patch.object(
            actions_runner.state_sync, "find_file_id", return_value=None
        ), mock.patch.object(
            actions_runner.state_sync,
            "gate_decision",
            return_value=("send", "obsolete vocabulary"),
        ):
            with self.assertRaisesRegex(RuntimeError, "unsupported mode"):
                actions_runner.main(["gate"])


class RuntimePredictionTests(unittest.TestCase):
    def _patched_pipeline(self, ensure_models=True, send_result=0):
        fake_root = Path("/tmp/footy-tipper-actions-test")
        return (
            mock.patch.object(runtime_prediction.pipeline_cli, "_project_root", return_value=fake_root),
            mock.patch.object(runtime_prediction.pipeline_cli, "load_dotenv"),
            mock.patch.object(runtime_prediction.pipeline_cli, "_build_env", return_value={}),
            mock.patch.object(runtime_prediction.pipeline_cli, "_run_lineups"),
            mock.patch.object(runtime_prediction.pipeline_cli, "_refresh_nrl_data"),
            mock.patch.object(
                runtime_prediction.pipeline_cli,
                "_ensure_models_for_prediction",
                return_value=ensure_models,
            ),
            mock.patch.object(runtime_prediction.pipeline_cli, "_run_inference"),
            mock.patch.object(runtime_prediction.pipeline_cli, "_log"),
            mock.patch.object(
                runtime_prediction.pipeline_cli,
                "_resolve_test_email",
                return_value="test@example.com",
            ),
            mock.patch.object(
                runtime_prediction.pipeline_cli,
                "_send_predictions",
                return_value=send_result,
            ),
        )

    def test_refresh_never_calls_send(self):
        patches = self._patched_pipeline()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5] as ensure, patches[6] as infer, patches[7], patches[8], patches[9] as send:
            result = runtime_prediction.run("refresh")

        self.assertEqual(result, 0)
        ensure.assert_called_once_with(
            mock.ANY,
            mock.ANY,
            auto_train=False,
            allow_lineup_bootstrap=False,
        )
        infer.assert_called_once()
        send.assert_not_called()

    def test_every_actions_prediction_mode_disables_auto_train(self):
        for mode in runtime_prediction.VALID_MODES:
            with self.subTest(mode=mode):
                patches = self._patched_pipeline()
                with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5] as ensure, patches[6], patches[7], patches[8], patches[9]:
                    result = runtime_prediction.run(mode)

                self.assertEqual(result, 0)
                ensure.assert_called_once_with(
                    mock.ANY,
                    mock.ANY,
                    auto_train=False,
                    allow_lineup_bootstrap=False,
                )

    def test_missing_models_fail_without_hosted_training(self):
        patches = self._patched_pipeline(ensure_models=False)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5] as ensure, patches[6] as infer, patches[7], patches[8], patches[9] as send:
            result = runtime_prediction.run("live")

        self.assertEqual(result, 1)
        ensure.assert_called_once_with(
            mock.ANY,
            mock.ANY,
            auto_train=False,
            allow_lineup_bootstrap=False,
        )
        infer.assert_not_called()
        send.assert_not_called()

    def test_test_send_cannot_upload_predictions_to_drive(self):
        patches = self._patched_pipeline(send_result=0)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9] as send:
            result = runtime_prediction.run("test")

        self.assertEqual(result, 0)
        send.assert_called_once_with(
            test_mode=True,
            test_email="test@example.com",
            skip_drive=True,
            use_llm=True,
            dry_run=False,
            force_resend=False,
        )

    def test_live_send_is_explicit(self):
        patches = self._patched_pipeline(send_result=0)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9] as send:
            result = runtime_prediction.run("live")

        self.assertEqual(result, 0)
        send.assert_called_once_with(
            test_mode=False,
            test_email="test@example.com",
            skip_drive=False,
            use_llm=True,
            dry_run=False,
            force_resend=False,
        )


if __name__ == "__main__":
    unittest.main()
