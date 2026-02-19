import os
import pathlib
import sys
import unittest
from unittest import mock

from pipeline import cli


class CLISmokeTests(unittest.TestCase):
    def test_run_command_respects_cwd(self):
        env = {"A": "1"}
        with mock.patch("pipeline.cli.subprocess.run") as run_mock:
            cli._run_command(["echo", "hello"], env, cwd=pathlib.Path("/tmp"))

        run_mock.assert_called_once_with(["echo", "hello"], check=True, env=env, cwd="/tmp")

    def test_run_data_prep_uses_absolute_path(self):
        root = pathlib.Path("/repo")
        env = {}
        with mock.patch("pipeline.cli._run_command") as run_mock:
            cli._run_data_prep(env, root)

        run_mock.assert_called_once_with(
            ["Rscript", "/repo/pipeline/data-prep.R"],
            env,
            cwd=root,
        )

    def test_run_train_uses_absolute_script_path(self):
        root = pathlib.Path("/repo")
        env = {}
        with mock.patch("pipeline.cli._run_data_prep") as prep_mock, mock.patch("pipeline.cli._run_command") as run_mock:
            cli._run_train(env, skip_prep=False, root=root)

        prep_mock.assert_called_once_with(env, root)
        run_mock.assert_called_once_with(
            [sys.executable, "/repo/pipeline/train.py"],
            env,
            cwd=root,
        )

    def test_run_inference_uses_absolute_script_path(self):
        root = pathlib.Path("/repo")
        env = {}
        with mock.patch("pipeline.cli._run_data_prep") as prep_mock, mock.patch("pipeline.cli._run_command") as run_mock:
            cli._run_inference(env, skip_prep=False, root=root)

        prep_mock.assert_called_once_with(env, root)
        run_mock.assert_called_once_with(
            [sys.executable, "/repo/pipeline/inference.py"],
            env,
            cwd=root,
        )

    def test_test_email_prefers_cli_value(self):
        with mock.patch.dict(os.environ, {"FOOTY_TIPPER_TEST_EMAIL": "from_env@example.com"}, clear=False):
            self.assertEqual(cli._resolve_test_email("from_cli@example.com"), "from_cli@example.com")

    def test_test_email_uses_env_when_cli_missing(self):
        with mock.patch.dict(os.environ, {"FOOTY_TIPPER_TEST_EMAIL": "from_env@example.com"}, clear=False):
            self.assertEqual(cli._resolve_test_email(None), "from_env@example.com")

    def test_test_email_falls_back_to_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(cli._resolve_test_email(None), cli.DEFAULT_TEST_EMAIL)

    def test_parser_defaults_test_email_to_none(self):
        parser = cli.build_parser()
        send_args = parser.parse_args(["send"])
        predict_args = parser.parse_args(["predict", "--skip-prep", "--skip-send"])

        self.assertIsNone(send_args.test_email)
        self.assertIsNone(predict_args.test_email)

    def test_main_send_reads_test_email_from_env(self):
        with mock.patch.dict(os.environ, {"FOOTY_TIPPER_TEST_EMAIL": "from_env@example.com"}, clear=False), \
             mock.patch("pipeline.cli._send_predictions", return_value=0) as send_mock, \
             mock.patch("pipeline.cli.load_dotenv"):
            rc = cli.main(["send", "--test", "--dry-run", "--without-openai"])

        self.assertEqual(rc, 0)
        self.assertEqual(send_mock.call_args.kwargs["test_email"], "from_env@example.com")


if __name__ == "__main__":
    unittest.main()
