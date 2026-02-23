import os
import pathlib
import subprocess
import sys
import unittest
from unittest import mock

import pandas as pd

from pipeline import cli


class CLISmokeTests(unittest.TestCase):
    def test_run_command_respects_cwd(self):
        env = {"A": "1"}
        proc = mock.Mock()
        proc.stdout = iter(["hello\n"])
        proc.wait.return_value = 0
        with mock.patch("pipeline.cli.subprocess.Popen", return_value=proc) as popen_mock, \
             mock.patch("pipeline.cli._log"):
            cli._run_command(["echo", "hello"], env, cwd=pathlib.Path("/tmp"))

        popen_mock.assert_called_once_with(
            ["echo", "hello"],
            env=env,
            cwd="/tmp",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        proc.wait.assert_called_once()

    def test_run_command_raises_on_nonzero_exit(self):
        env = {"A": "1"}
        proc = mock.Mock()
        proc.stdout = iter([])
        proc.wait.return_value = 2
        with mock.patch("pipeline.cli.subprocess.Popen", return_value=proc), \
             mock.patch("pipeline.cli._log"):
            with self.assertRaises(subprocess.CalledProcessError):
                cli._run_command(["echo", "hello"], env, cwd=pathlib.Path("/tmp"))

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

    def test_send_predictions_test_mode_never_writes_joker_usage(self):
        predictions = pd.DataFrame(
            [
                {
                    "game_id": 1,
                    "round_id": 1,
                    "competition_year": 2026,
                    "round_name": "Round 1",
                    "team_home": "Alpha",
                    "team_away": "Bravo",
                    "home_team_result": "Win",
                    "home_team_win_prob": 0.55,
                    "home_team_lose_prob": 0.45,
                    "team_head_to_head_odds_home": 1.8,
                    "team_head_to_head_odds_away": 2.0,
                }
            ]
        )
        recommendation = {
            "headline": "PLAY JOKER THIS ROUND",
            "detail": "Round 1 is ranked #1.",
            "should_use_this_round": True,
            "competition_year": 2026,
            "current_round_id": 1,
            "current_round_name": "Round 1",
        }
        payload = {
            "subject": "Subject",
            "plain_text": "Body",
            "html_text": "<p>Body</p>",
            "inline_images": [],
        }

        with mock.patch("pipeline.cli._project_root", return_value=pathlib.Path("/repo")), \
             mock.patch("pipeline.cli.load_dotenv"), \
             mock.patch("pipeline.cli._log"), \
             mock.patch("pipeline.common.use_predictions.sending_functions.get_predictions", return_value=predictions), \
             mock.patch("pipeline.common.use_predictions.sending_functions.get_tipper_picks", return_value=pd.DataFrame()), \
             mock.patch("pipeline.common.use_predictions.sending_functions.get_joker_round_recommendation", return_value=recommendation), \
             mock.patch("pipeline.common.use_predictions.sending_functions.generate_reg_regan_email_payload", return_value=payload), \
             mock.patch("pipeline.common.use_predictions.sending_functions.send_test_email", return_value=True), \
             mock.patch("pipeline.common.use_predictions.sending_functions.persist_joker_usage_if_applicable") as persist_mock:
            rc = cli._send_predictions(
                test_mode=True,
                test_email="test@example.com",
                skip_drive=True,
                use_openai=False,
                dry_run=False,
            )

        self.assertEqual(rc, 0)
        persist_mock.assert_not_called()

    def test_send_predictions_prod_writes_joker_usage_after_successful_send(self):
        predictions = pd.DataFrame(
            [
                {
                    "game_id": 1,
                    "round_id": 1,
                    "competition_year": 2026,
                    "round_name": "Round 1",
                    "team_home": "Alpha",
                    "team_away": "Bravo",
                    "home_team_result": "Win",
                    "home_team_win_prob": 0.55,
                    "home_team_lose_prob": 0.45,
                    "team_head_to_head_odds_home": 1.8,
                    "team_head_to_head_odds_away": 2.0,
                }
            ]
        )
        recommendation = {
            "headline": "PLAY JOKER THIS ROUND",
            "detail": "Round 1 is ranked #1.",
            "should_use_this_round": True,
            "competition_year": 2026,
            "current_round_id": 1,
            "current_round_name": "Round 1",
        }
        payload = {
            "subject": "Subject",
            "plain_text": "Body",
            "html_text": "<p>Body</p>",
            "inline_images": [],
        }

        with mock.patch("pipeline.cli._project_root", return_value=pathlib.Path("/repo")), \
             mock.patch("pipeline.cli.load_dotenv"), \
             mock.patch("pipeline.cli._log"), \
             mock.patch("pipeline.common.use_predictions.sending_functions.get_predictions", return_value=predictions), \
             mock.patch("pipeline.common.use_predictions.sending_functions.get_tipper_picks", return_value=pd.DataFrame()), \
             mock.patch("pipeline.common.use_predictions.sending_functions.get_joker_round_recommendation", return_value=recommendation), \
             mock.patch("pipeline.common.use_predictions.sending_functions.generate_reg_regan_email_payload", return_value=payload), \
             mock.patch("pipeline.common.use_predictions.sending_functions.send_emails", return_value=True), \
             mock.patch(
                 "pipeline.common.use_predictions.sending_functions.persist_joker_usage_if_applicable",
                 return_value={
                     "recorded": True,
                     "reason": "recorded",
                     "competition_year": 2026,
                     "round_id": 1,
                 },
             ) as persist_mock:
            rc = cli._send_predictions(
                test_mode=False,
                test_email=None,
                skip_drive=True,
                use_openai=False,
                dry_run=False,
            )

        self.assertEqual(rc, 0)
        persist_mock.assert_called_once()
        self.assertTrue(persist_mock.call_args.kwargs["allow_write"])


if __name__ == "__main__":
    unittest.main()
