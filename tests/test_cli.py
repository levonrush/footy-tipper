import os
import pathlib
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd

from pipeline import cli
from pipeline.common.use_predictions import sending_functions as sf


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

    def test_ensure_models_runs_train_when_artifacts_missing(self):
        env = {}
        root = pathlib.Path("/repo")
        with mock.patch("pipeline.cli._model_artifacts_exist", side_effect=[False, True]), \
             mock.patch("pipeline.cli._bootstrap_lineups_for_training_if_needed") as bootstrap_mock, \
             mock.patch("pipeline.cli._run_train") as run_train_mock, \
             mock.patch("pipeline.cli._log"):
            ok = cli._ensure_models_for_prediction(env, root, auto_train=True)

        self.assertTrue(ok)
        bootstrap_mock.assert_called_once()
        bootstrap_env = bootstrap_mock.call_args.args[0]
        self.assertEqual(bootstrap_env["FOOTY_TIPPER_PREP_MODE"], "train")
        run_train_mock.assert_called_once()

    def test_ensure_models_returns_false_when_auto_train_disabled(self):
        env = {}
        root = pathlib.Path("/repo")
        with mock.patch("pipeline.cli._model_artifacts_exist", return_value=False), \
             mock.patch("pipeline.cli._run_train") as run_train_mock, \
             mock.patch("pipeline.cli._log"):
            ok = cli._ensure_models_for_prediction(env, root, auto_train=False)

        self.assertFalse(ok)
        run_train_mock.assert_not_called()

    def test_ensure_models_respects_disabled_lineup_bootstrap(self):
        env = {}
        root = pathlib.Path("/repo")
        with mock.patch("pipeline.cli._model_artifacts_exist", side_effect=[False, True]), \
             mock.patch("pipeline.cli._bootstrap_lineups_for_training_if_needed") as bootstrap_mock, \
             mock.patch("pipeline.cli._run_train") as run_train_mock, \
             mock.patch("pipeline.cli._log"):
            ok = cli._ensure_models_for_prediction(env, root, auto_train=True, allow_lineup_bootstrap=False)

        self.assertTrue(ok)
        bootstrap_mock.assert_not_called()
        run_train_mock.assert_called_once()

    def test_lineup_backfill_bootstrapped_detects_recorded_backfill_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            data_dir = root / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = data_dir / "footy-tipper-db.sqlite"

            with sqlite3.connect(str(db_path)) as con:
                con.executescript(
                    """
                    CREATE TABLE lineup_ingestion_runs (
                        run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        mode TEXT NOT NULL,
                        requested_start_year INTEGER,
                        requested_end_year INTEGER,
                        completed_at_utc TEXT NOT NULL,
                        status TEXT NOT NULL
                    );
                    """
                )
                con.execute(
                    """
                    INSERT INTO lineup_ingestion_runs (
                        mode, requested_start_year, requested_end_year, completed_at_utc, status
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    ("backfill", 2018, 2026, "2026-02-28T00:00:00+00:00", "ok"),
                )

            env = {"FOOTY_TIPPER_START_YEAR": "2018", "FOOTY_TIPPER_END_YEAR": "2026"}
            self.assertTrue(cli._lineup_backfill_bootstrapped(root, env))

    def test_bootstrap_lineups_for_training_runs_backfill_when_history_missing(self):
        env = {"FOOTY_TIPPER_START_YEAR": "2018", "FOOTY_TIPPER_END_YEAR": "2026"}
        root = pathlib.Path("/repo")

        with mock.patch("pipeline.cli._lineup_backfill_bootstrapped", return_value=False), \
             mock.patch("pipeline.cli._run_lineups") as run_lineups_mock, \
             mock.patch("pipeline.cli._log"):
            cli._bootstrap_lineups_for_training_if_needed(env, root)

        run_lineups_mock.assert_called_once()
        bootstrap_env = run_lineups_mock.call_args.args[0]
        self.assertEqual(bootstrap_env["FOOTY_TIPPER_LINEUPS_MODE"], "backfill")
        self.assertEqual(bootstrap_env["FOOTY_TIPPER_LINEUPS_MAX_ARTICLES"], "2000")

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
                use_llm=False,
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
             mock.patch("pipeline.common.use_predictions.sending_functions.prepare_email_delivery", return_value=mock.sentinel.prepared_delivery), \
             mock.patch("pipeline.common.use_predictions.sending_functions.send_emails", return_value=2), \
             mock.patch("pipeline.common.use_predictions.sending_functions.email_send_already_recorded", return_value=None), \
             mock.patch("pipeline.common.use_predictions.sending_functions.record_email_send", return_value=True), \
             mock.patch("pipeline.ops.delivery_state.get_delivery", return_value=None), \
             mock.patch(
                 "pipeline.ops.delivery_state.begin_delivery",
                 return_value={
                     "allowed": True,
                     "marker": {"attempt_id": "attempt-1", "status": "pending"},
                 },
             ), \
             mock.patch("pipeline.ops.delivery_state.mark_sent") as marker_mock, \
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
                use_llm=False,
                dry_run=False,
            )

        self.assertEqual(rc, 0)
        marker_mock.assert_called_once_with(
            pathlib.Path("/repo"),
            2026,
            1,
            "attempt-1",
            recipients_count=2,
        )
        persist_mock.assert_called_once()
        self.assertTrue(persist_mock.call_args.kwargs["allow_write"])

    def test_sent_drive_marker_reconciles_ledger_without_resending(self):
        predictions = pd.DataFrame(
            [{"competition_year": 2026, "round_id": 21, "game_id": 1}]
        )
        with mock.patch(
            "pipeline.cli._project_root", return_value=pathlib.Path("/repo")
        ), mock.patch("pipeline.cli.load_dotenv"), mock.patch(
            "pipeline.cli._log"
        ), mock.patch(
            "pipeline.common.use_predictions.sending_functions.get_predictions",
            return_value=predictions,
        ), mock.patch(
            "pipeline.common.use_predictions.sending_functions.email_send_already_recorded",
            return_value=None,
        ), mock.patch(
            "pipeline.ops.delivery_state.get_delivery",
            return_value={"status": "sent", "recipients_count": 9},
        ), mock.patch(
            "pipeline.common.use_predictions.sending_functions.record_email_send",
            return_value=True,
        ) as record, mock.patch(
            "pipeline.common.use_predictions.sending_functions.send_emails"
        ) as send:
            result = cli._send_predictions(
                test_mode=False,
                test_email=None,
                skip_drive=True,
                use_llm=False,
                dry_run=False,
            )

        self.assertEqual(result, 0)
        record.assert_called_once()
        send.assert_not_called()

    def test_confirmed_live_round_mismatch_refuses_before_any_production_state(self):
        predictions = pd.DataFrame(
            [{"competition_year": 2026, "round_id": 21, "game_id": 1}]
        )
        with mock.patch.dict(
            os.environ, {cli.CONFIRMED_LIVE_ROUND_ENV: "22"}, clear=False
        ), mock.patch(
            "pipeline.cli._project_root", return_value=pathlib.Path("/repo")
        ), mock.patch("pipeline.cli.load_dotenv"), mock.patch(
            "pipeline.cli._log"
        ) as log, mock.patch(
            "pipeline.common.use_predictions.sending_functions.get_predictions",
            return_value=predictions,
        ), mock.patch(
            "pipeline.common.use_predictions.sending_functions.email_send_already_recorded"
        ) as ledger, mock.patch(
            "pipeline.ops.delivery_state.get_delivery"
        ) as marker_read, mock.patch(
            "pipeline.ops.delivery_state.begin_delivery"
        ) as marker_claim, mock.patch(
            "pipeline.common.use_predictions.sending_functions.send_emails"
        ) as send:
            result = cli._send_predictions(
                test_mode=False,
                test_email=None,
                skip_drive=True,
                use_llm=False,
                dry_run=False,
            )

        self.assertEqual(result, 3)
        self.assertIn("confirmed round 22", log.call_args.args[0])
        self.assertIn("round 21", log.call_args.args[0])
        ledger.assert_not_called()
        marker_read.assert_not_called()
        marker_claim.assert_not_called()
        send.assert_not_called()

    def test_invalid_confirmed_live_round_refuses_before_delivery_checks(self):
        predictions = pd.DataFrame(
            [{"competition_year": 2026, "round_id": 21, "game_id": 1}]
        )
        with mock.patch.dict(
            os.environ, {cli.CONFIRMED_LIVE_ROUND_ENV: "not-a-round"}, clear=False
        ), mock.patch(
            "pipeline.cli._project_root", return_value=pathlib.Path("/repo")
        ), mock.patch("pipeline.cli.load_dotenv"), mock.patch(
            "pipeline.cli._log"
        ), mock.patch(
            "pipeline.common.use_predictions.sending_functions.get_predictions",
            return_value=predictions,
        ), mock.patch(
            "pipeline.common.use_predictions.sending_functions.email_send_already_recorded"
        ) as ledger, mock.patch("pipeline.ops.delivery_state.get_delivery") as marker_read:
            result = cli._send_predictions(
                test_mode=False,
                test_email=None,
                skip_drive=True,
                use_llm=False,
                dry_run=False,
            )

        self.assertEqual(result, 3)
        ledger.assert_not_called()
        marker_read.assert_not_called()

    def test_scheduled_live_treats_blank_confirmed_round_as_absent(self):
        predictions = pd.DataFrame(
            [{"competition_year": 2026, "round_id": 21, "game_id": 1}]
        )
        for blank_value in ("", "   "):
            with self.subTest(blank_value=repr(blank_value)), mock.patch.dict(
                os.environ,
                {cli.CONFIRMED_LIVE_ROUND_ENV: blank_value},
                clear=False,
            ), mock.patch(
                "pipeline.cli._project_root", return_value=pathlib.Path("/repo")
            ), mock.patch("pipeline.cli.load_dotenv"), mock.patch(
                "pipeline.cli._log"
            ), mock.patch(
                "pipeline.common.use_predictions.sending_functions.get_predictions",
                return_value=predictions,
            ), mock.patch(
                "pipeline.common.use_predictions.sending_functions.email_send_already_recorded",
                return_value={"sent_at_utc": "2026-07-22T00:00:00+00:00"},
            ) as ledger, mock.patch(
                "pipeline.ops.delivery_state.get_delivery"
            ) as marker_read:
                result = cli._send_predictions(
                    test_mode=False,
                    test_email=None,
                    skip_drive=True,
                    use_llm=False,
                    dry_run=False,
                )

            self.assertEqual(result, 0)
            ledger.assert_called_once()
            marker_read.assert_not_called()

    def test_render_failure_does_not_claim_an_uncertain_delivery_marker(self):
        predictions = pd.DataFrame(
            [{"competition_year": 2026, "round_id": 21, "game_id": 1}]
        )
        with mock.patch(
            "pipeline.cli._project_root", return_value=pathlib.Path("/repo")
        ), mock.patch("pipeline.cli.load_dotenv"), mock.patch(
            "pipeline.cli._log"
        ), mock.patch(
            "pipeline.common.use_predictions.sending_functions.get_predictions",
            return_value=predictions,
        ), mock.patch(
            "pipeline.common.use_predictions.sending_functions.email_send_already_recorded",
            return_value=None,
        ), mock.patch(
            "pipeline.ops.delivery_state.get_delivery", return_value=None
        ), mock.patch(
            "pipeline.common.use_predictions.sending_functions.get_comp_strategy_recommendation",
            side_effect=RuntimeError("render setup failed"),
        ), mock.patch("pipeline.ops.delivery_state.begin_delivery") as begin:
            with self.assertRaisesRegex(RuntimeError, "render setup failed"):
                cli._send_predictions(
                    test_mode=False,
                    test_email=None,
                    skip_drive=True,
                    use_llm=False,
                    dry_run=False,
                )

        begin.assert_not_called()

    def test_email_preparation_failure_does_not_claim_delivery_marker(self):
        predictions = pd.DataFrame(
            [{"competition_year": 2026, "round_id": 21, "game_id": 1}]
        )
        payload = {
            "subject": "Subject",
            "plain_text": "Body",
            "html_text": "<p>Body</p>",
            "inline_images": [],
        }

        with mock.patch(
            "pipeline.cli._project_root", return_value=pathlib.Path("/repo")
        ), mock.patch("pipeline.cli.load_dotenv"), mock.patch(
            "pipeline.cli._log"
        ) as log, mock.patch(
            "pipeline.common.use_predictions.sending_functions.get_predictions",
            return_value=predictions,
        ), mock.patch(
            "pipeline.common.use_predictions.sending_functions.email_send_already_recorded",
            return_value=None,
        ), mock.patch(
            "pipeline.ops.delivery_state.get_delivery", return_value=None
        ), mock.patch(
            "pipeline.common.use_predictions.sending_functions.get_comp_strategy_recommendation",
            return_value={"status": "off", "mode": "off", "tips_changed": 0},
        ), mock.patch(
            "pipeline.common.use_predictions.sending_functions.persist_comp_strategy_decision"
        ), mock.patch(
            "pipeline.common.use_predictions.sending_functions.get_tipper_picks",
            return_value=pd.DataFrame(),
        ), mock.patch(
            "pipeline.common.use_predictions.sending_functions.get_joker_round_recommendation",
            return_value={"headline": "HOLD"},
        ), mock.patch(
            "pipeline.common.use_predictions.sending_functions.get_season_scoreboard",
            return_value=None,
        ), mock.patch(
            "pipeline.common.use_predictions.sending_functions.scoreboard_summary_line",
            return_value=None,
        ), mock.patch(
            "pipeline.common.use_predictions.sending_functions.generate_reg_regan_email_payload",
            return_value=payload,
        ), mock.patch(
            "pipeline.common.use_predictions.sending_functions.prepare_email_delivery",
            side_effect=sf.EmailPreparationError("recipient list is empty"),
        ) as prepare, mock.patch(
            "pipeline.ops.delivery_state.begin_delivery"
        ) as begin, mock.patch(
            "pipeline.common.use_predictions.sending_functions.send_emails"
        ) as send:
            result = cli._send_predictions(
                test_mode=False,
                test_email=None,
                skip_drive=True,
                use_llm=False,
                dry_run=False,
            )

        self.assertEqual(result, 1)
        prepare.assert_called_once()
        begin.assert_not_called()
        send.assert_not_called()
        self.assertIn("no pending marker was created", log.call_args.args[0].lower())

    def test_ambiguous_smtp_result_leaves_pending_marker(self):
        predictions = pd.DataFrame(
            [{"competition_year": 2026, "round_id": 21, "game_id": 1}]
        )
        payload = {
            "subject": "Subject",
            "plain_text": "Body",
            "html_text": "<p>Body</p>",
            "inline_images": [],
        }

        with mock.patch(
            "pipeline.cli._project_root", return_value=pathlib.Path("/repo")
        ), mock.patch("pipeline.cli.load_dotenv"), mock.patch(
            "pipeline.cli._log"
        ), mock.patch(
            "pipeline.common.use_predictions.sending_functions.get_predictions",
            return_value=predictions,
        ), mock.patch(
            "pipeline.common.use_predictions.sending_functions.email_send_already_recorded",
            return_value=None,
        ), mock.patch(
            "pipeline.ops.delivery_state.get_delivery", return_value=None
        ), mock.patch(
            "pipeline.common.use_predictions.sending_functions.get_comp_strategy_recommendation",
            return_value={"status": "off", "mode": "off", "tips_changed": 0},
        ), mock.patch(
            "pipeline.common.use_predictions.sending_functions.persist_comp_strategy_decision"
        ), mock.patch(
            "pipeline.common.use_predictions.sending_functions.get_tipper_picks",
            return_value=pd.DataFrame(),
        ), mock.patch(
            "pipeline.common.use_predictions.sending_functions.get_joker_round_recommendation",
            return_value={"headline": "HOLD"},
        ), mock.patch(
            "pipeline.common.use_predictions.sending_functions.get_season_scoreboard",
            return_value=None,
        ), mock.patch(
            "pipeline.common.use_predictions.sending_functions.scoreboard_summary_line",
            return_value=None,
        ), mock.patch(
            "pipeline.common.use_predictions.sending_functions.generate_reg_regan_email_payload",
            return_value=payload,
        ), mock.patch(
            "pipeline.common.use_predictions.sending_functions.prepare_email_delivery",
            return_value=mock.sentinel.prepared_delivery,
        ), mock.patch(
            "pipeline.ops.delivery_state.begin_delivery",
            return_value={
                "allowed": True,
                "marker": {"attempt_id": "attempt-1", "status": "pending"},
            },
        ) as begin, mock.patch(
            "pipeline.common.use_predictions.sending_functions.send_emails",
            return_value=False,
        ), mock.patch(
            "pipeline.ops.delivery_state.mark_sent"
        ) as mark_sent, mock.patch(
            "pipeline.common.use_predictions.sending_functions.record_email_send"
        ) as record:
            result = cli._send_predictions(
                test_mode=False,
                test_email=None,
                skip_drive=True,
                use_llm=False,
                dry_run=False,
            )

        self.assertEqual(result, 1)
        begin.assert_called_once()
        mark_sent.assert_not_called()
        record.assert_not_called()


if __name__ == "__main__":
    unittest.main()
