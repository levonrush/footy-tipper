import contextlib
import io
import inspect
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import pandas as pd

from pipeline import cli
from pipeline import cli_workflows
from pipeline import operator_cli


class OperatorCLITests(unittest.TestCase):
    def test_cli_loads_ignored_secrets_file_for_every_command(self):
        root = pathlib.Path("/repo")
        with mock.patch("pipeline.operator_cli.load_dotenv") as load, \
             mock.patch("pipeline.operator_cli.sys.stdin.isatty", return_value=False), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = operator_cli.run([], root=root)

        self.assertEqual(rc, 0)
        load.assert_called_once_with(dotenv_path=root / "secrets.env")

    def test_required_workflows_include_prediction_and_hosted_model_check(self):
        completed = mock.Mock(
            returncode=0,
            stdout=json.dumps(
                [
                    {
                        "name": "Predict",
                        "state": "active",
                        "path": ".github/workflows/predict.yml",
                        "id": 1,
                    },
                    {
                        "name": "Model release check",
                        "state": "active",
                        "path": ".github/workflows/model-check.yml",
                        "id": 2,
                    },
                ]
            ),
            stderr="",
        )
        with mock.patch("pipeline.operator_cli.shutil.which", return_value="/usr/bin/gh"), \
             mock.patch("pipeline.operator_cli._quiet_run", return_value=completed) as run_mock:
            status = operator_cli._github_workflow_readiness(pathlib.Path("/repo"))

        self.assertTrue(status["ready"])
        self.assertTrue(status["workflows"]["predict"]["ready"])
        self.assertTrue(status["workflows"]["model_check"]["ready"])
        run_mock.assert_called_once_with(
            ["gh", "workflow", "list", "--all", "--json", "name,state,path,id"],
            pathlib.Path("/repo"),
        )

    def test_disabled_model_check_makes_workflow_readiness_false(self):
        completed = mock.Mock(
            returncode=0,
            stdout=json.dumps(
                [
                    {
                        "name": "Predict",
                        "state": "active",
                        "path": ".github/workflows/predict.yml",
                    },
                    {
                        "name": "Model release check",
                        "state": "disabled_manually",
                        "path": ".github/workflows/model-check.yml",
                    },
                ]
            ),
            stderr="",
        )
        with mock.patch("pipeline.operator_cli.shutil.which", return_value="/usr/bin/gh"), \
             mock.patch("pipeline.operator_cli._quiet_run", return_value=completed):
            status = operator_cli._github_workflow_readiness(pathlib.Path("/repo"))

        self.assertFalse(status["ready"])
        self.assertTrue(status["workflows"]["predict"]["ready"])
        self.assertFalse(status["workflows"]["model_check"]["ready"])

    def test_setup_requires_hosted_workflows_and_never_local_docker(self):
        workflow_status = {
            "available": True,
            "ready": True,
            "workflows": {
                "predict": {"ready": True, "state": "active"},
                "model_check": {"ready": True, "state": "active"},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "secrets.env").touch()
            (root / "service-account-token.json").touch()
            output = io.StringIO()
            with mock.patch("pipeline.operator_cli.shutil.which", return_value="/usr/bin/tool"), \
                 mock.patch(
                     "pipeline.operator_cli._quiet_run",
                     return_value=mock.Mock(returncode=0, stdout="", stderr=""),
                 ), \
                 mock.patch(
                     "pipeline.operator_cli._github_workflow_readiness",
                     return_value=workflow_status,
                 ), \
                 mock.patch("pipeline.operator_cli.importlib.util.find_spec", return_value=object()), \
                 mock.patch("pipeline.operator_cli.shutil.disk_usage", return_value=mock.Mock(free=10 * 1024 ** 3)), \
                 mock.patch.dict("pipeline.operator_cli.os.environ", {"CONDA_DEFAULT_ENV": "footy-tipper"}), \
                 mock.patch("pipeline.ops.state_sync.drive_service", return_value=object()), \
                 mock.patch("pipeline.ops.state_sync._existing_state_folder", return_value="state-id"), \
                 mock.patch("pipeline.ops.state_sync.get_model_pointer", return_value={"release_id": "r1"}), \
                 contextlib.redirect_stdout(output):
                rc = operator_cli.command_setup(mock.Mock(), root=root)

        self.assertEqual(rc, 0)
        self.assertIn("Prediction automation workflow", output.getvalue())
        self.assertIn("Hosted model-validation workflow", output.getvalue())
        self.assertNotIn("Docker", output.getvalue())
        self.assertNotIn("docker", inspect.getsource(operator_cli.command_setup).lower())
        self.assertNotIn("gh auth login", inspect.getsource(operator_cli.command_setup))

    def test_status_prints_hosted_model_validation_state(self):
        status = {
            "git": {"available": False},
            "local_database": {"present": True},
            "local_models": {"ready": True, "missing": [], "release_id": None},
            "model_update": None,
            "configuration": {"secrets_env": True, "service_account": True},
            "github_workflows": {
                "available": True,
                "ready": True,
                "workflows": {
                    "predict": {"ready": True, "state": "active"},
                    "model_check": {"ready": True, "state": "active"},
                },
            },
        }
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            operator_cli._print_status(status)
        self.assertIn("Setup: ready", output.getvalue())
        self.assertIn("Hosted model validation: active", output.getvalue())

    def test_offline_status_never_claims_online_setup_is_ready(self):
        status = {
            "offline": True,
            "git": {"available": False},
            "local_database": {"present": True},
            "local_models": {"ready": True, "missing": [], "release_id": None},
            "model_update": None,
            "configuration": {"secrets_env": True, "service_account": True},
        }
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            operator_cli._print_status(status)

        self.assertIn("local files ready; online checks skipped", output.getvalue())
        self.assertNotIn("Setup: ready", output.getvalue())

    def test_parser_exposes_only_small_top_level_surface(self):
        parser = cli.build_parser()
        action = next(
            item
            for item in parser._actions
            if getattr(item, "choices", None)
        )
        self.assertEqual(
            set(action.choices),
            {"status", "setup", "tips", "update-model", "advanced"},
        )

    def test_entire_advanced_tree_parses(self):
        commands = [
            ["advanced", "data", "prepare", scope]
            for scope in ("all", "training", "tips")
        ]
        commands += [
            ["advanced", "data", "lineups", action]
            for action in ("refresh", "backfill")
        ]
        commands += [
            ["advanced", "data", "nrl", action]
            for action in ("refresh", "backfill", "validate")
        ]
        commands += [
            ["advanced", "data", "odds", action]
            for action in ("refresh", "backfill")
        ]
        commands += [
            ["advanced", "model", action]
            for action in ("train", "infer", "evaluate", "verify", "list", "rollback")
        ]
        commands.append(["advanced", "model", "activate", "release-123"])
        commands += [
            ["advanced", family, action]
            for family in ("local-run", "delivery")
            for action in ("preview", "test", "live")
        ]
        commands += [
            ["advanced", "cloud", action]
            for action in ("pull-runtime", "push-runtime", "schedule", "gate")
        ]
        commands += [
            ["advanced", "site", action]
            for action in ("build", "publish")
        ]
        parser = cli.build_parser()
        for command in commands:
            with self.subTest(command=command):
                parsed = parser.parse_args(command)
                self.assertEqual(parsed.command, "advanced")

    def test_every_retired_top_level_command_is_rejected_with_replacement(self):
        for command, replacement in operator_cli.RETIRED_COMMANDS.items():
            with self.subTest(command=command), contextlib.redirect_stderr(io.StringIO()) as stderr:
                rc = cli.main([command])
                self.assertEqual(rc, operator_cli.EXIT_INVOCATION)
                self.assertIn(replacement, stderr.getvalue())

    def test_retired_state_actions_get_specific_replacements(self):
        for action, replacement in operator_cli.STATE_ACTION_REPLACEMENTS.items():
            with self.subTest(action=action), contextlib.redirect_stderr(io.StringIO()) as stderr:
                rc = cli.main(["state", action])
                self.assertEqual(rc, operator_cli.EXIT_INVOCATION)
                self.assertIn(replacement, stderr.getvalue())

    def test_root_non_tty_prints_help_and_does_nothing(self):
        output = io.StringIO()
        with mock.patch("pipeline.operator_cli.sys.stdin.isatty", return_value=False), \
             mock.patch("pipeline.operator_cli.collect_status") as status_mock, \
             contextlib.redirect_stdout(output):
            rc = cli.main([])
        self.assertEqual(rc, 0)
        self.assertIn("NRL tips without", output.getvalue())
        status_mock.assert_not_called()

    def test_root_tty_menu_routes_selected_safe_action(self):
        status = {
            "git": {"available": False},
            "local_database": {"present": False},
            "local_models": {"ready": False, "missing": [], "release_id": None},
            "configuration": {"secrets_env": False, "service_account": False},
        }
        with contextlib.redirect_stdout(io.StringIO()), \
             mock.patch("pipeline.operator_cli.sys.stdin.isatty", return_value=True), \
             mock.patch("pipeline.operator_cli.sys.stdout.isatty", return_value=True), \
             mock.patch("pipeline.operator_cli.collect_status", return_value=status), \
             mock.patch("builtins.input", return_value="2"), \
             mock.patch("pipeline.operator_cli.run", return_value=0) as run_mock:
            rc = operator_cli._guided_menu(cli.build_parser(), root=pathlib.Path("/repo"), debug=False)
        self.assertEqual(rc, 0)
        run_mock.assert_called_once_with(
            ["tips", "test"], root=pathlib.Path("/repo"), inherited_debug=False
        )

    def test_tips_test_and_refresh_dispatch_exact_modes(self):
        for mode in ("test", "refresh"):
            with self.subTest(mode=mode), \
                 mock.patch("pipeline.operator_cli.cli_workflows.dispatch_and_wait", return_value={"conclusion": "success"}) as dispatch, \
                 contextlib.redirect_stdout(io.StringIO()):
                rc = operator_cli.run(["tips", mode], root=pathlib.Path("/repo"))
                self.assertEqual(rc, 0)
                dispatch.assert_called_once_with(mode, root=pathlib.Path("/repo"))

    def test_tips_live_requires_exact_round_confirmation(self):
        predictions = pd.DataFrame([{"competition_year": 2026, "round_id": 22}])
        with mock.patch("pipeline.operator_cli._published_predictions", return_value=predictions), \
             mock.patch("pipeline.operator_cli._ensure_delivery_clear"), \
             mock.patch("pipeline.operator_cli.sys.stdin.isatty", return_value=True), \
             mock.patch("builtins.input", return_value="SEND ROUND 22"), \
             mock.patch("pipeline.operator_cli.cli_workflows.dispatch_and_wait", return_value={"conclusion": "success"}) as dispatch, \
             contextlib.redirect_stdout(io.StringIO()):
            rc = operator_cli.run(["tips", "live"], root=pathlib.Path("/repo"))
        self.assertEqual(rc, 0)
        dispatch.assert_called_once_with(
            "live", root=pathlib.Path("/repo"), confirmed_round=22
        )

    def test_tips_live_refuses_mismatch_without_dispatch(self):
        predictions = pd.DataFrame([{"competition_year": 2026, "round_id": 22}])
        with mock.patch("pipeline.operator_cli._published_predictions", return_value=predictions), \
             mock.patch("pipeline.operator_cli._ensure_delivery_clear"), \
             mock.patch("pipeline.operator_cli.sys.stdin.isatty", return_value=True), \
             mock.patch("builtins.input", return_value="yes"), \
             mock.patch("pipeline.operator_cli.cli_workflows.dispatch_and_wait") as dispatch, \
             contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            rc = operator_cli.run(["tips", "live"], root=pathlib.Path("/repo"))
        self.assertEqual(rc, operator_cli.EXIT_SAFETY)
        dispatch.assert_not_called()

    def test_tips_live_has_no_noninteractive_bypass(self):
        predictions = pd.DataFrame([{"competition_year": 2026, "round_id": 22}])
        with mock.patch("pipeline.operator_cli._published_predictions", return_value=predictions), \
             mock.patch("pipeline.operator_cli._ensure_delivery_clear"), \
             mock.patch("pipeline.operator_cli.sys.stdin.isatty", return_value=False), \
             mock.patch("pipeline.operator_cli.cli_workflows.dispatch_and_wait") as dispatch, \
             contextlib.redirect_stderr(io.StringIO()):
            rc = operator_cli.run(["tips", "live"], root=pathlib.Path("/repo"))
        self.assertEqual(rc, operator_cli.EXIT_SAFETY)
        dispatch.assert_not_called()

    def test_status_json_is_schema_versioned(self):
        status = {"answer": 42}
        stdout = io.StringIO()
        with mock.patch("pipeline.operator_cli.collect_status", return_value=status), \
             contextlib.redirect_stdout(stdout):
            rc = operator_cli.run(["status", "--offline", "--json"], root=pathlib.Path("/repo"))
        self.assertEqual(rc, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema_version"], "1.0")
        self.assertEqual(payload["command"], "status")
        self.assertEqual(payload["status"], status)

    def test_tips_show_reads_published_copy_and_emits_json(self):
        predictions = pd.DataFrame(
            [{"competition_year": 2026, "round_id": 22, "team_home": "A", "team_away": "B"}]
        )
        stdout = io.StringIO()
        with mock.patch("pipeline.operator_cli._published_predictions", return_value=predictions), \
             contextlib.redirect_stdout(stdout):
            rc = operator_cli.run(["tips", "show", "--json"], root=pathlib.Path("/repo"))
        self.assertEqual(rc, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["tips"][0]["round_id"], 22)

    def test_model_infer_does_not_auto_train_by_default(self):
        with mock.patch("pipeline.cli._run_lineups"), \
             mock.patch("pipeline.cli._refresh_nrl_data"), \
             mock.patch("pipeline.cli._ensure_models_for_prediction", return_value=False) as ensure:
            rc = operator_cli.run(
                ["advanced", "model", "infer", "--skip-prepare"],
                root=pathlib.Path("/repo"),
            )
        self.assertEqual(rc, 1)
        self.assertFalse(ensure.call_args.kwargs["auto_train"])

    def test_advanced_cloud_calls_machine_interface_not_legacy_state_push(self):
        with mock.patch("pipeline.ops.actions_runner.main", return_value=0) as runner:
            rc = operator_cli.run(
                ["advanced", "cloud", "push-runtime"], root=pathlib.Path("/repo")
            )
        self.assertEqual(rc, 0)
        runner.assert_called_once_with(["runtime-push"])

    def test_advanced_live_aliases_route_to_serialized_hosted_send(self):
        for family in ("local-run", "delivery"):
            with self.subTest(family=family), mock.patch(
                "pipeline.operator_cli._dispatch_hosted_live", return_value=0
            ) as hosted, mock.patch(
                "pipeline.ops.runtime_prediction.run"
            ) as local_prediction, mock.patch(
                "pipeline.cli._send_predictions"
            ) as local_delivery, contextlib.redirect_stdout(io.StringIO()):
                rc = operator_cli.run(
                    ["advanced", family, "live"], root=pathlib.Path("/repo")
                )

            self.assertEqual(rc, 0)
            hosted.assert_called_once_with(root=pathlib.Path("/repo"))
            local_prediction.assert_not_called()
            local_delivery.assert_not_called()

    def test_errors_hide_traceback_unless_debug(self):
        with mock.patch("pipeline.operator_cli.collect_status", side_effect=RuntimeError("boom")), \
             contextlib.redirect_stderr(io.StringIO()) as stderr:
            rc = operator_cli.run(["status"], root=pathlib.Path("/repo"))
        self.assertEqual(rc, 1)
        self.assertNotIn("Traceback", stderr.getvalue())

        with mock.patch("pipeline.operator_cli.collect_status", side_effect=RuntimeError("boom")), \
             contextlib.redirect_stderr(io.StringIO()) as stderr:
            rc = operator_cli.run(["--debug", "status"], root=pathlib.Path("/repo"))
        self.assertEqual(rc, 1)
        self.assertIn("Traceback", stderr.getvalue())


class GitHubDispatchTests(unittest.TestCase):
    def test_unknown_mode_is_rejected_before_gh(self):
        with mock.patch("pipeline.cli_workflows._gh") as gh:
            with self.assertRaises(ValueError):
                cli_workflows.dispatch_and_wait("surprise", root=pathlib.Path("/repo"))
        gh.assert_not_called()

    def test_manual_live_requires_positive_confirmed_round_before_gh(self):
        for value in (None, 0, -1, True, "22"):
            with self.subTest(value=value), mock.patch("pipeline.cli_workflows._gh") as gh, \
                 mock.patch("pipeline.cli_workflows._workflow_runs") as runs:
                with self.assertRaises(ValueError):
                    cli_workflows.dispatch_and_wait(
                        "live", root=pathlib.Path("/repo"), confirmed_round=value
                    )
                gh.assert_not_called()
                runs.assert_not_called()

    def test_non_live_mode_rejects_confirmed_round_before_gh(self):
        with mock.patch("pipeline.cli_workflows._gh") as gh, \
             mock.patch("pipeline.cli_workflows._workflow_runs") as runs:
            with self.assertRaises(ValueError):
                cli_workflows.dispatch_and_wait(
                    "refresh", root=pathlib.Path("/repo"), confirmed_round=22
                )
            gh.assert_not_called()
            runs.assert_not_called()

    def test_live_dispatch_passes_exact_confirmed_round_input(self):
        listed = [
            [{"databaseId": 1}],
            [{"databaseId": 1}],
            [
                {"databaseId": 1},
                {"databaseId": 9, "displayTitle": "Predict manual (live)"},
            ],
        ]
        completed = mock.Mock(returncode=0, stdout="", stderr="")
        viewed = mock.Mock(
            returncode=0,
            stdout=json.dumps(
                {"databaseId": 9, "status": "completed", "conclusion": "success"}
            ),
            stderr="",
        )
        with mock.patch("pipeline.cli_workflows._workflow_runs", side_effect=listed), \
             mock.patch("pipeline.cli_workflows._gh", side_effect=[completed, completed, viewed]) as gh, \
             mock.patch("pipeline.cli_workflows.time.sleep"), \
             contextlib.redirect_stdout(io.StringIO()):
            cli_workflows.dispatch_and_wait(
                "live", root=pathlib.Path("/repo"), confirmed_round=22
            )

        dispatch_args = gh.call_args_list[0].args[0]
        self.assertIn("mode=live", dispatch_args)
        self.assertIn("confirmed_round=22", dispatch_args)

    def test_dispatch_waits_for_the_new_run_only(self):
        listed = [
            [{"databaseId": 1}, {"databaseId": 2}],
            [{"databaseId": 1}, {"databaseId": 2}],
            [
                {"databaseId": 1},
                {"databaseId": 2},
                {"databaseId": 9, "displayTitle": "Predict manual (test)"},
            ],
        ]
        completed = mock.Mock(returncode=0, stdout="", stderr="")
        viewed = mock.Mock(
            returncode=0,
            stdout=json.dumps(
                {"databaseId": 9, "status": "completed", "conclusion": "success", "url": "https://example/run/9"}
            ),
            stderr="",
        )
        with mock.patch("pipeline.cli_workflows._workflow_runs", side_effect=listed), \
             mock.patch("pipeline.cli_workflows._gh", side_effect=[completed, completed, viewed]) as gh, \
             mock.patch("pipeline.cli_workflows.time.sleep"), \
             contextlib.redirect_stdout(io.StringIO()):
            result = cli_workflows.dispatch_and_wait("test", root=pathlib.Path("/repo"))
        self.assertEqual(result["databaseId"], 9)
        self.assertIn("mode=test", gh.call_args_list[0].args[0])
        self.assertEqual(gh.call_args_list[1].args[0][:3], ["run", "watch", "9"])


if __name__ == "__main__":
    unittest.main()
