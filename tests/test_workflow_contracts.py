import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
PREDICT_WORKFLOW = WORKFLOW_DIR / "predict.yml"
MODEL_CHECK_WORKFLOW = WORKFLOW_DIR / "model-check.yml"
SMOKE_WORKFLOW = WORKFLOW_DIR / "smoke-checks.yml"
WATCHDOG_SOURCE = REPO_ROOT / "watchdog" / "src" / "Code.js"
WATCHDOG_MANIFEST = REPO_ROOT / "watchdog" / "src" / "appsscript.json"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"
COMPOSE_FILE = REPO_ROOT / "compose.yml"
POST_MERGE_CUTOVER = REPO_ROOT / "pipeline" / "ops" / "post_merge_cutover.sh"


def _workflow_paths():
    return sorted(
        path
        for path in WORKFLOW_DIR.iterdir()
        if path.suffix in {".yml", ".yaml"}
    )


class WorkflowContractTests(unittest.TestCase):
    def test_hosted_training_workflow_is_removed(self):
        paths = _workflow_paths()
        self.assertFalse(any(path.stem == "train" for path in paths))

        workflow_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in paths
        )
        self.assertNotIn("pipeline.cli train", workflow_text)
        self.assertNotIn("footy-tipper train", workflow_text)
        self.assertNotIn("pipeline/train.py", workflow_text)
        self.assertNotIn("FOOTY_TIPPER_TUNE_ITER", workflow_text)
        self.assertNotIn("footy-tipper update-model", workflow_text)
        self.assertNotIn("pipeline.ops.model_release", workflow_text)
        self.assertNotIn("model_release.py", workflow_text)
        self.assertNotIn("caffeinate", workflow_text)
        self.assertNotRegex(workflow_text, r"gh workflow (?:enable|run) train\.yml")

    def test_predict_uses_exact_actions_runner_mode(self):
        workflow_text = PREDICT_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("options: [test, refresh, live]", workflow_text)
        self.assertIn("default: test", workflow_text)
        self.assertIn(
            'python -m pipeline.ops.actions_runner predict --mode "$RUN_MODE"',
            workflow_text,
        )
        self.assertNotIn("pipeline.cli predict", workflow_text)
        self.assertNotRegex(workflow_text, r"(?m)^\s*\*\)\s+")
        self.assertNotIn("--skip-auto-train", workflow_text)

    def test_predict_uses_targeted_off_boundary_sydney_schedule(self):
        workflow_text = PREDICT_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn('cron: "7,22,37,52 11 * * *"', workflow_text)
        self.assertIn('cron: "7,37 12-14 * * *"', workflow_text)
        self.assertEqual(
            workflow_text.count('timezone: "Australia/Sydney"'),
            2,
        )
        self.assertNotIn('cron: "*/15 * * * *"', workflow_text)

    def test_manual_live_is_bound_to_a_confirmed_round(self):
        workflow_text = PREDICT_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("confirmed_round:", workflow_text)
        self.assertIn("Manual LIVE requires a numeric confirmed_round.", workflow_text)
        self.assertIn('--confirmed-round "$CONFIRMED_ROUND"', workflow_text)
        self.assertIn("confirmed_round is valid only for LIVE.", workflow_text)

    def test_watchdog_can_only_request_the_gate_as_configured_source(self):
        workflow_text = PREDICT_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("watchdog:", workflow_text)
        self.assertIn("FOOTY_TIPPER_WATCHDOG_ACTOR", workflow_text)
        self.assertIn(
            "Watchdog dispatch refused: actor is not the configured watchdog source.",
            workflow_text,
        )
        self.assertIn(
            "Bot actors may only use the guarded watchdog gate.",
            workflow_text,
        )
        self.assertIn("Watchdog dispatch must not provide confirmed_round.", workflow_text)
        self.assertIn("pipeline.ops.actions_runner gate", workflow_text)

    def test_all_prediction_sources_share_the_state_concurrency_lock(self):
        workflow_text = PREDICT_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("group: footy-tipper-state", workflow_text)
        self.assertIn("cancel-in-progress: false", workflow_text)

    def test_automated_failures_reconcile_one_operator_alert(self):
        workflow_text = PREDICT_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("delivery-alert:", workflow_text)
        self.assertIn("issues: write", workflow_text)
        self.assertIn("automation-alert", workflow_text)
        self.assertIn("state_reason: \"completed\"", workflow_text)

    def test_model_check_workflow_is_validation_only(self):
        workflow_text = MODEL_CHECK_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:", workflow_text)
        self.assertNotIn("schedule:", workflow_text)
        self.assertIn("contents: read", workflow_text)
        self.assertIn("packages: read", workflow_text)
        self.assertIn(
            'python -m pipeline.ops.actions_runner model-check --release "$RELEASE_ID"',
            workflow_text,
        )
        self.assertIn("printf 'FOLDER_ID=%s\\n'", workflow_text)
        self.assertNotIn(
            'printf \'%s\\n\' "$SECRETS_ENV" > secrets.env', workflow_text
        )
        for forbidden in (
            "pipeline.cli train",
            "pipeline/train.py",
            "FOOTY_TIPPER_TUNE_ITER",
            "publish_model_release",
            "activate_model_release",
            "runtime-push",
            "state_sync push",
            "predict --mode",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, workflow_text)

    def test_workflow_uses_runtime_only_state_interface(self):
        workflow_text = PREDICT_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("pipeline.ops.actions_runner runtime-pull", workflow_text)
        self.assertIn("pipeline.ops.actions_runner runtime-push", workflow_text)
        self.assertNotIn("pipeline.ops.state_sync pull", workflow_text)
        self.assertNotIn("pipeline.ops.state_sync push", workflow_text)

    def test_predict_job_materialises_dedicated_odds_api_secret(self):
        workflow_text = PREDICT_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("ODDS_API_KEY_SECRET: ${{ secrets.ODDS_API_KEY }}", workflow_text)
        self.assertIn(
            "printf 'ODDS_API_KEY=%s\\n' \"$ODDS_API_KEY_SECRET\" >> secrets.env",
            workflow_text,
        )

    def test_test_mode_cannot_publish_runtime_or_site(self):
        workflow_text = PREDICT_WORKFLOW.read_text(encoding="utf-8")
        guard = "if: success() && needs.gate.outputs.mode != 'test'"
        self.assertEqual(workflow_text.count(guard), 2)
        self.assertIn("pipeline.ops.actions_runner site-publish", workflow_text)
        self.assertNotIn("continue-on-error: true", workflow_text)

    def test_gate_uses_live_vocabulary_machine_interface(self):
        workflow_text = PREDICT_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("pipeline.ops.actions_runner gate", workflow_text)
        self.assertNotIn("pipeline.ops.state_sync gate", workflow_text)

    def test_no_workflow_can_override_an_operator_pause(self):
        workflow_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in _workflow_paths()
        )
        self.assertNotRegex(workflow_text, r"gh workflow enable ")

    def test_smoke_checks_cover_the_independent_watchdog(self):
        workflow_text = SMOKE_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("working-directory: watchdog", workflow_text)
        self.assertIn("node-version: \"24\"", workflow_text)
        self.assertIn("npm ci --ignore-scripts", workflow_text)
        self.assertIn("npm run check", workflow_text)
        self.assertIn("npm test", workflow_text)
        self.assertIn("npm audit --audit-level=high", workflow_text)

    def test_watchdog_is_apps_script_and_dispatches_only_the_guarded_gate(self):
        source = WATCHDOG_SOURCE.read_text(encoding="utf-8")
        manifest = WATCHDOG_MANIFEST.read_text(encoding="utf-8")
        self.assertIn('timeZone": "Australia/Sydney"', manifest)
        self.assertIn(".everyMinutes(5)", source)
        self.assertIn('inputs: { watchdog: true }', source)
        self.assertIn('tokenProperty: "GITHUB_TOKEN"', source)
        self.assertIn('lastSlotProperty: "LAST_SUCCESSFUL_SLOT"', source)
        self.assertNotIn("Cloudflare", source)
        self.assertFalse((REPO_ROOT / "watchdog" / "wrangler.jsonc").exists())

    def test_unsafe_legacy_job_launchers_are_removed(self):
        self.assertFalse(COMPOSE_FILE.exists())
        self.assertFalse(POST_MERGE_CUTOVER.exists())

    def test_docker_context_excludes_mutable_state_but_keeps_references(self):
        patterns = set(DOCKERIGNORE.read_text(encoding="utf-8").splitlines())

        self.assertIn("data/*", patterns)
        self.assertIn("!data/.gitkeep", patterns)
        self.assertIn("!data/reference/", patterns)
        self.assertIn("!data/reference/**", patterns)
        self.assertIn("models/*", patterns)
        self.assertIn("!models/.gitkeep", patterns)
        self.assertNotIn("data/", patterns)
        self.assertNotIn("models/", patterns)


if __name__ == "__main__":
    unittest.main()
