import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
PREDICT_WORKFLOW = WORKFLOW_DIR / "predict.yml"
MODEL_CHECK_WORKFLOW = WORKFLOW_DIR / "model-check.yml"
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

    def test_manual_live_is_bound_to_a_confirmed_round(self):
        workflow_text = PREDICT_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("confirmed_round:", workflow_text)
        self.assertIn("Manual LIVE requires a numeric confirmed_round.", workflow_text)
        self.assertIn('--confirmed-round "$CONFIRMED_ROUND"', workflow_text)
        self.assertIn("confirmed_round is valid only for LIVE.", workflow_text)

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
