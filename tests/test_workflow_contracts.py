import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
PREDICT_WORKFLOW = WORKFLOW_DIR / "predict.yml"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"


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
        self.assertNotRegex(workflow_text, r"gh workflow (?:enable|run) train\.yml")

    def test_every_predict_mode_disables_automatic_training(self):
        workflow_text = PREDICT_WORKFLOW.read_text(encoding="utf-8")
        mode_commands = dict(re.findall(
            r"^\s*(test|refresh|\*)\)\s+(python -m pipeline\.cli predict[^;\n]*)\s+;;$",
            workflow_text,
            flags=re.MULTILINE,
        ))
        self.assertEqual(
            mode_commands,
            {
                "test": "python -m pipeline.cli predict --test --skip-auto-train",
                "refresh": "python -m pipeline.cli predict --skip-send --skip-auto-train",
                "*": "python -m pipeline.cli predict --skip-auto-train",
            },
        )
        for command in mode_commands.values():
            with self.subTest(command=command):
                self.assertIn("--skip-auto-train", command)

    def test_no_workflow_can_override_an_operator_pause(self):
        workflow_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in _workflow_paths()
        )
        self.assertNotRegex(workflow_text, r"gh workflow enable ")

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
