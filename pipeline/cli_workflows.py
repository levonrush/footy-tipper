"""Small, strict GitHub Actions client for the human CLI.

The operator commands deliberately dispatch one of the three named Predict
workflow modes.  Keeping this separate from the pipeline implementation makes
it hard for a friendly command to accidentally turn into a local production
send.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import time


PREDICT_WORKFLOW = "predict.yml"
PREDICT_MODES = frozenset({"test", "refresh", "live"})


class GitHubWorkflowError(RuntimeError):
    """The requested workflow could not be dispatched or did not succeed."""


def _gh(args: list[str], *, root: pathlib.Path, capture: bool = True) -> subprocess.CompletedProcess:
    if shutil.which("gh") is None:
        raise GitHubWorkflowError(
            "GitHub CLI (`gh`) is not installed. Run `footy-tipper setup` for help."
        )
    return subprocess.run(
        ["gh", *args],
        cwd=str(root),
        check=False,
        text=True,
        capture_output=capture,
    )


def _workflow_runs(root: pathlib.Path) -> list[dict]:
    result = _gh(
        [
            "run",
            "list",
            "--workflow",
            PREDICT_WORKFLOW,
            "--event",
            "workflow_dispatch",
            "--limit",
            "30",
            "--json",
            "databaseId,displayTitle,createdAt",
        ],
        root=root,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown gh error").strip()
        raise GitHubWorkflowError(f"Could not read GitHub Actions runs: {detail}")
    try:
        runs = json.loads(result.stdout or "[]")
        return [item for item in runs if item.get("databaseId") is not None]
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise GitHubWorkflowError("GitHub returned an unreadable workflow-run list.") from exc


def dispatch_and_wait(
    mode: str,
    *,
    root: pathlib.Path,
    confirmed_round: int | None = None,
    discovery_timeout: float = 45.0,
) -> dict:
    """Dispatch exactly one supported Predict mode and wait for its result."""
    if mode not in PREDICT_MODES:
        allowed = ", ".join(sorted(PREDICT_MODES))
        raise ValueError(f"Unsupported Predict mode {mode!r}; expected one of: {allowed}.")
    if mode == "live":
        if confirmed_round is None:
            raise ValueError("A manual live run requires the confirmed round number.")
        if isinstance(confirmed_round, bool) or not isinstance(confirmed_round, int) or confirmed_round < 1:
            raise ValueError("The confirmed round must be a positive whole number.")
    elif confirmed_round is not None:
        raise ValueError("A confirmed round is valid only for a live run.")

    before = {int(item["databaseId"]) for item in _workflow_runs(root)}
    dispatch_args = [
        "workflow",
        "run",
        PREDICT_WORKFLOW,
        "--ref",
        "main",
        "-f",
        f"mode={mode}",
    ]
    if confirmed_round is not None:
        dispatch_args.extend(["-f", f"confirmed_round={confirmed_round}"])
    dispatched = _gh(
        dispatch_args,
        root=root,
    )
    if dispatched.returncode != 0:
        detail = (dispatched.stderr or dispatched.stdout or "unknown gh error").strip()
        raise GitHubWorkflowError(f"GitHub did not accept the {mode} run: {detail}")

    print(f"Started the {mode.upper()} run. Waiting for GitHub Actions…", flush=True)
    deadline = time.monotonic() + discovery_timeout
    run_id = None
    while time.monotonic() < deadline:
        new_runs = [
            item
            for item in _workflow_runs(root)
            if int(item["databaseId"]) not in before
        ]
        matching = [
            item
            for item in new_runs
            if f"({mode})" in str(item.get("displayTitle", "")).lower()
        ]
        # Older run-name formats may not include the mode. A single new run is
        # still unambiguous; with concurrent runs we wait for the named match.
        selected = matching or (new_runs if len(new_runs) == 1 else [])
        if selected:
            run_id = max(int(item["databaseId"]) for item in selected)
            break
        time.sleep(1.0)
    if run_id is None:
        raise GitHubWorkflowError(
            "GitHub accepted the request, but its run did not appear within 45 seconds. "
            "Open Actions in GitHub to check it."
        )

    watched = _gh(
        ["run", "watch", str(run_id), "--exit-status"],
        root=root,
        capture=False,
    )
    viewed = _gh(
        [
            "run",
            "view",
            str(run_id),
            "--json",
            "databaseId,status,conclusion,url,displayTitle",
        ],
        root=root,
    )
    details = {"databaseId": run_id, "status": "completed"}
    if viewed.returncode == 0:
        try:
            details.update(json.loads(viewed.stdout or "{}"))
        except json.JSONDecodeError:
            pass
    if watched.returncode != 0:
        url = details.get("url")
        suffix = f" See {url}" if url else ""
        raise GitHubWorkflowError(f"The {mode} run failed.{suffix}")
    return details


def workflow_summary(*, root: pathlib.Path) -> dict:
    """Return a quiet status snapshot used by ``footy-tipper status``."""
    workflow = _gh(
        ["workflow", "list", "--all", "--json", "name,state,path,id"],
        root=root,
    )
    if workflow.returncode != 0:
        detail = (workflow.stderr or workflow.stdout or "unavailable").strip()
        return {"available": False, "error": detail}
    try:
        workflows = json.loads(workflow.stdout or "[]")
        summary = next(
            (
                item
                for item in workflows
                if item.get("path") == f".github/workflows/{PREDICT_WORKFLOW}"
            ),
            None,
        )
    except json.JSONDecodeError:
        return {"available": False, "error": "unreadable response"}
    if summary is None:
        return {"available": False, "error": "Predict workflow was not found"}

    latest = _gh(
        [
            "run",
            "list",
            "--workflow",
            PREDICT_WORKFLOW,
            "--limit",
            "1",
            "--json",
            "databaseId,status,conclusion,url,displayTitle,createdAt",
        ],
        root=root,
    )
    runs = []
    if latest.returncode == 0:
        try:
            runs = json.loads(latest.stdout or "[]")
        except json.JSONDecodeError:
            runs = []
    summary["available"] = True
    summary["latest_run"] = runs[0] if runs else None
    return summary
