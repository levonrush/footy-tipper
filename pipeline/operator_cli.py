"""The small, safety-first human interface for Footy Tipper 1.0.

This module intentionally separates *operator intent* from the lower-level
pipeline functions that still live in :mod:`pipeline.cli`.  The beginner
commands either read published state or dispatch an exact GitHub Actions mode;
they never silently train a model or send a production email locally.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import importlib.util
import json
import os
import pathlib
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import traceback
from typing import Any
from zoneinfo import ZoneInfo

from pipeline import cli_workflows
from pipeline.common import console

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - setup reports missing/incomplete environments
    def load_dotenv(*_args, **_kwargs):
        return False


CLI_VERSION = "1.0.0"
JSON_SCHEMA_VERSION = "1.0"
EXIT_OK = 0
EXIT_OPERATIONAL = 1
EXIT_INVOCATION = 2
EXIT_SAFETY = 3
EXIT_INTERRUPT = 130

RETIRED_COMMANDS = {
    "prep": "footy-tipper advanced data prepare all",
    "train": "footy-tipper advanced model train",
    "infer": "footy-tipper advanced model infer",
    "send": "footy-tipper advanced delivery live",
    "predict": "footy-tipper advanced local-run live",
    "lineups": "footy-tipper advanced data lineups refresh",
    "nrl-data": "footy-tipper advanced data nrl refresh",
    "odds": "footy-tipper advanced data odds refresh",
    "site": "footy-tipper advanced site build",
    "evaluate": "footy-tipper advanced model evaluate",
    "state": "footy-tipper advanced cloud pull-runtime",
}

STATE_ACTION_REPLACEMENTS = {
    "pull": "footy-tipper advanced cloud pull-runtime",
    "push": "footy-tipper advanced cloud push-runtime",
    "schedule": "footy-tipper advanced cloud schedule",
    "gate": "footy-tipper advanced cloud gate",
}


class InvocationError(RuntimeError):
    pass


class SafetyRefusal(RuntimeError):
    pass


class FriendlyParser(argparse.ArgumentParser):
    def error(self, message):
        raise InvocationError(message)


def _project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def _emit_json(command: str, ok: bool, **payload: Any) -> None:
    document = {
        "schema_version": JSON_SCHEMA_VERSION,
        "command": command,
        "ok": bool(ok),
        **payload,
    }
    print(json.dumps(document, indent=2, sort_keys=True, default=str))


def _secret_values(root: pathlib.Path) -> list[str]:
    values = []
    sensitive = re.compile(r"(password|secret|token|api.?key)", re.IGNORECASE)
    for key, value in os.environ.items():
        if sensitive.search(key) and len(value) >= 4:
            values.append(value)
    secrets_path = root / "secrets.env"
    if secrets_path.is_file():
        try:
            for raw in secrets_path.read_text(encoding="utf-8").splitlines():
                key, separator, value = raw.partition("=")
                if separator and sensitive.search(key):
                    cleaned = value.strip().strip("\"").strip("'")
                    if len(cleaned) >= 4:
                        values.append(cleaned)
        except OSError:
            pass
    return sorted(set(values), key=len, reverse=True)


def _redact(message: object, root: pathlib.Path) -> str:
    cleaned = str(message)
    for value in _secret_values(root):
        cleaned = cleaned.replace(value, "[REDACTED]")
    cleaned = re.sub(
        r"(?i)((?:password|secret|token|api[_-]?key)\s*[=:]\s*)\S+",
        r"\1[REDACTED]",
        cleaned,
    )
    return cleaned


def _redact_payload(value: Any, root: pathlib.Path) -> Any:
    if isinstance(value, dict):
        return {key: _redact_payload(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_payload(item, root) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_payload(item, root) for item in value)
    if isinstance(value, str):
        return _redact(value, root)
    return value


def _quiet_run(args: list[str], root: pathlib.Path) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            args,
            cwd=str(root),
            check=False,
            text=True,
            capture_output=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(args, 124, stdout="", stderr="timed out")


REQUIRED_GITHUB_WORKFLOWS = {
    "predict": {
        "path": ".github/workflows/predict.yml",
        "label": "Prediction automation",
    },
    "model_check": {
        "path": ".github/workflows/model-check.yml",
        "label": "Hosted model validation",
    },
}


def _github_workflow_readiness(root: pathlib.Path) -> dict:
    """Check that both operator-facing workflows exist and are enabled."""
    if shutil.which("gh") is None:
        return {
            "available": False,
            "ready": False,
            "error": "GitHub CLI is not installed",
            "workflows": {},
        }
    result = _quiet_run(
        ["gh", "workflow", "list", "--all", "--json", "name,state,path,id"],
        root,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unavailable").strip()
        return {
            "available": False,
            "ready": False,
            "error": detail,
            "workflows": {},
        }
    try:
        listed = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        return {
            "available": False,
            "ready": False,
            "error": "GitHub returned an unreadable workflow list",
            "workflows": {},
        }

    by_path = {item.get("path"): item for item in listed if item.get("path")}
    workflows = {}
    for key, requirement in REQUIRED_GITHUB_WORKFLOWS.items():
        item = by_path.get(requirement["path"])
        state = item.get("state") if item else None
        workflows[key] = {
            "label": requirement["label"],
            "path": requirement["path"],
            "present": item is not None,
            "state": state,
            "ready": item is not None and str(state).lower() == "active",
        }
    return {
        "available": True,
        "ready": all(item["ready"] for item in workflows.values()),
        "workflows": workflows,
    }


def _git_status(root: pathlib.Path) -> dict:
    branch = _quiet_run(["git", "branch", "--show-current"], root)
    head = _quiet_run(["git", "rev-parse", "--short", "HEAD"], root)
    dirty = _quiet_run(["git", "status", "--porcelain"], root)
    if branch.returncode or head.returncode or dirty.returncode:
        return {"available": False}
    return {
        "available": True,
        "branch": branch.stdout.strip(),
        "commit": head.stdout.strip(),
        "clean": not bool(dirty.stdout.strip()),
    }


def _local_model_status(root: pathlib.Path) -> dict:
    models = root / "models"
    required = ("home_model.pkl", "away_model.pkl", "model_manifest.json")
    missing = [name for name in required if not (models / name).is_file()]
    release_id = None
    manifest = models / "model_manifest.json"
    if manifest.is_file():
        try:
            body = json.loads(manifest.read_text(encoding="utf-8"))
            release = body.get("release") if isinstance(body.get("release"), dict) else {}
            release_id = (
                body.get("release_id")
                or body.get("model_release_id")
                or release.get("release_id")
            )
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "ready": not missing,
        "missing": missing,
        "release_id": release_id,
    }


def _model_update_journal(root: pathlib.Path) -> dict | None:
    path = root / ".footy-tipper" / "model-update.json"
    if not path.is_file():
        return None
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {"status": "unreadable", "path": str(path)}
    return {
        "status": body.get("status"),
        "current_stage": body.get("current_stage"),
        "release_id": body.get("release_id"),
        "updated_at_utc": body.get("updated_at_utc"),
        "completed_at_utc": body.get("completed_at_utc"),
        "last_error": body.get("last_error"),
        "resumable": body.get("status") not in {None, "complete", "cancelled"},
    }


def _published_schedule(root: pathlib.Path) -> dict:
    """Read the small Drive schedule without altering any local state."""
    from pipeline.ops import state_sync

    service = state_sync.drive_service(root / "service-account-token.json")
    state_id = state_sync._existing_state_folder(service, root)
    schedule_id = state_sync.find_file_id(
        service, state_id, state_sync.SCHEDULE_FILE
    )
    if not schedule_id:
        raise RuntimeError("Published schedule.json is missing.")
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / state_sync.SCHEDULE_FILE
        state_sync.download_to(service, schedule_id, path)
        schedule = json.loads(path.read_text(encoding="utf-8"))

    next_round = None
    for entry in schedule.get("upcoming_rounds", []):
        if not entry.get("sent"):
            next_round = dict(entry)
            break
    result = {
        "generated_at_utc": schedule.get("generated_at_utc"),
        "competition_year": schedule.get("competition_year"),
        "next_unsent_round": next_round,
    }
    if next_round and next_round.get("first_kickoff_utc") is not None:
        target = state_sync.sydney_send_target_utc(next_round["first_kickoff_utc"])
        target_dt = dt.datetime.fromtimestamp(
            target, tz=dt.timezone.utc
        ).astimezone(ZoneInfo("Australia/Sydney"))
        result["next_target_utc"] = int(target)
        result["next_target_sydney"] = target_dt.isoformat()
    return result


def collect_status(*, root: pathlib.Path, offline: bool) -> dict:
    status = {
        "version": CLI_VERSION,
        "offline": bool(offline),
        "git": _git_status(root),
        "local_database": {
            "present": (root / "data" / "footy-tipper-db.sqlite").is_file()
        },
        "local_models": _local_model_status(root),
        "model_update": _model_update_journal(root),
        "configuration": {
            "secrets_env": (root / "secrets.env").is_file(),
            "service_account": (root / "service-account-token.json").is_file(),
        },
    }
    if offline:
        return status
    status["github_workflows"] = _github_workflow_readiness(root)
    try:
        status["github_actions"] = cli_workflows.workflow_summary(root=root)
    except Exception as exc:
        status["github_actions"] = {"available": False, "error": str(exc)}
    try:
        status["published_schedule"] = _published_schedule(root)
    except Exception as exc:
        status["published_schedule"] = {"available": False, "error": str(exc)}
    try:
        from pipeline.ops import state_sync

        status["active_model"] = {
            "available": True,
            "pointer": state_sync.get_model_pointer(root),
        }
    except Exception as exc:
        status["active_model"] = {"available": False, "error": str(exc)}
    try:
        from pipeline.ops import delivery_state

        deliveries = delivery_state.list_deliveries(root)
        unresolved = [
            marker for marker in deliveries if marker.get("status") != "sent"
        ]
        status["delivery_safety"] = {
            "available": True,
            "unresolved": unresolved,
            "latest": deliveries[0] if deliveries else None,
        }
    except Exception as exc:
        status["delivery_safety"] = {"available": False, "error": str(exc)}
    return status


def _print_status(status: dict) -> None:
    git = status["git"]
    if git.get("available"):
        worktree = "clean" if git.get("clean") else "has uncommitted changes"
        print(f"Code: {git.get('branch')} @ {git.get('commit')} ({worktree})")
    else:
        print("Code: unavailable")
    database = "ready" if status["local_database"]["present"] else "missing"
    models = status["local_models"]
    model_text = "ready" if models["ready"] else "missing files"
    if models.get("release_id"):
        model_text += f" ({models['release_id']})"
    print(f"This Mac: database {database}; models {model_text}")
    update = status.get("model_update")
    if update:
        if update.get("resumable"):
            print(
                "Model update: "
                f"{update.get('status', 'incomplete')} at {update.get('current_stage', 'unknown stage')}; "
                "run `footy-tipper update-model` to resume"
            )
        elif update.get("status") == "cancelled":
            print("Model update: cancelled safely; the next run starts a fresh release")
        else:
            print(
                "Model update: complete"
                + (f" ({update.get('release_id')})" if update.get("release_id") else "")
            )
    config = status["configuration"]
    configured = config["secrets_env"] and config["service_account"]
    workflow_readiness = status.get("github_workflows")
    if status.get("offline"):
        print(
            "Setup: local files ready; online checks skipped"
            if configured
            else "Setup: local files need attention; online checks skipped"
        )
    else:
        if workflow_readiness is not None:
            configured = configured and workflow_readiness.get("ready", False)
        active_model = status.get("active_model")
        if active_model is not None:
            configured = configured and active_model.get("available", False)
        print(f"Setup: {'ready' if configured else 'needs attention'}")

    workflow = status.get("github_actions")
    if workflow:
        if workflow.get("available"):
            latest = workflow.get("latest_run") or {}
            state = workflow.get("state", "unknown")
            if latest:
                result = latest.get("conclusion") or latest.get("status") or "unknown"
                print(f"GitHub prediction: {state}; latest run {result}")
            else:
                print(f"GitHub prediction: {state}; no runs found")
        else:
            print("GitHub prediction: unavailable")

    workflows = status.get("github_workflows")
    if workflows:
        model_check = (workflows.get("workflows") or {}).get("model_check")
        if model_check:
            state = model_check.get("state") or "missing"
            print(f"Hosted model validation: {state}")
        elif not workflows.get("available"):
            print("Hosted model validation: unavailable")

    active = status.get("active_model")
    if active:
        if active.get("available"):
            pointer = active.get("pointer") or {}
            print(f"Production model: {pointer.get('release_id', 'unknown')}")
        else:
            print("Production model: unavailable")

    schedule = status.get("published_schedule")
    if schedule:
        upcoming = schedule.get("next_unsent_round")
        if upcoming:
            target = schedule.get("next_target_sydney", "target unavailable")
            print(
                f"Next automatic send: round {upcoming.get('round_id')} at {target}"
            )
        elif schedule.get("available") is False:
            print("Published schedule: unavailable")
        else:
            print("Published schedule: no unsent round")

    delivery = status.get("delivery_safety")
    if delivery and delivery.get("available"):
        unresolved = delivery.get("unresolved") or []
        if unresolved:
            marker = unresolved[0]
            print(
                "Delivery safety: ATTENTION — "
                f"{marker.get('competition_year')} round {marker.get('round_id')} is "
                f"{marker.get('status', 'uncertain')}; do not resend automatically"
            )
        else:
            latest = delivery.get("latest")
            if latest:
                print(
                    "Delivery safety: clear; latest sent marker is "
                    f"{latest.get('competition_year')} round {latest.get('round_id')}"
                )
            else:
                print("Delivery safety: clear; no production send markers yet")
    elif delivery:
        print("Delivery safety: unavailable")


def command_status(args, *, root: pathlib.Path) -> int:
    status = collect_status(root=root, offline=args.offline)
    if args.json:
        _emit_json("status", True, status=_redact_payload(status, root))
    else:
        _print_status(status)
    return EXIT_OK


def command_setup(_args, *, root: pathlib.Path) -> int:
    everyday = [
        ("Git", shutil.which("git") is not None, "Install Git."),
        (
            "GitHub CLI",
            shutil.which("gh") is not None,
            "Install `gh`; Footy Tipper will use an existing saved login and will not open an auth flow.",
        ),
        ("secrets.env", (root / "secrets.env").is_file(), "Create secrets.env from secrets.env.example."),
        (
            "Google service account",
            (root / "service-account-token.json").is_file(),
            "Put service-account-token.json in this project folder.",
        ),
    ]
    if shutil.which("gh"):
        everyday.append(
            (
                "GitHub sign-in",
                _quiet_run(["gh", "auth", "status"], root).returncode == 0,
                "GitHub access is unavailable. This command stopped without changing anything; use a permitted computer or ask for help instead of starting a university auth flow.",
            )
        )

    drive_ready = False
    if everyday[2][1] and everyday[3][1]:
        try:
            from pipeline.ops import state_sync

            service = state_sync.drive_service(root / "service-account-token.json")
            state_sync._existing_state_folder(service, root)
            state_sync.get_model_pointer(root)
            drive_ready = True
        except Exception:
            drive_ready = False
    everyday.append(
        (
            "Published Drive state",
            drive_ready,
            "Check FOLDER_ID, service-account access, and the active model pointer.",
        )
    )

    workflow_readiness = _github_workflow_readiness(root)
    workflow_checks = workflow_readiness.get("workflows") or {}
    predict_workflow = workflow_checks.get("predict") or {}
    model_check_workflow = workflow_checks.get("model_check") or {}
    everyday.append(
        (
            "Prediction automation workflow",
            bool(predict_workflow.get("ready")),
            "Push and enable `.github/workflows/predict.yml` in GitHub Actions.",
        )
    )

    free_gib = shutil.disk_usage(root).free / (1024 ** 3)
    model_update = [
        (
            "Conda environment",
            os.getenv("CONDA_DEFAULT_ENV") == "footy-tipper",
            "Run `conda activate footy-tipper`.",
        ),
        (
            "Python 3.11+",
            sys.version_info >= (3, 11),
            "Use the footy-tipper Conda environment with Python 3.11 or newer.",
        ),
        ("R", shutil.which("Rscript") is not None, "Install R for model/data work."),
        (
            "Python model packages",
            all(
                importlib.util.find_spec(package) is not None
                for package in ("lightgbm", "skopt", "pandas", "dill", "sklearn")
            ),
            "Run `conda env update -f environment.yml --prune`.",
        ),
        (
            "Keep-awake tool",
            shutil.which("caffeinate") is not None,
            "`caffeinate` is included with macOS.",
        ),
        (
            "Hosted model-validation workflow",
            bool(model_check_workflow.get("ready")),
            "Push and enable `.github/workflows/model-check.yml` in GitHub Actions.",
        ),
        (
            "Free disk space (5 GB+)",
            free_gib >= 5,
            f"Free some disk space; only {free_gib:.1f} GB is available.",
        ),
    ]

    print("Footy Tipper setup check\n")
    for heading, checks in (("Everyday tips", everyday), ("Model updates", model_update)):
        print(f"{heading}:")
        for label, ready, advice in checks:
            print(f"  {'✓' if ready else '✗'} {label}")
            if not ready:
                print(f"      {advice}")
        print()
    everyday_missing = [label for label, ready, _ in everyday if not ready]
    model_missing = [label for label, ready, _ in model_update if not ready]
    if everyday_missing or model_missing:
        if not everyday_missing:
            print("Everyday tips are ready; only model-update items need attention.")
        print("\nFix the items marked ✗, then run `footy-tipper setup` again.")
        return EXIT_INVOCATION
    print("Everything needed for everyday tips and model updates is ready.")
    print("Next: run `footy-tipper` for the guided menu.")
    return EXIT_OK


def _download_published_db(destination: pathlib.Path, *, root: pathlib.Path) -> pathlib.Path:
    from pipeline.ops import state_sync

    downloaded = state_sync.download_runtime_db(destination, root=root)
    path = pathlib.Path(downloaded or destination)
    if not path.is_file():
        raise RuntimeError("Published runtime database download did not produce a file.")
    with sqlite3.connect(str(path)) as connection:
        result = connection.execute("PRAGMA quick_check").fetchone()
    if not result or str(result[0]).lower() != "ok":
        raise RuntimeError("Published runtime database failed its integrity check.")
    return path


def _published_predictions(root: pathlib.Path):
    with tempfile.TemporaryDirectory(prefix="footy-tipper-show-") as tmp:
        db_path = _download_published_db(
            pathlib.Path(tmp) / "runtime.sqlite", root=root
        )
        from pipeline.common.use_predictions import sending_functions as sf

        return sf.get_predictions(db_path, root)


def _prediction_records(predictions) -> list[dict]:
    if predictions.empty:
        return []
    safe = predictions.where(predictions.notna(), None)
    return [dict(row) for row in safe.to_dict(orient="records")]


def _print_predictions(predictions) -> None:
    if predictions.empty:
        print("There are no published tips for an upcoming round yet.")
        return
    first = predictions.iloc[0]
    round_label = first.get("round_name") or f"Round {int(first['round_id'])}"
    print(
        f"Published tips — {int(first['competition_year'])} "
        f"{round_label}\n"
    )
    for _, row in predictions.iterrows():
        home = str(row.get("team_home", "Home"))
        away = str(row.get("team_away", "Away"))
        home_result = str(row.get("home_team_result", "")).lower()
        tip = home if home_result == "win" else away
        home_score = row.get("predicted_home_score")
        away_score = row.get("predicted_away_score")
        score = ""
        if home_score is not None and away_score is not None:
            try:
                score = f" — {int(home_score)}–{int(away_score)}"
            except (TypeError, ValueError):
                pass
        print(f"  {home} v {away}: {tip}{score}")


def command_tips_show(args, *, root: pathlib.Path) -> int:
    if args.json:
        with contextlib.redirect_stdout(sys.stderr):
            predictions = _published_predictions(root)
        _emit_json("tips show", True, tips=_prediction_records(predictions))
    else:
        predictions = _published_predictions(root)
        _print_predictions(predictions)
    return EXIT_OK


def _round_from_predictions(predictions) -> int:
    if predictions.empty:
        raise SafetyRefusal("There is no upcoming published round to send.")
    try:
        return int(predictions.iloc[0]["round_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SafetyRefusal("The upcoming round number could not be verified.") from exc


def _ensure_delivery_clear(
    root: pathlib.Path, round_id: int, competition_year: int | None = None
) -> None:
    try:
        if competition_year is None:
            predictions = _published_predictions(root)
            if predictions.empty:
                raise SafetyRefusal("The published season could not be verified.")
            competition_year = int(predictions.iloc[0]["competition_year"])
        from pipeline.ops import delivery_state

        marker = delivery_state.get_delivery(root, competition_year, round_id)
    except SafetyRefusal:
        raise
    except Exception as exc:
        raise SafetyRefusal(
            "The production delivery marker could not be checked, so a live send is blocked."
        ) from exc
    if marker:
        raise SafetyRefusal(
            f"{competition_year} round {round_id} already has a "
            f"{marker.get('status', 'unknown')} delivery marker. Check `footy-tipper status`; "
            "do not resend automatically."
        )


def _confirm_phrase(expected: str, explanation: str) -> None:
    if not sys.stdin.isatty():
        raise SafetyRefusal(
            "This operation requires an interactive terminal; no non-interactive override exists."
        )
    print(f"\n{explanation}")
    entered = input(f"Type {expected} to continue: ").strip()
    if entered != expected:
        raise SafetyRefusal("Operation cancelled; the confirmation text did not match.")


def _confirm_live(round_id: int) -> None:
    _confirm_phrase(
        f"SEND ROUND {round_id}",
        "This sends email to the real production list.",
    )


def _dispatch(
    mode: str, *, root: pathlib.Path, confirmed_round: int | None = None
) -> int:
    if confirmed_round is None:
        details = cli_workflows.dispatch_and_wait(mode, root=root)
    else:
        details = cli_workflows.dispatch_and_wait(
            mode, root=root, confirmed_round=confirmed_round
        )
    result = details.get("conclusion") or "completed"
    url = details.get("url")
    print(f"GitHub Actions {mode.upper()} run {result}.")
    if url:
        print(url)
    return EXIT_OK


def _dispatch_hosted_live(*, root: pathlib.Path) -> int:
    """Bind every human production send to the serialized Actions workflow."""
    predictions = _published_predictions(root)
    round_id = _round_from_predictions(predictions)
    competition_year = int(predictions.iloc[0]["competition_year"])
    _ensure_delivery_clear(root, round_id, competition_year)
    _confirm_live(round_id)
    return _dispatch("live", root=root, confirmed_round=round_id)


def command_tips(args, *, root: pathlib.Path) -> int:
    if args.tips_command == "show":
        return command_tips_show(args, root=root)
    if args.tips_command == "live":
        return _dispatch_hosted_live(root=root)
    return _dispatch(args.tips_command, root=root)


def _call_model_release(name: str, root: pathlib.Path, *args, **kwargs) -> int:
    from pipeline.ops import model_release

    function = getattr(model_release, name, None)
    if not callable(function):
        raise RuntimeError(f"Model-release operation `{name}` is unavailable in this checkout.")
    result = function(root, *args, **kwargs)
    return EXIT_OK if result is None else int(result)


def command_update_model(args, *, root: pathlib.Path, debug: bool) -> int:
    return _call_model_release(
        "update_model", root, json_output=args.json, debug=debug
    )


def _engine():
    # Imported lazily to avoid a module cycle: pipeline.cli delegates back to
    # this operator layer after defining its reusable pipeline helpers.
    from pipeline import cli

    return cli


def _advanced_env(args):
    return _engine()._build_env(args)


def _advanced_data(args, *, root: pathlib.Path) -> int:
    engine = _engine()
    env = _advanced_env(args)
    if args.data_command == "prepare":
        scope = args.scope
        env["FOOTY_TIPPER_PREP_MODE"] = {
            "all": "full",
            "training": "train",
            "tips": "infer",
        }[scope]
        if not args.skip_lineups:
            if scope in {"all", "training"}:
                engine._bootstrap_lineups_for_training_if_needed(env, root)
            engine._run_lineups(env, root)
        if not args.skip_nrl_data:
            engine._refresh_nrl_data(env, root, include_bootstrap=scope in {"all", "training"})
        engine._run_data_prep(env, root)
        return EXIT_OK
    if args.data_command == "lineups":
        env["FOOTY_TIPPER_LINEUPS_MODE"] = "recent" if args.action == "refresh" else "backfill"
        if args.max_articles is not None:
            env["FOOTY_TIPPER_LINEUPS_MAX_ARTICLES"] = str(args.max_articles)
        if args.strict:
            env["FOOTY_TIPPER_LINEUPS_STRICT"] = "true"
        engine._run_lineups(env, root)
        return EXIT_OK
    if args.data_command == "nrl":
        extra = []
        for flag, value in (
            ("--start-year", args.start_year),
            ("--end-year", args.end_year),
            ("--season", args.season),
            ("--max-pages", args.max_pages),
            ("--report-path", args.report_path),
        ):
            if value is not None:
                extra.extend([flag, str(value)])
        if args.strict:
            extra.append("--strict")
        engine._run_nrl_data(env, root, args.action, extra)
        return EXIT_OK
    if args.data_command == "odds":
        extra = []
        if args.xlsx_path:
            extra.extend(["--xlsx-path", args.xlsx_path])
        if args.url:
            extra.extend(["--url", args.url])
        if args.strict:
            extra.append("--strict")
        engine._run_odds(env, root, "live" if args.action == "refresh" else "backfill", extra)
        return EXIT_OK
    raise InvocationError("Unknown advanced data command.")


def _advanced_model(args, *, root: pathlib.Path) -> int:
    engine = _engine()
    action = args.action
    if action == "verify":
        return _call_model_release("verify_active", root)
    if action == "list":
        return _call_model_release("list_releases", root, json_output=args.json)
    if action == "activate":
        _confirm_phrase(
            f"ACTIVATE {args.release_id}",
            "This changes the model used by the automatic production run.",
        )
        return _call_model_release("activate_release", root, args.release_id)
    if action == "rollback":
        _confirm_phrase(
            "ROLL BACK MODEL",
            "This changes production back to the previous model release.",
        )
        return _call_model_release("rollback", root)

    env = _advanced_env(args)
    if action == "train":
        env["FOOTY_TIPPER_PREP_MODE"] = "train"
        env["FOOTY_TIPPER_TUNE_ITER"] = str(args.tuning_candidates)
        if not args.skip_prepare:
            engine._bootstrap_lineups_for_training_if_needed(env, root)
            engine._run_lineups(env, root)
            engine._refresh_nrl_data(env, root, include_bootstrap=True)
        engine._run_train(env, skip_prep=args.skip_prepare, root=root)
        return EXIT_OK
    if action == "infer":
        env["FOOTY_TIPPER_PREP_MODE"] = "infer"
        if not args.skip_prepare:
            engine._run_lineups(env, root)
            engine._refresh_nrl_data(env, root)
        if not engine._ensure_models_for_prediction(
            env,
            root,
            auto_train=args.auto_train,
            allow_lineup_bootstrap=args.auto_train,
        ):
            return EXIT_OPERATIONAL
        engine._run_inference(env, skip_prep=args.skip_prepare, root=root)
        return EXIT_OK
    if action == "evaluate":
        env["FOOTY_TIPPER_PREP_MODE"] = "train"
        if not engine._model_artifacts_exist(root):
            raise RuntimeError("Model artifacts are missing; train a model first.")
        if args.seasons is not None:
            env["FOOTY_TIPPER_EVAL_SEASONS"] = str(args.seasons)
        engine._run_evaluate(env, skip_prep=args.skip_prepare, root=root)
        return EXIT_OK
    raise InvocationError("Unknown advanced model command.")


def _advanced_local_run(args, *, root: pathlib.Path) -> int:
    from pipeline.ops import runtime_prediction

    if args.action == "live":
        print(
            "Production sends are serialized in GitHub Actions; "
            "this live action will use the hosted workflow."
        )
        return _dispatch_hosted_live(root=root)
    mode = {"preview": "refresh", "test": "test", "live": "live"}[args.action]
    return int(runtime_prediction.run(mode))


def _advanced_delivery(args, *, root: pathlib.Path) -> int:
    engine = _engine()
    action = args.action
    if action == "live":
        print(
            "Production sends are serialized in GitHub Actions; "
            "this live action will use the hosted workflow."
        )
        return _dispatch_hosted_live(root=root)
    test_mode = action != "live"
    dry_run = action == "preview"
    return int(
        engine._send_predictions(
            test_mode=test_mode,
            test_email=engine._resolve_test_email(getattr(args, "test_email", None)),
            skip_drive=test_mode,
            use_llm=not getattr(args, "no_llm", False),
            dry_run=dry_run,
            force_resend=False,
        )
    )


def _advanced_cloud(args, *, root: pathlib.Path) -> int:
    from pipeline.ops import actions_runner, state_sync

    if args.action == "schedule":
        return int(state_sync.print_schedule(root))

    action = {
        "pull-runtime": "runtime-pull",
        "push-runtime": "runtime-push",
        "gate": "gate",
    }[args.action]
    return int(actions_runner.main([action]))


def _advanced_site(args, *, root: pathlib.Path) -> int:
    from pipeline.common.use_predictions import site

    db_path = root / "data" / "footy-tipper-db.sqlite"
    site.generate_site(db_path, root)
    if args.action == "publish":
        return EXIT_OK if site.publish_site(root) else EXIT_OPERATIONAL
    return EXIT_OK


def command_advanced(args, *, root: pathlib.Path) -> int:
    if args.advanced_command == "data":
        return _advanced_data(args, root=root)
    if args.advanced_command == "model":
        return _advanced_model(args, root=root)
    if args.advanced_command == "local-run":
        return _advanced_local_run(args, root=root)
    if args.advanced_command == "delivery":
        return _advanced_delivery(args, root=root)
    if args.advanced_command == "cloud":
        return _advanced_cloud(args, root=root)
    if args.advanced_command == "site":
        return _advanced_site(args, root=root)
    raise InvocationError("Unknown advanced command.")


def _add_years(parser) -> None:
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = FriendlyParser(
        prog="footy-tipper",
        description="NRL tips without the pipeline guesswork.",
    )
    parser.add_argument("--debug", action="store_true", help="Show a traceback when an operation fails.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {CLI_VERSION}")
    top = parser.add_subparsers(dest="command", parser_class=FriendlyParser)

    status = top.add_parser("status", help="Show whether this Mac and the automatic run are ready.")
    status.add_argument("--offline", action="store_true", help="Do not contact GitHub or Google Drive.")
    status.add_argument("--json", action="store_true", help="Emit stable machine-readable JSON.")

    top.add_parser("setup", help="Check the few things Footy Tipper needs.")

    tips = top.add_parser("tips", help="Show or safely run the weekly tips workflow.")
    tips_sub = tips.add_subparsers(dest="tips_command", required=True, parser_class=FriendlyParser)
    show = tips_sub.add_parser("show", help="Show tips from the published runtime database (read-only).")
    show.add_argument("--json", action="store_true", help="Emit stable machine-readable JSON.")
    tips_sub.add_parser("test", help="Run a test-email workflow in GitHub Actions and wait.")
    tips_sub.add_parser("refresh", help="Refresh published tips without sending email, then wait.")
    tips_sub.add_parser("live", help="Send the real round email after typed confirmation.")

    update = top.add_parser("update-model", help="Safely train, validate, and publish a new model.")
    update.add_argument("--json", action="store_true", help="Emit machine-readable progress/result output.")

    advanced = top.add_parser("advanced", help="Technical pipeline tools (normally unnecessary).")
    advanced_sub = advanced.add_subparsers(dest="advanced_command", required=True, parser_class=FriendlyParser)

    data = advanced_sub.add_parser("data", help="Prepare or ingest data.")
    data_sub = data.add_subparsers(dest="data_command", required=True, parser_class=FriendlyParser)
    prepare = data_sub.add_parser("prepare", help="Prepare all, training, or tips data.")
    prepare.add_argument("scope", choices=("all", "training", "tips"))
    _add_years(prepare)
    prepare.add_argument("--skip-lineups", action="store_true")
    prepare.add_argument("--skip-nrl-data", action="store_true")

    lineups = data_sub.add_parser("lineups", help="Refresh or backfill team lists.")
    lineups.add_argument("action", choices=("refresh", "backfill"))
    _add_years(lineups)
    lineups.add_argument("--max-articles", type=int)
    lineups.add_argument("--strict", action="store_true")

    nrl = data_sub.add_parser("nrl", help="Refresh, backfill, or validate nrl.com data.")
    nrl.add_argument("action", choices=("refresh", "backfill", "validate"))
    _add_years(nrl)
    nrl.add_argument("--season", type=int)
    nrl.add_argument("--max-pages", type=int)
    nrl.add_argument("--report-path")
    nrl.add_argument("--strict", action="store_true")

    odds = data_sub.add_parser("odds", help="Refresh live odds or backfill history.")
    odds.add_argument("action", choices=("refresh", "backfill"))
    odds.add_argument("--xlsx-path")
    odds.add_argument("--url")
    odds.add_argument("--strict", action="store_true")

    model = advanced_sub.add_parser("model", help="Technical model operations.")
    model_sub = model.add_subparsers(dest="action", required=True, parser_class=FriendlyParser)
    model_train = model_sub.add_parser("train", help="Train locally without publishing.")
    _add_years(model_train)
    model_train.add_argument("--skip-prepare", action="store_true")
    model_train.add_argument("--tuning-candidates", type=int, default=100)
    model_infer = model_sub.add_parser("infer", help="Run local inference without sending.")
    _add_years(model_infer)
    model_infer.add_argument("--skip-prepare", action="store_true")
    model_infer.add_argument(
        "--auto-train",
        action="store_true",
        help="Explicitly train if model files are missing (off by default).",
    )
    model_eval = model_sub.add_parser("evaluate", help="Run honest nested-season evaluation.")
    _add_years(model_eval)
    model_eval.add_argument("--skip-prepare", action="store_true")
    model_eval.add_argument("--seasons", type=int)
    model_sub.add_parser("verify", help="Verify the active published model.")
    model_list = model_sub.add_parser("list", help="List immutable model releases.")
    model_list.add_argument("--json", action="store_true")
    model_activate = model_sub.add_parser("activate", help="Activate a validated release.")
    model_activate.add_argument("release_id")
    model_sub.add_parser("rollback", help="Reactivate the previous valid release.")

    local_run = advanced_sub.add_parser("local-run", help="Run the full prediction pipeline on this Mac.")
    local_run.add_argument("action", choices=("preview", "test", "live"))

    delivery = advanced_sub.add_parser("delivery", help="Render/test/send existing local predictions.")
    delivery.add_argument("action", choices=("preview", "test", "live"))
    delivery.add_argument("--test-email")
    delivery.add_argument("--no-llm", action="store_true")

    cloud = advanced_sub.add_parser("cloud", help="Low-level runtime state operations.")
    cloud.add_argument("action", choices=("pull-runtime", "push-runtime", "schedule", "gate"))

    site = advanced_sub.add_parser("site", help="Build or publish the static site.")
    site.add_argument("action", choices=("build", "publish"))
    return parser


def _retired_replacement(argv: list[str]) -> str | None:
    if not argv:
        return None
    first = argv[0]
    if first == "send":
        if "--test" in argv:
            return "footy-tipper advanced delivery test"
        if "--dry-run" in argv:
            return "footy-tipper advanced delivery preview"
    if first == "predict":
        if "--test" in argv:
            return "footy-tipper advanced local-run test"
        if "--skip-send" in argv or "--dry-run" in argv:
            return "footy-tipper advanced local-run preview"
    if first == "nrl-data" and len(argv) > 1 and argv[1] in {"refresh", "backfill", "validate"}:
        return f"footy-tipper advanced data nrl {argv[1]}"
    if first == "odds" and len(argv) > 1 and argv[1] in {"live", "backfill"}:
        action = "refresh" if argv[1] == "live" else "backfill"
        return f"footy-tipper advanced data odds {action}"
    if first == "site" and "--publish" in argv:
        return "footy-tipper advanced site publish"
    if first != "state":
        return RETIRED_COMMANDS.get(first)
    if len(argv) > 1:
        return STATE_ACTION_REPLACEMENTS.get(argv[1], RETIRED_COMMANDS["state"])
    return RETIRED_COMMANDS["state"]


def _guided_menu(parser: argparse.ArgumentParser, *, root: pathlib.Path, debug: bool) -> int:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        parser.print_help()
        return EXIT_OK

    print("Footy Tipper\n")
    try:
        _print_status(collect_status(root=root, offline=False))
    except Exception as exc:
        print(f"Status: some online checks were unavailable ({_redact(exc, root)})")
    print(
        "\nWhat would you like to do?\n"
        "  1. Show the current tips\n"
        "  2. Send me a test email\n"
        "  3. Refresh the tips (no email)\n"
        "  4. Update the model\n"
        "  5. Send the real round email\n"
        "  6. Check setup\n"
        "  7. Show advanced tools\n"
        "  0. Exit"
    )
    choice = input("\nChoose 0–7: ").strip()
    if choice == "0" or not choice:
        return EXIT_OK
    commands = {
        "1": ["tips", "show"],
        "2": ["tips", "test"],
        "3": ["tips", "refresh"],
        "4": ["update-model"],
        "5": ["tips", "live"],
        "6": ["setup"],
    }
    if choice == "7":
        advanced = build_parser().parse_args(["advanced", "--help"])
        return EXIT_OK if advanced is None else EXIT_OK
    if choice not in commands:
        raise InvocationError("Please choose a number from 0 to 7.")
    return run(commands[choice], root=root, inherited_debug=debug)


def run(argv=None, *, root: pathlib.Path | None = None, inherited_debug: bool = False) -> int:
    root = pathlib.Path(root or _project_root())
    # Keep the simple CLI equivalent to the old wrapper: operators should not
    # have to export every value from the ignored secrets file by hand.
    load_dotenv(dotenv_path=root / "secrets.env")
    argv = list(sys.argv[1:] if argv is None else argv)
    debug = inherited_debug or "--debug" in argv
    json_requested = "--json" in argv
    # JSON callers need clean machine output, so silence the human reporter.
    console.configure(quiet=json_requested)
    if "--debug" in argv:
        argv = [item for item in argv if item != "--debug"]

    replacement = _retired_replacement(argv)
    if replacement:
        print(
            f"That command was retired in Footy Tipper 1.0. Use: {replacement}",
            file=sys.stderr,
        )
        return EXIT_INVOCATION

    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        args.debug = debug
        if args.command is None:
            return _guided_menu(parser, root=root, debug=debug)
        if args.command == "status":
            return command_status(args, root=root)
        if args.command == "setup":
            return command_setup(args, root=root)
        if args.command == "tips":
            return command_tips(args, root=root)
        if args.command == "update-model":
            return command_update_model(args, root=root, debug=debug)
        if args.command == "advanced":
            return command_advanced(args, root=root)
        raise InvocationError(f"Unknown command: {args.command}")
    except SafetyRefusal as exc:
        error = _redact(exc, root)
        if json_requested:
            _emit_json(" ".join(argv), False, error=error, exit_code=EXIT_SAFETY)
        print(f"Stopped safely: {error}", file=sys.stderr)
        return EXIT_SAFETY
    except InvocationError as exc:
        error = _redact(exc, root)
        if json_requested:
            _emit_json(" ".join(argv), False, error=error, exit_code=EXIT_INVOCATION)
        print(f"Usage error: {error}", file=sys.stderr)
        print("Run `footy-tipper --help` to see the available commands.", file=sys.stderr)
        return EXIT_INVOCATION
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return EXIT_INTERRUPT
    except SystemExit:
        raise
    except Exception as exc:
        error = _redact(exc, root)
        if json_requested:
            _emit_json(" ".join(argv), False, error=error, exit_code=EXIT_OPERATIONAL)
        if debug:
            traceback.print_exc()
        else:
            print(f"Could not complete that operation: {error}", file=sys.stderr)
            print("Run again with --debug if you need technical details.", file=sys.stderr)
        return EXIT_OPERATIONAL
