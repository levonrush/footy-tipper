"""Resumable, local-authoritative model training and publication.

``footy-tipper update-model`` is intentionally a staged transaction.  Long
work happens under the ignored ``.footy-tipper/`` directory; the live model
directory and Drive pointer move only after validation and publication have
completed.  A journal makes interruption recovery explicit and predictable.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import pathlib
import signal
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time

from pipeline.ops import state_sync


SCHEMA_VERSION = 1
DEFAULT_TUNING_CANDIDATES = 100
WORK_DIR_NAME = ".footy-tipper"
JOURNAL_FILE = "model-update.json"
HEARTBEAT_SECONDS = 30
LOCAL_BACKUPS_TO_KEEP = 4
LOCAL_UPDATES_TO_KEEP = 3
REDACT_KEYS = ("PASSWORD", "TOKEN", "SECRET", "KEY", "AUTH")
STAGE_ORDER = (
    "preflight",
    "backup",
    "data_prepared",
    "trained",
    "validated",
    "published",
    "container_checked",
    "pointer_activated",
    "activated",
    "refreshed",
)


class ProductionCodeChanged(RuntimeError):
    """The release can no longer be proven compatible with production code."""


class ModelUpdateAlreadyRunning(RuntimeError):
    """Another process owns the local model-update transaction."""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _log(message, *, json_output=False):
    if not json_output:
        print(f"[model-update] {message}", flush=True)


def _atomic_json(path, payload) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha(root) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _release_id(git_sha) -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}-{git_sha[:10]}"


def _work_root(root) -> pathlib.Path:
    return pathlib.Path(root) / WORK_DIR_NAME


def _journal_path(root) -> pathlib.Path:
    return _work_root(root) / JOURNAL_FILE


def _load_journal(root):
    path = _journal_path(root)
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Update journal is unreadable at {path}; move it aside before retrying"
        ) from exc
    if not isinstance(value, dict) or int(value.get("schema_version", 0)) != SCHEMA_VERSION:
        raise RuntimeError(f"Update journal at {path} has an unsupported schema")
    return value


def _save_journal(root, journal) -> None:
    journal["updated_at_utc"] = _utc_now()
    _atomic_json(_journal_path(root), journal)


def _set_stage(root, journal, stage, **extra) -> None:
    journal.setdefault("stages", {})[stage] = {
        "completed": True,
        "completed_at_utc": _utc_now(),
        **extra,
    }
    journal["current_stage"] = stage
    _save_journal(root, journal)


def _stage_done(journal, stage) -> bool:
    return bool(journal.get("stages", {}).get(stage, {}).get("completed"))


def _rewind_from(root, journal, stage, reason) -> None:
    """Forget a stage and everything after it when its durable evidence vanished."""
    start = STAGE_ORDER.index(stage)
    stages = journal.setdefault("stages", {})
    for name in STAGE_ORDER[start:]:
        stages.pop(name, None)
    journal["status"] = "running"
    journal["current_stage"] = STAGE_ORDER[start - 1] if start else "created"
    journal["last_error"] = f"Resume evidence was rewound: {reason}"
    _save_journal(root, journal)


def _download_release_evidence(root, release_id, destination=None) -> dict:
    """Re-download one immutable release and optionally restore its staged files."""
    root = pathlib.Path(root)
    service = state_sync.drive_service(root / "service-account-token.json")
    state_id = state_sync._existing_state_folder(service, root)
    with tempfile.TemporaryDirectory(prefix=".release-evidence-", dir=root) as tmp:
        staged, _, metadata, _ = state_sync._download_release(
            service, state_id, release_id, pathlib.Path(tmp) / "release"
        )
        if destination is not None:
            destination = pathlib.Path(destination)
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(staged, destination)
        return metadata


def _hosted_check_succeeded(root, run_id) -> bool:
    if not run_id:
        return False
    result = subprocess.run(
        ["gh", "run", "view", str(run_id), "--json", "status,conclusion"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return False
    try:
        body = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return False
    return body.get("status") == "completed" and body.get("conclusion") == "success"


def _revalidate_resume_evidence(root, journal, models_dir) -> None:
    """Make journal resumption prove files, Drive objects, checks, and pointer state."""
    release_id = journal["release_id"]
    if _stage_done(journal, "trained"):
        try:
            state_sync._validate_release_directory(models_dir, release_id=release_id)
        except (OSError, RuntimeError, ValueError):
            if _stage_done(journal, "published"):
                _download_release_evidence(root, release_id, destination=models_dir)
            else:
                _rewind_from(
                    root,
                    journal,
                    "trained",
                    "the staged model receipt/files are missing or invalid",
                )

    if _stage_done(journal, "published"):
        try:
            metadata = _download_release_evidence(root, release_id)
            expected = journal["stages"]["published"].get("archive_sha256")
            if expected and metadata.get("archive_sha256") != expected:
                raise ValueError("published archive hash changed")
        except (OSError, RuntimeError, ValueError):
            if _stage_done(journal, "pointer_activated"):
                raise RuntimeError(
                    "The active model release no longer has valid immutable Drive evidence. "
                    "Do not continue; verify Drive history and roll back explicitly."
                )
            _rewind_from(
                root,
                journal,
                "published",
                "the immutable Drive release is missing or invalid",
            )

    if _stage_done(journal, "container_checked"):
        check_run = journal["stages"]["container_checked"].get("run_id")
        if not _hosted_check_succeeded(root, check_run):
            _rewind_from(
                root,
                journal,
                "container_checked",
                "the successful hosted validation run cannot be proved",
            )

    if _stage_done(journal, "pointer_activated"):
        pointer = state_sync.get_model_pointer(root)
        if pointer.get("release_id") != release_id:
            raise ProductionCodeChanged(
                "Production was deliberately moved away from this candidate after activation. "
                "The old journal will not reactivate it; start a fresh model update if needed"
            )

    if _stage_done(journal, "activated"):
        try:
            state_sync._validate_release_directory(root / "models", release_id=release_id)
        except (OSError, RuntimeError, ValueError):
            _rewind_from(
                root,
                journal,
                "activated",
                "the local installed model does not match the active release",
            )


def _redacted_environment(env) -> dict:
    safe = {}
    for key, value in env.items():
        if any(fragment in key.upper() for fragment in REDACT_KEYS):
            safe[key] = "[redacted]"
        elif key.startswith("FOOTY_TIPPER_"):
            safe[key] = value
    return safe


def _acquire_update_lock(root):
    """Hold one non-blocking lock for the entire local publication transaction."""
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - production updates run on macOS
        raise RuntimeError("This platform cannot provide the required model-update lock") from exc

    path = _work_root(root) / "model-update.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise ModelUpdateAlreadyRunning(
            "Another `footy-tipper update-model` process is already running. "
            "Use `footy-tipper status` in this terminal and let the first process finish."
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} acquired_at_utc={_utc_now()}\n")
    handle.flush()
    return handle


def _release_update_lock(handle) -> None:
    if handle is None:
        return
    try:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _stop_process_group(process) -> None:
    """Terminate and reap a command plus children before interruption escapes."""
    if process.poll() is not None:
        process.wait()
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        process.terminate()
    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.kill()
    process.wait()


def _run_logged(command, *, root, env, log_path, label, prevent_sleep=False) -> None:
    command = [str(part) for part in command]
    if prevent_sleep and shutil.which("caffeinate"):
        command = ["caffeinate", "-dimsu", *command]
    log_path = pathlib.Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n[{_utc_now()}] {label}: {' '.join(command)}\n")
        log.write(f"environment: {json.dumps(_redacted_environment(env), sort_keys=True)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=str(root),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            last_heartbeat = time.monotonic()
            while process.poll() is None:
                time.sleep(1)
                if time.monotonic() - last_heartbeat >= HEARTBEAT_SECONDS:
                    print(
                        f"[model-update] {label} is still running; log: {log_path}",
                        file=sys.stderr,
                        flush=True,
                    )
                    last_heartbeat = time.monotonic()
        except BaseException:
            _stop_process_group(process)
            raise
        if process.returncode:
            try:
                tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
            except OSError:
                tail = []
            if tail:
                print("\n".join(tail), file=sys.stderr)
            raise RuntimeError(
                f"{label} failed with exit code {process.returncode}; see {log_path}"
            )


def _base_environment(root, tuning_candidates) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env.setdefault("R_LIBS_USER", os.path.expanduser("~/R/library"))
    env["FOOTY_TIPPER_TUNE_ITER"] = str(int(tuning_candidates))
    return env


def _require_command(name, explanation) -> None:
    if not shutil.which(name):
        raise RuntimeError(f"{name} is required {explanation}")


def _preflight(root, tuning_candidates) -> dict:
    root = pathlib.Path(root)
    if int(tuning_candidates) < 1:
        raise ValueError("tuning candidate count must be at least 1")
    if os.getenv("CONDA_DEFAULT_ENV") != "footy-tipper":
        raise RuntimeError(
            "The footy-tipper Conda environment is not active. Run "
            "`conda activate footy-tipper`, then retry"
        )
    missing_packages = []
    for package in ("lightgbm", "scikit-learn", "pandas", "dill", "scikit-optimize"):
        try:
            importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            missing_packages.append(package)
    if missing_packages:
        raise RuntimeError(
            "The active Python environment is incomplete (missing: "
            + ", ".join(missing_packages)
            + "). Run `conda env update -f environment.yml --prune`"
        )
    for path, label in (
        (root / "service-account-token.json", "Google service-account token"),
        (root / "secrets.env", "secrets.env"),
    ):
        if not path.is_file():
            raise RuntimeError(f"Missing {label}: {path}")
    state_sync._folder_id(root)
    _require_command("git", "to record model provenance")
    _require_command("Rscript", "to prepare training data")
    _require_command("gh", "to request the no-email production refresh")
    if sys.platform == "darwin":
        _require_command("caffeinate", "to keep the Mac awake during training")
    subprocess.run(
        ["gh", "auth", "status"], cwd=root, check=True, capture_output=True, text=True
    )
    free_gib = shutil.disk_usage(root).free / (1024 ** 3)
    if free_gib < 5:
        raise RuntimeError(
            f"At least 5 GB of free disk is required; only {free_gib:.1f} GB is available"
        )
    active_pointer = state_sync.get_model_pointer(root)
    workflows = subprocess.run(
        ["gh", "workflow", "list", "--all", "--json", "path,state"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    workflow_states = {
        item.get("path"): str(item.get("state", "")).lower()
        for item in json.loads(workflows.stdout or "[]")
    }
    for path, label in (
        (".github/workflows/predict.yml", "Predict"),
        (".github/workflows/model-check.yml", "Model release check"),
    ):
        if workflow_states.get(path) != "active":
            raise RuntimeError(
                f"The {label} workflow is not active. Enable {path.rsplit('/', 1)[-1]} "
                "before updating the model"
            )
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if branch != "main":
        raise RuntimeError(
            f"Model publication must start from main, but the current branch is {branch or 'detached'}"
        )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(
            "The repository has uncommitted changes. Commit or stash them before updating the model"
        )
    subprocess.run(
        ["git", "fetch", "--quiet", "origin", "main"], cwd=root, check=True
    )
    local_sha = _git_sha(root)
    remote_sha = subprocess.run(
        ["git", "rev-parse", "origin/main"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if local_sha != remote_sha:
        raise RuntimeError(
            "Local main is not exactly up to date with origin/main. Run `git pull --rebase`, then retry"
        )
    return {
        "git_sha": local_sha,
        "branch": branch,
        "active_release_id": active_pointer["release_id"],
        "free_disk_gib": round(free_gib, 1),
        "tuning_candidates": int(tuning_candidates),
    }


def _seed_training_db_if_missing(root, json_output=False) -> pathlib.Path:
    db_path = pathlib.Path(root) / "data" / "footy-tipper-db.sqlite"
    if db_path.exists():
        state_sync._validate_sqlite(db_path)
        return db_path
    _log("No local training database found; seeding one validated copy from Drive.", json_output=json_output)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        staged = pathlib.Path(tmp) / "seed.sqlite"
        state_sync.download_runtime_db(staged, root=root)
        os.replace(staged, db_path)
    return db_path


def _backup_training_db(root, db_path, release_id) -> pathlib.Path:
    backup_dir = _work_root(root) / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / f"before-{release_id}.sqlite"
    if not backup.exists():
        state_sync._snapshot_sqlite(db_path, backup)
        state_sync._validate_sqlite(backup)
    backups = sorted(
        backup_dir.glob("before-*.sqlite"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for stale in backups[LOCAL_BACKUPS_TO_KEEP:]:
        stale.unlink()
    return backup


def _prune_update_directories(root, current_release_id) -> None:
    updates_dir = _work_root(root) / "model-updates"
    if not updates_dir.exists():
        return
    directories = sorted(
        (path for path in updates_dir.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    keep = {path.name for path in directories[:LOCAL_UPDATES_TO_KEEP]}
    keep.add(str(current_release_id))
    for stale in directories:
        if stale.name not in keep:
            shutil.rmtree(stale)


def _prepare_training_data(root, env, log_path) -> None:
    env = dict(env)
    env["FOOTY_TIPPER_PREP_MODE"] = "train"
    # Reuse the established bootstrap detectors, but keep command execution in
    # this journalled/logged transaction. Historical ingestion runs only when
    # the local-authoritative DB has not already recorded it.
    from pipeline import cli as pipeline_cli

    commands = []
    if pipeline_cli._lineups_enabled(env):
        if (
            pipeline_cli._to_bool(env.get("FOOTY_TIPPER_LINEUPS_AUTO_BACKFILL"), True)
            and not pipeline_cli._lineup_backfill_bootstrapped(root, env)
        ):
            backfill_env = dict(env)
            backfill_env["FOOTY_TIPPER_LINEUPS_MODE"] = "backfill"
            backfill_env["FOOTY_TIPPER_LINEUPS_MAX_ARTICLES"] = str(
                pipeline_cli._env_int(
                    env,
                    "FOOTY_TIPPER_LINEUPS_BACKFILL_MAX_ARTICLES",
                    pipeline_cli.DEFAULT_LINEUP_BACKFILL_MAX_ARTICLES,
                )
            )
            _run_logged(
                [sys.executable, root / "pipeline" / "lineups.py"],
                root=root,
                env=backfill_env,
                log_path=log_path,
                label="Bootstrapping historical team lists",
            )
        commands.append(
            ([sys.executable, root / "pipeline" / "lineups.py"], "Refreshing team lists")
        )

    python_feeds = pipeline_cli._feed_source(env) != "feed"
    if python_feeds and pipeline_cli._nrl_data_enabled(env):
        if pipeline_cli._to_bool(
            env.get("FOOTY_TIPPER_NRL_DATA_AUTO_BACKFILL"), True
        ):
            if not pipeline_cli._nrl_backfill_bootstrapped(root):
                commands.append(
                    ([sys.executable, root / "pipeline" / "nrl_data.py", "backfill"],
                     "Bootstrapping historical NRL data")
                )
            if not pipeline_cli._odds_backfill_bootstrapped(root):
                commands.append(
                    ([sys.executable, root / "pipeline" / "odds.py", "backfill"],
                     "Bootstrapping historical odds")
                )
        commands.extend(
            [
                ([sys.executable, root / "pipeline" / "nrl_data.py", "refresh"],
                 "Refreshing NRL data"),
                ([sys.executable, root / "pipeline" / "odds.py", "live"],
                 "Refreshing odds"),
            ]
        )
    commands.append((["Rscript", root / "pipeline" / "data-prep.R"], "Preparing training data"))
    for command, label in commands:
        _run_logged(command, root=root, env=env, log_path=log_path, label=label)


def _runtime_versions() -> dict:
    result = {"python": sys.version.split()[0]}
    for package in ("lightgbm", "scikit-learn", "pandas", "numpy", "dill", "scikit-optimize"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _database_summary(db_path) -> dict:
    con = sqlite3.connect(str(db_path))
    try:
        row = con.execute(
            """
            SELECT COUNT(*), MIN(CAST(competition_year AS INTEGER)),
                   MAX(CAST(competition_year AS INTEGER))
            FROM footy_tipping_data
            WHERE game_state_name = 'Final'
            """
        ).fetchone()
    finally:
        con.close()
    return {
        "training_rows": int((row or (0, None, None))[0] or 0),
        "training_year_min": row[1] if row else None,
        "training_year_max": row[2] if row else None,
    }


def _augment_manifest(models_dir, *, release_id, git_sha, tuning_candidates) -> None:
    path = pathlib.Path(models_dir) / "model_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["release"] = {
        "schema_version": SCHEMA_VERSION,
        "release_id": release_id,
        "git_sha": git_sha,
        "tuning_candidates": int(tuning_candidates),
        "trained_at_utc": _utc_now(),
        "runtime_versions": _runtime_versions(),
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_receipt(models_dir, db_path, *, release_id, git_sha, tuning_candidates, source) -> dict:
    models_dir = pathlib.Path(models_dir)
    state_sync._validate_model_artifacts(models_dir)
    artifacts = {}
    for path in sorted(models_dir.iterdir()):
        if path.is_file() and path.name not in {".gitkeep", state_sync.TRAINING_RECEIPT_FILE}:
            artifacts[path.name] = {"size": path.stat().st_size, "sha256": _sha256(path)}
    return {
        "schema_version": SCHEMA_VERSION,
        "release_id": release_id,
        "source": source,
        "git_sha": git_sha,
        "created_at_utc": _utc_now(),
        "tuning_candidates": int(tuning_candidates),
        "database": {
            **_database_summary(db_path),
            "sha256": _sha256(db_path),
            "size": pathlib.Path(db_path).stat().st_size,
        },
        "runtime_versions": _runtime_versions(),
        "artifacts": artifacts,
    }


def _write_receipt_last(models_dir, receipt) -> pathlib.Path:
    path = pathlib.Path(models_dir) / state_sync.TRAINING_RECEIPT_FILE
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    state_sync._validate_release_directory(models_dir, release_id=receipt["release_id"])
    return path


def _validate_candidate(root, models_dir, db_path, env, log_path) -> None:
    state_sync._validate_release_directory(models_dir)
    candidate_db = pathlib.Path(models_dir).parent / "validation.sqlite"
    state_sync._snapshot_sqlite(db_path, candidate_db)
    validation_env = dict(env)
    validation_env["FOOTY_TIPPER_MODELS_DIR"] = str(models_dir)
    validation_env["FOOTY_TIPPER_DB_PATH"] = str(candidate_db)
    _run_logged(
        [sys.executable, root / "pipeline" / "inference.py"],
        root=root,
        env=validation_env,
        log_path=log_path,
        label="Validating candidate inference",
    )
    state_sync._validate_sqlite(candidate_db)
    candidate_db.unlink(missing_ok=True)


def _container_check(root, release_id, env, log_path) -> int:
    """Dispatch and wait for validation in the actual production image."""
    before = subprocess.run(
        [
            "gh", "run", "list", "--workflow", "model-check.yml", "--event",
            "workflow_dispatch", "--limit", "30", "--json", "databaseId",
        ],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    before_ids = {
        int(item["databaseId"]) for item in json.loads(before.stdout or "[]")
    }
    _run_logged(
        [
            "gh", "workflow", "run", "model-check.yml", "--ref", "main",
            "-f", f"release={release_id}",
        ],
        root=root,
        env=env,
        log_path=log_path,
        label="Requesting the production-container model check",
    )
    run_id = None
    deadline = time.time() + 90
    while time.time() < deadline and run_id is None:
        result = subprocess.run(
            [
                "gh", "run", "list", "--workflow", "model-check.yml", "--event",
                "workflow_dispatch", "--limit", "30", "--json",
                "databaseId,displayTitle",
            ],
            cwd=root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        matching = [
            item
            for item in json.loads(result.stdout or "[]")
            if int(item["databaseId"]) not in before_ids
            and release_id in str(item.get("displayTitle", ""))
        ]
        if matching:
            run_id = str(max(int(item["databaseId"]) for item in matching))
            break
        time.sleep(2)
    if run_id is None:
        raise RuntimeError(
            "GitHub accepted the model check, but its run could not be identified"
        )
    _run_logged(
        ["gh", "run", "watch", run_id, "--exit-status"],
        root=root,
        env=env,
        log_path=log_path,
        label=f"Waiting for production-container check {run_id}",
    )
    return int(run_id)


def _verify_production_code(root, journal) -> None:
    """Stop before activation if production code changed during a long run.

    Actions-generated static-site commits are harmless and expected. Any other
    remote change means the candidate must be trained again against the newer
    code instead of being activated on hope.
    """
    if _git_sha(root) != journal["git_sha"]:
        raise ProductionCodeChanged(
            "Local code changed during this model update. Pull main and start a fresh update"
        )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ProductionCodeChanged(
            "Uncommitted repository files appeared during this model update. "
            "Commit or stash them, pull main, and start a fresh update"
        )
    subprocess.run(
        ["git", "fetch", "--quiet", "origin", "main"], cwd=root, check=True
    )
    remote_sha = subprocess.run(
        ["git", "rev-parse", "origin/main"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if remote_sha == journal["git_sha"]:
        return
    changed = subprocess.run(
        ["git", "diff", "--name-only", f"{journal['git_sha']}..{remote_sha}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    material = [path for path in changed if not path.startswith("docs/site/")]
    if material:
        preview = ", ".join(material[:5])
        raise ProductionCodeChanged(
            "Production code changed while training "
            f"({preview}{'…' if len(material) > 5 else ''}). "
            "Pull main and run a fresh model update"
        )


def _install_local_models(root, models_dir) -> None:
    root = pathlib.Path(root)
    with tempfile.TemporaryDirectory(prefix=".model-install-", dir=root) as tmp:
        tmp = pathlib.Path(tmp)
        staged = tmp / "models-staged"
        shutil.copytree(models_dir, staged)
        (staged / ".gitkeep").touch(exist_ok=True)
        state_sync._replace_models_dir(staged, root / "models", tmp)


def _request_refresh(root, env, log_path) -> int:
    before = time.time()
    _run_logged(
        ["gh", "workflow", "run", "predict.yml", "--ref", "main", "-f", "mode=refresh"],
        root=root,
        env=env,
        log_path=log_path,
        label="Requesting a no-email production refresh",
    )
    run_id = None
    deadline = time.time() + 90
    while time.time() < deadline and run_id is None:
        result = subprocess.run(
            [
                "gh", "run", "list", "--workflow", "predict.yml", "--event",
                "workflow_dispatch", "--limit", "10", "--json",
                "databaseId,createdAt,displayTitle",
            ],
            cwd=root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        for item in json.loads(result.stdout):
            created = dt.datetime.fromisoformat(item["createdAt"].replace("Z", "+00:00")).timestamp()
            if created >= before - 2 and "refresh" in str(item.get("displayTitle", "")).lower():
                run_id = str(item["databaseId"])
                break
        if run_id is None:
            time.sleep(2)
    if run_id is None:
        raise RuntimeError("GitHub accepted the refresh but its run could not be identified")
    _run_logged(
        ["gh", "run", "watch", run_id, "--exit-status"],
        root=root,
        env=env,
        log_path=log_path,
        label=f"Waiting for refresh run {run_id}",
    )
    return int(run_id)


def _new_or_resumable_journal(root, tuning_candidates):
    existing = _load_journal(root)
    if existing and existing.get("status") not in {"complete", "cancelled"}:
        # A preflight-only failure may be fixed by committing/pulling, which
        # changes the provenance SHA. No expensive work exists yet, so start a
        # fresh transaction instead of resuming under the stale SHA.
        if existing.get("stages") or existing.get("git_sha") == _git_sha(root):
            return existing, True
    git_sha = _git_sha(root)
    release_id = _release_id(git_sha)
    journal = {
        "schema_version": SCHEMA_VERSION,
        "release_id": release_id,
        "git_sha": git_sha,
        "tuning_candidates": int(tuning_candidates),
        "status": "running",
        "current_stage": "created",
        "created_at_utc": _utc_now(),
        "stages": {},
    }
    _save_journal(root, journal)
    return journal, False


def update_model(root, *, json_output=False, debug=False) -> int:
    """Run or resume the safe one-command model update."""
    root = pathlib.Path(root).resolve()
    tuning_candidates = int(os.getenv("FOOTY_TIPPER_TUNE_ITER", DEFAULT_TUNING_CANDIDATES))
    journal = None
    lock_handle = None
    try:
        lock_handle = _acquire_update_lock(root)
        journal, resumed = _new_or_resumable_journal(root, tuning_candidates)
        release_id = journal["release_id"]
        tuning_candidates = int(journal["tuning_candidates"])
        update_dir = _work_root(root) / "model-updates" / release_id
        models_dir = update_dir / "models"
        log_path = update_dir / "update.log"
        update_dir.mkdir(parents=True, exist_ok=True)
        _log(
            f"{'Resuming' if resumed else 'Starting'} release {release_id}. "
            f"You can safely rerun this command after an interruption.",
            json_output=json_output,
        )

        preflight = _preflight(root, tuning_candidates)
        if not _stage_done(journal, "preflight"):
            journal["git_sha"] = preflight["git_sha"]
            _set_stage(root, journal, "preflight", **preflight)

        env = _base_environment(root, tuning_candidates)
        db_path = _seed_training_db_if_missing(root, json_output=json_output)
        _revalidate_resume_evidence(root, journal, models_dir)
        _verify_production_code(root, journal)

        if not _stage_done(journal, "backup"):
            backup = _backup_training_db(root, db_path, release_id)
            _set_stage(root, journal, "backup", path=str(backup))

        if not _stage_done(journal, "data_prepared"):
            _log("Refreshing and preparing the local training data.", json_output=json_output)
            _prepare_training_data(root, env, log_path)
            _set_stage(root, journal, "data_prepared")

        if not _stage_done(journal, "trained"):
            if models_dir.exists():
                shutil.rmtree(models_dir)
            models_dir.mkdir(parents=True)
            train_env = dict(env)
            train_env["FOOTY_TIPPER_MODELS_DIR"] = str(models_dir)
            train_env["FOOTY_TIPPER_DB_PATH"] = str(db_path)
            _log(
                f"Training {tuning_candidates} Bayesian candidates on this computer. "
                "caffeinate will keep it awake while the trainer runs.",
                json_output=json_output,
            )
            _run_logged(
                [sys.executable, root / "pipeline" / "train.py"],
                root=root,
                env=train_env,
                log_path=log_path,
                label="Training candidate models",
                prevent_sleep=True,
            )
            _augment_manifest(
                models_dir,
                release_id=release_id,
                git_sha=journal["git_sha"],
                tuning_candidates=tuning_candidates,
            )
            receipt = _build_receipt(
                models_dir,
                db_path,
                release_id=release_id,
                git_sha=journal["git_sha"],
                tuning_candidates=tuning_candidates,
                source="local_training",
            )
            _write_receipt_last(models_dir, receipt)
            _set_stage(root, journal, "trained", receipt=receipt)

        if not _stage_done(journal, "validated"):
            _log("Validating the staged models against a throwaway DB copy.", json_output=json_output)
            _validate_candidate(root, models_dir, db_path, env, log_path)
            _set_stage(root, journal, "validated")

        if not _stage_done(journal, "published"):
            metadata = state_sync.publish_model_release(
                root,
                models_dir,
                journal["stages"]["trained"]["receipt"],
                release_id,
            )
            _set_stage(
                root,
                journal,
                "published",
                archive_sha256=metadata["archive_sha256"],
            )

        if not _stage_done(journal, "container_checked"):
            _verify_production_code(root, journal)
            check_run_id = _container_check(root, release_id, env, log_path)
            _set_stage(
                root, journal, "container_checked", run_id=check_run_id
            )

        if not _stage_done(journal, "pointer_activated"):
            _verify_production_code(root, journal)
            pointer = state_sync.activate_model_release(root, release_id)
            _set_stage(
                root,
                journal,
                "pointer_activated",
                previous_release_id=pointer.get("previous_release_id"),
            )
        if not _stage_done(journal, "activated"):
            _install_local_models(root, models_dir)
            _set_stage(
                root,
                journal,
                "activated",
                previous_release_id=journal["stages"]["pointer_activated"].get(
                    "previous_release_id"
                ),
            )

        if not _stage_done(journal, "refreshed"):
            refresh_run_id = _request_refresh(root, env, log_path)
            _set_stage(root, journal, "refreshed", run_id=refresh_run_id)

        active_pointer = state_sync.get_model_pointer(root)
        if active_pointer.get("release_id") != release_id:
            raise ProductionCodeChanged(
                "Production no longer points at the candidate, so this update cannot "
                "be reported complete"
            )

        journal["status"] = "complete"
        journal["current_stage"] = "complete"
        journal["completed_at_utc"] = _utc_now()
        _save_journal(root, journal)
        _prune_update_directories(root, release_id)
        result = {
            "schema_version": SCHEMA_VERSION,
            "ok": True,
            "release_id": release_id,
            "status": "complete",
            "log_path": str(log_path),
        }
        if json_output:
            print(json.dumps(result, sort_keys=True))
        else:
            _log(
                f"Model {release_id} is active and the no-email refresh succeeded.",
                json_output=False,
            )
        return 0
    except KeyboardInterrupt:
        if journal is not None:
            journal["status"] = "interrupted"
            journal["last_error"] = "Interrupted by operator"
            _save_journal(root, journal)
        if json_output:
            print(json.dumps({"schema_version": 1, "ok": False, "error": "interrupted"}))
        else:
            print("Model update interrupted. Run `footy-tipper update-model` again to resume.", file=sys.stderr)
        return 130
    except ProductionCodeChanged as exc:
        if journal is not None:
            journal["status"] = "cancelled"
            journal["last_error"] = str(exc)
            _save_journal(root, journal)
        if json_output:
            print(
                json.dumps(
                    {"schema_version": 1, "ok": False, "error": str(exc)},
                    sort_keys=True,
                )
            )
        else:
            print(f"Model update stopped safely: {exc}", file=sys.stderr)
            print(
                "No new pointer was activated. Run `git pull --rebase`, then "
                "run `footy-tipper update-model` to start a fresh release.",
                file=sys.stderr,
            )
        return 1
    except Exception as exc:
        if journal is not None:
            journal["status"] = "failed"
            journal["last_error"] = str(exc)
            _save_journal(root, journal)
        if debug:
            raise
        if json_output:
            print(
                json.dumps(
                    {"schema_version": 1, "ok": False, "error": str(exc)},
                    sort_keys=True,
                )
            )
        else:
            print(f"Model update stopped safely: {exc}", file=sys.stderr)
            if journal is not None and _stage_done(journal, "pointer_activated"):
                print(
                    "The new model is active, but the no-email refresh did not finish. "
                    "Run the same command to resume at the refresh step.",
                    file=sys.stderr,
                )
            else:
                print(
                    "Nothing new was activated. Fix the problem, then run the same command to resume.",
                    file=sys.stderr,
                )
        return 1
    finally:
        _release_update_lock(lock_handle)


def verify_active(root) -> int:
    try:
        pointer = state_sync.get_model_pointer(root)
        return state_sync.check_model_release(root, pointer["release_id"])
    except Exception as exc:
        print(f"Active model verification failed: {exc}", file=sys.stderr)
        return 1


def list_releases(root, *, json_output=False) -> int:
    try:
        releases = state_sync.list_model_releases(root)
        pointer = state_sync.get_model_pointer(root)
    except Exception as exc:
        print(f"Could not list model releases: {exc}", file=sys.stderr)
        return 1
    if json_output:
        print(json.dumps({"schema_version": 1, "active": pointer, "releases": releases}))
    else:
        for release in releases:
            marker = "*" if release.get("release_id") == pointer.get("release_id") else " "
            print(f"{marker} {release.get('release_id')}  {release.get('created_at_utc', 'unknown date')}")
    return 0


def _hosted_validate_for_activation(root, release_id) -> bool:
    """Require local and production-image proof immediately before pointer movement."""
    root = pathlib.Path(root).resolve()
    if state_sync.check_model_release(root, release_id) != 0:
        return False
    env = _base_environment(root, DEFAULT_TUNING_CANDIDATES)
    log_path = _work_root(root) / "activation-checks" / f"{release_id}.log"
    _container_check(root, release_id, env, log_path)
    return True


def activate_release(root, release_id) -> int:
    try:
        if not _hosted_validate_for_activation(root, release_id):
            return 1
        state_sync.activate_model_release(
            root, release_id, repair_broken_pointer=True
        )
    except Exception as exc:
        print(f"Could not activate model release: {exc}", file=sys.stderr)
        return 1
    return 0


def rollback(root) -> int:
    try:
        current = state_sync.get_model_pointer(root)
        previous = current.get("previous_release_id")
        if not previous:
            raise RuntimeError(
                "the active model pointer has no previous release to roll back to"
            )
        if not _hosted_validate_for_activation(root, previous):
            return 1
        pointer = state_sync.activate_model_release(root, previous)
        print(f"Rolled back to model release {pointer['release_id']}.")
    except Exception as exc:
        print(f"Could not roll back model release: {exc}", file=sys.stderr)
        return 1
    return 0


def import_legacy_models(root, *, activate=True, container_check=True) -> dict:
    """One-time rollout helper: wrap today's validated local models as release 1."""
    root = pathlib.Path(root).resolve()
    git_sha = _git_sha(root)
    release_id = f"legacy-{_release_id(git_sha)}"
    source_models = root / "models"
    db_path = root / "data" / "footy-tipper-db.sqlite"
    state_sync._validate_model_artifacts(source_models)
    state_sync._validate_sqlite(db_path)
    with tempfile.TemporaryDirectory(prefix=".legacy-release-", dir=root) as tmp:
        staged = pathlib.Path(tmp) / "models"
        shutil.copytree(source_models, staged)
        _augment_manifest(
            staged,
            release_id=release_id,
            git_sha=git_sha,
            tuning_candidates=DEFAULT_TUNING_CANDIDATES,
        )
        receipt = _build_receipt(
            staged,
            db_path,
            release_id=release_id,
            git_sha=git_sha,
            tuning_candidates=DEFAULT_TUNING_CANDIDATES,
            source="legacy_rollout_import",
        )
        _write_receipt_last(staged, receipt)
        metadata = state_sync.publish_model_release(root, staged, receipt, release_id)
        if container_check:
            env = _base_environment(root, DEFAULT_TUNING_CANDIDATES)
            log_path = _work_root(root) / "model-updates" / release_id / "import.log"
            _container_check(root, release_id, env, log_path)
        pointer = state_sync.activate_model_release(root, release_id) if activate else None
    return {"metadata": metadata, "pointer": pointer, "release_id": release_id}


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m pipeline.ops.model_release")
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("import-legacy")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--release")
    args = parser.parse_args(argv)
    root = pathlib.Path(__file__).resolve().parents[2]
    if args.action == "import-legacy":
        result = import_legacy_models(root)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.release:
        return state_sync.check_model_release(root, args.release)
    return verify_active(root)


if __name__ == "__main__":
    raise SystemExit(main())
