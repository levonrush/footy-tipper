"""Publish mutable runtime state and consume immutable model releases.

GitHub Actions pulls the current SQLite runtime DB and the model release named
by ``model-current.json``.  It may push only the DB and ``schedule.json`` after
a run.  Local model publication uses the separate create-only release helpers;
runtime sync can never upload or select a model.

Deliberately imports only the stdlib plus google-api-python-client/google-auth,
so the gate job can run it after installing just those two packages (no pandas
import chain like pipeline.common.use_predictions.distribution).

Usage: python -m pipeline.ops.state_sync {runtime-push|runtime-pull|gate|schedule}
"""

import argparse
import datetime as dt
import gzip
import hashlib
import json
import math
import os
import pathlib
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import time
from zoneinfo import ZoneInfo

try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
except Exception:  # pragma: no cover - exercised only without google deps
    service_account = None
    build = None
    MediaFileUpload = None
    MediaIoBaseDownload = None

STATE_FOLDER_NAME = "state"
DB_ARCHIVE = "footy-tipper-db-latest.sqlite.gz"
MODELS_ARCHIVE = "models-latest.tar.gz"
SCHEDULE_FILE = "schedule.json"
MODEL_POINTER_FILE = "model-current.json"
MODEL_RELEASES_FOLDER = "model-releases"
TRAINING_RECEIPT_FILE = "training-receipt.json"
REQUIRED_MODEL_FILES = ("home_model.pkl", "away_model.pkl", "model_manifest.json")
PROBABILITY_STACK_V3_FILES = (
    "binary_model.pkl",
    "stacker.pkl",
    "win_prob_calibrator.pkl",
    "stacker_no_market.pkl",
    "win_prob_calibrator_no_market.pkl",
)

SYDNEY_TIMEZONE = ZoneInfo("Australia/Sydney")
SEND_HOUR_LOCAL = 11
GRACE_HOURS = 12
STALE_DAYS = 8
SCHEDULE_ROUND_LIMIT = 8


def _project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _log(message: str) -> None:
    print(f"[state-sync] {message}", file=sys.stderr, flush=True)


def load_env_file(path) -> dict:
    """Minimal KEY=VALUE parser so the gate job does not need python-dotenv."""
    values = {}
    path = pathlib.Path(path)
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        values[key.strip()] = value
    return values


def _folder_id(root) -> str:
    folder_id = os.getenv("FOLDER_ID") or load_env_file(root / "secrets.env").get("FOLDER_ID")
    if not folder_id:
        raise RuntimeError("FOLDER_ID is not set (env or secrets.env); cannot locate Drive state.")
    return folder_id


def drive_service(json_path):
    if service_account is None or build is None:
        raise RuntimeError("google-api-python-client and google-auth are required for state sync.")
    json_path = pathlib.Path(json_path)
    if not json_path.exists():
        raise RuntimeError(f"Service account file not found: {json_path}")
    creds = service_account.Credentials.from_service_account_file(str(json_path))
    return build("drive", "v3", credentials=creds)


def get_or_create_folder(service, folder_name, parent_folder_id) -> str:
    query = (
        f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' "
        f"and '{parent_folder_id}' in parents and trashed=false"
    )
    results = service.files().list(q=query, spaces="drive", fields="files(id, name)").execute()
    folders = results.get("files", [])
    if folders:
        return folders[0]["id"]
    file_metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_folder_id],
    }
    folder = service.files().create(body=file_metadata, fields="id").execute()
    return folder["id"]


def find_folder_id(service, folder_name, parent_folder_id):
    query = (
        f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' "
        f"and '{parent_folder_id}' in parents and trashed=false"
    )
    results = service.files().list(
        q=query, spaces="drive", fields="files(id, name)"
    ).execute()
    folders = results.get("files", [])
    return folders[0]["id"] if folders else None


def find_file_id(service, folder_id, name):
    query = f"name='{name}' and '{folder_id}' in parents and trashed=false"
    results = service.files().list(q=query, spaces="drive", fields="files(id, name)").execute()
    files = results.get("files", [])
    return files[0]["id"] if files else None


def list_files(service, folder_id):
    """Return non-trashed files in a Drive folder with the fields we use."""
    query = f"'{folder_id}' in parents and trashed=false"
    results = service.files().list(
        q=query,
        spaces="drive",
        fields="files(id, name, createdTime, modifiedTime, size, md5Checksum)",
        orderBy="name",
    ).execute()
    return results.get("files", [])


def upload_or_update(service, folder_id, name, local_path, mimetype) -> str:
    media = MediaFileUpload(str(local_path), mimetype=mimetype, resumable=True)
    existing_id = find_file_id(service, folder_id, name)
    if existing_id:
        updated = service.files().update(fileId=existing_id, media_body=media, fields="id").execute()
        return updated["id"]
    metadata = {"name": name, "parents": [folder_id]}
    created = service.files().create(body=metadata, media_body=media, fields="id").execute()
    return created["id"]


def upload_create_only(service, folder_id, name, local_path, mimetype) -> str:
    """Create an immutable Drive object, refusing to replace an existing name."""
    if find_file_id(service, folder_id, name):
        raise FileExistsError(f"Drive object already exists and is immutable: {name}")
    media = MediaFileUpload(str(local_path), mimetype=mimetype, resumable=True)
    metadata = {"name": name, "parents": [folder_id]}
    created = service.files().create(
        body=metadata, media_body=media, fields="id"
    ).execute()
    return created["id"]


def download_to(service, file_id, local_path) -> None:
    local_path = pathlib.Path(local_path)
    request = service.files().get_media(fileId=file_id)
    with open(local_path, "wb") as handle:
        downloader = MediaIoBaseDownload(handle, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _missing_required_models(models_dir):
    models_dir = pathlib.Path(models_dir)
    return [
        name for name in REQUIRED_MODEL_FILES
        if not (models_dir / name).is_file()
    ]


def _validate_model_artifacts(models_dir) -> None:
    """Prove a staged/publication model set is loadable before it can replace state."""
    models_dir = pathlib.Path(models_dir)
    missing_models = _missing_required_models(models_dir)
    if missing_models:
        raise ValueError(
            f"missing required artifacts: {', '.join(missing_models)}"
        )

    manifest_path = models_dir / "model_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("model_manifest.json is not valid JSON") from exc
    predictors = manifest.get("predictors") if isinstance(manifest, dict) else None
    if (
        not isinstance(predictors, list)
        or not predictors
        or not all(isinstance(name, str) and name for name in predictors)
    ):
        raise ValueError(
            "model_manifest.json must contain a non-empty string predictor list"
        )

    probability_stack = manifest.get("probability_stack")
    probability_stack_version = 0
    if isinstance(probability_stack, dict):
        probability_stack_version = int(
            probability_stack.get("schema_version", 0) or 0
        )
    if probability_stack_version >= 3:
        missing_probability_files = [
            name
            for name in PROBABILITY_STACK_V3_FILES
            if not (models_dir / name).is_file()
        ]
        if missing_probability_files:
            raise ValueError(
                "probability-stack v3 is missing required artifacts: "
                + ", ".join(missing_probability_files)
            )
        market_config = probability_stack.get("market")
        no_market_config = probability_stack.get("no_market")
        if (
            not isinstance(market_config, dict)
            or market_config.get("strategy") != "simplex"
            or market_config.get("stacker_file") != "stacker.pkl"
            or market_config.get("calibrator_file") != "win_prob_calibrator.pkl"
            or market_config.get("experts")
            != ["tier_a", "tier_b", "tier_c", "market"]
            or not isinstance(no_market_config, dict)
            or no_market_config.get("stacker_file") != "stacker_no_market.pkl"
            or no_market_config.get("calibrator_file")
            != "win_prob_calibrator_no_market.pkl"
            or no_market_config.get("strategy") not in {"simplex", "tier_b"}
            or no_market_config.get("experts")
            != ["tier_a", "tier_b", "tier_c"]
        ):
            raise ValueError("probability-stack v3 manifest contract is invalid")

    try:
        import dill as model_pickle
    except ImportError as exc:
        raise ValueError("dill is required to validate model artifacts") from exc

    loaded = {}
    for model_path in sorted(models_dir.glob("*.pkl")):
        try:
            with open(model_path, "rb") as handle:
                loaded[model_path.name] = model_pickle.load(handle)
        except Exception as exc:
            raise ValueError(f"{model_path.name} cannot be loaded") from exc

    for name in ("home_model.pkl", "away_model.pkl"):
        if not callable(getattr(loaded.get(name), "predict", None)):
            raise ValueError(f"{name} does not expose a predict method")
    if probability_stack_version >= 3:
        def validate_simplex(name, expected_experts, include_market, config):
            artifact = loaded.get(name)
            if (
                artifact.__class__.__name__ != "SimplexLogitPool"
                or artifact.__class__.__module__
                != "pipeline.common.model_training.calibration"
                or not callable(getattr(artifact, "predict", None))
                or getattr(artifact, "_is_fitted", False) is not True
                or bool(getattr(artifact, "include_market", None))
                is not include_market
                or tuple(getattr(artifact, "expert_names_", ()))
                != tuple(expected_experts)
            ):
                raise ValueError(f"{name} does not satisfy the fitted simplex contract")
            try:
                weights = [float(value) for value in artifact.weights_]
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} simplex weights are invalid") from exc
            if (
                len(weights) != len(expected_experts)
                or not all(math.isfinite(value) and value >= 0.0 for value in weights)
                or not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-8)
            ):
                raise ValueError(f"{name} simplex weights are invalid")
            manifest_weights = config.get("weights")
            if not isinstance(manifest_weights, dict) or any(
                expert not in manifest_weights
                or not math.isclose(
                    float(manifest_weights[expert]),
                    weights[index],
                    rel_tol=0.0,
                    abs_tol=1e-8,
                )
                for index, expert in enumerate(expected_experts)
            ):
                raise ValueError(f"{name} weights do not match model_manifest.json")

        def validate_temperature(name, config):
            artifact = loaded.get(name)
            try:
                temperature = float(artifact.temperature_)
                manifest_temperature = float(config.get("temperature"))
            except (AttributeError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} temperature is invalid") from exc
            if (
                artifact.__class__.__name__ != "TemperatureCalibrator"
                or artifact.__class__.__module__
                != "pipeline.common.model_training.calibration"
                or not callable(getattr(artifact, "predict", None))
                or getattr(artifact, "_is_fitted", False) is not True
                or not math.isfinite(temperature)
                or temperature <= 0.0
                or not math.isclose(
                    manifest_temperature,
                    temperature,
                    rel_tol=0.0,
                    abs_tol=1e-8,
                )
            ):
                raise ValueError(f"{name} temperature is invalid")

        validate_simplex(
            "stacker.pkl",
            ("tier_a", "tier_b", "tier_c", "market"),
            True,
            market_config,
        )
        validate_temperature("win_prob_calibrator.pkl", market_config)
        validate_simplex(
            "stacker_no_market.pkl",
            ("tier_a", "tier_b", "tier_c"),
            False,
            no_market_config,
        )
        validate_temperature(
            "win_prob_calibrator_no_market.pkl", no_market_config
        )
        if not callable(getattr(loaded.get("binary_model.pkl"), "predict_proba", None)):
            raise ValueError("binary_model.pkl does not expose a predict_proba method")

    for json_path in sorted(models_dir.glob("*.json")):
        if json_path == manifest_path:
            continue
        try:
            json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{json_path.name} is not valid JSON") from exc


def _extract_models_archive(archive_path, destination) -> None:
    """Safely extract a flat models archive into an isolated directory."""
    destination = pathlib.Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()

    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = (destination / member.name).resolve()
            try:
                member_path.relative_to(destination_root)
            except ValueError as exc:
                raise ValueError(
                    f"Unsafe path in models archive: {member.name}"
                ) from exc
            if member.issym() or member.islnk():
                raise ValueError(
                    f"Links are not allowed in models archive: {member.name}"
                )
            if not member.isfile() and not member.isdir():
                raise ValueError(
                    f"Unsupported entry in models archive: {member.name}"
                )
        try:
            tar.extractall(destination, filter="data")
        except TypeError:  # Python without extraction filters
            tar.extractall(destination)


def _write_models_archive(models_dir, archive_path) -> None:
    """Write a byte-for-byte reproducible flat ``tar.gz`` model archive."""
    models_dir = pathlib.Path(models_dir)
    archive_path = pathlib.Path(archive_path)
    raw_tar = archive_path.with_suffix("")
    with tarfile.open(raw_tar, "w") as tar:
        for item in sorted(models_dir.iterdir()):
            if not item.is_file():
                continue
            info = tarfile.TarInfo(item.name)
            info.size = item.stat().st_size
            info.mtime = 0
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            with open(item, "rb") as handle:
                tar.addfile(info, handle)
    with open(raw_tar, "rb") as source, open(archive_path, "wb") as destination:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=destination, mtime=0
        ) as compressed:
            shutil.copyfileobj(source, compressed)
    raw_tar.unlink(missing_ok=True)


def _replace_models_dir(staged_models, models_dir, transaction_dir) -> None:
    """Replace models as a directory unit, restoring the old set on failure."""
    staged_models = pathlib.Path(staged_models)
    models_dir = pathlib.Path(models_dir)
    backup_dir = pathlib.Path(transaction_dir) / "models-backup"
    had_existing_models = models_dir.exists()

    if had_existing_models:
        os.replace(models_dir, backup_dir)
    try:
        os.replace(staged_models, models_dir)
    except Exception:
        if had_existing_models and backup_dir.exists():
            os.replace(backup_dir, models_dir)
        raise


def _state_folder(service, root) -> str:
    return get_or_create_folder(service, STATE_FOLDER_NAME, _folder_id(root))


def _existing_state_folder(service, root) -> str:
    state_id = find_folder_id(service, STATE_FOLDER_NAME, _folder_id(root))
    if not state_id:
        raise RuntimeError("Drive state folder does not exist")
    return state_id


def compute_schedule(db_path, now=None) -> dict:
    """Upcoming-round kickoffs (true UTC epoch) plus per-round sent status.

    Carries several rounds, not just the next: rounds stay 'Pre Game' after
    their email goes out, so the gate must be able to look past an
    already-sent round to find the next actionable kickoff.
    """
    now = time.time() if now is None else now
    schedule = {
        "generated_at_utc": int(now),
        "competition_year": None,
        "upcoming_rounds": [],
    }
    con = sqlite3.connect(str(db_path))
    try:
        year_row = con.execute(
            "SELECT MAX(CAST(competition_year AS INTEGER)) FROM footy_tipping_data "
            "WHERE game_state_name = 'Pre Game'"
        ).fetchone()
        year = year_row[0] if year_row else None
        if year is None:
            return schedule
        schedule["competition_year"] = int(year)

        rounds = con.execute(
            """
            SELECT CAST(round_id AS INTEGER) AS round_id,
                   MIN(CAST(start_time_utc AS REAL)) AS first_kickoff_utc
            FROM footy_tipping_data
            WHERE game_state_name = 'Pre Game'
              AND CAST(competition_year AS INTEGER) = ?
            GROUP BY CAST(round_id AS INTEGER)
            ORDER BY first_kickoff_utc
            LIMIT ?
            """,
            (int(year), SCHEDULE_ROUND_LIMIT),
        ).fetchall()

        sent_rounds = set()
        try:
            sent_rows = con.execute(
                "SELECT DISTINCT CAST(round_id AS INTEGER) FROM email_sends "
                "WHERE CAST(competition_year AS INTEGER) = ?",
                (int(year),),
            ).fetchall()
            sent_rounds = {row[0] for row in sent_rows}
        except sqlite3.OperationalError:
            pass  # ledger not created yet: nothing sent

        schedule["upcoming_rounds"] = [
            {
                "round_id": round_id,
                "first_kickoff_utc": int(kickoff),
                "sent": round_id in sent_rounds,
            }
            for round_id, kickoff in rounds
            if kickoff is not None
        ]
        return schedule
    finally:
        con.close()


def sydney_send_target_utc(first_kickoff_utc) -> float:
    """Return 11:00 Australia/Sydney on the kickoff's local calendar day."""
    kickoff = dt.datetime.fromtimestamp(float(first_kickoff_utc), tz=dt.timezone.utc)
    local_kickoff = kickoff.astimezone(SYDNEY_TIMEZONE)
    local_target = local_kickoff.replace(
        hour=SEND_HOUR_LOCAL, minute=0, second=0, microsecond=0
    )
    return local_target.astimezone(dt.timezone.utc).timestamp()


def gate_decision(schedule, now=None, grace_hours=GRACE_HOURS,
                  stale_days=STALE_DAYS):
    """Decide what the scheduled gate should do. Returns (mode, reason).

    mode is one of:
      live    - from 11am Sydney on first-game day through kickoff + grace
      refresh - nothing actionable and schedule.json is stale; run predict
                --skip-send just to refresh fixtures (offseason-safe)
      skip    - nothing to do
    """
    now = time.time() if now is None else now
    if not schedule:
        return "skip", "runtime is not seeded: schedule.json is missing from Drive"

    grace = grace_hours * 3600
    for entry in schedule.get("upcoming_rounds", []):
        if entry.get("sent"):
            continue
        kickoff = entry.get("first_kickoff_utc")
        if kickoff is None:
            continue
        target = sydney_send_target_utc(kickoff)
        if now < target:
            hours_away = (target - now) / 3600
            return "skip", (
                f"too early: round {entry.get('round_id')} Sydney 11am target "
                f"opens in {hours_away:.1f}h"
            )
        if now < kickoff + grace:
            return "live", (
                f"round {entry.get('round_id')} reached Sydney 11am send target "
                f"(target {int(target)}, kickoff {int(kickoff)}, now {int(now)})"
            )
        # Past the grace window without a send: fall through to the next round.

    generated = schedule.get("generated_at_utc") or 0
    if now - generated > stale_days * 86400:
        return "refresh", (
            f"schedule.json is {int((now - generated) / 86400)} days old; refreshing fixtures"
        )
    return "skip", "no unsent round in window and schedule is fresh"


def _snapshot_sqlite(source_path, destination) -> None:
    source = sqlite3.connect(str(source_path))
    try:
        target = sqlite3.connect(str(destination))
        try:
            source.backup(target)
        finally:
            target.close()
    finally:
        source.close()


def _validate_sqlite(path) -> None:
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            result = con.execute("PRAGMA quick_check").fetchone()
            if not result or str(result[0]).lower() != "ok":
                raise ValueError(f"SQLite quick_check returned {result!r}")
            table = con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='footy_tipping_data'"
            ).fetchone()
            if not table:
                raise ValueError("required footy_tipping_data table is missing")
        finally:
            con.close()
    except sqlite3.Error as exc:
        raise ValueError(f"invalid SQLite database: {exc}") from exc


def _write_db_archive(db_path, archive_path) -> None:
    snapshot = pathlib.Path(archive_path).with_suffix(".sqlite")
    _snapshot_sqlite(db_path, snapshot)
    with open(snapshot, "rb") as src, gzip.open(archive_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    snapshot.unlink(missing_ok=True)


def _extract_db_archive(archive_path, destination) -> None:
    try:
        with gzip.open(archive_path, "rb") as src, open(destination, "wb") as dst:
            shutil.copyfileobj(src, dst)
    except (OSError, EOFError) as exc:
        raise ValueError(f"invalid DB archive: {exc}") from exc
    _validate_sqlite(destination)


def _release_folder(service, state_id) -> str:
    return get_or_create_folder(service, MODEL_RELEASES_FOLDER, state_id)


def _existing_release_folder(service, state_id) -> str:
    folder_id = find_folder_id(service, MODEL_RELEASES_FOLDER, state_id)
    if not folder_id:
        raise RuntimeError("Drive model-releases folder does not exist")
    return folder_id


def _safe_release_id(release_id) -> str:
    release_id = str(release_id or "").strip()
    if not release_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for character in release_id
    ):
        raise ValueError("release id may contain only letters, numbers, dot, dash, and underscore")
    return release_id


def _read_json(path, description):
    try:
        payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object")
    return payload


def _validate_release_directory(models_dir, release_id=None) -> dict:
    models_dir = pathlib.Path(models_dir)
    _validate_model_artifacts(models_dir)
    receipt_path = models_dir / TRAINING_RECEIPT_FILE
    receipt = _read_json(receipt_path, TRAINING_RECEIPT_FILE)
    if int(receipt.get("schema_version", 0)) != 1:
        raise ValueError("training receipt schema_version must be 1")
    receipt_release = _safe_release_id(receipt.get("release_id"))
    if release_id is not None and receipt_release != _safe_release_id(release_id):
        raise ValueError("training receipt release id does not match requested release")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("training receipt must contain artifact hashes")
    actual_artifacts = {
        path.name
        for path in models_dir.iterdir()
        if path.is_file() and path.name not in {".gitkeep", TRAINING_RECEIPT_FILE}
    }
    if set(artifacts) != actual_artifacts:
        missing = sorted(actual_artifacts - set(artifacts))
        extra = sorted(set(artifacts) - actual_artifacts)
        raise ValueError(
            "training receipt artifact inventory does not match release "
            f"(unrecorded: {missing or 'none'}, missing: {extra or 'none'})"
        )
    for name, expected in artifacts.items():
        if not isinstance(name, str) or pathlib.Path(name).name != name:
            raise ValueError("training receipt contains an unsafe artifact name")
        artifact = models_dir / name
        if not artifact.is_file():
            raise ValueError(f"receipt artifact is missing: {name}")
        if not isinstance(expected, dict):
            raise ValueError(f"receipt metadata is invalid for {name}")
        if int(expected.get("size", -1)) != artifact.stat().st_size:
            raise ValueError(f"artifact size does not match receipt: {name}")
        if expected.get("sha256") != _sha256(artifact):
            raise ValueError(f"artifact hash does not match receipt: {name}")
    return receipt


def _load_remote_pointer(service, state_id, destination) -> dict:
    pointer_id = find_file_id(service, state_id, MODEL_POINTER_FILE)
    if not pointer_id:
        raise RuntimeError(
            "model-current.json is missing from Drive; publish and activate a model release first"
        )
    download_to(service, pointer_id, destination)
    pointer = _read_json(destination, MODEL_POINTER_FILE)
    if int(pointer.get("schema_version", 0)) != 1:
        raise ValueError("model-current.json schema_version must be 1")
    release_id = _safe_release_id(pointer.get("release_id"))
    if pointer.get("archive") != f"{release_id}.tar.gz":
        raise ValueError("model-current.json archive does not match release id")
    if pointer.get("metadata") != f"{release_id}.json":
        raise ValueError("model-current.json metadata does not match release id")
    archive_hash = pointer.get("archive_sha256")
    if (
        not isinstance(archive_hash, str)
        or len(archive_hash) != 64
        or any(character not in "0123456789abcdef" for character in archive_hash.lower())
    ):
        raise ValueError("model-current.json archive_sha256 is invalid")
    return pointer


def _download_release(
    service, state_id, release_id, destination, expected_hash=None, *, create_folder=False
):
    release_id = _safe_release_id(release_id)
    destination = pathlib.Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    releases_id = (
        _release_folder(service, state_id)
        if create_folder
        else _existing_release_folder(service, state_id)
    )
    archive_name = f"{release_id}.tar.gz"
    metadata_name = f"{release_id}.json"
    archive_id = find_file_id(service, releases_id, archive_name)
    metadata_id = find_file_id(service, releases_id, metadata_name)
    if not archive_id or not metadata_id:
        raise RuntimeError(
            f"model release {release_id} is incomplete on Drive "
            f"(archive: {'found' if archive_id else 'MISSING'}, "
            f"metadata: {'found' if metadata_id else 'MISSING'})"
        )
    archive_path = destination / archive_name
    metadata_path = destination / metadata_name
    download_to(service, archive_id, archive_path)
    download_to(service, metadata_id, metadata_path)
    metadata = _read_json(metadata_path, metadata_name)
    if metadata.get("release_id") != release_id or int(metadata.get("schema_version", 0)) != 1:
        raise ValueError(f"model release metadata is invalid for {release_id}")
    archive_hash = _sha256(archive_path)
    if metadata.get("archive_sha256") != archive_hash:
        raise ValueError(f"model release archive hash does not match metadata: {release_id}")
    if expected_hash is not None and expected_hash != archive_hash:
        raise ValueError(f"model release archive hash does not match active pointer: {release_id}")
    staged_models = destination / "models-staged"
    _extract_models_archive(archive_path, staged_models)
    receipt = _validate_release_directory(staged_models, release_id=release_id)
    return staged_models, archive_path, metadata, receipt


def download_runtime_db(destination, root=None):
    """Download and validate the published runtime DB without mutating the repo."""
    root = pathlib.Path(root) if root is not None else _project_root()
    destination = pathlib.Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    service = drive_service(root / "service-account-token.json")
    state_id = _existing_state_folder(service, root)
    db_id = find_file_id(service, state_id, DB_ARCHIVE)
    if not db_id:
        raise RuntimeError(f"{DB_ARCHIVE} is missing from Drive")
    with tempfile.TemporaryDirectory() as tmp:
        archive = pathlib.Path(tmp) / DB_ARCHIVE
        staged = pathlib.Path(tmp) / "runtime.sqlite"
        download_to(service, db_id, archive)
        _extract_db_archive(archive, staged)
        shutil.copy2(staged, destination)
    return destination


def push_runtime_state(root) -> int:
    """Publish only mutable runtime DB/schedule state; never publish models."""
    root = pathlib.Path(root)
    db_path = root / "data" / "footy-tipper-db.sqlite"
    if not db_path.exists():
        _log(f"DB not found at {db_path}; runtime was not pushed.")
        return 1
    try:
        _validate_sqlite(db_path)
    except ValueError as exc:
        _log(f"Runtime DB is invalid; refusing to push: {exc}")
        return 1
    service = drive_service(root / "service-account-token.json")
    state_id = _state_folder(service, root)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        db_gz = tmp / DB_ARCHIVE
        _write_db_archive(db_path, db_gz)
        schedule = compute_schedule(db_path)
        schedule_path = tmp / SCHEDULE_FILE
        schedule_path.write_text(json.dumps(schedule, indent=2), encoding="utf-8")
        upload_or_update(service, state_id, DB_ARCHIVE, db_gz, "application/gzip")
        _log(f"Uploaded runtime DB ({db_gz.stat().st_size / 1e6:.1f} MB).")
        upload_or_update(service, state_id, SCHEDULE_FILE, schedule_path, "application/json")
        _log(
            f"Uploaded schedule: year {schedule['competition_year']}, "
            f"{len(schedule['upcoming_rounds'])} upcoming rounds."
        )
    return 0


def pull_runtime_state(root) -> int:
    """Atomically restore the mutable DB and the explicitly active model release."""
    root = pathlib.Path(root)
    service = drive_service(root / "service-account-token.json")
    try:
        state_id = _existing_state_folder(service, root)
    except RuntimeError as exc:
        _log(str(exc))
        return 1
    db_id = find_file_id(service, state_id, DB_ARCHIVE)
    if not db_id:
        _log(f"Runtime DB archive is missing from Drive: {DB_ARCHIVE}")
        return 1
    data_dir = root / "data"
    models_dir = root / "models"
    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".runtime-pull-", dir=root) as tmp:
        tmp = pathlib.Path(tmp)
        try:
            pointer = _load_remote_pointer(service, state_id, tmp / MODEL_POINTER_FILE)
            staged_models, _, _, _ = _download_release(
                service,
                state_id,
                pointer["release_id"],
                tmp / "release",
                expected_hash=pointer["archive_sha256"],
            )
            db_gz = tmp / DB_ARCHIVE
            download_to(service, db_id, db_gz)
            staged_db = tmp / "db-staged.sqlite"
            _extract_db_archive(db_gz, staged_db)
        except (OSError, tarfile.TarError, RuntimeError, ValueError) as exc:
            _log(f"Downloaded runtime state is invalid; local state was not changed: {exc}")
            return 1

        (staged_models / ".gitkeep").touch(exist_ok=True)
        db_path = data_dir / "footy-tipper-db.sqlite"
        _replace_models_dir(staged_models, models_dir, tmp)
        try:
            os.replace(staged_db, db_path)
        except Exception:
            failed_models = tmp / "models-failed"
            os.replace(models_dir, failed_models)
            backup_dir = tmp / "models-backup"
            if backup_dir.exists():
                os.replace(backup_dir, models_dir)
            raise
        _log(f"Restored runtime DB to {db_path}.")
        _log(f"Activated model release {pointer['release_id']} locally.")
    return 0


def publish_model_release(root, models_dir, receipt, release_id) -> dict:
    """Create and verify an immutable model archive + metadata pair on Drive."""
    root = pathlib.Path(root)
    models_dir = pathlib.Path(models_dir)
    release_id = _safe_release_id(release_id)
    receipt = dict(receipt)
    receipt["schema_version"] = 1
    receipt["release_id"] = release_id
    service = drive_service(root / "service-account-token.json")
    state_id = _state_folder(service, root)
    releases_id = _release_folder(service, state_id)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        staged = tmp / "models"
        staged.mkdir()
        for item in sorted(models_dir.iterdir()):
            if item.is_file() and item.name not in {".gitkeep", TRAINING_RECEIPT_FILE}:
                shutil.copy2(item, staged / item.name)
        (staged / TRAINING_RECEIPT_FILE).write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _validate_release_directory(staged, release_id=release_id)
        archive_name = f"{release_id}.tar.gz"
        metadata_name = f"{release_id}.json"
        archive = tmp / archive_name
        _write_models_archive(staged, archive)
        metadata = {
            "schema_version": 1,
            "release_id": release_id,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "archive": archive_name,
            "archive_sha256": _sha256(archive),
            "archive_size": archive.stat().st_size,
            "receipt": receipt,
        }
        metadata_path = tmp / metadata_name
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        existing_archive = find_file_id(service, releases_id, archive_name)
        if existing_archive:
            existing = tmp / f"existing-{archive_name}"
            download_to(service, existing_archive, existing)
            if _sha256(existing) != metadata["archive_sha256"]:
                raise FileExistsError(
                    f"immutable release {release_id} already exists with different contents"
                )
        else:
            upload_create_only(service, releases_id, archive_name, archive, "application/gzip")

        existing_metadata = find_file_id(service, releases_id, metadata_name)
        if existing_metadata:
            existing = tmp / f"existing-{metadata_name}"
            download_to(service, existing_metadata, existing)
            remote_metadata = _read_json(existing, metadata_name)
            if remote_metadata != metadata:
                # created_at differs on a retry, so accept an otherwise identical immutable pair.
                comparable_remote = dict(remote_metadata)
                comparable_local = dict(metadata)
                comparable_remote.pop("created_at_utc", None)
                comparable_local.pop("created_at_utc", None)
                if comparable_remote != comparable_local:
                    raise FileExistsError(
                        f"immutable release metadata {release_id} already differs"
                    )
                metadata = remote_metadata
        else:
            upload_create_only(service, releases_id, metadata_name, metadata_path, "application/json")

        # Publication is not successful until Drive can return and validate both objects.
        _download_release(service, state_id, release_id, tmp / "verification")
        _log(f"Published and re-verified immutable model release {release_id}.")
        return metadata


def get_model_pointer(root) -> dict:
    root = pathlib.Path(root)
    service = drive_service(root / "service-account-token.json")
    state_id = _existing_state_folder(service, root)
    with tempfile.TemporaryDirectory() as tmp:
        return _load_remote_pointer(service, state_id, pathlib.Path(tmp) / MODEL_POINTER_FILE)


def activate_model_release(root, release_id, *, repair_broken_pointer=False) -> dict:
    """Validate a release again, update the legacy copy, then move the active pointer.

    Normal callers fail closed if the current pointer is malformed. The
    separately confirmed advanced activation path may preserve that bad file
    as evidence and repair the pointer after hosted compatibility validation.
    """
    root = pathlib.Path(root)
    release_id = _safe_release_id(release_id)
    service = drive_service(root / "service-account-token.json")
    state_id = _existing_state_folder(service, root)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        _, archive, metadata, _ = _download_release(
            service, state_id, release_id, tmp / "candidate"
        )
        previous_release = None
        pointer_id = find_file_id(service, state_id, MODEL_POINTER_FILE)
        if pointer_id:
            previous_path = tmp / "previous-pointer.json"
            try:
                previous = _load_remote_pointer(service, state_id, previous_path)
            except (RuntimeError, ValueError):
                if not repair_broken_pointer:
                    raise
                if not previous_path.is_file():
                    raise RuntimeError(
                        "the broken active pointer could not be preserved for repair"
                    )
                stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                evidence_name = f"model-current.invalid-{stamp}.json"
                upload_create_only(
                    service,
                    state_id,
                    evidence_name,
                    previous_path,
                    "application/json",
                )
                _log(
                    f"Archived the malformed active pointer as {evidence_name} before repair."
                )
            else:
                if previous.get("release_id") == release_id:
                    _log(f"Model release {release_id} is already active; pointer unchanged.")
                    return previous
                previous_release = previous.get("release_id")
        pointer = {
            "schema_version": 1,
            "release_id": release_id,
            "archive": metadata["archive"],
            "metadata": f"{release_id}.json",
            "archive_sha256": metadata["archive_sha256"],
            "activated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "previous_release_id": previous_release,
        }
        pointer_path = tmp / MODEL_POINTER_FILE
        pointer_path.write_text(
            json.dumps(pointer, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # Keep the legacy archive usable during the rollout, but new consumers
        # trust only the immutable release named by model-current.json.
        upload_or_update(service, state_id, MODELS_ARCHIVE, archive, "application/gzip")
        upload_or_update(service, state_id, MODEL_POINTER_FILE, pointer_path, "application/json")
        _log(f"Activated model release {release_id} (previous: {previous_release or 'none'}).")
        return pointer


def list_model_releases(root) -> list:
    root = pathlib.Path(root)
    service = drive_service(root / "service-account-token.json")
    state_id = _existing_state_folder(service, root)
    releases_id = _existing_release_folder(service, state_id)
    result = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        for item in list_files(service, releases_id):
            name = item.get("name", "")
            if not name.endswith(".json"):
                continue
            path = tmp / name
            download_to(service, item["id"], path)
            try:
                result.append(_read_json(path, name))
            except ValueError:
                result.append({"release_id": name[:-5], "invalid": True})
    return sorted(result, key=lambda entry: str(entry.get("created_at_utc", "")), reverse=True)


def check_model_release(root, release_id) -> int:
    root = pathlib.Path(root)
    service = drive_service(root / "service-account-token.json")
    state_id = _existing_state_folder(service, root)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            staged_models, _, _, _ = _download_release(
                service, state_id, release_id, tmp / "release"
            )
            db_id = find_file_id(service, state_id, DB_ARCHIVE)
            if not db_id:
                raise RuntimeError(f"{DB_ARCHIVE} is missing from Drive")
            db_archive = tmp / DB_ARCHIVE
            candidate_db = tmp / "candidate-runtime.sqlite"
            download_to(service, db_id, db_archive)
            _extract_db_archive(db_archive, candidate_db)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(root) + (
                os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
            )
            env["FOOTY_TIPPER_PROJECT_ROOT"] = str(root)
            env["FOOTY_TIPPER_MODELS_DIR"] = str(staged_models)
            env["FOOTY_TIPPER_DB_PATH"] = str(candidate_db)
            result = subprocess.run(
                [sys.executable, str(root / "pipeline" / "inference.py")],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=30 * 60,
            )
            if result.returncode:
                detail = (result.stderr or result.stdout or "unknown inference failure")
                raise RuntimeError(
                    "candidate inference failed in this runtime: "
                    + detail.strip().splitlines()[-1]
                )
            _validate_sqlite(candidate_db)
    except (
        OSError,
        subprocess.SubprocessError,
        tarfile.TarError,
        RuntimeError,
        ValueError,
    ) as exc:
        _log(f"Model release {release_id} failed compatibility checks: {exc}")
        return 1
    _log(f"Model release {release_id} passed compatibility checks.")
    return 0


def rollback_model_release(root) -> dict:
    pointer = get_model_pointer(root)
    previous = pointer.get("previous_release_id")
    if not previous:
        raise RuntimeError("the active model pointer has no previous release to roll back to")
    return activate_model_release(root, previous)


# Transitional Python API names intentionally keep runtime-only semantics.
# They no longer upload or select model artifacts.
def push_state(root) -> int:
    return push_runtime_state(root)


def pull_state(root) -> int:
    return pull_runtime_state(root)


def _write_github_output(mode, reason) -> None:
    output_path = os.getenv("GITHUB_OUTPUT")
    if not output_path:
        return
    with open(output_path, "a", encoding="utf-8") as handle:
        handle.write(f"mode={mode}\n")
        handle.write(f"reason={reason}\n")


def run_gate(root) -> int:
    root = pathlib.Path(root)
    service = drive_service(root / "service-account-token.json")
    schedule = None
    try:
        state_id = _existing_state_folder(service, root)
    except RuntimeError:
        state_id = None
    if state_id:
        schedule_id = find_file_id(service, state_id, SCHEDULE_FILE)
        if schedule_id:
            with tempfile.TemporaryDirectory() as tmp:
                schedule_path = pathlib.Path(tmp) / SCHEDULE_FILE
                download_to(service, schedule_id, schedule_path)
                schedule = json.loads(schedule_path.read_text(encoding="utf-8"))

    mode, reason = gate_decision(schedule)
    _log(f"Gate decision: {mode} ({reason})")
    _write_github_output(mode, reason)
    return 0


def print_schedule(root) -> int:
    db_path = pathlib.Path(root) / "data" / "footy-tipper-db.sqlite"
    if not db_path.exists():
        _log(f"DB not found at {db_path}.")
        return 1
    print(json.dumps(compute_schedule(db_path), indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="state_sync",
        description="Sync mutable runtime state and gate scheduled runs.",
    )
    parser.add_argument(
        "action",
        choices=("runtime-push", "runtime-pull", "gate", "schedule"),
        help=(
            "runtime-push: upload DB+schedule.json only; "
            "runtime-pull: download DB+active immutable model; "
            "gate: read schedule.json and decide live/refresh/skip; "
            "schedule: print schedule.json computed from the local DB"
        ),
    )
    args = parser.parse_args(argv)
    root = _project_root()

    if args.action == "runtime-push":
        return push_runtime_state(root)
    if args.action == "runtime-pull":
        return pull_runtime_state(root)
    if args.action == "gate":
        return run_gate(root)
    return print_schedule(root)


if __name__ == "__main__":
    raise SystemExit(main())
