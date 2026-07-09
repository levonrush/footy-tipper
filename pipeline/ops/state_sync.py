"""Sync pipeline state (SQLite DB + model artifacts) with Google Drive.

Used by the GitHub Actions workflows so runs on fresh runners can pull the
current DB/models before a run and push them back after. Also computes a
small schedule.json (upcoming round kickoffs + sent status) that the hourly
gate job reads to decide whether it is time to run predict.

Deliberately imports only the stdlib plus google-api-python-client/google-auth,
so the gate job can run it after installing just those two packages (no pandas
import chain like pipeline.common.use_predictions.distribution).

Usage: python -m pipeline.ops.state_sync {push|pull|gate|schedule}
"""

import argparse
import gzip
import json
import os
import pathlib
import shutil
import sqlite3
import tarfile
import tempfile
import time

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

WINDOW_HOURS = 6
GRACE_HOURS = 12
STALE_DAYS = 8
SCHEDULE_ROUND_LIMIT = 8


def _project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _log(message: str) -> None:
    print(f"[state-sync] {message}", flush=True)


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


def find_file_id(service, folder_id, name):
    query = f"name='{name}' and '{folder_id}' in parents and trashed=false"
    results = service.files().list(q=query, spaces="drive", fields="files(id, name)").execute()
    files = results.get("files", [])
    return files[0]["id"] if files else None


def upload_or_update(service, folder_id, name, local_path, mimetype) -> str:
    media = MediaFileUpload(str(local_path), mimetype=mimetype, resumable=True)
    existing_id = find_file_id(service, folder_id, name)
    if existing_id:
        updated = service.files().update(fileId=existing_id, media_body=media, fields="id").execute()
        return updated["id"]
    metadata = {"name": name, "parents": [folder_id]}
    created = service.files().create(body=metadata, media_body=media, fields="id").execute()
    return created["id"]


def download_to(service, file_id, local_path) -> None:
    local_path = pathlib.Path(local_path)
    request = service.files().get_media(fileId=file_id)
    with open(local_path, "wb") as handle:
        downloader = MediaIoBaseDownload(handle, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def _state_folder(service, root) -> str:
    return get_or_create_folder(service, STATE_FOLDER_NAME, _folder_id(root))


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


def gate_decision(schedule, now=None, window_hours=WINDOW_HOURS,
                  grace_hours=GRACE_HOURS, stale_days=STALE_DAYS):
    """Decide what the hourly gate should do. Returns (mode, reason).

    mode is one of:
      send    - inside [kickoff - window, kickoff + grace] of an unsent round
      refresh - nothing actionable and schedule.json is stale; run predict
                --skip-send just to refresh fixtures (offseason-safe)
      skip    - nothing to do
    """
    now = time.time() if now is None else now
    if not schedule:
        return "skip", "state not seeded: schedule.json missing from Drive (run `footy-tipper state push` once)"

    window = window_hours * 3600
    grace = grace_hours * 3600
    for entry in schedule.get("upcoming_rounds", []):
        if entry.get("sent"):
            continue
        kickoff = entry.get("first_kickoff_utc")
        if kickoff is None:
            continue
        if now < kickoff - window:
            hours_away = (kickoff - window - now) / 3600
            return "skip", (
                f"too early: round {entry.get('round_id')} window opens in {hours_away:.1f}h"
            )
        if now < kickoff + grace:
            return "send", (
                f"round {entry.get('round_id')} in send window "
                f"(kickoff {int(kickoff)}, now {int(now)})"
            )
        # Past the grace window without a send: fall through to the next round.

    generated = schedule.get("generated_at_utc") or 0
    if now - generated > stale_days * 86400:
        return "refresh", (
            f"schedule.json is {int((now - generated) / 86400)} days old; refreshing fixtures"
        )
    return "skip", "no unsent round in window and schedule is fresh"


def push_state(root) -> int:
    root = pathlib.Path(root)
    db_path = root / "data" / "footy-tipper-db.sqlite"
    models_dir = root / "models"
    if not db_path.exists():
        _log(f"DB not found at {db_path}; nothing to push.")
        return 1
    if not models_dir.is_dir() or not any(models_dir.iterdir()):
        _log(f"Models directory {models_dir} is empty; nothing to push.")
        return 1

    service = drive_service(root / "service-account-token.json")
    state_id = _state_folder(service, root)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)

        # Consistent snapshot via the sqlite backup API, then gzip.
        snapshot = tmp / "db-snapshot.sqlite"
        source = sqlite3.connect(str(db_path))
        try:
            target = sqlite3.connect(str(snapshot))
            try:
                source.backup(target)
            finally:
                target.close()
        finally:
            source.close()
        db_gz = tmp / DB_ARCHIVE
        with open(snapshot, "rb") as src, gzip.open(db_gz, "wb") as dst:
            shutil.copyfileobj(src, dst)

        models_tar = tmp / MODELS_ARCHIVE
        with tarfile.open(models_tar, "w:gz") as tar:
            for item in sorted(models_dir.iterdir()):
                if item.is_file():
                    tar.add(item, arcname=item.name)

        schedule = compute_schedule(db_path)
        schedule_path = tmp / SCHEDULE_FILE
        schedule_path.write_text(json.dumps(schedule, indent=2), encoding="utf-8")

        upload_or_update(service, state_id, DB_ARCHIVE, db_gz, "application/gzip")
        _log(f"Uploaded {DB_ARCHIVE} ({db_gz.stat().st_size / 1e6:.1f} MB).")
        upload_or_update(service, state_id, MODELS_ARCHIVE, models_tar, "application/gzip")
        _log(f"Uploaded {MODELS_ARCHIVE} ({models_tar.stat().st_size / 1e6:.1f} MB).")
        upload_or_update(service, state_id, SCHEDULE_FILE, schedule_path, "application/json")
        _log(
            f"Uploaded {SCHEDULE_FILE}: year {schedule['competition_year']}, "
            f"{len(schedule['upcoming_rounds'])} upcoming rounds."
        )
    return 0


def pull_state(root) -> int:
    root = pathlib.Path(root)
    service = drive_service(root / "service-account-token.json")
    state_id = _state_folder(service, root)

    db_id = find_file_id(service, state_id, DB_ARCHIVE)
    models_id = find_file_id(service, state_id, MODELS_ARCHIVE)
    if not db_id or not models_id:
        _log(
            "State archives missing from Drive "
            f"(db: {'found' if db_id else 'MISSING'}, models: {'found' if models_id else 'MISSING'}). "
            "Seed once with `footy-tipper state push` from a machine that has them."
        )
        return 1

    data_dir = root / "data"
    models_dir = root / "models"
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)

        db_gz = tmp / DB_ARCHIVE
        download_to(service, db_id, db_gz)
        db_tmp = tmp / "db.sqlite"
        with gzip.open(db_gz, "rb") as src, open(db_tmp, "wb") as dst:
            shutil.copyfileobj(src, dst)
        db_path = data_dir / "footy-tipper-db.sqlite"
        os.replace(db_tmp, db_path)
        _log(f"Restored DB to {db_path} ({db_path.stat().st_size / 1e6:.1f} MB).")

        models_tar = tmp / MODELS_ARCHIVE
        download_to(service, models_id, models_tar)
        with tarfile.open(models_tar, "r:gz") as tar:
            try:
                tar.extractall(models_dir, filter="data")
            except TypeError:  # Python without extraction filters
                tar.extractall(models_dir)
        _log(f"Restored models into {models_dir}.")
    return 0


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
    state_id = _state_folder(service, root)

    schedule = None
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
        description="Sync DB/model state with Google Drive and gate scheduled runs.",
    )
    parser.add_argument(
        "action",
        choices=("push", "pull", "gate", "schedule"),
        help=(
            "push: upload DB+models+schedule.json to Drive; "
            "pull: download DB+models from Drive; "
            "gate: read schedule.json and decide send/refresh/skip; "
            "schedule: print schedule.json computed from the local DB"
        ),
    )
    args = parser.parse_args(argv)
    root = _project_root()

    if args.action == "push":
        return push_state(root)
    if args.action == "pull":
        return pull_state(root)
    if args.action == "gate":
        return run_gate(root)
    return print_schedule(root)


if __name__ == "__main__":
    raise SystemExit(main())
