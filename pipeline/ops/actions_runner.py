"""Exact, machine-only interface used by GitHub Actions.

This is intentionally separate from the operator CLI.  Every state-changing
operation and every delivery mode is named explicitly; unknown values are an
invocation error and can never become a live send by fall-through.

Usage::

    python -m pipeline.ops.actions_runner gate
    python -m pipeline.ops.actions_runner runtime-pull
    python -m pipeline.ops.actions_runner predict --mode test|refresh|live
    python -m pipeline.ops.actions_runner runtime-push
    python -m pipeline.ops.actions_runner site-publish
    python -m pipeline.ops.actions_runner model-check --release RELEASE_ID
"""

import argparse
import json
import os
import pathlib
import tempfile

from pipeline.ops import state_sync


PREDICT_MODES = ("test", "refresh", "live")


def _project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _run_gate(root: pathlib.Path) -> int:
    """Read Drive schedule state and emit only live/refresh/skip vocabulary."""
    service = state_sync.drive_service(root / "service-account-token.json")
    schedule = None
    try:
        state_id = state_sync._existing_state_folder(service, root)
    except RuntimeError:
        state_id = None
    if state_id:
        schedule_id = state_sync.find_file_id(
            service, state_id, state_sync.SCHEDULE_FILE
        )
        if schedule_id:
            with tempfile.TemporaryDirectory() as tmp:
                schedule_path = pathlib.Path(tmp) / state_sync.SCHEDULE_FILE
                state_sync.download_to(service, schedule_id, schedule_path)
                try:
                    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
                    state_sync._log(
                        "Published schedule.json is unreadable; requesting a refresh."
                    )
                    schedule = None

    mode, reason = state_sync.gate_decision(schedule)
    if mode not in {"live", "refresh", "skip"}:
        raise RuntimeError(f"Gate produced unsupported mode: {mode!r}")

    state_sync._log(f"Gate decision: {mode} ({reason})")
    state_sync._write_github_output(mode, reason)
    return 0


def _required_state_api(name: str):
    function = getattr(state_sync, name, None)
    if function is None:
        raise RuntimeError(
            f"Runtime state API {name} is unavailable in this image; "
            "rebuild the production image before running predictions."
        )
    return function


def _site_publish(root: pathlib.Path) -> int:
    from pipeline.common.use_predictions import site

    db_path = root / "data" / "footy-tipper-db.sqlite"
    site.generate_site(db_path, root)
    return 0 if site.publish_site(root) else 1


def _run_prediction(mode: str, confirmed_round: int | None = None) -> int:
    # Keep this import lazy: the lightweight gate job installs only Google
    # client dependencies and must not inherit the prediction/CLI dependency
    # chain just by importing this machine interface.
    from pipeline.ops import runtime_prediction

    variable = "FOOTY_TIPPER_CONFIRMED_LIVE_ROUND"
    previous = os.environ.get(variable)
    try:
        if confirmed_round is None:
            # An explicit empty value prevents load_dotenv from injecting a
            # stale confirmation into a scheduled run.
            os.environ[variable] = ""
        else:
            os.environ[variable] = str(confirmed_round)
        return runtime_prediction.run(mode)
    finally:
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous


def _positive_round(value: str) -> int:
    try:
        round_id = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("confirmed round must be a whole number") from exc
    if round_id < 1:
        raise argparse.ArgumentTypeError("confirmed round must be at least 1")
    return round_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="actions-runner",
        description="Strict machine interface for the prediction workflow.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("gate", help="Emit live, refresh, or skip from Drive schedule state.")
    commands.add_parser("runtime-pull", help="Pull the runtime DB and active model release.")

    predict = commands.add_parser("predict", help="Run one exact prediction/delivery mode.")
    predict.add_argument("--mode", required=True, choices=PREDICT_MODES)
    predict.add_argument(
        "--confirmed-round",
        type=_positive_round,
        help="Bind a manually confirmed live send to this exact round.",
    )

    commands.add_parser("runtime-push", help="Publish only the runtime DB and schedule.")
    commands.add_parser("site-publish", help="Generate and publish the static tips site.")

    model_check = commands.add_parser(
        "model-check", help="Validate an immutable candidate model release."
    )
    model_check.add_argument("--release", required=True, metavar="RELEASE_ID")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = _project_root()

    if args.command == "gate":
        return _run_gate(root)
    if args.command == "runtime-pull":
        return _required_state_api("pull_runtime_state")(root)
    if args.command == "predict":
        if args.confirmed_round is not None and args.mode != "live":
            parser.error("--confirmed-round is valid only with --mode live")
        if args.mode == "live" and os.getenv("GITHUB_ACTIONS", "").lower() != "true":
            parser.error(
                "--mode live is allowed only inside GitHub Actions; "
                "operators must use `footy-tipper tips live`"
            )
        return _run_prediction(args.mode, args.confirmed_round)
    if args.command == "runtime-push":
        return _required_state_api("push_runtime_state")(root)
    if args.command == "site-publish":
        return _site_publish(root)
    if args.command == "model-check":
        release_id = args.release.strip()
        if not release_id:
            parser.error("--release must not be empty")
        return _required_state_api("check_model_release")(root, release_id)

    # argparse requires a known subcommand, so this is defensive only.
    raise RuntimeError(f"Unsupported Actions command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
