import argparse
import os
import pathlib
import subprocess
import sys
import time

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return False


DEFAULT_TEST_EMAIL = "levon.rush@gmail.com"
CLI_START = time.monotonic()


def _project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def _format_elapsed(seconds: float) -> str:
    whole = int(max(0, seconds))
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _log(message: str, start_time: float = None) -> None:
    base = CLI_START if start_time is None else start_time
    elapsed = _format_elapsed(time.monotonic() - base)
    print(f"[+{elapsed}] {message}", flush=True)


def _run_command(cmd, env, cwd=None):
    cmd_text = " ".join(cmd)
    cmd_start = time.monotonic()
    _log(f"Running: {cmd_text}", start_time=cmd_start)
    proc = subprocess.Popen(
        cmd,
        env=env,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if proc.stdout is not None:
        for raw_line in proc.stdout:
            _log(raw_line.rstrip("\n"), start_time=cmd_start)

    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    _log(f"Completed: {cmd_text}", start_time=cmd_start)


def _build_env(args):
    env = os.environ.copy()
    if getattr(args, "start_year", None) is not None:
        env["FOOTY_TIPPER_START_YEAR"] = str(args.start_year)
    if getattr(args, "end_year", None) is not None:
        env["FOOTY_TIPPER_END_YEAR"] = str(args.end_year)
    if getattr(args, "include_performance", None) is not None:
        env["FOOTY_TIPPER_INCLUDE_PERFORMANCE"] = "true" if args.include_performance else "false"
    if getattr(args, "require_odds", None) is not None:
        env["FOOTY_TIPPER_REQUIRE_ODDS"] = "true" if args.require_odds else "false"
    if getattr(args, "prep_mode", None):
        env["FOOTY_TIPPER_PREP_MODE"] = args.prep_mode
    if getattr(args, "infer_context_years", None) is not None:
        env["FOOTY_TIPPER_INFER_CONTEXT_YEARS"] = str(args.infer_context_years)
    return env


def _run_data_prep(env, root):
    _run_command(["Rscript", str(root / "pipeline" / "data-prep.R")], env, cwd=root)


def _run_train(env, skip_prep, root):
    if not skip_prep:
        _run_data_prep(env, root)
    _run_command([sys.executable, str(root / "pipeline" / "train.py")], env, cwd=root)


def _run_inference(env, skip_prep, root):
    if not skip_prep:
        _run_data_prep(env, root)
    _run_command([sys.executable, str(root / "pipeline" / "inference.py")], env, cwd=root)


def _send_predictions(test_mode, test_email, skip_drive, use_openai, dry_run):
    try:
        from pipeline.common.use_predictions import sending_functions as sf
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", "dependency")
        raise RuntimeError(
            f"Send workflow requires project dependencies (missing: {missing}). "
            "Install requirements and retry."
        ) from exc

    root = _project_root()
    secrets_path = root / "secrets.env"
    db_path = root / "data" / "footy-tipper-db.sqlite"
    json_path = root / "service-account-token.json"

    load_dotenv(dotenv_path=secrets_path)

    predictions = sf.get_predictions(db_path, root)
    if predictions.empty:
        _log("No pre-game predictions available. Nothing to send.")
        return 0

    tipper_picks = sf.get_tipper_picks(predictions)
    joker_recommendation = sf.get_joker_round_recommendation(db_path, root, predictions)
    _log(joker_recommendation.get("headline", "Joker call unavailable"))
    if joker_recommendation.get("detail"):
        _log(joker_recommendation["detail"])

    # In test mode, skip Drive upload by default unless explicitly requested.
    if not skip_drive:
        sf.upload_df_to_drive(
            predictions,
            json_path,
            os.getenv("FOLDER_ID"),
            "predictions.csv",
        )
    else:
        _log("Drive upload skipped.")

    api_key = os.getenv("OPENAI_KEY") if use_openai else None
    if test_mode and not use_openai:
        _log("Test mode active: using fallback email content (OpenAI disabled).")

    email_payload = sf.generate_reg_regan_email_payload(
        predictions,
        tipper_picks,
        api_key,
        os.getenv("FOLDER_URL"),
        0.9,
        use_openai=use_openai,
        joker_recommendation=joker_recommendation,
    )

    subject = email_payload["subject"]
    email_body = email_payload["plain_text"]
    email_html = email_payload["html_text"]
    inline_images = email_payload["inline_images"]

    if test_mode:
        subject = f"[TEST] {subject}"
        if dry_run:
            _log("Dry run enabled. Email was not sent.")
            _log(f"To: {test_email}")
            _log(f"Subject: {subject}")
            _log("")
            _log(email_body)
            return 0

        sf.send_test_email(
            subject,
            email_body,
            os.getenv("MY_EMAIL"),
            os.getenv("EMAIL_PASSWORD"),
            test_email,
            html_message=email_html,
            inline_images=inline_images,
        )
        _log(f"Test email sent to {test_email}.")
        return 0

    if dry_run:
        _log("Dry run enabled. Production email was not sent.")
        _log(f"Subject: {subject}")
        _log("")
        _log(email_body)
        return 0

    sf.send_emails(
        "footy-tipper-email-list",
        subject,
        email_body,
        os.getenv("MY_EMAIL"),
        os.getenv("EMAIL_PASSWORD"),
        json_path,
        html_message=email_html,
        inline_images=inline_images,
    )
    _log("Production email flow complete.")
    return 0


def _resolve_test_email(cli_value):
    if cli_value:
        return cli_value
    return os.getenv("FOOTY_TIPPER_TEST_EMAIL", DEFAULT_TEST_EMAIL)


def _add_season_args(parser):
    parser.add_argument("--start-year", type=int, help="Override FOOTY_TIPPER_START_YEAR.")
    parser.add_argument("--end-year", type=int, help="Override FOOTY_TIPPER_END_YEAR.")
    perf = parser.add_mutually_exclusive_group()
    perf.add_argument(
        "--include-performance",
        dest="include_performance",
        action="store_true",
        default=None,
        help="Force include performance feed features.",
    )
    perf.add_argument(
        "--without-performance",
        dest="include_performance",
        action="store_false",
        default=None,
        help="Disable performance feed features for this run.",
    )

    odds = parser.add_mutually_exclusive_group()
    odds.add_argument(
        "--require-odds",
        dest="require_odds",
        action="store_true",
        default=None,
        help="Advanced: keep only rows with available head-to-head odds.",
    )
    odds.add_argument(
        "--allow-missing-odds",
        dest="require_odds",
        action="store_false",
        default=None,
        help="Default behavior: keep rows even when odds are missing.",
    )


def _add_prep_mode_args(
    parser,
    default_mode,
    choices=("full", "train", "infer"),
    include_infer_context_arg=True,
):
    if "infer" in choices:
        prep_help = (
            "Data prep strategy. full/train rebuild tables from configured season range; "
            "infer limits season scope and performs incremental upserts."
        )
    else:
        prep_help = "Data prep strategy for this command."

    parser.add_argument(
        "--prep-mode",
        choices=choices,
        default=default_mode,
        help=prep_help,
    )
    if include_infer_context_arg:
        parser.add_argument(
            "--infer-context-years",
            type=int,
            default=None,
            help=(
                "When prep mode is infer, include this many prior seasons "
                "for feature context (default from env or 1)."
            ),
        )


def _add_openai_args(parser):
    openai = parser.add_mutually_exclusive_group()
    openai.add_argument(
        "--use-openai",
        dest="use_openai",
        action="store_true",
        help="Use OpenAI-generated email text (default).",
    )
    openai.add_argument(
        "--without-openai",
        dest="use_openai",
        action="store_false",
        help="Use deterministic fallback email text instead of OpenAI.",
    )
    parser.set_defaults(use_openai=True)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="footy-tipper",
        description="Footy Tipper CLI: run prep, train, inference, and send workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prep", help="Run R data preparation and write SQLite tables.")
    _add_season_args(prep)
    _add_prep_mode_args(prep, default_mode="full", choices=("full", "train", "infer"))

    train = subparsers.add_parser("train", help="Run training workflow.")
    _add_season_args(train)
    _add_prep_mode_args(
        train,
        default_mode="full",
        choices=("full", "train"),
        include_infer_context_arg=False,
    )
    train.add_argument("--skip-prep", action="store_true", help="Skip R data prep and train from existing SQLite tables.")

    infer = subparsers.add_parser("infer", help="Run inference workflow.")
    _add_season_args(infer)
    _add_prep_mode_args(infer, default_mode="infer", choices=("infer", "full"))
    infer.add_argument("--skip-prep", action="store_true", help="Skip R data prep and infer from existing SQLite tables.")

    send = subparsers.add_parser("send", help="Send predictions (Drive + email list) or run test send.")
    send.add_argument("--test", action="store_true", help="Send a single test email instead of production list send.")
    send.add_argument(
        "--test-email",
        default=None,
        help=(
            "Recipient for --test mode "
            f"(default: FOOTY_TIPPER_TEST_EMAIL or {DEFAULT_TEST_EMAIL})."
        ),
    )
    send.add_argument(
        "--skip-drive",
        action="store_true",
        help="Skip Google Drive upload.",
    )
    _add_openai_args(send)
    send.add_argument("--dry-run", action="store_true", help="Print email output without sending.")

    predict = subparsers.add_parser("predict", help="Run full prediction workflow (prep -> infer -> send).")
    _add_season_args(predict)
    _add_prep_mode_args(predict, default_mode="infer", choices=("infer", "full"))
    predict.add_argument("--skip-prep", action="store_true", help="Skip R data prep.")
    predict.add_argument("--skip-send", action="store_true", help="Skip send step after inference.")
    predict.add_argument("--test", action="store_true", help="Use test send mode if send step is run.")
    predict.add_argument(
        "--test-email",
        default=None,
        help=(
            "Recipient for --test mode "
            f"(default: FOOTY_TIPPER_TEST_EMAIL or {DEFAULT_TEST_EMAIL})."
        ),
    )
    predict.add_argument("--skip-drive", action="store_true", help="Skip Google Drive upload during send step.")
    _add_openai_args(predict)
    predict.add_argument("--dry-run", action="store_true", help="Print email output without sending.")

    return parser


def main(argv=None):
    root = _project_root()
    load_dotenv(dotenv_path=root / "secrets.env")

    parser = build_parser()
    args = parser.parse_args(argv)
    env = _build_env(args)
    resolved_test_email = _resolve_test_email(getattr(args, "test_email", None))

    if args.command == "prep":
        _run_data_prep(env, root)
        return 0

    if args.command == "train":
        _run_train(env, skip_prep=args.skip_prep, root=root)
        return 0

    if args.command == "infer":
        _run_inference(env, skip_prep=args.skip_prep, root=root)
        return 0

    if args.command == "send":
        use_openai = args.use_openai
        skip_drive = args.skip_drive or args.test
        return _send_predictions(
            test_mode=args.test,
            test_email=resolved_test_email,
            skip_drive=skip_drive,
            use_openai=use_openai,
            dry_run=args.dry_run,
        )

    if args.command == "predict":
        _run_inference(env, skip_prep=args.skip_prep, root=root)
        if args.skip_send:
            _log("Send step skipped.")
            return 0
        skip_drive = args.skip_drive or args.test
        return _send_predictions(
            test_mode=args.test,
            test_email=resolved_test_email,
            skip_drive=skip_drive,
            use_openai=args.use_openai,
            dry_run=args.dry_run,
        )

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
