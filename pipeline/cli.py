import argparse
import os
import pathlib
import subprocess
import sys

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return False


DEFAULT_TEST_EMAIL = "levon.rush@gmail.com"


def _project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def _run_command(cmd, env):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


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


def _run_data_prep(env):
    _run_command(["Rscript", "pipeline/data-prep.R"], env)


def _run_train(env, skip_prep):
    if not skip_prep:
        _run_data_prep(env)
    _run_command([sys.executable, "pipeline/train.py"], env)


def _run_inference(env, skip_prep):
    if not skip_prep:
        _run_data_prep(env)
    _run_command([sys.executable, "pipeline/inference.py"], env)


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
        print("No pre-game predictions available. Nothing to send.")
        return 0

    tipper_picks = sf.get_tipper_picks(predictions)

    # In test mode, skip Drive upload by default unless explicitly requested.
    if not skip_drive:
        sf.upload_df_to_drive(
            predictions,
            json_path,
            os.getenv("FOLDER_ID"),
            "predictions.csv",
        )
    else:
        print("Drive upload skipped.")

    api_key = os.getenv("OPENAI_KEY") if use_openai else None
    if test_mode and not use_openai:
        print("Test mode active: using fallback email content (OpenAI disabled).")

    email_body = sf.generate_reg_regan_email(
        predictions,
        tipper_picks,
        api_key,
        os.getenv("FOLDER_URL"),
        0.9,
    )

    round_name = predictions["round_name"].iloc[0]
    competition_year = predictions["competition_year"].iloc[0]
    subject = f"Footy Tipper Predictions for {round_name} {competition_year}"

    if test_mode:
        subject = f"[TEST] {subject}"
        if dry_run:
            print("Dry run enabled. Email was not sent.")
            print(f"To: {test_email}")
            print(f"Subject: {subject}")
            print("")
            print(email_body)
            return 0

        sf.send_test_email(
            subject,
            email_body,
            os.getenv("MY_EMAIL"),
            os.getenv("EMAIL_PASSWORD"),
            test_email,
        )
        print(f"Test email sent to {test_email}.")
        return 0

    if dry_run:
        print("Dry run enabled. Production email was not sent.")
        print(f"Subject: {subject}")
        print("")
        print(email_body)
        return 0

    sf.send_emails(
        "footy-tipper-email-list",
        subject,
        email_body,
        os.getenv("MY_EMAIL"),
        os.getenv("EMAIL_PASSWORD"),
        json_path,
    )
    print("Production email flow complete.")
    return 0


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
        default=os.getenv("FOOTY_TIPPER_TEST_EMAIL", DEFAULT_TEST_EMAIL),
        help=f"Recipient for --test mode (default: {DEFAULT_TEST_EMAIL}).",
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
        default=os.getenv("FOOTY_TIPPER_TEST_EMAIL", DEFAULT_TEST_EMAIL),
        help=f"Recipient for --test mode (default: {DEFAULT_TEST_EMAIL}).",
    )
    predict.add_argument("--skip-drive", action="store_true", help="Skip Google Drive upload during send step.")
    _add_openai_args(predict)
    predict.add_argument("--dry-run", action="store_true", help="Print email output without sending.")

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    env = _build_env(args)

    if args.command == "prep":
        _run_data_prep(env)
        return 0

    if args.command == "train":
        _run_train(env, skip_prep=args.skip_prep)
        return 0

    if args.command == "infer":
        _run_inference(env, skip_prep=args.skip_prep)
        return 0

    if args.command == "send":
        use_openai = args.use_openai
        skip_drive = args.skip_drive or args.test
        return _send_predictions(
            test_mode=args.test,
            test_email=args.test_email,
            skip_drive=skip_drive,
            use_openai=use_openai,
            dry_run=args.dry_run,
        )

    if args.command == "predict":
        _run_inference(env, skip_prep=args.skip_prep)
        if args.skip_send:
            print("Send step skipped.")
            return 0
        skip_drive = args.skip_drive or args.test
        return _send_predictions(
            test_mode=args.test,
            test_email=args.test_email,
            skip_drive=skip_drive,
            use_openai=args.use_openai,
            dry_run=args.dry_run,
        )

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
