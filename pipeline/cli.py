import argparse
import os
import pathlib
import sqlite3
import subprocess
import sys
import tempfile
import time

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return False


DEFAULT_TEST_EMAIL = "levon.rush@gmail.com"
REQUIRED_MODEL_FILES = ("home_model.pkl", "away_model.pkl", "model_manifest.json")
CLI_START = time.monotonic()
DEFAULT_LINEUP_BACKFILL_MAX_ARTICLES = 2000


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
    env["R_LIBS_USER"] = os.path.expanduser("~/R/library")
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
    if getattr(args, "lineups_mode", None):
        env["FOOTY_TIPPER_LINEUPS_MODE"] = str(args.lineups_mode)
    if getattr(args, "lineups_max_articles", None) is not None:
        env["FOOTY_TIPPER_LINEUPS_MAX_ARTICLES"] = str(args.lineups_max_articles)
    if getattr(args, "lineups_include_sitemap_in_recent", None) is not None:
        env["FOOTY_TIPPER_LINEUPS_INCLUDE_SITEMAP_IN_RECENT"] = (
            "true" if bool(args.lineups_include_sitemap_in_recent) else "false"
        )
    if getattr(args, "lineups_strict", None) is not None:
        env["FOOTY_TIPPER_LINEUPS_STRICT"] = "true" if bool(args.lineups_strict) else "false"
    return env


def _run_data_prep(env, root):
    _run_command(["Rscript", str(root / "pipeline" / "data-prep.R")], env, cwd=root)


def _run_lineups(env, root):
    _run_command([sys.executable, str(root / "pipeline" / "lineups.py")], env, cwd=root)


def _to_bool(value, default):
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    return default


def _env_int(env, key, default):
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _lineups_enabled(env):
    return _to_bool(env.get("FOOTY_TIPPER_LINEUPS_ENABLED"), True)


def _lineups_mode(env):
    return str(env.get("FOOTY_TIPPER_LINEUPS_MODE", "recent")).strip().lower() or "recent"


def _lineup_backfill_db_path(root: pathlib.Path) -> pathlib.Path:
    return root / "data" / "footy-tipper-db.sqlite"


def _lineup_requested_year_window(env):
    start_year = _env_int(env, "FOOTY_TIPPER_START_YEAR", 2010)
    end_year = _env_int(env, "FOOTY_TIPPER_END_YEAR", time.gmtime().tm_year)
    if end_year < start_year:
        end_year = start_year
    return start_year, end_year


def _lineup_backfill_bootstrapped(root: pathlib.Path, env) -> bool:
    db_path = _lineup_backfill_db_path(root)
    if not db_path.exists():
        return False

    start_year, end_year = _lineup_requested_year_window(env)

    try:
        with sqlite3.connect(str(db_path)) as con:
            tables = {
                row[0]
                for row in con.execute(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type = 'table'
                      AND name IN ('lineup_ingestion_runs', 'lineup_entries')
                    """
                ).fetchall()
            }

            if "lineup_ingestion_runs" in tables:
                row = con.execute(
                    """
                    SELECT 1
                    FROM lineup_ingestion_runs
                    WHERE mode = 'backfill'
                      AND status IN ('ok', 'completed_with_errors')
                      AND COALESCE(requested_start_year, 9999) <= ?
                      AND COALESCE(requested_end_year, -9999) >= ?
                    ORDER BY completed_at_utc DESC
                    LIMIT 1
                    """,
                    (start_year, end_year),
                ).fetchone()
                if row:
                    return True

            if "lineup_entries" not in tables:
                return False

            year_count, min_year = con.execute(
                """
                SELECT
                    COUNT(DISTINCT competition_year) AS year_count,
                    MIN(competition_year) AS min_year
                FROM lineup_entries
                WHERE round_id IS NOT NULL
                  AND competition_year BETWEEN ? AND ?
                """,
                (start_year, end_year),
            ).fetchone()

            year_count = int(year_count or 0)
            min_year = int(min_year) if min_year is not None else None
            required_years = min(3, max(1, end_year - start_year + 1))

            if year_count < required_years:
                return False
            if min_year is None:
                return False
            if min_year > (start_year + 1):
                return False
            return True
    except Exception:
        return False


def _bootstrap_lineups_for_training_if_needed(env, root):
    if not _lineups_enabled(env):
        return
    if not _to_bool(env.get("FOOTY_TIPPER_LINEUPS_AUTO_BACKFILL"), True):
        return
    if _lineups_mode(env) == "backfill":
        return
    if _lineup_backfill_bootstrapped(root, env):
        return

    backfill_env = env.copy()
    backfill_env["FOOTY_TIPPER_LINEUPS_MODE"] = "backfill"
    backfill_env["FOOTY_TIPPER_LINEUPS_MAX_ARTICLES"] = str(
        _env_int(
            env,
            "FOOTY_TIPPER_LINEUPS_BACKFILL_MAX_ARTICLES",
            DEFAULT_LINEUP_BACKFILL_MAX_ARTICLES,
        )
    )

    start_year, end_year = _lineup_requested_year_window(env)
    _log(
        "Historical lineup backfill not found for "
        f"{start_year}-{end_year}. Running one-time lineup bootstrap."
    )
    _run_lineups(backfill_env, root)


def _run_train(env, skip_prep, root):
    if not skip_prep:
        _run_data_prep(env, root)
    _run_command([sys.executable, str(root / "pipeline" / "train.py")], env, cwd=root)


def _run_inference(env, skip_prep, root):
    if not skip_prep:
        _run_data_prep(env, root)
    _run_command([sys.executable, str(root / "pipeline" / "inference.py")], env, cwd=root)


def _run_evaluate(env, skip_prep, root):
    if not skip_prep:
        _run_data_prep(env, root)
    _run_command([sys.executable, str(root / "pipeline" / "evaluate.py")], env, cwd=root)


def _model_artifacts_exist(root: pathlib.Path) -> bool:
    models_dir = root / "models"
    return all((models_dir / filename).exists() for filename in REQUIRED_MODEL_FILES)


def _ensure_models_for_prediction(env, root, auto_train=True, allow_lineup_bootstrap=True) -> bool:
    if _model_artifacts_exist(root):
        return True

    if not auto_train:
        _log(
            "Model artifacts are missing. Run `footy-tipper train` "
            "or remove --skip-auto-train."
        )
        return False

    _log("Model artifacts are missing. Running `footy-tipper train` automatically.")
    train_env = env.copy()
    train_env["FOOTY_TIPPER_PREP_MODE"] = "train"
    if allow_lineup_bootstrap:
        _bootstrap_lineups_for_training_if_needed(train_env, root)
    _run_train(train_env, skip_prep=False, root=root)

    if _model_artifacts_exist(root):
        return True

    _log("Training completed but required model artifacts were still not found.")
    return False


def _send_predictions(test_mode, test_email, skip_drive, use_llm, dry_run, force_resend=False):
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

    send_year = None
    send_round_id = None
    try:
        send_year = int(predictions.iloc[0]["competition_year"])
        send_round_id = int(predictions.iloc[0]["round_id"])
    except Exception:
        pass

    # Idempotency gate: never email the production list twice for one round.
    if not test_mode and not dry_run:
        prior_send = sf.email_send_already_recorded(db_path, send_year, send_round_id)
        if prior_send and not force_resend:
            _log(
                f"Production email already sent for {send_year} round {send_round_id} "
                f"(recorded {prior_send.get('sent_at_utc')} UTC). "
                "Use --force-resend to send again."
            )
            return 0
        if prior_send and force_resend:
            _log(
                f"Production email already sent for {send_year} round {send_round_id}; "
                "resending because --force-resend was given."
            )

    tipper_picks = sf.get_tipper_picks(predictions)
    joker_recommendation = sf.get_joker_round_recommendation(db_path, root, predictions)
    _log(joker_recommendation.get("headline", "Joker call unavailable"))
    if joker_recommendation.get("detail"):
        _log(joker_recommendation["detail"])

    scoreboard = sf.get_season_scoreboard(db_path)
    scoreboard_line = sf.scoreboard_summary_line(scoreboard)
    if scoreboard_line:
        _log(f"Scoreboard: {scoreboard_line}")
    else:
        _log("Scoreboard: no completed predicted games yet this season.")

    # In test mode, skip Drive upload by default unless explicitly requested.
    if not skip_drive:
        temp_csv = os.path.join(tempfile.gettempdir(), "predictions.csv")
        sf.upload_df_to_drive(
            predictions,
            json_path,
            os.getenv("FOLDER_ID"),
            temp_csv,
        )
    else:
        _log("Drive upload skipped.")

    api_key = os.getenv("ANTHROPIC_API_KEY") if use_llm else None
    if test_mode and not use_llm:
        _log("Test mode active: using fallback email content (Claude disabled).")

    email_payload = sf.generate_reg_regan_email_payload(
        predictions,
        tipper_picks,
        api_key,
        os.getenv("FOLDER_URL"),
        0.9,
        use_llm=use_llm,
        joker_recommendation=joker_recommendation,
        openai_api_key=os.getenv("OPENAI_KEY") if use_llm else None,
        scoreboard=scoreboard,
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

        sent = sf.send_test_email(
            subject,
            email_body,
            os.getenv("MY_EMAIL"),
            os.getenv("EMAIL_PASSWORD"),
            test_email,
            html_message=email_html,
            inline_images=inline_images,
        )
        if sent:
            _log(f"Test email sent to {test_email}.")
        else:
            _log("Test email flow skipped or failed.")
        return 0

    if dry_run:
        _log("Dry run enabled. Production email was not sent.")
        _log(f"Subject: {subject}")
        _log("")
        _log(email_body)
        return 0

    sent = sf.send_emails(
        "footy-tipper-email-list",
        subject,
        email_body,
        os.getenv("MY_EMAIL"),
        os.getenv("EMAIL_PASSWORD"),
        json_path,
        html_message=email_html,
        inline_images=inline_images,
    )
    if not sent:
        _log("Production email flow skipped or failed. Joker usage state unchanged.")
        return 0

    recipients_count = sent if isinstance(sent, int) and not isinstance(sent, bool) else None
    if sf.record_email_send(
        db_path,
        send_year,
        send_round_id,
        recipients_count=recipients_count,
        source="cli_send_production",
    ):
        _log(f"Send recorded in ledger for {send_year} round {send_round_id}.")

    # Refresh the static site so the public page matches what was just sent.
    try:
        from pipeline.common.use_predictions import site as site_mod

        site_mod.generate_site(db_path, root)
    except Exception as exc:
        _log(f"Site refresh skipped ({exc}).")

    usage_outcome = sf.persist_joker_usage_if_applicable(
        db_path,
        joker_recommendation,
        allow_write=True,
        source="cli_send_production",
    )
    usage_reason = usage_outcome.get("reason")
    if usage_outcome.get("recorded"):
        _log(
            "Joker usage recorded for "
            f"{usage_outcome.get('competition_year')} in round {usage_outcome.get('round_id')}."
        )
    elif usage_reason == "already_recorded":
        _log("Joker usage was already recorded for this season.")
    elif usage_reason == "already_used":
        _log("Joker usage already marked for this season. No update needed.")
    elif usage_reason == "not_play_signal":
        _log("Joker recommendation is HOLD, so usage state was not updated.")
    elif usage_reason == "missing_round_context":
        _log("Joker recommendation lacked season/round context. Usage state was not updated.")
    elif usage_reason == "db_error":
        _log(f"Joker usage write failed: {usage_outcome.get('error', 'unknown db error')}")
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
    if "train" in choices and "infer" in choices:
        prep_help = (
            "Data prep strategy. full forces a fresh API pull for all requested seasons; "
            "train rebuilds prepared tables after a smart cache refresh of missing/current seasons; "
            "infer limits season scope and performs incremental upserts using the same smart refresh."
        )
    elif "train" in choices:
        prep_help = (
            "Data prep strategy. full forces a fresh API pull for all requested seasons; "
            "train rebuilds prepared tables after a smart cache refresh of missing/current seasons."
        )
    elif "infer" in choices:
        prep_help = (
            "Data prep strategy. full forces a fresh API pull for all requested seasons in scope; "
            "infer limits season scope and performs incremental upserts using smart cache refresh."
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


def _add_llm_args(parser):
    llm = parser.add_mutually_exclusive_group()
    llm.add_argument(
        "--with-llm",
        dest="use_llm",
        action="store_true",
        help="Use Claude-generated email copy (default).",
    )
    llm.add_argument(
        "--no-llm",
        dest="use_llm",
        action="store_false",
        help="Use deterministic fallback email text instead of Claude.",
    )
    # Deprecated aliases kept for muscle memory; the copy comes from Claude,
    # OpenAI is only used for the banner image.
    llm.add_argument("--use-openai", dest="use_llm", action="store_true", help=argparse.SUPPRESS)
    llm.add_argument("--without-openai", dest="use_llm", action="store_false", help=argparse.SUPPRESS)
    parser.set_defaults(use_llm=True)


def _add_lineup_args(parser, default_mode="recent", include_skip=True):
    if include_skip:
        parser.add_argument(
            "--skip-lineups",
            action="store_true",
            help="Skip lineup ingestion refresh before this command.",
        )
    parser.add_argument(
        "--lineups-mode",
        choices=("recent", "backfill"),
        default=None,
        help=(
            "Lineup ingestion mode: recent refreshes from team-lists hub; "
            f"backfill additionally crawls sitemap archives (default: {default_mode})."
        ),
    )
    parser.add_argument(
        "--lineups-max-articles",
        type=int,
        default=None,
        help="Limit number of lineup articles fetched in this run.",
    )
    parser.add_argument(
        "--lineups-include-sitemap-in-recent",
        action="store_true",
        default=None,
        help="In recent mode, also use sitemap URLs (broader but slower).",
    )
    parser.add_argument(
        "--lineups-strict",
        action="store_true",
        default=None,
        help="Fail command if lineup ingestion reports errors.",
    )


def build_parser():
    parser = argparse.ArgumentParser(
        prog="footy-tipper",
        description="Footy Tipper CLI: run prep, train, inference, and send workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prep", help="Run R data preparation and write SQLite tables.")
    _add_season_args(prep)
    _add_prep_mode_args(prep, default_mode="full", choices=("full", "train", "infer"))
    _add_lineup_args(prep, default_mode="recent")

    train = subparsers.add_parser("train", help="Run training workflow.")
    _add_season_args(train)
    _add_prep_mode_args(
        train,
        default_mode="train",
        choices=("full", "train"),
        include_infer_context_arg=False,
    )
    _add_lineup_args(train, default_mode="recent")
    train.add_argument("--skip-prep", action="store_true", help="Skip R data prep and train from existing SQLite tables.")

    infer = subparsers.add_parser("infer", help="Run inference workflow.")
    _add_season_args(infer)
    _add_prep_mode_args(infer, default_mode="infer", choices=("infer", "full"))
    _add_lineup_args(infer, default_mode="recent")
    infer.add_argument("--skip-prep", action="store_true", help="Skip R data prep and infer from existing SQLite tables.")
    infer.add_argument(
        "--skip-auto-train",
        action="store_true",
        help="Do not auto-run training when model artifacts are missing.",
    )

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
    _add_llm_args(send)
    send.add_argument("--dry-run", action="store_true", help="Print email output without sending.")
    send.add_argument(
        "--force-resend",
        action="store_true",
        help="Send to the production list even if this round was already emailed.",
    )

    predict = subparsers.add_parser("predict", help="Run full prediction workflow (prep -> infer -> send).")
    _add_season_args(predict)
    _add_prep_mode_args(predict, default_mode="infer", choices=("infer", "full"))
    _add_lineup_args(predict, default_mode="recent")
    predict.add_argument("--skip-prep", action="store_true", help="Skip R data prep.")
    predict.add_argument("--skip-send", action="store_true", help="Skip send step after inference.")
    predict.add_argument(
        "--skip-auto-train",
        action="store_true",
        help="Do not auto-run training when model artifacts are missing.",
    )
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
    _add_llm_args(predict)
    predict.add_argument("--dry-run", action="store_true", help="Print email output without sending.")
    predict.add_argument(
        "--force-resend",
        action="store_true",
        help="Send to the production list even if this round was already emailed.",
    )

    lineups = subparsers.add_parser("lineups", help="Run lineup ingestion only.")
    _add_season_args(lineups)
    _add_lineup_args(lineups, default_mode="recent", include_skip=False)

    site = subparsers.add_parser("site", help="Generate the static tips site into docs/site/.")
    site.add_argument(
        "--publish",
        action="store_true",
        help="Commit and push docs/site after generating (for GitHub Pages).",
    )

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Honest nested season-out evaluation (meta-layer never sees the test season).",
    )
    _add_season_args(evaluate)
    evaluate.add_argument(
        "--seasons",
        type=int,
        default=None,
        help="Number of recent seasons to hold out (default: FOOTY_TIPPER_EVAL_SEASONS or 3).",
    )
    evaluate.add_argument(
        "--skip-prep",
        action="store_true",
        help="Skip R data prep and evaluate from existing SQLite tables.",
    )

    return parser


def main(argv=None):
    root = _project_root()
    load_dotenv(dotenv_path=root / "secrets.env")

    parser = build_parser()
    args = parser.parse_args(argv)
    env = _build_env(args)
    resolved_test_email = _resolve_test_email(getattr(args, "test_email", None))

    if args.command == "prep":
        if not args.skip_lineups:
            _run_lineups(env, root)
        _run_data_prep(env, root)
        return 0

    if args.command == "train":
        if not args.skip_lineups:
            _bootstrap_lineups_for_training_if_needed(env, root)
            _run_lineups(env, root)
        _run_train(env, skip_prep=args.skip_prep, root=root)
        return 0

    if args.command == "infer":
        if not args.skip_lineups:
            _run_lineups(env, root)
        if not _ensure_models_for_prediction(
            env,
            root,
            auto_train=not args.skip_auto_train,
            allow_lineup_bootstrap=not args.skip_lineups,
        ):
            return 1
        _run_inference(env, skip_prep=args.skip_prep, root=root)
        return 0

    if args.command == "lineups":
        _run_lineups(env, root)
        return 0

    if args.command == "site":
        try:
            from pipeline.common.use_predictions import site as site_mod
        except ModuleNotFoundError as exc:
            missing = getattr(exc, "name", "dependency")
            _log(f"Site generation requires project dependencies (missing: {missing}).")
            return 1
        db_path = root / "data" / "footy-tipper-db.sqlite"
        site_mod.generate_site(db_path, root)
        if args.publish:
            return 0 if site_mod.publish_site(root) else 1
        return 0

    if args.command == "evaluate":
        if not _model_artifacts_exist(root):
            _log("Model artifacts are missing. Run `footy-tipper train` first; evaluate reuses their tuned hyperparameters.")
            return 1
        if args.seasons is not None:
            env["FOOTY_TIPPER_EVAL_SEASONS"] = str(args.seasons)
        eval_env = env.copy()
        eval_env.setdefault("FOOTY_TIPPER_PREP_MODE", "train")
        _run_evaluate(eval_env, skip_prep=args.skip_prep, root=root)
        return 0

    if args.command == "send":
        skip_drive = args.skip_drive or args.test
        return _send_predictions(
            test_mode=args.test,
            test_email=resolved_test_email,
            skip_drive=skip_drive,
            use_llm=args.use_llm,
            dry_run=args.dry_run,
            force_resend=args.force_resend,
        )

    if args.command == "predict":
        if not args.skip_lineups:
            _run_lineups(env, root)
        if not _ensure_models_for_prediction(
            env,
            root,
            auto_train=not args.skip_auto_train,
            allow_lineup_bootstrap=not args.skip_lineups,
        ):
            return 1
        _run_inference(env, skip_prep=args.skip_prep, root=root)
        if args.skip_send:
            _log("Send step skipped.")
            return 0
        skip_drive = args.skip_drive or args.test
        return _send_predictions(
            test_mode=args.test,
            test_email=resolved_test_email,
            skip_drive=skip_drive,
            use_llm=args.use_llm,
            dry_run=args.dry_run,
            force_resend=args.force_resend,
        )

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
