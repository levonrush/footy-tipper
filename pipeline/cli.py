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


DEFAULT_TEST_EMAIL = "levon_rush@hotmail.com"
REQUIRED_MODEL_FILES = ("home_model.pkl", "away_model.pkl", "model_manifest.json")
CONFIRMED_LIVE_ROUND_ENV = "FOOTY_TIPPER_CONFIRMED_LIVE_ROUND"
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
    env.setdefault("R_LIBS_USER", os.path.expanduser("~/R/library"))
    # Child scripts (lineups.py, inference.py, train.py) import the `pipeline`
    # package; make that work from a bare checkout with no editable install.
    root = str(_project_root())
    existing_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = root if not existing_path else root + os.pathsep + existing_path
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


def _run_nrl_data(env, root, action, extra_args=None):
    cmd = [sys.executable, str(root / "pipeline" / "nrl_data.py"), action]
    if extra_args:
        cmd.extend(extra_args)
    _run_command(cmd, env, cwd=root)


def _run_odds(env, root, action, extra_args=None):
    cmd = [sys.executable, str(root / "pipeline" / "odds.py"), action]
    if extra_args:
        cmd.extend(extra_args)
    _run_command(cmd, env, cwd=root)


def _feed_source(env):
    return str(env.get("FOOTY_TIPPER_FEED_SOURCE", "python")).strip().lower() or "python"


def _nrl_data_enabled(env):
    return _to_bool(env.get("FOOTY_TIPPER_NRL_DATA_ENABLED"), True)


def _nrl_backfill_bootstrapped(root: pathlib.Path) -> bool:
    db_path = root / "data" / "footy-tipper-db.sqlite"
    if not db_path.exists():
        return False
    try:
        with sqlite3.connect(str(db_path)) as con:
            row = con.execute(
                """
                SELECT 1 FROM sqlite_master WHERE type = 'table'
                  AND name = 'nrl_ingest_runs'
                """
            ).fetchone()
            if not row:
                return False
            row = con.execute(
                """
                SELECT 1 FROM nrl_ingest_runs
                WHERE mode = 'backfill'
                  AND status IN ('completed', 'completed_with_errors')
                LIMIT 1
                """
            ).fetchone()
            return bool(row)
    except Exception:
        return False


def _odds_backfill_bootstrapped(root: pathlib.Path) -> bool:
    db_path = root / "data" / "footy-tipper-db.sqlite"
    if not db_path.exists():
        return False
    try:
        with sqlite3.connect(str(db_path)) as con:
            row = con.execute(
                """
                SELECT 1 FROM sqlite_master WHERE type = 'table'
                  AND name = 'odds_history'
                """
            ).fetchone()
            if not row:
                return False
            row = con.execute(
                "SELECT 1 FROM odds_history WHERE source = 'aussportsbetting' LIMIT 1"
            ).fetchone()
            return bool(row)
    except Exception:
        return False


def _refresh_nrl_data(env, root, include_bootstrap=False):
    """Run nrl.com + odds ingestion ahead of R data prep.

    Skipped entirely in legacy feed mode (R fetches the XML feed itself).
    Individual steps fail soft; prep proceeds on cached data.
    """
    if _feed_source(env) == "feed":
        _log("Feed source is 'feed'; skipping nrl.com ingestion (legacy XML path).")
        return
    if not _nrl_data_enabled(env):
        _log("nrl.com ingestion disabled via FOOTY_TIPPER_NRL_DATA_ENABLED=false.")
        return

    if include_bootstrap and _to_bool(
        env.get("FOOTY_TIPPER_NRL_DATA_AUTO_BACKFILL"), True
    ):
        if not _nrl_backfill_bootstrapped(root):
            _log("Historical match-stats backfill not found. Running one-time nrl.com backfill.")
            _run_nrl_data(env, root, "backfill")
        if not _odds_backfill_bootstrapped(root):
            _log("Historical odds backfill not found. Running one-time odds backfill.")
            _run_odds(env, root, "backfill")

    _run_nrl_data(env, root, "refresh")
    _run_odds(env, root, "live")


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
            "Model artifacts are missing. Run `footy-tipper update-model` to publish "
            "a production model, or use `footy-tipper advanced model train` locally."
        )
        return False

    _log("Model artifacts are missing. Explicit advanced auto-train is starting now.")
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

    # A manually dispatched live workflow is authorized for one exact round.
    # Re-check after inference resolves the actual outgoing predictions and
    # before touching either idempotency store or any production side effect.
    confirmed_round_raw = os.getenv(CONFIRMED_LIVE_ROUND_ENV)
    confirmed_round_text = (confirmed_round_raw or "").strip()
    if not test_mode and not dry_run and confirmed_round_text:
        try:
            confirmed_round = int(confirmed_round_text)
        except (TypeError, ValueError):
            _log(
                f"Live send refused: {CONFIRMED_LIVE_ROUND_ENV} must be a whole round number."
            )
            return 3
        if confirmed_round < 1 or send_round_id != confirmed_round:
            actual = "unknown" if send_round_id is None else str(send_round_id)
            _log(
                "Live send refused: the operator confirmed round "
                f"{confirmed_round}, but the outgoing predictions resolve to round {actual}. "
                "No production state was changed and no email was sent."
            )
            return 3

    def _existing_delivery_result(marker, reason):
        marker = marker or {}
        if marker.get("status") == "sent":
            if sf.record_email_send(
                db_path,
                send_year,
                send_round_id,
                recipients_count=marker.get("recipients_count"),
                source="drive_delivery_reconciliation",
            ):
                _log(
                    f"Drive already confirms delivery for {send_year} round {send_round_id}. "
                    "Reconciled the local DB ledger without sending another email."
                )
                return 0
            _log(
                "Drive confirms this round was sent, but the local DB ledger "
                "could not be reconciled. No email was sent."
            )
            return 1
        _log(
            f"Production delivery blocked for {send_year} round {send_round_id}: "
            f"{reason}. Marker status is {marker.get('status', 'unknown')}; "
            "pending means the prior SMTP outcome must be checked by a human."
        )
        return 3

    # Read both idempotency stores before doing the comparatively expensive
    # rendering work. The pending marker itself is claimed immediately before
    # SMTP, so a pre-render failure cannot strand a round as uncertain.
    delivery_claim = None
    if not test_mode and not dry_run:
        prior_send = sf.email_send_already_recorded(db_path, send_year, send_round_id)
        if prior_send:
            _log(
                f"Production email already sent for {send_year} round {send_round_id} "
                f"(recorded {prior_send.get('sent_at_utc')} UTC). "
                "No email was sent. Reconcile the delivery records explicitly if "
                "the recorded outcome is wrong."
            )
            return 0
        try:
            from pipeline.ops import delivery_state

            existing_marker = delivery_state.get_delivery(
                root, send_year, send_round_id
            )
        except Exception as exc:
            _log(f"Could not check the production delivery safety marker: {exc}")
            return 1
        if existing_marker:
            return _existing_delivery_result(
                existing_marker,
                f"delivery is already {existing_marker.get('status', 'unknown')}",
            )

    # Competition-aware tip strategy: advisory logs deviations, auto applies
    # them to the outgoing email (predictions_table itself is never changed).
    comp_strategy = sf.get_comp_strategy_recommendation(db_path, root, predictions)
    if comp_strategy.get("status") != "off":
        _log(comp_strategy.get("headline", "Comp strategy unavailable"))
        if comp_strategy.get("detail"):
            _log(comp_strategy["detail"])
    if comp_strategy.get("mode") == "auto" and comp_strategy.get("tips_changed"):
        predictions = sf.apply_comp_strategy_to_predictions(predictions, comp_strategy)
        _log(f"Comp strategy AUTO: {comp_strategy['tips_changed']} tip(s) adjusted in this send.")
    if not test_mode and not dry_run:
        sf.persist_comp_strategy_decision(db_path, comp_strategy, predictions)

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
        comp_strategy=comp_strategy,
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
        return 0 if sent else 1

    if dry_run:
        _log("Dry run enabled. Production email was not sent.")
        _log(f"Subject: {subject}")
        _log("")
        _log(email_body)
        return 0

    # Resolve every deterministic production-email dependency before claiming
    # the Drive marker. A missing secret/token/dependency or an unreadable or
    # invalid recipient sheet proves that SMTP was never attempted, so it must
    # not strand the round as an uncertain pending delivery.
    try:
        prepared_delivery = sf.prepare_email_delivery(
            "footy-tipper-email-list",
            os.getenv("MY_EMAIL"),
            os.getenv("EMAIL_PASSWORD"),
            json_path,
        )
    except sf.EmailPreparationError as exc:
        _log(
            f"Production email preparation failed before delivery was claimed: {exc} "
            "No email was sent and no pending marker was created."
        )
        return 1
    except Exception as exc:
        _log(
            "Production email preparation failed unexpectedly before delivery was "
            f"claimed ({type(exc).__name__}). No email was sent and no pending marker "
            "was created."
        )
        return 1

    try:
        from pipeline.ops import delivery_state

        delivery_claim = delivery_state.begin_delivery(
            root,
            send_year,
            send_round_id,
            source="actions_live" if os.getenv("GITHUB_ACTIONS") else "local_live",
        )
    except Exception as exc:
        _log(f"Could not create the production delivery safety marker: {exc}")
        return 1
    if not delivery_claim.get("allowed"):
        return _existing_delivery_result(
            delivery_claim.get("marker"),
            delivery_claim.get("reason", "another runner claimed this round"),
        )

    sent = sf.send_emails(
        subject,
        email_body,
        prepared_delivery,
        html_message=email_html,
        inline_images=inline_images,
    )
    if not sent:
        _log("Production email flow skipped or failed. Joker usage state unchanged.")
        return 1

    recipients_count = sent if isinstance(sent, int) and not isinstance(sent, bool) else None
    try:
        from pipeline.ops import delivery_state

        marker = delivery_claim.get("marker") if delivery_claim else {}
        delivery_state.mark_sent(
            root,
            send_year,
            send_round_id,
            marker.get("attempt_id"),
            recipients_count=recipients_count,
        )
        _log(f"Drive delivery marker recorded as sent for {send_year} round {send_round_id}.")
    except Exception as exc:
        _log(
            "SMTP reported success, but the Drive marker could not be finalized. "
            f"The round remains blocked as uncertain and must not be resent automatically: {exc}"
        )
        return 1

    if sf.record_email_send(
        db_path,
        send_year,
        send_round_id,
        recipients_count=recipients_count,
        source="cli_send_production",
    ):
        _log(f"Send recorded in ledger for {send_year} round {send_round_id}.")
    else:
        _log(
            "Drive says this round was sent, but the local DB ledger could not be updated. "
            "Stopping so an operator can reconcile the runtime state."
        )
        return 1

    # Refresh the static site so the public page matches what was just sent.
    try:
        from pipeline.common.use_predictions import site as site_mod

        site_mod.generate_site(db_path, root)
    except Exception as exc:
        _log(f"Site refresh skipped ({exc}).")

    # Weekly recoverable snapshot of the runtime database.
    if _to_bool(os.getenv("FOOTY_TIPPER_DB_BACKUP"), True):
        sf.backup_db_to_drive(db_path, json_path, os.getenv("FOLDER_ID"))
    else:
        _log("DB backup disabled via FOOTY_TIPPER_DB_BACKUP.")

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


def build_parser():
    """Build the Footy Tipper 1.0 human-facing parser.

    Pipeline helpers remain in this module for the explicit advanced and
    machine interfaces, while the small operator surface lives separately.
    """
    from pipeline.operator_cli import build_parser as build_operator_parser

    return build_operator_parser()


def main(argv=None):
    from pipeline.operator_cli import run

    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
