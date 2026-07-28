"""Stable, machine-facing orchestration for production prediction runs.

The human CLI deliberately has a small, safety-oriented command surface.  GitHub
Actions needs a separate entry point whose modes cannot silently fall through to
a live email send.  This module owns that exact three-mode contract.
"""

from types import SimpleNamespace

from pipeline import cli as pipeline_cli
from pipeline.common import console
from pipeline.ops.odds_gate import current_round_odds_coverage


VALID_MODES = ("test", "refresh", "live")


def run(mode: str) -> int:
    """Run one exact Actions prediction mode without ever auto-training.

    ``test`` refreshes data, calculates tips, and sends only to the configured
    test address.  ``refresh`` refreshes data and calculates tips without an
    email.  ``live`` performs the production email send.  Runtime publication
    and site publication are intentionally handled by separate Actions runner
    commands so the workflow can omit both for test runs.
    """
    if mode not in VALID_MODES:
        raise ValueError(
            f"Unknown prediction mode {mode!r}; expected one of: "
            f"{', '.join(VALID_MODES)}"
        )

    console.section("predict", f"{mode} mode")
    root = pipeline_cli._project_root()
    pipeline_cli.load_dotenv(dotenv_path=root / "secrets.env")
    prediction_records = []

    # _build_env uses getattr for optional CLI arguments, so a deliberately
    # empty namespace gives the normal production defaults without exposing
    # the retired public CLI flag surface to Actions.
    env = pipeline_cli._build_env(SimpleNamespace())
    env["FOOTY_TIPPER_PREP_MODE"] = "infer"
    env["FOOTY_TIPPER_ACTIONS_MODE"] = mode

    pipeline_cli._run_lineups(env, root)
    inference_env = env
    if pipeline_cli._feed_source(env) == "feed":
        # The legacy XML path refreshes fixtures inside R. Do that before
        # fetching odds, then rebuild from the now-frozen cache so the gate and
        # inference cannot silently refer to different rounds.
        pipeline_cli._run_data_prep(env, root)
        prediction_records += pipeline_cli._refresh_nrl_data(env, root) or []
        inference_env = dict(env)
        inference_env["FOOTY_TIPPER_FEED_SOURCE"] = "python"
        pipeline_cli._run_data_prep(inference_env, root)
    else:
        prediction_records += pipeline_cli._refresh_nrl_data(env, root) or []
        pipeline_cli._run_data_prep(env, root)

    odds_coverage = current_round_odds_coverage(
        root / "data" / "footy-tipper-db.sqlite"
    )
    if odds_coverage.complete:
        pipeline_cli._log(odds_coverage.message())
    else:
        pipeline_cli._log(
            "WARNING: "
            + odds_coverage.message()
            + " Tips without valid prices are model-only; market edges and "
            "staking must remain disabled."
        )
        if mode == "live":
            pipeline_cli._log(
                "Live delivery blocked before inference because every current-round "
                "fixture requires fresh paired H2H odds."
            )
            return 1

    if not pipeline_cli._ensure_models_for_prediction(
        env,
        root,
        auto_train=False,
        allow_lineup_bootstrap=False,
    ):
        return 1
    prediction_records += (
        pipeline_cli._run_inference(inference_env, skip_prep=True, root=root) or []
    )
    pipeline_cli._render_prediction_results(prediction_records)

    if mode == "refresh":
        pipeline_cli._log("Refresh mode complete. Email send skipped.")
        return 0

    test_mode = mode == "test"
    return pipeline_cli._send_predictions(
        test_mode=test_mode,
        test_email=pipeline_cli._resolve_test_email(None),
        # A test run must not publish predictions to Drive.  The workflow also
        # omits runtime-state and site publication for this mode.
        skip_drive=test_mode,
        use_llm=True,
        dry_run=False,
        force_resend=False,
    )
