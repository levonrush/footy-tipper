# Getting started

The model has several moving pieces, but the safe entry is deliberately boring: create the environment, supply only the credentials your workflow needs, and dry-run before sending anything to anybody.

## Prerequisites

- Git
- Conda or Miniconda
- R and Python are installed through [`environment.yml`](../environment.yml)
- Network access for the default nrl.com ingestion, or legacy XML credentials only when explicitly using the rollback feed
- A Google service-account token only for Drive-backed state or distribution
- SMTP credentials only for an actual email send

Run commands from the repository root. Runtime code must not depend on a hardcoded local path.

## Create the environment

```bash
conda env create -f environment.yml
conda activate footy-tipper
cp secrets.env.example secrets.env
```

For an existing checkout:

```bash
conda env update -f environment.yml --prune
conda activate footy-tipper
hash -r
```

`secrets.env` and `service-account-token.json` are ignored by git. Never commit either file.

## Secrets by workflow

Use [`secrets.env.example`](../secrets.env.example) as the field-level reference.

| Workflow | Required configuration | Optional configuration |
| --- | --- | --- |
| Prepare/train/infer (default) | None for public nrl.com draw/match-centre ingestion | Betfair credentials, historical-odds URL, season, prep-mode, and lineup controls |
| Legacy XML rollback | `PASSWORD`, `BASE_URL`, `NRL_FIXTURES_EXTENTION`, `NRL_ROUND_LADDER_EXTENTION`; `NRL_PERFORMANCE_EXTENTION` when performance is enabled | `FOOTY_TIPPER_FEED_SOURCE=feed` selects this path |
| Drive state | `FOLDER_ID` plus `service-account-token.json` | `FOLDER_URL`, backup controls |
| Live email | `MY_EMAIL`, `EMAIL_PASSWORD` | `FOOTY_TIPPER_TEST_EMAIL`, recipient/state controls |
| Generated email prose | `ANTHROPIC_API_KEY` | `CLAUDE_MODEL`; `--no-llm` uses deterministic copy |
| Generated banner | `OPENAI_KEY` | `OPENAI_MODEL`; banner failure falls back without blocking delivery |

Claude/Anthropic generates email copy. OpenAI is used only for optional banner generation.

## Defaults that matter

- `FOOTY_TIPPER_START_YEAR=2010`
- `FOOTY_TIPPER_END_YEAR=<current year>`
- `FOOTY_TIPPER_INCLUDE_PERFORMANCE=true`
- `FOOTY_TIPPER_REQUIRE_ODDS=false`
- `FOOTY_TIPPER_INFER_CONTEXT_YEARS=1`
- `FOOTY_TIPPER_FEED_SOURCE=python`
- `FOOTY_TIPPER_NRL_DATA_ENABLED=true`
- `FOOTY_TIPPER_LINEUPS_ENABLED=true`
- `FOOTY_TIPPER_LINEUPS_MODE=recent`
- `FOOTY_TIPPER_LINEUPS_STRICT=false`
- `FOOTY_TIPPER_TUNE_ITER=100`
- `FOOTY_TIPPER_TEST_EMAIL=levon_rush@hotmail.com` when not overridden

Performance feed availability varies by season. With performance enabled, missing required performance data fails clearly. Lineup ingestion is different: it fails soft by default and becomes fatal only with `--lineups-strict` or `FOOTY_TIPPER_LINEUPS_STRICT=true`.

## First safe run

Inspect the interface and validate syntax first:

```bash
footy-tipper --help
python -m compileall -q pipeline
Rscript -e "parse(file='pipeline/data-prep.R')"
```

Then exercise the full prediction composition without sending email or writing Drive state:

```bash
footy-tipper predict --test --dry-run --skip-drive
```

This can fetch data, refresh lineups, write local SQLite/model artifacts, and auto-train when required models are missing. It does not send email in dry-run mode. To avoid all preparation writes and use existing local state, add `--skip-prep`; that will fail if the required tables or artifacts are absent.

For a targeted delivery check after predictions already exist:

```bash
footy-tipper send --test --dry-run --skip-drive
```

## Local production training and publication

```bash
conda activate footy-tipper

# Prevent a scheduled prediction from changing Drive state mid-training.
gh workflow disable predict.yml
# Wait until both commands list no runs, then disable once more.
gh run list --workflow predict.yml --status queued
gh run list --workflow predict.yml --status in_progress
gh workflow disable predict.yml
gh api 'repos/{owner}/{repo}/actions/workflows/predict.yml' --jq .state

footy-tipper state pull
FOOTY_TIPPER_TUNE_ITER=100 footy-tipper train

# Load and exercise the new artifacts without refreshing data or training again.
footy-tipper infer \
  --skip-prep \
  --skip-lineups \
  --skip-nrl-data \
  --skip-auto-train

footy-tipper state schedule
footy-tipper state push

gh workflow enable predict.yml
gh workflow run predict.yml -f mode=refresh
```

Local hardware is authoritative for production models. `train` bootstraps historical nrl.com/odds and lineup data when needed, refreshes current data, runs smart-cache preparation, and uses the normal 100-candidate Bayesian tuning budget. Search fits run in parallel across available cores; each LightGBM fit stays single-threaded to avoid nested CPU contention.

After disabling the workflow, wait for both queued and in-progress runs to drain (`gh run watch <run-id>`), disable it again, and confirm the API reports `disabled_manually` before pulling. The second disable protects the first cutover run from an older in-flight gate that may have started before self-enabling was removed. `state pull` then establishes the last published DB/model baseline. The explicit `infer` is the publication gate: if training or validation fails, **do not run `state push`**. Re-enable `predict.yml` so the last-known-good Drive state remains in service. A successful push publishes the consistent DB, required model artifacts, and derived schedule; the manual `refresh` then proves that Actions can consume them without sending email.

GitHub Actions polls every 15 minutes. On the first-game day of the next unsent round, the first poll at or after 11:00 `Australia/Sydney` pulls Drive state and runs prediction with `--skip-auto-train`. GitHub never performs a production retrain.

## Useful local variants

```bash
footy-tipper train --skip-prep
footy-tipper predict --skip-send
footy-tipper send --test --test-email you@example.com
footy-tipper evaluate --skip-prep --seasons 3
```

## Smoke checks

```bash
python -m compileall -q pipeline
Rscript -e "parse(file='pipeline/data-prep.R')"
python -m unittest discover -s tests -p 'test_*.py' -v
```

Run `git diff --check` before committing documentation or code.

## Common failures

| Symptom | Meaning and response |
| --- | --- |
| `footy-tipper: command not found` | Activate the Conda environment, update it from `environment.yml`, then run `hash -r`. |
| No pre-game rows | Offseason or feed timing. Inference/send exits cleanly; do not manufacture fixtures. |
| Performance data missing | Disable it explicitly with `--without-performance` only when that is an acceptable model change. Otherwise fix the provider/configuration. |
| Lineup scrape errors | Default behavior logs and continues. Re-run recent mode, use backfill for history, or enable strict mode while diagnosing. |
| Missing model artifacts locally | Standalone `infer` and `predict` auto-train by default. Remove `--skip-auto-train` or run `footy-tipper train`. |
| Missing model artifacts in Actions | Actions intentionally supplies `--skip-auto-train`. Publish a complete validated model set from local hardware; do not add cloud auto-training back. |
| Local training failed | Do not run `state push`. Re-enable `predict.yml`; Drive retains the last-known-good models. |
| Duplicate-send refusal | The `(season, round)` ledger is doing its job. Use `--force-resend` only after verifying a genuine resend is intended. |
| Drive integration skipped | Check `FOLDER_ID`, the service-account token, and dependency installation. Local prediction can still succeed. |
| GitHub Pages returns 404 | The URL is expected, not yet proven live. Generate `docs/site/`, then enable Pages from `main` and `/docs`. |

Continue with the [CLI reference](cli-reference.md) or the [operations runbook](operations-reliability.md).
