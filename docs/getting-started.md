# Getting started

The model has several moving pieces, but the safe entry is deliberately boring: create the environment, supply only the credentials your workflow needs, and dry-run before sending anything to anybody.

## Prerequisites

- Git
- Conda or Miniconda
- R and Python are installed through [`environment.yml`](../environment.yml)
- Feed credentials for preparation, if the local SQLite cache is not already usable
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
| Prepare/train/infer | `PASSWORD`, `BASE_URL`, `NRL_FIXTURES_EXTENTION`, `NRL_ROUND_LADDER_EXTENTION`; `NRL_PERFORMANCE_EXTENTION` when performance is enabled | season, odds, prep-mode, and lineup controls |
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
- `FOOTY_TIPPER_LINEUPS_ENABLED=true`
- `FOOTY_TIPPER_LINEUPS_MODE=recent`
- `FOOTY_TIPPER_LINEUPS_STRICT=false`
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

## Normal weekly workflow

```bash
footy-tipper train
footy-tipper predict
```

`train` bootstraps historical lineups when needed, refreshes recent lineups, runs smart-cache preparation, and trains the models. `predict` refreshes lineups, prepares a narrow inference window, auto-trains if required artifacts are missing, writes predictions, and runs distribution.

Useful controlled variants:

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
| Missing model artifacts | `infer` and `predict` auto-train by default. Remove `--skip-auto-train` or run `footy-tipper train`. |
| Duplicate-send refusal | The `(season, round)` ledger is doing its job. Use `--force-resend` only after verifying a genuine resend is intended. |
| Drive integration skipped | Check `FOLDER_ID`, the service-account token, and dependency installation. Local prediction can still succeed. |
| GitHub Pages returns 404 | The URL is expected, not yet proven live. Generate `docs/site/`, then enable Pages from `main` and `/docs`. |

Continue with the [CLI reference](cli-reference.md) or the [operations runbook](operations-reliability.md).
