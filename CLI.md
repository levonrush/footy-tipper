# Footy Tipper CLI (Reference)

Use this file as the command reference.

If you want the fastest setup and daily commands, use `cli/README.md`.

## Quick Start

```bash
conda env create -f environment.yml
conda activate footy-tipper
cp secrets.env.example secrets.env
# edit secrets.env with your feed credentials
footy-tipper --help
```

Notes:
- `environment.yml` installs the CLI entrypoint (`footy-tipper`) automatically.
- Most R packages install via conda. If one is missing (for example `elo`),
  the pipeline installs it automatically on first run.

## Common Commands

```bash
footy-tipper train --start-year 2012
footy-tipper infer
footy-tipper predict
footy-tipper send
footy-tipper predict --test --dry-run
```

## Command Reference

### `prep`

```bash
footy-tipper prep
footy-tipper prep --prep-mode infer --infer-context-years 1
```

### `train`

```bash
footy-tipper train
footy-tipper train --start-year 2012
footy-tipper train --skip-prep
```

### `infer`

```bash
footy-tipper infer
footy-tipper infer --skip-prep
```

### `predict`

```bash
footy-tipper predict
footy-tipper predict --skip-prep
footy-tipper predict --skip-send
footy-tipper predict --test --dry-run
```

### `send`

```bash
footy-tipper send
footy-tipper send --test --test-email levon.rush@gmail.com
footy-tipper send --dry-run
footy-tipper send --skip-drive
footy-tipper send --without-openai
```

## Key Options

Use only when needed:

```bash
--start-year 2012
--end-year 2026
--without-performance
--require-odds
--without-openai
--prep-mode full|train|infer
--infer-context-years 1
```

Defaults:
- `train`: `--prep-mode full`
- `infer`: `--prep-mode infer`
- `predict`: `--prep-mode infer`
- `prep`: `--prep-mode full`
- Missing odds: allowed by default
- OpenAI email generation: enabled by default
- `--test-email`: `FOOTY_TIPPER_TEST_EMAIL` from env/secrets, else `levon.rush@gmail.com`

## Required Runtime Files

Expected in project root:
- `secrets.env` (copy from `secrets.env.example`)
- `service-account-token.json` (only needed for Drive/email flows)

Required keys in `secrets.env`:
- feed: `PASSWORD`, `BASE_URL`, `NRL_FIXTURES_EXTENTION`, `NRL_ROUND_LADDER_EXTENTION`, `NRL_PERFORMANCE_EXTENTION`
- send: `FOLDER_ID`, `FOLDER_URL`, `MY_EMAIL`, `EMAIL_PASSWORD`
- optional: `OPENAI_KEY`, `OPENAI_MODEL`, `FOOTY_TIPPER_TEST_EMAIL`

## Smoke Checks

```bash
python -m compileall -q pipeline footy-tipper-train.py footy-tipper-predict.py
Rscript -e "parse(file='pipeline/data-prep.R')"
python -m unittest discover -s tests -p 'test_*.py' -v
```

## If `footy-tipper` Is Not Found

```bash
conda activate footy-tipper
conda env update -f environment.yml
hash -r
```
