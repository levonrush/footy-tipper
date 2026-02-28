# CLI Reference

## Quick Start

```bash
conda env create -f environment.yml
conda activate footy-tipper
cp secrets.env.example secrets.env
footy-tipper --help
```

## Common Commands

```bash
footy-tipper prep
footy-tipper lineups
footy-tipper train --start-year 2012
footy-tipper infer
footy-tipper predict
footy-tipper send
footy-tipper send --test --test-email you@example.com
footy-tipper send --test --dry-run
```

## Command Details

### prep

```bash
footy-tipper prep
footy-tipper prep --prep-mode infer --infer-context-years 1
footy-tipper prep --skip-lineups
```

### lineups

```bash
footy-tipper lineups
footy-tipper lineups --lineups-mode recent --lineups-max-articles 80
footy-tipper lineups --lineups-mode backfill --start-year 2018 --end-year 2026 --lineups-max-articles 2000
```

### train

```bash
footy-tipper train
footy-tipper train --start-year 2012
footy-tipper train --skip-prep
footy-tipper train --skip-lineups
```

Default behavior:
- if historical lineup backfill has not been bootstrapped for the requested training window, `train` runs a one-time backfill first
- then it runs the normal recent lineup refresh
- then it runs prep + model training

### infer

```bash
footy-tipper infer
footy-tipper infer --skip-prep
footy-tipper infer --skip-lineups
footy-tipper infer --skip-auto-train
```

### predict

```bash
footy-tipper predict
footy-tipper predict --skip-prep
footy-tipper predict --skip-send
footy-tipper predict --test --dry-run
footy-tipper predict --skip-lineups
footy-tipper predict --skip-auto-train
```

`infer` and `predict` automatically run training if required model artifacts are missing.  
Use `--skip-auto-train` only if you want hard failure instead.

When auto-training is triggered, the same historical lineup bootstrap logic from `train` is used unless `--skip-lineups` is set.

### send

```bash
footy-tipper send
footy-tipper send --test --test-email levon.rush@gmail.com
footy-tipper send --dry-run
footy-tipper send --skip-drive
footy-tipper send --without-openai
```

## Useful Options

```bash
--start-year 2012
--end-year 2026
--without-performance
--require-odds
--without-openai
--prep-mode full|train|infer
--infer-context-years 1
--skip-lineups
--lineups-mode recent|backfill
--lineups-max-articles 80
--lineups-include-sitemap-in-recent
--lineups-strict
--skip-auto-train
```

Advanced env-only default:
- `FOOTY_TIPPER_LINEUPS_BACKFILL_MAX_ARTICLES=2000`

## Defaults

- OpenAI email generation is enabled by default.
- Missing odds are allowed by default.
- Lineup ingestion runs before `prep/train/infer/predict` by default.
- `train` auto-bootstraps historical lineup backfill once when needed.
- `--test-email` defaults to `FOOTY_TIPPER_TEST_EMAIL`, else `levon.rush@gmail.com`.

## If `footy-tipper` Is Not Found

```bash
conda activate footy-tipper
conda env update -f environment.yml
hash -r
```
