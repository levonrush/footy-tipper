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
```

### train

```bash
footy-tipper train
footy-tipper train --start-year 2012
footy-tipper train --skip-prep
```

### infer

```bash
footy-tipper infer
footy-tipper infer --skip-prep
```

### predict

```bash
footy-tipper predict
footy-tipper predict --skip-prep
footy-tipper predict --skip-send
footy-tipper predict --test --dry-run
```

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
```

## Defaults

- OpenAI email generation is enabled by default.
- Missing odds are allowed by default.
- `--test-email` defaults to `FOOTY_TIPPER_TEST_EMAIL`, else `levon.rush@gmail.com`.

## If `footy-tipper` Is Not Found

```bash
conda activate footy-tipper
conda env update -f environment.yml
hash -r
```
