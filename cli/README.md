# Footy Tipper CLI (Simple Guide)

Use this file when you just want the command that works.

## 1) One-time setup

```bash
conda env create -f environment.yml
conda activate footy-tipper
```

Notes:
- `environment.yml` now installs the CLI (`footy-tipper`) automatically.
- Most R packages install via conda; if any are still missing (like `elo`),
  the pipeline auto-installs them on first run.

## 2) Most common commands

### Fresh training run from 2012

```bash
footy-tipper train --start-year 2012
```

Notes:
- This now defaults to full prep for training.
- Missing odds are allowed by default.

### Inference run (latest pre-game data)

```bash
footy-tipper infer
```

### Full weekly run (prep + infer + send)

```bash
footy-tipper predict
```

### Send only

```bash
footy-tipper send
```

### Safe email dry run

```bash
footy-tipper predict --test --dry-run
```

OpenAI note:
- OpenAI email generation is ON by default.
- Use `--without-openai` only when you want fallback text.

## 3) Optional switches (advanced)

Only use these if needed:

```bash
--start-year 2012
--end-year 2026
--without-performance
--require-odds
--without-openai
--prep-mode full|train|infer
--infer-context-years 1
```

## 4) If a command is not found

```bash
conda activate footy-tipper
conda env update -f environment.yml
hash -r
```
