# Getting Started

## 1) Setup

```bash
conda env create -f environment.yml
conda activate footy-tipper
cp secrets.env.example secrets.env
```

If your env already exists, update it after pulling changes:

```bash
conda env update -f environment.yml --prune
```

Edit `secrets.env` with your feed credentials.

## 2) Core Weekly Flow

```bash
footy-tipper train
footy-tipper predict
footy-tipper send --test --dry-run
```

Simple default:
- `footy-tipper train` handles historical lineup bootstrap when needed, then lineup refresh + data prep + training.
- `footy-tipper predict` handles lineup refresh + inference + send flow.
- If models are missing, `predict`/`infer` auto-run training unless you pass `--skip-auto-train`.

## 3) Required Files

Expected in repo root:
- `secrets.env`
- `service-account-token.json` (only needed for Drive/email integration)

## 4) Required Env Vars

Feed/API:
- `PASSWORD`
- `BASE_URL`
- `NRL_FIXTURES_EXTENTION`
- `NRL_ROUND_LADDER_EXTENTION`
- `NRL_PERFORMANCE_EXTENTION`

Send/integration:
- `FOLDER_ID`
- `FOLDER_URL`
- `MY_EMAIL`
- `EMAIL_PASSWORD`

Optional:
- `OPENAI_KEY`
- `OPENAI_MODEL`
- `FOOTY_TIPPER_TEST_EMAIL`
- `FOOTY_TIPPER_LINEUPS_MODE` (`recent` or `backfill`)
- `FOOTY_TIPPER_LINEUPS_MAX_ARTICLES`
- `FOOTY_TIPPER_LINEUPS_BACKFILL_MAX_ARTICLES`
- `FOOTY_TIPPER_LINEUPS_STRICT`

## 5) Season Controls

- `FOOTY_TIPPER_START_YEAR` (default `2018`)
- `FOOTY_TIPPER_END_YEAR` (default current year)
- `FOOTY_TIPPER_INCLUDE_PERFORMANCE` (default `true`)

## 6) Smoke Checks

```bash
python -m compileall -q pipeline footy-tipper-train.py footy-tipper-predict.py
Rscript -e "parse(file='pipeline/data-prep.R')"
python -m unittest discover -s tests -p 'test_*.py' -v
```
