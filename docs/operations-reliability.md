# Operations, Reliability, and Safety

This doc covers operational contracts and "what happens if I run it twice" behavior.

## 1) Prediction Storage Contract

`predictions_table` is keyed by `game_id` and upserted.

What that means:
- rerunning inference updates existing game rows
- it does not duplicate rows for the same game
- latest successful run is what send uses

Relevant SQL:
- `pipeline/common/sql/create_table.sql`
- `pipeline/common/sql/insert_into_table.sql`

## 2) Which Predictions Get Sent

Send query is intentionally scoped to:
- latest `competition_year` with `Pre Game` rows
- minimum `round_id` in that year

So historical/stale rounds are excluded from send output.

Relevant SQL:
- `pipeline/common/sql/prediction_table.sql`

## 3) Offseason and Empty Data Behavior

If there are no pre-game fixtures:
- send exits cleanly
- no crash path

This protects automation jobs outside active rounds.

## 4) Optional Integrations Degrade Safely

If optional providers are missing:
- Google deps missing -> upload/send steps skip with clear message
- OpenAI missing -> deterministic fallback email copy

Core tipping pipeline continues where possible.

Lineup ingestion is also fail-soft by default:
- parsing/network issues are logged
- train/infer continue with fallback lineup feature defaults
- enable strict failure with `--lineups-strict` (or `FOOTY_TIPPER_LINEUPS_STRICT=true`)

## 5) Joker Reliability Controls

Joker recommendation includes guardrails so it does not overreact when odds coverage is thin.

Defaults:
- minimum rounds with odds before `PLAY`
- minimum coverage per round
- minimum lead over next-best round

Single-use reliability:
- joker usage is persisted in `joker_usage` keyed by `competition_year`
- if rerun in the same recorded round, messaging stays sticky as `PLAY` ("already locked")
- once a season is marked used, recommendations are forced to hold
- test sends read this state but do not write it
- production sends write only after successful production email send

## 6) Scheduling (GitHub Actions)

Scheduling runs in GitHub Actions on the public repo (free minutes, no Mac
involved). Three workflows in `.github/workflows/`:

- `build-image.yml`: rebuilds `ghcr.io/levonrush/footy-tipper:latest` when
  Dockerfile / requirements.txt / install.R change on main.
- `predict.yml`: hourly gate. A tiny job downloads `state/schedule.json`
  from Drive and decides: `send` when now is inside the window from 6h before
  the first kickoff of the next unsent round until 12h after kickoff (the
  grace gives an hourly retry loop; the `email_sends` ledger stops double
  emails), `refresh` (predict `--skip-send`) when schedule.json is over 8
  days old, otherwise `skip`. Non-skip decisions run the full predict job in
  the pipeline container: pull state from Drive, predict, push state,
  publish site.
- `train.yml`: Monday 19:00 UTC (Tuesday 5am AEST / 6am AEDT): pull state,
  train, push state. Tuning is bounded by the repo variable
  `FOOTY_TIPPER_TUNE_ITER`; a timeout leaves the previous models in Drive
  untouched.

State (the SQLite DB, models/, and schedule.json) lives in a `state/` folder
under the Drive `FOLDER_ID`, synced by `footy-tipper state pull|push`
(`pipeline/ops/state_sync.py`). Seed it once from a machine that has the DB
and models: `footy-tipper state push`.

Manual triggers (predict defaults to `mode=test`, which emails only
`FOOTY_TIPPER_TEST_EMAIL` and never writes the send ledger):

```bash
gh workflow run predict.yml -f mode=test     # safe test email to yourself
gh workflow run predict.yml -f mode=live     # real production send
gh workflow run predict.yml -f mode=refresh  # no email, refresh data/state
gh workflow run train.yml                    # retrain now
gh run watch                                 # follow the latest run
```

Or on the website (works from a phone): repo, Actions tab, pick the
workflow, "Run workflow" button.

Secrets are repo Actions secrets: `SECRETS_ENV` (entire secrets.env file) and
`SERVICE_ACCOUNT_JSON` (service-account-token.json). Rotate with
`gh secret set SECRETS_ENV < secrets.env`. The repo is public, so never run
`--dry-run` in Actions (it prints the full email body into public logs); the
workflows mask every secrets.env value with `::add-mask::`.

Watch runs in the Actions tab, not `logs/*.log` (those were launchd-era).
The hourly gate re-enables both scheduled workflows via `gh workflow enable`,
so GitHub's 60-day inactivity auto-disable cannot silently stop them over the
offseason.

## 7) Send Idempotency, Backups, and the Site

- Every production send is recorded in the `email_sends` SQLite table keyed by
  (competition_year, round_id). Re-runs refuse to email the list again without
  `--force-resend`.
- After a successful production send: the static site under `docs/site/` is
  regenerated, and a gzipped consistent DB snapshot is uploaded to a
  `backups/` folder under the Drive `FOLDER_ID` (newest 8 kept). Disable with
  `FOOTY_TIPPER_DB_BACKUP=false`.
- GitHub Pages: enable once in repo Settings -> Pages -> Deploy from branch ->
  `main` / `/docs`. Then `footy-tipper site --publish` (or a manual push of
  `docs/site/`) updates the public page, and `FOOTY_TIPPER_SITE_URL` makes the
  email link to it.
- Lineup feature merge failures during train/infer print full tracebacks and
  can be made fatal with `FOOTY_TIPPER_LINEUP_FEATURES_STRICT=true`.

## 8) Recommended Weekly Runbook

1. `footy-tipper predict --test --dry-run`
2. sanity-check round scope + joker call + value picks
3. run production send

Bootstrap note:
- `footy-tipper train` auto-runs one historical lineup backfill when needed
- weekly `predict` keeps using the faster recent-refresh path unless auto-training is required

## 9) Debugging Checklist

When output looks wrong, check:
- round/year selected by `prediction_table.sql`
- row counts in `inference_data` and `predictions_table`
- lineup ingestion summary (`snapshots_inserted`, `entries_inserted`, parse failures)
- odds coverage by round
- `joker_usage` row for the active `competition_year`
- which joker strategy source was used (`explicit_env`, `policy_auto`, fallback)

## 10) Security / Hygiene

Never commit:
- `secrets.env`
- `service-account-token.json`
- credentials/API keys

Keep `.gitignore` protections intact.
