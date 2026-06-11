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

## 6) Scheduling (launchd)

Scheduling runs on the host Mac via launchd (Docker Compose has no scheduler;
the old compose cron labels did nothing).

```bash
./ops/install-launchd.sh            # install/refresh both jobs
./ops/install-launchd.sh uninstall  # remove them
launchctl list | grep footytipper   # check they're loaded
launchctl kickstart gui/$UID/com.footytipper.predict  # run one now
```

- `com.footytipper.train` — Tuesday 06:00 local, runs `footy-tipper train`
- `com.footytipper.predict` — Thursday 15:00 local, runs `footy-tipper predict`
- Logs: `logs/train.log`, `logs/predict.log`
- launchd skips a trigger if the Mac is asleep at that moment; keep it awake
  around run times (or plug it in with Energy Saver set accordingly).

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
