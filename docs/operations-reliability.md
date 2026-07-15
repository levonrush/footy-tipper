# Operations, reliability, and safety

This is the runbook for what owns state, what happens on a rerun, and how the automated path fails without turning one missing provider into a small constitutional crisis.

![Actions gate, Drive state, prediction, send, backup, and Pages flow](diagrams/operations-state.svg)

[Editable Mermaid source](diagrams/operations-state.mmd)

## State ownership

| State | Owner | Write rule |
| --- | --- | --- |
| Code and hand-written docs | Git | Human-reviewed commits |
| `data/footy-tipper-db.sqlite` | Local run / Drive `state/` | R prep, inference, and operational ledgers |
| `models/` | Training / Drive `state/` | Replace only after successful training and push |
| `schedule.json` | State scheduler / Drive `state/` | Derived from next unsent round |
| `docs/site/` | Site generator / GitHub Pages branch path | Generated; publish explicitly or from Actions |
| Email delivery | SMTP | One live `(competition_year, round_id)` unless forced |

`footy-tipper state pull` restores DB and models before cloud work. `state push` uploads DB, models, and schedule after successful work. All stateful Actions jobs share concurrency group `footy-tipper-state` with cancellation disabled.

## Prediction and send idempotency

- `predictions_table` is keyed/upserted by `game_id`; rerunning inference replaces the game's prediction rather than duplicating it.
- [`prediction_table.sql`](../pipeline/common/sql/prediction_table.sql) selects the latest year with pre-game rows and the minimum round in that year.
- `email_sends` prevents a second live email for the same season/round.
- `--force-resend` bypasses only that email guard and must be an explicit operator decision.
- `joker_usage` is written only after a successful live send whose recommendation is `PLAY`.
- `odds_snapshots` is append-only by preparation run; repeated observations are evidence, not duplicates.
- Lineup content hashes deduplicate identical article versions and repair old zero-entry snapshots in place.

## GitHub Actions

[The repository Actions dashboard](https://github.com/levonrush/footy-tipper/actions) is the live scheduler.

| Workflow | Trigger | Contract |
| --- | --- | --- |
| `predict.yml` | Hourly at minute 7; manual `test`, `live`, or `refresh` | Run a small Drive-backed gate, then pull state, predict, push state, and attempt site publication for non-skip modes. |
| `train.yml` | Monday 19:00 UTC; manual | Pull state, train in the container, push only if the job completes. |
| `build-image.yml` | Relevant runtime-file changes on `main`; manual | Build/push `ghcr.io/levonrush/footy-tipper:latest` and SHA tag. |
| `smoke-checks.yml` | Push and pull request | Python compile/lint/tests plus R source parsing; no publication. |

The scheduled prediction gate reads Drive `schedule.json` and chooses:

- `send`: from six hours before first kickoff of the next unsent round until twelve hours after it;
- `refresh`: when the schedule is more than eight days old;
- `skip`: otherwise.

A green gate-only run may mean `skip`; inspect whether the `predict` job ran. The twelve-hour grace window gives hourly retries while `email_sends` prevents duplicates. The gate also attempts to re-enable scheduled predict/train workflows to reduce offseason inactivity risk.

Manual dispatches:

```bash
gh workflow run predict.yml -f mode=test
gh workflow run predict.yml -f mode=refresh
gh workflow run predict.yml -f mode=live
gh workflow run train.yml
gh run watch
```

`test` is the default manual mode and does not write the production send ledger. `refresh` runs `predict --skip-send`. `live` sends to the real list.

## Actions secrets

GitHub Actions materializes:

- `SECRETS_ENV`: the complete contents of local `secrets.env`;
- `SERVICE_ACCOUNT_JSON`: the complete service-account JSON;
- `GITHUB_TOKEN`: GitHub's short-lived automatic token for checkout, packages, workflow enablement, and site pushes.

The workflow masks each environment value before subsequent steps. Because workflow logs are public with the repository, do not use `--dry-run` in Actions: it prints the rendered email. Rotate secrets through repository settings or `gh secret set`; never commit the materialized files.

## Optional integrations and failure behavior

| Condition | Expected behavior |
| --- | --- |
| No pre-game rows | Inference writes an empty batch; send exits cleanly. |
| Missing/sparse lineups | Log and fill safe defaults; strict mode alone fails hard. |
| Missing lineup feature merge | Continue by default; `FOOTY_TIPPER_LINEUP_FEATURES_STRICT=true` makes it fatal. |
| Missing performance data while enabled | Fail fast with a clear provider/configuration error. |
| Missing Claude/Anthropic | Use deterministic email copy when generation cannot run. |
| Missing OpenAI/banner generation | Continue with the normal/static presentation. |
| Missing Google dependencies or credentials | Skip Drive-dependent actions with a clear message where the command permits; cloud production should treat an unexpected skip as an alert. |
| Missing required models | `infer`/`predict` auto-train unless `--skip-auto-train` is set. |

Claude writes copy. OpenAI generates an optional banner. They are independent failure boundaries.

## Weekly operator runbook

1. Pull shared state when working from a fresh machine: `footy-tipper state pull`.
2. Inspect the derived schedule: `footy-tipper state schedule`.
3. Train if scheduled or if model/data contracts changed: `footy-tipper train`.
4. Run a safe full pass: `footy-tipper predict --test --dry-run --skip-drive`.
5. Verify season/round, fixtures, lineup coverage, confidence, value picks, stakes, joker call, and competition advice.
6. Use `footy-tipper predict` for the intended live workflow.
7. Confirm `email_sends`, Drive push, backup, and site publication messages.

The weekly `predict` path uses recent lineup refresh. It enters historical backfill only if missing models trigger auto-training and the requested lineup history is not bootstrapped.

## Recovery runbooks

### A live run fails before email

Fix the provider/configuration issue and rerun. No `email_sends` row or joker transition should have been written.

### Email succeeded but a later step failed

Check `email_sends` before doing anything. Rerun without `--force-resend`; the ledger should prevent a duplicate email while allowing diagnosis of remaining local/state tasks. Use explicit state/site commands where possible.

### Drive has older good models after training timeout

This is intentional: `train.yml` pushes only after training completes. Pull Drive state to restore the last successful artifact set.

### State is missing on a new installation

Create the local DB/models through prep/train, then seed once with `footy-tipper state push`. Do not push an empty state directory over a known-good remote without verifying contents.

### Lineup coverage collapses

Run recent ingestion with strict mode to expose the failure, inspect parse statuses, then backfill only if historical repair is needed. Do not make the entire prediction path strict as a permanent workaround.

## Backups and Pages

After a successful production send, a consistent gzipped DB snapshot can be uploaded under Drive `backups/`; the newest eight are retained. Set `FOOTY_TIPPER_DB_BACKUP=false` to disable it.

`footy-tipper site` builds locally. `site --publish` commits and pushes generated `docs/site/`, so it changes external Git state. Configure Pages to deploy `main` `/docs`, and set `FOOTY_TIPPER_SITE_URL` for email links.

The expected endpoint is [the Footy Tipper GitHub Pages site](https://levonrush.github.io/footy-tipper/site/). As of the documentation review on 2026-07-13 it returned 404; describe it as expected/unpublished until a live fetch succeeds.

## Diagnostic checklist

- Gate decision and reason in Actions logs
- latest season/round chosen by `prediction_table.sql`
- row counts in `inference_data` and `predictions_table`
- lineup run status, zero-entry rate, and eligible as-of coverage
- head-to-head and line-market coverage
- required/optional artifact files and manifest predictor schema
- `email_sends`, `joker_usage`, and `comp_strategy_decisions`
- Drive state timestamps and backup presence
- Pages configuration and `docs/site/` commit

## Security

Never commit `secrets.env`, `service-account-token.json`, passwords, API keys, or rendered logs containing them. Preserve [`.gitignore`](../.gitignore), review public Actions output, and use the least credentials required for each workflow.
