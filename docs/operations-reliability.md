# Operations, reliability, and safety

This is the runbook for model releases, scheduled tips, email idempotency, and recovery. Start with `footy-tipper status`; it translates the remote state into the next safe action.

![Immutable local model publication plus Actions runtime and delivery flow](diagrams/operations-state.svg)

[Editable Mermaid source](diagrams/operations-state.mmd)

## State ownership

| State | Authority | Write rule |
| --- | --- | --- |
| Code and hand-written docs | Git | Human-reviewed commits |
| Local training database | Operator hardware | Back up before model updates; never overwrite it with the Actions runtime DB |
| Active model release | Drive immutable release + `model-current.json` | Create releases once; move the pointer only after validation and a successful GitHub Actions production-image check |
| Runtime SQLite database | GitHub Actions / Drive | Pull at run start, push after stateful refresh/live success |
| `schedule.json` | Runtime scheduler / Drive | Derived from the next actionable unsent round |
| Delivery round marker | Drive | After pre-SMTP configuration/recipient validation, claim pending under serialized Actions concurrency; pending means uncertain until reconciled to sent |
| `docs/site/` | Site generator / GitHub Pages | Generated; publish explicitly or in the production workflow |
| Email delivery | SMTP | One live season/round after marker and DB-ledger checks |

Models and runtime state are deliberately separate. A runner that started with an old model may update predictions and schedule, but its runtime push cannot upload models or move the active pointer. A local training run cannot replace the smaller cloud runtime database.

## Drive layout

```text
state/
  footy-tipper-db-latest.sqlite.gz
  schedule.json
  models-latest.tar.gz                 # legacy migration compatibility only
  model-current.json                   # active release pointer
  model-releases/
    <release-id>.tar.gz                # create-only artifact archive
    <release-id>.json                  # create-only metadata/receipt
  deliveries/
    <season>-round-<round>.json         # pending (uncertain) or sent
```

Release IDs are immutable. Publishing an existing ID with different bytes is an error. Each archive is re-downloaded after upload and verified against its receipt. A missing or malformed `model-current.json` is a hard production failure; Actions must not guess, fall back to the legacy archive, or train.

`models-latest.tar.gz` is retained so the first 1.0 rollout can import the already validated legacy bundle as the initial immutable release without retraining. After the pointer exists, it is not the production selection mechanism.

### One-time 1.0 rollout

The maintainer performing the code rollout imports the currently validated local artifact set once:

```bash
python -m pipeline.ops.model_release import-legacy
footy-tipper status
footy-tipper tips refresh
```

The importer creates a receipt and immutable release, dispatches the GitHub Actions production-image check, and activates the pointer only if that workflow succeeds. Do not rerun it as a normal update mechanism; later production training always uses `footy-tipper update-model`.

## Read status first

```bash
footy-tipper status
```

Status should show, in plain language:

- setup/credential readiness;
- active release ID and validation metadata;
- the next season/round, first kickoff, and 11:00 Sydney target;
- delivery marker and SQLite ledger agreement;
- whether `predict.yml` is enabled and its latest meaningful result;
- the latest runtime-state timestamp;
- any resumable `.footy-tipper/` update journal.

Use `footy-tipper status --offline` only when remote checks are unavailable. Treat remote facts shown as unknown, not healthy. `--json` exists for monitoring and keeps logs separate from machine output.

## Scheduled prediction

[`predict.yml`](../.github/workflows/predict.yml) polls every 15 minutes. The gate derives the first fixture date in `Australia/Sydney` and returns:

- `live`: at the first available gate on or after 11:00 Sydney on that date, through the existing post-kickoff grace period;
- `refresh`: when published runtime state is stale;
- `skip`: before the target, after an expired round, or when there is no actionable round.

The ledger and delivery marker prevent duplicates. An expired unsent round progresses to the next actionable round. `zoneinfo` handles AEST/AEDT and UTC date boundaries. GitHub cron can be delayed; the 15-minute poll and grace window are recovery mechanisms, not an exact-time guarantee.

Actions uses the machine-only `pipeline.ops.actions_runner` interface. Its prediction mode is an exact allowlist of `test`, `refresh`, and `live`; an unknown value fails. There is no wildcard branch that can become live. Every mode refuses auto-training.

The other workflows do not own production training: `build-image.yml` builds the pinned runtime image after relevant runtime-file changes; `model-check.yml` is manually dispatched by `update-model` to load one exact immutable candidate in that image; and `smoke-checks.yml` compiles/tests Python plus parses the R entrypoint on pushes and pull requests.

### Manual operator commands

```bash
footy-tipper tips test
footy-tipper tips refresh
footy-tipper tips live
```

These dispatch the workflow and wait for completion. `tips live` also requires `SEND ROUND N` before dispatch. Prefer these commands over hand-written `gh workflow run` calls because they resolve the run, label it clearly, and preserve the confirmation contract.

All human production-live commands, including both advanced live aliases, route through this same serialized Actions workflow. No human-facing command sends production SMTP directly from the Mac.

### Mode side effects

| Mode | Email | Runtime push | Site publish | Send ledger / joker / delivery marker |
| --- | --- | --- | --- | --- |
| `test` | One test recipient | No | No | No mutation |
| `refresh` | None | Yes, after successful prediction | Yes | No live transition |
| `live` | Production list | Yes, after successful delivery/reconciliation | Yes | Claims/reconciles round and applies eligible live transitions |

A false/failed email result is a non-zero workflow failure. Test success cannot be inferred merely because rendering succeeded.

## Delivery idempotency

SQLite `email_sends` remains the historical ledger. The Drive marker closes the failure window where SMTP succeeds but the runner fails before its DB push becomes durable.

The live sequence is:

1. Resolve the exact season/round and verify it is not already sent in either store.
2. Render the message, validate the sender/password and service-account token, read and validate the Google Sheet, and freeze the deduplicated recipient envelope. A deterministic failure here creates no marker because SMTP has not been attempted.
3. Under the serialized Actions workflow, claim a Drive marker with status `pending` immediately before SMTP.
4. Send through SMTP using the frozen envelope.
5. If every recipient is accepted, mark Drive `sent` immediately; this is the durable no-resend fact.
6. Reconcile `email_sends` and eligible `joker_usage`, then publish the runtime DB. If that DB step fails, the next live gate copies the existing `sent` marker into the ledger without calling SMTP again.
7. If SMTP is ambiguous, refuses any recipient, or the Drive marker cannot be finalized, leave it `pending` (therefore uncertain) and fail the run. A refusal can mean partial delivery, so it is never safe to resend automatically.

`pending` blocks automatic resend because its SMTP outcome may be uncertain. Do not delete the marker or use a force option simply to make the workflow green. First establish whether recipients received the message, then reconcile the marker and DB ledger to the observed truth.

Inference remains upsert-safe by `game_id`. Odds and lineup observations remain versioned/append-only evidence rather than duplicate operational sends.

## Updating the model

Use the one-command runbook:

```bash
conda activate footy-tipper
footy-tipper update-model
```

Do not disable `predict.yml`. The existing release remains live throughout training; the new release does not become visible until final activation.

### Stages and invariants

1. **Preflight:** check tools, credentials, ignored workspace, disk, Git identity/SHA, Drive layout, active release, and update journal.
2. **Training data:** preserve the local-authoritative database and create a dated local backup. Seed from a staged/validated Drive DB only when no valid local training DB exists.
3. **Training:** write into a release staging directory, use 100 candidates by default, parallelize search candidates while each LightGBM fit uses one thread, keep macOS awake with `caffeinate`, and emit a heartbeat.
4. **Receipt:** after all model artifacts exist, write `training-receipt.json` last with release ID, Git SHA, tuning count, training row/year range, model/runtime library versions, artifact sizes, and hashes.
5. **Validation:** load the complete staged release and run a no-auto-train inference check against staged paths.
6. **Upload:** create the archive and metadata under a new immutable release ID. Never overwrite another release.
7. **Verify:** download the candidate to a fresh staging directory and verify receipt, members, sizes, hashes, and model loading.
8. **Production check:** dispatch `model-check.yml` for that release ID and wait while the GitHub Actions production image runs `model-check --release <id>` against the exact candidate.
9. **Activate:** update `model-current.json`, retaining previous release metadata for rollback.
10. **Refresh:** dispatch and wait for a no-email refresh using the new active release.

The journal, logs, and exclusive process lock live under ignored `.footy-tipper/`. A second update process is refused. Ctrl-C terminates and reaps the whole `caffeinate`/trainer process group before the journal records interruption. Completed stages are reused only after their staged receipt, immutable Drive release, hosted check, local install, and active pointer are revalidated as applicable. An incomplete training stage always restarts. If production was deliberately rolled back after activation, the old journal refuses to reactivate that candidate.

### Failure behavior

- Before activation: the prior pointer remains untouched and production continues on the old release.
- After activation but before refresh success: the new model remains active, status reports the failed refresh, and the operator chooses retry or rollback based on the failure.
- Failed or missing GitHub Actions model check: activation is forbidden.
- Missing/malformed active pointer: production fails loudly; repair or roll back explicitly.
- Upload collision: choose a new release ID; never mutate the existing object.

## Rollback

List and verify releases first:

```bash
footy-tipper advanced model list
footy-tipper advanced model verify
footy-tipper advanced model rollback
footy-tipper tips refresh
```

Rollback first runs the previous release through the current hosted production-image check, then moves the pointer; it does not delete the failed/newer release. Confirm the resulting active ID with `footy-tipper status`, then run the no-email refresh. `advanced model activate` performs the same current hosted check. If it repairs a malformed pointer, the bad pointer is archived in Drive as evidence before replacement.

## Recovery runbooks

### Model update was interrupted

Run `footy-tipper status`, then rerun `footy-tipper update-model`. Let the journal resume safe completed stages. Do not manually copy staged files into `models/` or move the Drive pointer.

### Training or validation failed

Fix the named source/configuration problem and rerun. The old release is still active. Do not upload partial output or edit `model-current.json`.

### A live run failed during pre-SMTP preparation

Credential, token, Google Sheet, and recipient validation happens before the marker claim. The failure message should confirm that no marker was created and no email was sent. After verifying the marker is absent and no DB ledger row exists, fix the named configuration issue and let the next eligible run retry.

### SMTP may have succeeded

Treat the message as sent until proven otherwise. The pending marker should block retries, including after a partial-recipient refusal. Check mailbox/provider evidence, then reconcile both Drive and SQLite before considering another live dispatch.

### Runtime state is missing or corrupt

Use `footy-tipper advanced cloud pull-runtime` to stage and validate what is published. Restore the last known-good runtime DB or backup without touching immutable model releases. Re-derive the schedule and run a refresh.

### Active model pointer is broken

Do not rely on `models-latest.tar.gz`. Verify the intended release metadata/archive, then use the explicit advanced activate/rollback path to repair the pointer and refresh.

### Lineup coverage collapses

Run the advanced recent refresh with strict diagnostics, inspect parse statuses, and use backfill only for historical repair:

```bash
footy-tipper advanced data lineups refresh --help
footy-tipper advanced data lineups backfill --help
```

## Optional integrations and failure behavior

| Condition | Expected behavior |
| --- | --- |
| No pre-game rows | Clean no-op; do not invent fixtures or send old tips. |
| Missing/sparse lineups | Fill safe defaults unless explicit strict diagnosis is requested. |
| Missing performance data while enabled | Fail model preparation clearly. |
| Missing Claude/Anthropic | Use deterministic copy when generation cannot run. |
| Missing OpenAI/banner generation | Continue with the normal/static presentation. |
| Missing Drive credentials in production | Fail the stateful workflow; do not claim success with unpersisted state. |
| Missing active model | Fail clearly; hosted training is forbidden. |

## Backups and Pages

Live runtime DB backups may be retained under Drive `backups/` according to configured retention. Model releases have their own immutable history and are not DB backups.

```bash
footy-tipper advanced site build
footy-tipper advanced site publish
```

`publish` changes external Git/Pages state. Configure Pages from `main` `/docs` and set `FOOTY_TIPPER_SITE_URL` for email links.

## Diagnostic checklist

- `footy-tipper status` explanation and active release ID
- latest Actions run mode, gate decision, and reason
- model receipt, archive hashes, and GitHub Actions production-image check
- next season/round selected by `prediction_table.sql`
- delivery marker and `email_sends` agreement
- lineup ingestion status and as-of coverage
- current odds/line coverage
- runtime DB and schedule timestamps
- site publication result

## Security

Never commit `secrets.env`, `service-account-token.json`, passwords, API keys, model-update logs, or rendered email output containing private data. Preserve [`.gitignore`](../.gitignore), review public Actions logs, redact secrets in human/JSON errors, and use the least credentials required for each action.
