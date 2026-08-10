# CLI reference

`footy-tipper` 1.0 is the supported human interface. Its top level is intentionally small; pipeline-shaped commands live under `advanced`, and GitHub Actions uses a separate machine-only runner.

```bash
footy-tipper --help
footy-tipper --version
footy-tipper
```

![Footy Tipper operator command hierarchy](diagrams/operator-cli.svg)

[Editable Mermaid source](diagrams/operator-cli.mmd)

## Top-level command map

| Command | Purpose | Remote side effects |
| --- | --- | --- |
| `footy-tipper` | Open the guided menu in a TTY | Only after choosing and confirming an action |
| `status [--offline\|--json]` | Explain readiness, active release, next gate, delivery state, workflow state, and resumable update | Read-only |
| `setup` | Guide and validate local configuration | Only explicit credential/auth setup steps |
| `tips show` | Display currently published tips from a temporary runtime DB | Read-only |
| `tips test` | Dispatch and wait for the exact Actions test mode | Test email only; no runtime/site/ledger/joker mutation |
| `tips refresh` | Dispatch and wait for exact no-email refresh mode | Updates mutable runtime state/site as the workflow defines |
| `tips live` | Confirm a round, dispatch, and wait for exact live mode | Production email and runtime state |
| `update-model` | Train, validate, publish, activate, and refresh one immutable model release | Creates a release, moves the active pointer after validation, requests refresh |
| `advanced ...` | Expose the technical toolbox | Depends on the selected leaf command |

Global `--debug` restores a full traceback for diagnosis. Normal human output stays concise. Commands that expose `--json` (`status`, `tips show`, `update-model`, and release listing) emit schema-versioned documents on standard output while logs remain on standard error.

## Guided menu

With no arguments and an interactive terminal, `footy-tipper` shows a status summary and numbered actions. It does not perform an action until one is selected. Destructive or live actions retain their own confirmation.

With no interactive terminal, the same invocation prints help and exits without prompting. Scripts should select a command explicitly rather than trying to drive the menu.

## `status`

```bash
footy-tipper status
footy-tipper status --offline
footy-tipper status --json
```

The human view answers six practical questions:

1. Is setup complete enough for the intended action?
2. Which immutable model release is active?
3. Which round is next, and when does its 11:00 Sydney gate open?
4. Is live delivery unsent, pending/uncertain, or sent?
5. Is the GitHub prediction workflow enabled and what happened most recently?
6. Is there a local model-update journal to resume?

Online status reads GitHub and Drive. `--offline` avoids those calls and clearly marks remote facts as unknown; it never presents cached data as current.

## `setup`

```bash
footy-tipper setup
```

This is the first-run and repair assistant. It checks the Conda/runtime tools, configuration files, credential presence, GitHub access, Drive access, and safe ignored paths. It names the next concrete fix instead of dumping a provider traceback. Re-running it is safe.

It only checks existing saved access. It does not open a browser or launch an interactive GitHub/Google authentication flow. On a managed laptop, failed access therefore stops safely and tells the operator to use a permitted computer or ask for help.

## `tips`

### Show

```bash
footy-tipper tips show
footy-tipper tips show --json
```

Downloads the published runtime database into a temporary directory, validates it, selects the current pre-game round, and prints tips. It does not replace the local database, update a ledger, refresh a site, or send email.

### Test

```bash
footy-tipper tips test
```

Dispatches `predict.yml` with exact mode `test`, prints the run link, and waits for completion. Test mode sends only to `FOOTY_TIPPER_TEST_EMAIL` (or its documented fallback). It must not push runtime state, publish the site, mark a round sent, or consume a joker.

### Refresh

```bash
footy-tipper tips refresh
```

Dispatches exact mode `refresh` and waits. It refreshes cloud inputs/predictions without email. It is the final stage of a successful model update and a safe manual recovery when the schedule or current inputs are stale.

### Live

```bash
footy-tipper tips live
```

Displays the selected season/round and requires the exact phrase `SEND ROUND N`. A wrong phrase, EOF, or Ctrl-C cancels without dispatch. Human CLI live mode has no non-interactive confirmation bypass.

This and every advanced human live alias dispatch the same serialized GitHub Actions workflow; none sends production SMTP directly from the Mac. The cloud run validates its sender credentials, token, Google Sheet access, and deduplicated recipient envelope before claiming a Drive-backed round marker. `pending` deliberately means the SMTP outcome may be uncertain and blocks another automatic send. A full success is reconciled into both the marker and SQLite ledger; a partial SMTP refusal leaves the marker pending for human reconciliation.

## `update-model`

```bash
footy-tipper update-model
```

This is the production model golden path. It has no required flags and defaults to 100 Bayesian candidates. It performs preflight, local DB backup/seed, staged training under `caffeinate`, last-written receipt creation, validation, immutable upload, download/hash verification, a GitHub Actions production-image check of the exact release, active-pointer update, and no-email refresh. It does not require local Docker.

The ignored `.footy-tipper/` directory contains the heartbeat log, resumable journal, and an exclusive update lock. A second terminal is refused while an update owns that lock. Ctrl-C terminates and reaps the trainer before the journal is marked interrupted. Re-running the command revalidates durable file, Drive, hosted-check, and active-pointer evidence before reusing a stage. An interrupted or incomplete training stage restarts rather than treating partial artifacts as valid. If committed production-code changes invalidate an unfinished journal, that same invocation abandons the stale candidate, creates a new release for the current commit, and performs fresh preparation and training; it does not reuse the stale candidate or require a second invocation. `training-receipt.json` is written only after the staged release is complete.

The update does not disable Actions and never replaces the remote runtime DB with the local training DB. Prediction continues on the old active release until the pointer is atomically moved to the verified candidate.

## Advanced command tree

The hierarchy is exact:

```text
footy-tipper advanced
├── data
│   ├── prepare {all|training|tips}
│   ├── lineups {refresh|backfill}
│   ├── nrl {refresh|backfill|validate}
│   └── odds {refresh|backfill}
├── model {train|infer|evaluate|verify|list|activate|rollback}
├── local-run {preview|test|live}
├── delivery {preview|test|live}
├── cloud {pull-runtime|push-runtime|schedule|gate}
├── site {build|publish}
└── explain {round|cohort|report}
```

Run `--help` at any branch or leaf for technical flags:

```bash
footy-tipper advanced --help
footy-tipper advanced data lineups backfill --help
footy-tipper advanced model evaluate --help
```

### `advanced data`

| Command | Contract |
| --- | --- |
| `prepare all` | Refresh the full requested source scope and rebuild prepared SQLite tables. |
| `prepare training` | Smart-refresh training history and rebuild training-oriented prepared state. |
| `prepare tips` | Refresh the narrow current inference window. |
| `lineups refresh` | Fetch recent Team Lists/Late Mail snapshots. |
| `lineups backfill` | Repair historical lineup coverage, including old zero-entry snapshots. |
| `nrl refresh` | Refresh current nrl.com draw/match-centre caches. |
| `nrl backfill` | Repair historical nrl.com coverage. |
| `nrl validate` | Produce parity/coverage evidence without changing source state. |
| `odds refresh` | Record available live The Odds API markets, with the configured provider fallback. |
| `odds backfill` | Import historical odds workbook observations. |

### `advanced model`

| Command | Contract |
| --- | --- |
| `train` | Technical local training only. It may stage artifacts but does not activate a production release. |
| `infer` | Load a selected/local artifact set and upsert predictions. Technical auto-training, where supported, must be an explicit flag here only. |
| `evaluate` | Run nested season-out evaluation and write evidence under `reports/`. `--explain` additionally captures out-of-fold feature attribution to `reports/explain-latest.json`. |
| `verify` | Validate artifact completeness, loading, manifest/receipt metadata, sizes, and hashes. |
| `list` | List immutable model releases and identify the active one. |
| `activate` | Recheck a selected release locally and in the hosted production image, then point production at it. A malformed old pointer is archived before an explicitly confirmed repair. |
| `rollback` | Recheck the previous release in the hosted production image, then activate it after explicit confirmation. |

`update-model`, not a hand-built chain of these commands, is the normal publication interface.

### `advanced local-run`

Runs the data/inference/delivery composition with an explicit mode:

- `preview`: render without SMTP or remote writes;
- `test`: send to the configured test recipient only;
- `live`: routes to the serialized GitHub Actions production workflow and requires the same exact-round confirmation.

These commands are for debugging the composition. Everyday `tips test|refresh|live` deliberately uses the production GitHub workflow.

### `advanced delivery`

Uses predictions already present in the selected local database:

- `preview`: render only;
- `test`: one test recipient;
- `live`: routes to the serialized GitHub Actions production workflow with the same round safety contract.

No human-facing command sends production SMTP directly from the Mac. This keeps scheduled and manual sends under one concurrency authority.

### `advanced cloud`

| Command | Contract |
| --- | --- |
| `pull-runtime` | Stage/validate the mutable runtime DB and schedule before replacing local runtime copies. Models are not part of this transfer. |
| `push-runtime` | Publish the mutable runtime DB and derived schedule. It cannot upload or overwrite model releases. |
| `schedule` | Derive and display the next actionable round schedule. |
| `gate` | Print `live`, `refresh`, or `skip` for the current schedule/time. |

### `advanced site`

- `build` writes generated pages under `docs/site/`.
- `publish` intentionally publishes that generated output and therefore changes external Git/Pages state.

### `advanced explain`

Read-only. Nothing here writes to the database or changes a tip.

| Command | Contract |
| --- | --- |
| `round` | Per-game drivers for the published round, read from `prediction_explanations`. `--trace` prints the exact arithmetic that produced the number, so it can be checked by hand. `--by feature` drops from family labels to raw predictors. |
| `cohort` | Attribute the deployed models across all history and print the family, dead-weight and coverage analyses. Fast, and in-sample: it answers what the model uses, not whether that helps. `--write-report` saves `reports/explain-latest.json`. |
| `report` | Render a stored report, whether written by `cohort` or by the honest `evaluate --explain` capture. |

Two of the analyses only mean something out of fold, because in-sample contributions overstate every family's apparent edge:

- `disagreement`: where the model departs from the market, which family supplied the departure, and whether the model or the market was right on those games.
- `confident-wrong`: which families push hardest when a confident tip is wrong, plus the worst individual calls and what drove them.

Both print an in-sample warning banner unless the report came from `advanced model evaluate --explain`.

Explanations are written by inference alongside the tips and stored in `prediction_explanations`, a sibling of `predictions_table` rather than extra columns on it. The email and site read a one-line `why` from that table through a left join in pandas, so a missing or broken explanations table costs a sentence rather than a send. Set `FOOTY_TIPPER_EXPLAIN=false` to skip the write.

## Retired top-level commands

The 1.0 interface is a clean break. Retired top-level names return exit code `2`, print the exact current replacement, and never forward. This prevents an old bare command from becoming a live delivery or surprise training run. The historical mapping is kept in the [changelog](../CHANGELOG.md), not mixed into the current operator reference.

## Actions machine interface

GitHub Actions does not invoke the human CLI. It uses `pipeline.ops.actions_runner` with an exact allowlist:

```text
gate
runtime-pull
predict --mode {test|refresh|live}
runtime-push
site-publish
model-check --release RELEASE_ID
```

Unknown modes fail. There is no wildcard or default-to-live branch. Actions prediction never trains or auto-trains.

## Output and exit contract

Normal failures name the failed operation and the next useful action without exposing a raw traceback or secret value. Sensitive environment values are redacted.

| Code | Meaning |
| --- | --- |
| `0` | Success or intentional no-op |
| `1` | Operational failure |
| `2` | Invalid invocation or configuration |
| `3` | Safety refusal or cancelled confirmation |
| `130` | User interrupt |
