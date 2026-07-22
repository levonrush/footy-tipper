# Getting started

The normal interface is deliberately small. Set it up once, use the guided menu or the five everyday commands, and leave the technical pipeline under `advanced` until you actually need it.

## Install the environment

You need Git, Conda or Miniconda, and network access. A production model update also needs an authenticated GitHub CLI and the repository's `predict.yml` and `model-check.yml` workflows enabled. The production-image check runs in GitHub Actions, so Docker Desktop is not required locally. Run these commands from the repository root:

```bash
conda env create -f environment.yml
conda activate footy-tipper
cp secrets.env.example secrets.env
footy-tipper setup
```

For an existing checkout:

```bash
git pull --rebase
conda env update -f environment.yml --prune
conda activate footy-tipper
hash -r
footy-tipper setup
```

`setup` walks through the local prerequisites and reports anything that still needs attention. `secrets.env` and `service-account-token.json` are ignored by Git. Never commit either file.

On a managed work or university laptop, Footy Tipper uses the GitHub login and Google service-account file that are already saved. It never opens a browser sign-in or starts an interactive authentication flow. If that saved access stops working, the command stops before changing anything; use a permitted computer or ask for help rather than trying to bypass the laptop's controls.

## Secrets by job

Use [`secrets.env.example`](../secrets.env.example) as the field-level reference.

| Job | Required configuration | Optional configuration |
| --- | --- | --- |
| Read status and published tips | GitHub/Drive access configured by `setup` | `--offline` uses local information only |
| Model update | `FOLDER_ID`, `service-account-token.json`, local R/Python environment | Betfair, season, lineup, and tuning controls |
| Live email | `MY_EMAIL`, `EMAIL_PASSWORD` | test recipient and staking controls |
| Generated email prose | `ANTHROPIC_API_KEY` | `CLAUDE_MODEL`; deterministic copy is the fallback |
| Generated banner | `OPENAI_KEY` | `OPENAI_MODEL`; banner failure does not block delivery |
| Legacy XML rollback only | `PASSWORD`, `BASE_URL`, and the relevant `NRL_*_EXTENTION` values | `FOOTY_TIPPER_FEED_SOURCE=feed` selects it |

The production feed is public nrl.com ingestion plus configured odds sources. XML credentials are not needed for the normal path.

## Your home screen

Run:

```bash
footy-tipper
```

In a terminal this opens a guided menu. The first screen summarizes:

- the active model release;
- the next NRL round and its 11:00 Sydney gate;
- whether that round is unsent, pending/uncertain, or sent;
- whether the prediction workflow is enabled and healthy;
- whether a local `update-model` journal can be resumed.

Choose an action by number. The menu confirms any action that can send email. If standard input is not an interactive terminal, the command prints help instead of guessing what you meant.

The same summary is available directly:

```bash
footy-tipper status
footy-tipper status --offline
footy-tipper status --json
```

Use `--offline` when GitHub or Drive is unavailable. JSON output is stable and schema-versioned for scripts; normal logs stay on standard error.

## The safe weekly workflow

Most weeks, Actions does the work automatically. These commands are for checking or deliberately nudging it:

```bash
footy-tipper tips show
footy-tipper tips test
footy-tipper tips refresh
```

- `tips show` downloads the published runtime database to a temporary location and displays the current tips. It does not change local or remote state.
- `tips test` dispatches the exact GitHub Actions test mode and waits for the result. It sends only to the configured test recipient and does not change the production send ledger, joker state, runtime database, or site.
- `tips refresh` dispatches the exact no-email refresh mode and waits. Use it after a model update or when current inputs need another pass.

Scheduled production polls every 15 minutes. On the day of a round's first game, the first available poll at or after 11:00 `Australia/Sydney` becomes eligible. GitHub can delay cron jobs slightly; duplicate-prevention state makes retries safe.

## A manual live send

Normally, let the scheduled run send. If an operator-triggered live run is genuinely required:

```bash
footy-tipper tips live
```

The command shows the selected season and round, then requires an exact typed phrase such as:

```text
SEND ROUND 21
```

Anything else cancels with no dispatch. There is no non-interactive bypass in the human CLI, and every human live alias routes through the same serialized GitHub Actions workflow. The cloud run validates the sender credentials, service-account token, Google Sheet access, and recipient list before claiming a Drive-backed pending marker. `pending` or `uncertain` state blocks another automatic attempt until it is reconciled; a partial SMTP refusal deliberately leaves the marker pending.

## Update the production model

Run one command on the better local hardware:

```bash
conda activate footy-tipper
footy-tipper update-model
```

That command performs the safe sequence for you:

1. Check credentials, disk space, Git state, local tools, and published state.
2. Back up the local training database and seed it from Drive only if no valid local copy exists.
3. Keep the Mac awake and train a staged model with the normal 100 Bayesian candidates.
4. Write `training-receipt.json` after every staged model artifact, including the release ID, Git SHA, tuning count, training range, versions, sizes, and hashes.
5. Validate the complete staged release without allowing another training fallback.
6. Upload an immutable release, download it again, and verify every hash.
7. Dispatch `model-check.yml` and wait while the GitHub Actions production image loads that exact candidate release.
8. Move `model-current.json` to the validated release and request a no-email refresh.

Progress, logs, the resumable journal, and a one-process lock live under ignored `.footy-tipper/`. The command uses `caffeinate` on macOS and prints a heartbeat during long quiet tuning stages. Do not start it in two terminals. Ctrl-C stops and reaps the trainer before recording the interruption. If the laptop closes, power fails, or the process is interrupted, rerun `footy-tipper update-model`; it rechecks the durable evidence before reusing a completed stage, and restarts an incomplete training stage cleanly.

Actions remains enabled throughout. Runtime prediction uses the previous active release until the candidate has passed every check and the pointer changes. The local training database is authoritative for training history and is never pushed over the smaller mutable Actions runtime database.

If the update fails, do not manually move the pointer. The previous release remains active. Read `footy-tipper status`, fix the named problem, and rerun the same command.

## Defaults that matter

- `FOOTY_TIPPER_START_YEAR=2010`
- `FOOTY_TIPPER_END_YEAR=<current year>`
- `FOOTY_TIPPER_INCLUDE_PERFORMANCE=true`
- `FOOTY_TIPPER_FEED_SOURCE=python`
- `FOOTY_TIPPER_LINEUPS_ENABLED=true`
- `FOOTY_TIPPER_LINEUPS_STRICT=false`
- `FOOTY_TIPPER_TUNE_ITER=100`
- `FOOTY_TIPPER_TEST_EMAIL=levon_rush@hotmail.com` when not overridden

Performance data is a required training input when enabled and fails clearly if missing. Lineup ingestion is deliberately softer: it logs and continues unless strict mode is explicitly selected in the advanced toolbox.

## Common problems

| What you see | What to do |
| --- | --- |
| `footy-tipper: command not found` | Activate the Conda environment, update it from `environment.yml`, then run `hash -r`. |
| Status says setup is incomplete | Run `footy-tipper setup` and follow the first failed check. |
| No current tips | The season may be between rounds, the feed may not expose pre-game rows yet, or refresh may be needed. Run `footy-tipper status`, then `footy-tipper tips refresh` if it recommends that action. |
| Test workflow fails | Read the linked Actions run. Fix the reported credential, feed, model, or SMTP problem; the production ledger was not changed. |
| Active model is missing or invalid | Do not add cloud auto-training. Resume/fix `footy-tipper update-model`, or use the advanced rollback command to select the previous valid release. |
| Model update was interrupted | Rerun `footy-tipper update-model`; the journal explains what will resume or restart. |
| Delivery is pending (therefore uncertain) | Do not force another live run. Verify the SMTP outcome and reconcile the delivery marker/DB ledger through the operations runbook. |
| Git says branches diverged | Commit or stash intentional work, then use `git pull --rebase`. This repository is configured to rebase local work over remote documentation/site commits. |

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Success, including an intentional no-op |
| `1` | Operational failure such as network, provider, or workflow failure |
| `2` | Invalid command or configuration |
| `3` | Safety refusal or cancelled confirmation |
| `130` | Interrupted with Ctrl-C |

The normal UI prints a concise explanation instead of a Python traceback. Add the global `--debug` flag when diagnosing code, not for everyday operation.

Continue with the [CLI reference](cli-reference.md) for the exact hierarchy or [Operations](operations-reliability.md) for recovery procedures.
