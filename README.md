# The Footy Tipper

An NRL prediction engine built from R, Python, SQLite, probability theory, and the stubborn belief that the pub tipping comp deserves production infrastructure.

Footy Tipper prepares match data, versions team lists, trains calibrated score and winner models, simulates coherent scorelines, finds value against the market, records why it thinks what it thinks, and turns the result into a weekly email and static site. It takes the football seriously. It remains open to the possibility that the football does not care.

![Footy Tipper logo](images/footy-tipper-logo.jpg)

## Start here

```bash
conda env create -f environment.yml
conda activate footy-tipper
cp secrets.env.example secrets.env

footy-tipper setup
footy-tipper
```

Running `footy-tipper` on its own opens a guided menu. It shows what is ready, what is next, and the safe actions available. You do not need to remember the technical pipeline commands.

## The everyday commands

```bash
footy-tipper status          # explain whether the system is ready
footy-tipper tips show       # display the currently published tips
footy-tipper tips test       # run and wait for a one-recipient Actions test
footy-tipper tips refresh    # refresh predictions without email
footy-tipper update-model    # train, validate, publish, and activate a model
```

`tips live` exists for an intentional manual production send. It asks you to type the current round exactly, for example `SEND ROUND 21`, before it can dispatch. Scheduled production delivery normally needs no manual command.

The technical data, model, delivery, cloud, and site tools live under `footy-tipper advanced`. The complete hierarchy is in the [CLI reference](docs/cli-reference.md).

## How production works

Python ingests the nrl.com draw and match centres, derives ladder/performance caches, and refreshes historical or live odds before R preparation. Official team-list articles supply versioned lineups. The credentialled XML feed is an explicit rollback, not the normal source.

Local hardware owns model training. `footy-tipper update-model` uses the normal 100-candidate search, keeps the Mac awake, validates the result, publishes an immutable release to Google Drive, asks GitHub Actions to load that exact release in the production image, activates it, and requests a no-email refresh. It prevents two local updates from running together and terminates the trainer cleanly on Ctrl-C. If interrupted, rerun the same command; its journal revalidates evidence before resuming safe completed stages. GitHub Actions remains enabled while this happens; Docker Desktop is not required locally.

GitHub Actions owns prediction and delivery. Targeted off-boundary polls become eligible at 11:00 `Australia/Sydney` on the calendar day of the round's first game, while an independent Google Apps Script watchdog requests the same safe gate if GitHub scheduling is delayed. Both clocks can only ask the Drive-backed gate what to do; the serialized workflow, delivery marker, and SQLite ledger remain authoritative. An automated failure creates or updates one assigned GitHub issue, and the next successful live run closes it. Actions pulls the active model release and mutable runtime database, but it never trains or silently falls back to training.

Every human live-send command also dispatches that same serialized Actions workflow. The live run validates its sender credentials, service-account token, Google Sheet access, and recipient list before claiming the round's pending marker. A partial SMTP refusal leaves that marker pending/uncertain, so the system cannot automatically send the round again.

Google Drive separates immutable model releases from mutable runtime state:

```text
state/
  footy-tipper-db-latest.sqlite.gz
  schedule.json
  model-current.json
  model-releases/<release-id>.tar.gz
  model-releases/<release-id>.json
```

The old `models-latest.tar.gz` remains only for the first migration and rollback compatibility. See [Operations](docs/operations-reliability.md) for recovery details.

## Documentation

The repository Markdown is the technical source of truth. Notion is a curated map back to it.

- [Documentation map](docs/README.md) — choose a path by task or audience.
- [Getting started](docs/getting-started.md) — setup and the safest everyday workflow.
- [CLI reference](docs/cli-reference.md) — the complete operator and advanced command trees.
- [Architecture](docs/how-it-works.md) — data, models, state, and delivery ownership.
- [Models and evidence](docs/modeling-techniques.md) — Tier A/B/C, calibration, simulation, and limitations.
- [Explainability](docs/explainability.md): the exact decision chain behind each tip, TreeSHAP attribution, and the cohort analyses.
- [Operations](docs/operations-reliability.md) — model releases, Actions, delivery safety, reruns, and recovery.
- [Watchdog operations](docs/watchdog-setup.md) — the deployed Google Apps Script fallback, verification, credential replacement, incident handling, and rollback.
- [Research and history](docs/research-and-history.md) — research-to-production status and the complete Medium series.

## Boundaries worth remembering

- Training rows are explicitly `game_state_name == "Final"`; inference rows are `game_state_name == "Pre Game"`.
- Beginner commands never trigger a surprise local training run.
- Missing lineup data fails soft unless strict mode is requested.
- Actions prediction consumes a named active release and fails clearly if it is missing or invalid.
- A pending live-delivery marker is deliberately treated as uncertain and blocks automatic resend until it is reconciled.
- Claude/Anthropic writes optional email copy. OpenAI is optional banner generation, not the copywriter.
- No season end year, secret, or machine-specific path belongs in runtime code.

## The story

The eleven-part [Footy Tipper Medium series](docs/research-and-history.md#the-medium-series) follows the project from a spreadsheet-shaped hunch through leakage, automation, model rebuilding, Reg R-ai-gan, and agent-led reconstruction. The repository records what runs now; the essays record how it learned to run.

If these tips nuke your comp, this README has never seen you before in its life.
