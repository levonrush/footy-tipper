# The Footy Tipper

An NRL prediction engine built from R, Python, SQLite, probability theory, and the stubborn belief that the pub tipping comp deserves production infrastructure.

Footy Tipper prepares match data, versions team lists, trains calibrated score and winner models, simulates coherent scorelines, finds value against the market, and turns the result into a weekly email and static site. It takes the football seriously. It remains open to the possibility that the football does not care.

![Footy Tipper logo](images/footy-tipper-logo.jpg)

## Fastest safe start

```bash
conda env create -f environment.yml
conda activate footy-tipper
cp secrets.env.example secrets.env
# Add only the credentials needed for the workflow you intend to run.

footy-tipper --help
footy-tipper predict --test --dry-run --skip-drive
```

The last command refreshes lineups, prepares inference data, auto-trains if required artifacts are missing, runs inference, and renders a test email without sending it. See [Getting started](docs/getting-started.md) before a live run.

## The operator commands

```bash
footy-tipper train                       # lineup bootstrap + prep + models
footy-tipper predict                     # lineup refresh + infer + send
footy-tipper send --test --dry-run       # inspect delivery without sending
footy-tipper evaluate --skip-prep        # nested season-out evidence
footy-tipper state pull                  # restore published DB/models from Drive
footy-tipper site                        # generate docs/site/
```

There are eleven commands in total: `prep`, `train`, `infer`, `send`, `predict`, `lineups`, `nrl-data`, `odds`, `site`, `evaluate`, and `state`. Their flags and defaults are in the [CLI reference](docs/cli-reference.md).

## Today and next

**Today:** Python ingests the nrl.com draw and match centres, derives ladder/performance caches, and refreshes historical or live odds before R preparation. Official team-list articles supply versioned lineups. SQLite owns prepared and operational state; calibrated models are trained on local hardware and published to Google Drive; GitHub Actions pulls that state and runs prediction/delivery only.

The credentialled XML feed remains available as the explicit `FOOTY_TIPPER_FEED_SOURCE=feed` rollback, not the default. See [Data-source migration](docs/data-source-migration.md) for the shipped cutover, evidence, and remaining feed extensions.

## Publishing a locally trained model

Production training is intentionally local. Pull the last-known-good Drive state, train with the default 100-candidate Bayesian search, validate the new artifacts without allowing auto-training, then push the DB/models/schedule together. Pause `predict.yml` during that sequence so a cloud prediction cannot race the publication.

```bash
gh workflow disable predict.yml
# Wait until both commands list no runs, then disable once more in case an
# older in-flight gate re-enabled the workflow while it drained.
gh run list --workflow predict.yml --status queued
gh run list --workflow predict.yml --status in_progress
gh workflow disable predict.yml
gh api 'repos/{owner}/{repo}/actions/workflows/predict.yml' --jq .state
footy-tipper state pull
FOOTY_TIPPER_TUNE_ITER=100 footy-tipper train
footy-tipper infer --skip-prep --skip-lineups --skip-nrl-data --skip-auto-train
footy-tipper state schedule
footy-tipper state push
gh workflow enable predict.yml
gh workflow run predict.yml -f mode=refresh
```

If training or validation fails, do not push; re-enable `predict.yml` and leave the published Drive models untouched. The complete runbook is in [Operations](docs/operations-reliability.md).

The expected GitHub Pages endpoint is [the Footy Tipper site](https://levonrush.github.io/footy-tipper/site/). It should be treated as unpublished while it returns 404; generate locally with `footy-tipper site` and enable Pages before calling it live.

## Documentation

The repository Markdown is the technical source of truth. Notion is a curated map back to it.

- [Documentation map](docs/README.md) — choose a path by task or audience.
- [Getting started](docs/getting-started.md) — setup, secrets, safe runs, and failures.
- [Architecture](docs/how-it-works.md) — data, models, state, and delivery ownership.
- [Models and evidence](docs/modeling-techniques.md) — Tier A/B/C, calibration, simulation, and limitations.
- [Research and history](docs/research-and-history.md) — research-to-production status and the complete Medium series.
- [Operations](docs/operations-reliability.md) — local training, Actions prediction, Drive publication, reruns, backups, and Pages.
- [Research notebooks](research/README.md) and [literature reviews](lit-review/README.md) — historical exploration and source material.

## Boundaries worth remembering

- Training rows are explicitly `game_state_name == "Final"`; inference rows are `game_state_name == "Pre Game"`.
- Missing lineup data fails soft unless strict mode is requested.
- Missing Google, Claude, or OpenAI integrations degrade to skips or deterministic fallbacks where documented.
- Actions prediction always uses `--skip-auto-train`; missing published models are an operational failure, never an invitation to train on a hosted runner.
- Claude/Anthropic writes optional email copy. OpenAI is optional banner generation, not the copywriter.
- Production sends are idempotent by season and round unless `--force-resend` is used.
- No season end year or local machine path belongs in runtime code.

## The story

The eleven-part [Footy Tipper Medium series](docs/research-and-history.md#the-medium-series) follows the project from a spreadsheet-shaped hunch through leakage, automation, model rebuilding, Reg R-ai-gan, and agent-led reconstruction. The repository records what runs now; the essays record how it learned to run.

If these tips nuke your comp, this README has never seen you before in its life.
