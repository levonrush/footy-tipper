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
footy-tipper state pull                  # restore DB and models from Drive
footy-tipper site                        # generate docs/site/
```

There are nine commands in total: `prep`, `train`, `infer`, `send`, `predict`, `lineups`, `site`, `evaluate`, and `state`. Their flags and defaults are in the [CLI reference](docs/cli-reference.md).

## Today and next

**Today:** the runnable production path uses credentialled XML feeds for fixtures, ladder, and performance data; nrl.com team-list articles for versioned lineups; SQLite for prepared data and operational state; calibrated Python models; GitHub Actions for scheduling; and Google Drive for mutable state.

**Next:** the `feed-migration` branch contains prototype nrl.com draw, match-centre, ladder, performance, and odds-ingestion modules. They are research-backed and useful, but they are **not yet invoked by the CLI or R preparation pipeline**. The production feed remains the current path until that wiring, parity testing, and cutover are complete. See [Data-source migration](docs/data-source-migration.md).

The expected GitHub Pages endpoint is [the Footy Tipper site](https://levonrush.github.io/footy-tipper/site/). It should be treated as unpublished while it returns 404; generate locally with `footy-tipper site` and enable Pages before calling it live.

## Documentation

The repository Markdown is the technical source of truth. Notion is a curated map back to it.

- [Documentation map](docs/README.md) — choose a path by task or audience.
- [Getting started](docs/getting-started.md) — setup, secrets, safe runs, and failures.
- [Architecture](docs/how-it-works.md) — data, models, state, and delivery ownership.
- [Models and evidence](docs/modeling-techniques.md) — Tier A/B/C, calibration, simulation, and limitations.
- [Research and history](docs/research-and-history.md) — research-to-production status and the complete Medium series.
- [Operations](docs/operations-reliability.md) — Actions, Drive state, reruns, backups, and Pages.
- [Research notebooks](research/README.md) and [literature reviews](lit-review/README.md) — historical exploration and source material.

## Boundaries worth remembering

- Training rows are explicitly `game_state_name == "Final"`; inference rows are `game_state_name == "Pre Game"`.
- Missing lineup data fails soft unless strict mode is requested.
- Missing Google, Claude, or OpenAI integrations degrade to skips or deterministic fallbacks where documented.
- Claude/Anthropic writes optional email copy. OpenAI is optional banner generation, not the copywriter.
- Production sends are idempotent by season and round unless `--force-resend` is used.
- No season end year or local machine path belongs in runtime code.

## The story

The eleven-part [Footy Tipper Medium series](docs/research-and-history.md#the-medium-series) follows the project from a spreadsheet-shaped hunch through leakage, automation, model rebuilding, Reg R-ai-gan, and agent-led reconstruction. The repository records what runs now; the essays record how it learned to run.

If these tips nuke your comp, this README has never seen you before in its life.
