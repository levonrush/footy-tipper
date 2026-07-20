# Footy Tipper documentation

Pick the door that matches the job. The repository Markdown is canonical; generated sites, Medium articles, research exports, and the Notion hub explain or present it but do not override it.

## I need to run it

1. [Getting started](getting-started.md) — prerequisites, Conda setup, workflow-specific secrets, and the first safe run.
2. [CLI reference](cli-reference.md) — all eleven commands, flags, defaults, and composition rules.
3. [Operations and reliability](operations-reliability.md) — local model publication, Actions prediction, Drive state, schedules, idempotency, backups, Pages, and incident runbooks.

## I need to understand it

1. [Architecture](how-it-works.md) — current production path, ownership boundaries, SQLite contracts, artifacts, and feed rollback.
2. [Models and evaluation](modeling-techniques.md) — Tier A/B/C, market separation, stacking, calibration, margin blending, simulation, and evidence.
3. [Lineup integration](lineup-integration.md) — versioned snapshots, as-of selection, feature families, uncertainty, and repair behavior.
4. [Principled odds integration](principled-odds-integration.md) — why odds are a separate signal and how the current stack avoids double-counting them.

## I need to operate the competition layer

- [Joker strategy](joker-strategy.md) — recommendation inputs, state transition, and replay-safe behavior.
- [Competition strategy](comp-strategy.md) — tipping, value, Kelly-derived stakes, and competition-win objectives.

## I need to change or research it

- [Research and history](research-and-history.md) — curated findings, primary references, implementation matrix, and the full Medium series.
- [Data-source migration](data-source-migration.md) — shipped nrl.com/odds production path, parity evidence, XML rollback, and remaining extensions.
- [Research notebook index](../research/README.md) — historical R Markdown exploration and stale-path warnings.
- [Literature-review index](../lit-review/README.md) — report theses and production influence.
- [Changelog](../CHANGELOG.md) — genuine history, including retired interfaces.
- [Agent guide](../AGENTS.md) — automation contracts and repo safety rules.

## Diagrams

Editable Mermaid sources and checked-in SVG previews live in [the diagram catalogue](diagrams/README.md). Shapes and labels carry meaning; colour is supplementary.
