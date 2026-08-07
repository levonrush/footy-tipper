# Diagram catalogue

This directory is the canonical home for editable Mermaid architecture sources and their checked-in SVG previews. Documentation embeds the SVG for consistent GitHub/Notion reading and links to the `.mmd` source for maintenance.

## Catalogue

| Diagram | Mermaid | SVG |
| --- | --- | --- |
| Operator CLI hierarchy | [source](operator-cli.mmd) | [preview](operator-cli.svg) |
| Current production architecture | [source](current-production.mmd) | [preview](current-production.svg) |
| Dual-clock delivery watchdog, guarded gate, and alert lifecycle | [source](delivery-watchdog.mmd) | [preview](delivery-watchdog.svg) |
| Model, market, calibration, margin, and simulation | [source](model-stack.mmd) | [preview](model-stack.svg) |
| Versioned lineup ingestion and as-of selection | [source](lineup-as-of.mmd) | [preview](lineup-as-of.svg) |
| Immutable model publication, Actions runtime state, and delivery safety | [source](operations-state.mmd) | [preview](operations-state.svg) |
| Python production feeds and legacy XML rollback | [source](feed-migration.mmd) | [preview](feed-migration.svg) |
| Odds integration before and after | [source](odds-before-after.mmd) | [preview](odds-before-after.svg) |

## Shape semantics

| Shape | Meaning |
| --- | --- |
| Parallelogram | External source or destination outside runtime ownership |
| Rounded rectangle | Process or orchestration step |
| Cylinder | SQLite table or mutable state store |
| Subroutine | Model or versioned artifact |
| Diamond | Decision, gate, or conditional branch |

Dashed arrows are unfinished/optional paths and are labelled. Solid arrows are current required or normal control/data flow. Colour is never the only carrier of status.

## Palette

The strokes use the established brand colours with accessible pastel fills and dark `#0F172A` text:

- navy `#0F172A` — owned state/output boundary
- blue `#0369A1` — orchestration and preparation
- cyan `#0891B2` — external source/destination
- teal `#0F766E` — SQLite and state
- gold `#F59E0B` — models and artifacts
- red `#DC2626` — decisions, warnings, and old/problem paths

## Regeneration

From the repository root, render one diagram with:

```bash
npx --yes @mermaid-js/mermaid-cli \
  -i docs/diagrams/current-production.mmd \
  -o docs/diagrams/current-production.svg \
  -b transparent
```

If Mermaid CLI cannot discover Chrome automatically on macOS, prefix the command with:

```bash
export PUPPETEER_EXECUTABLE_PATH='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
```

Render all sources with:

```bash
for source in docs/diagrams/*.mmd; do
  npx --yes @mermaid-js/mermaid-cli \
    -i "$source" \
    -o "${source%.mmd}.svg" \
    -b transparent
done
```

The checked-in previews are optimized after rendering so they stay small enough for GitHub and Notion:

```bash
npx --yes svgo --multipass docs/diagrams/*.svg
```

Keep labels short enough to read on a narrow page. If a diagram needs a paragraph inside a node, the paragraph belongs in the surrounding document.
