# Research notebooks

This directory is the project's workshop floor: useful experiments, old assumptions, and the occasional power tool left where the production pipeline no longer keeps it.

The notebooks are **historical exploration**, not supported entrypoints. Current preparation, training, inference, and evaluation live under [`pipeline/`](../pipeline/) and are operated through [`footy-tipper`](../footy-tipper). Results here should be reproduced against current data contracts before they inform a production change.

## R Markdown analyses

| Notebook | Original question | Current status |
| --- | --- | --- |
| [`byes.Rmd`](analysis/byes.Rmd) | Do byes alter subsequent match performance? | Exploratory; helper path points to removed `/R` scripts. |
| [`crowd-effects.Rmd`](analysis/crowd-effects.Rmd) | Is crowd size associated with home advantage or results? | Exploratory; old helper/data path and observational confounding remain. |
| [`elo.Rmd`](analysis/elo.Rmd) | How should Elo carryover and update strength be tuned? | Historically influential; production Tier A is now a separate sequential baseline implementation. |
| [`home-ground-advantage.Rmd`](analysis/home-ground-advantage.Rmd) | How does home advantage vary by team/venue context? | Exploratory; old `/R` helper discovery. |
| [`homeground-advantage.Rmd`](analysis/homeground-advantage.Rmd) | Earlier home-ground analysis variant. | Superseded duplicate; sources removed `R/get-data.R`. |
| [`last-ditch-effort.Rmd`](analysis/last-ditch-effort.Rmd) | Can late-season urgency or table position explain extra performance? | Historical; points to removed `pipeline/data-prep/functions`. |
| [`lets-make-a-model.Rmd`](analysis/lets-make-a-model.Rmd) | Train and assess an early random-forest outcome model. | Superseded by Tier A/B/C, nested evaluation, and current feature contracts. |
| [`soo.Rmd`](analysis/soo.Rmd) | Does State of Origin scheduling affect club match outcomes? | Exploratory; old `/R` helper path. |

Several notebooks refer to the deleted root `R/` directory or the old `pipeline/data-prep/functions` layout. Those paths are preserved as historical evidence rather than quietly rewritten to make an old analysis appear current.

## Other experiments

- [`phd-methods-transfer.ipynb`](phd-methods-transfer.ipynb) documents which methods from the *From Samples to Sensors* PhD were ported into the production stack and which were rejected, with the evaluation results that followed. Unlike the notebooks above it describes **shipped** code: the distributional metrics, the constraint-native probability/scoreline reconciliation, and the lineup-marginalisation fix are all live. Numbers are read from `reports/eval-latest.json` rather than typed, so the notebook goes stale the moment that report is regenerated.
- [`build_phd_transfer_notebook.py`](build_phd_transfer_notebook.py) generates the notebook above with its outputs already embedded, because the `footy-tipper` env has no Jupyter kernel. Regenerate with `python research/build_phd_transfer_notebook.py`; do not hand-edit the `.ipynb`.
- [`model-training.ipynb`](model-training.ipynb) is an earlier interactive model-training workflow. It is not the artifact-producing production trainer.
- [`model_properties.py`](model_properties.py) is a small model-inspection experiment, not an operator command.

## How to reuse this work safely

1. Restate the hypothesis and the decision it could change.
2. Rebuild inputs from current `footy_tipping_data` with an explicit temporal cutoff.
3. Keep `Final` outcomes out of any pre-game feature.
4. Evaluate by held-out season, not a random row split.
5. Add a production feature only when train and inference can construct it identically.
6. Record shipped/not-shipped status in [Research and history](../docs/research-and-history.md).

The curated research synthesis is [Research and history](../docs/research-and-history.md); formal reports are indexed in the [literature review](../lit-review/README.md).
