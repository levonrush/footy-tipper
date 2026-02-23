# Footy Tipper

Footy Tipper is a machine-learning NRL tipping engine built with R + Python + SQLite.

It has two goals:
- help you win your tipping comp
- teach what went into building a serious, production-ish footy model (not just dump predictions)

![Footy Tipper Logo](/images/footy-tipper-logo.jpg)

## Start Here

```bash
conda env create -f environment.yml
conda activate footy-tipper
cp secrets.env.example secrets.env
# edit secrets.env with your feed credentials
```

Core commands:

```bash
footy-tipper train --start-year 2012
footy-tipper infer
footy-tipper predict
footy-tipper send --test --dry-run
```

## Docs

The detailed docs now live in `docs/`.

- Getting started and runtime config: `docs/getting-started.md`
- Full CLI reference: `docs/cli-reference.md`
- End-to-end architecture and data flow: `docs/how-it-works.md`
- Modelling techniques (stacking, calibration, simulation): `docs/modeling-techniques.md`
- Joker strategy system (lit review to production): `docs/joker-strategy.md`
- Reliability, reruns, and production safety: `docs/operations-reliability.md`

You can start from `docs/README.md`.

## What This Project Tries To Teach

- how to split training vs inference data safely (`Final` vs `Pre Game`)
- how to blend model layers instead of betting on a single model class
- how to calibrate probabilities for decision quality, not just accuracy
- how to turn predictions into decisions (value picks, staking, joker timing)
- how to build resilient automation that degrades safely when providers fail

## Example Output

![Prediction Example](/images/example_simulation.png)

## Legacy / History

- Original CLI quick guide location: `cli/README.md`
- Original CLI reference location: `CLI.md`
- Literature and research notes: `lit-review/`
- Medium development story: <https://medium.com/@levonrush/the-footy-tipper-a-machine-learning-approach-to-winning-the-pub-tipping-comp-dc07a7325292>
