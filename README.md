# The Footy-Tipper: A Machine Learning Approach to Winning the Pub Tipping Comp

Footy Tipper is an open-source NRL prediction engine that aggressively mashes together R, Python, SQLite, probability theory, and a questionable amount of confidence.

The mission is twofold:
- help win tipping comps
- teach what actually went into building the thing, not just dump picks and pretend it was magic

It takes the game seriously, but not itself.

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
footy-tipper train
footy-tipper predict
footy-tipper send --test --dry-run
```

`train` and `predict` are designed as simple defaults:
- `train` runs historical lineup bootstrap when needed, then lineup refresh + prep + model training.
- `predict` runs lineup refresh + inference + send flow (and auto-trains if models are missing).

## Docs

Main README is intentionally lightweight.

The deep stuff lives in `docs/`:
- Quick setup and runtime config: `docs/getting-started.md`
- Full command reference: `docs/cli-reference.md`
- End-to-end architecture: `docs/how-it-works.md`
- Lineup ingestion + lineup-aware model features: `docs/lineup-integration.md`
- Modelling techniques and tradeoffs: `docs/modeling-techniques.md`
- Joker strategy (lit review to production): `docs/joker-strategy.md`
- Reliability, reruns, and ops contracts: `docs/operations-reliability.md`

Start at `docs/README.md`.

## What This Project Tries To Teach

- how to split train/infer data correctly (`Final` vs `Pre Game`)
- how to blend and calibrate models for decisions, not vibes
- how to turn probabilities into actions (value picks, staking, joker timing)
- how to build robust pipelines that fail gracefully when providers don’t play nice

## Example Output

![Prediction Example](/images/example_simulation.png)

## Legacy / History

- Literature and research notes: `lit-review/`
- CLI pointer docs: `cli/README.md`, `CLI.md`
- Medium dev write-up: <https://medium.com/@levonrush/the-footy-tipper-a-machine-learning-approach-to-winning-the-pub-tipping-comp-dc07a7325292>

If these tips nuke your comp, this README has never seen you before in its life.
