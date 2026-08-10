# Explainability

Every published tip carries a record of why the model produced it. That record has two halves, and they answer different questions.

**The chain** is exact arithmetic. It says *which model decided this tip*: the four expert probabilities, the simplex pool weights that combined them, the temperature that scaled the result, whether the consensus guard overrode it, and how the score means moved from raw model output through the Tier A blend and the market line to the scoreline on the page. Nothing here is estimated, and nothing is re-simulated.

**The attribution** is exact TreeSHAP over that model's features, from LightGBM's native `pred_contrib`. It says *why that model said what it said*, in percentage points of win probability and points of margin, grouped into feature families.

Both are computed during inference and stored in `prediction_explanations`. No new dependency: `shap` would give the same numbers for these models while breaking the pinned-version contract that keeps the pickles loadable.

## Reading a tip

```bash
footy-tipper advanced explain round --trace
footy-tipper advanced explain round --game-id 20261112440 --by feature --top 15
```

The chain panel is the part worth checking by hand:

```
market               p=0.7193  w=0.000  ->+0.0000
tier_a               p=0.9798  w=0.000  ->+0.0000
tier_b               p=0.8949  w=0.000  ->+0.0000
tier_c               p=0.7030  w=1.000  ->+0.8616
pooled logit         +0.8616
/ temperature        0.9277 -> +0.9288
consensus guard      not fired
published            0.7168 conditional
```

`sigmoid(0.9288) = 0.7168`, which is the number in the email. The deployed pool puts weight 1.0 on `tier_c` and 0.0 on everything else, so Tier A, Tier B and the market price contribute nothing to the tip: they touch only the margin path and the guard. The manifest records why (`fallback_reason: learned_pool_rejected_use_strongest_non_market_expert`).

## Which model gets attributed

This is the one rule that has to be right. On a normal row the published logit is `w_tier_c / T` times the classifier's logit, so the classifier's TreeSHAP explains the tip and `feature_multiplier` is that scalar.

But a row the consensus guard reversed, or one that fell back to Tier B, was not decided by the classifier at all. Those rows report `attribution_source = score_models` and their drivers come from the score models priced through `d(p)/d(margin)`, with the why line prefixed "Guard override". And if the classifier ever carries zero weight, the honest answer is that no feature of it moves the tip, which is what `experts_only` says.

## The one-line why

Composed at the family level, because "team-list strength" reads and `lineup_avg_spine_margin_rating_delta` does not. It is a pure function of the explanation: no LLM, no network, and deliberately kept out of the LLM copy prompt so it cannot be paraphrased into something the model never said.

The sentence follows the largest single driver, not the net, so a tip carried by the model's base rate against its own features reads honestly:

> Penrith Panthers tipped despite player recent form (-5 pts) and elo ratings (-2 pts); ladder and season totals (+2 pts) in their favour.

It appears in the email and on the site's tip card. Absent explanations render exactly as the email did before, which is asserted by a byte-identical test.

## Cohort analysis

```bash
footy-tipper advanced explain cohort --write-report
footy-tipper advanced model evaluate --explain      # the honest version
footy-tipper advanced explain report
```

`cohort` attributes the deployed models over all history in about two minutes. It is in-sample, which is fine for the question it answers: what does this model use? Every table it prints carries that caveat.

`evaluate --explain` captures attribution from the fold models inside the existing nested season-out loop, restricted to genuinely out-of-fold rows. It fits no extra model: the only added work is one `pred_contrib` call per fold.

### What the analyses are for

**Families, ranked by skill not volume.** Magnitude is a trap: a loud family pushing the wrong way is worse than a silent one. Three columns matter. `prob pts` is how loudly a family speaks. `lift LL` scores that family's contribution alone against the model's base rate, so a positive value means it moves probability toward the truth. The agreement rate with a Wilson interval says whether it points at the winner more often than chance; an interval covering 0.500 means no demonstrated directional value, however loud.

**Dead weight**, in three tiers because each implies a different action. *Never split* is provable from the boosters and safe to delete outright. *Effectively dead* features are split at least once but move nothing on 99% of games, so they cost tuning budget and variance for nothing. *Rare but strong* features have low coverage and a high peak: they do niche work on a handful of games and are exactly what a volume-based cut would destroy.

**Coverage gaps**, which is where a new dataset would actually buy something. Games are bucketed by margin residual, and total attribution is reported per bucket. A high-error cohort in which no family contributes much is a data gap: the model is not making a bad call, it is making a call with no information. The missingness cross-tab asks a related question: does the model get more out of a family when the data is present than when it is missing? If not, it has learned to ignore it.

**Market disagreement** and **confidently wrong** are only trustworthy out of fold. The first takes the decile of games where model and market diverge most, attributes the departure to a family, and then scores model against market on exactly those games. The second reports which families push hardest when a confident tip (>= 70%, matching the email's green badge) turns out wrong, standardized so loud families do not top the table by construction, plus the worst individual calls and what drove them. Those concrete matches are what generate hypotheses for new data.

## Where things live

| Piece | Path |
| --- | --- |
| TreeSHAP core and the raw-feature group map | `pipeline/common/explain/contributions.py` |
| Feature-family taxonomy | `pipeline/common/explain/families.py` |
| Link-scale to probability/margin points | `pipeline/common/explain/units.py` |
| Exact decision chain | `pipeline/common/explain/trace.py` |
| Per-game assembly and the why line | `pipeline/common/explain/game.py` |
| Cohort analyses and the fold collector | `pipeline/common/explain/cohort.py` |
| Persistence | `pipeline/common/explain/store.py`, `pipeline/common/sql/create_explanations_table.sql` |
| Report artifact | `pipeline/common/explain/report.py`, `reports/explain-latest.json` |
| Console rendering | `pipeline/common/explain/cli_views.py` |

## Boundaries

Explanations are diagnostics and are built so they cannot break anything they describe.

- Inference writes them after the tips are safely persisted, inside a try/except. `FOOTY_TIPPER_EXPLAIN=false` skips the write.
- They live in their own table. `prediction_table.sql` stays the ten-column published contract, pinned by a test, and the why line is left-joined in pandas.
- Diagnostics ride out of the simulation that already ran. Nothing re-simulates, no RNG salt changes, and a test asserts that asking for diagnostics returns byte-identical outcomes.
- Artifacts go to `reports/` and SQLite, never to `models/`, because the release receipt hashes every file in that directory.
- Linearised points do not sum exactly to the prediction, since the link is nonlinear. Surfaces therefore report both `points` (comparable across games) and `share` (exact, sums to one) and rank by magnitude.
