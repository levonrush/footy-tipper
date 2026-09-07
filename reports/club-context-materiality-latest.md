# Club Context materiality report

**Status:** `no_clear_lift`

No clear out-of-sample predictive lift from Club Context has been established.

> Club Context is shadow-only. This report cannot activate or alter production predictions.

## Paired out-of-fold comparison

| Cohort | Games | Δ accuracy | Δ log loss | Δ Brier | Tip flips |
|---|---:|---:|---:|---:|---:|
| All Games | 3180 | -0.0025 | +0.0001 | +0.0001 | 88 |
| Event Linked Games | 89 | -0.0337 | -0.0073 | -0.0031 | 5 |
| Market Backed | 3176 | -0.0025 | +0.0001 | +0.0001 | 88 |
| No Market | 4 | +0.0000 | +0.0106 | +0.0041 | 0 |

## Counterfactual without market inputs

| Cohort | Games | Δ accuracy | Δ log loss | Δ Brier | Tip flips |
|---|---:|---:|---:|---:|---:|
| All Games | 3180 | -0.0025 | +0.0001 | +0.0001 | 88 |
| Event Linked Games | 89 | -0.0337 | -0.0072 | -0.0031 | 5 |

## Event-cluster uncertainty

- Clusters: 36
- Paired Δ log-loss 95% interval: -0.0191 to +0.0049
- Bootstrap probability of improvement: 87.8%

## Market absorption

- Event-cohort Δ log loss with operational market routing: -0.0073
- Event-cohort Δ log loss without market inputs: -0.0072
- Read this comparison cautiously: similar deltas do not prove an independent psychosocial effect.

## Score and dispersion

- Event-cohort context-to-baseline residual-variance ratio: 1.001
- Production dispersion adjustment: disabled
- Event-cohort Δ margin MAE: -0.004 points

## Matched event study

- Event matches/clusters: 37/37
- Mean affected-team margin-residual difference: -0.77 points
- Event-cluster 95% interval: -6.38 to +5.14 points
- Permutation placebo p-value: 0.804

## By category

| Cohort | Games | Δ log loss | Δ accuracy | Tip flips |
|---|---:|---:|---:|---:|
| Club Crisis | 2 | -0.0196 | +0.0000 | 0 |
| Leadership Change | 81 | -0.0082 | -0.0494 | 4 |
| Serious Human Event | 3 | -0.0096 | +0.3333 | 1 |
| Tribute Milestone | 3 | +0.0291 | +0.0000 | 0 |

## By matches since event

| Cohort | Games | Δ log loss | Δ accuracy | Tip flips |
|---|---:|---:|---:|---:|
| Event Match | 34 | -0.0002 | -0.0588 | 4 |
| Matches 2 3 | 23 | -0.0122 | -0.0435 | 1 |
| Next Match | 32 | -0.0112 | +0.0000 | 0 |

## Evidence coverage

- Registry available: 3180/3180 games
- Eligible event-linked games: 89
- Event games with official evidence: 81

## Changed tips

All-game paired candidate tip flips: 88. Event-linked flips: 5.
- 2023 R12 · St. George Illawarra Dragons v Sydney Roosters: 54.6% → 46.8% (made the tip wrong)
- 2024 R12 · South Sydney Rabbitohs v Parramatta Eels: 53.2% → 47.5% (made the tip wrong)
- 2026 R12 · North Queensland Cowboys v South Sydney Rabbitohs: 49.1% → 50.5% (corrected the tip)
- 2021 R6 · Newcastle Knights v Cronulla-Sutherland Sharks: 50.3% → 49.2% (made the tip wrong)
- 2020 R9 · Gold Coast Titans v New Zealand Warriors: 50.1% → 49.7% (made the tip wrong)

## Interpretation

Negative log-loss/Brier deltas favour the context candidate. The event-cohort pattern is uncertain, aggregate probability quality is slightly worse, tip flips skew harmful, and neither the matched study nor dispersion test shows material benefit. Club Context therefore remains shadow-only.
