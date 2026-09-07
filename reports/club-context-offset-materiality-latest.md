# Club Context offset materiality

> The production baseline is held fixed. A pre-declared offset applies only where an event applies, so every game without one is byte-identical to the shipped prediction. This report cannot activate anything.

## Prevalence

- Paired games: 3180 over 16 seasons
- Event-exposed games: 89 (2.80%)
- Exposed games per season: 5.6

## Affected-side calibration

Baseline residual oriented to the club the event happened to, before anything is fitted.

- Mean residual: -16.78 percentage points
- Event-cluster 95% interval: [-24.91, -8.06]
- Clusters: 36
- Reading: affected side underperforms the baseline probability

## Pre-event placebo

The same clubs, oriented the same way, on their games before the event.

- Pre-event games: 100
- Mean residual: -20.13 percentage points
- Event-cluster 95% interval: [-28.29, -11.72]
- Reading: the same bias is present before the event, so this cohort is selected on underperformance

## Power

- Between-cluster standard error: 4.37 points
- Minimum detectable effect (95%/80%): 12.25 points
- Observed effect: -17.48 points

## Offset specifications

| Specification | Exposed Δ log loss | 95% cluster interval | Exposed Δ accuracy | Flips (correct/wrong) | Unexposed rows changed |
|---|---:|---|---:|---:|---:|
| side (primary) | -0.0564 | [-0.1210, +0.0160] | +0.0000 | 10/10 | 0 |
| exposure | -0.0221 | [-0.0550, +0.0140] | -0.0337 | 5/8 | 0 |
| side_and_exposure | -0.0554 | [-0.1190, +0.0170] | -0.0112 | 9/10 | 0 |

## What it is worth over a season

- Extra correct tips per season: +0.00
- Whole-competition log-loss delta: -0.001579

## High-prevalence comparator: season form shortfall

The same offset method, applied to a state variable defined for every club in every round rather than for 2.8% of fixtures.

- Coverage: 2388 of 3180 matches (75.1%)
- Residual-on-shortfall slope: 0.1982, 95% cluster interval [0.0906, 0.2899] over 260 team-seasons
- Out-of-season Δ log loss: -0.00242
- Out-of-season Δ accuracy: +0.0034 (134 flips)
- Extra correct tips per season: +0.50

## Matched control, by horizon

| Horizon | Matches | Mean margin-residual difference | 95% cluster interval |
|---|---:|---:|---|
| pooled post-event | 140 | -1.15 | [-4.67, +1.68] |
| relative_match_0 | 37 | -0.77 | [-6.38, +5.14] |
| relative_match_1 | 36 | -4.76 | [-10.82, +1.62] |
| relative_match_2 | 34 | -4.02 | [-10.11, +2.08] |
| relative_match_3 | 33 | +5.34 | [-0.45, +11.17] |

## Interpretation

The baseline is mis-calibrated on the event cohort: the affected side lands -16.8 points from its stated probability, 95% cluster interval [-24.9, -8.1]. A single pre-declared offset moves event-cohort log loss by -0.0564 with a 95% cluster interval of [-0.1210, +0.0160], which crosses zero. The same clubs already sat -20.1 points from their stated probability before the event, 95% interval [-28.3, -11.7], so this cohort is selected on underperformance and the post-event gap is not all event. Prevalence remains the binding constraint: at 5.6 exposed games a season this is worth +0.00 tips a season, so the value lies in probability quality rather than the tipping ladder.
