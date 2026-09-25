# Finals special edition

The tipping comps stop when the regular season does. The NRL does not, so the four
finals weeks run as special editions: the same recipients, the same models, a
different email.

Two things drive that difference. The joker and the competition-strategy layer both
optimise a season-long comp that has finished, so they are switched off. And a
finals series can answer a question no regular round can: who actually wins the
premiership.

## What changes

| Section | Regular round | Finals |
| --- | --- | --- |
| Subject | `Footy Tipper Predictions for Round N` | Stage-branded, for example `GRAND FINAL: the big dance` |
| Heading | `Finals Week 3 2026` | `Preliminary Finals 2026` |
| News | optional legacy highlight | recent finalist-first reporting woven into the opening and closing |
| Banner | topical scene when legacy news is enabled | suitable football news, plus an occasion brief that escalates to the decider |
| Theme | teal, green and amber | a per-stage accent and a SPECIAL EDITION ribbon, gold for the Grand Final |
| Joker call | joker box | replaced by **Road to the big dance** |
| Comp strategy | `P(win comp)` note in the closing | suppressed |
| Per fixture | tip, reason, confidence, head-to-head price | plus knockout stakes, margin shape, and the season series |
| Value picks | head-to-head only | plus line and totals |

Everything else, including the tips themselves, is the same pipeline. Finals rounds
were always predicted correctly; only the framing around them was missing.

## Round stages

`round_name` from the nrl.com draw carries `Finals Week 1`, `Finals Week 2`,
`Finals Week 3` and `Grand Final` at `round_id` 28 to 31 in a 27-round season.
[`pipeline/common/rounds.py`](../pipeline/common/rounds.py) turns that into a stage
constant and is the single place any part of the system decides whether a round is
finals.

It also recognises the older broadcast vocabulary that lineup article titles use
(`Qualifying Final`, `Elimination Final`, `Semi Final`, `Preliminary Final`), and an
unrecognised name containing `final` classifies as finals rather than regular, which
is the safe direction: under-hyping one email is recoverable, accumulating a ladder
through the finals is not.

Finals round numbers move between seasons (2020 played them at rounds 21 to 24, 2022
at 26 to 29), so numbers are only trusted when a caller supplies the season's last
regular round.

| Variable | Default | Meaning |
| --- | --- | --- |
| `FOOTY_TIPPER_FINALS_MODE` | `auto` | `auto` classifies from the round name; `on` forces the finals treatment so it can be rehearsed out of season; `off` restores the regular email |
| `FOOTY_TIPPER_FINALS_NEWS_ENABLED` | `true` | recent finalist-first news informs finals prose and suitable banner ideas; `false` keeps the occasion-based edition without news |

## News in the finals edition

The finals brief reads Google News RSS metadata from the previous seven days,
deduplicates stories, and prioritises this week's teams, then surviving finalists,
then wider league news. Publisher names, dates and source links accompany the
headlines and short snippets supplied to Reg. No article bodies are scraped or
stored, and the news never enters the model or the reviewed Club Context registry.

Reg weaves supported details into the existing opening and closing, attributing
reporting and preserving uncertainty. The separate top-of-email news highlight is
suppressed in finals. Serious human events remain with reviewed Context Watch;
ordinary injuries can inform prose, but neither can inspire cartoon imagery.
The banner receives a separately filtered football brief and otherwise uses the
finals occasion, never a fallback to the news-bearing opening. Missing feeds or
providers leave the existing fallback edition available.

Regular rounds retain `FOOTY_TIPPER_LEGACY_NEWS_ENABLED=false` by default. The
finals switch does not enable that older highlight or alter regular-round output.

## Discovering the next finals draw

Finals matchups may only acquire named teams after the previous week's games.
The published schedule therefore includes `refresh_after_utc`: 24 hours after
generation while the current season's finals are unfinished, otherwise eight
days. The classification includes completed fixtures, so an empty upcoming draw
between finals weeks still refreshes daily. A completed Grand Final ends this
policy. This does not depend on the presentation-only finals rehearsal switch.

The gate sends a known, due, unsent round first; otherwise it refreshes when the
deadline arrives, even if another known fixture is still in the future. Older
nonempty schedules containing only sent or expired rounds bootstrap a refresh
after 24 hours. Gate logs report schedule age and the next unsent round. Delivery
markers and the Sydney 11am send target are unchanged.

## Road to the big dance

[`premiership.py`](../pipeline/common/use_predictions/premiership.py) simulates the
rest of the bracket 20,000 times and reports each surviving team's probability of
reaching the Grand Final and of winning it.

Two probability sources, and the email says which is which:

- **Fixtures that have been drawn** use the calibrated model probability already in
  `predictions_table`, the same number the tip came from.
- **Matchups that do not exist yet** are priced from Tier-A team ratings, with the
  higher seed hosting and the Grand Final treated as neutral.

Raw Tier-A probabilities are badly overconfident. Their 0.9 to 1.0 bucket wins about
72% of the time, and nearly half of all games land outside `[0.1, 0.9]`, so
simulating on them would report a minor premier as an 85% chance. `compute_tier_a_ratings`
therefore fits a two-parameter Platt scaling from the same leak-safe walk that
produced the ratings. On 2015 onward that moves the Brier score from 0.274 to 0.225
and brings every bucket into line with its realised rate, without changing the
ordering. At a neutral venue only the slope applies: the intercept absorbs residual
home bias, and dropping it is what makes the two sides of a Grand Final sum to one.

The bracket rules are not asserted from memory. `tests/test_premiership.py`
reproduces the actual 2023, 2024 and 2025 week-two and week-three matchups from each
season's week-one results:

- Week 1: `1v4`, `2v3`, `5v8`, `6v7`
- Week 2: the loser of `1v4` hosts the winner of `5v8`; the loser of `2v3` hosts the
  winner of `6v7`
- Week 3: the qualifying-final winners host, and the semi-final winners cross over
- Week 4: the Grand Final, at a neutral venue

If the observed week-one pairings do not match that template, or the ladder is
missing, or a finalist has no rating, the simulation returns `available: False` and
the email drops the section instead of printing a fabricated table.

## Margin shape, line and totals

`predictions_table` stores a tip and a scoreline, not a distribution, so the finals
extras that need one read from `prediction_distributions`. That table is written
during inference from the samples the tip was already made from: no re-simulation, no
change to any random seed, and written after the tips are safely persisted inside a
try/except, exactly like `prediction_explanations`. A diagnostics failure costs a
section, never a send.

It carries margin bands, one-score and blowout probabilities, total-points quantiles,
and cover probabilities against the posted line and total. `get_market_picks` prices
the line and totals markets off those cover probabilities using the same bounded
Kelly staking as the head-to-head picks.

## Knockout stakes

Week one runs two qualifying finals and two elimination finals on the same weekend, so
the stakes differ by fixture and are read from the ladder seeds in the published view.
Later weeks are unambiguous.

## What is deliberately not here

- **The joker is suppressed, not fixed.** Its `min_rounds_with_odds = 2` guardrail
  means it reports HOLD once only one priced round remains, so an unused joker also
  expires worthless in the last regular round. That is a real pre-existing issue and
  a separate piece of work.
- **Ladder rate features are deflated in finals.** `get-data.R` divides counting stats
  by `round_id`, so at `round_id = 31` every `*_rate` and `avg_*` ladder feature is
  about 13% low against numerators frozen at 27 rounds. It is consistent between
  training and serving, so it is a systematic distortion the model has partly
  absorbed rather than a train/serve skew. Correcting it needs a retrain.
- **Finals training data is thin.** 84 finals games in the corpus, and `Finals Week 1`
  only from 2023 because the upstream feed has no week-one rows before then. The
  models do treat finals distinctly (`round_name` is a one-hot predictor and the
  finals values are seen categories), but on very few examples.

## Rehearsing out of season

The whole path can be exercised against a real finals series without touching
production state. Copy the runtime database, rewind `footy_tipping_data` and
`feed_cache_fixtures` so the target finals round is `Pre Game` again, rebuild
`inference_data` from the prepared `training_data` rows for those games, then point at
the copy with `FOOTY_TIPPER_DB_PATH` and run inference and
`footy-tipper advanced delivery preview`.

Both tables have to be rewound: the odds-freshness gate reads `feed_cache_fixtures`,
so rewinding only `footy_tipping_data` leaves it answering for the wrong round and
every price is masked as stale.

`FOOTY_TIPPER_FINALS_MODE=on` forces the finals treatment onto an ordinary round,
which is the quicker check when only the rendering matters.
