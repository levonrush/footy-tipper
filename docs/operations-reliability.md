# Operations, Reliability, and Safety

This doc covers operational contracts and "what happens if I run it twice" behavior.

## 1) Prediction Storage Contract

`predictions_table` is keyed by `game_id` and upserted.

What that means:
- rerunning inference updates existing game rows
- it does not duplicate rows for the same game
- latest successful run is what send uses

Relevant SQL:
- `pipeline/common/sql/create_table.sql`
- `pipeline/common/sql/insert_into_table.sql`

## 2) Which Predictions Get Sent

Send query is intentionally scoped to:
- latest `competition_year` with `Pre Game` rows
- minimum `round_id` in that year

So historical/stale rounds are excluded from send output.

Relevant SQL:
- `pipeline/common/sql/prediction_table.sql`

## 3) Offseason and Empty Data Behavior

If there are no pre-game fixtures:
- send exits cleanly
- no crash path

This protects automation jobs outside active rounds.

## 4) Optional Integrations Degrade Safely

If optional providers are missing:
- Google deps missing -> upload/send steps skip with clear message
- OpenAI missing -> deterministic fallback email copy

Core tipping pipeline continues where possible.

## 5) Joker Reliability Controls

Joker recommendation includes guardrails so it does not overreact when odds coverage is thin.

Defaults:
- minimum rounds with odds before `PLAY`
- minimum coverage per round
- minimum lead over next-best round

Single-use reliability:
- joker usage is persisted in `joker_usage` keyed by `competition_year`
- once a season is marked used, recommendations are forced to hold
- test sends read this state but do not write it
- production sends write only after successful production email send

## 6) Recommended Weekly Runbook

1. `footy-tipper infer`
2. `footy-tipper send --test --dry-run`
3. sanity-check round scope + joker call + value picks
4. run production send

## 7) Debugging Checklist

When output looks wrong, check:
- round/year selected by `prediction_table.sql`
- row counts in `inference_data` and `predictions_table`
- odds coverage by round
- `joker_usage` row for the active `competition_year`
- which joker strategy source was used (`explicit_env`, `policy_auto`, fallback)

## 8) Security / Hygiene

Never commit:
- `secrets.env`
- `service-account-token.json`
- credentials/API keys

Keep `.gitignore` protections intact.
