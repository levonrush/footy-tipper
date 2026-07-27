suppressPackageStartupMessages({
  library(dplyr)
  library(DBI)
  library(RSQLite)
  library(tibble)
  library(tidyr)
  library(stringr)
})

source("pipeline/common/data-prep/feature-engineering.R")
source("pipeline/common/data-prep/get-data.R")

assert_true <- function(condition, message) {
  if (!isTRUE(condition)) {
    stop(message, call. = FALSE)
  }
}

odds_rows <- tibble(
  team_head_to_head_odds_home = c(0, 1.91, 1, 1.91),
  team_head_to_head_odds_away = c(0, 1.91, 2.5, 1.91),
  team_line_odds_home = c(0, 1.91, Inf, 1.91),
  team_line_odds_away = c(0, 1.91, 1.91, 1.91),
  team_line_amount_home = c(0, 0, -4.5, 0),
  team_line_amount_away = c(0, 0, 4.5, 0),
  total_line = c(0, 42.5, 42.5, 0),
  total_over_odds = c(0, 1.91, -2, 1.91),
  total_under_odds = c(0, 1.91, 1.91, 1.91)
) %>%
  market_features() %>%
  missingness_flags()

assert_true(
  odds_rows$line_odds_missing[1] == 1L &&
    is.na(odds_rows$implied_spread_home[1]) &&
    is.na(odds_rows$implied_spread_away[1]),
  "Zero placeholder line prices must invalidate the implied spread."
)
assert_true(
  odds_rows$line_odds_missing[2] == 0L &&
    odds_rows$implied_spread_home[2] == 0 &&
    odds_rows$implied_spread_away[2] == 0 &&
    odds_rows$home_line_cover_prob_basic[2] == 0.5,
  "A two-sided 1.91 pick'em market must retain its genuine zero handicap."
)
assert_true(
  odds_rows$odds_missing[3] == 1L &&
    is.na(odds_rows$home_market_prob_basic[3]) &&
    odds_rows$line_odds_missing[3] == 1L &&
    odds_rows$totals_missing[3] == 1L,
  "Non-finite, negative, and <=1 decimal prices must be treated as missing."
)
assert_true(
  odds_rows$totals_missing[4] == 1L &&
    is.na(odds_rows$market_total_line[4]),
  "A zero or negative total line must stay missing even with valid prices."
)
market_values <- odds_rows %>%
  select(
    contains("market"),
    contains("overround"),
    contains("prob"),
    starts_with("implied_spread")
  ) %>%
  unlist(use.names = FALSE)
assert_true(
  !any(is.infinite(market_values)),
  "Market feature engineering must never emit infinite values."
)
assert_true(
  all(is.na(compute_fair_probs_basic(0, 2))) &&
    all(is.na(compute_fair_probs_power(Inf, 2))) &&
    all(is.na(compute_fair_probs_shin(-1, 2))) &&
    is.na(safe_logit(Inf)),
  "All fair-probability paths must reject invalid decimal prices safely."
)

shin_example <- compute_fair_probs_shin(2.50, 1.53)
shin_mirror <- compute_fair_probs_shin(1.53, 2.50)
assert_true(
  abs(shin_example[1] - 0.373202614379085) < 1e-12 &&
    abs(sum(shin_example[1:2]) - 1) < 1e-12 &&
    abs(shin_example[1] - shin_mirror[2]) < 1e-12 &&
    abs(shin_example[2] - shin_mirror[1]) < 1e-12,
  "Binary Shin probabilities must use the correct de-vig equation and preserve mirror symmetry."
)
assert_true(
  all(abs(compute_fair_probs_shin(1.91, 1.91)[1:2] - 0.5) < 1e-12),
  "A symmetric two-outcome market must remain neutral under Shin de-vigging."
)

movement_db <- tempfile(fileext = ".sqlite")
movement_con <- dbConnect(SQLite(), movement_db)
dbWriteTable(
  movement_con,
  "odds_history",
  tibble(
    id = 1:4,
    game_id = c(101, 101, 102, 102),
    snapshot_kind = c("open", "live", "open", "live"),
    h2h_odds_home = c(2.0, 1.5, 1.5, 3.0),
    h2h_odds_away = c(2.0, 3.0, 3.0, 1.5),
    line_amount_home = c(0, -2.5, -4.5, -1.5)
  )
)
dbDisconnect(movement_con)
movement_rows <- market_movement_features(
  tibble(game_id = c(101, 102)),
  movement_db
)
unlink(movement_db)
assert_true(
  movement_rows$h2h_move_logit[1] > 0 &&
    movement_rows$h2h_move_logit[2] < 0 &&
    movement_rows$h2h_move_logit[1] != movement_rows$h2h_move_logit[2],
  "Market movement probabilities must be calculated independently for each game."
)

fixtures <- tibble(
  game_id = c(1, 2, 3, 4, 5),
  competition_year = 2026L,
  round_id = c(1L, 2L, 3L, 3L, 3L),
  game_state_name = c("Final", "Final", "Pre Game", "Final", "Pre Game"),
  start_time = c(100, 200, 300, 300, 300),
  start_time_utc = c(100, 200, 300, 300, 300),
  team_home = c("A", "A", "A", "D", "D"),
  team_away = c("B", "C", "B", "E", "F")
)
performance <- tibble(
  team = c("A", "B", "A", "B", "C", "D", "E"),
  round_id = c(1L, 1L, 2L, 2L, 2L, 3L, 3L),
  competition_year = 2026L,
  points = c(10, 11, 20, 0, 21, 30, 31),
  set_completion_rate = c(70, 71, 80, 0, 81, 90, 91)
)

merged <- merge_latest_prior_performance(fixtures, performance)

assert_true(
  is.na(merged$points_home_performance[1]) &&
    is.na(merged$points_away_performance[1]),
  "Season-opening fixtures must remain missing when no prior final exists."
)
assert_true(
  merged$points_home_performance[2] == 10 &&
    is.na(merged$points_away_performance[2]),
  "A final fixture may use only earlier finalized performance, never its own row."
)
assert_true(
  merged$points_home_performance[3] == 20 &&
    merged$points_away_performance[3] == 11,
  "Upcoming fixtures must use each team's latest prior final, skipping bye rows."
)
assert_true(
  is.na(merged$points_home_performance[5]),
  "A performance observation at the same kickoff is not strictly prior."
)

message("data-prep integrity tests passed")
