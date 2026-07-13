# Function to add a result column based on the pipeline specification
fixture_result <- function(data, pipeline){
  
  # For a binomial classification, create a binary outcome variable: 'Win' or 'Loss'
  if (pipeline == 'binomial'){
    
    data <- data %>%
      mutate(
        home_team_result = ifelse(
          game_state_name == "Final",
          ifelse(team_final_score_home >= team_final_score_away, "Win", "Loss"),
          NA_character_
        ) %>% factor(levels = c("Win", "Loss"))
      )
    
    
  } 
  # For a multiclass classification, create an outcome variable with three levels: 'Win', 'Loss', or 'Draw'
  else if (pipeline == 'multiclass'){
    
    data <- data %>%
      mutate(
        home_team_result = case_when(
          game_state_name != "Final" ~ NA_character_,
          team_final_score_home > team_final_score_away ~ "Win",
          team_final_score_home < team_final_score_away ~ "Loss",
          TRUE ~ "Draw"
        ) %>% factor(levels = c("Win", "Loss", "Draw"))
      )
    
  } 
  # For Elo ratings, create variables to keep track of wins, losses, and draws for home and away teams and calculate game margin
  else if (pipeline == 'elo'){
    
    data <- data %>%
      mutate(
        home_team_result = case_when(
          game_state_name != "Final" ~ NA_character_,
          team_final_score_home > team_final_score_away ~ "Win",
          team_final_score_home < team_final_score_away ~ "Loss",
          TRUE ~ "Draw"
        ) %>% factor(levels = c("Win", "Loss", "Draw"))
      )
    
    data <- data %>% 
      mutate(
        home_result = case_when(
          game_state_name != "Final" ~ NA_real_,
          team_final_score_home > team_final_score_away ~ 1,  # win for home team
          team_final_score_home < team_final_score_away ~ 0,  # loss for home team
          TRUE ~ 0.5  # draw
        ),
        away_result = case_when(
          game_state_name != "Final" ~ NA_real_,
          team_final_score_home < team_final_score_away ~ 1,  # win for away team
          team_final_score_home > team_final_score_away ~ 0,  # loss for away team
          TRUE ~ 0.5  # draw
        ),
        margin = ifelse(
          game_state_name == "Final",
          abs(team_final_score_home - team_final_score_away),
          NA_real_
        )
      )  # absolute margin of the game
    
  }
  
  return(data)  # Return the modified dataset

}

rolling_sum_before <- function(observed_values, window = NULL) {
  vals <- as.numeric(observed_values)
  out <- numeric(length(vals))
  history <- numeric(0)

  for (i in seq_along(vals)) {
    if (length(history) == 0) {
      out[i] <- 0
    } else if (is.null(window)) {
      out[i] <- sum(history)
    } else {
      out[i] <- sum(tail(history, min(length(history), window)))
    }

    if (!is.na(vals[i])) {
      history <- c(history, vals[i])
    }
  }

  out
}

rolling_mean_before <- function(observed_values, window) {
  vals <- as.numeric(observed_values)
  out <- numeric(length(vals))
  history <- numeric(0)

  for (i in seq_along(vals)) {
    if (length(history) == 0) {
      out[i] <- 0
    } else {
      out[i] <- mean(tail(history, min(length(history), window)))
    }

    if (!is.na(vals[i])) {
      history <- c(history, vals[i])
    }
  }

  out
}

safe_logit <- function(prob, eps = 1e-6) {
  p <- pmin(pmax(as.numeric(prob), eps), 1 - eps)
  log(p / (1 - p))
}

compute_fair_probs_basic <- function(home_odds, away_odds) {
  q_home <- suppressWarnings(1 / as.numeric(home_odds))
  q_away <- suppressWarnings(1 / as.numeric(away_odds))

  if (is.na(q_home) || is.na(q_away) || q_home <= 0 || q_away <= 0) {
    return(c(NA_real_, NA_real_, NA_real_))
  }

  overround <- q_home + q_away
  if (overround <= 0) {
    return(c(NA_real_, NA_real_, NA_real_))
  }

  c(q_home / overround, q_away / overround, overround)
}

compute_fair_probs_power <- function(home_odds, away_odds) {
  q_home <- suppressWarnings(1 / as.numeric(home_odds))
  q_away <- suppressWarnings(1 / as.numeric(away_odds))

  if (is.na(q_home) || is.na(q_away) || q_home <= 0 || q_away <= 0) {
    return(c(NA_real_, NA_real_, NA_real_))
  }

  overround <- q_home + q_away
  if (overround <= 0) {
    return(c(NA_real_, NA_real_, NA_real_))
  }

  f <- function(k) {
    (q_home ^ k) + (q_away ^ k) - 1
  }

  power_k <- 1
  f_lower <- f(0.01)
  f_upper <- f(10)
  if (!is.na(f_lower) && !is.na(f_upper) && f_lower * f_upper <= 0) {
    power_k <- tryCatch(
      uniroot(f, lower = 0.01, upper = 10)$root,
      error = function(e) 1
    )
  }

  p_home_raw <- q_home ^ power_k
  p_away_raw <- q_away ^ power_k
  normalizer <- p_home_raw + p_away_raw
  if (normalizer <= 0) {
    return(c(NA_real_, NA_real_, overround))
  }

  c(p_home_raw / normalizer, p_away_raw / normalizer, overround)
}

compute_fair_probs_shin <- function(home_odds, away_odds) {
  q_home <- suppressWarnings(1 / as.numeric(home_odds))
  q_away <- suppressWarnings(1 / as.numeric(away_odds))

  if (is.na(q_home) || is.na(q_away) || q_home <= 0 || q_away <= 0) {
    return(c(NA_real_, NA_real_, NA_real_))
  }

  overround <- q_home + q_away
  if (overround <= 0) {
    return(c(NA_real_, NA_real_, NA_real_))
  }

  # Shin (1993): estimate insider-trading parameter z via closed-form discriminant.
  # z represents the fraction of bets from informed traders.
  disc <- overround^2 - 4 * (overround - 1) * (q_home^2 + q_away^2) / overround
  if (is.na(disc) || disc < 0) {
    # Fallback to basic normalization
    return(c(q_home / overround, q_away / overround, overround))
  }

  z <- tryCatch(
    (overround - sqrt(disc)) / (2 * (overround - 1)),
    error = function(e) 0
  )
  z <- max(0, min(z, 0.5))

  p_home <- (sqrt(z^2 + 4 * (1 - z) * (q_home / overround)^2) - z) / (2 * (1 - z))
  p_home <- max(0, min(1, p_home))
  p_away <- 1 - p_home

  c(p_home, p_away, overround)
}

col_or_na <- function(data, col_name) {
  if (col_name %in% names(data)) {
    suppressWarnings(as.numeric(data[[col_name]]))
  } else {
    rep(NA_real_, nrow(data))
  }
}

last_observed_before <- function(observed_values, default = NA_real_) {
  vals <- as.numeric(observed_values)
  out <- rep(default, length(vals))
  last_val <- NA_real_

  for (i in seq_along(vals)) {
    out[i] <- ifelse(is.na(last_val), default, last_val)
    if (!is.na(vals[i])) {
      last_val <- vals[i]
    }
  }

  out
}

build_team_rows <- function(data) {
  ordered <- data %>%
    arrange(competition_year, round_id, start_time, game_number, game_id)

  home_rows <- ordered %>%
    transmute(
      game_id,
      competition_year,
      round_id,
      game_number,
      start_time,
      game_state_name,
      side = "home",
      team = team_home,
      is_home_team = TRUE,
      points_for = team_final_score_home,
      points_against = team_final_score_away
    )

  away_rows <- ordered %>%
    transmute(
      game_id,
      competition_year,
      round_id,
      game_number,
      start_time,
      game_state_name,
      side = "away",
      team = team_away,
      is_home_team = FALSE,
      points_for = team_final_score_away,
      points_against = team_final_score_home
    )

  bind_rows(home_rows, away_rows) %>%
    arrange(team, competition_year, round_id, start_time, game_number, game_id, side)
}

# Function to get previous match result
get_prev_match_result <- function(data) {
  data %>%
    arrange(team, competition_year, round_id, start_time, game_number, game_id) %>%
    group_by(team, competition_year) %>%
    mutate(prev_result_diff = last_observed_before(team_final_score_diff, default = NA_real_)) %>%
    ungroup()
}

# Function to apply on the dataset
get_previous_results <- function(data){
  data <- data %>%
    select(-any_of(c(
      "team_final_score_diff",
      "home_prev_result_diff",
      "away_prev_result_diff",
      "prev_result_diff"
    )))
  
  team_rows <- build_team_rows(data) %>%
    mutate(
      team_final_score_diff = ifelse(
        game_state_name == "Final",
        points_for - points_against,
        NA_real_
      )
    ) %>%
    get_prev_match_result() %>%
    select(game_id, side, prev_result_diff)

  home_prev <- team_rows %>%
    filter(side == "home") %>%
    select(game_id, home_prev_result_diff = prev_result_diff)

  away_prev <- team_rows %>%
    filter(side == "away") %>%
    select(game_id, away_prev_result_diff = prev_result_diff)

  data %>%
    mutate(team_final_score_diff = team_final_score_home - team_final_score_away) %>%
    left_join(home_prev, by = "game_id") %>%
    left_join(away_prev, by = "game_id") %>%
    mutate(prev_result_diff = home_prev_result_diff - away_prev_result_diff) %>%
    replace_na(
      list(
        prev_result_diff = 0,
        home_prev_result_diff = 0,
        away_prev_result_diff = 0
      )
    )
}

# Function to calculate turnaround times between games for each team
turn_around <- function(data){
  data <- data %>%
    select(-any_of(c("turn_around_home", "turn_around_away", "turn_around_diff")))

  team_rows <- build_team_rows(data) %>%
    arrange(team, competition_year, round_id, start_time, game_number, game_id, side) %>%
    group_by(team, competition_year) %>%
    mutate(
      observed_start_time = ifelse(game_state_name == "Final", as.numeric(start_time), NA_real_),
      prev_observed_start_time = last_observed_before(observed_start_time, default = NA_real_),
      turn_around = (as.numeric(start_time) - prev_observed_start_time) / (60 * 60 * 24),
      # Season openers have no prior observed game in-year; use one week baseline.
      turn_around = replace_na(turn_around, 7)
    ) %>%
    ungroup() %>%
    select(game_id, side, turn_around)

  home_turn <- team_rows %>%
    filter(side == "home") %>%
    select(game_id, turn_around_home = turn_around)

  away_turn <- team_rows %>%
    filter(side == "away") %>%
    select(game_id, turn_around_away = turn_around)

  data %>%
    left_join(home_turn, by = "game_id") %>%
    left_join(away_turn, by = "game_id") %>%
    mutate(turn_around_diff = turn_around_home - turn_around_away)
}

# Function to create a feature indicating whether the game is a state of origin game
state_of_origin <- function(data){
  
  # Determine state of origin rounds
  round_data <- data %>%
    group_by(round_id, competition_year) %>%
    summarise(state_of_origin = ifelse(any(str_detect(round_name, "Round") & n() <= 5), 1, 0), .groups = "drop") %>%
    arrange(competition_year, round_id)
  
  # Determine post-origin rounds
  round_data <- round_data %>%
    group_by(competition_year) %>%
    mutate(post_origin = lag(state_of_origin, default = 0)) %>%
    ungroup()
  
  # Join back to game-level data
  data <- data %>%
    left_join(round_data, by = c("round_id", "competition_year"))
  
  return(data)
  
}

# This function adds new columns to the data which could be useful for further analysis or prediction
easy_pickings <- function(data){
  start_time_parsed <- suppressWarnings(as.POSIXct(data$start_time, tz = "UTC", origin = "1970-01-01"))

  data <- data %>%
    # Calculate the difference in positions between home and away teams
    mutate(position_diff = position_home_ladder - position_away_ladder,
           # Flag if the season is the 2020 season, which was affected by the Covid-19 pandemic
           corona_season = ifelse(competition_year == 2020, T, F),
           # Extract the start hour of the game from the start_time column
           start_hour = hour(start_time_parsed),
           # Determine the day of the week the game is played on
           game_day = weekdays(as.Date(start_time_parsed)) %>% as.factor())

  # Return the modified dataset
  return(data)
  
}

market_features <- function(data) {
  data <- data %>%
    select(-any_of(c(
      "home_market_prob_basic", "away_market_prob_basic", "home_market_prob_power", "away_market_prob_power",
      "home_market_prob_shin", "away_market_prob_shin",
      "market_overround_h2h", "home_market_logit_basic", "home_market_logit_power", "home_market_logit_shin",
      "market_entropy_basic", "market_entropy_power", "market_prob_delta_basic", "market_prob_delta_power",
      "home_line_cover_prob_basic", "away_line_cover_prob_basic", "line_overround_basic",
      "home_line_cover_prob_power", "away_line_cover_prob_power", "line_overround_power",
      "home_line_cover_prob_shin", "away_line_cover_prob_shin",
      "line_market_logit_home_basic", "line_market_logit_home_power",
      "implied_spread_home", "implied_spread_away", "implied_spread_diff",
      "market_total_line", "total_over_prob_basic", "total_under_prob_basic",
      "totals_overround", "market_total_logit"
    )))

  h2h_basic <- t(mapply(
    compute_fair_probs_basic,
    data$team_head_to_head_odds_home,
    data$team_head_to_head_odds_away
  ))
  h2h_power <- t(mapply(
    compute_fair_probs_power,
    data$team_head_to_head_odds_home,
    data$team_head_to_head_odds_away
  ))
  h2h_shin <- t(mapply(
    compute_fair_probs_shin,
    data$team_head_to_head_odds_home,
    data$team_head_to_head_odds_away
  ))

  line_basic <- t(mapply(
    compute_fair_probs_basic,
    data$team_line_odds_home,
    data$team_line_odds_away
  ))
  line_power <- t(mapply(
    compute_fair_probs_power,
    data$team_line_odds_home,
    data$team_line_odds_away
  ))
  line_shin <- t(mapply(
    compute_fair_probs_shin,
    data$team_line_odds_home,
    data$team_line_odds_away
  ))

  # Totals (over/under) market: same de-vig math as H2H, over vs under.
  # Columns arrive via the odds ingestion (aussportsbetting/Betfair) and are
  # absent under the legacy feed, so read them tolerantly.
  total_over_raw <- col_or_na(data, "total_over_odds")
  total_under_raw <- col_or_na(data, "total_under_odds")
  totals_basic <- t(mapply(
    compute_fair_probs_basic,
    total_over_raw,
    total_under_raw
  ))

  data <- data %>%
    mutate(
      home_market_prob_basic = as.numeric(h2h_basic[, 1]),
      away_market_prob_basic = as.numeric(h2h_basic[, 2]),
      market_overround_h2h = as.numeric(h2h_basic[, 3]),
      home_market_prob_power = as.numeric(h2h_power[, 1]),
      away_market_prob_power = as.numeric(h2h_power[, 2]),
      home_market_prob_shin = as.numeric(h2h_shin[, 1]),
      away_market_prob_shin = as.numeric(h2h_shin[, 2]),
      home_market_logit_basic = safe_logit(home_market_prob_basic),
      home_market_logit_power = safe_logit(home_market_prob_power),
      home_market_logit_shin = safe_logit(home_market_prob_shin),
      market_entropy_basic = ifelse(
        is.na(home_market_prob_basic),
        NA_real_,
        -(home_market_prob_basic * log(pmax(home_market_prob_basic, 1e-9)) +
            away_market_prob_basic * log(pmax(away_market_prob_basic, 1e-9)))
      ),
      market_entropy_power = ifelse(
        is.na(home_market_prob_power),
        NA_real_,
        -(home_market_prob_power * log(pmax(home_market_prob_power, 1e-9)) +
            away_market_prob_power * log(pmax(away_market_prob_power, 1e-9)))
      ),
      market_prob_delta_basic = home_market_prob_basic - away_market_prob_basic,
      market_prob_delta_power = home_market_prob_power - away_market_prob_power,

      home_line_cover_prob_basic = as.numeric(line_basic[, 1]),
      away_line_cover_prob_basic = as.numeric(line_basic[, 2]),
      line_overround_basic = as.numeric(line_basic[, 3]),
      home_line_cover_prob_power = as.numeric(line_power[, 1]),
      away_line_cover_prob_power = as.numeric(line_power[, 2]),
      line_overround_power = as.numeric(line_power[, 3]),
      home_line_cover_prob_shin = as.numeric(line_shin[, 1]),
      away_line_cover_prob_shin = as.numeric(line_shin[, 2]),
      line_market_logit_home_basic = safe_logit(home_line_cover_prob_basic),
      line_market_logit_home_power = safe_logit(home_line_cover_prob_power),
      implied_spread_home = suppressWarnings(as.numeric(team_line_amount_home)),
      implied_spread_away = suppressWarnings(as.numeric(team_line_amount_away)),
      implied_spread_diff = implied_spread_home - implied_spread_away,

      market_total_line = suppressWarnings(as.numeric(col_or_na(data, "total_line"))),
      total_over_prob_basic = as.numeric(totals_basic[, 1]),
      total_under_prob_basic = as.numeric(totals_basic[, 2]),
      totals_overround = as.numeric(totals_basic[, 3]),
      market_total_logit = safe_logit(total_over_prob_basic)
    )

  return(data)
}

# Odds movement features from the odds_history ledger (earliest vs latest
# observation per game). Fails soft to NA + flag when the table is absent
# (legacy feed mode) or a game has fewer than two observations.
market_movement_features <- function(data, db_path = NULL) {
  data <- data %>%
    select(-any_of(c("h2h_move_logit", "line_move_points", "movement_missing")))

  empty <- function(data) {
    data %>%
      mutate(
        h2h_move_logit = NA_real_,
        line_move_points = NA_real_,
        movement_missing = 1L
      )
  }

  if (is.null(db_path) || !file.exists(db_path)) {
    return(empty(data))
  }

  con <- dbConnect(SQLite(), db_path)
  on.exit(dbDisconnect(con), add = TRUE)
  if (!dbExistsTable(con, "odds_history")) {
    return(empty(data))
  }

  movement <- dbGetQuery(con, "
    WITH ordered AS (
      SELECT game_id,
             h2h_odds_home, h2h_odds_away, line_amount_home,
             ROW_NUMBER() OVER (
               PARTITION BY game_id
               ORDER BY CASE snapshot_kind
                 WHEN 'open' THEN 0 WHEN 'live' THEN 1 ELSE 2 END ASC, id ASC
             ) AS rn_first,
             ROW_NUMBER() OVER (
               PARTITION BY game_id
               ORDER BY CASE snapshot_kind
                 WHEN 'open' THEN 0 WHEN 'live' THEN 1 ELSE 2 END DESC, id DESC
             ) AS rn_last
      FROM odds_history
      WHERE h2h_odds_home IS NOT NULL OR line_amount_home IS NOT NULL
    )
    SELECT f.game_id,
           f.h2h_odds_home AS open_h2h_home, f.h2h_odds_away AS open_h2h_away,
           f.line_amount_home AS open_line_home,
           l.h2h_odds_home AS last_h2h_home, l.h2h_odds_away AS last_h2h_away,
           l.line_amount_home AS last_line_home
    FROM ordered f
    JOIN ordered l ON l.game_id = f.game_id AND l.rn_last = 1
    WHERE f.rn_first = 1
  ")

  if (nrow(movement) == 0) {
    return(empty(data))
  }

  fair_home_prob <- function(home_odds, away_odds) {
    q_home <- suppressWarnings(1 / as.numeric(home_odds))
    q_away <- suppressWarnings(1 / as.numeric(away_odds))
    ifelse(
      is.na(q_home) | is.na(q_away) | (q_home + q_away) <= 0,
      NA_real_,
      q_home / (q_home + q_away)
    )
  }

  movement <- movement %>%
    mutate(
      game_id = suppressWarnings(as.numeric(game_id)),
      h2h_move_logit = safe_logit(fair_home_prob(last_h2h_home, last_h2h_away)) -
        safe_logit(fair_home_prob(open_h2h_home, open_h2h_away)),
      line_move_points = suppressWarnings(as.numeric(last_line_home)) -
        suppressWarnings(as.numeric(open_line_home))
    ) %>%
    select(game_id, h2h_move_logit, line_move_points)

  data %>%
    mutate(game_id_join = suppressWarnings(as.numeric(game_id))) %>%
    left_join(movement, by = c("game_id_join" = "game_id")) %>%
    select(-game_id_join) %>%
    mutate(
      movement_missing = as.integer(is.na(h2h_move_logit) & is.na(line_move_points))
    )
}

missingness_flags <- function(data) {
  data <- data %>%
    select(-any_of(c(
      "odds_missing", "line_odds_missing", "market_features_missing", "totals_missing",
      "performance_home_missing", "performance_away_missing", "performance_features_missing"
    )))

  home_perf_cols <- names(data)[str_detect(names(data), "_home_performance$")]
  away_perf_cols <- names(data)[str_detect(names(data), "_away_performance$")]

  if (length(home_perf_cols) == 0) {
    home_perf_missing <- rep(1L, nrow(data))
  } else {
    home_perf_missing <- as.integer(rowSums(!is.na(data[, home_perf_cols, drop = FALSE])) == 0)
  }

  if (length(away_perf_cols) == 0) {
    away_perf_missing <- rep(1L, nrow(data))
  } else {
    away_perf_missing <- as.integer(rowSums(!is.na(data[, away_perf_cols, drop = FALSE])) == 0)
  }

  data %>%
    mutate(
      odds_missing = as.integer(is.na(team_head_to_head_odds_home) | is.na(team_head_to_head_odds_away)),
      line_odds_missing = as.integer(is.na(team_line_odds_home) | is.na(team_line_odds_away)),
      totals_missing = as.integer(is.na(col_or_na(data, "total_over_odds")) | is.na(col_or_na(data, "total_under_odds"))),
      market_features_missing = as.integer(is.na(home_market_prob_basic) | is.na(home_line_cover_prob_basic)),
      performance_home_missing = home_perf_missing,
      performance_away_missing = away_perf_missing,
      performance_features_missing = as.integer(performance_home_missing == 1L | performance_away_missing == 1L)
    )
}

delta_features <- function(data) {
  data <- data %>%
    select(-any_of(c(
      "ladder_points_delta", "ladder_points_difference_delta", "ladder_rank_delta",
      "ladder_win_rate_delta", "ladder_close_game_rate_delta",
      "form_delta", "points_for_form_delta", "points_against_form_delta", "diff_form_delta",
      "attack_delta", "defence_delta", "rest_delta",
      "points_performance_delta", "set_completion_rate_performance_delta",
      "effective_tackle_percentage_performance_delta", "all_run_metres_performance_delta"
    )))

  data %>%
    mutate(
      ladder_points_delta = competition_points_home_ladder - competition_points_away_ladder,
      ladder_points_difference_delta = points_difference_home_ladder - points_difference_away_ladder,
      ladder_rank_delta = position_home_ladder - position_away_ladder,
      ladder_win_rate_delta = win_rate_home_ladder - win_rate_away_ladder,
      ladder_close_game_rate_delta = close_game_rate_home_ladder - close_game_rate_away_ladder,
      form_delta = season_form_home - season_form_away,
      points_for_form_delta = season_points_for_form_home - season_points_for_form_away,
      points_against_form_delta = season_points_against_form_home - season_points_against_form_away,
      diff_form_delta = season_diff_form_home - season_diff_form_away,
      attack_delta = season_points_for_form_home - season_points_against_form_away,
      defence_delta = season_points_against_form_home - season_points_for_form_away,
      rest_delta = turn_around_diff,
      points_performance_delta = col_or_na(data, "points_home_performance") - col_or_na(data, "points_away_performance"),
      set_completion_rate_performance_delta = col_or_na(data, "set_completion_rate_home_performance") - col_or_na(data, "set_completion_rate_away_performance"),
      effective_tackle_percentage_performance_delta = col_or_na(data, "effective_tackle_percentage_home_performance") - col_or_na(data, "effective_tackle_percentage_away_performance"),
      all_run_metres_performance_delta = col_or_na(data, "all_run_metres_home_performance") - col_or_na(data, "all_run_metres_away_performance")
    )
}

# The 'season_stats' function calculates and adds season statistics to the dataset for each team, both when playing home and away
season_stats <- function(data){
  data <- data %>%
    select(-any_of(c(
      "season_record_home", "season_record_away",
      "season_points_for_home", "season_points_for_away",
      "season_points_against_home", "season_points_against_away",
      "season_points_diff_home", "season_points_diff_away"
    )))

  team_rows <- build_team_rows(data) %>%
    mutate(
      result_observed = case_when(
        game_state_name == "Final" & points_for > points_against ~ 1,
        game_state_name == "Final" & points_for < points_against ~ -1,
        game_state_name == "Final" ~ 0,
        TRUE ~ NA_real_
      ),
      points_for_observed = ifelse(game_state_name == "Final", points_for, NA_real_),
      points_against_observed = ifelse(game_state_name == "Final", points_against, NA_real_),
      points_diff_observed = ifelse(game_state_name == "Final", points_for - points_against, NA_real_)
    ) %>%
    arrange(team, competition_year, round_id, start_time, game_number, game_id, side) %>%
    group_by(team, competition_year) %>%
    mutate(
      season_record = rolling_sum_before(result_observed),
      season_points_for = rolling_sum_before(points_for_observed),
      season_points_against = rolling_sum_before(points_against_observed),
      season_points_diff = rolling_sum_before(points_diff_observed)
    ) %>%
    ungroup() %>%
    select(game_id, side, season_record, season_points_for, season_points_against, season_points_diff)

  home_stats <- team_rows %>%
    filter(side == "home") %>%
    select(
      game_id,
      season_record_home = season_record,
      season_points_for_home = season_points_for,
      season_points_against_home = season_points_against,
      season_points_diff_home = season_points_diff
    )

  away_stats <- team_rows %>%
    filter(side == "away") %>%
    select(
      game_id,
      season_record_away = season_record,
      season_points_for_away = season_points_for,
      season_points_against_away = season_points_against,
      season_points_diff_away = season_points_diff
    )

  data %>%
    left_join(home_stats, by = "game_id") %>%
    left_join(away_stats, by = "game_id")
  
}

# The 'form_stats' function calculates and adds short-term form statistics (over a specified period of games) to the dataset for each team, both when playing at home and away
form_stats <- function(data, form_period){
  data <- data %>%
    select(-any_of(c(
      "season_form_home", "season_form_away",
      "season_points_for_form_home", "season_points_for_form_away",
      "season_points_against_form_home", "season_points_against_form_away",
      "season_diff_form_home", "season_diff_form_away"
    )))

  team_rows <- build_team_rows(data) %>%
    mutate(
      result_observed = case_when(
        game_state_name == "Final" & points_for > points_against ~ 1,
        game_state_name == "Final" & points_for < points_against ~ -1,
        game_state_name == "Final" ~ 0,
        TRUE ~ NA_real_
      ),
      points_for_observed = ifelse(game_state_name == "Final", points_for, NA_real_),
      points_against_observed = ifelse(game_state_name == "Final", points_against, NA_real_),
      points_diff_observed = ifelse(game_state_name == "Final", points_for - points_against, NA_real_)
    ) %>%
    arrange(team, competition_year, round_id, start_time, game_number, game_id, side) %>%
    group_by(team, competition_year) %>%
    mutate(
      season_form = rolling_sum_before(result_observed, window = form_period),
      season_points_for_form = rolling_mean_before(points_for_observed, window = form_period),
      season_points_against_form = rolling_mean_before(points_against_observed, window = form_period),
      season_diff_form = rolling_mean_before(points_diff_observed, window = form_period)
    ) %>%
    ungroup() %>%
    select(game_id, side, season_form, season_points_for_form, season_points_against_form, season_diff_form)

  home_form <- team_rows %>%
    filter(side == "home") %>%
    select(
      game_id,
      season_form_home = season_form,
      season_points_for_form_home = season_points_for_form,
      season_points_against_form_home = season_points_against_form,
      season_diff_form_home = season_diff_form
    )

  away_form <- team_rows %>%
    filter(side == "away") %>%
    select(
      game_id,
      season_form_away = season_form,
      season_points_for_form_away = season_points_for_form,
      season_points_against_form_away = season_points_against_form,
      season_diff_form_away = season_diff_form
    )

  data %>%
    left_join(home_form, by = "game_id") %>%
    left_join(away_form, by = "game_id")
  
}

# The 'matchup_form' function calculates and adds a matchup form statistic to the dataset for each pair of teams (team_home, team_away) over a specified number of their most recent games

matchup_form <- function(data, form_period){
  data <- data %>%
    select(-any_of(c("matchup_form")))

  matchup_state <- data %>%
    arrange(competition_year, round_id, start_time, game_number, game_id) %>%
    group_by(team_home, team_away) %>%
    mutate(
      matchup_observed = case_when(
        game_state_name == "Final" & team_final_score_home > team_final_score_away ~ 1,
        game_state_name == "Final" & team_final_score_home < team_final_score_away ~ -1,
        game_state_name == "Final" ~ 0,
        TRUE ~ NA_real_
      ),
      matchup_form = rolling_sum_before(matchup_observed, window = form_period)
    ) %>%
    ungroup() %>%
    select(game_id, matchup_form)

  data %>%
    left_join(matchup_state, by = "game_id") %>%
    mutate(matchup_form = replace_na(matchup_form, 0))
  
}

# The 'feature_engineering' function is a wrapper function that applies multiple data transformation and feature engineering functions on the input data
feature_engineering <- function(data, form_period, db_path = NULL){

  print("Feature Engineering: Calculating Season Statistics...")

  data <- data %>%
    easy_pickings() %>%
    turn_around() %>%
    season_stats() %>%
    form_stats(form_period = form_period) %>%
    matchup_form(form_period = form_period) %>%
    state_of_origin() %>%
    get_previous_results() %>%
    market_features() %>%
    market_movement_features(db_path = db_path) %>%
    missingness_flags() %>%
    delta_features()

  return(data)

}
