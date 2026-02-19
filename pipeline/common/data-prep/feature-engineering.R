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
feature_engineering <- function(data, form_period){

  print("Feature Engineering: Calculating Season Statistics...")
 
  data <- data %>%
    easy_pickings() %>%
    turn_around() %>%
    season_stats() %>%
    form_stats(form_period = form_period) %>%
    matchup_form(form_period = form_period) %>%
    state_of_origin() %>%
    get_previous_results()

  return(data)

}
