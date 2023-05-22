library(tidyverse)
library(lubridate)
library(elo)
library(tidyverse)

turn_around <- function(data){
  
  home_games <- data %>%
    rename(team = team_home)
  
  away_games <- data %>%
    rename(team = team_away)
  
  turn_arounds <- bind_rows(home_games, away_games) %>%
    arrange(start_time) %>%
    group_by(team) %>%
    mutate(turn_around = difftime(start_time, lag(start_time), units = 'days') %>% as.numeric()) %>%
    ungroup() %>%
    select(game_id, team, turn_around)
  
  data <- data %>%
    left_join(turn_arounds, by = c("game_id", "team_home" = "team")) %>%
    left_join(turn_arounds, by = c("game_id", "team_away" = "team"), suffix = c("_home", "_away")) %>%
    mutate(turn_around_diff = turn_around_home - turn_around_away)
  
  return(data)
    
}

thing <- train_df %>% turn_around() 
thing %>% select(game_id, team_home, team_away, start_time, turn_around_home, turn_around_away, turn_around_diff) %>% View()

# crowd <- function(data){
#   
#   rf <- randomForest(crowd ~ city, venue_name, data = data %>% filter(!is.na(crowd)))
#   
#   data <- data %>%
#     mutate(case_when(
#       competition_year %in% 2020:2021 ~ ~replace_na(.x, 0),
#       TRUE ~ predict(rf, data)
#       
#     ))
#   
# }

state_of_origin <- function(data){
  
  data <- data %>%
    mutate(state_of_origin = ifelse(str_detect(round_name, "Round") & n() <= 5, 1, 0))
  
  return(data)
  
}

home_ground_advantage <- function(data){
  
  set.seed(69)
  
  hga_data <- data %>%
    mutate(points_diff = team_final_score_home - team_final_score_away,
           game_hour = hour(start_time),
           game_day = weekdays(start_time)) %>% 
    filter(game_state_name == 'Final' & !is.na(team_head_to_head_odds_away)) %>%
    select(all_of(c("round_name", "team_head_to_head_odds_home", "team_head_to_head_odds_away", "venue_name", "team_away", "team_home", "home_elo_prob", "away_elo_prob", "city", "away_elo", "position_diff", "home_elo", "average_losing_margin_home", "matchup_form", "close_game_rate_home", "avg_points_difference_away", "average_winning_margin_away", "avg_points_for_away", "points_difference_away", "average_losing_margin_away")), points_diff)
  
  rf <- randomForest(points_diff ~ ., data = hga_data)
  
  train_data <- data %>%
    filter(game_state_name == 'Final' & !is.na(team_head_to_head_odds_away)) %>%
    mutate(home_ground_advantage = rf$predicted) %>%
    select(game_id, home_ground_advantage)
  
  inference_data <- data %>%
    filter(game_state_name != 'Final') %>%
    mutate(home_ground_advantage = predict(rf, .)) %>%
    select(game_id, home_ground_advantage)
  
  hga_data <- bind_rows(train_data, inference_data)
  
  data <- data %>%
    left_join(hga_data, by = c("game_id"))
           
  return(data)
  
}

fixture_result <- function(data, pipeline){
  
  if (pipeline == 'binomial'){
    
    data <- data %>%
      mutate(home_team_result = ifelse(team_final_score_home > team_final_score_away,
                                       "Win", "Loss") %>% as.factor())
    
    
  } else if (pipeline == 'multiclass'){
    
    data <- data %>%
      mutate(home_team_result = case_when(
        team_final_score_home > team_final_score_away ~ "Win",
        team_final_score_home < team_final_score_away ~ "Loss",
        TRUE                                          ~ "Draw") %>% as.factor())
    
  } else if (pipeline == 'elo'){
    
    data <- data %>%
      mutate(home_team_result = case_when(
        team_final_score_home > team_final_score_away ~ "Win",
        team_final_score_home < team_final_score_away ~ "Loss",
        TRUE                                          ~ "Draw") %>% as.factor())
    
    data <- data %>% 
      mutate(home_result = case_when(team_final_score_home > team_final_score_away ~ 1,
                                     team_final_score_home < team_final_score_away ~ 0,
                                     team_final_score_home == team_final_score_away ~ 0.5),
             away_result = case_when(team_final_score_home < team_final_score_away ~ 1,
                                     team_final_score_home > team_final_score_away ~ 0,
                                     team_final_score_home == team_final_score_away ~ 0.5),
             margin = abs(team_final_score_home - team_final_score_away))
    
  }
  
  return(data)

}

easy_pickings <- function(data){
  
  data <- data %>%
    mutate(position_diff = position_home - position_away)
  
  return(data)
  
}

corona_season <- function(data){
  
  data <- data %>%
    mutate(corona_season = ifelse(competition_year == 2020, T, F))
  
  return(data)
  
}

timing_vars <- function(data){
  
  data <- data %>%
    mutate(start_hour = hour(start_time),
           game_day = weekdays(as.Date(start_time)) %>% as.factor())
  
  return(data)

}

season_stats <- function(data){
  
  home_games <- data %>%
    select(game_id, competition_year, team_home, team_final_score_home, team_final_score_away, home_team_result) %>%
    rename(team = team_home, points_for = team_final_score_home, points_against = team_final_score_away) %>%
    arrange(game_id) 
  
  away_games <- data %>%
    select(game_id, competition_year, team_away, team_final_score_home, team_final_score_away, home_team_result) %>%
    rename(team = team_away, points_for = team_final_score_away, points_against = team_final_score_home) %>%
    arrange(game_id) 
  
  season_record <- bind_rows(
    home_games %>% mutate(is_home_team = T), 
    away_games %>% mutate(is_home_team = F)
  ) %>%
    arrange(game_id) %>%
    group_by(team, competition_year) %>%
    mutate(season_record =
             cumsum(case_when(
               home_team_result == 'Win' & is_home_team == T  ~ 1, 
               home_team_result == 'Loss' & is_home_team == T ~ -1,
               home_team_result == 'Win' & is_home_team == F  ~ -1, 
               home_team_result == 'Loss' & is_home_team == F ~ 1,
               TRUE                       ~ 0)),
           season_points_for = cumsum(points_for),
           season_points_against = cumsum(points_against),
           season_points_diff = points_for - points_against) %>%
    mutate_at(vars(season_record, season_points_for, season_points_against, season_points_diff), lag) %>%
    replace(is.na(.), 0) %>%
    ungroup() %>%
    select(-c(competition_year, home_team_result, points_for, points_against, team)) %>%
    group_by(is_home_team) %>%
    group_split()
  
  data <- data %>%
    left_join(season_record[[1]] %>% select(-is_home_team), by = "game_id") %>%
    left_join(season_record[[2]] %>% select(-is_home_team), by = "game_id", suffix = c("_away", "_home"))
  
  return(data)
  
}

form_stats <- function(data, form_period){
  
  home_games <- data %>%
    select(game_id, competition_year, team_home, team_final_score_home, team_final_score_away, home_team_result) %>%
    rename(team = team_home, points_for = team_final_score_home, points_against = team_final_score_away) %>%
    arrange(game_id) 
  
  away_games <- data %>%
    select(game_id, competition_year, team_away, team_final_score_home, team_final_score_away, home_team_result) %>%
    rename(team = team_away, points_for = team_final_score_away, points_against = team_final_score_home) %>%
    arrange(game_id) 
  
  season_form <- bind_rows(
    home_games %>% mutate(is_home_team = T), 
    away_games %>% mutate(is_home_team = F)
  ) %>%  
    arrange(game_id) %>%
    group_by(team, competition_year) %>%
    mutate(season_form = ifelse(seq(n()) < form_period,
                                cumsum(case_when(
                                  home_team_result == 'Win' & is_home_team == T  ~ 1, 
                                  home_team_result == 'Loss' & is_home_team == T ~ -1,
                                  home_team_result == 'Win' & is_home_team == F  ~ -1, 
                                  home_team_result == 'Loss' & is_home_team == F ~ 1,
                                  TRUE                       ~ 0)),
                                rollsum(case_when(
                                  home_team_result == 'Win' & is_home_team == T  ~ 1, 
                                  home_team_result == 'Loss' & is_home_team == T ~ -1,
                                  home_team_result == 'Win' & is_home_team == F  ~ -1, 
                                  home_team_result == 'Loss' & is_home_team == F ~ 1,
                                  TRUE                       ~ 0), form_period, align = "right", fill = 0)),
           season_points_for_form = ifelse(seq(n()) < form_period,
                                           cumsum(points_for),
                                           rollapply(points_for, FUN = mean, width = form_period, align = "right", fill = 0)),
           season_points_against_form = ifelse(seq(n()) < 5,
                                               cumsum(points_against),
                                               rollapply(points_against, FUN = mean, width = form_period, align = "right", fill = 0)),
           season_diff_form = ifelse(seq(n()) < form_period,
                                     cumsum(points_for - points_against),
                                     rollapply(points_for - points_against, FUN = mean, width = form_period, align = "right", fill = 0))) %>%
    mutate_at(vars(season_form, season_points_for_form, season_points_against_form, season_diff_form), lag) %>% 
    replace(is.na(.), 0) %>%
    ungroup() %>%
    select(-c(competition_year, home_team_result, points_for, points_against, team)) %>%
    group_by(is_home_team) %>%
    group_split()
  
  data <- data %>%
    left_join(season_form[[1]] %>% select(-is_home_team), by = "game_id") %>%
    left_join(season_form[[2]] %>% select(-is_home_team), by = "game_id", suffix = c("_away", "_home"))
  
  
  return(data)
  
}

matchup_form <- function(data, form_period){
  
  data <- data %>%
    group_by(team_home, team_away) %>%
    mutate(
      form = case_when(
        home_team_result == 'Win'  ~ 1,
        home_team_result == 'Loss' ~ -1,
        TRUE                       ~ 0
        ),
      matchup_form = ifelse(seq(n()) < 5,
                            cumsum(form),
                            rollsum(form, 5, align = "right", fill = 0)) %>% lag() %>% replace_na(0)) %>%
    select(-form) %>%
    ungroup()
  
  return(data)
  
}

feature_engineering <- function(data, form_period, pipeline){
 
  data <- data %>%
    easy_pickings() %>%
    turn_around() %>%
    # corona_season() %>%
    timing_vars() %>%
    # season_stats() %>%
    # form_stats(form_period = form_period)  %>%
    matchup_form(form_period = form_period) %>%
    state_of_origin()

  return(data)

}
