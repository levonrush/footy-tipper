library(tidyverse)
library(lubridate)
library(elo)

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
    mutate(position_diff = team_position_home - team_position_away)
  
  return(data)
  
}

corona_season <- function(data){
  
  data <- data %>%
    mutate(corona_season = ifelse(competition_year == 2020, T, F))
  
  return(data)
  
}


timing_vars <- function(data){
  
  data <- data %>%
    mutate(start_hour = hour(start_time))
  
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

matchup_form

feature_engineering <- function(data, form_period, pipeline){
 
  data <- data %>%
    fixture_result(pipeline = pipeline) %>%
    easy_pickings() %>%
    corona_season() %>%
    timing_vars() %>%
    season_stats() %>%
    form_stats(form_period = form_period)  %>%
    matchup_form(form_period = form_period)

  return(data)

}

elo_variables <- function(data){
  
  data <- data %>% 
    mutate(home_result = case_when(team_final_score_home > team_final_score_away ~ 1,
                                   team_final_score_home < team_final_score_away ~ 0,
                                   team_final_score_home == team_final_score_away ~ 0.5),
           away_result = case_when(team_final_score_home < team_final_score_away ~ 1,
                                   team_final_score_home > team_final_score_away ~ 0,
                                   team_final_score_home == team_final_score_away ~ 0.5),
           margin = abs(team_final_score_home - team_final_score_away))
  
  
  elo_model <- elo.run(formula = home_result ~ team_home + team_away + k(3 + 3*margin),
                       data = data)
  
  elo_results <- elo_model %>% as.data.frame()
  
  draw_rates <- data.frame(win_prob = elo_model$elos[,3],
                           win_loss_draw = elo_model$elos[,4]) %>%
    mutate(prob_bucket = abs(round((win_prob)*20)) / 20) %>%
    group_by(prob_bucket) %>%
    summarise(draw_prob = sum(ifelse(win_loss_draw == 0.5, 1, 0)) / n())
  
  data <- data %>%
    mutate(home_elo = elo_results$elo.A - elo_results$update.A,
           away_elo = elo_results$elo.B - elo_results$update.B,
           home_prob = elo_results$p.A,
           away_prob = 1 - home_prob) %>%
    mutate(prob_bucket = round(20*home_prob)/20) %>%
    left_join(draw_rates, by = "prob_bucket") %>%
    select(-prob_bucket)
  
    data <- data %>% 
      mutate(home_prob = home_prob - home_prob * draw_prob,
             away_prob = away_prob - away_prob * draw_prob)
  
  return(data)
  
}
