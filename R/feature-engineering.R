library(tidyverse)
library(lubridate)

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



feature_engineering <- function(data){
 
  data <- data %>%
    corona_season() %>%
    timing_vars() %>%
    season_stats() %>%
    form_stats(form_period = 5)

  return(data)

}
