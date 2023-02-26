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
  
  home_record <- data %>%
    select(game_id, competition_year, team_home, team_final_score_home, team_final_score_away, home_team_result) %>%
    rename(team = team_home, points_for = team_final_score_home, points_against = team_final_score_away) %>%
    arrange(game_id) %>%
    group_by(team, competition_year) %>%
    mutate(home_record =
             cumsum(case_when(
               home_team_result == 'Win'  ~ 1, 
               home_team_result == 'Loss' ~ -1,
               TRUE                       ~ 0)),
           points_for = cumsum(points_for),
           points_against = cumsum(points_against),
           points_diff = points_for - points_against) %>%
    mutate_at(vars(home_record, points_for, points_against, points_diff), lag) %>%
    replace(is.na(.), 0) %>%
    ungroup() %>%
    select(-c(competition_year, home_team_result, ))
  
  
  
  away_record <- data %>%
    select(game_id, competition_year, team_away, team_final_score_home, team_final_score_away, home_team_result) %>%
    rename(team = team_away, points_for = team_final_score_away, points_against = team_final_score_home) %>%
    arrange(game_id) %>%
    group_by(team, competition_year) %>%
    mutate(away_record =
             cumsum(case_when(
               home_team_result == 'Win'  ~ -1, 
               home_team_result == 'Loss' ~ 1,
               TRUE                       ~ 0)),
           points_for = cumsum(points_for),
           points_against = cumsum(points_against),
           points_diff = points_for - points_against) %>%
    mutate_at(vars(away_record, points_for, points_against, points_diff), lag) %>%
    replace(is.na(.), 0) %>%
    ungroup() %>%
    select(-c(competition_year, home_team_result))
  
  season_record <- home_record %>%
    left_join(away_record, by = "game_id", suffix = c("_home", "_away")) %>%
    mutate(season_record = home_record + away_record,
           points_for_season = points_for_home + points_for_away,
           points_against_season = points_against_home + points_against_away,
           points_diff_season = points_for_season - points_against_season)
  
  data <- data %>%
    select(-c(team_home, team_away)) %>%
    left_join(season_record, by = "game_id")
  
  return(data)
  
}



feature_engineering <- function(data){
 
  data <- data %>%
    corona_season() %>%
    timing_vars() %>%
    season_stats()

  return(data)

}
