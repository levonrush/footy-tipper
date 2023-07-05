# Function to add a result column based on the pipeline specification
fixture_result <- function(data, pipeline){
  
  # For a binomial classification, create a binary outcome variable: 'Win' or 'Loss'
  if (pipeline == 'binomial'){
    
    data <- data %>%
      mutate(home_team_result = ifelse(team_final_score_home >= team_final_score_away,
                                       "Win", "Loss") %>% as.factor())
    
    
  } 
  # For a multiclass classification, create an outcome variable with three levels: 'Win', 'Loss', or 'Draw'
  else if (pipeline == 'multiclass'){
    
    data <- data %>%
      mutate(home_team_result = case_when(
        team_final_score_home > team_final_score_away ~ "Win",
        team_final_score_home < team_final_score_away ~ "Loss",
        TRUE                                          ~ "Draw") %>% as.factor())
    
  } 
  # For Elo ratings, create variables to keep track of wins, losses, and draws for home and away teams and calculate game margin
  else if (pipeline == 'elo'){
    
    data <- data %>%
      mutate(home_team_result = case_when(
        team_final_score_home > team_final_score_away ~ "Win",
        team_final_score_home < team_final_score_away ~ "Loss",
        TRUE                                          ~ "Draw") %>% as.factor())
    
    data <- data %>% 
      mutate(home_result = case_when(team_final_score_home > team_final_score_away ~ 1,  # win for home team
                                     team_final_score_home < team_final_score_away ~ 0,  # loss for home team
                                     team_final_score_home == team_final_score_away ~ 0.5),  # draw
             away_result = case_when(team_final_score_home < team_final_score_away ~ 1,  # win for away team
                                     team_final_score_home > team_final_score_away ~ 0,  # loss for away team
                                     team_final_score_home == team_final_score_away ~ 0.5),  # draw
             margin = abs(team_final_score_home - team_final_score_away))  # absolute margin of the game
    
  }
  
  return(data)  # Return the modified dataset

}

# Function to calculate turnaround times between games for each team
turn_around <- function(data){
  
  # Creating two datasets, one for home games and one for away games
  home_games <- data %>%
    rename(team = team_home)  # Rename team_home to team in home_games dataset
  
  away_games <- data %>%
    rename(team = team_away)  # Rename team_away to team in away_games dataset
  
  # Combine the two datasets and calculate the time difference between successive games for each team
  turn_arounds <- bind_rows(home_games, away_games) %>%
    arrange(start_time) %>%
    group_by(team) %>%
    mutate(turn_around = difftime(start_time, lag(start_time), units = 'days') %>% as.numeric()) %>%
    ungroup() %>%
    select(game_id, team, turn_around) %>%
    mutate(turn_around = replace_na(turn_around, mean(turn_around, na.rm = T)))  # Replace NA values with the mean turnaround time
  
  # Add the calculated turnaround times to the original dataset
  data <- data %>%
    left_join(turn_arounds, by = c("game_id", "team_home" = "team")) %>%
    left_join(turn_arounds, by = c("game_id", "team_away" = "team"), suffix = c("_home", "_away")) %>%
    mutate(turn_around_diff = turn_around_home - turn_around_away)  # Calculate the difference between home and away turnarounds
  
  return(data)  # Return the modified dataset
}

# Function to create a feature indicating whether the game is a state of origin game
state_of_origin <- function(data){
  
  data <- data %>%
    mutate(state_of_origin = ifelse(str_detect(round_name, "Round") & n() <= 5, 1, 0))  # If the round name contains "Round" and there are 5 or less records, set state_of_origin to 1, else 0
  
  return(data)  # Return the modified dataset
}

# This function adds new columns to the data which could be useful for further analysis or prediction
easy_pickings <- function(data){
  
  data <- data %>%
    # Calculate the difference in positions between home and away teams
    mutate(position_diff = position_home - position_away,
           # Flag if the season is the 2020 season, which was affected by the Covid-19 pandemic
           corona_season = ifelse(competition_year == 2020, T, F),
           # Extract the start hour of the game from the start_time column
           start_hour = hour(start_time),
           # Determine the day of the week the game is played on
           game_day = weekdays(as.Date(start_time)) %>% as.factor())
  
  # Return the modified dataset
  return(data)
  
}

# The 'season_stats' function calculates and adds season statistics to the dataset for each team, both when playing home and away
season_stats <- function(data){
  
  # Split data into home and away games
  home_games <- data %>%
    select(game_id, competition_year, team_home, team_final_score_home, team_final_score_away, home_team_result) %>%
    rename(team = team_home, points_for = team_final_score_home, points_against = team_final_score_away) %>%
    arrange(game_id) 

  away_games <- data %>%
    select(game_id, competition_year, team_away, team_final_score_home, team_final_score_away, home_team_result) %>%
    rename(team = team_away, points_for = team_final_score_away, points_against = team_final_score_home) %>%
    arrange(game_id)
  
  # Combine home and away data, calculate cumulative season stats (points for, against and record)
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
  
  # Join the computed season stats back to the original data
  data <- data %>%
    left_join(season_record[[1]] %>% select(-is_home_team), by = "game_id") %>%
    left_join(season_record[[2]] %>% select(-is_home_team), by = "game_id", suffix = c("_away", "_home"))
  
  return(data)
  
}

# The 'form_stats' function calculates and adds short-term form statistics (over a specified period of games) to the dataset for each team, both when playing at home and away
form_stats <- function(data, form_period){
  
  # Split data into home and away games
  home_games <- data %>%
    select(game_id, competition_year, team_home, team_final_score_home, team_final_score_away, home_team_result) %>%
    rename(team = team_home, points_for = team_final_score_home, points_against = team_final_score_away) %>%
    arrange(game_id) 

  away_games <- data %>%
    select(game_id, competition_year, team_away, team_final_score_home, team_final_score_away, home_team_result) %>%
    rename(team = team_away, points_for = team_final_score_away, points_against = team_final_score_home) %>%
    arrange(game_id) 
  
  # Combine home and away data, calculate form stats (record of results and average points for and against over the specified period)
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
  
  # Join the computed form stats back to the original data
  data <- data %>%
    left_join(season_form[[1]] %>% select(-is_home_team), by = "game_id") %>%
    left_join(season_form[[2]] %>% select(-is_home_team), by = "game_id", suffix = c("_away", "_home"))
  
  return(data)
  
}

# The 'matchup_form' function calculates and adds a matchup form statistic to the dataset for each pair of teams (team_home, team_away) over a specified number of their most recent games

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

# The 'feature_engineering' function is a wrapper function that applies multiple data transformation and feature engineering functions on the input data
feature_engineering <- function(data, form_period){
 
  data <- data %>%
    easy_pickings() %>%
    turn_around() %>%
    matchup_form(form_period = form_period) %>%
    state_of_origin()

  return(data)

}
