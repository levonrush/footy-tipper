# Function to calculate home ground advantage using a random forest model
home_ground_advantage <- function(data){

  set.seed(69)  # Set seed for reproducibility

  # Create a dataset for the random forest model
  hga_data <- data %>%
    mutate(points_diff = team_final_score_home - team_final_score_away,
           game_hour = hour(start_time),
           game_day = weekdays(start_time)) %>% 
    filter(game_state_name == 'Final' & !is.na(team_head_to_head_odds_away)) %>%
    select(all_of(c("round_name", "punt_odds_diff", "venue_name", "team_away", "team_home", "elo_prob_diff", "city", "elo_diff", "position_diff", "average_losing_margin_diff", "matchup_form", "close_game_rate_diff", "avg_points_difference_diff", "average_winning_margin_diff", "avg_points_for_diff")), points_diff)

  # Fit the random forest model
  rf <- randomForest(points_diff ~ ., data = hga_data)

  # Create a dataset for training data and another for inference data
  train_data <- data %>%
    filter(game_state_name == 'Final' & !is.na(team_head_to_head_odds_away)) %>%
    mutate(home_ground_advantage = rf$predicted) %>%
    select(game_id, home_ground_advantage)

  inference_data <- data %>%
    filter(game_state_name != 'Final') %>%
    mutate(home_ground_advantage = predict(rf, .)) %>%
    select(game_id, home_ground_advantage)

  # Combine the two datasets
  hga_data <- bind_rows(train_data, inference_data)

  # Add the calculated home ground advantage to the original dataset
  data <- data %>%
    left_join(hga_data, by = c("game_id"))

  return(data)  # Return the modified dataset
}