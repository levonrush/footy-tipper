# The 'map_margin_to_outcome' function is used to normalize the game margins to a value between 0 and 1.
map_margin_to_outcome <- function(margin, marg.max = 80, marg.min = -80){
  # Normalizing the margins
  norm <- (margin - marg.min)/(marg.max - marg.min)
  # Making sure the normalized values lie in the range [0,1]
  norm %>% pmin(1) %>% pmax(0)
}

# The 'elo_variables' function calculates the ELO ratings and probabilities for each team in the dataset.
elo_variables <- function(data, marg.max = 80, marg.min = -80, carry_over, k_val, elo_init){
  
  # Setting the carryover, K-value, and initial ELO values
  carry_over <- carry_over
  k_val <- k_val
  # Calculating home ground advantage (hga) as mean of home team score difference
  hga <- data %>%
    mutate(home_points_diff = team_final_score_home - team_final_score_away) %>%
    summarise(mean = mean(home_points_diff)) %>% .[['mean']]
  
  # Running the ELO model with the specified features
  elo_model <- elo.run(
    map_margin_to_outcome(team_final_score_home - team_final_score_away) ~
      adjust(team_home, hga) + # Adjusting for home team with home ground advantage
      team_away + # Including team away as a feature
      regress(competition_year, elo_init, carry_over), # Including competition year with regression
    k = k_val,
    data = data
  )
  
  # Extracting ELO results as a data frame
  elo_results <- elo_model %>% as.data.frame()
  
  # Computing draw rates for each probability bucket
  draw_rates <- data.frame(win_prob = elo_model$elos[,3],
                           win_loss_draw = elo_model$elos[,4]) %>%
    mutate(prob_bucket = abs(round((win_prob)*20)) / 20) %>%
    group_by(prob_bucket) %>%
    summarise(draw_prob = sum(ifelse(win_loss_draw == 0.5, 1, 0)) / n())
  
  # Adding ELO scores and probabilities to the original data
  data <- data %>%
    mutate(home_elo = elo_results$elo.A - elo_results$update.A,
           away_elo = elo_results$elo.B - elo_results$update.B,
           home_elo_prob = elo_results$p.A,
           away_elo_prob = 1 - home_elo_prob) %>%
    mutate(prob_bucket = round(20*home_elo_prob)/20) %>% # Creating buckets of probabilities
    left_join(draw_rates, by = "prob_bucket") %>% # Joining draw rates based on the probability buckets
    select(-prob_bucket)
  
  # Adjusting home and away ELO probabilities based on the draw probability
  data <- data %>% 
    mutate(home_elo_prob = home_elo_prob - home_elo_prob * draw_prob,
           away_elo_prob = away_elo_prob - away_elo_prob * draw_prob)
  
  return(data)
  
}

