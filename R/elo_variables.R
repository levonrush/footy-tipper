map_margin_to_outcome <- function(margin, marg.max = 80, marg.min = -80){
  norm <- (margin - marg.min)/(marg.max - marg.min)
  norm %>% pmin(1) %>% pmax(0)
}

elo_variables <- function(data, marg.max = 80, marg.min = -80, carry_over, k_val, elo_init){
  
  carry_over <- carry_over
  k_val <- k_val
  hga <- data %>%
    mutate(home_points_diff = team_final_score_home - team_final_score_away) %>%
    summarise(mean = mean(home_points_diff)) %>% .[['mean']]
  
  elo_model <- elo.run(
    map_margin_to_outcome(team_final_score_home - team_final_score_away) ~
      adjust(team_home, hga) +
      team_away +
      regress(competition_year, elo_init, carry_over) #+
      #group(round_id)
    ,
    k = k_val,
    data = data
  )
  
  elo_results <- elo_model %>% as.data.frame()
  
  draw_rates <- data.frame(win_prob = elo_model$elos[,3],
                           win_loss_draw = elo_model$elos[,4]) %>%
    mutate(prob_bucket = abs(round((win_prob)*20)) / 20) %>%
    group_by(prob_bucket) %>%
    summarise(draw_prob = sum(ifelse(win_loss_draw == 0.5, 1, 0)) / n())
  
  data <- data %>%
    mutate(home_elo = elo_results$elo.A - elo_results$update.A,
           away_elo = elo_results$elo.B - elo_results$update.B,
           home_elo_prob = elo_results$p.A,
           away_elo_prob = 1 - home_elo_prob) %>%
    mutate(prob_bucket = round(20*home_elo_prob)/20) %>%
    left_join(draw_rates, by = "prob_bucket") %>%
    select(-prob_bucket)
  
  data <- data %>% 
    mutate(home_elo_prob = home_elo_prob - home_elo_prob * draw_prob,
           away_elo_prob = away_elo_prob - away_elo_prob * draw_prob)
  
  return(data)
  
}
