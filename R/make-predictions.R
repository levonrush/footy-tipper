make_predicitons <- function(data, model, predictors, outcome_var = "home_team_result"){

  predictions <- data.frame(
    game_id = data$game_id,
    team_home = data$team_home,
    team_away = data$team_away,
    home_team_result = predict(model, data %>% select(all_of(c(predictors, outcome_var)))),
    home_team_win_prob = predict(model, data %>% select(all_of(c(predictors, outcome_var))), type = "prob")[,"Win"],
    home_team_draw_prob = predict(model, data %>% select(all_of(c(predictors, outcome_var))), type = "prob")[,"Draw"],
    home_team_lose_prob = predict(model, data %>% select(all_of(c(predictors, outcome_var))), type = "prob")[,"Loss"],
    home_elo = data$home_elo,
    away_elo = data$away_elo
  )
  
  return(predictions)
  
}