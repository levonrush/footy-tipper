model_properties <- function(model, train_df, positive){
  
  confusion_matrix <- model$confusion[-1,2:3] %>% confusionMatrix(positive = positive)
  
  importance <- importance(model) %>% t() %>% .[1,] %>% 
    as.tibble(rownames = "variable") %>%
    rename(importance = value) %>%
    mutate(variable = as.factor(variable)) %>%
    mutate(variable = fct_reorder(variable, importance, .fun = max)) %>%
    arrange(desc(importance))
  
  round_performance <- train_df %>%
    bind_cols(model$votes) %>%
    mutate(prediction = ifelse(Win >= 0.5, "Win", "Loss")) %>%
    mutate(round_name = fct_reorder(round_name, round_id)) %>%
    group_by(round_name) %>%
    summarise(accuracy = sum(home_team_result == prediction) / n()) %>%
    ggplot(aes(x = factor(round_name), y = accuracy)) +
      geom_bar(stat="identity", fill="steelblue") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))
  
  benchmarking <- train_df %>%
    mutate(ladder_prediction = ifelse(position_home < position_away, 'Win', 'Loss'),
           bet_prediction = case_when(
             team_head_to_head_odds_home < team_head_to_head_odds_away ~ 'Win',
             team_head_to_head_odds_home > team_head_to_head_odds_away ~ 'Loss',
             TRUE ~ as.character(NA)),
           elo_prediction = case_when(
             home_elo > away_elo ~ 'Win',
             home_elo < away_elo ~ 'Loss',
             TRUE ~ as.character(NA))
    ) %>%
    summarise(
      home_team_accuracy = sum(home_team_result == 'Win')/n(),
      ladder_accuracy = sum(ladder_prediction == home_team_result)/n(),
      elo_accuracy = sum(elo_prediction == home_team_result)/n(),
      bet_accuracy = sum(bet_prediction == home_team_result, na.rm = T)/n(),
      model_accuracy = confusion_matrix$overall['Accuracy'])
  
  return(list(confusion_matrix = confusion_matrix,
              importance = importance,
              round_performance = round_performance, 
              benchmarking = benchmarking))
  
}
