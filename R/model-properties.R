model_properties <- function(model, train_df, positive){
  
  confusion_matrix <- model$confusion[-1,2:3] %>% confusionMatrix(positive = positive)
  
  importance_plot <- importance(model) %>% t() %>% .[1,] %>% 
    as.tibble(rownames = "variable") %>%
    mutate(variable = as.factor(variable)) %>%
    mutate(variable = fct_reorder(variable, value, .fun = max)) %>%
    ggplot(aes(x = variable, y = value, fill = variable)) +
    geom_bar(stat = "identity") +
    coord_flip() +
    theme_minimal() +
    theme(legend.position = "none") +
    labs(title = "Variable importance plot",
         x = NULL, y = "Importance")
  
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
  
  return(list(confusion_matrix = confusion_matrix,
              importance_plot = importance_plot,
              round_performance = round_performance))
  
  
}
