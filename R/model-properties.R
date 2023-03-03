model_properties <- function(positive){
  
  confusion_matrix <- model$confusion[-1,2:3] %>% confusionMatrix(positive = positive)
  
  importance_plot <- importance(model) %>% t() %>% .[1,] %>% 
    as.tibble(rownames = "variable") %>%
    arrange(desc(value)) %>%
    ggplot(aes(x = variable, y = value, fill = variable)) +
    geom_bar(stat = "identity") +
    coord_flip() +
    theme_minimal() +
    theme(legend.position = "none") +
    labs(title = "Importance of variables",
         x = NULL, y = "Importance")
  
  return(list(confusion_matrix = confusion_matrix,
              importance_plot = importance_plot))
  
  
}