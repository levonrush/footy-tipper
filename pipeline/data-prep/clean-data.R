dumb_impute <- function(data, num_val, char_val){
  
  data <- data %>% 
    mutate_if(is.integer, ~replace(., is.na(.), num_val)) %>%
    mutate_if(is.character, ~replace(., is.na(.), char_val)) %>%
    mutate_if(is.numeric, ~replace(., is.na(.), num_val))
  
  return(data)
  
} 

clean_data <- function(data){
  
  data <- data %>%
    mutate(broadcast_channel1 = if_else(is.na(broadcast_channel1), "None", broadcast_channel1),
           broadcast_channel2 = if_else(is.na(broadcast_channel2), "None", broadcast_channel2),
           broadcast_channel3 = if_else(is.na(broadcast_channel3), "None", broadcast_channel3),
           venueName = fct_lump(venue_name, 42)) %>%
    mutate_if(is.character, as.factor) %>%
    mutate_if(is.factor, fct_lump_n, n = 40, ties.method = "max") %>%
    clean_names()
  
  # introduce the Dolphins as a factor level for R1 2023
  data <- data %>%
    mutate(team_home = fct_expand(team_home, "Dolphins"),
           team_away = fct_expand(team_away, "Dolphins"))
  
  return(data)
  
}

