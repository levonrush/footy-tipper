clean_data <- function(data){
  
  data <- data %>%
    mutate(broadcastChannel1 = if_else(is.na(broadcastChannel1), "None", broadcastChannel1),
           broadcastChannel2 = if_else(is.na(broadcastChannel2), "None", broadcastChannel2),
           broadcastChannel3 = if_else(is.na(broadcastChannel3), "None", broadcastChannel3),
           venueName = fct_lump(venueName, 42)) %>%
    mutate_if(is.character, as.factor) %>%
    clean_names()
  
  # introduce the Dolphins as a factor level for R1 2023
  data <- data %>%
    mutate(team_home = fct_expand(team_home, "Dolphins"),
           team_away = fct_expand(team_away, "Dolphins"))
  
  # # R1 ladder positions move to the end of last year - this is a bad soln and needs updating
  # data <- data %>%
  #   arrange(game_id) %>%
  #   group_by(team_home) %>%
  #   mutate(team_position_home = ifelse(round_id == 1, lag(team_position_home), team_position_home)) %>%
  #   group_by(team_away) %>%
  #   mutate(team_position_away = ifelse(round_id == 1, lag(team_position_away), team_position_away)) %>%
  #   ungroup() %>%
  #   # fix intro for dolphins
  #   mutate(team_position_home = ifelse(game_id == 20231110170, 17, team_position_home))
  #   
    return(data)
  
}

