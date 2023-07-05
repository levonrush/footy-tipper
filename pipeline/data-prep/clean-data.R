# A function that replaces missing values in a dataset. 
# If a variable is an integer or numeric, missing values are replaced with 'num_val'. 
# If a variable is a character, missing values are replaced with 'char_val'.
dumb_impute <- function(data, num_val, char_val){
  
  # mutate_if applies a function to a subset of variables, which are selected with a predicate function. 
  # The predicate function is applied to the data frame to select the variables, 
  # then the selected variables are transformed as specified.
  data <- data %>% 
    # Replace missing values in integer columns with 'num_val'
    mutate_if(is.integer, ~replace(., is.na(.), num_val)) %>%
    # Replace missing values in character columns with 'char_val'
    mutate_if(is.character, ~replace(., is.na(.), char_val)) %>%
    # Replace missing values in numeric columns with 'num_val'
    mutate_if(is.numeric, ~replace(., is.na(.), num_val))
  
  # Return the imputed data
  return(data)
}

# A function to clean a dataset by handling missing values, categorising variables, 
# collapsing levels of factors, and renaming variables to be snake case.
clean_data <- function(data){
  
  # This pipeline performs several data cleaning operations:
  # - Replaces missing values in 'broadcast_channel1', 'broadcast_channel2', and 'broadcast_channel3' with "None"
  # - Collapses levels of 'venue_name' to the top 42 levels (all other levels are grouped into a single level)
  # - Converts character variables to factors
  # - Collapses levels of factor variables to the top 40 levels
  # - Converts column names to snake case (lowercase with underscores)
  data <- data %>%
    mutate(broadcast_channel1 = if_else(is.na(broadcast_channel1), "None", broadcast_channel1),
           broadcast_channel2 = if_else(is.na(broadcast_channel2), "None", broadcast_channel2),
           broadcast_channel3 = if_else(is.na(broadcast_channel3), "None", broadcast_channel3),
           venueName = fct_lump(venue_name, 42)) %>%
    mutate_if(is.character, as.factor) %>%
    mutate_if(is.factor, fct_lump_n, n = 40, ties.method = "max") %>%
    clean_names()
  
  # Ensures the factor levels for 'team_home' and 'team_away' include 'Dolphins', regardless of whether it appears in the data.
  data <- data %>%
    mutate(team_home = fct_expand(team_home, "Dolphins"),
           team_away = fct_expand(team_away, "Dolphins"))
  
  # Return the cleaned data
  return(data)
}
