data_pipeline <- function(year_span, pipeline, form_period, carry_over, k_val, elo_init, use_odds) {
    
    # Step 1: Calling 'get_data' function to fetch data for the specified range of years.
    # Step 2: The fetched data is then passed to 'clean_data' function for data cleaning.
    # Step 3: The cleaned data is passed to 'fixture_result' function to get the fixture and result data.
    # Step 4: The data from the previous step is passed to 'feature_engineering' function for feature extraction and engineering.
    # Step 5: 'elo_variables' function is called with the data from previous step, carry_over, k_val and elo_init for ELO rating calculations.
    # Step 6: The resulting data is passed to 'home_ground_advantage' function to calculate the home ground advantage.
    # Step 7: The resulting data is filtered to remove the first round of each competition.
    footy_tipping_data <- get_data(year_span = year_span) %>%
      clean_data() %>%
      fixture_result(pipeline = pipeline) %>%
      feature_engineering(form_period = form_period) %>%
      elo_variables(
        carry_over = carry_over,
        k_val = k_val, elo_init = elo_init
      ) %>%
      home_ground_advantage() %>%
      filter(competition_year != min(competition_year))

    # If use_odds is TRUE, then only rows where team_head_to_head_odds_away is not NA are filtered.
    if (use_odds == TRUE) {
      footy_tipping_data <- footy_tipping_data %>%
        filter(!is.na(team_head_to_head_odds_away))
    } 
    
    # The data is grouped by game_state_name and split into two lists.
    train_inference_split <- footy_tipping_data  %>%
      group_by(game_state_name) %>%
      group_split()

    # The first element of the list is assigned to train_df.
    training_data <- train_inference_split[[1]]

    # For the second element of the list, only rows where round_id equals the minimum round_id are filtered and assigned to inference_df.
    inference_data <- train_inference_split[[2]] %>%
      filter(round_id == min(round_id))

    # The final processed data is returned as a list containing 'footy_tipping_data', 'train_df' and 'inference_df'.
    return(list(footy_tipping_data = footy_tipping_data,
                training_data = training_data,
                inference_data = inference_data))

}
