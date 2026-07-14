data_pipeline <- function(year_span, pipeline, form_period, carry_over, k_val, elo_init, use_odds, include_performance = TRUE, prep_mode = "full", db_path = NULL) {

    # Step 1: Calling 'get_data' function to fetch data for the specified range of years.
    # Step 2: The fetched data is then passed to 'clean_data' function for data cleaning.
    # Step 3: The cleaned data is passed to 'fixture_result' function to get the fixture and result data.
    # Step 4: The data from the previous step is passed to 'feature_engineering' function for feature extraction and engineering.
    # Step 5: 'elo_variables' function is called with the data from previous step, carry_over, k_val and elo_init for ELO rating calculations.
    # Step 6: The resulting data is passed to 'home_ground_advantage' function to calculate the home ground advantage.
    # Step 7: The resulting data is filtered to remove the first round of each competition.

    print("Data Pipeline: Beginning data preparation...")
    footy_tipping_data <- get_data(
      year_span = year_span,
      include_performance = include_performance,
      prep_mode = prep_mode,
      db_path = db_path
    )

    footy_tipping_data <- footy_tipping_data %>%
      clean_data()

    footy_tipping_data <- footy_tipping_data %>%
      fixture_result(pipeline = pipeline)

    footy_tipping_data <- footy_tipping_data %>%
      feature_engineering(form_period = form_period, db_path = db_path)

    footy_tipping_data <- footy_tipping_data %>%
      elo_variables(
        carry_over = carry_over,
        k_val = k_val, elo_init = elo_init
      )

    print("Data Pipeline: Filtering data...")
    unique_years <- unique(footy_tipping_data$competition_year)
    if (length(unique_years) > 1) {
      footy_tipping_data <- footy_tipping_data %>%
        filter(competition_year != min(competition_year))
    } else {
      print("Data Pipeline: Single season in scope. Skipping first-season drop.")
    }

    # If use_odds is TRUE, then only rows where team_head_to_head_odds_away is not NA are filtered.
    if (use_odds == TRUE) {
      footy_tipping_data <- footy_tipping_data %>%
        filter(!is.na(team_head_to_head_odds_away))
    }

    print("Data Pipeline: Splitting and assigning datasets...")
    training_data <- footy_tipping_data %>%
      filter(game_state_name == "Final")
    
    if (nrow(training_data) == 0 && prep_mode %in% c("full", "train")) {
      stop("Data Pipeline: No training rows found with game_state_name == 'Final'.")
    } else if (nrow(training_data) == 0) {
      print("Data Pipeline: No final rows in current infer scope. Training table upsert will be skipped.")
    }
    
    inference_data <- footy_tipping_data %>%
      filter(game_state_name == "Pre Game")
    
    if (nrow(inference_data) > 0) {
      latest_year <- max(inference_data$competition_year, na.rm = TRUE)
      inference_data <- inference_data %>%
        filter(competition_year == latest_year) %>%
        filter(round_id == min(round_id, na.rm = TRUE))
    } else {
      print("Data Pipeline: No pre-game rows found. Inference dataset will be empty.")
    }

    # The final processed data is returned as a list containing 'footy_tipping_data', 'train_df' and 'inference_df'.
    print("Data Pipeline: Data preparation complete!")
    return(list(footy_tipping_data = footy_tipping_data,
                training_data = training_data,
                inference_data = inference_data))

}
