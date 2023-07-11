data_pipeline <- function(year_span, pipeline, form_period, carry_over, k_val, elo_init) {

    # The 'data_pipeline' function acts as a wrapper for running the entire data processing pipeline.
    
    # Step 1: 'get_data' function is called to fetch data for the specified range of years.
    # Step 2: 'clean_data' function is called for data cleaning.
    # Step 3: 'fixture_result' function is called to get the fixture and result data.
    # Step 4: 'feature_engineering' function is called for feature extraction and engineering.
    # Step 5: 'elo_variables' function is called for ELO rating calculations.
    # Step 6: 'home_ground_advantage' function is called to calculate the home ground advantage.
    
    footy_tipping_data <- get_data(year_span = year_span) %>%
      clean_data() %>%
      fixture_result(pipeline = pipeline) %>%
      feature_engineering(form_period = form_period) %>%
      elo_variables(
        carry_over = carry_over,
        k_val = k_val, elo_init = elo_init
      ) %>%
      home_ground_advantage()

    # The final processed data is returned.
    return(footy_tipping_data)

}
