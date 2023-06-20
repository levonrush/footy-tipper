data_pipeline <- function(year_span, pipeline, form_period, carry_over, k_val, elo_init) {

    footy_tipping_data <- get_data(year_span = year_span) %>%
      clean_data() %>%
      fixture_result(pipeline = pipeline) %>%
      feature_engineering(form_period = form_period) %>%
      elo_variables(
        carry_over = carry_over,
        k_val = k_val, elo_init = elo_init
      ) %>%
      home_ground_advantage()

    return(footy_tipping_data)

}
