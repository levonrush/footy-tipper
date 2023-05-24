data_pipeline <- function(year_span, pipeline, form_period, carry_over, k_val, elo_init) {

    footy_tipping_data <- get_data(year_span = year_span) %>%
      clean_data() %>%
      fixture_result(pipeline = pipeline) %>%
      feature_engineering(form_period = form_period, pipeline = pipeline) %>%
      elo_variables(carry_over = carry_over, k_val = k_val, elo_init = elo_init) %>%
      home_ground_advantage() %>%
      filter(competition_year != min(competition_year),
              !is.na(team_head_to_head_odds_away)) %>%
      group_by(game_state_name) %>%
      group_split()

    train_df <- footy_tipping_data[[1]]

    inference_df <- footy_tipping_data[[2]] %>%
      filter(round_id == min(round_id))

    return(list(train_df = train_df,
                inference_df = inference_df))

}
