# The 'map_margin_to_outcome' function is used to normalize the game margins to a value between 0 and 1.
map_margin_to_outcome <- function(margin, marg.max = 80, marg.min = -80){
  # Normalizing the margins
  norm <- (margin - marg.min)/(marg.max - marg.min)
  # Making sure the normalized values lie in the range [0,1]
  norm %>% pmin(1) %>% pmax(0)
}

last_observed_before_elo <- function(observed_values, default = NA_real_) {
  vals <- as.numeric(observed_values)
  out <- rep(default, length(vals))
  last_val <- NA_real_

  for (i in seq_along(vals)) {
    out[i] <- ifelse(is.na(last_val), default, last_val)
    if (!is.na(vals[i])) {
      last_val <- vals[i]
    }
  }

  out
}

# The 'elo_variables' function calculates leak-safe ELO ratings and probabilities.
# Ratings are fit on finalized games only, then carried forward to pre-game rows.
elo_variables <- function(data, marg.max = 80, marg.min = -80, carry_over, k_val, elo_init){

  print("ELO Variables: Calculating ELO rankings...")

  ordered <- data %>%
    select(-any_of(c(
      "home_elo", "away_elo", "elo_diff",
      "home_elo_prob", "away_elo_prob", "elo_draw_prob", "elo_prob_diff"
    ))) %>%
    arrange(competition_year, round_id, start_time, game_number, game_id)

  finals <- ordered %>%
    filter(game_state_name == "Final")

  if (nrow(finals) == 0) {
    return(
      ordered %>%
        mutate(
          home_elo = elo_init,
          away_elo = elo_init,
          elo_diff = 0,
          home_elo_prob = 0.5,
          away_elo_prob = 0.5,
          elo_draw_prob = 0,
          elo_prob_diff = 0
        )
    )
  }

  hga <- finals %>%
    mutate(home_points_diff = team_final_score_home - team_final_score_away) %>%
    summarise(mean = mean(home_points_diff, na.rm = TRUE)) %>%
    .[["mean"]]
  if (is.na(hga)) {
    hga <- 0
  }

  elo_model <- elo.run(
    map_margin_to_outcome(
      finals$team_final_score_home - finals$team_final_score_away,
      marg.max = marg.max,
      marg.min = marg.min
    ) ~ adjust(team_home, hga) +
      team_away +
      regress(competition_year, elo_init, carry_over),
    k = k_val,
    data = finals
  )

  elo_results <- as.data.frame(elo_model)
  finals_pre_game_elo <- finals %>%
    mutate(
      pre_home_elo = elo_results$elo.A - elo_results$update.A,
      pre_away_elo = elo_results$elo.B - elo_results$update.B
    ) %>%
    select(game_id, pre_home_elo, pre_away_elo)

  team_rows <- ordered %>%
    transmute(
      game_id,
      competition_year,
      round_id,
      game_number,
      start_time,
      side = "home",
      team = team_home
    ) %>%
    bind_rows(
      ordered %>%
        transmute(
          game_id,
          competition_year,
          round_id,
          game_number,
          start_time,
          side = "away",
          team = team_away
        )
    ) %>%
    arrange(team, competition_year, round_id, start_time, game_number, game_id, side)

  observed_elos <- bind_rows(
    finals_pre_game_elo %>%
      transmute(game_id, side = "home", observed_elo = pre_home_elo),
    finals_pre_game_elo %>%
      transmute(game_id, side = "away", observed_elo = pre_away_elo)
  )

  team_rows <- team_rows %>%
    left_join(observed_elos, by = c("game_id", "side")) %>%
    group_by(team) %>%
    arrange(competition_year, round_id, start_time, game_number, game_id, .by_group = TRUE) %>%
    mutate(current_elo = last_observed_before_elo(observed_elo, default = elo_init)) %>%
    ungroup()

  home_elo_lookup <- team_rows %>%
    filter(side == "home") %>%
    select(game_id, home_elo = current_elo)

  away_elo_lookup <- team_rows %>%
    filter(side == "away") %>%
    select(game_id, away_elo = current_elo)

  result <- ordered %>%
    left_join(home_elo_lookup, by = "game_id") %>%
    left_join(away_elo_lookup, by = "game_id") %>%
    mutate(
      home_elo = replace_na(home_elo, elo_init),
      away_elo = replace_na(away_elo, elo_init),
      elo_diff = home_elo - away_elo,
      home_elo_prob_raw = 1 / (1 + 10 ^ ((away_elo - (home_elo + hga)) / 400)),
      away_elo_prob_raw = 1 - home_elo_prob_raw,
      prob_bucket = round(20 * home_elo_prob_raw) / 20
    )

  # Leak-safe draw probabilities: each row only sees draw outcomes from prior Final games
  # in the same probability bucket.
  bucket_levels <- sort(unique(result$prob_bucket))
  bucket_index <- match(result$prob_bucket, bucket_levels)
  final_counts <- numeric(length(bucket_levels))
  draw_counts <- numeric(length(bucket_levels))
  draw_prob_before <- numeric(nrow(result))

  for (i in seq_len(nrow(result))) {
    b_idx <- bucket_index[i]

    draw_prob_before[i] <- ifelse(
      final_counts[b_idx] > 0,
      draw_counts[b_idx] / final_counts[b_idx],
      0
    )

    if (result$game_state_name[i] == "Final") {
      final_counts[b_idx] <- final_counts[b_idx] + 1
      if (!is.na(result$team_final_score_home[i]) &&
          !is.na(result$team_final_score_away[i]) &&
          result$team_final_score_home[i] == result$team_final_score_away[i]) {
        draw_counts[b_idx] <- draw_counts[b_idx] + 1
      }
    }
  }

  result <- result %>%
    mutate(elo_draw_prob = draw_prob_before)

  result %>%
    mutate(
      elo_draw_prob = replace_na(elo_draw_prob, 0),
      home_elo_prob = home_elo_prob_raw - home_elo_prob_raw * elo_draw_prob,
      away_elo_prob = away_elo_prob_raw - away_elo_prob_raw * elo_draw_prob,
      elo_prob_diff = home_elo_prob - away_elo_prob
    ) %>%
    select(-home_elo_prob_raw, -away_elo_prob_raw, -prob_bucket)
}
