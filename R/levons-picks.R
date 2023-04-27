levons_picks <- function(predictions){
  
  punt_thresholds <- predictions %>%
    mutate(home_odds_thresh = 1/home_team_win_prob,
           away_odds_thresh = 1/home_team_lose_prob)
  
  home_picks <- punt_thresholds %>%
    filter(home_team_result == 'Win') %>%
    select(team_home, team_head_to_head_odds_home, home_odds_thresh) %>%
    rename(team = team_home, price = team_head_to_head_odds_home, price_min = home_odds_thresh)
  
  away_picks <- punt_thresholds %>%
    filter(home_team_result == 'Loss') %>%
    select(team_away, team_head_to_head_odds_away, away_odds_thresh) %>%
    rename(team = team_away, price = team_head_to_head_odds_away, price_min = away_odds_thresh)
  
  levons_picks <- bind_rows(home_picks, away_picks) %>%
    filter(price > price_min) %>%
    mutate_if(is.numeric, dollar)
  
  write.csv(x = predictions, file = "levons_picks.csv")
  
  # Upload the file
  drive_upload(media = "levons_picks.csv",
               path = "footy-tipping-predictions/levons_picks",
               name = paste0("levons_picks_round", unique(inference_df$round_id), "_", unique(inference_df$competition_year), ".csv"), 
               type = NULL,
               overwrite = TRUE)
  
  unlink("levons_picks.csv")
  
  return(levons_picks)
  
}
