save_predictions <- function(predictions, prod_run = FALSE){

  if (prod_run == TRUE){

    googledrive::drive_auth(
      path = paste0(here(), "/footy-tipper-c5bcb9639ee2.json"),
      scopes = "https://www.googleapis.com/auth/drive"
    )
    write.csv(x = predictions, file = "predictions.csv")

    # Upload the file
    drive_upload(media = "predictions.csv",
                 path = "footy-tipping-predictions/",
                #  name = paste0("round", unique(predictions$round_id), "_",
                #                unique(predictions$competition_year), ".csv"),
                name = "test",
                 type = NULL,
                 overwrite = TRUE)

    unlink("predictions.csv")

  }

}

get_punt_thresholds <- function(predictions){

  punt_thresholds <- predictions %>%
    mutate(home_odds_thresh = 1/home_team_win_prob,
           away_odds_thresh = 1/home_team_lose_prob) %>%
    select(team_home, home_odds_thresh, team_away, away_odds_thresh)
  
  return(punt_thresholds)
  
}

levons_picks <- function(predictions, prod_run = FALSE){
  
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
    filter(price > (price_min*1.15)) %>%
    mutate_if(is.numeric, dollar)
  
  if (prod_run == TRUE){
  
    write.csv(x = levons_picks, file = "levons_picks.csv")
    
    # Upload the file
    drive_upload(media = "levons_picks.csv",
                 path = "footy-tipping-predictions/levons_picks",
                 name = paste0("levons_picks_round", unique(predictions$round_id), "_", unique(predictions$competition_year), ".csv"), 
                 type = NULL,
                 overwrite = TRUE)
    
    unlink("levons_picks.csv")
  
  }
  
  return(levons_picks)
  
}
