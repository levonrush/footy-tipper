get_punt_thresholds <- function(predictions){

  punt_thresholds <- predictions %>%
    mutate(home_odds_thresh = 1/home_team_win_prob,
           away_odds_thresh = 1/home_team_lose_prob) %>%
    select(team_home, home_odds_thresh, team_away, away_odds_thresh)
  
  return(punt_thresholds)
  
}
