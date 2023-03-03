# data params
year_span = 2012:2023
pipeline = 'multiclass'
form_period <- 5

# predictors
predictors <- c("round_id"
                , "round_name"
                , "game_number"
                , "venue_name"
                , "team_home"
                , "team_position_home"
                , "team_away"
                , "team_position_away"
                , "competition_year"
                , "broadcast_channel1"
                , "broadcast_channel2"
                , "broadcast_channel3"
                , "corona_season"
                , "start_hour"
                , "game_day"
                , "season_record_away"
                , "season_points_for_away"
                , "season_points_against_away"
                , "season_points_diff_away"
                , "season_record_home"
                , "season_points_for_home"
                , "season_points_against_home"
                , "season_points_diff_home"
                , "season_form_away"
                , "season_points_for_form_away"
                , "season_points_against_form_away"
                , "season_diff_form_away"
                , "season_form_home"
                , "season_points_for_form_home"
                , "season_points_against_form_home"
                , "season_diff_form_home"
                , "matchup_form"
                , "home_elo"
                , "away_elo"
                , "home_prob"
                , "draw_prob"
                , "away_prob"
                )

# outcome variable
outcome_var = "home_team_result"
positive = "Win"

# training params
opt_metric = "Kappa"
