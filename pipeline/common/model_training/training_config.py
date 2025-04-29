import numpy as np
from sklearn.metrics import make_scorer, mean_poisson_deviance

def custom_poisson_deviance(y_true, y_pred):
    y_pred = np.maximum(y_pred, 1e-9)  # Ensure predictions are strictly positive
    return mean_poisson_deviance(y_true, y_pred)

opt_metric = make_scorer(custom_poisson_deviance, greater_is_better=False)

# modelling parameters
num_folds = 3
use_rfe = False

# predictors
include_performance = True

predictors = [
    "round_id", "round_name", "game_number", "game_state_name",
    "start_time", "start_time_utc", "venue_name", "city", "crowd",
    "broadcast_channel1", "broadcast_channel2", "broadcast_channel3",
    "team_home", "team_head_to_head_odds_home",
    "team_line_odds_home", "team_line_amount_home", "team_away",
    "team_head_to_head_odds_away", "team_line_odds_away",
    "team_line_amount_away", "competition_year", "position_home_ladder",
    "wins_home_ladder", "draws_home_ladder", "losses_home_ladder",
    "byes_home_ladder", "competition_points_home_ladder", "points_for_home_ladder",
    "points_against_home_ladder", "points_difference_home_ladder",
    "home_wins_home_ladder", "home_draws_home_ladder", "home_losses_home_ladder",
    "away_wins_home_ladder", "away_draws_home_ladder", "away_losses_home_ladder",
    "recent_form_home_ladder", "season_form_home_ladder", "tries_for_home_ladder",
    "tries_conceded_home_ladder", "goals_for_home_ladder", "goals_conceded_home_ladder",
    "field_goals_for_home_ladder", "field_goals_conceded_home_ladder",
    "players_used_home_ladder", "average_winning_margin_home_ladder",
    "average_losing_margin_home_ladder", "close_games_home_ladder",
    "win_rate_home_ladder", "draw_rate_home_ladder", "loss_rate_home_ladder",
    "competition_point_rate_home_ladder", "avg_points_for_home_ladder",
    "avg_points_against_home_ladder", "avg_points_difference_home_ladder",
    "home_win_rate_home_ladder", "home_draw_rate_home_ladder", "home_loss_rate_home_ladder",
    "away_win_rate_home_ladder", "away_draw_rate_home_ladder", "away_loss_rate_home_ladder",
    "avg_tries_for_home_ladder", "avg_tries_conceded_home_ladder", "avg_goals_for_home_ladder",
    "avg_goals_conceded_home_ladder", "close_game_rate_home_ladder", "position_away_ladder",
    "wins_away_ladder", "draws_away_ladder", "losses_away_ladder", "byes_away_ladder",
    "competition_points_away_ladder", "points_for_away_ladder", "points_against_away_ladder",
    "points_difference_away_ladder", "home_wins_away_ladder", "home_draws_away_ladder",
    "home_losses_away_ladder", "away_wins_away_ladder", "away_draws_away_ladder",
    "away_losses_away_ladder", "recent_form_away_ladder", "season_form_away_ladder",
    "tries_for_away_ladder", "tries_conceded_away_ladder", "goals_for_away_ladder",
    "goals_conceded_away_ladder", "field_goals_for_away_ladder", "field_goals_conceded_away_ladder",
    "players_used_away_ladder", "average_winning_margin_away_ladder",
    "average_losing_margin_away_ladder", "close_games_away_ladder", "win_rate_away_ladder",
    "draw_rate_away_ladder", "loss_rate_away_ladder", "competition_point_rate_away_ladder",
    "avg_points_for_away_ladder", "avg_points_against_away_ladder", "avg_points_difference_away_ladder",
    "home_win_rate_away_ladder", "home_draw_rate_away_ladder", "home_loss_rate_away_ladder",
    "away_win_rate_away_ladder", "away_draw_rate_away_ladder", "away_loss_rate_away_ladder",
    "avg_tries_for_away_ladder", "avg_tries_conceded_away_ladder", "avg_goals_for_away_ladder",


    "avg_goals_conceded_away_ladder", "close_game_rate_away_ladder", "sin_bin_home_performance",
    "conversion_made_home_performance", "conversion_missed_home_performance",
    "tackle_made_home_performance", "tackle_missed_home_performance", "possession_home_performance",
    "territory_home_performance", "offloads_home_performance", "tackle_break_home_performance",
    "linebreak_home_performance", "lb_assist_home_performance", "kicks_home_performance",
    "kicks_occur_home_performance", "try_assist_home_performance", "error_home_performance",
    "try_home_performance", "penalty_shot_made_home_performance", "penalty_shot_missed_home_performance",
    "field_goal_made_home_performance", "field_goal_missed_home_performance", "points_home_performance",
    "kick_return_home_performance", "dh_run_home_performance", "dh_run_occur_home_performance",
    "x40_20_kick_home_performance", "kick_bomb_home_performance", "try_assists_and_involvements_home_performance",
    "line_break_assists_and_involvements_home_performance", "shifts_home_performance",
    "shortsides_home_performance", "captains_challenge_success_percent_home_performance",
    "x20_att_gl_possession_seconds_home_performance", "marker_tackle_home_performance",
    "trebles_home_performance", "markers_home_performance", "shortside_left_home_performance",
    "shortside_right_home_performance", "charge_downs_home_performance", "kick_charged_down_home_performance",
    "decoys_home_performance", "ineffective_tackle_home_performance", "intercept_home_performance",
    "pass_intercepted_home_performance", "one_on_one_steal_home_performance", "on_report_home_performance",
    "one_on_one_lost_home_performance", "one_on_one_tackle_home_performance",
    "bomb_kicks_defused_home_performance", "bomb_kicks_not_defused_home_performance",
    "kick_defused_home_performance", "kick_not_defused_home_performance", "line_dropout_home_performance",
    "supports_home_performance", "try_cause_home_performance", "try_saver_tackle_home_performance",
    "conversion_attempted_home_performance", "field_goal_attempted_home_performance", "half_break_home_performance",
    "kick_chip_home_performance", "kick_crossfield_home_performance", "kick_grubber_home_performance",
    "line_engaged_home_performance", "penalties_home_performance", "penalty_shot_attempted_home_performance",
    "receipts_home_performance", "play_the_ball_home_performance", "all_goals_made_home_performance",
    "all_goals_attempted_home_performance", "all_goals_missed_home_performance", "all_run_metres_home_performance",
    "all_runs_home_performance", "doubles_home_performance", "dummy_pass_home_performance",
    "handling_errors_home_performance", "kick_forces_dropout_home_performance", "kick_pressures_home_performance",
    "long_kicks_finding_space_home_performance", "post_contact_metres_home_performance", "shifts_left_home_performance",
    "shifts_right_home_performance", "goal_conversion_rate_home_performance", "set_completion_rate_home_performance",
    "passes_per_run_home_performance", "field_goal_conversion_rate_home_performance",
    "effective_tackle_percentage_home_performance", "foul_play_penalties_home_performance",
    "offside_penalties_home_performance", "wins_home_performance", "losses_home_performance",
    "draws_home_performance", "captains_challenge_upheld_home_performance", "captains_challenge_overturned_home_performance",
    "ruck_infringement_home_performance", "one_point_field_goal_home_performance", "attacking_kicks_home_performance",
    "one_point_field_goal_missed_home_performance", "two_point_field_goal_missed_home_performance",
    "one_point_field_goal_attempted_home_performance", "two_point_field_goal_attempted_home_performance",
    "set_restart_conceded10m_offside_home_performance", "try_involvement_home_performance",
    "ptb_in_opposition_20_home_performance", "linebreak_involvement_home_performance", "short_dropout_home_performance",
    "sin_bin_away_performance", "conversion_made_away_performance", "conversion_missed_away_performance",
    "tackle_made_away_performance", "tackle_missed_away_performance", "possession_away_performance",
    "territory_away_performance", "offloads_away_performance", "tackle_break_away_performance",
    "linebreak_away_performance", "lb_assist_away_performance", "kicks_away_performance",
    "kicks_occur_away_performance", "try_assist_away_performance", "error_away_performance",
    "try_away_performance", "penalty_shot_made_away_performance", "penalty_shot_missed_away_performance",
    "field_goal_made_away_performance", "field_goal_missed_away_performance",
    "points_away_performance", "kick_return_away_performance", "dh_run_away_performance",
    "dh_run_occur_away_performance", "x40_20_kick_away_performance", "kick_bomb_away_performance",
    "try_assists_and_involvements_away_performance", "line_break_assists_and_involvements_away_performance",
    "shifts_away_performance", "shortsides_away_performance", "captains_challenge_success_percent_away_performance",
    "x20_att_gl_possession_seconds_away_performance", "marker_tackle_away_performance",
    "trebles_away_performance", "markers_away_performance", "shortside_left_away_performance",
    "shortside_right_away_performance", "charge_downs_away_performance", "kick_charged_down_away_performance",
    "decoys_away_performance", "ineffective_tackle_away_performance", "intercept_away_performance",
    "pass_intercepted_away_performance", "one_on_one_steal_away_performance", "on_report_away_performance",
    "one_on_one_lost_away_performance", "one_on_one_tackle_away_performance",
    "bomb_kicks_defused_away_performance", "bomb_kicks_not_defused_away_performance",
    "kick_defused_away_performance", "kick_not_defused_away_performance", "line_dropout_away_performance",
    "supports_away_performance", "try_cause_away_performance", "try_saver_tackle_away_performance",
    "conversion_attempted_away_performance", "field_goal_attempted_away_performance", "half_break_away_performance",
    "kick_chip_away_performance", "kick_crossfield_away_performance", "kick_grubber_away_performance",
    "line_engaged_away_performance", "penalties_away_performance", "penalty_shot_attempted_away_performance",
    "receipts_away_performance", "play_the_ball_away_performance", "all_goals_made_away_performance",
    "all_goals_attempted_away_performance", "all_goals_missed_away_performance", "all_run_metres_away_performance",
    "all_runs_away_performance", "doubles_away_performance", "dummy_pass_away_performance",
    "handling_errors_away_performance", "kick_forces_dropout_away_performance", "kick_pressures_away_performance",
    "long_kicks_finding_space_away_performance", "post_contact_metres_away_performance", "shifts_left_away_performance",
    "shifts_right_away_performance", "goal_conversion_rate_away_performance", "set_completion_rate_away_performance",
    "passes_per_run_away_performance", "field_goal_conversion_rate_away_performance",
    "effective_tackle_percentage_away_performance", "foul_play_penalties_away_performance",
    "offside_penalties_away_performance", "wins_away_performance", "losses_away_performance",
    "draws_away_performance", "captains_challenge_upheld_away_performance", "captains_challenge_overturned_away_performance",
    "ruck_infringement_away_performance", "one_point_field_goal_away_performance", "attacking_kicks_away_performance",
    "one_point_field_goal_missed_away_performance", "two_point_field_goal_missed_away_performance",
    "one_point_field_goal_attempted_away_performance", "two_point_field_goal_attempted_away_performance",
    "set_restart_conceded10m_offside_away_performance", "try_involvement_away_performance",
    "ptb_in_opposition_20_away_performance", "linebreak_involvement_away_performance", "short_dropout_away_performance"
]

def filter_predictors(include_performance=True, predictor_list=predictors):
    if include_performance:
        return predictor_list
    else:
        return [p for p in predictor_list if not p.endswith('_performance')]
    