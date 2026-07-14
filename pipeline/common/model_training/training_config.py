import os
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, mean_poisson_deviance

def custom_poisson_deviance(y_true, y_pred):
    y_pred = np.maximum(y_pred, 1e-9)  # Ensure predictions are strictly positive
    return mean_poisson_deviance(y_true, y_pred)

opt_metric = make_scorer(custom_poisson_deviance, greater_is_better=False)

# modelling parameters
num_folds = 3
use_rfe = False

# predictors
include_performance = os.getenv("FOOTY_TIPPER_INCLUDE_PERFORMANCE", "true").strip().lower() in {"1", "true", "yes", "y"}
sparse_min_support = int(os.getenv("FOOTY_TIPPER_SPARSE_MIN_SUPPORT", "30"))

predictors = [
    "round_id", "round_name", "game_number", "game_state_name",
    "start_time", "start_time_utc", "venue_name", "city", "crowd",
    "broadcast_channel1", "broadcast_channel2", "broadcast_channel3",
    "team_home", "team_away", "competition_year", "position_home_ladder",
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
    "avg_goals_conceded_away_ladder", "close_game_rate_away_ladder",

    # NOTE: streak/day-night ladder splits (current_streak_*, day_win_rate_*,
    # night_win_rate_*) are parsed in R and available in the DB, but the
    # honest eval showed no gain from adding them here (2026-07); re-test
    # after the next full hyperparameter retune before enabling.

    # Engineered rolling/state features built from prior observed (Final) games.
    "position_diff", "corona_season", "start_hour", "game_day",
    "turn_around_home", "turn_around_away", "turn_around_diff",
    "season_record_home", "season_record_away",
    "season_points_for_home", "season_points_for_away",
    "season_points_against_home", "season_points_against_away",
    "season_points_diff_home", "season_points_diff_away",
    "season_form_home", "season_form_away",
    "season_points_for_form_home", "season_points_for_form_away",
    "season_points_against_form_home", "season_points_against_form_away",
    "season_diff_form_home", "season_diff_form_away",
    "matchup_form", "state_of_origin", "post_origin",
    "home_prev_result_diff", "away_prev_result_diff", "prev_result_diff",
    "home_elo", "away_elo", "elo_diff",
    "home_elo_prob", "away_elo_prob", "elo_draw_prob", "elo_prob_diff",

    "sin_bin_home_performance",
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

# Missingness, delta, and other non-market features.
# Market/odds features removed from base learners (naïve market integration).
# Market signal enters only at the stacker level where it is cross-validated.
predictors += [
    "performance_home_missing", "performance_away_missing", "performance_features_missing",
    "ladder_points_delta", "ladder_points_difference_delta", "ladder_rank_delta",
    "ladder_win_rate_delta", "ladder_close_game_rate_delta",
    "form_delta", "points_for_form_delta", "points_against_form_delta",
    "diff_form_delta", "attack_delta", "defence_delta", "rest_delta",
    "points_performance_delta", "set_completion_rate_performance_delta",
    "effective_tackle_percentage_performance_delta", "all_run_metres_performance_delta",

    # Lineup-aware team list features (generated from lineup snapshots).
    "lineup_data_available_home", "lineup_data_available_away",
    "lineup_named_count_home", "lineup_named_count_away",
    "lineup_interchange_count_home", "lineup_interchange_count_away",
    "lineup_reserve_count_home", "lineup_reserve_count_away",
    "lineup_spine_count_home", "lineup_spine_count_away",
    "lineup_spine_complete_home", "lineup_spine_complete_away",
    "lineup_bench_hooker_count_home", "lineup_bench_hooker_count_away",
    "lineup_bench_spine_cover_count_home", "lineup_bench_spine_cover_count_away",
    "lineup_source_age_hours_home", "lineup_source_age_hours_away",
    "lineup_retained_ratio_home", "lineup_retained_ratio_away",
    "lineup_starters_retained_ratio_home", "lineup_starters_retained_ratio_away",
    "lineup_spine_retained_ratio_home", "lineup_spine_retained_ratio_away",
    "lineup_spine_same_as_prev_home", "lineup_spine_same_as_prev_away",
    "lineup_halves_pair_same_as_prev_home", "lineup_halves_pair_same_as_prev_away",
    "lineup_avg_named_experience_home", "lineup_avg_named_experience_away",
    "lineup_avg_spine_experience_home", "lineup_avg_spine_experience_away",
    "lineup_avg_halves_experience_home", "lineup_avg_halves_experience_away",
    "lineup_avg_middles_experience_home", "lineup_avg_middles_experience_away",
    "lineup_avg_edges_experience_home", "lineup_avg_edges_experience_away",
    "lineup_avg_outside_backs_experience_home", "lineup_avg_outside_backs_experience_away",
    "lineup_avg_interchange_experience_home", "lineup_avg_interchange_experience_away",
    "lineup_avg_named_margin_rating_home", "lineup_avg_named_margin_rating_away",
    "lineup_avg_spine_margin_rating_home", "lineup_avg_spine_margin_rating_away",
    "lineup_avg_halves_margin_rating_home", "lineup_avg_halves_margin_rating_away",
    "lineup_avg_middles_margin_rating_home", "lineup_avg_middles_margin_rating_away",
    "lineup_avg_edges_margin_rating_home", "lineup_avg_edges_margin_rating_away",
    "lineup_avg_outside_backs_margin_rating_home", "lineup_avg_outside_backs_margin_rating_away",
    "lineup_avg_interchange_margin_rating_home", "lineup_avg_interchange_margin_rating_away",
    "lineup_debutant_count_home", "lineup_debutant_count_away",
    "lineup_named_cohesion_home", "lineup_named_cohesion_away",
    "lineup_spine_cohesion_home", "lineup_spine_cohesion_away",
    "lineup_halves_pair_cohesion_home", "lineup_halves_pair_cohesion_away",
    "lineup_recent_named_stability_home", "lineup_recent_named_stability_away",
    "lineup_recent_spine_stability_home", "lineup_recent_spine_stability_away",
    "lineup_snapshot_count_home", "lineup_snapshot_count_away",
    "lineup_snapshot_window_hours_home", "lineup_snapshot_window_hours_away",
    "lineup_named_change_count_home", "lineup_named_change_count_away",
    "lineup_named_change_rate_home", "lineup_named_change_rate_away",
    "lineup_spine_change_count_home", "lineup_spine_change_count_away",
    "lineup_spine_change_rate_home", "lineup_spine_change_rate_away",
    "lineup_expected_named_count_home", "lineup_expected_named_count_away",
    "lineup_expected_interchange_count_home", "lineup_expected_interchange_count_away",
    "lineup_expected_spine_count_home", "lineup_expected_spine_count_away",
    "lineup_selection_uncertainty_home", "lineup_selection_uncertainty_away",
    "lineup_named_count_delta", "lineup_interchange_count_delta",
    "lineup_reserve_count_delta", "lineup_spine_count_delta",
    "lineup_spine_complete_delta", "lineup_bench_hooker_count_delta",
    "lineup_bench_spine_cover_count_delta",
    "lineup_source_age_hours_delta", "lineup_retained_ratio_delta",
    "lineup_starters_retained_ratio_delta", "lineup_spine_retained_ratio_delta",
    "lineup_spine_same_as_prev_delta", "lineup_halves_pair_same_as_prev_delta",
    "lineup_avg_named_experience_delta", "lineup_avg_spine_experience_delta",
    "lineup_avg_halves_experience_delta", "lineup_avg_middles_experience_delta",
    "lineup_avg_edges_experience_delta", "lineup_avg_outside_backs_experience_delta",
    "lineup_avg_interchange_experience_delta", "lineup_avg_named_margin_rating_delta",
    "lineup_avg_spine_margin_rating_delta", "lineup_avg_halves_margin_rating_delta",
    "lineup_avg_middles_margin_rating_delta", "lineup_avg_edges_margin_rating_delta",
    "lineup_avg_outside_backs_margin_rating_delta", "lineup_avg_interchange_margin_rating_delta",
    "lineup_debutant_count_delta", "lineup_named_cohesion_delta",
    "lineup_spine_cohesion_delta", "lineup_halves_pair_cohesion_delta",
    "lineup_recent_named_stability_delta", "lineup_recent_spine_stability_delta",
    "lineup_snapshot_count_delta", "lineup_snapshot_window_hours_delta",
    "lineup_named_change_count_delta", "lineup_named_change_rate_delta",
    "lineup_spine_change_count_delta", "lineup_spine_change_rate_delta",
    "lineup_expected_named_count_delta", "lineup_expected_interchange_count_delta",
    "lineup_expected_spine_count_delta", "lineup_selection_uncertainty_delta",
    "lineup_features_missing",
]

# Tier-A baseline features are computed in Python before training/inference.
predictors += [
    "baseline_mu_home",
    "baseline_mu_away",
    "baseline_draw_prob",
    "baseline_home_win_prob_conditional",
]

# Match-context families from the nrl.com ingestion (pipeline/common/nrl_data
# features + lineups player form). Built in Python and merged on game_id;
# form_ prefix obeys the FOOTY_TIPPER_INCLUDE_MATCH_FORM toggle.
_form_stats = [
    "possession_pct",
    "completion_rate",
    "all_run_metres",
    "post_contact_metres",
    "line_breaks",
    "tackle_breaks",
    "offloads",
    "kicking_metres",
    "effective_tackle_pct",
    "missed_tackles",
    "errors",
    "penalties_conceded",
]
match_form_predictors = [
    f"form_{stat}_{suffix}"
    for stat in _form_stats
    for suffix in ("home", "away", "delta")
] + [
    "form_features_missing_home",
    "form_features_missing_away",
]
predictors += match_form_predictors

predictors += [
    "referee_name",
    "ref_games_officiated",
    "ref_penalty_rate_ewma",
    "ref_sin_bin_rate_ewma",
    "ref_missing",
]

predictors += [
    "wx_temp",
    "wx_rain_3h",
    "wx_rain_24h",
    "wx_wind",
    "wx_humidity",
    "wx_wet",
    "ground_condition",
    "weather_missing",
]

predictors += [
    "travel_km_home",
    "travel_km_away",
    "travel_km_delta",
    "tz_shift_home",
    "tz_shift_away",
    "travel_missing",
]

_player_form_stats = ["fantasy", "run_metres", "tackles", "errors", "involvements"]
predictors += [
    f"lineup_form_{stat}_{side}"
    for stat in _player_form_stats
    for side in ("home", "away")
]
predictors += [
    "lineup_form_coverage_home",
    "lineup_form_coverage_away",
    "lineup_form_missing_home",
    "lineup_form_missing_away",
    "lineup_spine_form_fantasy_home",
    "lineup_spine_form_fantasy_away",
    "lineup_form_fantasy_delta",
    "lineup_spine_form_fantasy_delta",
]

# Predictors that should be treated as categorical if missing from the source
# data and created as fallback columns.
categorical_predictors = {
    "round_name",
    "game_state_name",
    "start_time",
    "start_time_utc",
    "venue_name",
    "city",
    "broadcast_channel1",
    "broadcast_channel2",
    "broadcast_channel3",
    "team_home",
    "team_away",
    "game_day",
    "referee_name",
    "ground_condition",
}

include_match_form = os.getenv(
    "FOOTY_TIPPER_INCLUDE_MATCH_FORM", "true"
).strip().lower() in {"1", "true", "yes", "y"}


def filter_predictors(
    include_performance=True,
    predictor_list=predictors,
    include_form=None,
):
    if include_form is None:
        include_form = include_match_form
    filtered = list(predictor_list)
    if not include_performance:
        filtered = [p for p in filtered if "_performance" not in p]
    if not include_form:
        # exact family membership; a bare startswith("form_") would also
        # drop the legacy R feature form_delta
        form_family = set(match_form_predictors)
        filtered = [p for p in filtered if p not in form_family]
    return filtered


sparse_feature_whitelist = {
    "state_of_origin",
    "post_origin",
    "performance_home_missing",
    "performance_away_missing",
    "performance_features_missing",
    "form_features_missing_home",
    "form_features_missing_away",
    "ref_missing",
    "wx_wet",
    "weather_missing",
    "tz_shift_home",
    "tz_shift_away",
    "travel_missing",
    "lineup_form_missing_home",
    "lineup_form_missing_away",
    "lineup_form_coverage_home",
    "lineup_form_coverage_away",
    "baseline_mu_home",
    "baseline_mu_away",
    "baseline_draw_prob",
    "baseline_home_win_prob_conditional",
    "lineup_data_available_home",
    "lineup_data_available_away",
    "lineup_named_count_home",
    "lineup_named_count_away",
    "lineup_spine_count_home",
    "lineup_spine_count_away",
    "lineup_spine_complete_home",
    "lineup_spine_complete_away",
    "lineup_bench_spine_cover_count_home",
    "lineup_bench_spine_cover_count_away",
    "lineup_retained_ratio_home",
    "lineup_retained_ratio_away",
    "lineup_starters_retained_ratio_home",
    "lineup_starters_retained_ratio_away",
    "lineup_spine_retained_ratio_home",
    "lineup_spine_retained_ratio_away",
    "lineup_spine_same_as_prev_home",
    "lineup_spine_same_as_prev_away",
    "lineup_halves_pair_same_as_prev_home",
    "lineup_halves_pair_same_as_prev_away",
    "lineup_avg_named_experience_home",
    "lineup_avg_named_experience_away",
    "lineup_avg_spine_experience_home",
    "lineup_avg_spine_experience_away",
    "lineup_avg_halves_experience_home",
    "lineup_avg_halves_experience_away",
    "lineup_avg_middles_experience_home",
    "lineup_avg_middles_experience_away",
    "lineup_avg_edges_experience_home",
    "lineup_avg_edges_experience_away",
    "lineup_avg_outside_backs_experience_home",
    "lineup_avg_outside_backs_experience_away",
    "lineup_avg_interchange_experience_home",
    "lineup_avg_interchange_experience_away",
    "lineup_avg_named_margin_rating_home",
    "lineup_avg_named_margin_rating_away",
    "lineup_avg_spine_margin_rating_home",
    "lineup_avg_spine_margin_rating_away",
    "lineup_avg_halves_margin_rating_home",
    "lineup_avg_halves_margin_rating_away",
    "lineup_avg_middles_margin_rating_home",
    "lineup_avg_middles_margin_rating_away",
    "lineup_avg_edges_margin_rating_home",
    "lineup_avg_edges_margin_rating_away",
    "lineup_avg_outside_backs_margin_rating_home",
    "lineup_avg_outside_backs_margin_rating_away",
    "lineup_avg_interchange_margin_rating_home",
    "lineup_avg_interchange_margin_rating_away",
    "lineup_debutant_count_home",
    "lineup_debutant_count_away",
    "lineup_named_cohesion_home",
    "lineup_named_cohesion_away",
    "lineup_spine_cohesion_home",
    "lineup_spine_cohesion_away",
    "lineup_halves_pair_cohesion_home",
    "lineup_halves_pair_cohesion_away",
    "lineup_recent_named_stability_home",
    "lineup_recent_named_stability_away",
    "lineup_recent_spine_stability_home",
    "lineup_recent_spine_stability_away",
    "lineup_snapshot_count_home",
    "lineup_snapshot_count_away",
    "lineup_snapshot_window_hours_home",
    "lineup_snapshot_window_hours_away",
    "lineup_named_change_count_home",
    "lineup_named_change_count_away",
    "lineup_named_change_rate_home",
    "lineup_named_change_rate_away",
    "lineup_spine_change_count_home",
    "lineup_spine_change_count_away",
    "lineup_spine_change_rate_home",
    "lineup_spine_change_rate_away",
    "lineup_expected_named_count_home",
    "lineup_expected_named_count_away",
    "lineup_expected_spine_count_home",
    "lineup_expected_spine_count_away",
    "lineup_selection_uncertainty_home",
    "lineup_selection_uncertainty_away",
    "lineup_named_count_delta",
    "lineup_spine_count_delta",
    "lineup_bench_spine_cover_count_delta",
    "lineup_retained_ratio_delta",
    "lineup_starters_retained_ratio_delta",
    "lineup_spine_retained_ratio_delta",
    "lineup_spine_same_as_prev_delta",
    "lineup_halves_pair_same_as_prev_delta",
    "lineup_avg_named_experience_delta",
    "lineup_avg_spine_experience_delta",
    "lineup_avg_halves_experience_delta",
    "lineup_avg_middles_experience_delta",
    "lineup_avg_edges_experience_delta",
    "lineup_avg_outside_backs_experience_delta",
    "lineup_avg_interchange_experience_delta",
    "lineup_avg_named_margin_rating_delta",
    "lineup_avg_spine_margin_rating_delta",
    "lineup_avg_halves_margin_rating_delta",
    "lineup_avg_middles_margin_rating_delta",
    "lineup_avg_edges_margin_rating_delta",
    "lineup_avg_outside_backs_margin_rating_delta",
    "lineup_avg_interchange_margin_rating_delta",
    "lineup_debutant_count_delta",
    "lineup_named_cohesion_delta",
    "lineup_spine_cohesion_delta",
    "lineup_halves_pair_cohesion_delta",
    "lineup_recent_named_stability_delta",
    "lineup_recent_spine_stability_delta",
    "lineup_snapshot_count_delta",
    "lineup_snapshot_window_hours_delta",
    "lineup_named_change_count_delta",
    "lineup_named_change_rate_delta",
    "lineup_spine_change_count_delta",
    "lineup_spine_change_rate_delta",
    "lineup_expected_named_count_delta",
    "lineup_expected_spine_count_delta",
    "lineup_selection_uncertainty_delta",
    "lineup_features_missing",
}


def prune_sparse_predictors(
    df: pd.DataFrame,
    predictor_list,
    min_support: int = sparse_min_support,
    whitelist=None,
):
    """Drop unstable predictors with very low historical support."""
    whitelist = set(whitelist or sparse_feature_whitelist)
    kept, dropped = [], []

    for col in predictor_list:
        if col in whitelist:
            kept.append(col)
            continue
        if col not in df.columns:
            dropped.append(col)
            continue

        series = df[col]
        non_missing = int(series.notna().sum())
        if non_missing < min_support:
            dropped.append(col)
            continue

        if pd.api.types.is_numeric_dtype(series):
            numeric_series = pd.to_numeric(series, errors="coerce")
            non_zero = int((numeric_series.fillna(0) != 0).sum())
            if non_zero < min_support:
                dropped.append(col)
                continue

        kept.append(col)

    if dropped:
        print(
            f"Pruned {len(dropped)} sparse predictors (min_support={min_support}): "
            + ", ".join(sorted(dropped))
        )
    print(f"Predictor count after sparse pruning: {len(kept)}")
    return kept


def align_predictor_columns(df: pd.DataFrame, predictor_list, categorical_cols=None) -> pd.DataFrame:
    """Ensure all expected predictor columns exist for training/inference."""
    aligned = df.copy()
    categorical_cols = set(categorical_cols or categorical_predictors)
    missing_predictors = [p for p in predictor_list if p not in aligned.columns]

    for col in missing_predictors:
        aligned[col] = "Unknown" if col in categorical_cols else 0.0

    if missing_predictors:
        print(
            "Missing predictors were added with fallback values: "
            + ", ".join(sorted(missing_predictors))
        )

    return aligned
    
