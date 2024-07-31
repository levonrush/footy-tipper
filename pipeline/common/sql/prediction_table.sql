WITH min_round_id AS (
    SELECT MIN(round_id) AS round_id
    FROM footy_tipping_data
    WHERE game_state_name = 'Pre Game'
)

SELECT CAST(ft.game_id AS INTEGER) AS game_id
    , p.home_team_result
    , ft.team_home
    , CAST(ft.position_home_ladder AS INTEGER) AS position_home
    , ft.team_head_to_head_odds_home
    , ft.team_away
    , CAST(ft.position_away_ladder AS INTEGER) AS position_away
    , ft.team_head_to_head_odds_away
    , p.home_team_win_prob
    , p.home_team_lose_prob
    , CAST(ft.round_id AS INTEGER) AS round_id
    , CAST(ft.competition_year AS INTEGER) AS competition_year
    , ft.round_name
FROM predictions_table p
LEFT JOIN footy_tipping_data ft ON p.game_id = ft.game_id
WHERE ft.game_state_name = 'Pre Game'
AND round_id = (SELECT * FROM min_round_id)
