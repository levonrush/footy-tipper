WITH latest_year AS (
    SELECT MAX(competition_year) AS competition_year
    FROM footy_tipping_data
    WHERE game_state_name = 'Pre Game'
),
min_round_id AS (
    SELECT MIN(round_id) AS round_id
    FROM footy_tipping_data
    WHERE game_state_name = 'Pre Game'
      AND competition_year = (SELECT competition_year FROM latest_year)
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
    , p.draw_prob
    , CAST(p.predicted_home_score AS INTEGER) AS predicted_home_score
    , CAST(p.predicted_away_score AS INTEGER) AS predicted_away_score
    , CAST(p.predicted_margin AS INTEGER) AS predicted_margin
    , p.bayes_factor
    , p.evidence_strength
    , CAST(ft.round_id AS INTEGER) AS round_id
    , CAST(ft.competition_year AS INTEGER) AS competition_year
    , ft.round_name
    -- Line and totals prices, so the finals email can value those markets too.
    -- `_sanitize_market_freshness` already masks these by their own freshness
    -- flags, independently of the head-to-head pair.
    , ft.team_line_amount_home
    , ft.team_line_odds_home
    , ft.team_line_odds_away
    , ft.total_line
    , ft.total_over_odds
    , ft.total_under_odds
    -- Kickoff ordering. game_id encodes round and game number, which is kickoff
    -- order in a regular round but bracket order in the finals, so the display
    -- sort needs the real start time rather than the identifier.
    , CAST(ft.start_time AS REAL) AS start_time
    , CAST(ft.game_number AS REAL) AS game_number
FROM predictions_table p
LEFT JOIN footy_tipping_data ft ON p.game_id = ft.game_id
WHERE ft.game_state_name = 'Pre Game'
AND ft.competition_year = (SELECT competition_year FROM latest_year)
AND ft.round_id = (SELECT round_id FROM min_round_id)
ORDER BY CAST(ft.start_time AS REAL), CAST(ft.game_number AS REAL), CAST(ft.game_id AS INTEGER)
