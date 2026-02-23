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

SELECT CAST(game_id AS INTEGER) AS game_id
    , CAST(round_id AS INTEGER) AS round_id
    , CAST(competition_year AS INTEGER) AS competition_year
    , round_name
    , team_home
    , team_away
    , team_head_to_head_odds_home
    , team_head_to_head_odds_away
FROM footy_tipping_data
WHERE game_state_name = 'Pre Game'
AND competition_year = (SELECT competition_year FROM latest_year)
AND round_id >= (SELECT round_id FROM min_round_id)
ORDER BY round_id, game_id
