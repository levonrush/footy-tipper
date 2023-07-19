INSERT INTO predictions_table (
    game_id,  
    home_team_result, 
    home_team_win_prob, 
    home_team_lose_prob
) VALUES (?, ?, ?, ?)
ON CONFLICT(game_id) DO UPDATE SET
    home_team_result= excluded.home_team_result,
    home_team_win_prob = excluded.home_team_win_prob,
    home_team_lose_prob = excluded.home_team_lose_prob
