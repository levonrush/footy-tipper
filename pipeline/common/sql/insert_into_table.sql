INSERT INTO predictions_table (
    game_id,  
    home_team_result, 
    home_team_win_prob, 
    home_team_lose_prob,
    draw_prob,
    bayes_factor,
    evidence_strength,
    predicted_home_score,
    predicted_away_score,
    predicted_margin
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(game_id) DO UPDATE SET
    home_team_result= excluded.home_team_result,
    home_team_win_prob = excluded.home_team_win_prob,
    home_team_lose_prob = excluded.home_team_lose_prob,
    draw_prob = excluded.draw_prob,
    bayes_factor = excluded.bayes_factor,
    evidence_strength = excluded.evidence_strength,
    predicted_home_score = excluded.predicted_home_score,
    predicted_away_score = excluded.predicted_away_score,
    predicted_margin = excluded.predicted_margin
