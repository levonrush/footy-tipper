CREATE TABLE IF NOT EXISTS predictions_table (
    game_id INTEGER PRIMARY KEY,
    home_team_result TEXT,
    home_team_win_prob REAL,
    home_team_lose_prob REAL,
    draw_prob REAL,
    bayes_factor REAL,
    evidence_strength TEXT,
    predicted_home_score INTEGER,
    predicted_away_score INTEGER,
    predicted_margin INTEGER
)
