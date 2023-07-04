CREATE TABLE IF NOT EXISTS predictions_table (
    game_id INTEGER PRIMARY KEY,
    home_team_result TEXT,
    home_team_win_prob REAL,
    home_team_lose_prob REAL
)
