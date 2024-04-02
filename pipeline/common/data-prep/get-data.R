# A function to extract game results from XML data
get_game_results <- function(fixtures_xml){
  
  # Extract relevant information from each 'gameFixture' node in the XML data
  game_results_long <- fixtures_xml %>% xml_find_all(".//gameFixture") %>%
    map_df(~{
      bind_cols(
        # Extract gameId and various team attributes for each game
        gameId =  xml_attr(.x, "gameId"),
        team = xml_find_all(.x, ".//teams/team") %>% xml_attr("team"),
        teamFinalScore = xml_find_all(.x, ".//teams/team") %>% xml_attr("teamFinalScore"),
        isHomeTeam = xml_find_all(.x, ".//teams/team") %>% xml_attr("isHomeTeam"),
        teamHeadToHeadOdds = xml_find_all(.x, ".//teams/team") %>% xml_attr("teamHeadToHeadOdds"),
        teamLineOdds = xml_find_all(.x, ".//teams/team") %>% xml_attr("teamLineOdds"),
        teamLineAmount = xml_find_all(.x, ".//teams/team") %>% xml_attr("teamLineAmount")
      ) 
    }) 
  
  # Split the results data into two sets based on 'isHomeTeam' 
  split_game_results <- game_results_long %>%
    group_by(isHomeTeam) %>%
    group_split()
  
  # Remove the 'isHomeTeam' column from the away and home results
  away_game_results <- split_game_results[[1]] %>%
    select(-isHomeTeam)
  home_game_results <- split_game_results[[2]] %>%
    select(-isHomeTeam)
  
  # Join the home and away results data on 'gameId'
  game_results <- home_game_results %>% 
    inner_join(away_game_results, by = "gameId", suffix = c('_home', '_away'))
  
  return(game_results)
}

# A function to extract fixture information from XML data
get_fixture_info <- function(fixtures_xml){
  
  # Extract relevant information from each 'roundFixtures' node in the XML data
  fixture_info <- fixtures_xml %>% xml_find_all(".//roundFixtures") %>%
    map_df(~{
      bind_cols(
        # Extract gameId and various other game attributes for each game
        gameId = xml_find_all(.x, ".//gameFixture") %>% xml_attr("gameId"),
        roundId = xml_attr(.x, "roundId"),
        roundName = xml_attr(.x, "roundName"),
        gameNumber = xml_find_all(.x, ".//gameFixture") %>% xml_attr("gameNumber"),
        gameStateName = xml_find_all(.x, ".//gameFixture") %>% xml_attr("gameStateName"),
        startTime = xml_find_all(.x, ".//gameFixture") %>% xml_attr("startTime"),
        startTimeUTC = xml_find_all(.x, ".//gameFixture") %>% xml_attr("startTimeUTC"),
        venueName = xml_find_all(.x, ".//gameFixture") %>% xml_attr("venueName"),
        city = xml_find_all(.x, ".//gameFixture") %>% xml_attr("city"),
        crowd = xml_find_all(.x, ".//gameFixture") %>% xml_attr("crowd"),
        broadcastChannel1 = xml_find_all(.x, ".//gameFixture") %>% xml_attr("broadcastChannel"),
        broadcastChannel2 = xml_find_all(.x, ".//gameFixture") %>% xml_attr("broadcastChannel2"),
        broadcastChannel3 = xml_find_all(.x, ".//gameFixture") %>% xml_attr("broadcastChannel3")
      ) 
    })

  return(fixture_info)
}

# A function to extract yearly ladder data
get_year_ladder <- function(password, year){

  password <- Sys.getenv("PASSWORD")
  base_url <- Sys.getenv("BASE_URL")
  ladder_ext <- Sys.getenv("NRL_ROUND_LADDER_EXTENTION")
  
  year_ladder <- vector(mode = "list")
  
  for (round in 1:40){
    
    # Try to read XML data for a specific round, if it fails return NA
    ladder_xml <- tryCatch(read_xml(paste0("http://", password, base_url, ladder_ext, year, "/", round)),
                           error = function(e){NA})
    
    if (is.na(ladder_xml)) break
    
    # Extract relevant information from each 'ladderposition' node in the XML data
    year_ladder[[round]] <- ladder_xml %>% xml_find_all(".//ladderposition") %>%
      map_df(~{
        bind_cols(
          # Extract team and various other attributes for each ladder position
          position = xml_attr(.x, "position"),
          team = xml_attr(.x, "teamName"),
          wins = xml_attr(.x, "wins"),
          draws = xml_attr(.x, "draws"),
          losses = xml_attr(.x, "losses"),
          byes = xml_attr(.x, "byes"),
          competition_points = xml_attr(.x, "competitionPoints"),
          pointsFor = xml_attr(.x, "pointsFor"),
          pointsAgainst = xml_attr(.x, "pointsAgainst"),
          pointsDifference = xml_attr(.x, "pointsDifference"),
          homeWins = xml_attr(.x, "homeWins"),
          homeDraws = xml_attr(.x, "homeDraws"),
          homeLosses = xml_attr(.x, "homeLosses"),
          awayWins = xml_attr(.x, "awayWins"),
          awayDraws = xml_attr(.x, "awayDraws"),
          awayLosses = xml_attr(.x, "awayLosses"),
          recentForm = xml_attr(.x, "recentForm"),
          seasonForm = xml_attr(.x, "seasonForm"),
          triesFor = xml_attr(.x, "triesFor"),
          triesConceded = xml_attr(.x, "triesConceded"),
          goalsFor = xml_attr(.x, "goalsFor"),
          goalsConceded = xml_attr(.x, "goalsConceded"),
          fieldGoalsFor = xml_attr(.x, "fieldGoalsFor"),
          fieldGoalsConceded = xml_attr(.x, "fieldGoalsConceded"),
          playersUsed = xml_attr(.x, "playersUsed"),
          averageWinningMargin = xml_attr(.x, "averageWinningMargin"),
          averageLosingMargin = xml_attr(.x, "averageLosingMargin"),
          closeGames = xml_attr(.x, "closeGames"),
          dayRecord = xml_attr(.x, "dayRecord"),
          nightRecord = xml_attr(.x, "nightRecord"),
          currentStreak = xml_attr(.x, "currentStreak")
        )
      }) %>%
      # Add a column to indicate the round and competition year
      mutate(round_id = round,
             competition_year = year)
    
  }
  
  year_ladder <- bind_rows(year_ladder)
  
  return(year_ladder)
  
}

# A function to extract all ladder data within a specific year span
get_ladders <- function(password, year_span){
  
  every_ladder <- vector(mode = "list")
  
  for (year in year_span){
    
    # Get the ladder data for each year and store it in the 'every_ladder' list
    table <- get_year_ladder(password, year)
    every_ladder[[year]] <- table
    
  }
  
  ladder_df <- bind_rows(every_ladder)
  
  return(ladder_df)
  
}

# The main function to extract all data
get_data <- function(year_span){

  password <- Sys.getenv("PASSWORD")
  base_url <- Sys.getenv("BASE_URL")
  fixtures_ext <- Sys.getenv("NRL_FIXTURES_EXTENTION")

  print("Get Data: Fetching fixture data...")
  
  # Initialize a list to store the results for each fixture
  all_fixtures <- vector(mode = "list", length = length(year_span))
  
  for (y in 1:length(year_span)){
    
    # Get the XML data for the fixtures of a specific year
    fixtures_xml <- read_xml(paste0("http://", password, base_url, fixtures_ext, year_span[y]))
    
    # Extract fixture information and game results
    fixture_info <- get_fixture_info(fixtures_xml)
    game_results <- get_game_results(fixtures_xml)
    
    # Merge the fixture information and game results, and add a column to indicate the competition year
    all_fixtures[[y]] <- fixture_info %>% 
      inner_join(game_results, by = c('gameId')) %>%
      mutate(competition_year = year_span[y])
    
  }
  
  fixtures_df <- bind_rows(all_fixtures) %>% clean_names() %>% type_convert()
  
  # Get the ladder data for each year, shift data to the previous round, and clean names and types
  ladders_df <- get_ladders(password, year_span) %>% clean_names() %>% type_convert() %>%
    arrange(competition_year, round_id) %>%
    group_by(team, competition_year) %>%
    mutate_at(vars(-team, -round_id, -competition_year), lag) %>%
    ungroup()
  
  print("Get Data: Fetching ladder data...")
  # Perform some data cleaning and feature engineering
  ladders_df <- ladders_df %>%
    select(-c(day_record, night_record, current_streak)) %>%
    mutate(recent_form = str_count(recent_form, "W") - str_count(recent_form, "L"),
           season_form = str_count(season_form, "W") - str_count(season_form, "L")) %>%
    mutate_at(vars(-team, -round_id, -competition_year), list(~ replace_na(., 0))) %>%
    mutate(win_rate = wins/round_id,
           draw_rate = draws/round_id,
           loss_rate = losses/round_id,
           competition_point_rate = competition_points/round_id,
           avg_points_for = points_for/round_id,
           avg_points_against = points_against/round_id,
           avg_points_difference = points_difference/round_id,
           home_win_rate = home_wins/round_id,
           home_draw_rate = home_draws/round_id,
           home_loss_rate = home_losses/round_id,
           away_win_rate = away_wins/round_id,
           away_draw_rate = away_draws/round_id,
           away_loss_rate = away_losses/round_id,
           avg_tries_for = tries_for/round_id,
           avg_tries_conceded = tries_conceded/round_id,
           avg_goals_for = goals_for/round_id,
           avg_goals_conceded = goals_conceded/round_id,
           close_game_rate = close_games/round_id)
  
  # Merge the fixture and ladder data, and return the final dataframe
  print("Get Data: Merging fixture and ladder data...")
  footy_tipper_df <- fixtures_df %>%
    left_join(ladders_df, by = c("competition_year", "round_id", "team_home" = "team")) %>%
    left_join(ladders_df, by = c("competition_year", "round_id", "team_away" = "team"), suffix = c("_home", "_away"))
  
  return(footy_tipper_df)
  
}
