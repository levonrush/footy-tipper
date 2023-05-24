get_game_results <- function(fixtures_xml){
  
  game_results_long <- fixtures_xml %>% xml_find_all(".//gameFixture") %>%
    map_df(~{
      bind_cols(
        gameId =  xml_attr(.x, "gameId"),
        team = xml_find_all(.x, ".//teams/team") %>% xml_attr("team"),
        teamFinalScore = xml_find_all(.x, ".//teams/team") %>% xml_attr("teamFinalScore"),
        isHomeTeam = xml_find_all(.x, ".//teams/team") %>% xml_attr("isHomeTeam"),
        teamHeadToHeadOdds = xml_find_all(.x, ".//teams/team") %>% xml_attr("teamHeadToHeadOdds"),
        teamLineOdds = xml_find_all(.x, ".//teams/team") %>% xml_attr("teamLineOdds"),
        teamLineAmount = xml_find_all(.x, ".//teams/team") %>% xml_attr("teamLineAmount")
        
      ) 
    }) 
  
  split_game_results <- game_results_long %>%
    group_by(isHomeTeam) %>%
    group_split()
  
  away_game_results <- split_game_results[[1]] %>%
    select(-isHomeTeam)
  
  home_game_results <- split_game_results[[2]] %>%
    select(-isHomeTeam)
  
  game_results <- home_game_results %>% 
    inner_join(away_game_results, by = "gameId", suffix = c('_home', '_away'))
  
  return(game_results)
  
}

get_fixture_info <- function(fixtures_xml){
  
  fixture_info <- fixtures_xml %>% xml_find_all(".//roundFixtures") %>%
    map_df(~{
      bind_cols(
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

##### then get the historic ladder placings

get_year_ladder <- function(password, year){
  
  year_ladder <- vector(mode = "list")
  
  for (round in 1:40){
    
    ladder_xml <- tryCatch(read_xml(paste0("http://", password, BASE_URL, NRL_ROUND_LADDER_EXTENTION, year, "/", round)),
                           error = function(e){NA})
    
    if (is.na(ladder_xml)) break
    
    year_ladder[[round]] <- ladder_xml %>% xml_find_all(".//ladderposition") %>%
      map_df(~{
        bind_cols(
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
      mutate(round_id = round,
             competition_year = year)
    
  }
  
  year_ladder <- bind_rows(year_ladder)
  
  return(year_ladder)
  
}

get_ladders <- function(password, year_span){
  
  every_ladder <- vector(mode = "list")
  
  for (year in year_span){
    
    table <- get_year_ladder(password, year)
    every_ladder[[year]] <- table
    
  }
  
  ladder_df <- bind_rows(every_ladder)
  
  return(ladder_df)
  
}

##### finally put it all together

get_data <- function(year_span){
  
  # Get the password
  if(interactive()){
    password <- rstudioapi::askForPassword("Enter your password")
  } else {
    password <- Sys.getenv("PASSWORD")
  }
  
  # get the results for each fixture
  all_fixtures <- vector(mode = "list", length = length(year_span))
  
  for (y in 1:length(year_span)){
    
    # get that year's xml file
    fixtures_xml <- read_xml(paste0("http://", password, BASE_URL, NRL_FIXTURES_EXTENTION, year_span[y]))
    
    # get fixture information
    fixture_info <- get_fixture_info(fixtures_xml)
    
    # get results from the game
    game_results <- get_game_results(fixtures_xml)
    
    # join it together and jot down the year
    all_fixtures[[y]] <- fixture_info %>% 
      inner_join(game_results, by = c('gameId')) %>%
      mutate(competition_year = year_span[y])
    
  }
  
  fixtures_df <- bind_rows(all_fixtures) %>% clean_names() %>% type_convert()
  
  # get all the associated ladders - need to move them back a step to be pre game ladder stats
  ladders_df <- get_ladders(password, year_span) %>% clean_names() %>% type_convert() %>%
    arrange(competition_year, round_id) %>%
    group_by(team, competition_year) %>%
    mutate_at(vars(-team, -round_id, -competition_year), lag) %>%
    ungroup()
  
  ### ladder cleaning and engineering engineering here
  ladders_df <- ladders_df %>%
    # i'll need to work out the variables for these later
    select(-c(day_record, night_record, current_streak)) %>%
    # convert form things to numbers first here
    mutate(recent_form = str_count(recent_form, coll("W")) - str_count(recent_form, coll("L")),
           season_form = str_count(season_form, coll("W")) - str_count(season_form, coll("L"))) %>%
    dumb_impute(0, "nothing") %>%
    # do some engineering off that's there
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
           away_loss_rate = away_losses/round_id,
           avg_tries_for = tries_for/round_id,
           avg_tries_conceded = tries_conceded/round_id,
           avg_goals_for = goals_for/round_id,
           avg_goals_conceded = goals_conceded/round_id,
           close_game_rate = close_games/round_id)
  
  # finally join on the ladder data for that round - remember i have to do this for both home and away teams
  footy_tipper_df <- fixtures_df %>%
    left_join(ladders_df, by = c("competition_year", "round_id", "team_home" = "team")) %>%
    left_join(ladders_df, by = c("competition_year", "round_id", "team_away" = "team"), suffix = c("_home", "_away"))
    
  # return it to the env
  return(footy_tipper_df)

}
