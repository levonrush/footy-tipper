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
  
  # Split into explicit home/away rows to avoid relying on group order.
  home_game_results <- game_results_long %>%
    filter(tolower(isHomeTeam) == "true") %>%
    select(-isHomeTeam)
  away_game_results <- game_results_long %>%
    filter(tolower(isHomeTeam) == "false") %>%
    select(-isHomeTeam)
  
  if (nrow(home_game_results) == 0 || nrow(away_game_results) == 0) {
    stop("Get Data: Could not split fixture results into home and away teams.")
  }
  
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

get_year_performance <- function(password, year){
  
  base_url <- Sys.getenv("BASE_URL")
  performance_ext <- Sys.getenv("NRL_PERFORMANCE_EXTENTION")
  
  year_performance <- vector(mode = "list")  # Initialize the list to store performance data for each round
  
  for (round in 1:40){
    
    performance_xml <- tryCatch(read_xml(paste0("http://", password, base_url, performance_ext, year, "/", round)),
                                error = function(e){NA})
    
    
    if (is.na(performance_xml)) break
    
    seasonId <- xml_attr(performance_xml, "seasonId")
    competitionId <- xml_attr(performance_xml, "competitionId")
    competitionName <- xml_attr(performance_xml, "competitionName")
    roundNumber <- xml_attr(performance_xml, "roundNumber")
    
    stats <- xml_find_all(performance_xml, "//leaderboardStat")
    
    flattened_data <- stats %>%
      map_df(function(stat) {
        entries <- xml_find_all(stat, "leaderboardEntry")
        data.frame(
          seasonId = seasonId,
          competitionId = competitionId,
          competitionName = competitionName,
          roundNumber = roundNumber,
          statID = xml_attr(stat, "ID"),
          statName = xml_attr(stat, "statName"),
          TeamID = xml_attr(entries, "TeamID"),
          TeamName = xml_attr(entries, "TeamName"),
          TeamAbbrev = xml_attr(entries, "TeamAbbrev"),
          Value = readr::parse_number(xml_attr(entries, "Value")),
          Rank = as.integer(xml_attr(entries, "Rank")),    # Ensure integer values are correctly typed
          Appearances = as.integer(xml_attr(entries, "Appearances")),  # Ensure integer values are correctly typed
          stringsAsFactors = FALSE
        )
      })

    if (nrow(flattened_data) == 0) {
      next
    }
    
    pivoted_data <- flattened_data %>%
      select(seasonId, roundNumber, TeamName, statName, Value) %>%
      # Some seasons/rounds can have duplicate stat/team rows; force to one numeric
      # value per team/stat so bind_rows does not receive list columns.
      pivot_wider(
        names_from = statName,
        values_from = Value,
        values_fn = list(Value = ~ {
          vals <- suppressWarnings(as.numeric(.x))
          vals <- vals[!is.na(vals)]
          if (length(vals) == 0) NA_real_ else dplyr::last(vals)
        }),
        values_fill = list(Value = NA_real_)
      )
    
    pivoted_data <- pivoted_data %>%
      mutate(
        across(-c(seasonId, roundNumber, TeamName), ~ suppressWarnings(as.numeric(.))),
        across(-c(seasonId, roundNumber, TeamName), ~ replace_na(., 0))
      )
    
    year_performance[[round]] <- pivoted_data %>%
      mutate(round_id = round,
             competition_year = year)
  }

  if (length(year_performance) == 0) {
    return(tibble())
  }

  year_performance <- bind_rows(year_performance)
  if (!("TeamName" %in% names(year_performance))) {
    return(tibble())
  }

  year_performance <- year_performance %>% # Combine all rounds' data for the year
    rename(team = TeamName) # Rename column to ensure consistency

  return(year_performance)
}

# A function to extract all ladder data within a specific year span
get_ladders <- function(password, year_span){
  
  every_ladder <- list()
  
  for (year in year_span){
    
    # Get the ladder data for each year and store it in the 'every_ladder' list
    table <- get_year_ladder(password, year)
    if (nrow(table) > 0){
      every_ladder[[as.character(year)]] <- table
    }
    
  }
  
  if (length(every_ladder) == 0){
    return(tibble())
  }
  
  ladder_df <- bind_rows(every_ladder)
  
  return(ladder_df)
  
}

# A function to extract all performance data within a specific year span
get_performance <- function(password, year_span){
  
  every_performance <- list()
  
  for (year in year_span){
    
    # Get the performance data for each year and store it in the 'every_performance' list
    table <- get_year_performance(password, year)
    if (nrow(table) > 0){
      every_performance[[as.character(year)]] <- table
    }
    
  }
  
  if (length(every_performance) == 0){
    return(tibble())
  }
  
  performance_df <- bind_rows(every_performance)
  
  return(performance_df)
  
}

# The main function to extract all data
get_data <- function(year_span, include_performance = TRUE){
  
  password <- Sys.getenv("PASSWORD")
  base_url <- Sys.getenv("BASE_URL")
  fixtures_ext <- Sys.getenv("NRL_FIXTURES_EXTENTION")
  
  print("Get Data: Fetching fixture data...")
  
  # Fetch fixture data for each year. Missing future seasons are skipped.
  all_fixtures <- list()
  available_years <- c()
  for (year in year_span){
    fixtures_xml <- tryCatch(
      read_xml(paste0("http://", password, base_url, fixtures_ext, year)),
      error = function(e){
        message(paste0("Get Data: No fixture feed available for ", year, " (", conditionMessage(e), "). Skipping year."))
        NULL
      }
    )
    if (is.null(fixtures_xml)) next
    
    fixture_info <- get_fixture_info(fixtures_xml)
    game_results <- get_game_results(fixtures_xml)
    
    all_fixtures[[as.character(year)]] <- fixture_info %>% 
      inner_join(game_results, by = "gameId") %>% 
      mutate(competition_year = year)
    available_years <- c(available_years, year)
  }
  
  if (length(all_fixtures) == 0){
    stop("Get Data: No fixture data available for the configured year span.")
  }
  
  fixtures_df <- bind_rows(all_fixtures) %>% clean_names() %>% type_convert()
  available_years <- sort(unique(available_years))
  
  print("Get Data: Fetching ladder data...")
  ladders_raw <- get_ladders(password, available_years)
  if (nrow(ladders_raw) == 0){
    stop("Get Data: No ladder data available for available fixture years.")
  }
  
  ladders_df <- ladders_raw %>% 
    clean_names() %>% 
    type_convert() %>%
    arrange(competition_year, round_id) %>%
    group_by(team, competition_year) %>%
    mutate_at(vars(-team, -round_id, -competition_year), lag) %>%
    ungroup() %>%
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
  
  print("Get Data: Merging fixture and ladder data...")
  footy_tipper_df <- fixtures_df %>%
    left_join(ladders_df, by = c("competition_year", "round_id", "team_home" = "team")) %>%
    left_join(ladders_df, by = c("competition_year", "round_id", "team_away" = "team"), 
              suffix = c("_home_ladder", "_away_ladder"))
  
  if(include_performance){
    print("Get Data: Fetching performance data...")
    performance_df <- get_performance(password, available_years)
    
    if (nrow(performance_df) == 0){
      stop("Get Data: include_performance is TRUE but no performance data is available. Set FOOTY_TIPPER_INCLUDE_PERFORMANCE=false to run without these features.")
    }
    
    performance_df <- performance_df %>%
      clean_names() %>%
      type_convert()
    
    # Ensure all necessary columns are numeric
    numeric_cols <- names(performance_df)[sapply(performance_df, is.numeric)]
    performance_df[numeric_cols] <- lapply(performance_df[numeric_cols], as.numeric)
    
    performance_df <- performance_df %>%
      arrange(competition_year, round_id) %>%
      group_by(team, competition_year) %>%
      mutate_at(vars(-team, -round_id, -competition_year), lag) %>%
      ungroup() %>%
      mutate_at(vars(-team, -round_id, -competition_year), list(~ replace_na(., 0)))
    
    print("Get Data: Merging match and performance data...")
    footy_tipper_df <- footy_tipper_df %>%
      left_join(performance_df, by = c("competition_year", "round_id", "team_home" = "team")) %>%
      left_join(performance_df, by = c("competition_year", "round_id", "team_away" = "team"), 
                suffix = c("_home_performance", "_away_performance"))
  } else {
    print("Get Data: Skipping performance data merge for this run.")
  }
  
  return(footy_tipper_df)
}
