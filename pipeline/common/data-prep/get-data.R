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

sanitize_feed_error <- function(error_message, password) {
  safe_error <- as.character(error_message)
  if (!is.null(password) && nzchar(password)) {
    safe_error <- gsub(password, "<redacted>", safe_error, fixed = TRUE)
  }
  safe_error <- gsub("https?://[^[:space:]]+", "<feed-url>", safe_error)
  safe_error
}

read_feed_xml <- function(url, password, context_message = NULL, log_error = FALSE) {
  tryCatch(
    suppressWarnings(read_xml(url)),
    error = function(e) {
      if (isTRUE(log_error) && !is.null(context_message)) {
        safe_error <- sanitize_feed_error(conditionMessage(e), password)
        message(paste0(context_message, " (", safe_error, ")"))
      }
      NULL
    }
  )
}

# A function to extract yearly ladder data
get_year_ladder <- function(password, year){

  base_url <- Sys.getenv("BASE_URL")
  ladder_ext <- Sys.getenv("NRL_ROUND_LADDER_EXTENTION")
  
  year_ladder <- vector(mode = "list")
  
  for (round in 1:40){
    
    ladder_xml <- read_feed_xml(
      paste0("http://", password, base_url, ladder_ext, year, "/", round),
      password
    )
    
    if (is.null(ladder_xml)) break
    
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
    
    performance_xml <- read_feed_xml(
      paste0("http://", password, base_url, performance_ext, year, "/", round),
      password
    )
    
    
    if (is.null(performance_xml)) break
    
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

fetch_fixture_year <- function(password, year){
  base_url <- Sys.getenv("BASE_URL")
  fixtures_ext <- Sys.getenv("NRL_FIXTURES_EXTENTION")

  fixtures_xml <- read_feed_xml(
    paste0("http://", password, base_url, fixtures_ext, year),
    password,
    context_message = paste0("Get Data: No fixture feed available for ", year, "."),
    log_error = TRUE
  )
  if (is.null(fixtures_xml)) {
    return(NULL)
  }

  fixture_info <- get_fixture_info(fixtures_xml)
  game_results <- get_game_results(fixtures_xml)

  fixture_info %>%
    inner_join(game_results, by = "gameId") %>%
    mutate(competition_year = year) %>%
    clean_names() %>%
    type_convert()
}

fetch_ladder_year <- function(password, year){
  table <- get_year_ladder(password, year)
  if (nrow(table) == 0) {
    return(NULL)
  }

  table %>%
    clean_names() %>%
    type_convert()
}

fetch_performance_year <- function(password, year){
  table <- get_year_performance(password, year)
  if (nrow(table) == 0) {
    return(NULL)
  }

  table %>%
    clean_names() %>%
    type_convert()
}

# The main function to extract all data
get_data <- function(year_span, include_performance = TRUE, prep_mode = "full", db_path = NULL){
  
  password <- Sys.getenv("PASSWORD")
  if (is.null(db_path) || !nzchar(db_path)) {
    stop("Get Data: db_path is required for feed cache access.")
  }

  refresh_mode <- if (tolower(prep_mode) == "full") "full" else "smart"
  current_year <- as.integer(format(Sys.Date(), "%Y"))
  requested_years <- normalize_year_vector(year_span)

  con <- dbConnect(SQLite(), db_path)
  on.exit(dbDisconnect(con), add = TRUE)

  print(paste0("Get Data: Feed refresh mode = ", refresh_mode))
  print("Get Data: Fetching fixture data...")

  fixture_cache_table <- feed_cache_table_name("fixtures")
  cached_fixtures <- load_cached_feed(con, fixture_cache_table, requested_years)
  fixture_refresh_years <- resolve_fixture_refresh_years(
    requested_years,
    cached_fixtures$competition_year,
    current_year,
    refresh_mode = refresh_mode
  )
  print(paste0("Get Data: Fixture refresh years = ", format_year_vector(fixture_refresh_years)))
  refresh_feed_cache_years(
    con,
    fixture_cache_table,
    fixture_refresh_years,
    function(year) fetch_fixture_year(password, year),
    "fixture"
  )

  fixtures_df <- load_cached_feed(con, fixture_cache_table, requested_years)
  if (nrow(fixtures_df) == 0){
    stop("Get Data: No fixture data available for the configured year span.")
  }

  available_years <- normalize_year_vector(fixtures_df$competition_year)
  print(paste0("Get Data: Loaded fixture cache for seasons ", format_year_vector(available_years)))

  print("Get Data: Fetching ladder data...")
  ladder_cache_table <- feed_cache_table_name("ladders")
  cached_ladders <- load_cached_feed(con, ladder_cache_table, available_years)
  fixture_rounds <- year_max_round_lookup(fixtures_df)
  cached_ladder_rounds <- year_max_round_lookup(cached_ladders)
  ladder_refresh_years <- resolve_ladder_refresh_years(
    available_years,
    cached_ladders$competition_year,
    fixture_rounds,
    cached_ladder_rounds,
    current_year,
    refresh_mode = refresh_mode
  )
  print(paste0("Get Data: Ladder refresh years = ", format_year_vector(ladder_refresh_years)))
  refresh_feed_cache_years(
    con,
    ladder_cache_table,
    ladder_refresh_years,
    function(year) fetch_ladder_year(password, year),
    "ladder"
  )

  ladders_raw <- load_cached_feed(con, ladder_cache_table, available_years)
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
    performance_cache_table <- feed_cache_table_name("performance")
    cached_performance <- load_cached_feed(con, performance_cache_table, available_years)
    final_fixture_rounds <- year_max_round_lookup(fixtures_df, state_name = "Final")
    cached_performance_rounds <- year_max_round_lookup(cached_performance)
    performance_refresh_years <- resolve_performance_refresh_years(
      available_years,
      cached_performance$competition_year,
      final_fixture_rounds,
      cached_performance_rounds,
      current_year,
      refresh_mode = refresh_mode
    )
    print(paste0("Get Data: Performance refresh years = ", format_year_vector(performance_refresh_years)))
    refresh_feed_cache_years(
      con,
      performance_cache_table,
      performance_refresh_years,
      function(year) fetch_performance_year(password, year),
      "performance"
    )

    performance_df <- load_cached_feed(con, performance_cache_table, available_years)
    
    if (nrow(performance_df) == 0){
      stop("Get Data: include_performance is TRUE but no performance data is available. Set FOOTY_TIPPER_INCLUDE_PERFORMANCE=false to run without these features.")
    }
    
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
