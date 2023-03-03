library(tidyverse)
library(xml2)
library(janitor)
library(zoo)

get_ladder_by_round <- function(ladder_xml, year, round){
  
 ladder_by_round <- ladder_xml %>% xml_find_all(".//ladderposition") %>%
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

get_game_results <- function(fixtures_xml){
  
  game_results_long <- fixtures_xml %>% xml_find_all(".//gameFixture") %>%
    map_df(~{
      bind_cols(
        gameId =  xml_attr(.x, "gameId"),
        team = xml_find_all(.x, ".//teams/team") %>% xml_attr("team"),
        teamFinalScore = xml_find_all(.x, ".//teams/team") %>% xml_attr("teamFinalScore"),
        isHomeTeam = xml_find_all(.x, ".//teams/team") %>% xml_attr("isHomeTeam"),
        teamPosition = xml_find_all(.x, ".//teams/team") %>% xml_attr("teamPosition")
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

get_data <- function(password = rstudioapi::askForPassword, year_span){
  
  password <- rstudioapi::askForPassword("Enter your password")
  
  footy_tipper_dfs <- vector(mode = "list", length = length(year_span))
  
  for (y in 1:length(year_span)){
    
    # get that year's xml file
    fixtures_xml <- read_xml(paste0("http://", password, "@rugbyleague-api.stats.com/api/NRL/competitions/fixtures/111/", year_span[y]))
    
    # get fixture information
    fixture_info <- get_fixture_info(fixtures_xml)
    
    # get results from the game
    game_results <- get_game_results(fixtures_xml)
    
    # join it together and jot down the year
    footy_tipper_dfs[[y]] <- fixture_info %>% 
      inner_join(game_results, by = c('gameId')) %>%
      mutate(competition_year = year_span[y])
    
  }
  
  # bind each of the years together and do some cleaning
  footy_tipper_df <- bind_rows(footy_tipper_dfs) %>% type_convert()

  # return it to the env
  return(footy_tipper_df)

}
