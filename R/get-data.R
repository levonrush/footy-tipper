library(tidyverse)
library(xml2)
library(janitor)
library(zoo)

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
  
  # do some cleaning
  footy_tipper_df <- footy_tipper_df %>%
    mutate(broadcastChannel2 = if_else(is.na(broadcastChannel2), "None", broadcastChannel2),
           broadcastChannel3 = if_else(is.na(broadcastChannel3), "None", broadcastChannel3),
           venueName = fct_lump(venueName, 42)) %>%
    mutate_if(is.character, as.factor) %>%
    clean_names()
  
  # introduce the Dolphins as a factor level for R1 2023
  footy_tipper_df <- footy_tipper_df %>%
    mutate(team_home = fct_expand(team_home, "Dolphins"),
           team_away = fct_expand(team_away, "Dolphins"))
    
  # and create a response variable
  footy_tipper_df <- footy_tipper_df %>%
    mutate(home_team_result = if_else(team_final_score_home > team_final_score_away, "Win", "Loss") %>% as.factor()) %>%

  # return it to the env
  return(footy_tipper_df)

}
