Get, Save and Use Predicitons
================

# Get the predicitons from the DB and save them in the Google Drive

``` r
# Connect to the SQLite database
con <- dbConnect(SQLite(), paste0(here(), "/data/footy-tipper-db.sqlite"))

# Query to select all records from the predictions_table
query <- "

    with min_round_id as (

        select min(round_id) as round_id
        from footy_tipping_data
        where game_state_name = 'Pre Game'

    )

    select ft.game_id
        , p.home_team_result
        , ft.team_home
        , ft.position_home
        , ft.team_head_to_head_odds_home
        , ft.team_away
        , ft.position_away
        , ft.team_head_to_head_odds_away
        , p.home_team_win_prob
        , p.home_team_lose_prob
        , ft.home_elo
        , ft.away_elo
        , ft.round_id
        , ft.competition_year
    from predictions_table p
    left join footy_tipping_data ft on p.game_id = ft.game_id
    where ft.game_state_name = 'Pre Game'
    and round_id = (select * from min_round_id)

"

# Execute the query and fetch the results into a data frame
predictions <- dbGetQuery(con, query)
predictions
```

    ##       game_id home_team_result                    team_home position_home
    ## 1 20231111710             Loss St. George Illawarra Dragons            15
    ## 2 20231111720             Loss                     Dolphins            10
    ## 3 20231111730              Win             Penrith Panthers             2
    ## 4 20231111740              Win              Melbourne Storm             3
    ## 5 20231111750              Win             Brisbane Broncos             1
    ## 6 20231111760              Win       South Sydney Rabbitohs             4
    ## 7 20231111770              Win              Sydney Roosters            11
    ##   team_head_to_head_odds_home                  team_away position_away
    ## 1                        2.50       New Zealand Warriors             6
    ## 2                        2.45            Parramatta Eels             8
    ## 3                        1.28          Newcastle Knights            14
    ## 4                        1.35 Manly-Warringah Sea Eagles            12
    ## 5                        1.35          Gold Coast Titans             9
    ## 6                        1.45   North Queensland Cowboys            13
    ## 7                        1.55           Canberra Raiders             7
    ##   team_head_to_head_odds_away home_team_win_prob home_team_lose_prob home_elo
    ## 1                        1.54          0.4545149           0.5454851 1486.459
    ## 2                        1.55          0.4118439           0.5881561 1477.222
    ## 3                        3.75          0.6332769           0.3667231 1543.328
    ## 4                        3.25          0.6229361           0.3770639 1529.594
    ## 5                        3.25          0.6226031           0.3773969 1517.723
    ## 6                        2.75          0.5910776           0.4089224 1524.943
    ## 7                        2.45          0.5463827           0.4536173 1481.271
    ##   away_elo round_id competition_year
    ## 1 1509.067       17             2023
    ## 2 1538.374       17             2023
    ## 3 1487.508       17             2023
    ## 4 1494.580       17             2023
    ## 5 1485.104       17             2023
    ## 6 1498.625       17             2023
    ## 7 1484.288       17             2023

``` r
# Disconnect from the SQLite database
dbDisconnect(con)
```

# Save them to the drive

``` r
save_predictions(predictions, prod_run = prod_run)
```

# extra thangs…

``` r
levons_picks(predictions, prod_run = prod_run)
```

    ## [1] team      price     price_min
    ## <0 rows> (or 0-length row.names)
