import unittest

from pipeline.common.lineups.ingest import (
    _repair_existing_snapshot_if_needed,
    ensure_lineup_tables,
    extract_match_centre_urls,
    extract_topic_urls,
    is_nrl_mens_article,
    parse_match_centre_page,
    parse_team_list_article,
    utc_now_iso,
)


SAMPLE_ARTICLE_HTML = """
<html>
  <head><title>NRL Team Lists: Round 2</title></head>
  <body>
    <h1>NRL Team Lists: Round 2</h1>
    <time datetime="2026-03-10T04:00:00Z">Tue 10 Mar 2026</time>

    <div class="match-header">Round 2 - Match: Knights v Cowboys</div>
    <h4 class="teamsheet-group__title">Backs</h4>
    <ul>
      <li class="team-list">
        <div class="team-list-profile team-list-profile--home">
          <div class="team-list-profile__name">Fullback for Knights is number 1 Kalyn Ponga</div>
        </div>
        <div class="team-list-position"><p><span class="team-list-position__number">1</span></p></div>
        <div class="team-list-profile team-list-profile--away">
          <div class="team-list-profile__name">Fullback for Cowboys is number 1 Scott Drinkwater</div>
        </div>
      </li>
      <li class="team-list">
        <div class="team-list-profile team-list-profile--home">
          <div class="team-list-profile__name">Halfback for Knights is number 7 Jackson Hastings</div>
        </div>
        <div class="team-list-position"><p><span class="team-list-position__number">7</span></p></div>
        <div class="team-list-profile team-list-profile--away">
          <div class="team-list-profile__name">Halfback for Cowboys is number 7 Tom Dearden</div>
        </div>
      </li>
    </ul>

    <h4 class="teamsheet-group__title">Interchange</h4>
    <ul>
      <li class="team-list">
        <div class="team-list-profile team-list-profile--home">
          <div class="team-list-profile__name">Hooker for Knights is number 14 Phoenix Crossland</div>
        </div>
        <div class="team-list-position"><p><span class="team-list-position__number">14</span></p></div>
        <div class="team-list-profile team-list-profile--away">
          <div class="team-list-profile__name">Hooker for Cowboys is number 14 Reece Robson</div>
        </div>
      </li>
    </ul>
  </body>
</html>
"""


LEGACY_NAME_LIST_ARTICLE_HTML = """
<html>
  <head><title>Updated: Round 2 Official NRL Team Lists</title></head>
  <body>
    <h1>Updated: Round 2 Official NRL Team Lists</h1>
    <time datetime="2012-03-09T09:00:00+10:00">Fri 9 Mar 2012, 09:00 AM</time>
    <article>
      <div class="s-cms-content">
        <div>FRIDAY</div>
        <div>Manly Sea Eagles v Wests Tigers at Blue Tongue Stadium, 7:30pm.</div>
        <div>SEA EAGLES: Brett Stewart, Michael Oldfield, Jamie Lyon (c), Dean Whare, David Williams, Kieran Foran, Daly Cherry-Evans, Jason King, Matt Ballin, Brent Kite, Anthony Watmough, Tony Williams, Steve Matai. Interchange: Jamie Buhrer, Vic Mauro, Darcy Lussick, Tim Robinson, Daniel Harrison Late Mail: Jamie Lyon passed fit.</div>
        <div>WESTS TIGERS: Tom Humble, Beau Ryan, Blake Ayshford, Chris Lawrence, Matt Utai, Benji Marshall, Tim Moltzen, Aaron Woods, Robbie Farah (c), Matt Groat, Adam Blair, Gareth Ellis, Chris Heighington. Interchange (from): Liam Fulton, Junior Moors, Matt Bell, Joel Reddy, Ben Murdoch-Masila</div>
      </div>
    </article>
  </body>
</html>
"""


LEGACY_NUMBERED_ARTICLE_HTML = """
<html>
  <head><title>Round 1 NRL team lists</title></head>
  <body>
    <h1>Round 1 NRL team lists</h1>
    <time datetime="2018-03-06T07:00:00Z">Tue 6 Mar 2018</time>
    <article>
      <div class="s-cms-content">
        <div>St George Illawarra Dragons v Brisbane Broncos, 8.05pm (AEDT) Thursday, Jubilee Oval</div>
        <div>Dragons: 18 Matthew Dufty, 2 Nene Macdonald, 3 Euan Aitken, 4 Tim Lafai, 5 Jason Nightingale, 6 Gareth Widdop (c), 7 Ben Hunt, 8 James Graham, 9 Cameron McInnes, 10 Paul Vaughan, 11 Tyson Frizell, 12 Tariq Sims, 13 Jack De Belin</div>
        <div>Interchange: 14 Luciano Leilua, 15 Kurt Mann, 16 Leeson Ah Mau, 19 Jeremy Latimore</div>
        <div>Broncos: 1 Darius Boyd (c), 2 Corey Oates, 3 James Roberts, 4 Jordan Kahu, 5 Jamayne Isaako, 6 Anthony Milford, 7 Kodi Nikorima, 8 Matthew Lodge, 10 Samuel Thaiday, 14 Tevita Pangai Jnr, 11 Alex Glenn, 12 Matt Gillett, 13 Josh McGuire</div>
        <div>Interchange: 9 Andrew McCullough, 15 Joe Ofahengaue, 16 Korbin Sims, 17 Jaydn Su'a</div>
      </div>
    </article>
  </body>
</html>
"""


SAMPLE_TOPIC_HTML = """
<html>
  <body>
    <a href="/news/2026/03/10/nrl-team-lists-round-2/">NRL Team Lists</a>
    <a href="/news/2026/03/14/nrl-late-mail-round-2/">NRL Late Mail</a>
    <a href="/news/2026/03/10/nrlw-team-lists-round-2/">NRLW Team Lists</a>
    <a href="/watch/2026/03/10/highlights/">Highlights</a>
  </body>
</html>
"""


SAMPLE_DRAW_HTML = """
<html>
  <body>
    <div
      id="vue-draw"
      q-data="{&quot;fixtures&quot;:[{&quot;matchCentreUrl&quot;:&quot;/draw/nrl-premiership/2026/round-1/knights-v-cowboys/&quot;},{&quot;matchCentreUrl&quot;:&quot;/draw/nrl-premiership/2026/round-1/eels-v-tigers/&quot;}]}"
    ></div>
  </body>
</html>
"""


SAMPLE_MATCH_CENTRE_HTML = """
<html>
  <body>
    <h1>Knights v Cowboys</h1>
    <div
      id="vue-match-centre"
      q-data="{&quot;match&quot;:{&quot;matchId&quot;:&quot;20261110110&quot;,&quot;matchMode&quot;:&quot;Post&quot;,&quot;matchState&quot;:&quot;FullTime&quot;,&quot;updated&quot;:&quot;2026-03-01T04:15:00Z&quot;,&quot;roundNumber&quot;:1,&quot;roundTitle&quot;:&quot;Round 1&quot;,&quot;startTime&quot;:&quot;2026-03-01T02:15:00Z&quot;,&quot;homeTeam&quot;:{&quot;name&quot;:&quot;Newcastle Knights&quot;,&quot;nickName&quot;:&quot;Knights&quot;,&quot;players&quot;:[{&quot;firstName&quot;:&quot;Kalyn&quot;,&quot;lastName&quot;:&quot;Ponga&quot;,&quot;position&quot;:&quot;Fullback&quot;,&quot;playerId&quot;:504870,&quot;number&quot;:1,&quot;isOnField&quot;:true},{&quot;firstName&quot;:&quot;Phoenix&quot;,&quot;lastName&quot;:&quot;Crossland&quot;,&quot;position&quot;:&quot;Interchange&quot;,&quot;playerId&quot;:505463,&quot;number&quot;:14,&quot;isOnField&quot;:false},{&quot;firstName&quot;:&quot;Jack&quot;,&quot;lastName&quot;:&quot;Hetherington&quot;,&quot;position&quot;:&quot;Replacement&quot;,&quot;playerId&quot;:509999,&quot;number&quot;:18,&quot;isOnField&quot;:false}]},&quot;awayTeam&quot;:{&quot;name&quot;:&quot;North Queensland Cowboys&quot;,&quot;nickName&quot;:&quot;Cowboys&quot;,&quot;players&quot;:[{&quot;firstName&quot;:&quot;Scott&quot;,&quot;lastName&quot;:&quot;Drinkwater&quot;,&quot;position&quot;:&quot;Fullback&quot;,&quot;playerId&quot;:504087,&quot;number&quot;:1,&quot;isOnField&quot;:true},{&quot;firstName&quot;:&quot;Kai&quot;,&quot;lastName&quot;:&quot;ODonnell&quot;,&quot;position&quot;:&quot;Interchange&quot;,&quot;playerId&quot;:506555,&quot;number&quot;:17,&quot;isOnField&quot;:false},{&quot;firstName&quot;:&quot;Robert&quot;,&quot;lastName&quot;:&quot;Derby&quot;,&quot;position&quot;:&quot;Replacement&quot;,&quot;playerId&quot;:507777,&quot;number&quot;:18,&quot;isOnField&quot;:false}]}}}"
    ></div>
  </body>
</html>
"""


SAMPLE_DRAW_JSON = """
{"fixtures":[{"matchCentreUrl":"/draw/nrl-premiership/2026/round-1/knights-v-cowboys/"},{"matchCentreUrl":"/draw/nrl-premiership/2026/round-1/eels-v-tigers/"}]}
"""


SAMPLE_MATCH_CENTRE_JSON = """
{"matchId":"20261110110","matchMode":"Post","matchState":"FullTime","updated":"2026-03-01T04:15:00Z","roundNumber":1,"roundTitle":"Round 1","startTime":"2026-03-01T02:15:00Z","homeTeam":{"name":"Newcastle Knights","nickName":"Knights","players":[{"firstName":"Kalyn","lastName":"Ponga","position":"Fullback","playerId":504870,"number":1,"isOnField":true},{"firstName":"Phoenix","lastName":"Crossland","position":"Interchange","playerId":505463,"number":14,"isOnField":false},{"firstName":"Jack","lastName":"Hetherington","position":"Replacement","playerId":509999,"number":18,"isOnField":false}]},"awayTeam":{"name":"North Queensland Cowboys","nickName":"Cowboys","players":[{"firstName":"Scott","lastName":"Drinkwater","position":"Fullback","playerId":504087,"number":1,"isOnField":true},{"firstName":"Kai","lastName":"ODonnell","position":"Interchange","playerId":506555,"number":17,"isOnField":false},{"firstName":"Robert","lastName":"Derby","position":"Replacement","playerId":507777,"number":18,"isOnField":false}]}}
"""


class LineupIngestParsingTests(unittest.TestCase):
    def test_extract_match_centre_urls_from_draw_page(self):
        urls = extract_match_centre_urls(SAMPLE_DRAW_HTML)
        self.assertEqual(
            urls,
            [
                "https://www.nrl.com/draw/nrl-premiership/2026/round-1/knights-v-cowboys/",
                "https://www.nrl.com/draw/nrl-premiership/2026/round-1/eels-v-tigers/",
            ],
        )

    def test_extract_match_centre_urls_from_draw_json(self):
        urls = extract_match_centre_urls(SAMPLE_DRAW_JSON)
        self.assertEqual(
            urls,
            [
                "https://www.nrl.com/draw/nrl-premiership/2026/round-1/knights-v-cowboys/",
                "https://www.nrl.com/draw/nrl-premiership/2026/round-1/eels-v-tigers/",
            ],
        )

    def test_extract_topic_urls_filters_team_list_patterns(self):
        urls = extract_topic_urls(SAMPLE_TOPIC_HTML)
        self.assertEqual(len(urls), 3)
        self.assertEqual(
            urls[0],
            "https://www.nrl.com/news/2026/03/10/nrl-team-lists-round-2/",
        )

    def test_parse_team_list_article_extracts_entries(self):
        payload = parse_team_list_article(
            "https://www.nrl.com/news/2026/03/10/nrl-team-lists-round-2/",
            SAMPLE_ARTICLE_HTML,
        )

        self.assertEqual(payload["competition_year"], 2026)
        self.assertEqual(payload["round_id"], 2)
        self.assertEqual(payload["round_name"], "Round 2")
        self.assertEqual(payload["article_type"], "team_list")
        self.assertEqual(payload["source_published_at_utc"], "2026-03-10T04:00:00+00:00")
        self.assertEqual(len(payload["entries"]), 6)

        first = payload["entries"][0]
        self.assertEqual(first["team_name"], "Knights")
        self.assertEqual(first["side"], "home")
        self.assertEqual(first["listed_position"], "Fullback")
        self.assertEqual(first["jersey_number"], 1)
        self.assertEqual(first["player_name"], "Kalyn Ponga")

    def test_parse_match_centre_page_extracts_final_17_with_player_ids(self):
        payload = parse_match_centre_page(
            "https://www.nrl.com/draw/nrl-premiership/2026/round-1/knights-v-cowboys/",
            SAMPLE_MATCH_CENTRE_HTML,
        )

        self.assertEqual(payload["article_type"], "match_centre")
        self.assertEqual(payload["competition_year"], 2026)
        self.assertEqual(payload["round_id"], 1)
        self.assertEqual(payload["match_id"], "20261110110")
        self.assertEqual(payload["match_state"], "FullTime")
        self.assertEqual(len(payload["entries"]), 4)

        home = [row for row in payload["entries"] if row["side"] == "home"]
        away = [row for row in payload["entries"] if row["side"] == "away"]
        self.assertEqual(len(home), 2)
        self.assertEqual(len(away), 2)
        self.assertEqual(home[0]["player_external_id"], "504870")
        self.assertEqual(home[1]["squad_group"], "interchange")
        self.assertTrue(all(row["jersey_number"] <= 17 for row in payload["entries"]))

    def test_parse_match_centre_json_extracts_final_17_with_player_ids(self):
        payload = parse_match_centre_page(
            "https://www.nrl.com/draw/nrl-premiership/2026/round-1/knights-v-cowboys/",
            SAMPLE_MATCH_CENTRE_JSON,
        )

        self.assertEqual(payload["article_type"], "match_centre")
        self.assertEqual(payload["competition_year"], 2026)
        self.assertEqual(payload["round_id"], 1)
        self.assertEqual(payload["article_title"], "Knights v Cowboys")
        self.assertEqual(payload["match_id"], "20261110110")
        self.assertEqual(len(payload["entries"]), 4)
        self.assertEqual(payload["entries"][0]["player_external_id"], "504870")

    def test_parse_legacy_name_list_article_extracts_entries(self):
        payload = parse_team_list_article(
            "https://www.nrl.com/news/2012/03/09/updated-round-2-official-nrl-team-lists/",
            LEGACY_NAME_LIST_ARTICLE_HTML,
        )

        self.assertEqual(payload["competition_year"], 2012)
        self.assertEqual(payload["round_id"], 2)
        self.assertEqual(payload["round_name"], "Round 2")
        self.assertEqual(payload["source_published_at_utc"], "2012-03-08T23:00:00+00:00")
        self.assertEqual(len(payload["entries"]), 36)

        first = payload["entries"][0]
        self.assertEqual(first["team_name"], "Manly Sea Eagles")
        self.assertEqual(first["side"], "home")
        self.assertEqual(first["jersey_number"], 1)
        self.assertEqual(first["listed_position"], "Fullback")
        self.assertEqual(first["player_name"], "Brett Stewart")

        reserve = next(row for row in payload["entries"] if row["player_name"] == "Daniel Harrison")
        self.assertEqual(reserve["jersey_number"], 18)
        self.assertEqual(reserve["squad_group"], "reserves")

    def test_parse_legacy_numbered_article_canonicalizes_slots(self):
        payload = parse_team_list_article(
            "https://www.nrl.com/news/2018/03/06/round-1-nrl-team-lists/",
            LEGACY_NUMBERED_ARTICLE_HTML,
        )

        self.assertEqual(payload["competition_year"], 2018)
        self.assertEqual(payload["round_id"], 1)
        self.assertEqual(payload["round_name"], "Round 1")
        self.assertEqual(len(payload["entries"]), 34)

        dufty = next(row for row in payload["entries"] if row["player_name"] == "Matthew Dufty")
        self.assertEqual(dufty["team_name"], "St George Illawarra Dragons")
        self.assertEqual(dufty["jersey_number"], 1)
        self.assertEqual(dufty["listed_position"], "Fullback")

        mccullough = next(row for row in payload["entries"] if row["player_name"] == "Andrew McCullough")
        self.assertEqual(mccullough["team_name"], "Brisbane Broncos")
        self.assertEqual(mccullough["jersey_number"], 14)
        self.assertEqual(mccullough["squad_group"], "interchange")
        self.assertEqual(mccullough["listed_position"], "Hooker")

    def test_existing_zero_entry_snapshot_is_repaired(self):
        import hashlib
        import sqlite3

        payload = parse_team_list_article(
            "https://www.nrl.com/news/2018/03/06/round-1-nrl-team-lists/",
            LEGACY_NUMBERED_ARTICLE_HTML,
        )
        content_hash = hashlib.sha256(LEGACY_NUMBERED_ARTICLE_HTML.encode("utf-8")).hexdigest()

        con = sqlite3.connect(":memory:")
        try:
            ensure_lineup_tables(con)
            con.execute(
                """
                INSERT INTO lineup_article_snapshots (
                    article_url,
                    article_title,
                    article_type,
                    competition_year,
                    round_id,
                    round_name,
                    source_published_at_utc,
                    scraped_at_utc,
                    content_hash,
                    parse_status,
                    parse_error,
                    entry_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'ok', NULL, 0)
                """,
                (
                    payload["article_url"],
                    payload["article_title"],
                    payload["article_type"],
                    payload["competition_year"],
                    payload["round_id"],
                    payload["round_name"],
                    payload["source_published_at_utc"],
                    utc_now_iso(),
                    content_hash,
                ),
            )

            snapshot_id, inserted = _repair_existing_snapshot_if_needed(
                con,
                parsed_article=payload,
                content_hash=content_hash,
                scraped_at_utc=utc_now_iso(),
            )

            self.assertIsNotNone(snapshot_id)
            self.assertEqual(inserted, len(payload["entries"]))
            repaired_entry_count = con.execute(
                "SELECT entry_count FROM lineup_article_snapshots WHERE snapshot_id = ?",
                (snapshot_id,),
            ).fetchone()[0]
            self.assertEqual(repaired_entry_count, len(payload["entries"]))
        finally:
            con.close()

    def test_existing_zero_entry_snapshot_is_repaired_on_same_url_with_new_hash(self):
        import hashlib
        import sqlite3

        payload = parse_match_centre_page(
            "https://www.nrl.com/draw/nrl-premiership/2026/round-1/knights-v-cowboys/",
            SAMPLE_MATCH_CENTRE_JSON,
        )
        old_hash = hashlib.sha256(SAMPLE_MATCH_CENTRE_HTML.encode("utf-8")).hexdigest()
        new_hash = hashlib.sha256(SAMPLE_MATCH_CENTRE_JSON.encode("utf-8")).hexdigest()

        con = sqlite3.connect(":memory:")
        try:
            ensure_lineup_tables(con)
            con.execute(
                """
                INSERT INTO lineup_article_snapshots (
                    article_url,
                    article_title,
                    article_type,
                    competition_year,
                    round_id,
                    round_name,
                    source_published_at_utc,
                    scraped_at_utc,
                    content_hash,
                    parse_status,
                    parse_error,
                    entry_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'ok', NULL, 0)
                """,
                (
                    payload["article_url"],
                    payload["article_title"],
                    payload["article_type"],
                    payload["competition_year"],
                    payload["round_id"],
                    payload["round_name"],
                    payload["source_published_at_utc"],
                    utc_now_iso(),
                    old_hash,
                ),
            )

            snapshot_id, inserted = _repair_existing_snapshot_if_needed(
                con,
                parsed_article=payload,
                content_hash=new_hash,
                scraped_at_utc=utc_now_iso(),
            )

            self.assertIsNotNone(snapshot_id)
            self.assertEqual(inserted, len(payload["entries"]))
            repaired_row = con.execute(
                "SELECT entry_count, content_hash FROM lineup_article_snapshots WHERE snapshot_id = ?",
                (snapshot_id,),
            ).fetchone()
            self.assertEqual(repaired_row[0], len(payload["entries"]))
            self.assertEqual(repaired_row[1], new_hash)
        finally:
            con.close()

    def test_is_nrl_mens_article_blocks_nrlw(self):
        self.assertFalse(
            is_nrl_mens_article(
                "https://www.nrl.com/news/2026/03/10/nrlw-team-lists-round-2/",
                "NRLW Team Lists: Round 2",
            )
        )
        self.assertTrue(
            is_nrl_mens_article(
                "https://www.nrl.com/news/2026/03/10/nrl-team-lists-round-2/",
                "NRL Team Lists: Round 2",
            )
        )


if __name__ == "__main__":
    unittest.main()
