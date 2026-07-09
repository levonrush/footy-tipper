import pathlib
import shutil
import sqlite3
import tempfile
import unittest

from pipeline.common.use_predictions.site import generate_site

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _make_project_root(tmp_dir):
    """Temp project root with the real SQL files the site generator reads."""
    root = pathlib.Path(tmp_dir)
    sql_dst = root / "pipeline" / "common" / "sql"
    sql_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPO_ROOT / "pipeline" / "common" / "sql", sql_dst)
    (root / "data").mkdir()
    return root


def _build_db(db_path, with_pregame=True):
    con = sqlite3.connect(str(db_path))
    con.executescript(
        """
        CREATE TABLE footy_tipping_data (
            game_id INTEGER PRIMARY KEY,
            game_state_name TEXT,
            competition_year INTEGER,
            round_id INTEGER,
            round_name TEXT,
            team_home TEXT,
            position_home_ladder INTEGER,
            team_head_to_head_odds_home REAL,
            team_away TEXT,
            position_away_ladder INTEGER,
            team_head_to_head_odds_away REAL,
            team_final_score_home REAL,
            team_final_score_away REAL,
            start_time REAL,
            game_number INTEGER
        );
        CREATE TABLE predictions_table (
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
        );
        """
    )
    rows = [
        (1, "Final", 2026, 14, "Round 14", "Knights", 4, 1.6, "Sharks", 7, 2.3, 24, 12, 1.0, 1),
    ]
    predictions = [
        (1, "Win", 0.66, 0.32, 0.02, 2.1, "Anecdotal evidence", 24, 14, 10),
    ]
    if with_pregame:
        rows.append(
            (2, "Pre Game", 2026, 15, "Round 15", "Storm", 1, 1.3, "Titans", 12, 3.6, None, None, 2.0, 1)
        )
        predictions.append(
            (2, "Win", 0.81, 0.17, 0.02, 4.8, "Moderate evidence", 28, 12, 15)
        )
    con.executemany(
        "INSERT INTO footy_tipping_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    con.executemany(
        "INSERT INTO predictions_table VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        predictions,
    )
    con.commit()
    con.close()


class SiteGenerationTests(unittest.TestCase):
    def test_generates_index_round_archive_and_results(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = _make_project_root(tmp_dir)
            db_path = root / "data" / "db.sqlite"
            _build_db(db_path)

            written = generate_site(db_path, root)
            names = {p.relative_to(root).as_posix() for p in written}
            self.assertEqual(
                names,
                {
                    "docs/site/index.html",
                    "docs/site/rounds/2026-round-15.html",
                    "docs/site/results.html",
                },
            )

            index = (root / "docs" / "site" / "index.html").read_text()
            self.assertIn("Round 15", index)
            self.assertIn("Storm", index)
            self.assertIn("The ledger", index)

            results = (root / "docs" / "site" / "results.html").read_text()
            self.assertIn("Season results", results)
            self.assertIn("Knights", results)

    def test_offseason_site_still_generates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = _make_project_root(tmp_dir)
            db_path = root / "data" / "db.sqlite"
            _build_db(db_path, with_pregame=False)

            written = generate_site(db_path, root)
            names = {p.relative_to(root).as_posix() for p in written}
            self.assertEqual(names, {"docs/site/index.html", "docs/site/results.html"})

            index = (root / "docs" / "site" / "index.html").read_text()
            self.assertIn("No upcoming pre-game fixtures", index)


if __name__ == "__main__":
    unittest.main()
