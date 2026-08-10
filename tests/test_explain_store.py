import pathlib
import shutil
import sqlite3
import tempfile
import unittest

from pipeline.common.explain import game as xgame
from pipeline.common.explain import store as xstore
from pipeline.common.explain import trace as xt

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _make_project_root(tmp_dir):
    root = pathlib.Path(tmp_dir)
    (root / "data").mkdir(parents=True, exist_ok=True)
    shutil.copytree(PROJECT_ROOT / "pipeline" / "common" / "sql", root / "pipeline" / "common" / "sql")
    return root


def _explanation(game_id=1, why="Storm favoured on ladder (+9 pts)."):
    return xgame.GameExplanation(
        game_id=game_id,
        team_home="Storm",
        team_away="Titans",
        probability=xt.ProbabilityTrace(
            game_id=game_id,
            tier_a=0.61,
            tier_b=0.58,
            tier_c=0.70,
            market=0.66,
            valid_market=True,
            weights={"tier_c": 1.0, "market": 0.0},
            expert_logit_terms={"tier_c": 0.8473},
            pooled_logit=0.8473,
            temperature=0.9277,
            calibrated_logit=0.9133,
            route="market",
            published_cond=0.7137,
        ),
        score=xt.ScoreTrace(
            game_id=game_id,
            mu_model_home=25.9,
            mu_model_away=17.8,
            line_applied=True,
            displayed_home=29,
            displayed_away=20,
            displayed_margin=9,
        ),
        prob_drivers=(
            xgame.Driver(
                key="position_home_ladder",
                label="position_home_ladder",
                family="ladder",
                points=4.2,
                share=0.31,
                detail="position_home_ladder = 3",
            ),
        ),
        prob_families=(
            xgame.Driver(key="ladder", label="Ladder", family="ladder", points=9.0, share=0.5),
        ),
        why_line=why,
    )


class ExplanationStoreTests(unittest.TestCase):
    def test_round_trip_preserves_the_trace_and_drivers(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = _make_project_root(tmp_dir)
            db_path = root / "data" / "db.sqlite"

            written = xstore.save_explanations([_explanation()], db_path, root)
            self.assertEqual(written, 1)

            loaded = xstore.load_game_explanations(db_path)
            self.assertEqual(len(loaded), 1)
            restored = loaded[0]
            self.assertEqual(restored.team_home, "Storm")
            self.assertEqual(restored.tipped_team, "Storm")
            self.assertAlmostEqual(restored.probability.published_cond, 0.7137)
            self.assertEqual(restored.probability.route, "market")
            self.assertTrue(restored.score.line_applied)
            self.assertEqual(restored.prob_families[0].label, "Ladder")
            self.assertEqual(restored.prob_drivers[0].detail, "position_home_ladder = 3")

    def test_rerunning_inference_upserts_rather_than_duplicating(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = _make_project_root(tmp_dir)
            db_path = root / "data" / "db.sqlite"

            xstore.save_explanations([_explanation(why="first")], db_path, root)
            xstore.save_explanations([_explanation(why="second")], db_path, root)

            frame = xstore.load_explanations(db_path)
            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.iloc[0]["why_line"], "second")

    def test_missing_table_returns_empty_rather_than_raising(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = pathlib.Path(tmp_dir) / "empty.sqlite"
            sqlite3.connect(str(db_path)).close()

            self.assertTrue(xstore.load_explanations(db_path).empty)
            self.assertEqual(xstore.load_game_explanations(db_path), [])
            self.assertEqual(xstore.why_lines(db_path), {})

    def test_column_migration_is_additive(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = pathlib.Path(tmp_dir) / "old.sqlite"
            con = sqlite3.connect(str(db_path))
            con.execute(
                f"CREATE TABLE {xstore.TABLE_NAME} (game_id INTEGER PRIMARY KEY, why_line TEXT)"
            )
            con.execute(
                f"INSERT INTO {xstore.TABLE_NAME} (game_id, why_line) VALUES (1, 'kept')"
            )
            xstore._ensure_explanations_table_columns(con)
            con.commit()

            columns = {row[1] for row in con.execute(
                f"PRAGMA table_info({xstore.TABLE_NAME})"
            ).fetchall()}
            self.assertIn("attribution_source", columns)
            self.assertIn("trace_json", columns)
            # The pre-existing row survives.
            self.assertEqual(
                con.execute(f"SELECT why_line FROM {xstore.TABLE_NAME}").fetchone()[0],
                "kept",
            )
            con.close()

    def test_why_lines_filters_to_requested_games(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = _make_project_root(tmp_dir)
            db_path = root / "data" / "db.sqlite"
            xstore.save_explanations(
                [_explanation(1, "one"), _explanation(2, "two")], db_path, root
            )

            self.assertEqual(xstore.why_lines(db_path, [2]), {2: "two"})
            self.assertEqual(xstore.why_lines(db_path, []), {})

    def test_manifest_release_dict_is_stored_as_text(self):
        # models/model_manifest.json stores release as a dict, and sqlite cannot
        # bind one. Coercing here is what keeps the write from failing silently.
        self.assertEqual(
            xstore._release_label({"release_id": "abc", "git_sha": "def"}), "abc"
        )
        self.assertEqual(xstore._release_label({"git_sha": "def"}), "def")
        self.assertIsNone(xstore._release_label(None))
        self.assertEqual(xstore._release_label("plain"), "plain")

    def test_unreadable_row_is_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = _make_project_root(tmp_dir)
            db_path = root / "data" / "db.sqlite"
            xstore.save_explanations([_explanation(1), _explanation(2)], db_path, root)

            con = sqlite3.connect(str(db_path))
            con.execute(
                f"UPDATE {xstore.TABLE_NAME} SET trace_json = 'not json' WHERE game_id = 1"
            )
            con.commit()
            con.close()

            loaded = xstore.load_game_explanations(db_path)
            self.assertEqual([e.game_id for e in loaded], [2])


if __name__ == "__main__":
    unittest.main()
