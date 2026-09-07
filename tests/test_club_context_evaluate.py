import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import club_context as context_cli
from pipeline import club_context_evaluate as context_eval


CATALOG_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "reference"
    / "club_context_events.json"
)


def _paired_frame() -> pd.DataFrame:
    games = 48
    index = np.arange(games)
    baseline = np.where(index % 2 == 0, 0.49, 0.58)
    candidate = baseline.copy()
    candidate[:3] = np.array([0.53, 0.47, 0.52])
    relative = np.tile(np.arange(-3, 4), 7)[:games].astype(float)
    annotated = index < 35
    active = annotated & (relative >= 0)
    return pd.DataFrame(
        {
            "game_id": index + 1,
            "competition_year": np.where(index < 24, 2024, 2025),
            "round_id": (index % 24) + 1,
            "team_home": [f"Home {value}" for value in index],
            "team_away": [f"Away {value}" for value in index],
            "actual_home_win": index % 2,
            "baseline_home_win_prob": baseline,
            "context_home_win_prob": candidate,
            "baseline_no_market_home_win_prob": np.clip(baseline - 0.02, 0.01, 0.99),
            "context_no_market_home_win_prob": np.clip(candidate - 0.01, 0.01, 0.99),
            "actual_home_score": 18 + (index % 9),
            "actual_away_score": 16 + (index % 7),
            "baseline_home_score": np.full(games, 21.0),
            "baseline_away_score": np.full(games, 19.0),
            "context_home_score": np.full(games, 20.5),
            "context_away_score": np.full(games, 18.5),
            "market_available": index % 3 != 0,
            "market_spread": np.where(index % 3 != 0, 2.5, np.nan),
            "event_id": np.where(annotated, "event-" + (index // 7).astype(str), None),
            "category": np.where(annotated, "leadership_change", None),
            "event_relative_match": np.where(annotated, relative, np.nan),
            "affected_side": np.where(index % 4 == 0, "away", "home"),
            "context_event_count": active.astype(float),
            "matches_since_event": np.where(active, np.maximum(relative, 0), np.nan),
            "context_data_available": np.ones(games),
            "context_source_diversity": active.astype(float) * 2,
            "context_official_count": active.astype(float),
            "form_delta": np.sin(index),
            "rest_delta": index % 3,
            "elo_diff": np.cos(index) * 10,
            "lineup_selection_uncertainty_delta": np.sin(index / 3),
            "venue_name": np.where(index % 2, "A", "B"),
        }
    )


class ClubContextMaterialityTests(unittest.TestCase):
    def test_shadow_report_separates_active_features_from_pretrend_labels(self):
        frame = _paired_frame()
        report = context_eval.evaluate_shadow_frame(
            frame, seed=73, bootstrap_reps=50
        )

        self.assertEqual(report["rows"], len(frame))
        self.assertEqual(
            report["event_linked_rows"], int(frame["context_event_count"].sum())
        )
        self.assertIn("all_games", report["counterfactual_without_market_inputs"])
        self.assertTrue(report["market_absorption"]["available"])
        self.assertTrue(report["dispersion_hypothesis"]["tested"])
        self.assertTrue(report["event_study"]["available"])
        self.assertIn("matched_event_match", report["event_study"])
        self.assertEqual(
            report["source_coverage"]["registry_available_games"], len(frame)
        )
        self.assertGreater(report["changed_tip_examples"]["count"], 0)

    def test_seeded_cluster_results_are_reproducible(self):
        frame = _paired_frame()
        first = context_eval.evaluate_shadow_frame(
            frame, seed=20100308, bootstrap_reps=80
        )
        second = context_eval.evaluate_shadow_frame(
            frame, seed=20100308, bootstrap_reps=80
        )

        self.assertEqual(
            first["event_cluster_bootstrap"], second["event_cluster_bootstrap"]
        )
        self.assertEqual(first["event_study"], second["event_study"])

    def test_materiality_report_contains_market_and_changed_tip_sections(self):
        report = context_eval.evaluate_shadow_frame(
            _paired_frame(), seed=8, bootstrap_reps=10
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path, markdown_path = context_eval.write_materiality_report(
                report, Path(temp_dir) / "materiality.json"
            )
            restored = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual(restored["shadow_only"], True)
        self.assertIn("Counterfactual without market inputs", markdown)
        self.assertIn("Evidence coverage", markdown)
        self.assertIn("Changed tips", markdown)


class ClubContextResearchAnnotationTests(unittest.TestCase):
    def test_arrow_pretrend_labels_do_not_require_prematch_feature_exposure(self):
        matches = pd.DataFrame(
            [
                {
                    "game_id": 9000 + offset,
                    "competition_year": 2026,
                    "round_id": 10 + offset,
                    "start_time_utc": kickoff,
                    "team_home": "South Sydney Rabbitohs"
                    if offset % 2 == 0
                    else "Brisbane Broncos",
                    "team_away": "Brisbane Broncos"
                    if offset % 2 == 0
                    else "South Sydney Rabbitohs",
                }
                for offset, kickoff in enumerate(
                    (
                        "2026-05-10T06:00:00+00:00",
                        "2026-05-17T06:00:00+00:00",
                        "2026-05-24T06:00:00+00:00",
                        "2026-05-31T06:00:00+00:00",
                        "2026-06-07T06:00:00+00:00",
                    )
                )
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "context.sqlite"
            result = context_cli._backfill(
                db_path,
                CATALOG_PATH,
                start_year=2026,
                end_year=2026,
            )
            annotations = context_eval.build_event_annotations(matches, db_path)

        self.assertEqual(result["errors"], [])
        arrow = annotations[
            annotations["event_key"]
            == "human-rabbitohs-arrow-retirement-announcement-2026"
        ]
        self.assertFalse(arrow.empty)
        self.assertIn(-1, set(arrow["event_relative_match"]))
        self.assertIn(0, set(arrow["event_relative_match"]))


if __name__ == "__main__":
    unittest.main()
