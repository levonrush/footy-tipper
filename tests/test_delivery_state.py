import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline.ops import delivery_state


class DeliveryStateTests(unittest.TestCase):
    def _valid_marker(self, **overrides):
        marker = {
            "schema_version": 1,
            "competition_year": 2026,
            "round_id": 21,
            "status": "pending",
            "attempt_id": "attempt-1",
        }
        marker.update(overrides)
        return marker

    def test_marker_round_must_match_requested_filename(self):
        with self.assertRaisesRegex(RuntimeError, "year/round.*filename"):
            delivery_state._validate_marker(
                "2026-round-21.json", self._valid_marker(round_id=22)
            )

    def test_marker_status_must_be_known(self):
        with self.assertRaisesRegex(RuntimeError, "invalid status"):
            delivery_state._validate_marker(
                "2026-round-21.json", self._valid_marker(status="uncertain")
            )

    def test_marker_attempt_id_must_be_present(self):
        for attempt_id in (None, "", "   ", 123):
            with self.subTest(attempt_id=attempt_id), self.assertRaisesRegex(
                RuntimeError, "invalid attempt_id"
            ):
                delivery_state._validate_marker(
                    "2026-round-21.json",
                    self._valid_marker(attempt_id=attempt_id),
                )

    def test_existing_pending_marker_blocks_a_second_claim(self):
        marker = {
            "schema_version": 1,
            "competition_year": 2026,
            "round_id": 21,
            "status": "pending",
            "attempt_id": "first-attempt",
        }
        with mock.patch.object(
            delivery_state, "_context", return_value=(Path("/repo"), object(), "folder")
        ), mock.patch.object(
            delivery_state, "_download_marker", return_value=marker
        ), mock.patch.object(
            delivery_state.state_sync, "upload_create_only"
        ) as upload:
            result = delivery_state.begin_delivery("/repo", 2026, 21)

        self.assertFalse(result["allowed"])
        self.assertEqual(result["marker"]["status"], "pending")
        upload.assert_not_called()

    def test_claim_is_persisted_as_pending_before_smtp(self):
        captured = {}

        def capture_upload(_service, _folder, _name, path, _mime):
            captured.update(json.loads(Path(path).read_text(encoding="utf-8")))

        with mock.patch.object(
            delivery_state, "_context", return_value=(Path("/repo"), object(), "folder")
        ), mock.patch.object(
            delivery_state, "_download_marker", return_value=None
        ), mock.patch.object(
            delivery_state.state_sync,
            "upload_create_only",
            side_effect=capture_upload,
        ):
            result = delivery_state.begin_delivery("/repo", 2026, 21)

        self.assertTrue(result["allowed"])
        self.assertEqual(captured["status"], "pending")
        self.assertEqual(captured["competition_year"], 2026)
        self.assertTrue(captured["attempt_id"])

    def test_only_the_claiming_attempt_can_mark_sent(self):
        marker = {
            "schema_version": 1,
            "competition_year": 2026,
            "round_id": 21,
            "status": "pending",
            "attempt_id": "first-attempt",
        }
        with mock.patch.object(
            delivery_state, "_context", return_value=(Path("/repo"), object(), "folder")
        ), mock.patch.object(
            delivery_state, "_download_marker", return_value=marker
        ):
            with self.assertRaisesRegex(RuntimeError, "different attempt"):
                delivery_state.mark_sent("/repo", 2026, 21, "other-attempt")

    def test_successful_smtp_marker_becomes_sent(self):
        marker = {
            "schema_version": 1,
            "competition_year": 2026,
            "round_id": 21,
            "status": "pending",
            "attempt_id": "attempt-1",
        }
        captured = {}

        def capture_upload(_service, _folder, _name, path, _mime):
            captured.update(json.loads(Path(path).read_text(encoding="utf-8")))

        with mock.patch.object(
            delivery_state, "_context", return_value=(Path("/repo"), object(), "folder")
        ), mock.patch.object(
            delivery_state, "_download_marker", return_value=marker
        ), mock.patch.object(
            delivery_state.state_sync, "upload_or_update", side_effect=capture_upload
        ):
            result = delivery_state.mark_sent(
                "/repo", 2026, 21, "attempt-1", recipients_count=12
            )

        self.assertEqual(result["status"], "sent")
        self.assertEqual(captured["recipients_count"], 12)


if __name__ == "__main__":
    unittest.main()
