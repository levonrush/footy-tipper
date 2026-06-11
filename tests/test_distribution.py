import os
import sqlite3
import tempfile
import unittest

from pipeline.common.use_predictions import distribution as dist


class MimeMessageTests(unittest.TestCase):
    def test_list_send_hides_recipients_via_bcc(self):
        recipients = ["mate1@example.com", "mate2@example.com"]
        msg = dist._build_mime_message(
            subject="Tips",
            sender_email="sender@example.com",
            recipients=recipients,
            plain_message="body",
            html_message="<p>body</p>",
            bcc_recipients=True,
        )
        self.assertEqual(msg["To"], "sender@example.com")
        raw = msg.as_string()
        for address in recipients:
            self.assertNotIn(address, raw)

    def test_test_send_keeps_single_recipient_in_to(self):
        msg = dist._build_mime_message(
            subject="Tips",
            sender_email="sender@example.com",
            recipients=["one@example.com"],
            plain_message="body",
        )
        self.assertEqual(msg["To"], "one@example.com")

    def test_list_unsubscribe_header_present(self):
        msg = dist._build_mime_message(
            subject="Tips",
            sender_email="sender@example.com",
            recipients=["one@example.com"],
            plain_message="body",
        )
        self.assertEqual(msg["List-Unsubscribe"], "<mailto:sender@example.com?subject=unsubscribe>")


class SendLedgerTests(unittest.TestCase):
    def setUp(self):
        handle, self.db_path = tempfile.mkstemp(suffix=".sqlite")
        os.close(handle)

    def tearDown(self):
        os.remove(self.db_path)

    def test_round_trip(self):
        self.assertIsNone(dist.email_send_already_recorded(self.db_path, 2026, 15))

        self.assertTrue(
            dist.record_email_send(self.db_path, 2026, 15, recipients_count=12, source="test")
        )
        record = dist.email_send_already_recorded(self.db_path, 2026, 15)
        self.assertIsNotNone(record)
        self.assertEqual(record["recipients_count"], 12)
        self.assertEqual(record["source"], "test")

        # Second record for the same round is ignored (primary key), first wins.
        dist.record_email_send(self.db_path, 2026, 15, recipients_count=99, source="other")
        record = dist.email_send_already_recorded(self.db_path, 2026, 15)
        self.assertEqual(record["recipients_count"], 12)

        # Other rounds are unaffected.
        self.assertIsNone(dist.email_send_already_recorded(self.db_path, 2026, 16))

    def test_lookup_is_fail_soft_on_bad_path(self):
        self.assertIsNone(dist.email_send_already_recorded("/nonexistent/dir/db.sqlite", 2026, 1))

    def test_missing_context_returns_none(self):
        self.assertIsNone(dist.email_send_already_recorded(self.db_path, None, 1))
        self.assertFalse(dist.record_email_send(self.db_path, None, 1))


if __name__ == "__main__":
    unittest.main()
