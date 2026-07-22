import os
import tempfile
import unittest
from unittest import mock

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


class ProductionEmailTests(unittest.TestCase):
    def test_full_smtp_success_returns_recipient_count(self):
        prepared = dist.PreparedEmailDelivery(
            sender_email="sender@example.com",
            sender_password="app-password",
            recipients=("one@example.com", "two@example.com"),
        )
        server = mock.Mock()
        server.sendmail.return_value = {}

        with mock.patch("pipeline.common.use_predictions.distribution.smtplib.SMTP", return_value=server):
            result = dist.send_emails("Tips", "Body", prepared)

        self.assertEqual(result, 2)
        server.sendmail.assert_called_once()
        self.assertEqual(server.sendmail.call_args.args[1], ["one@example.com", "two@example.com"])
        server.quit.assert_called_once()

    def test_partial_recipient_refusal_is_ambiguous_failure(self):
        prepared = dist.PreparedEmailDelivery(
            sender_email="sender@example.com",
            sender_password="app-password",
            recipients=("accepted@example.com", "refused@example.com"),
        )
        server = mock.Mock()
        server.sendmail.return_value = {
            "refused@example.com": (550, b"mailbox unavailable")
        }

        with mock.patch("pipeline.common.use_predictions.distribution.smtplib.SMTP", return_value=server):
            result = dist.send_emails("Tips", "Body", prepared)

        self.assertFalse(result)
        server.quit.assert_called_once()

    def test_prepare_resolves_recipients_once_and_deduplicates(self):
        service_account = mock.Mock()
        gspread = mock.Mock()
        gspread.authorize.return_value.open.return_value.sheet1.get_all_records.return_value = [
            {"Email": "one@example.com"},
            {"Email": "One@example.com"},
            {"Email": "Two Person <two@example.com>"},
            {"Email": ""},
        ]

        with tempfile.NamedTemporaryFile() as token, mock.patch.object(
            dist, "service_account", service_account
        ), mock.patch.object(dist, "gspread", gspread):
            prepared = dist.prepare_email_delivery(
                "footy-tipper-email-list",
                "sender@example.com",
                "app-password",
                token.name,
            )

        self.assertEqual(
            prepared.recipients,
            ("one@example.com", "two@example.com"),
        )
        service_account.Credentials.from_service_account_file.assert_called_once()
        gspread.authorize.return_value.open.assert_called_once_with(
            "footy-tipper-email-list"
        )

    def test_test_email_recipient_refusal_returns_false(self):
        server = mock.Mock()
        server.sendmail.return_value = {
            "test@example.com": (550, b"mailbox unavailable")
        }

        with mock.patch("pipeline.common.use_predictions.distribution.smtplib.SMTP", return_value=server):
            result = dist.send_test_email(
                "Tips",
                "Body",
                "sender@example.com",
                "app-password",
                "test@example.com",
            )

        self.assertFalse(result)


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
