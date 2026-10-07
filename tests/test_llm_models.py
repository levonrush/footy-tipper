"""Exercise hosted-model migrations through the SDKs without network or secrets."""

import base64
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import httpx
import pandas as pd
from PIL import Image

from pipeline.common.use_predictions import banner, email_copy, llm, news


def _predictions():
    return pd.DataFrame([{
        "game_id": 1, "competition_year": 2026, "round_id": 15,
        "round_name": "Round 15", "team_home": "Knights", "team_away": "Storm",
        "home_team_result": "Win", "home_team_win_prob": 0.6, "home_team_lose_prob": 0.4,
        "team_head_to_head_odds_home": 1.8, "team_head_to_head_odds_away": 2.1,
        "predicted_home_score": 24, "predicted_away_score": 18, "predicted_margin": 6,
    }])


class ClaudeTextTests(unittest.TestCase):
    def test_only_text_blocks_are_used_in_order(self):
        response = SimpleNamespace(stop_reason="end_turn", content=[
            SimpleNamespace(type="thinking", thinking="Private reasoning"),
            SimpleNamespace(type="redacted_thinking", data="opaque"),
            SimpleNamespace(type="text", text="First "),
            SimpleNamespace(type="tool_use", input={"text": "Not the answer"}),
            SimpleNamespace(type="text", text="second."),
        ])
        self.assertEqual(llm.claude_response_text(response), "First second.")

    def test_incomplete_or_refused_output_is_not_published(self):
        for reason in ("max_tokens", "refusal", "model_context_window_exceeded"):
            with self.subTest(reason=reason):
                response = SimpleNamespace(stop_reason=reason, content=[
                    SimpleNamespace(type="text", text="A plausible but incomplete answer"),
                ])
                self.assertEqual(llm.claude_response_text(response), "")
        self.assertEqual(llm.claude_response_text(SimpleNamespace(content=[])), "")


@unittest.skipIf(banner.Anthropic is None or banner.OpenAIClient is None, "Optional AI SDKs unavailable")
class ModelSDKTests(unittest.TestCase):
    def setUp(self):
        from anthropic import Anthropic
        from openai import OpenAI

        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.dict(os.environ, {}, clear=True))
        self.stack.enter_context(mock.patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden")))
        self.stack.enter_context(mock.patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Network forbidden")))
        self.logs = io.StringIO()
        self.stack.enter_context(redirect_stdout(self.logs))
        self.stack.enter_context(redirect_stderr(self.logs))
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        (self.root / "images").mkdir()
        buffer = io.BytesIO()
        Image.new("RGB", (1536, 1024), "blue").save(buffer, format="PNG")
        self.png = buffer.getvalue()
        self.static = self.root / "images/email-banner.png"
        self.static.write_bytes(self.png)
        self.stack.enter_context(mock.patch.object(banner, "__file__", str(self.root / "pipeline/common/use_predictions/banner.py")))
        self.anthropic_requests = []
        self.image_requests = []
        self.anthropic_status = 200
        self.image_status = 200
        self.stop_reason = "end_turn"
        self.answer = "Reg and Ernie wave beside the logo."
        self.copy = {"subject": "Round 15", "opening": "Opening.", "closing": "Bring back the biff.", "news_hit": None}
        self.claude = Anthropic(api_key="offline-placeholder", max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(self._claude_response)))
        self.openai = OpenAI(api_key="offline-placeholder", max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(self._image_response)))
        self.stack.callback(self.claude.close)
        self.stack.callback(self.openai.close)
        self.stack.enter_context(mock.patch.object(banner, "Anthropic", return_value=self.claude))
        self.stack.enter_context(mock.patch.object(email_copy, "Anthropic", return_value=self.claude))
        self.stack.enter_context(mock.patch.object(banner, "OpenAIClient", return_value=self.openai))

    def _claude_response(self, request):
        self.assertEqual(request.url.path, "/v1/messages")
        payload = json.loads(request.read())
        self.anthropic_requests.append(payload)
        if self.anthropic_status != 200:
            return httpx.Response(self.anthropic_status, json={"type": "error", "error": {
                "type": "permission_error", "message": "DO-NOT-LOG-CREDENTIAL"}})
        answer = json.dumps(self.copy) if "You are Reg Reagan" in payload["system"] else self.answer
        return httpx.Response(200, json={
            "id": "msg_offline", "type": "message", "role": "assistant", "model": payload["model"],
            "stop_reason": self.stop_reason, "stop_sequence": None,
            "content": [{"type": "thinking", "thinking": "", "signature": "offline"},
                        {"type": "text", "text": answer}],
            "usage": {"input_tokens": 10, "output_tokens": 20},
        })

    def _image_response(self, request):
        self.assertEqual(request.url.path, "/v1/images/edits")
        self.image_requests.append(request.read())
        if self.image_status != 200:
            return httpx.Response(self.image_status, json={"error": {
                "type": "permission_error", "message": "DO-NOT-LOG-CREDENTIAL"}})
        return httpx.Response(200, json={"created": 0, "data": [
            {"b64_json": base64.b64encode(self.png).decode()}]})

    def _payload(self, openai_key="offline-placeholder"):
        return email_copy.generate_reg_regan_email_payload(
            _predictions(), pd.DataFrame(), "offline-placeholder", None, 0.9,
            openai_api_key=openai_key, context_cards=[],
        )

    def test_current_models_reach_all_consumers_with_thinking_safe_text(self):
        payload = self._payload()
        self.assertEqual(payload["subject"], self.copy["subject"])
        generated = Path(payload["inline_images"][0]["path"])
        self.assertNotEqual(generated, self.static)
        self.assertEqual(generated.read_bytes(), self.png)
        with Image.open(generated) as image:
            image.verify()
        with mock.patch.dict(os.environ, {"FOOTY_TIPPER_LEGACY_NEWS_ENABLED": "true"}), \
             mock.patch.object(news, "_fetch_rss_headlines", return_value="NRL training news"):
            self.assertEqual(news._fetch_nrl_news_context(self.claude), self.answer)
        self.assertEqual(len(self.anthropic_requests), 3)
        for request in self.anthropic_requests:
            self.assertEqual(request["model"], "claude-fable-5-1")
            self.assertNotIn("temperature", request)
            self.assertEqual(request["output_config"], {"effort": "low"})
            self.assertGreater(request["max_tokens"], 4096)
        self.assertEqual(len(self.image_requests), 1)
        self.assertIn(b"gpt-image-2.5-sunburst", self.image_requests[0])
        self.assertIn(b"1536x1024", self.image_requests[0])

    def test_blank_overrides_use_defaults(self):
        with mock.patch.dict(os.environ, {"CLAUDE_MODEL": "  ", "OPENAI_MODEL": "\t"}):
            self._payload()
        self.assertEqual(self.anthropic_requests[0]["model"], "claude-fable-5-1")
        self.assertIn(b"gpt-image-2.5-sunburst", self.image_requests[0])

    def test_rollback_overrides_keep_old_sampling_and_caps(self):
        with mock.patch.dict(os.environ, {"CLAUDE_MODEL": " claude-sonnet-4-6 ", "OPENAI_MODEL": " gpt-image-1.5 "}):
            self._payload()
        prose, scene = self.anthropic_requests
        self.assertEqual(prose["model"], "claude-sonnet-4-6")
        self.assertEqual(prose["temperature"], 0.9)
        self.assertEqual(prose["max_tokens"], 2500)
        self.assertEqual(scene["temperature"], 1.0)
        self.assertEqual(scene["max_tokens"], 150)
        self.assertNotIn("output_config", prose)
        self.assertIn(b"gpt-image-1.5", self.image_requests[0])

    def test_latest_sonnet_override_also_uses_compatible_requests(self):
        with mock.patch.dict(os.environ, {"CLAUDE_MODEL": "claude-sonnet-5-5"}):
            self._payload()
        for request in self.anthropic_requests:
            self.assertEqual(request["model"], "claude-sonnet-5-5")
            self.assertNotIn("temperature", request)

    def test_image_permission_failure_keeps_static_banner_and_hides_error_body(self):
        self.image_status = 403
        payload = self._payload()
        self.assertEqual(payload["subject"], self.copy["subject"])
        self.assertEqual(payload["inline_images"][0]["path"], str(self.static))
        self.assertIn("cid:footy_tipper_email_banner", payload["html_text"])
        self.assertNotIn("DO-NOT-LOG-CREDENTIAL", self.logs.getvalue())

    def test_claude_permission_failure_keeps_deterministic_copy_and_static_banner(self):
        self.anthropic_status = 403
        payload = self._payload()
        expected = email_copy._build_fallback_copy(_predictions(), None)
        self.assertEqual(payload["subject"], expected["subject"])
        self.assertEqual(payload["inline_images"][0]["path"], str(self.static))
        self.assertEqual(self.image_requests, [])
        self.assertNotIn("DO-NOT-LOG-CREDENTIAL", self.logs.getvalue())

    def test_truncated_claude_response_does_not_send_partial_copy_or_call_images(self):
        self.stop_reason = "max_tokens"
        payload = self._payload()
        expected = email_copy._build_fallback_copy(_predictions(), None)
        self.assertEqual(payload["subject"], expected["subject"])
        self.assertEqual(payload["inline_images"][0]["path"], str(self.static))
        self.assertEqual(self.image_requests, [])

    def test_missing_image_key_keeps_static_banner_without_image_request(self):
        payload = self._payload(openai_key=None)
        self.assertEqual(payload["inline_images"][0]["path"], str(self.static))
        self.assertEqual(self.image_requests, [])


if __name__ == "__main__":
    unittest.main()
