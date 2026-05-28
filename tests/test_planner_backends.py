"""Planner backend tests that do not call the network."""

from __future__ import annotations

import json
import tempfile
import unittest
import urllib.request
from base64 import b64decode
from pathlib import Path
from typing import Any

from ai_edit_service.assets import AssetStore
from ai_edit_service.planner_backends import VenicePlannerBackend


class VenicePlannerBackendTests(unittest.TestCase):
    """Verify local request construction, key loading, and JSON extraction."""

    def test_chat_request_uses_local_key_and_returns_planner_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            key_path = Path(temp_dir) / "venice-key"
            key_path.write_text("local_secret_key\n", encoding="utf-8")
            captured: dict[str, Any] = {}

            def opener(request: urllib.request.Request, timeout: float) -> bytes:
                captured["url"] = request.full_url
                captured["timeout"] = timeout
                captured["authorization"] = request.get_header("Authorization")
                captured["content_type"] = request.get_header("Content-type")
                captured["payload"] = json.loads(request.data.decode("utf-8"))
                content = {
                    "schema_version": "ai_edit_planner_output.v1",
                    "description": "No edit needed.",
                    "stop_on_error": True,
                    "actions": [{"type": "no_op"}],
                    "metadata": {},
                }
                return json.dumps({"choices": [{"message": {"content": json.dumps(content)}}]}).encode("utf-8")

            backend = VenicePlannerBackend(api_key_path=key_path, timeout=7.5, opener=opener)
            result = backend.plan(minimal_planner_request())

            self.assertIsInstance(result, dict)
            self.assertEqual(result["schema_version"], "ai_edit_planner_output.v1")
            self.assertEqual(result["actions"], [{"type": "no_op"}])
            self.assertEqual(captured["url"], "https://api.venice.ai/api/v1/chat/completions")
            self.assertEqual(captured["timeout"], 7.5)
            self.assertEqual(captured["authorization"], "Bearer local_secret_key")
            self.assertEqual(captured["content_type"], "application/json")
            self.assertEqual(captured["payload"]["model"], "openai-gpt-54")
            self.assertEqual(captured["payload"]["max_tokens"], 16384)
            self.assertEqual(captured["payload"]["messages"][0]["role"], "system")
            self.assertEqual(captured["payload"]["messages"][1]["role"], "user")
            self.assertIn("available_actions", captured["payload"]["messages"][1]["content"])
            self.assertNotIn("api_key", captured["payload"])

    def test_attaches_current_image_asset_as_vision_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            assets = AssetStore(Path(temp_dir) / "assets")
            record = assets.put_bytes(tiny_png_bytes(), suffix=".png", media_type="image/png")
            captured: dict[str, Any] = {}

            def opener(request: urllib.request.Request, timeout: float) -> bytes:
                captured["payload"] = json.loads(request.data.decode("utf-8"))
                content = {
                    "schema_version": "ai_edit_planner_output.v1",
                    "description": "No edit needed.",
                    "stop_on_error": True,
                    "actions": [{"type": "no_op"}],
                    "metadata": {},
                }
                return json.dumps({"choices": [{"message": {"content": json.dumps(content)}}]}).encode("utf-8")

            request = minimal_planner_request()
            request["asset_refs"] = {
                "current_image": f"asset:{record.id}",
                "current_image_role": "attached_vision_input",
            }
            backend = VenicePlannerBackend(api_key="secret", asset_store=assets, opener=opener)

            result = backend.plan(request)

            self.assertIsInstance(result, dict)
            user_content = captured["payload"]["messages"][1]["content"]
            self.assertIsInstance(user_content, list)
            self.assertEqual(user_content[0]["type"], "text")
            self.assertIn("attached_to_this_message", user_content[0]["text"])
            self.assertNotIn(record.id, user_content[0]["text"])
            self.assertEqual(user_content[1]["type"], "image_url")
            self.assertTrue(user_content[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_extracts_json_from_fenced_response(self) -> None:
        def opener(request: urllib.request.Request, timeout: float) -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "```json\n"
                                    "{\"schema_version\":\"ai_edit_planner_output.v1\","
                                    "\"description\":\"No edit needed.\","
                                    "\"stop_on_error\":true,"
                                    "\"actions\":[{\"type\":\"no_op\"}],"
                                    "\"metadata\":{}}\n"
                                    "```"
                                )
                            }
                        }
                    ]
                }
            ).encode("utf-8")

        backend = VenicePlannerBackend(api_key="secret", opener=opener)
        result = backend.plan(minimal_planner_request())

        self.assertIsInstance(result, dict)
        self.assertEqual(result["actions"], [{"type": "no_op"}])

    def test_missing_key_fails_without_calling_network(self) -> None:
        called = False

        def opener(request: urllib.request.Request, timeout: float) -> bytes:
            nonlocal called
            called = True
            return b"{}"

        backend = VenicePlannerBackend(api_key_path=None, opener=opener)

        with self.assertRaisesRegex(RuntimeError, "Venice API key not found"):
            backend.plan(minimal_planner_request())

        self.assertFalse(called)


def minimal_planner_request() -> dict[str, Any]:
    return {
        "schema_version": "ai_edit_planner_request.v1",
        "user_prompt": "do nothing",
        "document_summary": {
            "id": "doc_test",
            "canvas": {"width": 2, "height": 2, "color_space": "srgb"},
            "layers": [{"id": "layer_source", "name": "source"}],
            "masks": [],
            "active_layer_id": "layer_source",
        },
        "available_actions": [{"type": "no_op", "params": {}}],
        "output_contract": {},
        "previous_errors": [],
    }


def tiny_png_bytes() -> bytes:
    return b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNg+M/wHwAEBgIA"
        "E2m3tQAAAABJRU5ErkJggg=="
    )


if __name__ == "__main__":
    unittest.main()
