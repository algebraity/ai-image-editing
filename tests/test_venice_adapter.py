"""Venice adapter tests that do not call the network."""

from __future__ import annotations

import json
import tempfile
import unittest
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from ai_edit_kernel.diffusion.adapters.venice import VeniceImageBackend
from ai_edit_kernel.diffusion.types import DiffusionJob, DiffusionOptions


class VeniceAdapterTests(unittest.TestCase):
    """Verify local request construction, key loading, and response decoding."""

    def test_generation_request_uses_local_key_and_decodes_b64_response(self) -> None:
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
                return json.dumps({"data": [{"b64_json": tiny_png_b64()}]}).encode("utf-8")

            backend = VeniceImageBackend(
                api_key_path=key_path,
                model="krea-v2-medium",
                timeout=7.5,
                opener=opener,
            )
            job = DiffusionJob(
                job_id="job_venice_adapter",
                operation="txt2img",
                options=DiffusionOptions(
                    prompt="a small stained glass moon over calm water",
                    job={"size": "1024x1024"},
                ),
                canvas_width=2,
                canvas_height=2,
            )

            result = backend.run(job)

            self.assertEqual(result.status, "completed", result.to_json())
            self.assertEqual(result.pixels.shape, (2, 2, 4))
            self.assertEqual(captured["url"], "https://api.venice.ai/api/v1/images/generations")
            self.assertEqual(captured["timeout"], 7.5)
            self.assertEqual(captured["authorization"], "Bearer local_secret_key")
            self.assertEqual(captured["content_type"], "application/json")
            self.assertEqual(captured["payload"]["model"], "krea-v2-medium")
            self.assertEqual(captured["payload"]["prompt"], "a small stained glass moon over calm water")
            self.assertEqual(captured["payload"]["size"], "1024x1024")
            self.assertEqual(captured["payload"]["response_format"], "b64_json")
            self.assertNotIn("api_key", result.metadata["request"])

    def test_region_operation_is_marked_as_generation_candidate(self) -> None:
        backend = VeniceImageBackend(
            api_key="secret",
            opener=lambda request, timeout: json.dumps(
                {"data": [{"b64_json": tiny_png_b64()}]}
            ).encode("utf-8"),
        )
        job = DiffusionJob(
            job_id="job_venice_region",
            operation="inpaint",
            options=DiffusionOptions(prompt="glowing flowers inside the selected area"),
            canvas_width=2,
            canvas_height=2,
        )

        result = backend.run(job)

        self.assertEqual(result.status, "completed", result.to_json())
        self.assertEqual(result.metadata["adapter_mode"], "generation_endpoint_candidate_for_region")

    def test_missing_key_fails_without_calling_network(self) -> None:
        called = False

        def opener(request: urllib.request.Request, timeout: float) -> bytes:
            nonlocal called
            called = True
            return b"{}"

        backend = VeniceImageBackend(api_key_path=None, opener=opener)
        job = DiffusionJob(
            job_id="job_missing_key",
            operation="txt2img",
            options=DiffusionOptions(prompt="a quiet test image"),
            canvas_width=2,
            canvas_height=2,
        )

        result = backend.run(job)

        self.assertEqual(result.status, "failed")
        self.assertFalse(called)
        self.assertIn("Venice API key not found", result.error)


def tiny_png_b64() -> str:
    image = Image.new("RGBA", (2, 2), (32, 64, 128, 255))
    image.putpixel((1, 1), (255, 128, 0, 255))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    import base64

    return base64.b64encode(buffer.getvalue()).decode("ascii")


if __name__ == "__main__":
    unittest.main()
