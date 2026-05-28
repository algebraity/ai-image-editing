"""Service-layer tests that avoid network calls."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from ai_edit_kernel.document.document_state import CanvasSpec, DocumentState
from ai_edit_kernel.document.layer import Layer
from ai_edit_kernel.planning import ActionBatchNormalizer
from ai_edit_service.assets import AssetStore
from ai_edit_service.config import ServiceConfig
from ai_edit_service.kernel_runner import KernelRunner
from ai_edit_service.models import EditRequest
from ai_edit_service.payloads import encode_image_payload


class CapturingPlannerBackend:
    """Planner backend that records the request it receives."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def plan(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(request)
        return {
            "schema_version": "ai_edit_planner_output.v1",
            "description": "No edit needed.",
            "stop_on_error": True,
            "actions": [{"type": "no_op"}],
            "metadata": {},
        }


class AiEditServiceTests(unittest.TestCase):
    """Verify service-kernel glue behavior."""

    def test_service_workspace_defaults_to_artifacts_directory(self) -> None:
        config = ServiceConfig()

        self.assertEqual(config.workspace_dir, Path("artifacts") / "ai_edit_service")
        self.assertEqual(config.resolved_assets_dir(), Path("artifacts") / "ai_edit_service" / "assets")
        self.assertEqual(config.resolved_traces_dir(), Path("artifacts") / "ai_edit_service" / "traces")
        self.assertEqual(config.planner_timeout, 360.0)

    def test_runner_adds_current_image_asset_ref_for_model_planning(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            assets = AssetStore(Path(temp_dir) / "assets")
            backend = CapturingPlannerBackend()
            runner = KernelRunner(planner_backend=backend, asset_store=assets)
            image = np.ones((2, 2, 4), dtype=np.float32)
            image[0, 0] = [0.0, 0.0, 0.0, 1.0]
            request = EditRequest.from_json(
                {
                    "schema_version": "ai_edit_service.edit_request.v1",
                    "prompt": "do nothing",
                    "canvas_width": 2,
                    "canvas_height": 2,
                    "layers": [
                        {
                            "id": "layer_source",
                            "name": "source",
                            "image": encode_image_payload(image).to_json(),
                        }
                    ],
                    "active_layer_id": "layer_source",
                }
            )

            result = runner.run_edit(request)

            self.assertEqual(result.status.value, "succeeded")
            self.assertEqual(len(backend.requests), 1)
            asset_refs = backend.requests[0]["asset_refs"]
            self.assertEqual(asset_refs["current_image_role"], "attached_vision_input")
            self.assertTrue(asset_refs["current_image"].startswith("asset:"))
            current_image_id = asset_refs["current_image"].split(":", 1)[1]
            current_image_record = assets.get_record(current_image_id)
            self.assertEqual(current_image_record.media_type, "image/png")
            self.assertGreater(current_image_record.size_bytes, 0)

    def test_normalizer_uses_planned_active_selection_as_write_mask(self) -> None:
        pixels = np.zeros((4, 4, 4), dtype=np.float32)
        pixels[..., 3] = 1.0
        document = DocumentState(
            id="doc_test",
            canvas=CanvasSpec(width=4, height=4),
            layers=[Layer(id="layer_source", name="source", pixels=pixels)],
            active_layer_id="layer_source",
        )
        planner_output = {
            "schema_version": "ai_edit_planner_output.v1",
            "actions": [
                {
                    "type": "select_color_range",
                    "target": {"mask_id": "mask_eye"},
                    "params": {
                        "name": "eye mask",
                        "color": "#000000",
                        "tolerance": 0.1,
                        "set_active": True,
                    },
                },
                {
                    "type": "colorize",
                    "target": {"layer_id": "layer_source"},
                    "params": {"color": "#0000ff", "amount": 1.0},
                },
            ],
        }

        batch = ActionBatchNormalizer().normalize(planner_output, document, "make eyes blue")

        self.assertEqual(len(batch.actions), 2)
        self.assertEqual(batch.actions[1].type.value, "colorize")
        self.assertEqual(batch.actions[1].write_mask_id, "mask_eye")

    def test_normalizer_rejects_duplicate_generated_mask_ids(self) -> None:
        pixels = np.zeros((4, 4, 4), dtype=np.float32)
        pixels[..., 3] = 1.0
        document = DocumentState(
            id="doc_test",
            canvas=CanvasSpec(width=4, height=4),
            layers=[Layer(id="layer_source", name="source", pixels=pixels)],
            active_layer_id="layer_source",
        )
        planner_output = {
            "schema_version": "ai_edit_planner_output.v1",
            "actions": [
                {
                    "type": "select_ellipse",
                    "target": {"mask_id": "mask_eye"},
                    "params": {"bbox_xyxy": [0, 0, 2, 2]},
                },
                {
                    "type": "feather_mask",
                    "target": {"mask_id": "mask_eye"},
                    "params": {"source_mask_id": "mask_eye", "radius": 1.0},
                },
            ],
        }

        with self.assertRaisesRegex(ValueError, "already exists"):
            ActionBatchNormalizer().normalize(planner_output, document, "make eyes blue")


if __name__ == "__main__":
    unittest.main()
