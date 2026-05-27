"""Diffusion orchestration tests using deterministic local backends."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from ai_edit_kernel.diffusion import FakeDiffusionBackend
from ai_edit_kernel.diffusion.types import DiffusionJob, DiffusionResult
from ai_edit_kernel.document.document_state import CanvasSpec, DocumentState
from ai_edit_kernel.document.layer import Layer
from ai_edit_kernel.document.mask import Mask, MaskKind
from ai_edit_kernel.planning import AIPlanner, StaticPlannerBackend
from ai_edit_kernel.runtime.executor import ExecutionContext, Executor
from ai_edit_kernel.schema.actions import ActionBatch, SCHEMA_VERSION
from ai_edit_kernel.trace.trace_logger import TraceLogger


ARTIFACT_ROOT = Path(__file__).resolve().parent / "artifacts" / "diffusion_pipeline"


class DiffusionPipelineTests(unittest.TestCase):
    """Verify crop-local diffusion execution, clipping, tracing, and planning."""

    def test_inpaint_region_uses_crop_local_job_and_preserves_outside_mask(self) -> None:
        backend = RecordingBackend()
        document = make_document()
        batch = ActionBatch.from_json(
            {
                "schema_version": SCHEMA_VERSION,
                "id": "batch_diffusion_crop_local",
                "actions": [
                    action(
                        "action_inpaint",
                        "inpaint_region",
                        target={"layer_id": "layer_base", "output_layer_id": "layer_unused"},
                        write_mask_id="mask_box",
                        params={"prompt": "blue square", "padding": 1},
                    )
                ],
            }
        )

        result = Executor(ExecutionContext(diffusion_backend=backend)).execute_action(document, batch.actions[0])

        self.assertTrue(result.succeeded(), result.to_json())
        self.assertEqual(len(backend.jobs), 1)
        job = backend.jobs[0]
        self.assertEqual(job.operation, "inpaint")
        self.assertEqual(job.region.mask_bbox_xyxy.as_list(), [3, 2, 7, 5])
        self.assertEqual(job.region.padded_bbox_xyxy.as_list(), [2, 1, 8, 6])
        self.assertEqual(job.source_image.shape, (5, 6, 4))
        self.assertEqual(job.source_mask.shape, (5, 6))
        self.assertEqual(job.target_pixels.shape, (5, 6, 4))

        layer = document.get_layer("layer_base").pixels
        self.assert_rgba_close(layer[3, 4], [0.0, 0.0, 1.0, 1.0])
        self.assert_rgba_close(layer[1, 2], [0.1, 0.1, 0.1, 1.0])
        self.assert_rgba_close(layer[5, 7], [0.1, 0.1, 0.1, 1.0])

    def test_inpaint_new_layer_is_clipped_to_write_mask_even_with_padding(self) -> None:
        document = make_document()
        batch = ActionBatch.from_json(
            {
                "schema_version": SCHEMA_VERSION,
                "id": "batch_diffusion_new_layer_clip",
                "actions": [
                    action(
                        "action_new_layer",
                        "inpaint_region",
                        target={"layer_id": "layer_base", "output_layer_id": "layer_generated"},
                        write_mask_id="mask_box",
                        params={"prompt": "blue square", "mode": "new_layer", "padding": 2},
                    )
                ],
            }
        )

        result = Executor(ExecutionContext(diffusion_backend=FakeDiffusionBackend())).execute_action(document, batch.actions[0])

        self.assertTrue(result.succeeded(), result.to_json())
        generated = document.get_layer("layer_generated").pixels
        self.assert_rgba_close(generated[3, 4], [0.0, 0.0, 1.0, 1.0])
        self.assert_rgba_close(generated[1, 2], [0.0, 0.0, 0.0, 0.0])
        self.assert_rgba_close(generated[5, 7], [0.0, 0.0, 0.0, 0.0])
        self.assert_rgba_close(document.get_layer("layer_base").pixels[3, 4], [0.1, 0.1, 0.1, 1.0])

    def test_trace_records_diffusion_events_and_crop_assets(self) -> None:
        case_dir = ARTIFACT_ROOT / "trace_events"
        traces_dir = case_dir / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        document = make_document()
        logger = TraceLogger(traces_dir, metadata={"task_type": "diffusion_trace_test"})
        logger.start_session("Generate blue content in the selected box.")
        logger.log_document_snapshot(document, "initial")

        batch = ActionBatch.from_json(
            {
                "schema_version": SCHEMA_VERSION,
                "id": "batch_diffusion_trace",
                "actions": [
                    action(
                        "action_trace",
                        "inpaint_region",
                        target={"layer_id": "layer_base", "output_layer_id": "layer_unused"},
                        write_mask_id="mask_box",
                        params={"prompt": "blue square", "padding": 1},
                    )
                ],
            }
        )
        executor = Executor(ExecutionContext(diffusion_backend=FakeDiffusionBackend(), trace_sink=logger))
        results = executor.execute_batch(document, batch)
        logger.log_document_snapshot(document, "final")
        session = logger.end_session()

        self.assertTrue(all(result.succeeded() for result in results), [result.to_json() for result in results])
        events = read_jsonl(traces_dir / session.id / "events.jsonl")
        event_types = [event["type"] for event in events]
        self.assertIn("diffusion_job_started", event_types)
        self.assertIn("diffusion_job_result", event_types)

        started = next(event for event in events if event["type"] == "diffusion_job_started")
        finished = next(event for event in events if event["type"] == "diffusion_job_result")
        self.assertEqual(started["payload"]["params"]["region"]["padded_bbox_xyxy"], [2, 1, 8, 6])
        self.assertIn("source_crop", started["asset_refs"])
        self.assertIn("mask_crop", started["asset_refs"])
        self.assertIn("generated_crop", finished["asset_refs"])
        for relative_path in [*started["asset_refs"].values(), *finished["asset_refs"].values()]:
            self.assertTrue((traces_dir / session.id / relative_path).exists(), relative_path)

    def test_planner_output_executes_diffusion_action_through_kernel(self) -> None:
        document = make_document(with_mask=False)
        backend = StaticPlannerBackend(
            [
                {
                    "schema_version": "ai_edit_planner_output.v1",
                    "description": "Generate blue pixels inside a rectangular write mask.",
                    "actions": [
                        {
                            "type": "create_mask_from_shape",
                            "target": {"mask_id": "mask_prompt_box"},
                            "params": {
                                "name": "prompt box",
                                "kind": "write_guard",
                                "shape": {"type": "rectangle", "bbox_xyxy": [3, 2, 7, 5]},
                                "set_active": False,
                            },
                        },
                        {
                            "type": "inpaint_region",
                            "target": {"layer_id": "layer_base", "output_layer_id": "layer_generated"},
                            "write_mask_id": "mask_prompt_box",
                            "params": {"prompt": "blue square", "padding": 1},
                        },
                    ],
                }
            ]
        )
        planner = AIPlanner(backend=backend)
        executor = Executor(ExecutionContext(diffusion_backend=FakeDiffusionBackend()))

        execution = planner.plan_and_execute("Fill the box with generated blue color.", document, executor)

        self.assertTrue(execution.succeeded(), [result.to_json() for result in execution.action_results])
        self.assertIn("mask_prompt_box", document.masks)
        self.assert_rgba_close(document.get_layer("layer_base").pixels[3, 4], [0.0, 0.0, 1.0, 1.0])
        self.assert_rgba_close(document.get_layer("layer_base").pixels[0, 0], [0.1, 0.1, 0.1, 1.0])

    def assert_rgba_close(self, actual: np.ndarray, expected: list[float], tolerance: float = 0.01) -> None:
        np.testing.assert_allclose(np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32), atol=tolerance)


class RecordingBackend:
    """Fake backend that records typed jobs for assertions."""

    name = "recording_fake"

    def __init__(self) -> None:
        self.jobs: list[DiffusionJob] = []

    def run(self, job: DiffusionJob) -> DiffusionResult:
        self.jobs.append(job)
        pixels = np.zeros((*job.source_image.shape[:2], 4), dtype=np.float32)
        pixels[..., :] = [0.0, 0.0, 1.0, 1.0]
        return DiffusionResult(job_id=job.job_id, status="completed", pixels=pixels, assets={"backend": self.name})


def make_document(*, with_mask: bool = True) -> DocumentState:
    pixels = np.zeros((8, 10, 4), dtype=np.float32)
    pixels[..., :] = [0.1, 0.1, 0.1, 1.0]
    document = DocumentState(
        id="doc_diffusion_pipeline",
        canvas=CanvasSpec(width=10, height=8),
        layers=[Layer(id="layer_base", name="base", pixels=pixels)],
        active_layer_id="layer_base",
    )
    if with_mask:
        mask = np.zeros((8, 10), dtype=np.float32)
        mask[2:5, 3:7] = 1.0
        document.add_mask(Mask(id="mask_box", name="box", data=mask, kind=MaskKind.WRITE_GUARD, hard=True))
    document.validate()
    return document


def action(
    action_id: str,
    action_type: str,
    *,
    params: dict[str, Any] | None = None,
    target: dict[str, Any] | None = None,
    write_mask_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": action_id,
        "type": action_type,
        "params": {} if params is None else params,
        "target": {} if target is None else target,
    }
    if write_mask_id is not None:
        payload["write_mask_id"] = write_mask_id
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    unittest.main()
