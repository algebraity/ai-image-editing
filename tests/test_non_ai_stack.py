"""Preliminary end-to-end tests for the non-AI editing stack.

These tests intentionally exercise the same path a prototype planner will use:
canonical action JSON is parsed into an `ActionBatch`, actions are executed by
the runtime, validation reports and document snapshots are written to a trace,
and exported preview images are left in `tests/artifacts` for manual inspection.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ai_edit_kernel.document.document_state import CanvasSpec, DocumentState
from ai_edit_kernel.runtime.executor import ExecutionContext, Executor
from ai_edit_kernel.runtime.validator import Validator
from ai_edit_kernel.schema.actions import ActionBatch, ActionResult, ActionStatus, SCHEMA_VERSION
from ai_edit_kernel.trace.trace_logger import TraceLogger


ARTIFACT_ROOT = Path(__file__).resolve().parent / "artifacts"


class NonAIStackTests(unittest.TestCase):
    """Exercise document state, actions, executor, validation, export, and traces."""

    maxDiff = None

    def test_01_create_layer_draw_circle(self) -> None:
        """Create a document, add a layer, draw a filled ellipse, and export it."""
        actions = [
            create_layer("action_001", "layer_circle", "circle", color="#00000000"),
            full_canvas_mask("action_002", "mask_full_canvas", 32, 32),
            draw_shape(
                "action_003",
                "layer_circle",
                ellipse([8, 8, 24, 24]),
                write_mask_id="mask_full_canvas",
                fill={"color": "#ff0000"},
            ),
            export_flat("action_004", self.export_path("test_01_create_layer_draw_circle", "final.png")),
        ]
        doc, results, summary = self.run_case("test_01_create_layer_draw_circle", 32, 32, actions)

        preview = doc.flatten_preview()
        self.assert_all_succeeded(results)
        self.assert_color_close(preview[16, 16], [1.0, 0.0, 0.0, 1.0])
        self.assert_color_close(preview[2, 2], [0.0, 0.0, 0.0, 0.0])
        self.assertGreater(int(np.count_nonzero(preview[..., 3] > 0.5)), 170)
        self.assertLess(int(np.count_nonzero(preview[..., 3] > 0.5)), 230)
        self.assert_trace_healthy(summary, expected_results=len(actions), min_snapshots=len(actions) + 1)

    def test_02_rectangle_border_clear_interior(self) -> None:
        """Draw a stroked rectangle and clear its interior with a second mask."""
        actions = [
            create_layer("action_001", "layer_border", "border", color="#00000000"),
            full_canvas_mask("action_002", "mask_full_canvas", 32, 32),
            draw_shape(
                "action_003",
                "layer_border",
                rectangle([0, 0, 32, 32]),
                write_mask_id="mask_full_canvas",
                fill={"color": "#ffffff"},
            ),
            export_flat("action_004", self.export_path("test_02_rectangle_border_clear_interior", "stage_01_white.png")),
            draw_shape(
                "action_005",
                "layer_border",
                rectangle([6, 6, 26, 26]),
                write_mask_id="mask_full_canvas",
                stroke={"color": "#000000", "width": 4},
            ),
            export_flat("action_006", self.export_path("test_02_rectangle_border_clear_interior", "stage_02_border.png")),
            select_rect("action_007", "mask_inner", [10, 10, 22, 22], name="inner clear mask"),
            clear_region("action_008", "layer_border", "mask_inner"),
            export_flat("action_009", self.export_path("test_02_rectangle_border_clear_interior", "final.png")),
        ]
        doc, results, summary = self.run_case("test_02_rectangle_border_clear_interior", 32, 32, actions)

        preview = doc.flatten_preview()
        self.assert_all_succeeded(results)
        self.assert_color_close(preview[8, 8], [0.0, 0.0, 0.0, 1.0])
        self.assert_color_close(preview[16, 16], [0.0, 0.0, 0.0, 0.0])
        self.assert_color_close(preview[2, 2], [1.0, 1.0, 1.0, 1.0])
        self.assert_trace_healthy(summary, expected_results=len(actions), min_snapshots=len(actions) + 1)

    def test_03_multilayer_opacity_compositing(self) -> None:
        """Composite a half-opacity green layer over an opaque blue background."""
        actions = [
            create_layer("action_001", "layer_background", "background", color="#0000ff"),
            export_flat("action_002", self.export_path("test_03_multilayer_opacity_compositing", "stage_01_background.png")),
            create_layer("action_003", "layer_tint", "green tint", color="#00000000", opacity=0.5),
            full_canvas_mask("action_004", "mask_full_canvas", 32, 32),
            draw_shape(
                "action_005",
                "layer_tint",
                rectangle([0, 0, 32, 32]),
                write_mask_id="mask_full_canvas",
                fill={"color": "#00ff00"},
            ),
            export_flat("action_006", self.export_path("test_03_multilayer_opacity_compositing", "final.png")),
        ]
        doc, results, summary = self.run_case("test_03_multilayer_opacity_compositing", 32, 32, actions)

        preview = doc.flatten_preview()
        self.assert_all_succeeded(results)
        self.assert_color_close(preview[16, 16], [0.0, 0.5, 0.5, 1.0], tolerance=0.02)
        self.assertEqual([layer.id for layer in doc.layers], ["layer_background", "layer_tint"])
        self.assert_trace_healthy(summary, expected_results=len(actions), min_snapshots=len(actions) + 1)

    def test_04_union_masks_clear_two_holes(self) -> None:
        """Union two rectangular masks and clear both regions from one layer."""
        actions = [
            create_layer("action_001", "layer_panel", "panel", color="#ffffff"),
            full_canvas_mask("action_002", "mask_full_canvas", 32, 32),
            export_flat("action_003", self.export_path("test_04_union_masks_clear_two_holes", "stage_01_panel.png")),
            select_rect("action_004", "mask_top_left", [2, 2, 10, 10], name="top left hole"),
            select_rect("action_005", "mask_bottom_right", [22, 22, 30, 30], name="bottom right hole"),
            combine_masks("action_006", "mask_two_holes", "union", ["mask_top_left", "mask_bottom_right"]),
            clear_region("action_007", "layer_panel", "mask_two_holes"),
            export_flat("action_008", self.export_path("test_04_union_masks_clear_two_holes", "final.png")),
        ]
        doc, results, summary = self.run_case("test_04_union_masks_clear_two_holes", 32, 32, actions)

        preview = doc.flatten_preview()
        self.assert_all_succeeded(results)
        self.assert_color_close(preview[5, 5], [0.0, 0.0, 0.0, 0.0])
        self.assert_color_close(preview[25, 25], [0.0, 0.0, 0.0, 0.0])
        self.assert_color_close(preview[16, 16], [1.0, 1.0, 1.0, 1.0])
        self.assert_trace_healthy(summary, expected_results=len(actions), min_snapshots=len(actions) + 1)

    def test_05_subtract_mask_draw_ring(self) -> None:
        """Subtract an inner mask from the full canvas and paint only the ring."""
        actions = [
            create_layer("action_001", "layer_ring", "ring", color="#000000"),
            full_canvas_mask("action_002", "mask_full_canvas", 32, 32),
            select_rect("action_003", "mask_inner", [8, 8, 24, 24], name="protected center"),
            combine_masks("action_004", "mask_ring", "subtract", ["mask_full_canvas", "mask_inner"]),
            draw_shape(
                "action_005",
                "layer_ring",
                rectangle([0, 0, 32, 32]),
                write_mask_id="mask_ring",
                fill={"color": "#ff0000"},
            ),
            export_flat("action_006", self.export_path("test_05_subtract_mask_draw_ring", "final.png")),
        ]
        doc, results, summary = self.run_case("test_05_subtract_mask_draw_ring", 32, 32, actions)

        preview = doc.flatten_preview()
        self.assert_all_succeeded(results)
        self.assert_color_close(preview[2, 2], [1.0, 0.0, 0.0, 1.0])
        self.assert_color_close(preview[16, 16], [0.0, 0.0, 0.0, 1.0])
        self.assert_trace_healthy(summary, expected_results=len(actions), min_snapshots=len(actions) + 1)

    def test_06_feather_mask_soft_clear(self) -> None:
        """Feather a rectangular mask and verify the clear operation has soft edges."""
        actions = [
            create_layer("action_001", "layer_soft_clear", "soft clear", color="#ffffff"),
            select_rect("action_002", "mask_rect", [8, 8, 24, 24], name="hard clear mask"),
            feather_mask("action_003", "mask_soft", "mask_rect", radius=2.0),
            clear_region("action_004", "layer_soft_clear", "mask_soft"),
            export_flat("action_005", self.export_path("test_06_feather_mask_soft_clear", "final.png")),
        ]
        doc, results, summary = self.run_case("test_06_feather_mask_soft_clear", 32, 32, actions)

        alpha = doc.get_layer("layer_soft_clear").pixels[..., 3]
        self.assert_all_succeeded(results)
        self.assertLess(float(alpha[16, 16]), 0.05)
        self.assertGreater(float(alpha[2, 2]), 0.95)
        self.assertGreater(int(np.count_nonzero((alpha > 0.1) & (alpha < 0.9))), 0)
        self.assertFalse(doc.get_mask("mask_soft").hard)
        self.assert_trace_healthy(summary, expected_results=len(actions), min_snapshots=len(actions) + 1)

    def test_07_ellipse_write_mask_clips_rectangle(self) -> None:
        """Use an ellipse write mask to clip a full-canvas rectangle fill."""
        actions = [
            create_layer("action_001", "layer_clipped", "clipped", color="#00000000"),
            create_mask_from_shape(
                "action_002",
                "mask_ellipse",
                ellipse([8, 4, 24, 28]),
                name="ellipse write mask",
                kind="write_guard",
            ),
            draw_shape(
                "action_003",
                "layer_clipped",
                rectangle([0, 0, 32, 32]),
                write_mask_id="mask_ellipse",
                fill={"color": "#ff0000"},
            ),
            export_flat("action_004", self.export_path("test_07_ellipse_write_mask_clips_rectangle", "final.png")),
        ]
        doc, results, summary = self.run_case("test_07_ellipse_write_mask_clips_rectangle", 32, 32, actions)

        preview = doc.flatten_preview()
        painted = int(np.count_nonzero(preview[..., 3] > 0.5))
        self.assert_all_succeeded(results)
        self.assert_color_close(preview[16, 16], [1.0, 0.0, 0.0, 1.0])
        self.assert_color_close(preview[0, 0], [0.0, 0.0, 0.0, 0.0])
        self.assertGreater(painted, 250)
        self.assertLess(painted, 330)
        self.assert_trace_healthy(summary, expected_results=len(actions), min_snapshots=len(actions) + 1)

    def test_08_rollback_invalid_draw_preserves_previous_image(self) -> None:
        """A failed draw rolls back cleanly and leaves the prior image intact."""
        case_name = "test_08_rollback_invalid_draw_preserves_previous_image"
        actions = [
            create_layer("action_001", "layer_rollback", "rollback", color="#00000000"),
            full_canvas_mask("action_002", "mask_full_canvas", 32, 32),
            draw_shape(
                "action_003",
                "layer_rollback",
                rectangle([4, 4, 12, 12]),
                write_mask_id="mask_full_canvas",
                fill={"color": "#0000ff"},
            ),
            export_flat("action_004", self.export_path(case_name, "stage_01_before_failure.png")),
            draw_shape(
                "action_005",
                "layer_rollback",
                rectangle([0, 0, 40, 40]),
                write_mask_id="mask_full_canvas",
                fill={"color": "#ff0000"},
            ),
            export_flat("action_006", self.export_path(case_name, "unreached_after_failure.png")),
        ]
        doc, results, summary = self.run_case(case_name, 32, 32, actions)
        final_manual_export = self.export_path(case_name, "final_after_rollback.png")
        save_preview_png(doc, final_manual_export)
        self.append_manual_export(summary, final_manual_export)

        self.assertEqual([result.status for result in results], [ActionStatus.EXECUTED] * 4 + [ActionStatus.ROLLED_BACK])
        self.assertEqual(doc.revision, 3)
        preview = doc.flatten_preview()
        self.assert_color_close(preview[6, 6], [0.0, 0.0, 1.0, 1.0])
        self.assert_color_close(preview[20, 20], [0.0, 0.0, 0.0, 0.0])
        self.assertFalse((self.export_path(case_name, "unreached_after_failure.png")).exists())
        self.assert_trace_healthy(summary, expected_results=len(results), min_snapshots=len(results) + 1, expect_success=False)

    def test_09_trace_reload_and_training_export(self) -> None:
        """Reload a written trace and export a planner-imitation training example."""
        case_name = "test_09_trace_reload_and_training_export"
        actions = [
            create_layer("action_001", "layer_training", "training demo", color="#00000000"),
            full_canvas_mask("action_002", "mask_full_canvas", 32, 32),
            draw_shape(
                "action_003",
                "layer_training",
                rectangle([6, 6, 26, 26]),
                write_mask_id="mask_full_canvas",
                stroke={"color": "#111111", "width": 3},
                fill={"color": "#dddddd"},
            ),
            export_flat("action_004", self.export_path(case_name, "final.png")),
        ]
        doc, results, summary = self.run_case(case_name, 32, 32, actions, task_type="training_export")

        loader = TraceLogger(summary["trace_dir"].parent)
        loaded_session = loader.load_session(summary["trace_dir"])
        training_example = loader.export_training_example(loaded_session)
        dataset_path = summary["test_dir"] / "training_examples.jsonl"
        loader.export_dataset([loaded_session], dataset_path)

        self.assert_all_succeeded(results)
        self.assertEqual(training_example["schema_version"], "ai_edit_training_example.v1")
        self.assertEqual(training_example["source_session_id"], loaded_session.id)
        self.assertEqual(training_example["target"]["action_batch"]["id"], f"batch_{case_name}")
        self.assertEqual(len(training_example["target"]["action_batch"]["actions"]), len(actions))
        self.assertTrue(dataset_path.exists())
        self.assert_color_close(doc.flatten_preview()[16, 16], [0.8667, 0.8667, 0.8667, 1.0], tolerance=0.02)
        self.assert_trace_healthy(summary, expected_results=len(actions), min_snapshots=len(actions) + 1)

    def test_10_active_layer_switching_multistage(self) -> None:
        """Switch active layers across stages while editing explicit layer targets."""
        actions = [
            create_layer("action_001", "layer_background", "background", color="#0000ff"),
            create_layer("action_002", "layer_foreground", "foreground", color="#00000000"),
            full_canvas_mask("action_003", "mask_full_canvas", 32, 32),
            set_active_layer("action_004", "layer_background"),
            draw_shape(
                "action_005",
                "layer_background",
                rectangle([4, 4, 28, 28]),
                write_mask_id="mask_full_canvas",
                fill={"color": "#ffff00"},
            ),
            export_flat("action_006", self.export_path("test_10_active_layer_switching_multistage", "stage_01_background.png")),
            set_active_layer("action_007", "layer_foreground"),
            draw_shape(
                "action_008",
                "layer_foreground",
                ellipse([8, 8, 24, 24]),
                write_mask_id="mask_full_canvas",
                fill={"color": "#ff00ff"},
            ),
            export_flat("action_009", self.export_path("test_10_active_layer_switching_multistage", "final.png")),
        ]
        doc, results, summary = self.run_case("test_10_active_layer_switching_multistage", 32, 32, actions)

        preview = doc.flatten_preview()
        background = doc.get_layer("layer_background").pixels
        foreground = doc.get_layer("layer_foreground").pixels
        self.assert_all_succeeded(results)
        self.assertEqual(doc.active_layer_id, "layer_foreground")
        self.assert_color_close(background[16, 16], [1.0, 1.0, 0.0, 1.0])
        self.assert_color_close(foreground[16, 16], [1.0, 0.0, 1.0, 1.0])
        self.assert_color_close(preview[16, 16], [1.0, 0.0, 1.0, 1.0])
        self.assert_color_close(preview[5, 5], [1.0, 1.0, 0.0, 1.0])
        self.assert_trace_healthy(summary, expected_results=len(actions), min_snapshots=len(actions) + 1)

    def test_11_kiririn_import_frame_eye_background_paint(self) -> None:
        """Edit the supplied Kiririn image with frame, selection, and paint actions."""
        case_name = "test_11_kiririn_import_frame_eye_background_paint"
        source_path = Path(__file__).resolve().parent / "kiririn.png"
        if not source_path.exists():
            source_path = Path(__file__).resolve().parent / "kiririn.jpg"
        self.assertTrue(source_path.exists(), "expected tests/kiririn.png or tests/kiririn.jpg")

        with Image.open(source_path) as image:
            source_width, source_height = image.size

        transparent_margin = 48
        frame_width = 18
        image_x = transparent_margin + frame_width
        image_y = transparent_margin + frame_width
        canvas_width = source_width + 2 * (transparent_margin + frame_width)
        canvas_height = source_height + 2 * (transparent_margin + frame_width)
        frame_outer = [
            transparent_margin,
            transparent_margin,
            transparent_margin + source_width + 2 * frame_width,
            transparent_margin + source_height + 2 * frame_width,
        ]
        left_eye_bbox = offset_bbox([95, 170, 190, 280], image_x, image_y)
        right_eye_bbox = offset_bbox([260, 140, 365, 255], image_x, image_y)
        background_seeds = offset_points(
            [(5, 5), (source_width - 5, 5), (5, 300), (source_width - 5, 300), (5, source_height - 5)],
            image_x,
            image_y,
        )

        actions = [
            full_canvas_mask("action_001", "mask_full_canvas", canvas_width, canvas_height),
            create_layer("action_002", "layer_frame", "purple frame", color="#00000000"),
            draw_shape(
                "action_003",
                "layer_frame",
                rectangle(frame_outer),
                write_mask_id="mask_full_canvas",
                stroke={"color": "#8a2be2", "width": frame_width},
            ),
            import_image_as_layer(
                "action_004",
                "layer_kiririn",
                source_path,
                x=image_x,
                y=image_y,
                name="kiririn source",
            ),
            export_flat("action_005", self.export_path(case_name, "stage_01_imported_with_frame.png")),
            select_color_range(
                "action_006",
                "layer_kiririn",
                "mask_left_eye",
                "#62beb4",
                tolerance=0.36,
                bbox_xyxy=left_eye_bbox,
                name="left iris color range",
            ),
            select_color_range(
                "action_007",
                "layer_kiririn",
                "mask_right_eye",
                "#62beb4",
                tolerance=0.36,
                bbox_xyxy=right_eye_bbox,
                name="right iris color range",
            ),
            combine_masks("action_008", "mask_eyes", "union", ["mask_left_eye", "mask_right_eye"]),
            paint_bucket_fill("action_009", "layer_kiririn", "mask_eyes", "#ff2020"),
            export_flat("action_010", self.export_path(case_name, "stage_02_red_eyes.png")),
            magic_wand_select(
                "action_011",
                "layer_kiririn",
                "mask_original_white_background",
                background_seeds,
                tolerance=0.08,
                name="connected white background",
            ),
            paint_bucket_fill("action_012", "layer_kiririn", "mask_original_white_background", "#ffd6ea"),
            export_flat("action_013", self.export_path(case_name, "final.png")),
        ]
        doc, results, summary = self.run_case(case_name, canvas_width, canvas_height, actions, task_type="kiririn_image_edit")

        preview = doc.flatten_preview()
        kiririn = doc.get_layer("layer_kiririn").pixels
        eyes_mask = doc.get_mask("mask_eyes")
        background_mask = doc.get_mask("mask_original_white_background")
        purple = [0x8A / 255.0, 0x2B / 255.0, 0xE2 / 255.0, 1.0]
        pink = [1.0, 0xD6 / 255.0, 0xEA / 255.0, 1.0]

        self.assert_all_succeeded(results)
        self.assertEqual(preview.shape, (canvas_height, canvas_width, 4))
        self.assert_color_close(preview[10, 10], [0.0, 0.0, 0.0, 0.0])
        self.assert_color_close(preview[transparent_margin + 5, transparent_margin + 5], purple, tolerance=0.02)
        self.assert_color_close(preview[image_y + 20, image_x + 20], pink, tolerance=0.02)
        self.assert_color_close(kiririn[image_y + 257, image_x + 161], [1.0, 0x20 / 255.0, 0x20 / 255.0, 1.0], tolerance=0.02)
        self.assert_color_close(kiririn[image_y + 237, image_x + 297], [1.0, 0x20 / 255.0, 0x20 / 255.0, 1.0], tolerance=0.02)
        self.assertFalse(np.allclose(preview[image_y + 80, image_x + 250], pink, atol=0.03))
        self.assertGreater(eyes_mask.stats().area_pixels, 1500)
        self.assertLess(eyes_mask.stats().area_pixels, 2500)
        self.assertGreater(background_mask.stats().area_pixels, 70000)
        self.assertLess(background_mask.stats().area_pixels, 80000)
        self.assert_trace_healthy(summary, expected_results=len(actions), min_snapshots=len(actions) + 1)

    def run_case(
        self,
        name: str,
        width: int,
        height: int,
        actions: list[dict[str, Any]],
        *,
        task_type: str = "non_ai_stack_test",
    ) -> tuple[DocumentState, list[ActionResult], dict[str, Any]]:
        """Run one action sequence and persist trace/export metadata."""
        test_dir = ARTIFACT_ROOT / name
        exports_dir = test_dir / "exports"
        traces_dir = test_dir / "traces"
        exports_dir.mkdir(parents=True, exist_ok=True)
        traces_dir.mkdir(parents=True, exist_ok=True)

        prompt = f"Test fixture: {name}"
        document = DocumentState(id=f"doc_{name}", canvas=CanvasSpec(width=width, height=height))
        logger = TraceLogger(
            traces_dir,
            metadata={
                "task_type": task_type,
                "source": "non_ai_stack_unittest",
                "split": "test",
                "rights": {
                    "source_image_license": "generated",
                    "user_provided_content": False,
                    "allowed_for_training": False,
                    "contains_personal_data": False,
                },
                "environment": {
                    "platform": "local_test",
                    "python_version": "test-runtime",
                    "deterministic_mode": True,
                    "random_seed": None,
                },
            },
        )
        logger.start_session(prompt, metadata={"test_name": name})
        logger.log_document_snapshot(document, "initial")

        batch = ActionBatch.from_json(
            {
                "schema_version": SCHEMA_VERSION,
                "id": f"batch_{name}",
                "user_prompt": prompt,
                "description": f"Non-AI stack regression case {name}",
                "stop_on_error": True,
                "actions": actions,
                "metadata": {"test_name": name},
            }
        )
        logger.log_action_batch_planned(batch, document)

        executor = Executor(ExecutionContext(trace_sink=logger))
        validator = Validator()
        results: list[ActionResult] = []
        for action in batch.actions:
            result = executor.execute_action(document, action)
            results.append(result)
            logger.log_validation_report(validator.validate_document(document), document, action)
            logger.log_document_snapshot(document, f"after_{action.id}")
            if batch.stop_on_error and not result.succeeded():
                break

        session = logger.end_session()
        session_dir = traces_dir / session.id
        manifest = read_json(session_dir / "manifest.json")
        events = read_jsonl(session_dir / "events.jsonl")
        exports = sorted(exports_dir.glob("*"))
        summary = {
            "test_name": name,
            "test_dir": test_dir,
            "trace_dir": session_dir,
            "manifest_path": session_dir / "manifest.json",
            "events_path": session_dir / "events.jsonl",
            "exports": exports,
            "manual_exports": [],
            "result_statuses": [result.status.value for result in results],
            "results": [result.to_json() for result in results],
            "event_types": [event["type"] for event in events],
            "event_count": len(events),
            "manifest": manifest,
            "final_document": document.snapshot_summary(),
        }
        write_json(test_dir / "summary.json", path_safe(summary))
        return document, results, summary

    def export_path(self, case_name: str, filename: str) -> Path:
        """Return the canonical export path for a test image."""
        return ARTIFACT_ROOT / case_name / "exports" / filename

    def append_manual_export(self, summary: dict[str, Any], path: Path) -> None:
        """Record a manual export generated after the trace session ended."""
        summary["manual_exports"].append(path)
        summary["exports"] = sorted([*summary["exports"], path])
        write_json(summary["test_dir"] / "summary.json", path_safe(summary))

    def assert_all_succeeded(self, results: list[ActionResult]) -> None:
        """Assert every action result is successful."""
        self.assertTrue(results, "case did not execute any actions")
        self.assertTrue(all(result.succeeded() for result in results), [result.to_json() for result in results])

    def assert_trace_healthy(
        self,
        summary: dict[str, Any],
        *,
        expected_results: int,
        min_snapshots: int,
        expect_success: bool = True,
    ) -> None:
        """Check that trace files and core trace events are present."""
        self.assertTrue(summary["manifest_path"].exists())
        self.assertTrue(summary["events_path"].exists())
        self.assertEqual(summary["manifest"]["schema_version"], "ai_edit_trace.v1")
        self.assertEqual(summary["manifest"]["success"], expect_success)
        self.assertEqual(summary["manifest"]["summary"]["action_count"], expected_results)

        event_types = summary["event_types"]
        for event_type in (
            "session_started",
            "user_prompt",
            "action_batch_planned",
            "action_started",
            "action_result",
            "validation_report",
            "document_snapshot",
            "session_ended",
        ):
            self.assertIn(event_type, event_types)
        self.assertEqual(event_types.count("action_result"), expected_results)
        self.assertGreaterEqual(event_types.count("document_snapshot"), min_snapshots)

        for export in summary["exports"]:
            self.assertTrue(export.exists(), export)
            if export.suffix == ".png":
                with Image.open(export) as image:
                    self.assertEqual(image.mode, "RGBA")

    def assert_color_close(self, actual: np.ndarray, expected: list[float], tolerance: float = 0.01) -> None:
        """Assert an RGBA pixel is approximately equal to an expected value."""
        np.testing.assert_allclose(np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32), atol=tolerance)


def create_layer(
    action_id: str,
    layer_id: str,
    name: str,
    *,
    color: str,
    opacity: float = 1.0,
    insert_index: int | None = None,
    set_active: bool = True,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "name": name,
        "kind": "raster",
        "opacity": opacity,
        "blend_mode": "normal",
        "set_active": set_active,
        "color": color,
    }
    if insert_index is not None:
        params["insert_index"] = insert_index
    return action(action_id, "create_layer", params=params, target={"output_layer_id": layer_id})


def import_image_as_layer(
    action_id: str,
    layer_id: str,
    path: Path,
    *,
    x: int,
    y: int,
    name: str,
    opacity: float = 1.0,
    set_active: bool = True,
) -> dict[str, Any]:
    return action(
        action_id,
        "import_image_as_layer",
        params={
            "path": str(path),
            "name": name,
            "x": x,
            "y": y,
            "opacity": opacity,
            "blend_mode": "normal",
            "set_active": set_active,
        },
        target={"output_layer_id": layer_id},
    )


def set_active_layer(action_id: str, layer_id: str) -> dict[str, Any]:
    return action(action_id, "set_active_layer", target={"layer_id": layer_id})


def full_canvas_mask(action_id: str, mask_id: str, width: int, height: int) -> dict[str, Any]:
    return create_mask_from_shape(
        action_id,
        mask_id,
        rectangle([0, 0, width, height]),
        name="full canvas",
        kind="write_guard",
    )


def select_rect(action_id: str, mask_id: str, bbox_xyxy: list[int], *, name: str) -> dict[str, Any]:
    return action(
        action_id,
        "select_rect",
        params={"name": name, "bbox_xyxy": bbox_xyxy, "set_active": True},
        target={"mask_id": mask_id},
    )


def select_color_range(
    action_id: str,
    layer_id: str,
    mask_id: str,
    color: str,
    *,
    tolerance: float,
    bbox_xyxy: list[int],
    name: str,
) -> dict[str, Any]:
    return action(
        action_id,
        "select_color_range",
        params={
            "name": name,
            "color": color,
            "tolerance": tolerance,
            "bbox_xyxy": bbox_xyxy,
            "alpha_min": 0.9,
            "kind": "selection",
            "set_active": True,
        },
        target={"layer_id": layer_id, "mask_id": mask_id},
        preconditions={"required_layer_ids": [layer_id]},
    )


def magic_wand_select(
    action_id: str,
    layer_id: str,
    mask_id: str,
    seed_points: list[list[int]],
    *,
    tolerance: float,
    name: str,
) -> dict[str, Any]:
    return action(
        action_id,
        "magic_wand_select",
        params={
            "name": name,
            "seed_points": seed_points,
            "tolerance": tolerance,
            "alpha_min": 0.9,
            "diagonal": False,
            "kind": "selection",
            "set_active": True,
        },
        target={"layer_id": layer_id, "mask_id": mask_id},
        preconditions={"required_layer_ids": [layer_id]},
    )


def create_mask_from_shape(
    action_id: str,
    mask_id: str,
    shape: dict[str, Any],
    *,
    name: str,
    kind: str = "selection",
) -> dict[str, Any]:
    return action(
        action_id,
        "create_mask_from_shape",
        params={"name": name, "kind": kind, "shape": shape, "set_active": False},
        target={"mask_id": mask_id},
    )


def combine_masks(action_id: str, output_mask_id: str, operation: str, mask_ids: list[str]) -> dict[str, Any]:
    return action(
        action_id,
        "combine_masks",
        params={"operation": operation, "mask_ids": mask_ids, "name": output_mask_id},
        target={"mask_id": output_mask_id},
    )


def feather_mask(action_id: str, output_mask_id: str, source_mask_id: str, *, radius: float) -> dict[str, Any]:
    return action(
        action_id,
        "feather_mask",
        params={"source_mask_id": source_mask_id, "radius": radius, "name": output_mask_id},
        target={"mask_id": output_mask_id},
    )


def draw_shape(
    action_id: str,
    layer_id: str,
    shape: dict[str, Any],
    *,
    write_mask_id: str,
    stroke: dict[str, Any] | None = None,
    fill: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return action(
        action_id,
        "draw_shape",
        params={"shape": shape, "stroke": stroke, "fill": fill},
        target={"layer_id": layer_id},
        write_mask_id=write_mask_id,
        preconditions={"required_mask_ids": [write_mask_id], "required_layer_ids": [layer_id]},
        expected_result={"changed_layer_ids": [layer_id]},
    )


def paint_bucket_fill(action_id: str, layer_id: str, write_mask_id: str, color: str) -> dict[str, Any]:
    return action(
        action_id,
        "paint_bucket_fill",
        params={"color": color, "mode": "replace_rgb_preserve_alpha"},
        target={"layer_id": layer_id},
        write_mask_id=write_mask_id,
        preconditions={"required_mask_ids": [write_mask_id], "required_layer_ids": [layer_id]},
        expected_result={"changed_layer_ids": [layer_id]},
    )


def clear_region(action_id: str, layer_id: str, write_mask_id: str) -> dict[str, Any]:
    return action(
        action_id,
        "clear_region",
        params={"mode": "alpha_to_zero", "preserve_rgb": False},
        target={"layer_id": layer_id},
        write_mask_id=write_mask_id,
        preconditions={"required_mask_ids": [write_mask_id], "required_layer_ids": [layer_id]},
        expected_result={"changed_layer_ids": [layer_id]},
    )


def export_flat(action_id: str, path: Path) -> dict[str, Any]:
    return action(action_id, "export_flat", params={"path": str(path)})


def action(
    action_id: str,
    action_type: str,
    *,
    params: dict[str, Any] | None = None,
    target: dict[str, Any] | None = None,
    write_mask_id: str | None = None,
    preconditions: dict[str, Any] | None = None,
    expected_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": action_id,
        "type": action_type,
        "target": {} if target is None else target,
        "write_mask_id": write_mask_id,
        "params": {} if params is None else params,
        "preconditions": preconditions_with_defaults(preconditions),
        "expected_result": expected_result_with_defaults(expected_result),
        "created_by": "test",
        "metadata": {"fixture": True},
    }


def preconditions_with_defaults(overrides: dict[str, Any] | None) -> dict[str, Any]:
    data = {
        "required_layer_ids": [],
        "required_mask_ids": [],
        "require_active_layer": False,
        "require_active_selection": False,
        "require_unlocked_target_layer": True,
        "require_write_mask": True,
        "allow_hidden_layers": False,
        "custom": {},
    }
    if overrides is not None:
        data.update(overrides)
    return data


def expected_result_with_defaults(overrides: dict[str, Any] | None) -> dict[str, Any]:
    data = {
        "changed_layer_ids": [],
        "created_layer_names": [],
        "created_mask_names": [],
        "geometry_expectations": {},
        "visual_expectations": {},
        "custom": {},
    }
    if overrides is not None:
        data.update(overrides)
    return data


def rectangle(bbox_xyxy: list[int]) -> dict[str, Any]:
    return {"type": "rectangle", "bbox_xyxy": bbox_xyxy, "corner_radius": 0.0}


def ellipse(bbox_xyxy: list[int]) -> dict[str, Any]:
    return {"type": "ellipse", "bbox_xyxy": bbox_xyxy, "corner_radius": 0.0}


def offset_bbox(bbox_xyxy: list[int], dx: int, dy: int) -> list[int]:
    return [bbox_xyxy[0] + dx, bbox_xyxy[1] + dy, bbox_xyxy[2] + dx, bbox_xyxy[3] + dy]


def offset_points(points: list[tuple[int, int]], dx: int, dy: int) -> list[list[int]]:
    return [[x + dx, y + dy] for x, y in points]


def save_preview_png(document: DocumentState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.clip(document.flatten_preview() * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGBA")
    image.save(path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"{path} did not contain a JSON object")
    return data


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    events = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                event = json.loads(stripped)
                if not isinstance(event, dict):
                    raise TypeError(f"{path} contained a non-object event")
                events.append(event)
    return events


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def path_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): path_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [path_safe(item) for item in value]
    if isinstance(value, tuple):
        return [path_safe(item) for item in value]
    return value


if __name__ == "__main__":
    unittest.main()
