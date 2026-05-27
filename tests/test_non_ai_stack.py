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
CUTE_ROOT = Path(__file__).resolve().parent / "cute"
VECTOR_ROOT = Path(__file__).resolve().parent / "vector"


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

    def test_12_layer_management_and_merge_down(self) -> None:
        """Exercise trivial layer metadata actions and GIMP-style merge down."""
        case_name = "test_12_layer_management_and_merge_down"
        actions = [
            create_layer("action_001", "layer_background", "background", color="#0000ff"),
            create_layer("action_002", "layer_foreground", "foreground", color="#ff0000", opacity=0.5),
            rename_layer("action_003", "layer_foreground", "renamed foreground"),
            duplicate_layer("action_004", "layer_foreground", "layer_duplicate", name="temporary duplicate"),
            set_layer_opacity("action_005", "layer_duplicate", 1.0),
            set_layer_visibility("action_006", "layer_duplicate", False),
            reorder_layer("action_007", "layer_duplicate", 1),
            set_blend_mode("action_008", "layer_foreground", "normal"),
            delete_layer("action_009", "layer_duplicate"),
            merge_layers("action_010", mode="down", layer_id="layer_foreground", output_layer_name="merged result"),
            export_flat("action_011", self.export_path(case_name, "final.png")),
        ]
        doc, results, summary = self.run_case(case_name, 16, 16, actions)

        self.assert_all_succeeded(results)
        self.assertEqual([layer.id for layer in doc.layers], ["layer_background"])
        self.assertEqual(doc.layers[0].name, "merged result")
        self.assertEqual(doc.layers[0].opacity, 1.0)
        self.assert_color_close(doc.flatten_preview()[8, 8], [0.5, 0.0, 0.5, 1.0], tolerance=0.02)
        self.assert_trace_healthy(summary, expected_results=len(actions), min_snapshots=len(actions) + 1)

    def test_13_selection_grow_shrink_and_invert(self) -> None:
        """Create, grow, shrink, and invert an elliptical selection."""
        case_name = "test_13_selection_grow_shrink_and_invert"
        actions = [
            select_ellipse("action_001", "mask_ellipse", [4, 4, 12, 12], name="ellipse"),
            grow_mask("action_002", "mask_grown", "mask_ellipse", pixels=2),
            shrink_mask("action_003", "mask_shrunk", "mask_grown", pixels=1),
            invert_mask("action_004", "mask_inverted", "mask_shrunk", set_active=True),
            export_flat("action_005", self.export_path(case_name, "final.png")),
        ]
        doc, results, summary = self.run_case(case_name, 16, 16, actions)

        base_area = doc.get_mask("mask_ellipse").stats().area_pixels
        grown_area = doc.get_mask("mask_grown").stats().area_pixels
        shrunk_area = doc.get_mask("mask_shrunk").stats().area_pixels
        inverted_area = doc.get_mask("mask_inverted").stats().area_pixels
        self.assert_all_succeeded(results)
        self.assertGreater(grown_area, base_area)
        self.assertLess(shrunk_area, grown_area)
        self.assertEqual(inverted_area, 16 * 16 - shrunk_area)
        self.assertEqual(doc.active_selection_mask_id, "mask_inverted")
        self.assert_trace_healthy(summary, expected_results=len(actions), min_snapshots=len(actions) + 1)

    def test_14_resize_canvas_center_and_document_crop(self) -> None:
        """Resize around the center, then crop the whole document back down."""
        case_name = "test_14_resize_canvas_center_and_document_crop"
        actions = [
            create_layer("action_001", "layer_square", "square", color="#00000000"),
            full_canvas_mask("action_002", "mask_full_canvas", 4, 4),
            draw_shape(
                "action_003",
                "layer_square",
                rectangle([1, 1, 3, 3]),
                write_mask_id="mask_full_canvas",
                fill={"color": "#ff0000"},
            ),
            select_rect("action_004", "mask_square", [1, 1, 3, 3], name="square mask"),
            resize_canvas("action_005", 8, 6),
            export_flat("action_006", self.export_path(case_name, "stage_01_resized.png")),
            crop("action_007", [2, 1, 6, 5], scope="document"),
            export_flat("action_008", self.export_path(case_name, "final.png")),
        ]
        doc, results, summary = self.run_case(case_name, 4, 4, actions)

        self.assert_all_succeeded(results)
        self.assertEqual((doc.canvas.width, doc.canvas.height), (4, 4))
        self.assertEqual(doc.get_mask("mask_square").data.shape, (4, 4))
        self.assert_color_close(doc.flatten_preview()[1, 1], [1.0, 0.0, 0.0, 1.0])
        self.assert_color_close(doc.flatten_preview()[0, 0], [0.0, 0.0, 0.0, 0.0])
        self.assert_trace_healthy(summary, expected_results=len(actions), min_snapshots=len(actions) + 1)

    def test_15_targeted_layer_and_mask_crop(self) -> None:
        """Crop one layer and one mask while preserving the document canvas."""
        case_name = "test_15_targeted_layer_and_mask_crop"
        actions = [
            create_layer("action_001", "layer_panel", "panel", color="#ffff00"),
            full_canvas_mask("action_002", "mask_full_canvas", 8, 8),
            crop("action_003", [2, 2, 6, 6], scope="layer", layer_id="layer_panel"),
            crop("action_004", [1, 1, 7, 7], scope="mask", mask_id="mask_full_canvas"),
            export_flat("action_005", self.export_path(case_name, "final.png")),
        ]
        doc, results, summary = self.run_case(case_name, 8, 8, actions)

        layer = doc.get_layer("layer_panel")
        mask = doc.get_mask("mask_full_canvas")
        self.assert_all_succeeded(results)
        self.assertEqual((doc.canvas.width, doc.canvas.height), (8, 8))
        self.assert_color_close(layer.pixels[0, 0], [0.0, 0.0, 0.0, 0.0])
        self.assert_color_close(layer.pixels[3, 3], [1.0, 1.0, 0.0, 1.0])
        self.assertEqual(float(mask.data[0, 0]), 0.0)
        self.assertEqual(float(mask.data[3, 3]), 1.0)
        self.assert_trace_healthy(summary, expected_results=len(actions), min_snapshots=len(actions) + 1)

    def test_16_blur_region_rgb_and_alpha_channels(self) -> None:
        """Blur RGB separately from alpha through explicit write masks."""
        case_name = "test_16_blur_region_rgb_and_alpha_channels"
        actions = [
            full_canvas_mask("action_001", "mask_full_canvas", 9, 9),
            create_layer("action_002", "layer_rgb", "rgb blur", color="#000000"),
            draw_shape(
                "action_003",
                "layer_rgb",
                rectangle([4, 0, 5, 9]),
                write_mask_id="mask_full_canvas",
                fill={"color": "#ffffff"},
            ),
            blur_region("action_004", "layer_rgb", "mask_full_canvas", radius=1.0, channels="rgb"),
            create_layer("action_005", "layer_alpha", "alpha blur", color="#00000000"),
            draw_shape(
                "action_006",
                "layer_alpha",
                rectangle([4, 0, 5, 9]),
                write_mask_id="mask_full_canvas",
                fill={"color": "#ffffff"},
            ),
            blur_region("action_007", "layer_alpha", "mask_full_canvas", radius=1.0, channels="alpha"),
            export_flat("action_008", self.export_path(case_name, "final.png")),
        ]
        doc, results, summary = self.run_case(case_name, 9, 9, actions)

        rgb_layer = doc.get_layer("layer_rgb").pixels
        alpha_layer = doc.get_layer("layer_alpha").pixels
        self.assert_all_succeeded(results)
        self.assertGreater(float(rgb_layer[4, 3, 0]), 0.0)
        self.assertEqual(float(rgb_layer[4, 3, 3]), 1.0)
        self.assertGreater(float(alpha_layer[4, 3, 3]), 0.0)
        self.assertEqual(float(alpha_layer[4, 3, 0]), 0.0)
        self.assert_trace_healthy(summary, expected_results=len(actions), min_snapshots=len(actions) + 1)

    def test_17_merge_visible_and_flatten(self) -> None:
        """Merge visible layers while preserving hidden layers, then flatten."""
        case_name = "test_17_merge_visible_and_flatten"
        actions = [
            create_layer("action_001", "layer_background", "background", color="#00ff00"),
            create_layer("action_002", "layer_top", "top", color="#ff0000", opacity=0.5),
            create_layer("action_003", "layer_hidden", "hidden", color="#0000ff"),
            set_layer_visibility("action_004", "layer_hidden", False),
            merge_layers("action_005", mode="visible", output_layer_id="layer_merged", output_layer_name="merged visible"),
            export_flat("action_006", self.export_path(case_name, "stage_01_merged_visible.png")),
            merge_layers("action_007", mode="flatten", output_layer_id="layer_flat", output_layer_name="flat"),
            export_flat("action_008", self.export_path(case_name, "final.png")),
        ]
        doc, results, summary = self.run_case(case_name, 8, 8, actions)

        self.assert_all_succeeded(results)
        self.assertEqual([layer.id for layer in doc.layers], ["layer_flat"])
        self.assert_color_close(doc.flatten_preview()[4, 4], [0.5, 0.5, 0.0, 1.0], tolerance=0.02)
        self.assert_trace_healthy(summary, expected_results=len(actions), min_snapshots=len(actions) + 1)

    def test_18_resize_canvas_with_cute1(self) -> None:
        """Resize a document containing a real imported image around the center."""
        case_name = "test_18_resize_canvas_with_cute1"
        fixture = cute_fixture(1)
        resized_width = fixture["canvas_width"] + 32
        resized_height = fixture["canvas_height"] + 16
        actions = [
            import_image_as_layer("action_001", "layer_cute1", fixture["path"], x=fixture["x"], y=fixture["y"], name="cute1"),
            resize_canvas("action_002", resized_width, resized_height),
            export_flat("action_003", self.export_path(case_name, "final.png")),
        ]
        doc, results, summary = self.run_case(case_name, fixture["canvas_width"], fixture["canvas_height"], actions)

        self.assert_all_succeeded(results)
        self.assertEqual((doc.canvas.width, doc.canvas.height), (resized_width, resized_height))
        self.assertEqual(doc.get_layer("layer_cute1").pixels.shape, (resized_height, resized_width, 4))
        self.assertGreater(float(doc.get_layer("layer_cute1").pixels[resized_height // 2, resized_width // 2, 3]), 0.9)
        self.assert_color_close(doc.get_layer("layer_cute1").pixels[0, 0], [0.0, 0.0, 0.0, 0.0])
        self.assert_trace_healthy(summary, expected_results=len(actions), min_snapshots=len(actions) + 1)

    def test_19_crop_document_with_cute2(self) -> None:
        """Crop the whole document down to an imported cute image region."""
        case_name = "test_19_crop_document_with_cute2"
        fixture = cute_fixture(2)
        actions = [
            import_image_as_layer("action_001", "layer_cute2", fixture["path"], x=fixture["x"], y=fixture["y"], name="cute2"),
            crop("action_002", fixture["image_bbox"], scope="document"),
            export_flat("action_003", self.export_path(case_name, "final.png")),
        ]
        doc, results, summary = self.run_case(case_name, fixture["canvas_width"], fixture["canvas_height"], actions)

        self.assert_all_succeeded(results)
        self.assertEqual((doc.canvas.width, doc.canvas.height), (fixture["image_width"], fixture["image_height"]))
        self.assertEqual(doc.get_layer("layer_cute2").pixels.shape, (fixture["image_height"], fixture["image_width"], 4))
        self.assertGreater(float(doc.get_layer("layer_cute2").pixels[fixture["image_height"] // 2, fixture["image_width"] // 2, 3]), 0.9)
        self.assert_trace_healthy(summary, expected_results=len(actions), min_snapshots=len(actions) + 1)

    def test_20_crop_layer_with_cute3(self) -> None:
        """Clear pixels outside a layer crop without changing the canvas size."""
        case_name = "test_20_crop_layer_with_cute3"
        fixture = cute_fixture(3)
        crop_bbox = cute_relative_bbox(3, 0.25, 0.25, 0.75, 0.75)
        outside_x, outside_y = cute_relative_point(3, 0.1, 0.1)
        center_x, center_y = fixture["center"]
        actions = [
            import_image_as_layer("action_001", "layer_cute3", fixture["path"], x=fixture["x"], y=fixture["y"], name="cute3"),
            crop("action_002", crop_bbox, scope="layer", layer_id="layer_cute3"),
            export_flat("action_003", self.export_path(case_name, "final.png")),
        ]
        doc, results, summary = self.run_case(case_name, fixture["canvas_width"], fixture["canvas_height"], actions)

        layer = doc.get_layer("layer_cute3")
        self.assert_all_succeeded(results)
        self.assertEqual((doc.canvas.width, doc.canvas.height), (fixture["canvas_width"], fixture["canvas_height"]))
        self.assert_color_close(layer.pixels[outside_y, outside_x], [0.0, 0.0, 0.0, 0.0])
        self.assertGreater(float(layer.pixels[center_y, center_x, 3]), 0.9)
        self.assert_trace_healthy(summary, expected_results=len(actions), min_snapshots=len(actions) + 1)

    def test_21_crop_mask_with_cute4(self) -> None:
        """Crop a write mask and use it to constrain painting on an imported image."""
        case_name = "test_21_crop_mask_with_cute4"
        fixture = cute_fixture(4)
        crop_bbox = cute_relative_bbox(4, 0.25, 0.25, 0.75, 0.75)
        outside_x, outside_y = cute_relative_point(4, 0.1, 0.1)
        center_x, center_y = fixture["center"]
        actions = [
            import_image_as_layer("action_001", "layer_cute4", fixture["path"], x=fixture["x"], y=fixture["y"], name="cute4"),
            full_canvas_mask("action_002", "mask_cropped_write", fixture["canvas_width"], fixture["canvas_height"]),
            crop("action_003", crop_bbox, scope="mask", mask_id="mask_cropped_write"),
            paint_bucket_fill("action_004", "layer_cute4", "mask_cropped_write", "#ff00ff"),
            export_flat("action_005", self.export_path(case_name, "final.png")),
        ]
        doc, results, summary = self.run_case(case_name, fixture["canvas_width"], fixture["canvas_height"], actions)

        mask = doc.get_mask("mask_cropped_write")
        self.assert_all_succeeded(results)
        self.assertEqual(float(mask.data[outside_y, outside_x]), 0.0)
        self.assertEqual(float(mask.data[center_y, center_x]), 1.0)
        self.assert_color_close(doc.get_layer("layer_cute4").pixels[center_y, center_x, :3], [1.0, 0.0, 1.0], tolerance=0.01)
        self.assert_trace_healthy(summary, expected_results=len(actions), min_snapshots=len(actions) + 1)

    def test_22_delete_layer_with_cute5(self) -> None:
        """Delete an imported image layer from a document stack."""
        case_name = "test_22_delete_layer_with_cute5"
        fixture = cute_fixture(5)
        center_x, center_y = fixture["center"]
        actions = [
            create_layer("action_001", "layer_background", "background", color="#112233"),
            import_image_as_layer("action_002", "layer_cute5", fixture["path"], x=fixture["x"], y=fixture["y"], name="cute5"),
            delete_layer("action_003", "layer_cute5"),
            export_flat("action_004", self.export_path(case_name, "final.png")),
        ]
        doc, results, summary = self.run_case(case_name, fixture["canvas_width"], fixture["canvas_height"], actions)

        self.assert_all_succeeded(results)
        self.assertEqual([layer.id for layer in doc.layers], ["layer_background"])
        self.assert_color_close(doc.flatten_preview()[center_y, center_x], [0x11 / 255.0, 0x22 / 255.0, 0x33 / 255.0, 1.0])
        self.assert_trace_healthy(summary, expected_results=len(actions), min_snapshots=len(actions) + 1)

    def test_23_duplicate_layer_with_cute1(self) -> None:
        """Duplicate an imported layer as an independent deep copy."""
        case_name = "test_23_duplicate_layer_with_cute1"
        fixture = cute_fixture(1)
        actions = [
            import_image_as_layer("action_001", "layer_cute1", fixture["path"], x=fixture["x"], y=fixture["y"], name="cute1"),
            duplicate_layer("action_002", "layer_cute1", "layer_cute1_copy", name="cute1 copy"),
            export_flat("action_003", self.export_path(case_name, "final.png")),
        ]
        doc, results, summary = self.run_case(case_name, fixture["canvas_width"], fixture["canvas_height"], actions)

        source = doc.get_layer("layer_cute1")
        duplicate = doc.get_layer("layer_cute1_copy")
        self.assert_all_succeeded(results)
        self.assertEqual([layer.id for layer in doc.layers], ["layer_cute1", "layer_cute1_copy"])
        np.testing.assert_allclose(source.pixels, duplicate.pixels)
        self.assertFalse(np.shares_memory(source.pixels, duplicate.pixels))
        self.assert_trace_healthy(summary, expected_results=len(actions), min_snapshots=len(actions) + 1)

    def test_24_rename_layer_with_cute2(self) -> None:
        """Rename an imported layer while keeping its stable ID."""
        case_name = "test_24_rename_layer_with_cute2"
        fixture = cute_fixture(2)
        actions = [
            import_image_as_layer("action_001", "layer_cute2", fixture["path"], x=fixture["x"], y=fixture["y"], name="cute2"),
            rename_layer("action_002", "layer_cute2", "renamed cute2"),
            export_flat("action_003", self.export_path(case_name, "final.png")),
        ]
        doc, results, summary = self.run_case(case_name, fixture["canvas_width"], fixture["canvas_height"], actions)

        self.assert_all_succeeded(results)
        self.assertEqual(doc.get_layer("layer_cute2").name, "renamed cute2")
        self.assertEqual([layer.id for layer in doc.layers], ["layer_cute2"])
        self.assert_trace_healthy(summary, expected_results=len(actions), min_snapshots=len(actions) + 1)

    def test_25_reorder_layer_with_cute3(self) -> None:
        """Move an imported layer below an opaque layer and verify stack order."""
        case_name = "test_25_reorder_layer_with_cute3"
        fixture = cute_fixture(3)
        center_x, center_y = fixture["center"]
        actions = [
            create_layer("action_001", "layer_cover", "cover", color="#000000"),
            import_image_as_layer("action_002", "layer_cute3", fixture["path"], x=fixture["x"], y=fixture["y"], name="cute3"),
            reorder_layer("action_003", "layer_cute3", 0),
            export_flat("action_004", self.export_path(case_name, "final.png")),
        ]
        doc, results, summary = self.run_case(case_name, fixture["canvas_width"], fixture["canvas_height"], actions)

        self.assert_all_succeeded(results)
        self.assertEqual([layer.id for layer in doc.layers], ["layer_cute3", "layer_cover"])
        self.assert_color_close(doc.flatten_preview()[center_y, center_x], [0.0, 0.0, 0.0, 1.0])
        self.assert_trace_healthy(summary, expected_results=len(actions), min_snapshots=len(actions) + 1)

    def test_26_set_layer_visibility_with_cute4(self) -> None:
        """Hide an imported layer and verify it no longer contributes to preview."""
        case_name = "test_26_set_layer_visibility_with_cute4"
        fixture = cute_fixture(4)
        actions = [
            import_image_as_layer("action_001", "layer_cute4", fixture["path"], x=fixture["x"], y=fixture["y"], name="cute4"),
            set_layer_visibility("action_002", "layer_cute4", False),
            export_flat("action_003", self.export_path(case_name, "final.png")),
        ]
        doc, results, summary = self.run_case(case_name, fixture["canvas_width"], fixture["canvas_height"], actions)

        self.assert_all_succeeded(results)
        self.assertFalse(doc.get_layer("layer_cute4").visible)
        self.assertEqual(int(np.count_nonzero(doc.flatten_preview()[..., 3] > 0.0)), 0)
        self.assert_trace_healthy(summary, expected_results=len(actions), min_snapshots=len(actions) + 1)

    def test_27_set_layer_opacity_with_cute5(self) -> None:
        """Set opacity on an imported image layer and verify source-over output."""
        case_name = "test_27_set_layer_opacity_with_cute5"
        fixture = cute_fixture(5)
        center_x, center_y = fixture["center"]
        actions = [
            create_layer("action_001", "layer_background", "background", color="#0000ff"),
            import_image_as_layer("action_002", "layer_cute5", fixture["path"], x=fixture["x"], y=fixture["y"], name="cute5"),
            set_layer_opacity("action_003", "layer_cute5", 0.25),
            export_flat("action_004", self.export_path(case_name, "final.png")),
        ]
        doc, results, summary = self.run_case(case_name, fixture["canvas_width"], fixture["canvas_height"], actions)

        layer = doc.get_layer("layer_cute5")
        source_rgb = layer.pixels[center_y, center_x, :3]
        expected = np.array([source_rgb[0] * 0.25, source_rgb[1] * 0.25, source_rgb[2] * 0.25 + 0.75, 1.0], dtype=np.float32)
        self.assert_all_succeeded(results)
        self.assertEqual(layer.opacity, 0.25)
        self.assert_color_close(doc.flatten_preview()[center_y, center_x], expected.tolist(), tolerance=0.02)
        self.assert_trace_healthy(summary, expected_results=len(actions), min_snapshots=len(actions) + 1)

    def test_28_set_blend_mode_with_cute1(self) -> None:
        """Apply a supported blend-mode metadata update to an imported layer."""
        case_name = "test_28_set_blend_mode_with_cute1"
        fixture = cute_fixture(1)
        center_x, center_y = fixture["center"]
        actions = [
            import_image_as_layer("action_001", "layer_cute1", fixture["path"], x=fixture["x"], y=fixture["y"], name="cute1"),
            set_blend_mode("action_002", "layer_cute1", "normal"),
            export_flat("action_003", self.export_path(case_name, "final.png")),
        ]
        doc, results, summary = self.run_case(case_name, fixture["canvas_width"], fixture["canvas_height"], actions)

        self.assert_all_succeeded(results)
        self.assertEqual(doc.get_layer("layer_cute1").blend_mode.value, "normal")
        self.assertGreater(float(doc.flatten_preview()[center_y, center_x, 3]), 0.9)
        self.assert_trace_healthy(summary, expected_results=len(actions), min_snapshots=len(actions) + 1)

    def test_29_merge_down_with_cute2(self) -> None:
        """Merge an imported image layer down into an opaque background."""
        case_name = "test_29_merge_down_with_cute2"
        fixture = cute_fixture(2)
        center_x, center_y = fixture["center"]
        actions = [
            create_layer("action_001", "layer_background", "background", color="#224466"),
            import_image_as_layer("action_002", "layer_cute2", fixture["path"], x=fixture["x"], y=fixture["y"], name="cute2", opacity=0.5),
            merge_layers("action_003", mode="down", layer_id="layer_cute2", output_layer_name="cute2 merged down"),
            export_flat("action_004", self.export_path(case_name, "final.png")),
        ]
        doc, results, summary = self.run_case(case_name, fixture["canvas_width"], fixture["canvas_height"], actions)

        self.assert_all_succeeded(results)
        self.assertEqual([layer.id for layer in doc.layers], ["layer_background"])
        self.assertEqual(doc.layers[0].name, "cute2 merged down")
        self.assertGreater(float(doc.flatten_preview()[center_y, center_x, 3]), 0.9)
        self.assert_trace_healthy(summary, expected_results=len(actions), min_snapshots=len(actions) + 1)

    def test_30_merge_visible_with_cute3(self) -> None:
        """Merge visible imported image layers while preserving a hidden layer."""
        case_name = "test_30_merge_visible_with_cute3"
        fixture = cute_fixture(3)
        center_x, center_y = fixture["center"]
        actions = [
            import_image_as_layer("action_001", "layer_cute3_a", fixture["path"], x=fixture["x"], y=fixture["y"], name="cute3 a"),
            import_image_as_layer("action_002", "layer_cute3_hidden", fixture["path"], x=fixture["x"], y=fixture["y"], name="cute3 hidden"),
            set_layer_visibility("action_003", "layer_cute3_hidden", False),
            create_layer("action_004", "layer_overlay", "overlay", color="#ff000080", opacity=0.5),
            merge_layers("action_005", mode="visible", output_layer_id="layer_merged_visible", output_layer_name="merged visible cute3"),
            export_flat("action_006", self.export_path(case_name, "final.png")),
        ]
        doc, results, summary = self.run_case(case_name, fixture["canvas_width"], fixture["canvas_height"], actions)

        self.assert_all_succeeded(results)
        self.assertEqual([layer.id for layer in doc.layers], ["layer_merged_visible", "layer_cute3_hidden"])
        self.assertFalse(doc.get_layer("layer_cute3_hidden").visible)
        self.assertGreater(float(doc.get_layer("layer_merged_visible").pixels[center_y, center_x, 3]), 0.9)
        self.assert_trace_healthy(summary, expected_results=len(actions), min_snapshots=len(actions) + 1)

    def test_31_merge_selected_with_cute4(self) -> None:
        """Merge explicitly selected imported layers in stack order."""
        case_name = "test_31_merge_selected_with_cute4"
        fixture = cute_fixture(4)
        center_x, center_y = fixture["center"]
        actions = [
            import_image_as_layer("action_001", "layer_cute4_a", fixture["path"], x=fixture["x"], y=fixture["y"], name="cute4 a"),
            import_image_as_layer("action_002", "layer_cute4_b", fixture["path"], x=fixture["x"], y=fixture["y"], name="cute4 b", opacity=0.5),
            create_layer("action_003", "layer_unselected", "unselected", color="#00000000"),
            merge_layers(
                "action_004",
                mode="selected",
                output_layer_id="layer_selected_merge",
                output_layer_name="selected merge cute4",
                layer_ids=["layer_cute4_a", "layer_cute4_b"],
            ),
            export_flat("action_005", self.export_path(case_name, "final.png")),
        ]
        doc, results, summary = self.run_case(case_name, fixture["canvas_width"], fixture["canvas_height"], actions)

        self.assert_all_succeeded(results)
        self.assertEqual([layer.id for layer in doc.layers], ["layer_selected_merge", "layer_unselected"])
        self.assertGreater(float(doc.get_layer("layer_selected_merge").pixels[center_y, center_x, 3]), 0.9)
        self.assert_trace_healthy(summary, expected_results=len(actions), min_snapshots=len(actions) + 1)

    def test_32_flatten_with_cute5(self) -> None:
        """Flatten visible imported layers and discard hidden layers."""
        case_name = "test_32_flatten_with_cute5"
        fixture = cute_fixture(5)
        actions = [
            import_image_as_layer("action_001", "layer_cute5", fixture["path"], x=fixture["x"], y=fixture["y"], name="cute5"),
            import_image_as_layer("action_002", "layer_hidden", fixture["path"], x=fixture["x"], y=fixture["y"], name="hidden cute5"),
            set_layer_visibility("action_003", "layer_hidden", False),
            merge_layers("action_004", mode="flatten", output_layer_id="layer_flat", output_layer_name="flat cute5"),
            export_flat("action_005", self.export_path(case_name, "final.png")),
        ]
        doc, results, summary = self.run_case(case_name, fixture["canvas_width"], fixture["canvas_height"], actions)

        self.assert_all_succeeded(results)
        self.assertEqual([layer.id for layer in doc.layers], ["layer_flat"])
        self.assertTrue(bool(np.all(doc.get_layer("layer_flat").pixels[..., 3] == 1.0)))
        self.assert_trace_healthy(summary, expected_results=len(actions), min_snapshots=len(actions) + 1)

    def test_33_select_ellipse_with_cute1(self) -> None:
        """Use an elliptical selection as a paint mask on an imported image."""
        case_name = "test_33_select_ellipse_with_cute1"
        fixture = cute_fixture(1)
        center_x, center_y = fixture["center"]
        mask_bbox = cute_relative_bbox(1, 0.25, 0.2, 0.75, 0.8)
        actions = [
            import_image_as_layer("action_001", "layer_cute1", fixture["path"], x=fixture["x"], y=fixture["y"], name="cute1"),
            select_ellipse("action_002", "mask_ellipse", mask_bbox, name="ellipse paint region"),
            paint_bucket_fill("action_003", "layer_cute1", "mask_ellipse", "#00ffff"),
            export_flat("action_004", self.export_path(case_name, "final.png")),
        ]
        doc, results, summary = self.run_case(case_name, fixture["canvas_width"], fixture["canvas_height"], actions)

        self.assert_all_succeeded(results)
        self.assertGreater(doc.get_mask("mask_ellipse").stats().area_pixels, 200000)
        self.assert_color_close(doc.get_layer("layer_cute1").pixels[center_y, center_x, :3], [0.0, 1.0, 1.0], tolerance=0.01)
        self.assert_trace_healthy(summary, expected_results=len(actions), min_snapshots=len(actions) + 1)

    def test_34_grow_mask_with_cute2(self) -> None:
        """Grow a rectangular selection and paint the expanded area."""
        case_name = "test_34_grow_mask_with_cute2"
        fixture = cute_fixture(2)
        mask_bbox = cute_relative_bbox(2, 0.4, 0.4, 0.6, 0.6)
        paint_x = (mask_bbox[0] + mask_bbox[2]) // 2
        paint_y = mask_bbox[1] - 6
        actions = [
            import_image_as_layer("action_001", "layer_cute2", fixture["path"], x=fixture["x"], y=fixture["y"], name="cute2"),
            select_rect("action_002", "mask_rect", mask_bbox, name="base rect"),
            grow_mask("action_003", "mask_grown", "mask_rect", pixels=12),
            paint_bucket_fill("action_004", "layer_cute2", "mask_grown", "#ffff00"),
            export_flat("action_005", self.export_path(case_name, "final.png")),
        ]
        doc, results, summary = self.run_case(case_name, fixture["canvas_width"], fixture["canvas_height"], actions)

        self.assert_all_succeeded(results)
        self.assertGreater(doc.get_mask("mask_grown").stats().area_pixels, doc.get_mask("mask_rect").stats().area_pixels)
        self.assert_color_close(doc.get_layer("layer_cute2").pixels[paint_y, paint_x, :3], [1.0, 1.0, 0.0], tolerance=0.01)
        self.assert_trace_healthy(summary, expected_results=len(actions), min_snapshots=len(actions) + 1)

    def test_35_shrink_mask_with_cute3(self) -> None:
        """Shrink a selection before painting an imported image."""
        case_name = "test_35_shrink_mask_with_cute3"
        fixture = cute_fixture(3)
        center_x, center_y = fixture["center"]
        mask_bbox = cute_relative_bbox(3, 0.3, 0.3, 0.7, 0.7)
        edge_x = (mask_bbox[0] + mask_bbox[2]) // 2
        edge_y = mask_bbox[1] + 4
        actions = [
            import_image_as_layer("action_001", "layer_cute3", fixture["path"], x=fixture["x"], y=fixture["y"], name="cute3"),
            select_rect("action_002", "mask_rect", mask_bbox, name="base rect"),
            shrink_mask("action_003", "mask_shrunk", "mask_rect", pixels=12),
            paint_bucket_fill("action_004", "layer_cute3", "mask_shrunk", "#ff8800"),
            export_flat("action_005", self.export_path(case_name, "final.png")),
        ]
        doc, results, summary = self.run_case(case_name, fixture["canvas_width"], fixture["canvas_height"], actions)

        self.assert_all_succeeded(results)
        self.assertLess(doc.get_mask("mask_shrunk").stats().area_pixels, doc.get_mask("mask_rect").stats().area_pixels)
        self.assertEqual(float(doc.get_mask("mask_shrunk").data[edge_y, edge_x]), 0.0)
        self.assert_color_close(doc.get_layer("layer_cute3").pixels[center_y, center_x, :3], [1.0, 0x88 / 255.0, 0.0], tolerance=0.01)
        self.assert_trace_healthy(summary, expected_results=len(actions), min_snapshots=len(actions) + 1)

    def test_36_invert_mask_with_cute4(self) -> None:
        """Invert a central selection and clear everything outside it."""
        case_name = "test_36_invert_mask_with_cute4"
        fixture = cute_fixture(4)
        center_x, center_y = fixture["center"]
        mask_bbox = cute_relative_bbox(4, 0.25, 0.25, 0.75, 0.75)
        outside_x, outside_y = cute_relative_point(4, 0.1, 0.1)
        actions = [
            import_image_as_layer("action_001", "layer_cute4", fixture["path"], x=fixture["x"], y=fixture["y"], name="cute4"),
            select_rect("action_002", "mask_center", mask_bbox, name="center"),
            invert_mask("action_003", "mask_outside", "mask_center", set_active=True),
            clear_region("action_004", "layer_cute4", "mask_outside"),
            export_flat("action_005", self.export_path(case_name, "final.png")),
        ]
        doc, results, summary = self.run_case(case_name, fixture["canvas_width"], fixture["canvas_height"], actions)

        self.assert_all_succeeded(results)
        self.assertEqual(doc.active_selection_mask_id, "mask_outside")
        self.assertGreater(float(doc.get_layer("layer_cute4").pixels[center_y, center_x, 3]), 0.9)
        self.assertEqual(float(doc.get_layer("layer_cute4").pixels[outside_y, outside_x, 3]), 0.0)
        self.assert_trace_healthy(summary, expected_results=len(actions), min_snapshots=len(actions) + 1)

    def test_37_blur_region_with_cute5(self) -> None:
        """Blur RGB channels inside a write mask on an imported image layer."""
        case_name = "test_37_blur_region_with_cute5"
        fixture = cute_fixture(5)
        center_x, center_y = fixture["center"]
        blur_bbox = cute_relative_bbox(5, 0.45, 0.25, 0.55, 0.75)
        black_bbox = cute_relative_bbox(5, 0.47, 0.25, 0.53, 0.75)
        strip_bbox = [center_x - 2, blur_bbox[1], center_x + 2, blur_bbox[3]]
        actions = [
            import_image_as_layer("action_001", "layer_cute5", fixture["path"], x=fixture["x"], y=fixture["y"], name="cute5"),
            full_canvas_mask("action_002", "mask_full_canvas", fixture["canvas_width"], fixture["canvas_height"]),
            draw_shape(
                "action_003",
                "layer_cute5",
                rectangle(black_bbox),
                write_mask_id="mask_full_canvas",
                fill={"color": "#000000"},
            ),
            draw_shape(
                "action_004",
                "layer_cute5",
                rectangle(strip_bbox),
                write_mask_id="mask_full_canvas",
                fill={"color": "#ffffff"},
            ),
            select_rect("action_005", "mask_blur_strip", blur_bbox, name="blur strip"),
            blur_region("action_006", "layer_cute5", "mask_blur_strip", radius=2.0, channels="rgb"),
            export_flat("action_007", self.export_path(case_name, "final.png")),
        ]
        doc, results, summary = self.run_case(case_name, fixture["canvas_width"], fixture["canvas_height"], actions)

        layer = doc.get_layer("layer_cute5")
        self.assert_all_succeeded(results)
        self.assertGreater(float(layer.pixels[center_y, center_x - 5, 0]), 0.1)
        self.assertGreater(float(layer.pixels[center_y, center_x - 5, 3]), 0.9)
        self.assert_trace_healthy(summary, expected_results=len(actions), min_snapshots=len(actions) + 1)

    def test_38_import_vector_as_raster(self) -> None:
        """Rasterize an SVG asset directly into a normal full-canvas raster layer."""
        case_name = "test_38_import_vector_as_raster"
        actions = [
            import_vector_as_raster(
                "action_001",
                "layer_vector_icon",
                vector_fixture_path("simple_icon.svg"),
                x=16,
                y=12,
                width=160,
                height=120,
                name="vector icon rasterized",
                background_color=None,
            ),
            export_flat("action_002", self.export_path(case_name, "final.png")),
        ]
        doc, results, summary = self.run_case(case_name, 200, 160, actions)

        layer = doc.get_layer("layer_vector_icon")
        self.assert_all_succeeded(results)
        self.assertEqual(layer.kind.value, "raster")
        self.assertEqual(layer.metadata["source_format"], "vector")
        self.assertEqual(layer.metadata["rasterized_size"], [160, 120])
        self.assertGreater(float(layer.pixels[72, 96, 1]), 0.35)
        self.assertGreater(float(layer.pixels[72, 96, 3]), 0.9)
        self.assert_trace_healthy(summary, expected_results=len(actions), min_snapshots=len(actions) + 1)

    def test_39_rasterize_vector_asset(self) -> None:
        """Rasterize an SVG asset to a standalone PNG without mutating the document."""
        case_name = "test_39_rasterize_vector_asset"
        output_path = self.export_path(case_name, "simple_icon.png")
        actions = [
            rasterize_vector_asset(
                "action_001",
                vector_fixture_path("simple_icon.svg"),
                output_path,
                width=120,
                height=90,
                background_color="#ffffff",
            )
        ]
        doc, results, summary = self.run_case(case_name, 64, 64, actions)

        self.assert_all_succeeded(results)
        self.assertEqual(doc.revision, 0)
        self.assertEqual(doc.layers, [])
        self.assertTrue(output_path.exists())
        with Image.open(output_path) as image:
            self.assertEqual(image.mode, "RGBA")
            self.assertEqual(image.size, (120, 90))
            self.assertEqual(image.getpixel((0, 0))[3], 255)
        self.assertEqual(results[0].output_assets["path"], str(output_path))
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


def cute_fixture_path(index: int) -> Path:
    """Return the full-resolution cute fixture path."""
    path = CUTE_ROOT / f"cute{index}.jpg"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def cute_image_size(index: int) -> tuple[int, int]:
    """Return the native width and height for a cute fixture."""
    with Image.open(cute_fixture_path(index)) as image:
        return image.size


def cute_fixture(index: int, padding: int = 16) -> dict[str, Any]:
    """Return native-size import geometry for a cute fixture on a padded canvas."""
    image_width, image_height = cute_image_size(index)
    return {
        "path": cute_fixture_path(index),
        "x": padding,
        "y": padding,
        "image_width": image_width,
        "image_height": image_height,
        "canvas_width": image_width + padding * 2,
        "canvas_height": image_height + padding * 2,
        "image_bbox": [padding, padding, padding + image_width, padding + image_height],
        "center": (padding + image_width // 2, padding + image_height // 2),
    }


def cute_relative_bbox(
    index: int,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    padding: int = 16,
) -> list[int]:
    """Return a bbox inside a native-size cute fixture using relative coordinates."""
    image_width, image_height = cute_image_size(index)
    return [
        padding + int(round(image_width * x0)),
        padding + int(round(image_height * y0)),
        padding + int(round(image_width * x1)),
        padding + int(round(image_height * y1)),
    ]


def cute_relative_point(index: int, x: float, y: float, padding: int = 16) -> tuple[int, int]:
    """Return one point inside a native-size cute fixture using relative coordinates."""
    image_width, image_height = cute_image_size(index)
    return padding + int(round(image_width * x)), padding + int(round(image_height * y))


def vector_fixture_path(name: str) -> Path:
    """Return a vector fixture path."""
    path = VECTOR_ROOT / name
    if not path.exists():
        raise FileNotFoundError(path)
    return path


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


def import_vector_as_raster(
    action_id: str,
    layer_id: str,
    path: Path,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    name: str,
    opacity: float = 1.0,
    set_active: bool = True,
    background_color: Any = None,
) -> dict[str, Any]:
    return action(
        action_id,
        "import_vector_as_raster",
        params={
            "path": str(path),
            "name": name,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "opacity": opacity,
            "blend_mode": "normal",
            "set_active": set_active,
            "background_color": background_color,
        },
        target={"output_layer_id": layer_id},
    )


def rasterize_vector_asset(
    action_id: str,
    path: Path,
    output_path: Path,
    *,
    width: int,
    height: int,
    background_color: Any = None,
) -> dict[str, Any]:
    return action(
        action_id,
        "rasterize_vector_asset",
        params={
            "path": str(path),
            "output_path": str(output_path),
            "width": width,
            "height": height,
            "background_color": background_color,
        },
        preconditions={"require_write_mask": False},
    )


def resize_canvas(action_id: str, width: int, height: int) -> dict[str, Any]:
    return action(action_id, "resize_canvas", params={"width": width, "height": height, "anchor": "center"})


def crop(
    action_id: str,
    bbox_xyxy: list[int],
    *,
    scope: str,
    layer_id: str | None = None,
    mask_id: str | None = None,
) -> dict[str, Any]:
    target: dict[str, Any] = {}
    if layer_id is not None:
        target["layer_id"] = layer_id
    if mask_id is not None:
        target["mask_id"] = mask_id
    return action(action_id, "crop", params={"bbox_xyxy": bbox_xyxy, "scope": scope}, target=target)


def set_active_layer(action_id: str, layer_id: str) -> dict[str, Any]:
    return action(action_id, "set_active_layer", target={"layer_id": layer_id})


def delete_layer(action_id: str, layer_id: str) -> dict[str, Any]:
    return action(
        action_id,
        "delete_layer",
        target={"layer_id": layer_id},
        preconditions={"required_layer_ids": [layer_id], "allow_hidden_layers": True},
    )


def duplicate_layer(action_id: str, source_layer_id: str, output_layer_id: str, *, name: str) -> dict[str, Any]:
    return action(
        action_id,
        "duplicate_layer",
        params={"name": name, "set_active": True},
        target={"layer_id": source_layer_id, "output_layer_id": output_layer_id},
        preconditions={"required_layer_ids": [source_layer_id]},
    )


def rename_layer(action_id: str, layer_id: str, name: str) -> dict[str, Any]:
    return action(
        action_id,
        "rename_layer",
        params={"name": name},
        target={"layer_id": layer_id},
        preconditions={"required_layer_ids": [layer_id]},
    )


def reorder_layer(action_id: str, layer_id: str, index: int) -> dict[str, Any]:
    return action(
        action_id,
        "reorder_layer",
        params={"index": index},
        target={"layer_id": layer_id},
        preconditions={"required_layer_ids": [layer_id], "allow_hidden_layers": True},
    )


def set_layer_visibility(action_id: str, layer_id: str, visible: bool) -> dict[str, Any]:
    return action(
        action_id,
        "set_layer_visibility",
        params={"visible": visible},
        target={"layer_id": layer_id},
        preconditions={"required_layer_ids": [layer_id], "allow_hidden_layers": True},
    )


def set_layer_opacity(action_id: str, layer_id: str, opacity: float) -> dict[str, Any]:
    return action(
        action_id,
        "set_layer_opacity",
        params={"opacity": opacity},
        target={"layer_id": layer_id},
        preconditions={"required_layer_ids": [layer_id], "allow_hidden_layers": True},
    )


def set_blend_mode(action_id: str, layer_id: str, blend_mode: str) -> dict[str, Any]:
    return action(
        action_id,
        "set_blend_mode",
        params={"blend_mode": blend_mode},
        target={"layer_id": layer_id},
        preconditions={"required_layer_ids": [layer_id], "allow_hidden_layers": True},
    )


def merge_layers(
    action_id: str,
    *,
    mode: str,
    layer_id: str | None = None,
    output_layer_id: str | None = None,
    output_layer_name: str | None = None,
    layer_ids: list[str] | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"mode": mode}
    if output_layer_name is not None:
        params["output_layer_name"] = output_layer_name
    if layer_ids is not None:
        params["layer_ids"] = layer_ids
    target: dict[str, Any] = {}
    if layer_id is not None:
        target["layer_id"] = layer_id
    if output_layer_id is not None:
        target["output_layer_id"] = output_layer_id
    return action(action_id, "merge_layers", params=params, target=target)


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


def select_ellipse(action_id: str, mask_id: str, bbox_xyxy: list[int], *, name: str) -> dict[str, Any]:
    return action(
        action_id,
        "select_ellipse",
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


def grow_mask(action_id: str, output_mask_id: str, source_mask_id: str, *, pixels: int) -> dict[str, Any]:
    return action(
        action_id,
        "grow_mask",
        params={"source_mask_id": source_mask_id, "pixels": pixels, "name": output_mask_id, "set_active": False},
        target={"mask_id": output_mask_id},
        preconditions={"required_mask_ids": [source_mask_id]},
    )


def shrink_mask(action_id: str, output_mask_id: str, source_mask_id: str, *, pixels: int) -> dict[str, Any]:
    return action(
        action_id,
        "shrink_mask",
        params={"source_mask_id": source_mask_id, "pixels": pixels, "name": output_mask_id, "set_active": False},
        target={"mask_id": output_mask_id},
        preconditions={"required_mask_ids": [source_mask_id]},
    )


def invert_mask(action_id: str, output_mask_id: str, source_mask_id: str, *, set_active: bool = False) -> dict[str, Any]:
    return action(
        action_id,
        "invert_mask",
        params={"source_mask_id": source_mask_id, "name": output_mask_id, "set_active": set_active},
        target={"mask_id": output_mask_id},
        preconditions={"required_mask_ids": [source_mask_id]},
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


def blur_region(
    action_id: str,
    layer_id: str,
    write_mask_id: str,
    *,
    radius: float,
    channels: str | list[str],
) -> dict[str, Any]:
    return action(
        action_id,
        "blur_region",
        params={"radius": radius, "channels": channels, "edge_mode": "nearest"},
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
