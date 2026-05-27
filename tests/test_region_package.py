"""Tests for the shared region utility package."""

from __future__ import annotations

import unittest

import numpy as np

from ai_edit_kernel.document.document_state import CanvasSpec, DocumentState
from ai_edit_kernel.document.layer import Layer
from ai_edit_kernel.document.mask import Mask, MaskKind
from ai_edit_kernel.region import (
    BBoxXYXY,
    apply_crop_with_mask,
    apply_write_mask,
    bbox_from_mask,
    bbox_from_xyxy,
    changed_bbox,
    changed_pixels_outside_mask,
    clip_bbox,
    expand_crop_to_canvas,
    extract_layer,
    extract_mask,
    hard_clip_rgba_to_mask,
    make_region_view,
    max_delta_outside_mask,
    multiply_alpha_by_mask,
    pad_bbox,
    paste_crop,
    rect_mask,
    resolve_region_bbox,
    resolve_region_mask,
    snap_bbox_to_multiple,
    source_over,
)


class RegionPackageTests(unittest.TestCase):
    """Verify non-mutating geometry, extraction, and compositing primitives."""

    def test_bbox_validation_padding_and_backend_snapping(self) -> None:
        box = bbox_from_xyxy([2, 1, 5, 4], 8, 6)
        self.assertEqual(box.as_list(), [2, 1, 5, 4])
        self.assertEqual((box.width, box.height, box.area), (3, 3, 9))

        with self.assertRaises(ValueError):
            bbox_from_xyxy([2.25, 1, 5, 4], 8, 6)

        detector_box = bbox_from_xyxy([2.25, 1.5, 4.1, 4.01], 8, 6, rounding="floor_ceil")
        self.assertEqual(detector_box.as_list(), [2, 1, 5, 5])

        clipped = clip_bbox(BBoxXYXY(-3, -1, 4, 3), 8, 6)
        self.assertEqual(clipped.as_list(), [0, 0, 4, 3])

        padded = pad_bbox(box, (3, 2, 4, 5), 8, 6)
        self.assertEqual(padded.as_list(), [0, 0, 8, 6])

        snapped = snap_bbox_to_multiple(BBoxXYXY(5, 4, 7, 5), 10, 8, multiple=4)
        self.assertEqual(snapped.as_list(), [4, 2, 8, 6])

    def test_mask_bbox_and_region_resolution_match_kernel_priority(self) -> None:
        document = self.make_document()

        bbox = bbox_from_mask(document.get_mask("mask_object").data)
        self.assertIsNotNone(bbox)
        self.assertEqual(bbox.as_list(), [3, 2, 7, 5])

        explicit_mask = resolve_region_mask(document, mask_id="mask_object", bbox=[0, 0, 2, 2])
        self.assertEqual(int(np.count_nonzero(explicit_mask > 0.0)), 12)

        explicit_bbox = resolve_region_mask(document, bbox=[0, 0, 2, 2])
        self.assertEqual(int(np.count_nonzero(explicit_bbox > 0.0)), 4)

        limited_mask = resolve_region_mask(document, mask_id="mask_object", bbox=[5, 0, 8, 6], intersect_bbox=True)
        self.assertEqual(int(np.count_nonzero(limited_mask > 0.0)), 6)

        document.set_active_selection("mask_object")
        active_bbox = resolve_region_bbox(document, use_active_selection=True, default_full_canvas=False)
        self.assertEqual(active_bbox.as_list(), [3, 2, 7, 5])

    def test_make_region_view_extracts_independent_crops(self) -> None:
        document = self.make_document()

        view = make_region_view(
            document,
            mask_id="mask_object",
            source_layer_id="layer_source",
            include_preview=True,
            include_layer_pixels=True,
            padding=1,
        )

        self.assertEqual(view.bbox.as_list(), [2, 1, 8, 6])
        self.assertEqual(view.shape_hw, (5, 6))
        self.assertEqual(view.mask.shape, (5, 6))
        self.assertEqual(view.preview.shape, (5, 6, 4))
        self.assertEqual(view.layer_pixels.shape, (5, 6, 4))
        self.assertEqual(float(view.mask[1, 1]), 1.0)
        self.assertEqual(float(view.mask[0, 0]), 0.0)

        view.layer_pixels[:] = 0.0
        self.assertGreater(float(document.get_layer("layer_source").pixels[2, 3, 3]), 0.9)

        layer_crop = extract_layer(document, "layer_source", [3, 2, 7, 5])
        mask_crop = extract_mask(document.get_mask("mask_object").data, [3, 2, 7, 5])
        self.assertEqual(layer_crop.shape, (3, 4, 4))
        self.assertTrue(np.all(mask_crop == 1.0))

    def test_write_mask_and_crop_compositing_are_consistent(self) -> None:
        before = np.zeros((6, 8, 4), dtype=np.float32)
        before[..., 3] = 1.0
        proposed = np.zeros((6, 8, 4), dtype=np.float32)
        proposed[..., :] = [1.0, 0.0, 0.0, 1.0]
        mask = rect_mask(8, 6, [2, 1, 6, 5])
        mask[1, 2] = 0.5

        blended = apply_write_mask(before, proposed, mask)
        self.assertTrue(np.allclose(blended[0, 0], [0.0, 0.0, 0.0, 1.0]))
        self.assertTrue(np.allclose(blended[2, 3], [1.0, 0.0, 0.0, 1.0]))
        self.assertTrue(np.allclose(blended[1, 2], [0.5, 0.0, 0.0, 1.0]))

        crop = np.zeros((4, 4, 4), dtype=np.float32)
        crop[..., :] = [0.0, 0.0, 1.0, 1.0]
        crop_mask = np.ones((4, 4), dtype=np.float32)
        crop_mask[0, 0] = 0.25
        applied = apply_crop_with_mask(before, crop, [2, 1, 6, 5], crop_mask)
        self.assertTrue(np.allclose(applied[0, 0], before[0, 0]))
        self.assertTrue(np.allclose(applied[2, 3], [0.0, 0.0, 1.0, 1.0]))
        self.assertTrue(np.allclose(applied[1, 2], [0.0, 0.0, 0.25, 1.0]))

    def test_crop_placement_clipping_and_change_metrics(self) -> None:
        destination = np.zeros((4, 5, 4), dtype=np.float32)
        source = np.ones((3, 3, 4), dtype=np.float32)
        pasted = paste_crop(destination, source, -1, 2)

        self.assertEqual(int(np.count_nonzero(pasted[..., 3] > 0.0)), 4)
        self.assertTrue(np.allclose(pasted[2, 0], [1.0, 1.0, 1.0, 1.0]))
        self.assertTrue(np.allclose(pasted[3, 1], [1.0, 1.0, 1.0, 1.0]))

        crop = np.ones((2, 2, 4), dtype=np.float32)
        canvas = expand_crop_to_canvas(crop, BBoxXYXY(1, 1, 3, 3), width=5, height=4)
        self.assertEqual(int(np.count_nonzero(canvas[..., 3] > 0.0)), 4)

        changes = changed_bbox(destination, canvas)
        self.assertIsNotNone(changes)
        self.assertEqual(changes.as_list(), [1, 1, 3, 3])

        guard = rect_mask(5, 4, [1, 1, 3, 3])
        self.assertEqual(changed_pixels_outside_mask(destination, canvas, guard), 0)
        self.assertEqual(changed_pixels_outside_mask(destination, canvas, np.zeros((4, 5), dtype=np.float32)), 4)
        self.assertAlmostEqual(max_delta_outside_mask(destination, canvas, guard), 0.0)
        self.assertAlmostEqual(max_delta_outside_mask(destination, canvas, np.zeros((4, 5), dtype=np.float32)), 1.0)

    def test_mask_clipping_alpha_multiply_and_source_over(self) -> None:
        pixels = np.ones((2, 3, 4), dtype=np.float32)
        mask = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.25]], dtype=np.float32)

        alpha_scaled = multiply_alpha_by_mask(pixels, mask)
        self.assertTrue(np.allclose(alpha_scaled[..., 3], mask))

        hard_clipped = hard_clip_rgba_to_mask(pixels, mask, threshold=0.5)
        self.assertTrue(np.allclose(hard_clipped[0, 0], [1.0, 1.0, 1.0, 1.0]))
        self.assertTrue(np.allclose(hard_clipped[0, 1], [0.0, 0.0, 0.0, 0.0]))

        destination = np.zeros((1, 1, 4), dtype=np.float32)
        destination[0, 0] = [0.0, 0.0, 1.0, 1.0]
        source = np.zeros((1, 1, 4), dtype=np.float32)
        source[0, 0] = [1.0, 0.0, 0.0, 0.5]
        composited = source_over(destination, source)
        self.assertTrue(np.allclose(composited[0, 0], [0.5, 0.0, 0.5, 1.0], atol=0.001))

    def make_document(self) -> DocumentState:
        """Create a small document with a source layer and object mask."""
        pixels = np.zeros((6, 8, 4), dtype=np.float32)
        pixels[..., :] = [0.1, 0.2, 0.3, 1.0]
        pixels[2:5, 3:7, :] = [1.0, 0.0, 0.0, 1.0]

        mask = np.zeros((6, 8), dtype=np.float32)
        mask[2:5, 3:7] = 1.0

        document = DocumentState(
            id="doc_region",
            canvas=CanvasSpec(width=8, height=6),
            layers=[Layer(id="layer_source", name="source", pixels=pixels)],
            masks={
                "mask_object": Mask(
                    id="mask_object",
                    name="object",
                    data=mask,
                    kind=MaskKind.OBJECT,
                    hard=True,
                )
            },
            active_layer_id="layer_source",
        )
        document.validate()
        return document


if __name__ == "__main__":
    unittest.main()
