"""Kernel-level region helpers.

The region package provides non-mutating primitives for resolving document work
areas, extracting crop-sized arrays, and compositing crop results back through
masks. User-facing actions may use these helpers, but the helpers themselves do
not execute actions or mutate `DocumentState`.
"""

from ai_edit_kernel.region.composite import (
    apply_crop_with_mask,
    apply_write_mask,
    changed_bbox,
    changed_pixels_outside_mask,
    expand_crop_to_canvas,
    hard_clip_rgba_to_mask,
    max_delta_outside_mask,
    multiply_alpha_by_mask,
    paste_crop,
    source_over,
)
from ai_edit_kernel.region.extract import (
    extract_document_mask,
    extract_layer,
    extract_mask,
    extract_preview,
    extract_rgba,
    make_region_view,
    rect_mask,
    resolve_region_bbox,
    resolve_region_mask,
)
from ai_edit_kernel.region.geometry import (
    bbox_from_mask,
    bbox_from_xywh,
    bbox_from_xyxy,
    clip_bbox,
    content_bbox_rgba,
    full_canvas_bbox,
    intersect_bbox,
    normalize_padding,
    pad_bbox,
    snap_bbox_to_multiple,
    union_bbox,
)
from ai_edit_kernel.region.types import BBoxXYXY, RegionPlacement, RegionSpec, RegionView

__all__ = [
    "BBoxXYXY",
    "RegionPlacement",
    "RegionSpec",
    "RegionView",
    "apply_crop_with_mask",
    "apply_write_mask",
    "bbox_from_mask",
    "bbox_from_xywh",
    "bbox_from_xyxy",
    "changed_bbox",
    "changed_pixels_outside_mask",
    "clip_bbox",
    "content_bbox_rgba",
    "expand_crop_to_canvas",
    "extract_document_mask",
    "extract_layer",
    "extract_mask",
    "extract_preview",
    "extract_rgba",
    "full_canvas_bbox",
    "hard_clip_rgba_to_mask",
    "intersect_bbox",
    "make_region_view",
    "max_delta_outside_mask",
    "multiply_alpha_by_mask",
    "normalize_padding",
    "pad_bbox",
    "paste_crop",
    "rect_mask",
    "resolve_region_bbox",
    "resolve_region_mask",
    "snap_bbox_to_multiple",
    "source_over",
    "union_bbox",
]
