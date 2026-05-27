"""Non-mutating extraction helpers for document regions."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ai_edit_kernel.document.document_state import DocumentState
from ai_edit_kernel.document.layer import Layer
from ai_edit_kernel.region.geometry import bbox_from_mask, bbox_from_xyxy, full_canvas_bbox, pad_bbox
from ai_edit_kernel.region.types import BBoxXYXY, RegionSpec, RegionView


def rect_mask(width: int, height: int, bbox: BBoxXYXY | list[int] | tuple[int, int, int, int]) -> np.ndarray:
    """Return a full-canvas hard rectangular mask for `bbox`."""
    box = bbox_from_xyxy(bbox, width, height)
    data = np.zeros((height, width), dtype=np.float32)
    y_slice, x_slice = box.to_slices()
    data[y_slice, x_slice] = 1.0
    return data


def resolve_region_mask(
    document: DocumentState,
    *,
    mask_id: Optional[str] = None,
    bbox: Optional[BBoxXYXY | list[int] | tuple[int, int, int, int]] = None,
    use_active_selection: bool = True,
    default_full_canvas: bool = True,
    intersect_bbox: bool = False,
) -> np.ndarray:
    """Resolve a full-canvas mask from a mask ID, bbox, active selection, or canvas.

    Priority matches existing action behavior by default: explicit mask, explicit
    bbox, active selection, then full canvas. When `intersect_bbox=True`, an
    explicit bbox limits an explicit mask or active selection instead of being
    ignored.
    """
    width = document.canvas.width
    height = document.canvas.height
    box = bbox_from_xyxy(bbox, width, height) if bbox is not None else None

    if mask_id is not None:
        data = np.array(document.get_mask(mask_id).data, dtype=np.float32, copy=True)
    elif box is not None:
        data = rect_mask(width, height, box)
    elif use_active_selection and document.active_selection_mask_id is not None:
        data = np.array(document.get_mask(document.active_selection_mask_id).data, dtype=np.float32, copy=True)
    elif default_full_canvas:
        data = np.ones((height, width), dtype=np.float32)
    else:
        data = np.zeros((height, width), dtype=np.float32)

    _validate_mask_shape(data, width, height, "resolved mask")
    if intersect_bbox and box is not None and mask_id is not None:
        data *= rect_mask(width, height, box)
    elif intersect_bbox and box is not None and mask_id is None and use_active_selection and document.active_selection_mask_id is not None:
        data *= rect_mask(width, height, box)
    return np.clip(data, 0.0, 1.0).astype(np.float32)


def resolve_region_bbox(
    document: DocumentState,
    *,
    mask_id: Optional[str] = None,
    bbox: Optional[BBoxXYXY | list[int] | tuple[int, int, int, int]] = None,
    use_active_selection: bool = False,
    default_full_canvas: bool = True,
    padding: int | tuple[int, int, int, int] = 0,
    mask_threshold: float = 0.0,
) -> BBoxXYXY:
    """Resolve a non-empty work-area bbox for a document region."""
    width = document.canvas.width
    height = document.canvas.height

    if bbox is not None:
        resolved = bbox_from_xyxy(bbox, width, height)
    else:
        source_mask_id = mask_id
        if source_mask_id is None and use_active_selection:
            source_mask_id = document.active_selection_mask_id
        if source_mask_id is not None:
            resolved = bbox_from_mask(document.get_mask(source_mask_id).data, threshold=mask_threshold)
            if resolved is None:
                raise ValueError(f"mask {source_mask_id!r} does not contain any selected pixels")
        elif default_full_canvas:
            resolved = full_canvas_bbox(width, height)
        else:
            raise ValueError("no region bbox, mask, active selection, or full-canvas default is available")

    if padding != 0 and padding != (0, 0, 0, 0):
        resolved = pad_bbox(resolved, padding, width, height)
    return resolved.require_non_empty("region bbox")


def extract_rgba(pixels: np.ndarray, bbox: BBoxXYXY | list[int] | tuple[int, int, int, int], *, copy: bool = True) -> np.ndarray:
    """Extract an RGBA crop from an array."""
    _validate_rgba_array(pixels, "pixels")
    box = bbox_from_xyxy(bbox, pixels.shape[1], pixels.shape[0])
    y_slice, x_slice = box.to_slices()
    crop = pixels[y_slice, x_slice, :]
    return np.array(crop, dtype=np.float32, copy=copy)


def extract_mask(mask: np.ndarray, bbox: BBoxXYXY | list[int] | tuple[int, int, int, int], *, copy: bool = True) -> np.ndarray:
    """Extract a mask crop from a full-canvas mask array."""
    _validate_mask_array(mask, "mask")
    box = bbox_from_xyxy(bbox, mask.shape[1], mask.shape[0])
    y_slice, x_slice = box.to_slices()
    crop = mask[y_slice, x_slice]
    return np.array(crop, dtype=np.float32, copy=copy)


def extract_layer(document: DocumentState, layer_id: str, bbox: BBoxXYXY | list[int] | tuple[int, int, int, int], *, copy: bool = True) -> np.ndarray:
    """Extract a crop from a document layer with raster pixel data."""
    layer = document.get_layer(layer_id)
    _require_layer_pixels(layer)
    return extract_rgba(layer.pixels, bbox, copy=copy)


def extract_document_mask(document: DocumentState, mask_id: str, bbox: BBoxXYXY | list[int] | tuple[int, int, int, int], *, copy: bool = True) -> np.ndarray:
    """Extract a crop from a registered document mask."""
    return extract_mask(document.get_mask(mask_id).data, bbox, copy=copy)


def extract_preview(document: DocumentState, bbox: BBoxXYXY | list[int] | tuple[int, int, int, int], *, include_hidden: bool = False, copy: bool = True) -> np.ndarray:
    """Flatten the document and extract an RGBA preview crop."""
    return extract_rgba(document.flatten_preview(include_hidden=include_hidden), bbox, copy=copy)


def make_region_view(
    document: DocumentState,
    spec: Optional[RegionSpec] = None,
    *,
    bbox: Optional[BBoxXYXY | list[int] | tuple[int, int, int, int]] = None,
    mask_id: Optional[str] = None,
    source_layer_id: Optional[str] = None,
    include_preview: bool = False,
    include_layer_pixels: bool = False,
    include_mask: bool = True,
    use_active_selection: bool = False,
    default_full_canvas: bool = False,
    intersect_bbox_with_mask: bool = False,
    padding: int | tuple[int, int, int, int] = 0,
    mask_threshold: float = 0.0,
) -> RegionView:
    """Create an extracted work-area view from the current document state.

    The returned arrays are crop-sized copies. The document itself is untouched.
    This helper is intended for current actions as well as future backend calls
    that need local source, mask, and target context.
    """
    if spec is not None:
        if not isinstance(spec, RegionSpec):
            raise TypeError("spec must be a RegionSpec or None")
        bbox = spec.bbox if bbox is None else bbox
        mask_id = spec.mask_id if mask_id is None else mask_id
        use_active_selection = spec.use_active_selection
        default_full_canvas = spec.default_full_canvas
        padding = spec.padding
        mask_threshold = spec.mask_threshold

    resolved_bbox = resolve_region_bbox(
        document,
        mask_id=mask_id,
        bbox=bbox,
        use_active_selection=use_active_selection,
        default_full_canvas=default_full_canvas,
        padding=padding,
        mask_threshold=mask_threshold,
    )

    mask_crop: Optional[np.ndarray] = None
    if include_mask:
        full_mask = resolve_region_mask(
            document,
            mask_id=mask_id,
            bbox=bbox,
            use_active_selection=use_active_selection,
            default_full_canvas=default_full_canvas,
            intersect_bbox=intersect_bbox_with_mask,
        )
        mask_crop = extract_mask(full_mask, resolved_bbox)

    preview_crop = extract_preview(document, resolved_bbox) if include_preview else None
    layer_crop = None
    if include_layer_pixels:
        if source_layer_id is None:
            raise ValueError("source_layer_id is required when include_layer_pixels is true")
        layer_crop = extract_layer(document, source_layer_id, resolved_bbox)

    return RegionView(
        canvas_width=document.canvas.width,
        canvas_height=document.canvas.height,
        bbox=resolved_bbox,
        mask=mask_crop,
        preview=preview_crop,
        layer_pixels=layer_crop,
        source_layer_id=source_layer_id,
        source_mask_id=mask_id,
        metadata={
            "padding": padding,
            "mask_threshold": float(mask_threshold),
            "intersect_bbox_with_mask": bool(intersect_bbox_with_mask),
        },
    )


def _require_layer_pixels(layer: Layer) -> None:
    if layer.pixels is None:
        raise ValueError(f"layer {layer.id!r} has no pixel data")
    _validate_rgba_array(layer.pixels, f"layer {layer.id!r} pixels")


def _validate_rgba_array(pixels: np.ndarray, field_name: str) -> None:
    if not isinstance(pixels, np.ndarray):
        raise TypeError(f"{field_name} must be a NumPy array")
    if pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError(f"{field_name} must have shape H x W x 4")


def _validate_mask_array(mask: np.ndarray, field_name: str) -> None:
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"{field_name} must be a NumPy array")
    if mask.ndim != 2:
        raise ValueError(f"{field_name} must have shape H x W")


def _validate_mask_shape(mask: np.ndarray, width: int, height: int, field_name: str) -> None:
    _validate_mask_array(mask, field_name)
    if mask.shape != (height, width):
        raise ValueError(f"{field_name} shape must match the document canvas")
