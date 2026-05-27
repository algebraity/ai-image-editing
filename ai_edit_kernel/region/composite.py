"""Compositing and placement helpers for region crops."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ai_edit_kernel.region.geometry import bbox_from_xyxy
from ai_edit_kernel.region.types import BBoxXYXY, RegionPlacement


def apply_write_mask(before_pixels: np.ndarray, proposed_pixels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend proposed RGBA pixels into existing pixels through a soft mask."""
    _validate_rgba_array(before_pixels, "before_pixels")
    _validate_rgba_array(proposed_pixels, "proposed_pixels")
    _validate_mask_array(mask, "mask")
    if before_pixels.shape != proposed_pixels.shape:
        raise ValueError("before_pixels and proposed_pixels must have the same shape")
    if mask.shape != before_pixels.shape[:2]:
        raise ValueError("mask shape must match pixel array dimensions")

    alpha = np.clip(mask.astype(np.float32, copy=False), 0.0, 1.0)[..., np.newaxis]
    blended = before_pixels.astype(np.float32, copy=False) * (1.0 - alpha) + proposed_pixels.astype(np.float32, copy=False) * alpha
    return np.clip(blended, 0.0, 1.0).astype(np.float32)


def apply_crop_with_mask(
    before_pixels: np.ndarray,
    proposed_crop: np.ndarray,
    bbox: BBoxXYXY | list[int] | tuple[int, int, int, int],
    mask: np.ndarray,
) -> np.ndarray:
    """Blend a crop into a full-canvas RGBA array through a crop or canvas mask."""
    _validate_rgba_array(before_pixels, "before_pixels")
    _validate_rgba_array(proposed_crop, "proposed_crop")
    box = bbox_from_xyxy(bbox, before_pixels.shape[1], before_pixels.shape[0])
    _validate_crop_shape(proposed_crop, box, "proposed_crop")

    mask_crop = _mask_crop_for_bbox(mask, box, before_pixels.shape[1], before_pixels.shape[0])
    output = np.array(before_pixels, dtype=np.float32, copy=True)
    y_slice, x_slice = box.to_slices()
    output[y_slice, x_slice, :] = apply_write_mask(output[y_slice, x_slice, :], proposed_crop, mask_crop)
    return output


def paste_crop(
    destination: np.ndarray,
    source: np.ndarray,
    x: int,
    y: int,
    *,
    copy: bool = True,
    require_intersection: bool = True,
) -> np.ndarray:
    """Paste a crop into a destination array at integer canvas coordinates.

    Pasting uses direct replacement, matching the current clipboard helper. The
    destination is copied by default.
    """
    _validate_rgba_array(destination, "destination")
    _validate_rgba_array(source, "source")
    if isinstance(x, bool) or not isinstance(x, int):
        raise TypeError("x must be an integer")
    if isinstance(y, bool) or not isinstance(y, int):
        raise TypeError("y must be an integer")

    output = np.array(destination, dtype=np.float32, copy=copy)
    height, width = output.shape[:2]
    src_h, src_w = source.shape[:2]
    dst_x0 = max(0, x)
    dst_y0 = max(0, y)
    src_x0 = max(0, -x)
    src_y0 = max(0, -y)
    copy_w = min(src_w - src_x0, width - dst_x0)
    copy_h = min(src_h - src_y0, height - dst_y0)
    if copy_w <= 0 or copy_h <= 0:
        if require_intersection:
            raise ValueError("pasted pixels do not intersect the canvas")
        return output
    output[dst_y0 : dst_y0 + copy_h, dst_x0 : dst_x0 + copy_w, :] = source[src_y0 : src_y0 + copy_h, src_x0 : src_x0 + copy_w, :]
    return np.clip(output, 0.0, 1.0).astype(np.float32)


def expand_crop_to_canvas(
    crop: np.ndarray,
    placement: RegionPlacement | BBoxXYXY | list[int] | tuple[int, int, int, int],
    width: Optional[int] = None,
    height: Optional[int] = None,
    *,
    fill_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Place a crop on a new full-canvas RGBA array."""
    _validate_rgba_array(crop, "crop")
    if isinstance(placement, RegionPlacement):
        canvas_width = placement.canvas_width
        canvas_height = placement.canvas_height
        box = placement.bbox
    else:
        if width is None or height is None:
            raise ValueError("width and height are required when placement is a bbox")
        canvas_width = width
        canvas_height = height
        box = bbox_from_xyxy(placement, canvas_width, canvas_height)
    _validate_crop_shape(crop, box, "crop")
    fill = _validate_fill_color(fill_color)

    source = np.array(crop, dtype=np.float32, copy=True)
    if mask is not None:
        source = multiply_alpha_by_mask(source, _mask_crop_for_bbox(mask, box, canvas_width, canvas_height))

    output = np.zeros((canvas_height, canvas_width, 4), dtype=np.float32)
    output[..., :] = fill
    y_slice, x_slice = box.to_slices()
    output[y_slice, x_slice, :] = source
    return np.clip(output, 0.0, 1.0).astype(np.float32)


def hard_clip_rgba_to_mask(
    pixels: np.ndarray,
    mask: np.ndarray,
    *,
    threshold: float = 0.0,
    preserve_rgb: bool = False,
) -> np.ndarray:
    """Set pixels outside a hard mask to transparent or zero RGBA."""
    _validate_rgba_array(pixels, "pixels")
    _validate_mask_array(mask, "mask")
    if mask.shape != pixels.shape[:2]:
        raise ValueError("mask shape must match pixel array dimensions")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not np.isfinite(float(threshold)):
        raise TypeError("threshold must be a finite number")
    output = np.array(pixels, dtype=np.float32, copy=True)
    outside = mask <= float(threshold)
    if preserve_rgb:
        output[outside, 3] = 0.0
    else:
        output[outside, :] = 0.0
    return np.clip(output, 0.0, 1.0).astype(np.float32)


def multiply_alpha_by_mask(pixels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return RGBA pixels whose alpha channel is multiplied by a soft mask."""
    _validate_rgba_array(pixels, "pixels")
    _validate_mask_array(mask, "mask")
    if mask.shape != pixels.shape[:2]:
        raise ValueError("mask shape must match pixel array dimensions")
    output = np.array(pixels, dtype=np.float32, copy=True)
    output[..., 3] *= np.clip(mask.astype(np.float32, copy=False), 0.0, 1.0)
    return np.clip(output, 0.0, 1.0).astype(np.float32)


def source_over(destination: np.ndarray, source: np.ndarray, *, source_opacity: float = 1.0) -> np.ndarray:
    """Composite one straight-alpha RGBA array over another."""
    _validate_rgba_array(destination, "destination")
    _validate_rgba_array(source, "source")
    if destination.shape != source.shape:
        raise ValueError("destination and source must have the same shape")
    if isinstance(source_opacity, bool) or not isinstance(source_opacity, (int, float)):
        raise TypeError("source_opacity must be a number")
    if not np.isfinite(float(source_opacity)) or source_opacity < 0.0 or source_opacity > 1.0:
        raise ValueError("source_opacity must be in [0, 1]")

    destination = destination.astype(np.float32, copy=False)
    source = source.astype(np.float32, copy=False)
    source_alpha = source[..., 3:4] * np.float32(source_opacity)
    destination_alpha = destination[..., 3:4]
    output_alpha = source_alpha + destination_alpha * (1.0 - source_alpha)
    output_premultiplied = (
        source[..., :3] * source_alpha
        + destination[..., :3] * destination_alpha * (1.0 - source_alpha)
    )
    output_rgb = np.zeros_like(destination[..., :3])
    np.divide(output_premultiplied, output_alpha, out=output_rgb, where=output_alpha > 0.0)
    output = np.concatenate([output_rgb, output_alpha], axis=2)
    return np.clip(output, 0.0, 1.0).astype(np.float32)


def changed_bbox(before_pixels: np.ndarray, after_pixels: np.ndarray, *, tolerance: float = 0.0) -> Optional[BBoxXYXY]:
    """Return the bbox of pixels whose RGBA values changed beyond tolerance."""
    _validate_rgba_array(before_pixels, "before_pixels")
    _validate_rgba_array(after_pixels, "after_pixels")
    if before_pixels.shape != after_pixels.shape:
        raise ValueError("before_pixels and after_pixels must have the same shape")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not np.isfinite(float(tolerance)):
        raise TypeError("tolerance must be a finite number")
    if tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative")
    delta = np.max(np.abs(after_pixels.astype(np.float32, copy=False) - before_pixels.astype(np.float32, copy=False)), axis=2)
    ys, xs = np.nonzero(delta > float(tolerance))
    if xs.size == 0:
        return None
    return BBoxXYXY(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def changed_pixels_outside_mask(before_pixels: np.ndarray, after_pixels: np.ndarray, mask: np.ndarray, *, tolerance: float = 0.0) -> int:
    """Count changed pixels outside a write mask."""
    _validate_rgba_array(before_pixels, "before_pixels")
    _validate_rgba_array(after_pixels, "after_pixels")
    _validate_mask_array(mask, "mask")
    if before_pixels.shape != after_pixels.shape:
        raise ValueError("before_pixels and after_pixels must have the same shape")
    if mask.shape != before_pixels.shape[:2]:
        raise ValueError("mask shape must match pixel array dimensions")
    delta = np.max(np.abs(after_pixels.astype(np.float32, copy=False) - before_pixels.astype(np.float32, copy=False)), axis=2)
    return int(np.count_nonzero((delta > float(tolerance)) & (mask <= 0.0)))


def max_delta_outside_mask(before_pixels: np.ndarray, after_pixels: np.ndarray, mask: np.ndarray) -> float:
    """Return the largest RGBA channel delta outside a write mask."""
    _validate_rgba_array(before_pixels, "before_pixels")
    _validate_rgba_array(after_pixels, "after_pixels")
    _validate_mask_array(mask, "mask")
    if before_pixels.shape != after_pixels.shape:
        raise ValueError("before_pixels and after_pixels must have the same shape")
    if mask.shape != before_pixels.shape[:2]:
        raise ValueError("mask shape must match pixel array dimensions")

    protected = mask <= 0.0
    if not bool(np.any(protected)):
        return 0.0
    deltas = np.abs(after_pixels.astype(np.float32, copy=False)[protected] - before_pixels.astype(np.float32, copy=False)[protected])
    return float(deltas.max()) if deltas.size else 0.0


def _mask_crop_for_bbox(mask: np.ndarray, bbox: BBoxXYXY, width: int, height: int) -> np.ndarray:
    _validate_mask_array(mask, "mask")
    if mask.shape == (bbox.height, bbox.width):
        return np.clip(mask.astype(np.float32, copy=False), 0.0, 1.0)
    if mask.shape == (height, width):
        y_slice, x_slice = bbox.to_slices()
        return np.clip(mask[y_slice, x_slice].astype(np.float32, copy=False), 0.0, 1.0)
    raise ValueError("mask must be either crop-sized or full-canvas-sized")


def _validate_crop_shape(crop: np.ndarray, bbox: BBoxXYXY, field_name: str) -> None:
    if crop.shape[:2] != (bbox.height, bbox.width):
        raise ValueError(f"{field_name} shape must match bbox dimensions")


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


def _validate_fill_color(fill_color: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if not isinstance(fill_color, tuple) or len(fill_color) != 4:
        raise TypeError("fill_color must be a four-channel RGBA tuple")
    values = tuple(float(channel) for channel in fill_color)
    if any(not np.isfinite(channel) or channel < 0.0 or channel > 1.0 for channel in values):
        raise ValueError("fill_color channels must be finite values in [0, 1]")
    return values
