"""Deterministic cleanup helpers for diffusion candidate pixels."""

from __future__ import annotations

import numpy as np

from ai_edit_kernel.region import BBoxXYXY, apply_crop_with_mask, expand_crop_to_canvas, multiply_alpha_by_mask


def fit_rgba_to_size(pixels: np.ndarray, width: int, height: int) -> np.ndarray:
    """Center crop or pad candidate pixels to an exact RGBA size."""
    _validate_rgba(pixels)
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive integer")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError("height must be a positive integer")
    source = pixels.astype(np.float32, copy=False)
    if source.shape[:2] == (height, width):
        return np.clip(source, 0.0, 1.0).astype(np.float32, copy=True)

    output = np.zeros((height, width, 4), dtype=np.float32)
    src_y, dst_y, copy_h = _centered_copy_axis(source.shape[0], height)
    src_x, dst_x, copy_w = _centered_copy_axis(source.shape[1], width)
    output[dst_y : dst_y + copy_h, dst_x : dst_x + copy_w, :] = source[src_y : src_y + copy_h, src_x : src_x + copy_w, :]
    return np.clip(output, 0.0, 1.0).astype(np.float32)


def clip_generated_crop(pixels: np.ndarray, mask_crop: np.ndarray) -> np.ndarray:
    """Return candidate crop pixels with alpha multiplied by the write mask."""
    clipped = multiply_alpha_by_mask(pixels, mask_crop)
    clipped[mask_crop <= 0.0, :] = 0.0
    return clipped


def composite_generated_crop(before_pixels: np.ndarray, generated_crop: np.ndarray, bbox: BBoxXYXY, mask_crop: np.ndarray) -> np.ndarray:
    """Composite candidate crop into a target layer through a soft write mask."""
    return apply_crop_with_mask(before_pixels, generated_crop, bbox, mask_crop)


def generated_crop_to_layer(pixels: np.ndarray, bbox: BBoxXYXY, width: int, height: int, mask_crop: np.ndarray) -> np.ndarray:
    """Place a masked candidate crop on a transparent full-canvas layer."""
    clipped = clip_generated_crop(pixels, mask_crop)
    return expand_crop_to_canvas(clipped, bbox, width=width, height=height)


def _centered_copy_axis(old_size: int, new_size: int) -> tuple[int, int, int]:
    copy_size = min(old_size, new_size)
    source_start = max((old_size - new_size) // 2, 0)
    destination_start = max((new_size - old_size) // 2, 0)
    return source_start, destination_start, copy_size


def _validate_rgba(pixels: np.ndarray) -> None:
    if not isinstance(pixels, np.ndarray):
        raise TypeError("pixels must be a NumPy array")
    if pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError("pixels must have shape H x W x 4")
