"""Geometry helpers for document regions and masks.

All bboxes use the kernel's canonical `bbox_xyxy` convention: integer half-open
pixel bounds `[x0, y0, x1, y1]`. Helpers in this module are pure and never
mutate documents, layers, masks, or pixel arrays.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Optional

import numpy as np

from ai_edit_kernel.region.types import BBoxXYXY

RoundingMode = Literal["error", "floor_ceil", "round"]


def full_canvas_bbox(width: int, height: int) -> BBoxXYXY:
    """Return the non-empty bbox covering the whole canvas."""
    _validate_canvas_size(width, height)
    return BBoxXYXY(0, 0, width, height)


def bbox_from_xyxy(
    value: Any,
    width: Optional[int] = None,
    height: Optional[int] = None,
    *,
    allow_empty: bool = False,
    clip: bool = False,
    rounding: RoundingMode = "error",
    field_name: str = "bbox_xyxy",
) -> BBoxXYXY:
    """Validate user or model-provided `bbox_xyxy` data.

    `rounding="error"` matches the current executor behavior and rejects
    fractional coordinates. `rounding="floor_ceil"` is useful for detector or
    VLM output where an inclusive-looking floating box should conservatively
    cover all touched pixels.
    """
    bbox = _coerce_bbox(value, rounding, field_name)
    if width is not None or height is not None:
        if width is None or height is None:
            raise ValueError("width and height must be provided together")
        _validate_canvas_size(width, height)
        if clip:
            bbox = clip_bbox(bbox, width, height)
        elif bbox.x0 < 0 or bbox.y0 < 0 or bbox.x1 > width or bbox.y1 > height:
            raise ValueError(f"{field_name} must be inside the canvas")
    if not allow_empty:
        bbox.require_non_empty(field_name)
    return bbox


def bbox_from_xywh(
    value: Any,
    width: Optional[int] = None,
    height: Optional[int] = None,
    *,
    allow_empty: bool = False,
    clip: bool = False,
    rounding: RoundingMode = "error",
    field_name: str = "bbox_xywh",
) -> BBoxXYXY:
    """Validate `(x, y, width, height)` data and return `BBoxXYXY`."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise TypeError(f"{field_name} must be a four-number list")
    x, y, box_width, box_height = _coerce_numbers(value, rounding, field_name)
    if box_width < 0 or box_height < 0:
        raise ValueError(f"{field_name} width and height must be nonnegative")
    return bbox_from_xyxy(
        [x, y, x + box_width, y + box_height],
        width,
        height,
        allow_empty=allow_empty,
        clip=clip,
        rounding="error",
        field_name=field_name,
    )


def clip_bbox(bbox: BBoxXYXY, width: int, height: int) -> BBoxXYXY:
    """Clip a bbox to canvas bounds, returning an empty bbox if it misses."""
    _validate_bbox_instance(bbox)
    _validate_canvas_size(width, height)
    x0 = min(max(bbox.x0, 0), width)
    y0 = min(max(bbox.y0, 0), height)
    x1 = min(max(bbox.x1, 0), width)
    y1 = min(max(bbox.y1, 0), height)
    if x1 < x0:
        x1 = x0
    if y1 < y0:
        y1 = y0
    return BBoxXYXY(x0, y0, x1, y1)


def pad_bbox(bbox: BBoxXYXY, padding: int | tuple[int, int, int, int], width: int, height: int) -> BBoxXYXY:
    """Expand a bbox by padding and clip it to the canvas.

    Tuple padding is `(left, top, right, bottom)`.
    """
    _validate_bbox_instance(bbox)
    bbox.require_non_empty()
    left, top, right, bottom = normalize_padding(padding)
    padded = BBoxXYXY(bbox.x0 - left, bbox.y0 - top, bbox.x1 + right, bbox.y1 + bottom)
    return clip_bbox(padded, width, height).require_non_empty("padded bbox")


def snap_bbox_to_multiple(
    bbox: BBoxXYXY,
    width: int,
    height: int,
    multiple: int,
    *,
    min_width: int = 1,
    min_height: int = 1,
) -> BBoxXYXY:
    """Expand a bbox so its size is compatible with backend tile constraints.

    The snapped box stays inside the canvas and remains centered on the input as
    much as possible. If the canvas itself is smaller than the requested multiple,
    the canvas dimension wins.
    """
    _validate_bbox_instance(bbox)
    bbox = bbox_from_xyxy(bbox.as_list(), width, height)
    if isinstance(multiple, bool) or not isinstance(multiple, int) or multiple <= 0:
        raise ValueError("multiple must be a positive integer")
    if isinstance(min_width, bool) or not isinstance(min_width, int) or min_width <= 0:
        raise ValueError("min_width must be a positive integer")
    if isinstance(min_height, bool) or not isinstance(min_height, int) or min_height <= 0:
        raise ValueError("min_height must be a positive integer")

    target_width = _ceil_to_multiple(max(bbox.width, min_width), multiple)
    target_height = _ceil_to_multiple(max(bbox.height, min_height), multiple)
    target_width = min(target_width, width)
    target_height = min(target_height, height)

    center_x = (bbox.x0 + bbox.x1) / 2.0
    center_y = (bbox.y0 + bbox.y1) / 2.0
    x0, x1 = _fit_interval(center_x, target_width, width)
    y0, y1 = _fit_interval(center_y, target_height, height)
    return BBoxXYXY(x0, y0, x1, y1).require_non_empty("snapped bbox")


def intersect_bbox(left: BBoxXYXY, right: BBoxXYXY) -> BBoxXYXY:
    """Return the intersection of two bboxes, possibly empty."""
    _validate_bbox_instance(left)
    _validate_bbox_instance(right)
    x0 = max(left.x0, right.x0)
    y0 = max(left.y0, right.y0)
    x1 = min(left.x1, right.x1)
    y1 = min(left.y1, right.y1)
    if x1 < x0:
        x1 = x0
    if y1 < y0:
        y1 = y0
    return BBoxXYXY(x0, y0, x1, y1)


def union_bbox(boxes: list[BBoxXYXY] | tuple[BBoxXYXY, ...], *, allow_empty: bool = False) -> BBoxXYXY:
    """Return the smallest bbox containing all non-empty input boxes."""
    if not boxes:
        raise ValueError("boxes must not be empty")
    non_empty: list[BBoxXYXY] = []
    for box in boxes:
        _validate_bbox_instance(box)
        if not box.is_empty:
            non_empty.append(box)
    if not non_empty:
        if allow_empty:
            return BBoxXYXY(0, 0, 0, 0)
        raise ValueError("at least one bbox must be non-empty")
    return BBoxXYXY(
        min(box.x0 for box in non_empty),
        min(box.y0 for box in non_empty),
        max(box.x1 for box in non_empty),
        max(box.y1 for box in non_empty),
    )


def bbox_from_mask(mask: np.ndarray, threshold: float = 0.0) -> Optional[BBoxXYXY]:
    """Return the bbox of mask values greater than `threshold`, or `None`."""
    _validate_mask_array(mask, "mask")
    _validate_threshold(threshold)
    included_y, included_x = np.nonzero(mask > float(threshold))
    if included_x.size == 0:
        return None
    return BBoxXYXY(
        int(included_x.min()),
        int(included_y.min()),
        int(included_x.max()) + 1,
        int(included_y.max()) + 1,
    )


def content_bbox_rgba(pixels: np.ndarray, alpha_threshold: float = 0.0) -> Optional[BBoxXYXY]:
    """Return the bbox of RGBA pixels whose alpha is greater than a threshold."""
    _validate_rgba_array(pixels, "pixels")
    _validate_threshold(alpha_threshold)
    return bbox_from_mask(pixels[..., 3], alpha_threshold)


def normalize_padding(padding: int | tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Return padding as `(left, top, right, bottom)`."""
    if isinstance(padding, bool):
        raise TypeError("padding must be an integer or four-integer tuple")
    if isinstance(padding, int):
        if padding < 0:
            raise ValueError("padding must be nonnegative")
        return (padding, padding, padding, padding)
    if not isinstance(padding, tuple) or len(padding) != 4:
        raise TypeError("padding must be an integer or four-integer tuple")
    values: list[int] = []
    for index, value in enumerate(padding):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"padding[{index}] must be an integer")
        if value < 0:
            raise ValueError("padding entries must be nonnegative")
        values.append(value)
    return (values[0], values[1], values[2], values[3])


def _coerce_bbox(value: Any, rounding: RoundingMode, field_name: str) -> BBoxXYXY:
    if isinstance(value, BBoxXYXY):
        return value
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise TypeError(f"{field_name} must be a four-number list")
    x0, y0, x1, y1 = _coerce_numbers(value, rounding, field_name)
    return BBoxXYXY(x0, y0, x1, y1)


def _coerce_numbers(value: Any, rounding: RoundingMode, field_name: str) -> tuple[int, int, int, int]:
    if rounding not in {"error", "floor_ceil", "round"}:
        raise ValueError("rounding must be 'error', 'floor_ceil', or 'round'")
    coords: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{field_name}[{index}] must be a number")
        number = float(item)
        if not np.isfinite(number):
            raise ValueError(f"{field_name}[{index}] must be finite")
        if rounding == "error":
            if number != int(number):
                raise ValueError(f"{field_name}[{index}] must be an integer pixel coordinate")
            coords.append(int(number))
        elif rounding == "round":
            coords.append(int(round(number)))
        elif index in {0, 1}:
            coords.append(int(math.floor(number)))
        else:
            coords.append(int(math.ceil(number)))
    return (coords[0], coords[1], coords[2], coords[3])


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _fit_interval(center: float, size: int, maximum: int) -> tuple[int, int]:
    if size >= maximum:
        return (0, maximum)
    start = int(math.floor(center - size / 2.0))
    end = start + size
    if start < 0:
        end -= start
        start = 0
    if end > maximum:
        start -= end - maximum
        end = maximum
    return (start, end)


def _validate_bbox_instance(bbox: BBoxXYXY) -> None:
    if not isinstance(bbox, BBoxXYXY):
        raise TypeError("bbox must be a BBoxXYXY")


def _validate_canvas_size(width: int, height: int) -> None:
    if isinstance(width, bool) or not isinstance(width, int):
        raise TypeError("width must be an integer")
    if isinstance(height, bool) or not isinstance(height, int):
        raise TypeError("height must be an integer")
    if width <= 0 or height <= 0:
        raise ValueError("canvas dimensions must be positive")


def _validate_threshold(threshold: float) -> None:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError("threshold must be a number")
    if not np.isfinite(float(threshold)):
        raise ValueError("threshold must be finite")


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
