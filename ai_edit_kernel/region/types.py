"""Shared region data structures for non-mutating document work areas.

The editing kernel stores layers and masks as full-canvas arrays, but many
operations act on a bounded part of that canvas: selections, copy/paste,
filters, validation, perception crops, and diffusion jobs. The classes in this
module describe those temporary work areas without implying any mutation of the
document itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True, slots=True)
class BBoxXYXY:
    """Integer half-open rectangle in `[x0, y0, x1, y1]` order.

    `x1` and `y1` are exclusive bounds. Empty boxes are representable so helpers
    can return a safe value for empty intersections, but extraction and most edit
    operations should call `require_non_empty()` before using a box as a work
    area.
    """

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        for field_name in ("x0", "y0", "x1", "y1"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
        if self.x1 < self.x0:
            raise ValueError("x1 must be greater than or equal to x0")
        if self.y1 < self.y0:
            raise ValueError("y1 must be greater than or equal to y0")

    @classmethod
    def from_iterable(cls, value: Any, field_name: str = "bbox_xyxy") -> "BBoxXYXY":
        """Validate and convert a four-integer iterable to a bbox."""
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise TypeError(f"{field_name} must be a four-number list")
        coords: list[int] = []
        for index, item in enumerate(value):
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise TypeError(f"{field_name}[{index}] must be a number")
            if not np.isfinite(float(item)):
                raise ValueError(f"{field_name}[{index}] must be finite")
            if float(item) != int(item):
                raise ValueError(f"{field_name}[{index}] must be an integer pixel coordinate")
            coords.append(int(item))
        return cls(coords[0], coords[1], coords[2], coords[3])

    @property
    def width(self) -> int:
        """Return `x1 - x0`."""
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        """Return `y1 - y0`."""
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        """Return the number of pixels covered by this half-open rectangle."""
        return self.width * self.height

    @property
    def is_empty(self) -> bool:
        """Return whether this rectangle has zero width or height."""
        return self.width == 0 or self.height == 0

    def require_non_empty(self, field_name: str = "bbox_xyxy") -> "BBoxXYXY":
        """Return this bbox, or raise if it has no area."""
        if self.is_empty:
            raise ValueError(f"{field_name} must be non-empty")
        return self

    def as_tuple(self) -> tuple[int, int, int, int]:
        """Return the bbox as an immutable tuple."""
        return (self.x0, self.y0, self.x1, self.y1)

    def as_list(self) -> list[int]:
        """Return the bbox as a JSON-friendly list."""
        return [self.x0, self.y0, self.x1, self.y1]

    def to_slices(self) -> tuple[slice, slice]:
        """Return NumPy slices in row-major `(y, x)` order."""
        return (slice(self.y0, self.y1), slice(self.x0, self.x1))

    def translated(self, dx: int, dy: int) -> "BBoxXYXY":
        """Return this bbox shifted by integer offsets."""
        if isinstance(dx, bool) or not isinstance(dx, int):
            raise TypeError("dx must be an integer")
        if isinstance(dy, bool) or not isinstance(dy, int):
            raise TypeError("dy must be an integer")
        return BBoxXYXY(self.x0 + dx, self.y0 + dy, self.x1 + dx, self.y1 + dy)


@dataclass(frozen=True, slots=True)
class RegionSpec:
    """Declarative request for a temporary document work area.

    A spec may refer to an explicit bbox, a document mask, the active selection,
    or the full canvas. It describes how a caller wants a region resolved; it
    does not store extracted pixels and does not mutate the document.
    """

    bbox: Optional[BBoxXYXY] = None
    mask_id: Optional[str] = None
    padding: int | tuple[int, int, int, int] = 0
    use_active_selection: bool = False
    default_full_canvas: bool = False
    mask_threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.bbox is not None and not isinstance(self.bbox, BBoxXYXY):
            raise TypeError("bbox must be a BBoxXYXY or None")
        if self.mask_id is not None and not isinstance(self.mask_id, str):
            raise TypeError("mask_id must be a string or None")
        if not isinstance(self.use_active_selection, bool):
            raise TypeError("use_active_selection must be a bool")
        if not isinstance(self.default_full_canvas, bool):
            raise TypeError("default_full_canvas must be a bool")
        if isinstance(self.mask_threshold, bool) or not isinstance(self.mask_threshold, (int, float)):
            raise TypeError("mask_threshold must be a number")
        if not np.isfinite(float(self.mask_threshold)):
            raise ValueError("mask_threshold must be finite")


@dataclass(slots=True)
class RegionView:
    """Extracted, non-mutating view of a document region.

    The bbox records where the crop belongs on the full document canvas. Optional
    arrays are independent float32 copies so callers can safely pass them to
    backends, validators, or trace writers without accidentally mutating the
    source document.
    """

    canvas_width: int
    canvas_height: int
    bbox: BBoxXYXY
    mask: Optional[np.ndarray] = None
    preview: Optional[np.ndarray] = None
    layer_pixels: Optional[np.ndarray] = None
    source_layer_id: Optional[str] = None
    source_mask_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        """Return the crop width."""
        return self.bbox.width

    @property
    def height(self) -> int:
        """Return the crop height."""
        return self.bbox.height

    @property
    def shape_hw(self) -> tuple[int, int]:
        """Return the crop shape as `(height, width)`."""
        return (self.height, self.width)

    def placement(self) -> "RegionPlacement":
        """Return a placement object for mapping crop pixels back to the canvas."""
        return RegionPlacement(self.canvas_width, self.canvas_height, self.bbox)


@dataclass(frozen=True, slots=True)
class RegionPlacement:
    """Mapping from a crop-sized array back into full-canvas coordinates."""

    canvas_width: int
    canvas_height: int
    bbox: BBoxXYXY

    def __post_init__(self) -> None:
        if isinstance(self.canvas_width, bool) or not isinstance(self.canvas_width, int):
            raise TypeError("canvas_width must be an integer")
        if isinstance(self.canvas_height, bool) or not isinstance(self.canvas_height, int):
            raise TypeError("canvas_height must be an integer")
        if self.canvas_width <= 0 or self.canvas_height <= 0:
            raise ValueError("canvas dimensions must be positive")
        if self.bbox.x0 < 0 or self.bbox.y0 < 0 or self.bbox.x1 > self.canvas_width or self.bbox.y1 > self.canvas_height:
            raise ValueError("bbox must be inside the canvas")

    @property
    def crop_shape_hw(self) -> tuple[int, int]:
        """Return the crop shape this placement expects."""
        return (self.bbox.height, self.bbox.width)
