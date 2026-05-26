"""Mask and selection model for the AI Editing Kernel.

Masks are full-canvas spatial controls. They are used for selections, layer
alpha, object regions, diffusion regions, and write guards. Every mask stores a
`height x width` NumPy array of `float32` values in `[0, 1]`: `0` means fully
excluded or protected, `1` means fully included or editable, and intermediate
values represent antialiasing or feathered transitions.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

try:
    from scipy import ndimage as _ndimage
except ImportError:  # pragma: no cover - exercised only when SciPy is absent
    _ndimage = None


class MaskKind(str, Enum):
    """Semantic role of a mask."""

    SELECTION = "selection"
    WRITE_GUARD = "write_guard"
    LAYER_ALPHA = "layer_alpha"
    OBJECT = "object"
    SHAPE = "shape"
    LINE_ART_REGION = "line_art_region"
    DIFFUSION = "diffusion"


@dataclass(slots=True)
class MaskStats:
    """Geometric and numerical properties of a mask.

    `bbox` is `(x, y, width, height)` over all pixels whose value is greater than
    zero. Empty masks use `(0, 0, 0, 0)` and centroid `(0.0, 0.0)`.
    """

    area_pixels: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    min_value: float
    max_value: float
    is_binary: bool


@dataclass(slots=True)
class Mask:
    """A full-canvas floating point mask.

    Hard masks contain only `0.0` and `1.0`. Soft masks may contain intermediate
    values and are used for antialiasing, feathered selections, diffusion blends,
    and gradual write guards.
    """

    id: str
    name: str
    data: np.ndarray
    kind: MaskKind = MaskKind.SELECTION
    hard: bool = True
    source: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, canvas_width: int, canvas_height: int) -> None:
        """Validate identity fields, shape, dtype, range, and canvas compatibility."""
        _validate_positive_int(canvas_width, "canvas_width")
        _validate_positive_int(canvas_height, "canvas_height")
        _validate_identifier(self.id, "mask.id")
        if not isinstance(self.name, str):
            raise TypeError("mask.name must be a string")
        _validate_enum_member(self.kind, MaskKind, "mask.kind")
        if not isinstance(self.hard, bool):
            raise TypeError("mask.hard must be a bool")
        if self.source is not None and not isinstance(self.source, str):
            raise TypeError("mask.source must be a string or None")
        if not isinstance(self.metadata, dict):
            raise TypeError("mask.metadata must be a dictionary")

        _validate_mask_array(self.data, canvas_width, canvas_height, "mask.data")
        if self.hard and not _is_binary_array(self.data):
            raise ValueError("hard masks must contain only 0.0 and 1.0 values")

    def stats(self) -> MaskStats:
        """Compute area, bounding box, centroid, min/max, and binary status."""
        _validate_mask_array_no_canvas(self.data, "mask.data")
        included_y, included_x = np.nonzero(self.data > 0.0)
        area = int(included_x.size)

        if area == 0:
            bbox = (0, 0, 0, 0)
            centroid = (0.0, 0.0)
        else:
            x_min = int(included_x.min())
            x_max = int(included_x.max())
            y_min = int(included_y.min())
            y_max = int(included_y.max())
            bbox = (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
            centroid = (float(included_x.mean()), float(included_y.mean()))

        return MaskStats(
            area_pixels=area,
            bbox=bbox,
            centroid=centroid,
            min_value=float(self.data.min()) if self.data.size else 0.0,
            max_value=float(self.data.max()) if self.data.size else 0.0,
            is_binary=_is_binary_array(self.data),
        )

    def clone(self, new_id: str, new_name: Optional[str] = None) -> "Mask":
        """Return an independent copy of this mask and its metadata."""
        _validate_identifier(new_id, "new_id")
        if new_name is not None and not isinstance(new_name, str):
            raise TypeError("new_name must be a string or None")

        return Mask(
            id=new_id,
            name=self.name if new_name is None else new_name,
            data=np.array(self.data, dtype=np.float32, copy=True),
            kind=self.kind,
            hard=self.hard,
            source=self.source,
            metadata=deepcopy(self.metadata),
        )

    def threshold(self, threshold: float = 0.5, new_id: Optional[str] = None) -> "Mask":
        """Return a hard binary mask using `data >= threshold`.

        If `new_id` is omitted, the returned mask keeps this mask's ID. Callers
        inserting the result beside the original in a document should pass a new
        ID.
        """
        _validate_unit_float(threshold, "threshold")
        output_id = self.id if new_id is None else new_id
        _validate_identifier(output_id, "new_id")

        return Mask(
            id=output_id,
            name=self.name,
            data=(self.data >= float(threshold)).astype(np.float32),
            kind=self.kind,
            hard=True,
            source=self.source,
            metadata=deepcopy(self.metadata),
        )

    def invert(self, new_id: str, name: Optional[str] = None) -> "Mask":
        """Return a mask where included and excluded areas are swapped."""
        _validate_identifier(new_id, "new_id")
        output_name = _derived_name(name, f"{self.name} inverted")
        return self._copy_with(new_id, output_name, 1.0 - self.data, hard=self.hard)

    def union(self, other: "Mask", new_id: str, name: Optional[str] = None) -> "Mask":
        """Return a mask containing pixels included by either mask."""
        _validate_compatible_masks(self, other)
        output_name = _derived_name(name, f"{self.name} union {other.name}")
        data = np.maximum(self.data, other.data).astype(np.float32, copy=False)
        return self._copy_with(new_id, output_name, data, hard=self.hard and other.hard)

    def intersect(self, other: "Mask", new_id: str, name: Optional[str] = None) -> "Mask":
        """Return a mask containing only pixels included by both masks."""
        _validate_compatible_masks(self, other)
        output_name = _derived_name(name, f"{self.name} intersect {other.name}")
        data = np.minimum(self.data, other.data).astype(np.float32, copy=False)
        return self._copy_with(new_id, output_name, data, hard=self.hard and other.hard)

    def subtract(self, other: "Mask", new_id: str, name: Optional[str] = None) -> "Mask":
        """Return this mask with the other mask's included area removed.

        For soft masks, subtraction uses the conservative alpha-style operation
        `A * (1 - B)`. This never increases editability and keeps write guards on
        the safer side when the subtracting mask has feathered edges.
        """
        _validate_compatible_masks(self, other)
        output_name = _derived_name(name, f"{self.name} subtract {other.name}")
        data = (self.data * (1.0 - other.data)).astype(np.float32, copy=False)
        return self._copy_with(new_id, output_name, data, hard=self.hard and other.hard)

    def dilate(self, pixels: int, new_id: str, name: Optional[str] = None) -> "Mask":
        """Grow the included region by `pixels` using a disk-shaped footprint."""
        _validate_nonnegative_int(pixels, "pixels")
        _require_scipy("dilate")
        output_name = _derived_name(name, f"{self.name} dilated {pixels}px")
        if pixels == 0:
            return self.clone(new_id, output_name)

        footprint = _disk_footprint(pixels)
        if self.hard:
            data = _ndimage.binary_dilation(self.data > 0.0, structure=footprint).astype(np.float32)
            hard = True
        else:
            data = _ndimage.grey_dilation(self.data, footprint=footprint).astype(np.float32)
            hard = False
        return self._copy_with(new_id, output_name, np.clip(data, 0.0, 1.0), hard=hard)

    def erode(self, pixels: int, new_id: str, name: Optional[str] = None) -> "Mask":
        """Shrink the included region by `pixels` using a disk-shaped footprint."""
        _validate_nonnegative_int(pixels, "pixels")
        _require_scipy("erode")
        output_name = _derived_name(name, f"{self.name} eroded {pixels}px")
        if pixels == 0:
            return self.clone(new_id, output_name)

        footprint = _disk_footprint(pixels)
        if self.hard:
            data = _ndimage.binary_erosion(self.data > 0.0, structure=footprint).astype(np.float32)
            hard = True
        else:
            data = _ndimage.grey_erosion(self.data, footprint=footprint).astype(np.float32)
            hard = False
        return self._copy_with(new_id, output_name, np.clip(data, 0.0, 1.0), hard=hard)

    def feather(self, radius: float, new_id: str, name: Optional[str] = None) -> "Mask":
        """Return a softened mask by applying a Gaussian blur.

        `radius` is used as the Gaussian sigma in pixels. The result is marked as
        soft whenever `radius` is greater than zero.
        """
        _validate_nonnegative_number(radius, "radius")
        _require_scipy("feather")
        output_name = _derived_name(name, f"{self.name} feathered {radius:g}px")
        if radius == 0:
            return self.clone(new_id, output_name)

        data = _ndimage.gaussian_filter(self.data, sigma=float(radius), mode="nearest")
        return self._copy_with(new_id, output_name, np.clip(data, 0.0, 1.0), hard=False)

    def as_write_guard(self, new_id: str, name: Optional[str] = None) -> "Mask":
        """Return an independent copy marked as a write guard."""
        output_name = _derived_name(name, f"{self.name} write guard")
        mask = self.clone(new_id, output_name)
        mask.kind = MaskKind.WRITE_GUARD
        return mask

    @classmethod
    def full_canvas(cls, id: str, width: int, height: int, name: str = "full canvas") -> "Mask":
        """Create a hard write guard that allows editing every canvas pixel."""
        _validate_identifier(id, "id")
        _validate_positive_int(width, "width")
        _validate_positive_int(height, "height")
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        return cls(
            id=id,
            name=name,
            data=np.ones((height, width), dtype=np.float32),
            kind=MaskKind.WRITE_GUARD,
            hard=True,
        )

    @classmethod
    def empty(cls, id: str, width: int, height: int, name: str = "empty") -> "Mask":
        """Create a hard write guard that protects every canvas pixel."""
        _validate_identifier(id, "id")
        _validate_positive_int(width, "width")
        _validate_positive_int(height, "height")
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        return cls(
            id=id,
            name=name,
            data=np.zeros((height, width), dtype=np.float32),
            kind=MaskKind.WRITE_GUARD,
            hard=True,
        )

    def _copy_with(self, new_id: str, name: str, data: np.ndarray, hard: bool) -> "Mask":
        """Create a derived mask while preserving semantic metadata."""
        _validate_identifier(new_id, "new_id")
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        copied = np.array(data, dtype=np.float32, copy=True)
        _validate_mask_array_no_canvas(copied, "derived mask data")
        return Mask(
            id=new_id,
            name=name,
            data=copied,
            kind=self.kind,
            hard=hard,
            source=self.source,
            metadata=deepcopy(self.metadata),
        )


def _validate_identifier(value: Any, field_name: str) -> None:
    """Validate a caller-provided ID or reference field."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if value == "":
        raise ValueError(f"{field_name} must not be empty")


def _validate_positive_int(value: Any, field_name: str) -> None:
    """Validate a positive integer while rejecting bools."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero")


def _validate_nonnegative_int(value: Any, field_name: str) -> None:
    """Validate a nonnegative integer while rejecting bools."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must not be negative")


def _validate_nonnegative_number(value: Any, field_name: str) -> None:
    """Validate a finite nonnegative scalar number."""
    if not _is_real_number(value):
        raise TypeError(f"{field_name} must be a number")
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{field_name} must be finite and nonnegative")


def _validate_unit_float(value: Any, field_name: str) -> None:
    """Validate a finite scalar in `[0, 1]`."""
    if not _is_real_number(value):
        raise TypeError(f"{field_name} must be a number")
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{field_name} must be finite and in [0, 1]")


def _validate_enum_member(value: Any, enum_type: type[Enum], field_name: str) -> None:
    """Validate an enum value while accepting its raw string value."""
    try:
        enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = [member.value for member in enum_type]
        raise ValueError(f"{field_name} must be one of {allowed!r}") from exc


def _validate_mask_array(array: Any, width: int, height: int, field_name: str) -> None:
    """Validate canonical full-canvas mask storage."""
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{field_name} must be a NumPy array")
    if array.shape != (height, width):
        raise ValueError(f"{field_name} must have shape {(height, width)!r}")
    _validate_mask_array_no_canvas(array, field_name)


def _validate_mask_array_no_canvas(array: Any, field_name: str) -> None:
    """Validate mask dtype, rank, finite values, and value range."""
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{field_name} must be a NumPy array")
    if array.ndim != 2:
        raise ValueError(f"{field_name} must be a 2D array")
    if array.dtype != np.float32:
        raise TypeError(f"{field_name} must have dtype float32")
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{field_name} must contain only finite values")
    if bool(np.any((array < 0.0) | (array > 1.0))):
        raise ValueError(f"{field_name} values must be in [0, 1]")


def _validate_compatible_masks(left: Mask, right: Mask) -> None:
    """Validate two masks before pairwise mask algebra."""
    if not isinstance(right, Mask):
        raise TypeError("other must be a Mask")
    _validate_mask_array_no_canvas(left.data, "mask.data")
    _validate_mask_array_no_canvas(right.data, "other.data")
    if left.data.shape != right.data.shape:
        raise ValueError("masks must have the same shape")


def _is_binary_array(array: np.ndarray) -> bool:
    """Return whether an array contains only values close to 0 or 1."""
    return bool(np.all(np.isclose(array, 0.0) | np.isclose(array, 1.0)))


def _disk_footprint(radius: int) -> np.ndarray:
    """Return a boolean disk footprint with the requested integer radius."""
    if radius == 0:
        return np.ones((1, 1), dtype=bool)
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= radius * radius


def _derived_name(name: Optional[str], fallback: str) -> str:
    """Return a caller-provided name or a deterministic derived fallback."""
    if name is None:
        return fallback
    if not isinstance(name, str):
        raise TypeError("name must be a string or None")
    return name


def _require_scipy(operation: str) -> None:
    """Raise a clear error if a SciPy-backed operation is unavailable."""
    if _ndimage is None:
        raise RuntimeError(f"Mask.{operation}() requires scipy.ndimage")


def _is_real_number(value: Any) -> bool:
    """Return whether `value` is a real scalar number and not a bool."""
    return not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, float, np.integer, np.floating))
