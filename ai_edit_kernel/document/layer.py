"""Layer model for the AI Editing Kernel.

A `Layer` is one editable plane in a `DocumentState`. The kernel's document
contract uses full-canvas raster storage: renderable pixel data is a
`height x width x 4` NumPy array of straight-alpha RGBA `float32` values in
`[0, 1]`.

Layers are ordered by `DocumentState.layers` from bottom to top. Raster layers
carry pixels directly. Group layers are organizational metadata and do not carry
pixel data. Vector, text, and adjustment layers may be metadata-only or may carry
a rasterized full-canvas pixel representation when an executor or adapter needs
them to participate in preview compositing.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


class LayerKind(str, Enum):
    """Supported layer categories."""

    RASTER = "raster"
    VECTOR = "vector"
    TEXT = "text"
    GROUP = "group"
    ADJUSTMENT = "adjustment"


class BlendMode(str, Enum):
    """Compositing modes for merging a layer into a flattened preview."""

    NORMAL = "normal"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"
    ADD = "add"
    SUBTRACT = "subtract"


@dataclass(slots=True)
class Transform2D:
    """Affine transform applied to a layer before compositing.

    The matrix convention is column-vector based: a point is represented as
    `[x, y, 1]`, and the returned matrix is multiplied on the left. `x` and `y`
    are translations. `scale_x`, `scale_y`, and `rotation_degrees` are applied
    around `anchor_x`, `anchor_y`, which are interpreted in the same coordinate
    space as the points being transformed.
    """

    x: float = 0.0
    y: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    rotation_degrees: float = 0.0
    anchor_x: float = 0.5
    anchor_y: float = 0.5

    def as_matrix(self) -> np.ndarray:
        """Return this transform as a 3x3 homogeneous affine matrix."""
        _validate_transform_numbers(self)

        radians = np.deg2rad(float(self.rotation_degrees))
        cos_theta = float(np.cos(radians))
        sin_theta = float(np.sin(radians))

        translate = np.array(
            [
                [1.0, 0.0, float(self.x)],
                [0.0, 1.0, float(self.y)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        to_anchor = np.array(
            [
                [1.0, 0.0, float(self.anchor_x)],
                [0.0, 1.0, float(self.anchor_y)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        from_anchor = np.array(
            [
                [1.0, 0.0, -float(self.anchor_x)],
                [0.0, 1.0, -float(self.anchor_y)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        rotate = np.array(
            [
                [cos_theta, -sin_theta, 0.0],
                [sin_theta, cos_theta, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        scale = np.array(
            [
                [float(self.scale_x), 0.0, 0.0],
                [0.0, float(self.scale_y), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        return translate @ to_anchor @ rotate @ scale @ from_anchor


@dataclass(slots=True)
class LockFlags:
    """Layer lock settings checked by executor preconditions.

    These flags do not prevent a layer from existing, validating, flattening, or
    becoming active. They answer whether an action is allowed to mutate a class
    of layer data. `fully_locked` overrides the narrower lock flags.
    """

    pixels_locked: bool = False
    alpha_locked: bool = False
    position_locked: bool = False
    visibility_locked: bool = False
    fully_locked: bool = False

    def allows_pixel_edit(self) -> bool:
        """Return whether an action may modify this layer's RGB channels."""
        self._validate()
        return not self.fully_locked and not self.pixels_locked

    def allows_alpha_edit(self) -> bool:
        """Return whether an action may modify this layer's alpha channel."""
        self._validate()
        return not self.fully_locked and not self.alpha_locked

    def allows_transform_edit(self) -> bool:
        """Return whether an action may modify this layer's transform."""
        self._validate()
        return not self.fully_locked and not self.position_locked

    def _validate(self) -> None:
        """Validate that all lock flags are booleans."""
        for field_name in (
            "pixels_locked",
            "alpha_locked",
            "position_locked",
            "visibility_locked",
            "fully_locked",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a bool")


@dataclass(slots=True)
class Layer:
    """A single layer in a layered image document.

    `pixels` is either `None` or a full-canvas `height x width x 4` `float32`
    array. Values are straight-alpha RGBA channels in `[0, 1]`. Raster layers
    require pixel data. Group layers must not contain pixel data. Other layer
    kinds may store a rasterized representation when one is available.
    """

    id: str
    name: str
    kind: LayerKind = LayerKind.RASTER
    pixels: Optional[np.ndarray] = None
    opacity: float = 1.0
    visible: bool = True
    blend_mode: BlendMode = BlendMode.NORMAL
    transform: Transform2D = field(default_factory=Transform2D)
    mask_id: Optional[str] = None
    parent_group_id: Optional[str] = None
    child_layer_ids: list[str] = field(default_factory=list)
    locks: LockFlags = field(default_factory=LockFlags)
    metadata: dict[str, Any] = field(default_factory=dict)

    def width(self) -> int:
        """Return this layer's raster width.

        Layers without raster pixel data raise `ValueError`; callers that accept
        metadata-only layers should check `has_pixels()` first.
        """
        self._require_pixels()
        return int(self.pixels.shape[1])

    def height(self) -> int:
        """Return this layer's raster height.

        Layers without raster pixel data raise `ValueError`; callers that accept
        metadata-only layers should check `has_pixels()` first.
        """
        self._require_pixels()
        return int(self.pixels.shape[0])

    def has_pixels(self) -> bool:
        """Return whether this layer has a raster pixel array attached."""
        return isinstance(self.pixels, np.ndarray)

    def validate_pixel_shape(self, canvas_width: int, canvas_height: int) -> None:
        """Validate this layer against the canonical full-canvas pixel contract."""
        _validate_positive_int(canvas_width, "canvas_width")
        _validate_positive_int(canvas_height, "canvas_height")
        _validate_identifier(self.id, "layer.id")
        if not isinstance(self.name, str):
            raise TypeError("layer.name must be a string")
        _validate_enum_member(self.kind, LayerKind, "layer.kind")
        _validate_unit_float(self.opacity, "layer.opacity")
        if not isinstance(self.visible, bool):
            raise TypeError("layer.visible must be a bool")
        _validate_enum_member(self.blend_mode, BlendMode, "layer.blend_mode")
        _validate_transform_numbers(self.transform)
        self.locks._validate()

        if self.mask_id is not None:
            _validate_identifier(self.mask_id, "layer.mask_id")
        if self.parent_group_id is not None:
            _validate_identifier(self.parent_group_id, "layer.parent_group_id")
        if not isinstance(self.child_layer_ids, list):
            raise TypeError("layer.child_layer_ids must be a list")
        for child_id in self.child_layer_ids:
            _validate_identifier(child_id, "layer.child_layer_ids entry")
        if not isinstance(self.metadata, dict):
            raise TypeError("layer.metadata must be a dictionary")

        kind = LayerKind(self.kind)
        if kind is LayerKind.GROUP:
            if self.pixels is not None:
                raise ValueError("group layers must not contain pixel data")
            return
        if kind is LayerKind.RASTER and self.pixels is None:
            raise ValueError("raster layers must contain pixel data")
        if self.pixels is not None:
            _validate_pixel_array(self.pixels, canvas_width, canvas_height, "layer.pixels")

    def clone_shallow(self, new_id: str, new_name: Optional[str] = None) -> "Layer":
        """Return a layer copy that shares pixel storage with this layer.

        All mutable metadata containers are copied. The pixel array reference is
        intentionally shared, which is useful for callers implementing explicit
        copy-on-write behavior.
        """
        _validate_identifier(new_id, "new_id")
        if new_name is not None and not isinstance(new_name, str):
            raise TypeError("new_name must be a string or None")

        return Layer(
            id=new_id,
            name=self.name if new_name is None else new_name,
            kind=self.kind,
            pixels=self.pixels,
            opacity=self.opacity,
            visible=self.visible,
            blend_mode=self.blend_mode,
            transform=deepcopy(self.transform),
            mask_id=self.mask_id,
            parent_group_id=self.parent_group_id,
            child_layer_ids=list(self.child_layer_ids),
            locks=deepcopy(self.locks),
            metadata=deepcopy(self.metadata),
        )

    def clone_deep(self, new_id: str, new_name: Optional[str] = None) -> "Layer":
        """Return a full independent layer copy, including pixel data."""
        clone = self.clone_shallow(new_id, new_name)
        if self.pixels is not None:
            clone.pixels = np.array(self.pixels, copy=True)
        return clone

    def _require_pixels(self) -> None:
        """Raise a clear error when pixel data is unavailable."""
        if self.pixels is None:
            raise ValueError(f"layer {self.id!r} does not contain pixel data")
        if not isinstance(self.pixels, np.ndarray):
            raise TypeError(f"layer {self.id!r} pixels must be a NumPy array")


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


def _validate_pixel_array(array: Any, width: int, height: int, field_name: str) -> None:
    """Validate full-canvas straight-alpha RGBA pixel storage."""
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{field_name} must be a NumPy array")
    if array.shape != (height, width, 4):
        raise ValueError(f"{field_name} must have shape {(height, width, 4)!r}")
    if array.dtype != np.float32:
        raise TypeError(f"{field_name} must have dtype float32")
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{field_name} must contain only finite values")
    if bool(np.any((array < 0.0) | (array > 1.0))):
        raise ValueError(f"{field_name} values must be in [0, 1]")


def _validate_transform_numbers(transform: Any) -> None:
    """Validate numeric transform fields."""
    for field_name in ("x", "y", "scale_x", "scale_y", "rotation_degrees", "anchor_x", "anchor_y"):
        if not hasattr(transform, field_name):
            raise TypeError(f"transform must provide {field_name}")
        value = getattr(transform, field_name)
        if not _is_real_number(value):
            raise TypeError(f"transform.{field_name} must be a number")
        if not np.isfinite(value):
            raise ValueError(f"transform.{field_name} must be finite")
    if transform.scale_x == 0.0 or transform.scale_y == 0.0:
        raise ValueError("transform scale values must be nonzero")


def _is_real_number(value: Any) -> bool:
    """Return whether `value` is a real scalar number and not a bool."""
    return not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, float, np.integer, np.floating))
