"""Layer model for the AI Editing Kernel.

A Layer is an editable plane in the document. In v0, raster layers are the most
important. Vector/text/adjustment/group layers are represented in the schema so
that the architecture can grow without changing the public document model.

Function bodies are intentionally omitted. This file defines the data contract
and the methods that later implementations must provide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


class LayerKind(str, Enum):
    """Supported layer categories.

    V0 should fully support RASTER and basic GROUP semantics. VECTOR, TEXT, and
    ADJUSTMENT can initially be metadata-only or rasterized at execution time.
    """

    RASTER = "raster"
    VECTOR = "vector"
    TEXT = "text"
    GROUP = "group"
    ADJUSTMENT = "adjustment"


class BlendMode(str, Enum):
    """Compositing modes for merging a layer into the flattened preview.

    V0 should implement NORMAL first. MULTIPLY, SCREEN, OVERLAY, ADD, and
    SUBTRACT can be added once layer compositing is stable.
    """

    NORMAL = "normal"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"
    ADD = "add"
    SUBTRACT = "subtract"


@dataclass(slots=True)
class Transform2D:
    """Affine transform applied to a layer before compositing.

    The executor should eventually use this for move, scale, rotate, and align
    operations. V0 can restrict transforms to translation and scale.
    """

    x: float = 0.0
    y: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    rotation_degrees: float = 0.0
    anchor_x: float = 0.5
    anchor_y: float = 0.5

    def as_matrix(self) -> np.ndarray:
        """Return this transform as a 3x3 homogeneous affine matrix.

        Must combine translation, scale, rotation, and anchor handling into one
        matrix that downstream rasterization/compositing code can use.
        """
        raise NotImplementedError


@dataclass(slots=True)
class LockFlags:
    """Layer lock settings.

    These flags protect layers from accidental destructive actions. The executor
    must check them before changing pixels, alpha, transform, or metadata.
    """

    pixels_locked: bool = False
    alpha_locked: bool = False
    position_locked: bool = False
    visibility_locked: bool = False
    fully_locked: bool = False

    def allows_pixel_edit(self) -> bool:
        """Return whether an action may modify this layer's RGB pixel data."""
        raise NotImplementedError

    def allows_alpha_edit(self) -> bool:
        """Return whether an action may modify this layer's alpha channel."""
        raise NotImplementedError

    def allows_transform_edit(self) -> bool:
        """Return whether an action may modify this layer's transform."""
        raise NotImplementedError


@dataclass(slots=True)
class Layer:
    """A single layer in a layered image document.

    For raster layers, `pixels` should be an H x W x 4 float32 or uint8 RGBA
    array. The project should standardize on one internal representation early;
    float32 in 0..1 is usually easier for compositing, while uint8 is easier for
    file IO.
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
        """Return the raster width of this layer.

        Must raise a clear error if called on a non-raster layer without
        rasterized pixel data.
        """
        raise NotImplementedError

    def height(self) -> int:
        """Return the raster height of this layer.

        Must raise a clear error if called on a non-raster layer without
        rasterized pixel data.
        """
        raise NotImplementedError

    def has_pixels(self) -> bool:
        """Return True when the layer has editable raster pixel data."""
        raise NotImplementedError

    def validate_pixel_shape(self, canvas_width: int, canvas_height: int) -> None:
        """Check that this layer's pixel array is compatible with the canvas.

        V0 should probably require full-canvas H x W x 4 pixel arrays. Later, the
        runtime can support sparse/tiled layers with offsets.
        """
        raise NotImplementedError

    def clone_shallow(self, new_id: str, new_name: Optional[str] = None) -> "Layer":
        """Return a new Layer object sharing no mutable metadata with this layer.

        This should copy metadata and settings but may share pixel storage only if
        the caller explicitly wants copy-on-write semantics.
        """
        raise NotImplementedError

    def clone_deep(self, new_id: str, new_name: Optional[str] = None) -> "Layer":
        """Return a full independent copy of this layer, including pixel data."""
        raise NotImplementedError
