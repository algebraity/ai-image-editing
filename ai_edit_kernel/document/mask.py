"""Mask and selection model for the AI Editing Kernel.

Masks are the primary safety mechanism of the kernel. Every pixel-changing
operation should be clipped through a Mask, and the validator should confirm that
protected pixels did not change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


class MaskKind(str, Enum):
    """Semantic role of a mask.

    SELECTION masks are user/editor selections. WRITE_GUARD masks define the
    exact pixels an action may modify. LAYER_ALPHA masks may be attached to a
    layer. OBJECT masks are inferred object regions from perception tools.
    """

    SELECTION = "selection"
    WRITE_GUARD = "write_guard"
    LAYER_ALPHA = "layer_alpha"
    OBJECT = "object"
    SHAPE = "shape"
    LINE_ART_REGION = "line_art_region"
    DIFFUSION = "diffusion"


@dataclass(slots=True)
class MaskStats:
    """Cached geometric and numerical properties of a mask."""

    area_pixels: int
    bbox: tuple[int, int, int, int]  # x, y, width, height
    centroid: tuple[float, float]
    min_value: float
    max_value: float
    is_binary: bool


@dataclass(slots=True)
class Mask:
    """A 2D floating point mask with values in [0.0, 1.0].

    A value of 0.0 means fully protected/excluded. A value of 1.0 means fully
    included/editable. Values between 0 and 1 are used for antialiasing and
    feathered transitions.
    """

    id: str
    name: str
    data: np.ndarray
    kind: MaskKind = MaskKind.SELECTION
    hard: bool = True
    source: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, canvas_width: int, canvas_height: int) -> None:
        """Validate shape, dtype, range, and canvas compatibility.

        Must ensure data is H x W, values are finite, and all values are within
        [0.0, 1.0]. Hard masks should contain only 0/1 values after thresholding.
        """
        raise NotImplementedError

    def stats(self) -> MaskStats:
        """Compute area, bounding box, centroid, min/max, and binary status."""
        raise NotImplementedError

    def clone(self, new_id: str, new_name: Optional[str] = None) -> "Mask":
        """Return an independent copy of this mask and its metadata."""
        raise NotImplementedError

    def threshold(self, threshold: float = 0.5, new_id: Optional[str] = None) -> "Mask":
        """Return a hard binary mask from this mask using the given threshold."""
        raise NotImplementedError

    def invert(self, new_id: str, name: Optional[str] = None) -> "Mask":
        """Return a mask where included and excluded areas are swapped."""
        raise NotImplementedError

    def union(self, other: "Mask", new_id: str, name: Optional[str] = None) -> "Mask":
        """Return a mask containing pixels included by either mask."""
        raise NotImplementedError

    def intersect(self, other: "Mask", new_id: str, name: Optional[str] = None) -> "Mask":
        """Return a mask containing only pixels included by both masks."""
        raise NotImplementedError

    def subtract(self, other: "Mask", new_id: str, name: Optional[str] = None) -> "Mask":
        """Return this mask with the other mask's included pixels removed."""
        raise NotImplementedError

    def dilate(self, pixels: int, new_id: str, name: Optional[str] = None) -> "Mask":
        """Grow the included region by the requested number of pixels.

        Used for operations like growing a selection or creating overlap for edge
        cleanup. Must preserve canvas dimensions.
        """
        raise NotImplementedError

    def erode(self, pixels: int, new_id: str, name: Optional[str] = None) -> "Mask":
        """Shrink the included region by the requested number of pixels.

        Used for creating safe interior masks, e.g. inside a circle border or
        inside a crystal ball rim.
        """
        raise NotImplementedError

    def feather(self, radius: float, new_id: str, name: Optional[str] = None) -> "Mask":
        """Return a softened mask with a feathered boundary.

        The validator should still distinguish between fully protected pixels and
        feather-transition pixels.
        """
        raise NotImplementedError

    def as_write_guard(self, new_id: str, name: Optional[str] = None) -> "Mask":
        """Return a copy of this mask marked as a WRITE_GUARD mask."""
        raise NotImplementedError

    @classmethod
    def full_canvas(cls, id: str, width: int, height: int, name: str = "full canvas") -> "Mask":
        """Create a mask that allows editing every canvas pixel."""
        raise NotImplementedError

    @classmethod
    def empty(cls, id: str, width: int, height: int, name: str = "empty") -> "Mask":
        """Create a mask that protects every canvas pixel."""
        raise NotImplementedError
