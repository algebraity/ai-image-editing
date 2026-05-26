"""DocumentState model for the AI Editing Kernel.

DocumentState is the authoritative in-memory representation of a layered image
being edited. Planners inspect it, Executors mutate it through Actions, Validators
check it, and TraceLogger records snapshots of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

from ai_edit_kernel.document.layer import Layer
from ai_edit_kernel.document.mask import Mask


class ColorSpace(str, Enum):
    """Supported document color spaces.

    V0 can operate only in SRGB. The enum exists so the file format and API do
    not have to change when linear RGB, display P3, or CMYK are considered later.
    """

    SRGB = "srgb"
    LINEAR_RGB = "linear_rgb"
    DISPLAY_P3 = "display_p3"


@dataclass(slots=True)
class CanvasSpec:
    """Canvas-wide settings."""

    width: int
    height: int
    color_space: ColorSpace = ColorSpace.SRGB
    background_color_rgba: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    dpi: Optional[float] = None

    def validate(self) -> None:
        """Validate canvas dimensions and color configuration.

        Must reject non-positive width/height and invalid background channel
        values. Later this can also validate supported DPI and color spaces.
        """
        raise NotImplementedError


@dataclass(slots=True)
class DocumentMetadata:
    """Human-readable and machine-readable document metadata."""

    title: Optional[str] = None
    author: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    source_file: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    custom: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DocumentState:
    """Complete editable state of a document.

    `layers` should be ordered from bottom to top unless the project standardizes
    otherwise. All layer/mask access should go through helper methods so the
    executor and validator get consistent error handling.
    """

    id: str
    canvas: CanvasSpec
    layers: list[Layer] = field(default_factory=list)
    masks: dict[str, Mask] = field(default_factory=dict)
    active_layer_id: Optional[str] = None
    active_selection_mask_id: Optional[str] = None
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    revision: int = 0
    annotations: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate the complete document structure.

        Must check canvas validity, unique layer IDs, active layer existence,
        layer pixel shapes, mask dimensions, layer mask references, group
        references, and other document invariants.
        """
        raise NotImplementedError

    def get_layer(self, layer_id: str) -> Layer:
        """Return the layer with the given ID or raise a clear lookup error."""
        raise NotImplementedError

    def get_layer_by_name(self, name: str) -> Layer:
        """Return the first layer with the given name or raise a clear lookup error.

        Names are human-friendly and may not be globally unique in later versions;
        prefer IDs for executor internals.
        """
        raise NotImplementedError

    def add_layer(self, layer: Layer, index: Optional[int] = None) -> None:
        """Insert a layer into the layer stack.

        Must validate ID uniqueness, pixel shape, group references, and optional
        insertion index. Should update active_layer_id if requested by the action
        calling this method.
        """
        raise NotImplementedError

    def remove_layer(self, layer_id: str) -> Layer:
        """Remove and return a layer.

        Must protect against removing locked layers unless the caller has already
        validated an override. Must update active_layer_id if needed.
        """
        raise NotImplementedError

    def reorder_layer(self, layer_id: str, new_index: int) -> None:
        """Move a layer to a new stack index."""
        raise NotImplementedError

    def add_mask(self, mask: Mask) -> None:
        """Register a mask in the document.

        Must validate dimensions, ID uniqueness, and value range.
        """
        raise NotImplementedError

    def get_mask(self, mask_id: str) -> Mask:
        """Return a mask by ID or raise a clear lookup error."""
        raise NotImplementedError

    def remove_mask(self, mask_id: str) -> Mask:
        """Remove and return a mask.

        Must ensure no layer or active selection still references the mask unless
        the calling action explicitly handles reference cleanup.
        """
        raise NotImplementedError

    def set_active_layer(self, layer_id: str) -> None:
        """Set the active layer after confirming the layer exists and is editable."""
        raise NotImplementedError

    def set_active_selection(self, mask_id: Optional[str]) -> None:
        """Set or clear the active selection mask."""
        raise NotImplementedError

    def next_revision(self) -> int:
        """Increment and return the document revision number.

        Executors should call this after each successful document mutation.
        """
        raise NotImplementedError

    def flatten_preview(self, include_hidden: bool = False) -> np.ndarray:
        """Composite visible layers into a single RGBA preview image.

        Must respect layer order, opacity, blend mode, transforms, layer masks,
        and canvas background. V0 can implement normal blend mode first.
        """
        raise NotImplementedError

    def clone_deep(self, new_id: Optional[str] = None) -> "DocumentState":
        """Return a full independent copy of the document.

        Used for rollback, before/after validation, and trace snapshots.
        """
        raise NotImplementedError

    def snapshot_summary(self) -> dict[str, Any]:
        """Return a lightweight JSON-serializable summary of the document.

        This is what the LLM planner should usually see instead of raw pixels:
        canvas size, layer stack, mask names/stats, active layer, and metadata.
        """
        raise NotImplementedError
