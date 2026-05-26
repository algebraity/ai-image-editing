"""Authoritative document state for the AI Editing Kernel.

`DocumentState` is the kernel's in-memory source of truth for a layered image.
The planner may inspect summaries of it, the executor mutates it through action
handlers, validators check its invariants, and trace logging records snapshots
of it for replay and training data.

The document contract is intentionally strict:

* Raster pixel data is stored as full-canvas `height x width x 4` NumPy arrays.
* Pixel arrays use straight-alpha RGBA channels with `float32` values in `[0, 1]`.
* Masks are full-canvas `height x width` `float32` arrays with values in `[0, 1]`.
* `layers` are ordered from bottom to top, so later layers composite over earlier
  layers.
* Layer IDs are unique within the layer stack, and mask IDs are unique within the
  mask registry.
* `DocumentState` helper methods perform structural mutations only. Higher-level
  edit policy, including lock enforcement and revision increments, belongs to the
  executor.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

from ai_edit_kernel.document.layer import BlendMode, Layer, LayerKind
from ai_edit_kernel.document.mask import Mask, MaskKind


class ColorSpace(str, Enum):
    """Supported canvas color spaces.

    The document stores numeric channel values independently of file format. The
    color space records how those channels should be interpreted by import,
    export, preview, and backend adapters.
    """

    SRGB = "srgb"
    LINEAR_RGB = "linear_rgb"
    DISPLAY_P3 = "display_p3"


@dataclass(slots=True)
class CanvasSpec:
    """Canvas-wide geometry and color defaults.

    `width` and `height` are measured in pixels. `background_color_rgba` is the
    transparent or opaque base color used by `DocumentState.flatten_preview()`
    before layers are composited. Channel values are straight-alpha RGBA floats
    in `[0, 1]`.
    """

    width: int
    height: int
    color_space: ColorSpace = ColorSpace.SRGB
    background_color_rgba: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    dpi: Optional[float] = None

    def validate(self) -> None:
        """Validate canvas dimensions, color space, background color, and DPI."""
        _validate_positive_int(self.width, "canvas.width")
        _validate_positive_int(self.height, "canvas.height")
        _validate_enum_member(self.color_space, ColorSpace, "canvas.color_space")
        _validate_rgba(self.background_color_rgba, "canvas.background_color_rgba")

        if self.dpi is not None:
            if not _is_real_number(self.dpi):
                raise TypeError("canvas.dpi must be a positive number or None")
            if not np.isfinite(self.dpi) or self.dpi <= 0:
                raise ValueError("canvas.dpi must be finite and greater than zero")


@dataclass(slots=True)
class DocumentMetadata:
    """Human-readable and machine-readable document metadata.

    Metadata does not control execution. It is copied with the document, included
    in snapshot summaries, and may be used by UI or training-data pipelines.
    """

    title: Optional[str] = None
    author: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    source_file: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    custom: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DocumentState:
    """Complete editable state of a layered image document.

    The state is deliberately small and explicit. Layers hold pixel data and
    per-layer metadata; masks hold reusable spatial constraints; active IDs point
    to the editor's current layer and selection. An empty layer stack is valid,
    which allows callers to create a document before importing or generating
    content.

    Mutating helpers such as `add_layer()` and `remove_mask()` preserve structural
    invariants but do not increment `revision`. The executor owns revision changes
    so a whole action can validate, mutate, trace, and then advance the revision
    exactly once.
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
        """Validate the full document structure.

        This method checks the in-memory invariants that all executors and
        validators rely on: canvas validity, layer ID uniqueness, full-canvas
        pixel arrays, mask registry consistency, active ID references, layer mask
        references, and basic group metadata references.
        """
        _validate_identifier(self.id, "document.id")

        if not isinstance(self.canvas, CanvasSpec):
            raise TypeError("document.canvas must be a CanvasSpec")
        self.canvas.validate()

        if not isinstance(self.layers, list):
            raise TypeError("document.layers must be a list")
        if not isinstance(self.masks, dict):
            raise TypeError("document.masks must be a dictionary keyed by mask ID")
        if not isinstance(self.metadata, DocumentMetadata):
            raise TypeError("document.metadata must be a DocumentMetadata")
        if not isinstance(self.annotations, dict):
            raise TypeError("document.annotations must be a dictionary")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise TypeError("document.revision must be an integer")
        if self.revision < 0:
            raise ValueError("document.revision must not be negative")

        self._validate_metadata()

        mask_ids: set[str] = set()
        for key, mask in self.masks.items():
            if not isinstance(key, str):
                raise TypeError("document.masks keys must be strings")
            if not isinstance(mask, Mask):
                raise TypeError(f"document.masks[{key!r}] must be a Mask")
            self._validate_mask(mask)
            if key != mask.id:
                raise ValueError(f"mask registry key {key!r} does not match mask id {mask.id!r}")
            if mask.id in mask_ids:
                raise ValueError(f"duplicate mask id {mask.id!r}")
            mask_ids.add(mask.id)

        layer_ids: set[str] = set()
        layer_by_id: dict[str, Layer] = {}
        for index, layer in enumerate(self.layers):
            if not isinstance(layer, Layer):
                raise TypeError(f"document.layers[{index}] must be a Layer")
            self._validate_layer(layer, index)
            if layer.id in layer_ids:
                raise ValueError(f"duplicate layer id {layer.id!r}")
            layer_ids.add(layer.id)
            layer_by_id[layer.id] = layer

        if self.active_layer_id is not None:
            _validate_identifier(self.active_layer_id, "document.active_layer_id")
            if self.active_layer_id not in layer_by_id:
                raise ValueError(f"active layer id {self.active_layer_id!r} does not exist")
        if self.active_selection_mask_id is not None:
            _validate_identifier(self.active_selection_mask_id, "document.active_selection_mask_id")
            if self.active_selection_mask_id not in self.masks:
                raise ValueError(f"active selection mask id {self.active_selection_mask_id!r} does not exist")

        for layer in self.layers:
            if layer.mask_id is not None and layer.mask_id not in self.masks:
                raise ValueError(f"layer {layer.id!r} references missing mask {layer.mask_id!r}")

        self._validate_group_references(layer_by_id)

    def get_layer(self, layer_id: str) -> Layer:
        """Return the layer with `layer_id`.

        Layer IDs are the stable way for executor internals to address layers.
        Missing IDs raise `KeyError`.
        """
        _validate_identifier(layer_id, "layer_id")
        for layer in self.layers:
            if layer.id == layer_id:
                return layer
        raise KeyError(f"layer id {layer_id!r} does not exist")

    def get_layer_by_name(self, name: str) -> Layer:
        """Return the first layer whose human-readable name matches `name`.

        Names are allowed to repeat, matching common image-editor behavior. This
        method searches the stack from bottom to top and returns the first match.
        Use IDs whenever uniqueness matters.
        """
        if not isinstance(name, str):
            raise TypeError("layer name must be a string")
        for layer in self.layers:
            if layer.name == name:
                return layer
        raise KeyError(f"layer named {name!r} does not exist")

    def add_layer(self, layer: Layer, index: Optional[int] = None) -> None:
        """Insert `layer` into the bottom-to-top layer stack.

        `index=None` appends the layer at the top of the stack. Explicit indexes
        must be in `0..len(layers)`; negative indexes are rejected so callers do
        not accidentally depend on Python list insertion quirks.

        The caller provides the layer ID. This method validates structural
        compatibility and ID uniqueness, but it does not change the active layer
        or increment the document revision.
        """
        if not isinstance(layer, Layer):
            raise TypeError("layer must be a Layer")
        self.canvas.validate()
        if any(existing.id == layer.id for existing in self.layers):
            raise ValueError(f"duplicate layer id {layer.id!r}")

        insert_index = self._normalize_insert_index(index)
        self._validate_layer(layer, insert_index)
        self._validate_new_layer_references(layer)
        self.layers.insert(insert_index, layer)

    def remove_layer(self, layer_id: str) -> Layer:
        """Remove and return a layer by ID.

        Removal fails if another layer's group metadata references the target.
        The caller should explicitly clean up group metadata before removing a
        referenced layer. If the removed layer was active, the active layer moves
        to the topmost remaining layer, or `None` when the stack becomes empty.
        """
        _validate_identifier(layer_id, "layer_id")
        for index, layer in enumerate(self.layers):
            if layer.id == layer_id:
                self._assert_layer_not_referenced(layer_id)
                removed = self.layers.pop(index)
                if self.active_layer_id == layer_id:
                    self.active_layer_id = self.layers[-1].id if self.layers else None
                return removed

        raise KeyError(f"layer id {layer_id!r} does not exist")

    def reorder_layer(self, layer_id: str, new_index: int) -> None:
        """Move a layer to `new_index` in the bottom-to-top layer stack."""
        _validate_identifier(layer_id, "layer_id")
        if isinstance(new_index, bool) or not isinstance(new_index, int):
            raise TypeError("new_index must be an integer")
        if new_index < 0 or new_index >= len(self.layers):
            raise IndexError("new_index must be in 0..len(layers)-1")

        for old_index, layer in enumerate(self.layers):
            if layer.id == layer_id:
                if old_index == new_index:
                    return
                moved = self.layers.pop(old_index)
                self.layers.insert(new_index, moved)
                return

        raise KeyError(f"layer id {layer_id!r} does not exist")

    def add_mask(self, mask: Mask) -> None:
        """Register `mask` in the document mask registry.

        The caller provides the mask ID. This method validates full-canvas shape,
        range, dtype, and ID uniqueness. It does not make the mask active.
        """
        if not isinstance(mask, Mask):
            raise TypeError("mask must be a Mask")
        self.canvas.validate()
        self._validate_mask(mask)
        if mask.id in self.masks:
            raise ValueError(f"duplicate mask id {mask.id!r}")
        self.masks[mask.id] = mask

    def get_mask(self, mask_id: str) -> Mask:
        """Return the mask with `mask_id`, or raise `KeyError` if it is missing."""
        _validate_identifier(mask_id, "mask_id")
        try:
            return self.masks[mask_id]
        except KeyError as exc:
            raise KeyError(f"mask id {mask_id!r} does not exist") from exc

    def remove_mask(self, mask_id: str) -> Mask:
        """Remove and return an unreferenced mask.

        Removal fails with `ValueError` when the mask is the active selection or
        when any layer references it as a layer mask. The executor should perform
        explicit cleanup before calling this method.
        """
        _validate_identifier(mask_id, "mask_id")
        if mask_id not in self.masks:
            raise KeyError(f"mask id {mask_id!r} does not exist")
        if self.active_selection_mask_id == mask_id:
            raise ValueError(f"cannot remove active selection mask {mask_id!r}")

        referencing_layers = [layer.id for layer in self.layers if layer.mask_id == mask_id]
        if referencing_layers:
            raise ValueError(
                f"cannot remove mask {mask_id!r}; it is referenced by layers {referencing_layers!r}"
            )

        return self.masks.pop(mask_id)

    def set_active_layer(self, layer_id: str) -> None:
        """Set the active layer after confirming that the ID exists.

        Lock flags do not prevent activation. They are edit-policy checks enforced
        by the executor before actions modify pixels, alpha, transforms, or layer
        metadata.
        """
        self.get_layer(layer_id)
        self.active_layer_id = layer_id

    def set_active_selection(self, mask_id: Optional[str]) -> None:
        """Set or clear the active selection.

        The active selection is stored as a normal mask reference so it can be
        inspected, cloned, traced, and reused like any other mask. `None` clears
        the active selection.
        """
        if mask_id is None:
            self.active_selection_mask_id = None
            return

        self.get_mask(mask_id)
        self.active_selection_mask_id = mask_id

    def next_revision(self) -> int:
        """Increment and return the document revision number.

        Executors call this once after a successful action-level mutation. Lower
        level document helpers leave `revision` unchanged.
        """
        self.revision += 1
        return self.revision

    def flatten_preview(self, include_hidden: bool = False) -> np.ndarray:
        """Composite the document into a full-canvas straight-alpha RGBA preview.

        Preview compositing is intentionally conservative. It supports visible
        full-canvas pixel layers using normal source-over blending, layer opacity,
        and optional layer masks. Group layers are organizational metadata and are
        skipped. Hidden layers are skipped unless `include_hidden=True`.

        Non-normal blend modes and non-identity transforms raise
        `NotImplementedError` so callers do not receive a preview that silently
        omits important rendering semantics.
        """
        self.canvas.validate()
        preview = _rgba_array_from_color(
            self.canvas.background_color_rgba,
            self.canvas.width,
            self.canvas.height,
        )

        for index, layer in enumerate(self.layers):
            if not include_hidden and not layer.visible:
                continue
            if _enum_value(layer.kind) == LayerKind.GROUP.value:
                continue

            self._validate_layer(layer, index)

            if layer.pixels is None:
                raise NotImplementedError(f"layer {layer.id!r} has no raster pixels to preview")
            if _enum_value(layer.blend_mode) != BlendMode.NORMAL.value:
                raise NotImplementedError(f"blend mode {layer.blend_mode!r} is not supported by flatten_preview")
            if not _is_identity_transform(layer.transform):
                raise NotImplementedError(f"layer {layer.id!r} uses a transform that flatten_preview cannot render")

            layer_pixels = layer.pixels.astype(np.float32, copy=False)
            effective_alpha = layer_pixels[..., 3:4] * np.float32(layer.opacity)

            if layer.mask_id is not None:
                mask = self.get_mask(layer.mask_id)
                self._validate_mask(mask)
                effective_alpha = effective_alpha * mask.data[..., np.newaxis]

            preview = _source_over(preview, layer_pixels[..., :3], effective_alpha)

        return preview

    def clone_deep(self, new_id: Optional[str] = None) -> "DocumentState":
        """Return a full independent copy of the document.

        The clone owns independent layer arrays, mask arrays, metadata, and
        annotations. IDs are preserved unless `new_id` is supplied for the document
        itself. This is the snapshot primitive used by rollback, validation, and
        trace capture.
        """
        if new_id is not None:
            _validate_identifier(new_id, "new_id")

        return DocumentState(
            id=self.id if new_id is None else new_id,
            canvas=deepcopy(self.canvas),
            layers=[_clone_layer(layer) for layer in self.layers],
            masks={mask_id: _clone_mask(mask) for mask_id, mask in self.masks.items()},
            active_layer_id=self.active_layer_id,
            active_selection_mask_id=self.active_selection_mask_id,
            metadata=deepcopy(self.metadata),
            revision=self.revision,
            annotations=deepcopy(self.annotations),
        )

    def snapshot_summary(self) -> dict[str, Any]:
        """Return a lightweight JSON-compatible summary of the document.

        The summary is designed for planners, trace files, and diagnostics. It
        includes IDs, canvas settings, layer/mask metadata, mask statistics, active
        references, revision, and user metadata. It never includes raw pixel arrays
        or raw mask arrays.
        """
        return {
            "id": self.id,
            "revision": self.revision,
            "canvas": {
                "width": self.canvas.width,
                "height": self.canvas.height,
                "color_space": _enum_value(self.canvas.color_space),
                "background_color_rgba": _json_safe(self.canvas.background_color_rgba),
                "dpi": _json_safe(self.canvas.dpi),
            },
            "active_layer_id": self.active_layer_id,
            "active_selection_mask_id": self.active_selection_mask_id,
            "layer_count": len(self.layers),
            "mask_count": len(self.masks),
            "layers": [self._layer_summary(layer, index) for index, layer in enumerate(self.layers)],
            "masks": [self._mask_summary(mask) for mask in self.masks.values()],
            "metadata": _json_safe(self.metadata),
            "annotations": _json_safe(self.annotations),
        }

    def _validate_metadata(self) -> None:
        """Validate the structured document metadata container."""
        for field_name in ("title", "author", "created_at", "updated_at", "source_file"):
            value = getattr(self.metadata, field_name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"document.metadata.{field_name} must be a string or None")
        if not isinstance(self.metadata.tags, list):
            raise TypeError("document.metadata.tags must be a list")
        if not all(isinstance(tag, str) for tag in self.metadata.tags):
            raise TypeError("document.metadata.tags must contain only strings")
        if not isinstance(self.metadata.custom, dict):
            raise TypeError("document.metadata.custom must be a dictionary")

    def _normalize_insert_index(self, index: Optional[int]) -> int:
        """Return the concrete insertion index for `add_layer()`."""
        if index is None:
            return len(self.layers)
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("layer insertion index must be an integer or None")
        if index < 0 or index > len(self.layers):
            raise IndexError("layer insertion index must be in 0..len(layers)")
        return index

    def _validate_layer(self, layer: Layer, index: int) -> None:
        """Validate a layer's local fields and full-canvas pixel data."""
        _validate_identifier(layer.id, f"layers[{index}].id")
        if not isinstance(layer.name, str):
            raise TypeError(f"layer {layer.id!r} name must be a string")
        _validate_enum_member(layer.kind, LayerKind, f"layer {layer.id!r} kind")
        _validate_unit_float(layer.opacity, f"layer {layer.id!r} opacity")
        if not isinstance(layer.visible, bool):
            raise TypeError(f"layer {layer.id!r} visible must be a bool")
        _validate_enum_member(layer.blend_mode, BlendMode, f"layer {layer.id!r} blend_mode")
        _validate_transform(layer.transform, f"layer {layer.id!r} transform")
        _validate_lock_flags(layer.locks, f"layer {layer.id!r} locks")

        if layer.mask_id is not None:
            _validate_identifier(layer.mask_id, f"layer {layer.id!r} mask_id")
        if layer.parent_group_id is not None:
            _validate_identifier(layer.parent_group_id, f"layer {layer.id!r} parent_group_id")
        if not isinstance(layer.child_layer_ids, list):
            raise TypeError(f"layer {layer.id!r} child_layer_ids must be a list")
        for child_id in layer.child_layer_ids:
            _validate_identifier(child_id, f"layer {layer.id!r} child_layer_ids entry")
        if layer.child_layer_ids and _enum_value(layer.kind) != LayerKind.GROUP.value:
            raise ValueError(f"non-group layer {layer.id!r} cannot have child layers")
        if not isinstance(layer.metadata, dict):
            raise TypeError(f"layer {layer.id!r} metadata must be a dictionary")

        if _enum_value(layer.kind) == LayerKind.GROUP.value and layer.pixels is not None:
            raise ValueError(f"group layer {layer.id!r} must not have pixel data")
        if _enum_value(layer.kind) == LayerKind.RASTER.value and layer.pixels is None:
            raise ValueError(f"raster layer {layer.id!r} must have pixel data")
        if layer.pixels is not None:
            _validate_pixel_array(layer.pixels, self.canvas.width, self.canvas.height, f"layer {layer.id!r} pixels")

    def _validate_new_layer_references(self, layer: Layer) -> None:
        """Validate references from a layer before it is inserted."""
        existing_ids = {existing.id for existing in self.layers}
        existing_by_id = {existing.id: existing for existing in self.layers}

        if layer.mask_id is not None and layer.mask_id not in self.masks:
            raise ValueError(f"layer {layer.id!r} references missing mask {layer.mask_id!r}")

        if layer.parent_group_id is not None:
            parent = existing_by_id.get(layer.parent_group_id)
            if parent is None:
                raise ValueError(f"layer {layer.id!r} references missing parent group {layer.parent_group_id!r}")
            if _enum_value(parent.kind) != LayerKind.GROUP.value:
                raise ValueError(f"layer {layer.id!r} parent {parent.id!r} is not a group layer")

        for child_id in layer.child_layer_ids:
            if child_id not in existing_ids:
                raise ValueError(f"group layer {layer.id!r} references missing child layer {child_id!r}")
            if child_id == layer.id:
                raise ValueError(f"group layer {layer.id!r} cannot be its own child")

    def _validate_mask(self, mask: Mask) -> None:
        """Validate a mask's fields and full-canvas data array."""
        _validate_identifier(mask.id, "mask.id")
        if not isinstance(mask.name, str):
            raise TypeError(f"mask {mask.id!r} name must be a string")
        _validate_enum_member(mask.kind, MaskKind, f"mask {mask.id!r} kind")
        if not isinstance(mask.hard, bool):
            raise TypeError(f"mask {mask.id!r} hard must be a bool")
        if mask.source is not None and not isinstance(mask.source, str):
            raise TypeError(f"mask {mask.id!r} source must be a string or None")
        if not isinstance(mask.metadata, dict):
            raise TypeError(f"mask {mask.id!r} metadata must be a dictionary")

        _validate_mask_array(mask.data, self.canvas.width, self.canvas.height, f"mask {mask.id!r} data")
        if mask.hard:
            binary = np.isclose(mask.data, 0.0) | np.isclose(mask.data, 1.0)
            if not bool(np.all(binary)):
                raise ValueError(f"hard mask {mask.id!r} must contain only 0.0 and 1.0 values")

    def _validate_group_references(self, layer_by_id: dict[str, Layer]) -> None:
        """Check that group metadata references existing group/layer IDs."""
        for layer in self.layers:
            if layer.parent_group_id is not None:
                parent = layer_by_id.get(layer.parent_group_id)
                if parent is None:
                    raise ValueError(f"layer {layer.id!r} references missing parent group {layer.parent_group_id!r}")
                if _enum_value(parent.kind) != LayerKind.GROUP.value:
                    raise ValueError(f"layer {layer.id!r} parent {parent.id!r} is not a group layer")

            for child_id in layer.child_layer_ids:
                if child_id not in layer_by_id:
                    raise ValueError(f"group layer {layer.id!r} references missing child layer {child_id!r}")
                if child_id == layer.id:
                    raise ValueError(f"group layer {layer.id!r} cannot be its own child")

        for layer in self.layers:
            seen: set[str] = set()
            current = layer
            while current.parent_group_id is not None:
                if current.parent_group_id in seen:
                    raise ValueError(f"group cycle detected from layer {layer.id!r}")
                seen.add(current.parent_group_id)
                current = layer_by_id[current.parent_group_id]

    def _assert_layer_not_referenced(self, layer_id: str) -> None:
        """Reject layer removal while group metadata still points at the layer."""
        for layer in self.layers:
            if layer.id == layer_id:
                continue
            if layer.parent_group_id == layer_id:
                raise ValueError(f"cannot remove layer {layer_id!r}; layer {layer.id!r} has it as parent group")
            if layer_id in layer.child_layer_ids:
                raise ValueError(f"cannot remove layer {layer_id!r}; group layer {layer.id!r} references it")

    def _layer_summary(self, layer: Layer, index: int) -> dict[str, Any]:
        """Return a JSON-compatible summary for one layer."""
        width = int(layer.pixels.shape[1]) if layer.pixels is not None else None
        height = int(layer.pixels.shape[0]) if layer.pixels is not None else None
        return {
            "id": layer.id,
            "name": layer.name,
            "index": index,
            "kind": _enum_value(layer.kind),
            "width": width,
            "height": height,
            "has_pixels": layer.pixels is not None,
            "opacity": float(layer.opacity),
            "visible": layer.visible,
            "blend_mode": _enum_value(layer.blend_mode),
            "transform": _json_safe(layer.transform),
            "mask_id": layer.mask_id,
            "parent_group_id": layer.parent_group_id,
            "child_layer_ids": list(layer.child_layer_ids),
            "locks": _json_safe(layer.locks),
            "metadata": _json_safe(layer.metadata),
        }

    def _mask_summary(self, mask: Mask) -> dict[str, Any]:
        """Return a JSON-compatible summary for one mask."""
        stats = _mask_stats(mask.data)
        return {
            "id": mask.id,
            "name": mask.name,
            "kind": _enum_value(mask.kind),
            "width": int(mask.data.shape[1]),
            "height": int(mask.data.shape[0]),
            "hard": mask.hard,
            "source": mask.source,
            "stats": stats,
            "metadata": _json_safe(mask.metadata),
        }


def _validate_identifier(value: Any, field_name: str) -> None:
    """Validate a caller-provided document, layer, mask, or reference ID."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if value == "":
        raise ValueError(f"{field_name} must not be empty")


def _validate_positive_int(value: Any, field_name: str) -> None:
    """Validate a positive integer field while rejecting bools."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero")


def _validate_unit_float(value: Any, field_name: str) -> None:
    """Validate a finite numeric value in `[0, 1]`."""
    if not _is_real_number(value):
        raise TypeError(f"{field_name} must be a number")
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{field_name} must be finite and in [0, 1]")


def _validate_rgba(value: Any, field_name: str) -> None:
    """Validate a four-channel RGBA color with values in `[0, 1]`."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (4,):
        raise ValueError(f"{field_name} must contain exactly four RGBA channels")
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{field_name} must contain only finite values")
    if bool(np.any((array < 0.0) | (array > 1.0))):
        raise ValueError(f"{field_name} channels must be in [0, 1]")


def _validate_enum_member(value: Any, enum_type: type[Enum], field_name: str) -> None:
    """Validate an enum value while accepting the enum's raw string value."""
    try:
        enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = [member.value for member in enum_type]
        raise ValueError(f"{field_name} must be one of {allowed!r}") from exc


def _validate_pixel_array(array: Any, width: int, height: int, field_name: str) -> None:
    """Validate canonical full-canvas straight-alpha RGBA pixel storage."""
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


def _validate_mask_array(array: Any, width: int, height: int, field_name: str) -> None:
    """Validate canonical full-canvas mask storage."""
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{field_name} must be a NumPy array")
    if array.shape != (height, width):
        raise ValueError(f"{field_name} must have shape {(height, width)!r}")
    if array.dtype != np.float32:
        raise TypeError(f"{field_name} must have dtype float32")
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{field_name} must contain only finite values")
    if bool(np.any((array < 0.0) | (array > 1.0))):
        raise ValueError(f"{field_name} values must be in [0, 1]")


def _validate_transform(transform: Any, field_name: str) -> None:
    """Validate the numeric fields of a layer transform."""
    required_fields = ("x", "y", "scale_x", "scale_y", "rotation_degrees", "anchor_x", "anchor_y")
    for attr in required_fields:
        if not hasattr(transform, attr):
            raise TypeError(f"{field_name} must provide {attr}")
        value = getattr(transform, attr)
        if not _is_real_number(value):
            raise TypeError(f"{field_name}.{attr} must be a number")
        if not np.isfinite(value):
            raise ValueError(f"{field_name}.{attr} must be finite")
    if transform.scale_x == 0.0 or transform.scale_y == 0.0:
        raise ValueError(f"{field_name} scale values must be nonzero")


def _validate_lock_flags(locks: Any, field_name: str) -> None:
    """Validate the boolean lock fields used by executor preconditions."""
    required_fields = ("pixels_locked", "alpha_locked", "position_locked", "visibility_locked", "fully_locked")
    for attr in required_fields:
        if not hasattr(locks, attr):
            raise TypeError(f"{field_name} must provide {attr}")
        if not isinstance(getattr(locks, attr), bool):
            raise TypeError(f"{field_name}.{attr} must be a bool")


def _is_identity_transform(transform: Any) -> bool:
    """Return whether a transform leaves pixel coordinates unchanged."""
    return (
        np.isclose(transform.x, 0.0)
        and np.isclose(transform.y, 0.0)
        and np.isclose(transform.scale_x, 1.0)
        and np.isclose(transform.scale_y, 1.0)
        and np.isclose(transform.rotation_degrees, 0.0)
    )


def _is_real_number(value: Any) -> bool:
    """Return whether `value` is a real scalar number and not a bool."""
    return not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, float, np.integer, np.floating))


def _enum_value(value: Any) -> Any:
    """Return the JSON-friendly value for enums and enum-like strings."""
    return value.value if isinstance(value, Enum) else value


def _rgba_array_from_color(color: Any, width: int, height: int) -> np.ndarray:
    """Create a full-canvas RGBA array filled with `color`."""
    _validate_rgba(color, "canvas.background_color_rgba")
    pixel = np.asarray(color, dtype=np.float32)
    return np.broadcast_to(pixel, (height, width, 4)).copy()


def _source_over(destination: np.ndarray, source_rgb: np.ndarray, source_alpha: np.ndarray) -> np.ndarray:
    """Composite a straight-alpha source over a straight-alpha destination."""
    destination_alpha = destination[..., 3:4]
    output_alpha = source_alpha + destination_alpha * (1.0 - source_alpha)

    source_premultiplied = source_rgb * source_alpha
    destination_premultiplied = destination[..., :3] * destination_alpha
    output_premultiplied = source_premultiplied + destination_premultiplied * (1.0 - source_alpha)

    output_rgb = np.zeros_like(destination[..., :3])
    np.divide(
        output_premultiplied,
        output_alpha,
        out=output_rgb,
        where=output_alpha > 0.0,
    )

    output = np.empty_like(destination)
    output[..., :3] = output_rgb
    output[..., 3:4] = output_alpha
    return np.clip(output, 0.0, 1.0).astype(np.float32, copy=False)


def _clone_layer(layer: Layer) -> Layer:
    """Deep-copy a layer and force pixel storage to be independent."""
    cloned = deepcopy(layer)
    if layer.pixels is not None:
        cloned.pixels = np.array(layer.pixels, copy=True)
    return cloned


def _clone_mask(mask: Mask) -> Mask:
    """Deep-copy a mask and force mask storage to be independent."""
    cloned = deepcopy(mask)
    cloned.data = np.array(mask.data, copy=True)
    return cloned


def _mask_stats(data: np.ndarray) -> dict[str, Any]:
    """Compute JSON-compatible mask statistics without exposing raw mask data."""
    included_y, included_x = np.nonzero(data > 0.0)
    area = int(included_x.size)
    if area == 0:
        bbox = [0, 0, 0, 0]
        centroid = None
    else:
        x_min = int(included_x.min())
        x_max = int(included_x.max())
        y_min = int(included_y.min())
        y_max = int(included_y.max())
        bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]
        centroid = [float(included_x.mean()), float(included_y.mean())]

    binary = bool(np.all(np.isclose(data, 0.0) | np.isclose(data, 1.0)))
    return {
        "area_pixels": area,
        "bbox": bbox,
        "centroid": centroid,
        "min_value": float(data.min()) if data.size else 0.0,
        "max_value": float(data.max()) if data.size else 0.0,
        "is_binary": binary,
    }


def _json_safe(value: Any) -> Any:
    """Convert common project objects into JSON-compatible structures."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else repr(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else repr(number)
    if isinstance(value, np.ndarray):
        return {"array": {"shape": list(value.shape), "dtype": str(value.dtype)}}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=repr)]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return {
            field_name: _json_safe(getattr(value, field_name))
            for field_name in value.__dataclass_fields__
        }
    return repr(value)
