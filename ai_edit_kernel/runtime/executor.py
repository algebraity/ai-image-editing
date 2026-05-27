"""Action executor for the AI Editing Kernel.

The executor is the only runtime component that should mutate `DocumentState`.
Planners produce `Action` objects; validators check them; the executor applies
them; trace sinks record what happened.
"""

from __future__ import annotations

import gzip
import json
import re
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np

try:
    from scipy import ndimage as _ndimage
except ImportError:  # pragma: no cover - exercised only when SciPy is absent
    _ndimage = None

from ai_edit_kernel.document.document_state import CanvasSpec, ColorSpace, DocumentMetadata, DocumentState
from ai_edit_kernel.document.layer import BlendMode, Layer, LayerKind
from ai_edit_kernel.document.mask import Mask, MaskKind
from ai_edit_kernel.region import (
    apply_write_mask as _region_apply_write_mask,
    bbox_from_mask as _region_bbox_from_mask,
    bbox_from_xyxy as _region_bbox_from_xyxy,
    content_bbox_rgba as _region_content_bbox_rgba,
    extract_mask as _region_extract_mask,
    extract_rgba as _region_extract_rgba,
    multiply_alpha_by_mask as _region_multiply_alpha_by_mask,
    paste_crop as _region_paste_crop,
    rect_mask as _region_rect_mask,
    resolve_region_mask as _region_resolve_mask,
)
from ai_edit_kernel.runtime.validator import ValidationReport, Validator
from ai_edit_kernel.schema.actions import Action, ActionBatch, ActionError, ActionResult, ActionStatus, ActionType


class DiffusionBackend(Protocol):
    """Protocol for pluggable diffusion/image-generation backends."""

    def inpaint(self, job: dict[str, Any]) -> dict[str, Any]:
        """Generate pixels for a masked inpainting job and return result assets."""
        ...

    def img2img(self, job: dict[str, Any]) -> dict[str, Any]:
        """Generate an image-to-image result and return result assets."""
        ...

    def txt2img(self, job: dict[str, Any]) -> dict[str, Any]:
        """Generate a text-to-image result and return result assets."""
        ...


class TraceSink(Protocol):
    """Minimal protocol implemented by trace loggers."""

    def log_action_started(self, action: Action, document: DocumentState) -> None:
        """Record that action execution has started."""
        ...

    def log_action_result(self, action: Action, result: ActionResult, document: DocumentState) -> None:
        """Record the final result of an action."""
        ...


@dataclass(slots=True)
class ExecutionOptions:
    """Runtime behavior switches for action execution."""

    dry_run: bool = False
    validate_before: bool = True
    validate_after: bool = True
    rollback_on_failure: bool = True
    allow_full_canvas_writes: bool = False
    record_intermediate_snapshots: bool = True
    strict_mask_guard: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionContext:
    """External dependencies and per-run settings for the executor."""

    options: ExecutionOptions = field(default_factory=ExecutionOptions)
    diffusion_backend: Optional[DiffusionBackend] = None
    trace_sink: Optional[TraceSink] = None
    asset_store: Optional[Any] = None
    validator: Optional[Validator] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Executor:
    """Apply actions to a `DocumentState`."""

    context: ExecutionContext = field(default_factory=ExecutionContext)

    def execute_batch(self, document: DocumentState, batch: ActionBatch) -> list[ActionResult]:
        """Execute a sequence of actions against a document."""
        batch.validate_schema()
        results: list[ActionResult] = []
        for action in batch.actions:
            result = self.execute_action(document, action)
            results.append(result)
            if batch.stop_on_error and not result.succeeded():
                break
        return results

    def execute_action(self, document: DocumentState, action: Action) -> ActionResult:
        """Execute one action against a document."""
        before_revision = document.revision
        snapshot = self.create_rollback_snapshot(document)
        validator = self._validator()

        try:
            action.validate_schema()
            if self.context.options.validate_before:
                report = validator.validate_preconditions(document, action)
                if report.has_errors():
                    return self._failed_result(
                        action,
                        "validation.precondition_failed",
                        "action preconditions failed",
                        before_revision,
                        details={"validation": report.to_json()},
                    )

            if self.context.trace_sink is not None:
                self.context.trace_sink.log_action_started(action, document)

            if self.context.options.dry_run:
                result = ActionResult(
                    action_id=action.id,
                    status=ActionStatus.VALIDATED,
                    document_id=document.id,
                    before_revision=before_revision,
                    after_revision=document.revision,
                    metadata={"dry_run": True},
                )
                self._log_result(action, result, document)
                return result

            result = self.dispatch(document, action)
            result.before_revision = before_revision
            result.document_id = document.id

            if result.succeeded() and _mutates_document(action):
                result.after_revision = document.next_revision()
            else:
                result.after_revision = document.revision

            if result.succeeded() and self.context.options.validate_after:
                report = validator.validate_result(snapshot, document, action, result)
                if report.has_errors():
                    raise RuntimeError(f"post-action validation failed: {report.to_json()}")

            self._log_result(action, result, document)
            return result
        except Exception as exc:
            rolled_back = False
            if self.context.options.rollback_on_failure:
                self.rollback(document, snapshot)
                rolled_back = True
            result = ActionResult(
                action_id=action.id,
                status=ActionStatus.ROLLED_BACK if rolled_back else ActionStatus.FAILED,
                document_id=document.id,
                before_revision=before_revision,
                after_revision=document.revision,
                error=ActionError(
                    code="execution.exception",
                    message=str(exc),
                    action_id=action.id,
                    recoverable=rolled_back,
                ),
                metadata={"rolled_back": rolled_back},
            )
            self._log_result(action, result, document)
            return result

    def dispatch(self, document: DocumentState, action: Action) -> ActionResult:
        """Route an action to its implementation method."""
        handlers = {
            ActionType.NEW_DOCUMENT: self._execute_new_document,
            ActionType.RESIZE_CANVAS: self._execute_resize_canvas,
            ActionType.CROP: self._execute_crop,
            ActionType.IMPORT_IMAGE_AS_LAYER: self._execute_import_image_as_layer,
            ActionType.IMPORT_VECTOR_AS_RASTER: self._execute_import_vector_as_raster,
            ActionType.RASTERIZE_VECTOR_ASSET: self._execute_rasterize_vector_asset,
            ActionType.CREATE_LAYER: self._execute_create_layer,
            ActionType.DELETE_LAYER: self._execute_delete_layer,
            ActionType.DUPLICATE_LAYER: self._execute_duplicate_layer,
            ActionType.RENAME_LAYER: self._execute_rename_layer,
            ActionType.REORDER_LAYER: self._execute_reorder_layer,
            ActionType.SET_ACTIVE_LAYER: self._execute_set_active_layer,
            ActionType.SET_LAYER_VISIBILITY: self._execute_set_layer_visibility,
            ActionType.SET_LAYER_OPACITY: self._execute_set_layer_opacity,
            ActionType.SET_BLEND_MODE: self._execute_set_blend_mode,
            ActionType.MERGE_LAYERS: self._execute_merge_layers,
            ActionType.MOVE_LAYER: self._execute_move_layer,
            ActionType.SCALE_LAYER: self._execute_scale_layer,
            ActionType.ROTATE_LAYER: self._execute_rotate_layer,
            ActionType.FLIP_LAYER: self._execute_flip_layer,
            ActionType.TRANSFORM_LAYER: self._execute_transform_layer,
            ActionType.ALIGN_LAYER: self._execute_align_layer,
            ActionType.ADD_LAYER_MASK: self._execute_add_layer_mask,
            ActionType.APPLY_LAYER_MASK: self._execute_apply_layer_mask,
            ActionType.REMOVE_LAYER_MASK: self._execute_remove_layer_mask,
            ActionType.SELECT_RECT: self._execute_select_rect,
            ActionType.SELECT_ELLIPSE: self._execute_select_ellipse,
            ActionType.SELECT_POLYGON: self._execute_select_polygon,
            ActionType.SELECT_FREEHAND: self._execute_select_polygon,
            ActionType.SELECT_FROM_ALPHA: self._execute_select_from_alpha,
            ActionType.SELECT_COLOR_RANGE: self._execute_select_color_range,
            ActionType.MAGIC_WAND_SELECT: self._execute_magic_wand_select,
            ActionType.SAVE_SELECTION_AS_MASK: self._execute_save_selection_as_mask,
            ActionType.CREATE_MASK_FROM_SHAPE: self._execute_create_mask_from_shape,
            ActionType.GROW_MASK: self._execute_grow_mask,
            ActionType.SHRINK_MASK: self._execute_shrink_mask,
            ActionType.INVERT_MASK: self._execute_invert_mask,
            ActionType.COMBINE_MASKS: self._execute_combine_masks,
            ActionType.FEATHER_MASK: self._execute_feather_mask,
            ActionType.REFINE_SELECTION: self._execute_refine_selection,
            ActionType.REMOVE_SMALL_ISLANDS: self._execute_remove_small_islands,
            ActionType.FILL_MASK_HOLES: self._execute_fill_mask_holes,
            ActionType.DRAW_SHAPE: self._execute_draw_shape,
            ActionType.DRAW_PATH: self._execute_draw_path,
            ActionType.BRUSH_STROKE: self._execute_brush_stroke,
            ActionType.ERASE_STROKE: self._execute_erase_stroke,
            ActionType.PAINT_BUCKET_FILL: self._execute_paint_bucket_fill,
            ActionType.GRADIENT_FILL: self._execute_gradient_fill,
            ActionType.PATTERN_FILL: self._execute_pattern_fill,
            ActionType.BLUR_REGION: self._execute_blur_region,
            ActionType.SHARPEN_REGION: self._execute_sharpen_region,
            ActionType.NOISE_REDUCE: self._execute_noise_reduce,
            ActionType.MEDIAN_FILTER: self._execute_median_filter,
            ActionType.EDGE_DETECT: self._execute_edge_detect,
            ActionType.DROP_SHADOW: self._execute_drop_shadow,
            ActionType.STROKE_SELECTION: self._execute_stroke_selection,
            ActionType.CLEAR_REGION: self._execute_clear_region,
            ActionType.CUT: self._execute_cut,
            ActionType.COPY: self._execute_copy,
            ActionType.PASTE: self._execute_paste,
            ActionType.PASTE_AS_NEW_LAYER: self._execute_paste,
            ActionType.DUPLICATE_REGION_TO_LAYER: self._execute_duplicate_region_to_layer,
            ActionType.ADJUST_BRIGHTNESS_CONTRAST: self._execute_adjust_brightness_contrast,
            ActionType.ADJUST_HUE_SATURATION: self._execute_adjust_hue_saturation,
            ActionType.ADJUST_LEVELS: self._execute_adjust_levels,
            ActionType.ADJUST_CURVES: self._execute_adjust_curves,
            ActionType.COLORIZE: self._execute_colorize,
            ActionType.REPLACE_COLOR: self._execute_replace_color,
            ActionType.DESATURATE: self._execute_desaturate,
            ActionType.CREATE_TEXT_LAYER: self._execute_create_text_layer,
            ActionType.EDIT_TEXT_LAYER: self._execute_edit_text_layer,
            ActionType.RASTERIZE_TEXT_LAYER: self._execute_rasterize_text_layer,
            ActionType.DETECT_SHAPE: self._execute_detect_shape,
            ActionType.DETECT_OBJECTS: self._execute_detect_objects,
            ActionType.SEGMENT_OBJECT: self._execute_segment_object,
            ActionType.ESTIMATE_DEPTH: self._execute_estimate_depth,
            ActionType.EXTRACT_LINE_ART: self._execute_extract_line_art,
            ActionType.DECOMPOSE_TO_LAYERS: self._execute_decompose_to_layers,
            ActionType.TXT2IMG_TO_LAYER: self._execute_txt2img_to_layer,
            ActionType.IMG2IMG_TO_LAYER: self._execute_img2img_to_layer,
            ActionType.INPAINT_REGION: self._execute_inpaint_region,
            ActionType.OUTPAINT_REGION: self._execute_outpaint_region,
            ActionType.EXPORT_FLAT: self._execute_export_flat,
            ActionType.EXPORT_LAYERED_BUNDLE: self._execute_export_layered_bundle,
            ActionType.VALIDATE: self._execute_validate,
            ActionType.NO_OP: self._execute_no_op,
        }
        handler = handlers.get(ActionType(action.type))
        if handler is None:
            return self._failed_result(
                action,
                "execution.unsupported_action",
                f"unsupported action type {ActionType(action.type).value!r}",
                document.revision,
            )
        return handler(document, action)

    def apply_write_mask(self, before_pixels: Any, proposed_pixels: Any, write_mask_id: str, document: DocumentState) -> Any:
        """Blend proposed pixels into old pixels only where the write mask allows."""
        mask = document.get_mask(write_mask_id)
        return _region_apply_write_mask(before_pixels, proposed_pixels, mask.data)

    def create_rollback_snapshot(self, document: DocumentState) -> DocumentState:
        """Create a deep snapshot for action-level rollback."""
        return document.clone_deep()

    def rollback(self, document: DocumentState, snapshot: Any) -> None:
        """Restore a document to a previous deep snapshot."""
        if not isinstance(snapshot, DocumentState):
            raise TypeError("rollback snapshot must be a DocumentState")
        document.id = snapshot.id
        document.canvas = snapshot.canvas
        document.layers = snapshot.layers
        document.masks = snapshot.masks
        document.active_layer_id = snapshot.active_layer_id
        document.active_selection_mask_id = snapshot.active_selection_mask_id
        document.metadata = snapshot.metadata
        document.revision = snapshot.revision
        document.annotations = snapshot.annotations

    def _execute_new_document(self, document: DocumentState, action: Action) -> ActionResult:
        """Replace the current document contents with a new empty canvas."""
        params = action.params
        document.id = action.target.document_id or document.id
        document.canvas = CanvasSpec(
            width=int(params["width"]),
            height=int(params["height"]),
            color_space=ColorSpace(params.get("color_space", ColorSpace.SRGB.value)),
            background_color_rgba=_parse_color(params.get("background_color", "#00000000")),
            dpi=None if params.get("dpi") is None else float(params["dpi"]),
        )
        document.layers = []
        document.masks = {}
        document.active_layer_id = None
        document.active_selection_mask_id = None
        document.metadata = DocumentMetadata(
            title=params.get("title"),
            author=params.get("author"),
            source_file=params.get("source_file"),
            tags=list(params.get("tags", [])),
            custom=dict(params.get("custom_metadata", {})),
        )
        document.annotations = {}
        document.revision = 0
        return ActionResult(
            action_id=action.id,
            status=ActionStatus.EXECUTED,
            metadata={"new_document": {"width": document.canvas.width, "height": document.canvas.height}},
        )

    def _execute_resize_canvas(self, document: DocumentState, action: Action) -> ActionResult:
        """Resize the canvas around its center, padding or cropping all arrays."""
        new_width = int(action.params["width"])
        new_height = int(action.params["height"])
        fill_color = _parse_color(action.params.get("fill_color", "#00000000"))
        old_width = document.canvas.width
        old_height = document.canvas.height

        for layer in document.layers:
            if layer.pixels is not None:
                layer.pixels = _resize_rgba_centered(layer.pixels, new_width, new_height, fill_color)
        for mask in document.masks.values():
            mask.data = _resize_mask_centered(mask.data, new_width, new_height)

        document.canvas = CanvasSpec(
            width=new_width,
            height=new_height,
            color_space=document.canvas.color_space,
            background_color_rgba=document.canvas.background_color_rgba,
            dpi=document.canvas.dpi,
        )
        return ActionResult(
            action_id=action.id,
            status=ActionStatus.EXECUTED,
            changed_layer_ids=[layer.id for layer in document.layers],
            created_mask_ids=[],
            metadata={"old_size": [old_width, old_height], "new_size": [new_width, new_height], "anchor": "center"},
        )

    def _execute_crop(self, document: DocumentState, action: Action) -> ActionResult:
        """Crop the whole document or clear outside a crop on one layer or mask."""
        scope = action.params.get("scope", "document")
        bbox = _region_bbox_from_xyxy(action.params["bbox_xyxy"], document.canvas.width, document.canvas.height)
        y_slice, x_slice = bbox.to_slices()
        if scope == "document":
            old_size = [document.canvas.width, document.canvas.height]
            for layer in document.layers:
                if layer.pixels is not None:
                    layer.pixels = _region_extract_rgba(layer.pixels, bbox)
            for mask in document.masks.values():
                mask.data = _region_extract_mask(mask.data, bbox)
            document.canvas = CanvasSpec(
                width=bbox.width,
                height=bbox.height,
                color_space=document.canvas.color_space,
                background_color_rgba=document.canvas.background_color_rgba,
                dpi=document.canvas.dpi,
            )
            return ActionResult(
                action_id=action.id,
                status=ActionStatus.EXECUTED,
                changed_layer_ids=[layer.id for layer in document.layers],
                created_mask_ids=[],
                metadata={"scope": scope, "old_size": old_size, "new_size": [bbox.width, bbox.height]},
            )

        if scope == "layer":
            layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
            if layer.pixels is None:
                raise ValueError(f"target layer {layer.id!r} has no pixel data")
            fill_color = _parse_color(action.params.get("fill_color", "#00000000"))
            cropped = np.zeros_like(layer.pixels)
            cropped[..., :] = fill_color
            cropped[y_slice, x_slice, :] = layer.pixels[y_slice, x_slice, :]
            layer.pixels = cropped
            return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, changed_layer_ids=[layer.id], metadata={"scope": scope})

        if scope == "mask":
            mask = document.get_mask(_required_target(action.target.mask_id, "target.mask_id"))
            cropped = np.zeros_like(mask.data)
            cropped[y_slice, x_slice] = mask.data[y_slice, x_slice]
            mask.data = cropped.astype(np.float32, copy=False)
            return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, created_mask_ids=[], metadata={"scope": scope, "changed_mask_id": mask.id})

        raise ValueError(f"unsupported crop scope {scope!r}")

    def _execute_create_layer(self, document: DocumentState, action: Action) -> ActionResult:
        """Create a new full-canvas layer and insert it into the document."""
        params = action.params
        output_layer_id = _required_target(action.target.output_layer_id, "target.output_layer_id")
        kind = LayerKind(params.get("kind", LayerKind.RASTER.value))
        width = params.get("width", document.canvas.width)
        height = params.get("height", document.canvas.height)
        if width != document.canvas.width or height != document.canvas.height:
            raise ValueError("prototype layers must match the document canvas size")

        pixels = None
        if kind is not LayerKind.GROUP:
            color = _color_from_params(params, default=(0.0, 0.0, 0.0, 0.0))
            pixels = np.zeros((document.canvas.height, document.canvas.width, 4), dtype=np.float32)
            pixels[..., :] = color

        layer = Layer(
            id=output_layer_id,
            name=params["name"],
            kind=kind,
            pixels=pixels,
            opacity=float(params.get("opacity", 1.0)),
            blend_mode=BlendMode(params.get("blend_mode", BlendMode.NORMAL.value)),
        )
        document.add_layer(layer, params.get("insert_index"))
        if params.get("set_active", True):
            document.set_active_layer(layer.id)

        return ActionResult(
            action_id=action.id,
            status=ActionStatus.EXECUTED,
            created_layer_ids=[layer.id],
        )

    def _execute_import_image_as_layer(self, document: DocumentState, action: Action) -> ActionResult:
        """Load an image file and place it into a full-canvas raster layer."""
        output_layer_id = _required_target(action.target.output_layer_id, "target.output_layer_id")
        path = Path(action.params["path"])
        imported = _load_rgba_image(path)
        image_height, image_width = imported.shape[:2]
        x = _integer_coordinate(action.params.get("x", 0), "params.x")
        y = _integer_coordinate(action.params.get("y", 0), "params.y")
        if x < 0 or y < 0:
            raise ValueError("import_image_as_layer coordinates must be nonnegative")
        if x + image_width > document.canvas.width or y + image_height > document.canvas.height:
            raise ValueError("imported image must fit inside the document canvas")

        pixels = np.zeros((document.canvas.height, document.canvas.width, 4), dtype=np.float32)
        pixels[y : y + image_height, x : x + image_width, :] = imported
        layer = Layer(
            id=output_layer_id,
            name=action.params.get("name", path.stem),
            kind=LayerKind.RASTER,
            pixels=pixels,
            opacity=float(action.params.get("opacity", 1.0)),
            blend_mode=BlendMode(action.params.get("blend_mode", BlendMode.NORMAL.value)),
            metadata={
                "source_path": str(path),
                "import_offset_xy": [x, y],
                "import_size": [image_width, image_height],
            },
        )
        document.add_layer(layer)
        if action.params.get("set_active", True):
            document.set_active_layer(layer.id)

        return ActionResult(
            action_id=action.id,
            status=ActionStatus.EXECUTED,
            created_layer_ids=[layer.id],
            output_assets={"source_path": str(path)},
        )

    def _execute_import_vector_as_raster(self, document: DocumentState, action: Action) -> ActionResult:
        """Rasterize a vector asset and place it into a full-canvas raster layer."""
        output_layer_id = _required_target(action.target.output_layer_id, "target.output_layer_id")
        path = Path(action.params["path"])
        rasterized, render_metadata = _rasterize_vector_asset(
            path,
            width=action.params.get("width"),
            height=action.params.get("height"),
            background_color=action.params.get("background_color"),
        )
        image_height, image_width = rasterized.shape[:2]
        x = _integer_coordinate(action.params.get("x", 0), "params.x")
        y = _integer_coordinate(action.params.get("y", 0), "params.y")
        if x < 0 or y < 0:
            raise ValueError("import_vector_as_raster coordinates must be nonnegative")
        if x + image_width > document.canvas.width or y + image_height > document.canvas.height:
            raise ValueError("rasterized vector image must fit inside the document canvas")

        pixels = np.zeros((document.canvas.height, document.canvas.width, 4), dtype=np.float32)
        pixels[y : y + image_height, x : x + image_width, :] = rasterized
        layer = Layer(
            id=output_layer_id,
            name=action.params.get("name", path.stem),
            kind=LayerKind.RASTER,
            pixels=pixels,
            opacity=float(action.params.get("opacity", 1.0)),
            blend_mode=BlendMode(action.params.get("blend_mode", BlendMode.NORMAL.value)),
            metadata={
                "source_path": str(path),
                "source_format": "vector",
                "rasterized_size": [image_width, image_height],
                "import_offset_xy": [x, y],
                "vector_render": render_metadata,
            },
        )
        document.add_layer(layer)
        if action.params.get("set_active", True):
            document.set_active_layer(layer.id)

        return ActionResult(
            action_id=action.id,
            status=ActionStatus.EXECUTED,
            created_layer_ids=[layer.id],
            output_assets={
                "source_path": str(path),
                "source_format": "vector",
                "rasterized_size": [image_width, image_height],
            },
            metadata={"vector_render": render_metadata},
        )

    def _execute_rasterize_vector_asset(self, document: DocumentState, action: Action) -> ActionResult:
        """Rasterize a vector asset to a standalone PNG or NPY artifact."""
        source_path = Path(action.params["path"])
        output_path = Path(action.params["output_path"])
        rasterized, render_metadata = _rasterize_vector_asset(
            source_path,
            width=action.params.get("width"),
            height=action.params.get("height"),
            background_color=action.params.get("background_color"),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = output_path.suffix.lower()
        if suffix == ".npy":
            np.save(output_path, rasterized)
            output_format = "npy"
        elif suffix == ".png":
            _save_rgba_png(rasterized, output_path)
            output_format = "png"
        else:
            raise ValueError("rasterize_vector_asset output_path must end in .npy or .png")

        return ActionResult(
            action_id=action.id,
            status=ActionStatus.EXECUTED,
            output_assets={"source_path": str(source_path), "path": str(output_path), "format": output_format},
            metadata={"vector_render": render_metadata},
        )

    def _execute_delete_layer(self, document: DocumentState, action: Action) -> ActionResult:
        """Remove a layer after lock and dependency checks."""
        layer_id = _required_target(action.target.layer_id, "target.layer_id")
        document.remove_layer(layer_id)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, changed_layer_ids=[layer_id])

    def _execute_duplicate_layer(self, document: DocumentState, action: Action) -> ActionResult:
        """Create a deep copy of a layer and insert it into the stack."""
        source_id = _required_target(action.target.layer_id, "target.layer_id")
        output_id = _required_target(action.target.output_layer_id, "target.output_layer_id")
        source = document.get_layer(source_id)
        duplicate = source.clone_deep(output_id, action.params.get("name"))
        document.add_layer(duplicate, action.params.get("insert_index"))
        if action.params.get("set_active", True):
            document.set_active_layer(duplicate.id)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, created_layer_ids=[duplicate.id])

    def _execute_rename_layer(self, document: DocumentState, action: Action) -> ActionResult:
        """Rename a layer without changing its ID or pixels."""
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        layer.name = action.params["name"]
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, changed_layer_ids=[layer.id])

    def _execute_reorder_layer(self, document: DocumentState, action: Action) -> ActionResult:
        """Move a layer to a different stack index."""
        layer_id = _required_target(action.target.layer_id, "target.layer_id")
        document.reorder_layer(layer_id, int(action.params["index"]))
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, changed_layer_ids=[layer_id])

    def _execute_set_active_layer(self, document: DocumentState, action: Action) -> ActionResult:
        """Set the document's active layer."""
        layer_id = _required_target(action.target.layer_id, "target.layer_id")
        document.set_active_layer(layer_id)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, changed_layer_ids=[layer_id])

    def _execute_set_layer_visibility(self, document: DocumentState, action: Action) -> ActionResult:
        """Set whether a layer participates in preview compositing."""
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        layer.visible = bool(action.params["visible"])
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, changed_layer_ids=[layer.id])

    def _execute_set_layer_opacity(self, document: DocumentState, action: Action) -> ActionResult:
        """Set layer opacity."""
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        layer.opacity = float(action.params["opacity"])
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, changed_layer_ids=[layer.id])

    def _execute_set_blend_mode(self, document: DocumentState, action: Action) -> ActionResult:
        """Set layer blend mode metadata."""
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        layer.blend_mode = BlendMode(action.params["blend_mode"])
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, changed_layer_ids=[layer.id])

    def _execute_merge_layers(self, document: DocumentState, action: Action) -> ActionResult:
        """Merge layers using normal source-over compositing."""
        mode = action.params.get("mode", "down")
        if mode == "down":
            return self._merge_down(document, action)
        if mode == "visible":
            return self._merge_visible(document, action)
        if mode == "selected":
            return self._merge_selected(document, action)
        if mode == "flatten":
            return self._flatten_image(document, action)
        raise ValueError(f"unsupported merge mode {mode!r}")

    def _execute_move_layer(self, document: DocumentState, action: Action) -> ActionResult:
        """Translate a layer's pixels on the full canvas."""
        params = dict(action.params)
        params["operation"] = "translate"
        return self._transform_layer_pixels(document, action, params)

    def _execute_scale_layer(self, document: DocumentState, action: Action) -> ActionResult:
        """Scale a layer around an anchor point."""
        params = dict(action.params)
        params["operation"] = "scale"
        return self._transform_layer_pixels(document, action, params)

    def _execute_rotate_layer(self, document: DocumentState, action: Action) -> ActionResult:
        """Rotate a layer around an anchor point."""
        params = dict(action.params)
        params["operation"] = "rotate"
        return self._transform_layer_pixels(document, action, params)

    def _execute_flip_layer(self, document: DocumentState, action: Action) -> ActionResult:
        """Flip a layer horizontally and/or vertically around an anchor point."""
        params = dict(action.params)
        params["operation"] = "flip"
        return self._transform_layer_pixels(document, action, params)

    def _execute_transform_layer(self, document: DocumentState, action: Action) -> ActionResult:
        """Apply a destructive affine transform to a full-canvas raster layer."""
        return self._transform_layer_pixels(document, action, action.params)

    def _execute_align_layer(self, document: DocumentState, action: Action) -> ActionResult:
        """Move a layer's nontransparent content to a canvas edge or center."""
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        _require_pixel_layer(layer)
        bbox = _content_bbox(layer.pixels)
        if bbox is None:
            return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, changed_layer_ids=[layer.id], metadata={"empty": True})
        x0, y0, x1, y1 = bbox
        margin = int(action.params.get("margin", 0))
        horizontal = action.params.get("horizontal", "none")
        vertical = action.params.get("vertical", "none")
        dx = 0
        dy = 0
        if horizontal == "left":
            dx = margin - x0
        elif horizontal == "center":
            dx = int(round((document.canvas.width - (x1 - x0)) / 2.0)) - x0
        elif horizontal == "right":
            dx = document.canvas.width - margin - x1
        if vertical == "top":
            dy = margin - y0
        elif vertical == "center":
            dy = int(round((document.canvas.height - (y1 - y0)) / 2.0)) - y0
        elif vertical == "bottom":
            dy = document.canvas.height - margin - y1
        params = {"operation": "translate", "dx": dx, "dy": dy, "fill_color": action.params.get("fill_color", "#00000000")}
        result = self._transform_layer_pixels(document, action, params)
        result.metadata["alignment_delta_xy"] = [dx, dy]
        return result

    def _transform_layer_pixels(self, document: DocumentState, action: Action, params: dict[str, Any]) -> ActionResult:
        """Apply an affine transform by resampling the layer's RGBA pixels."""
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        _require_pixel_layer(layer)
        operation = params.get("operation", "affine")
        anchor = _anchor_point(params.get("anchor"), document.canvas.width, document.canvas.height)
        matrix = _transform_matrix(params, operation, anchor)
        fill_color = _parse_color(params.get("fill_color", "#00000000"))
        resample = params.get("resample", "bilinear")
        layer.pixels = _affine_transform_rgba(layer.pixels, matrix, fill_color, resample)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, changed_layer_ids=[layer.id], metadata={"operation": operation})

    def _execute_add_layer_mask(self, document: DocumentState, action: Action) -> ActionResult:
        """Create a layer-alpha mask and attach it to a layer."""
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        mask_id = _required_target(action.target.mask_id, "target.mask_id")
        mode = action.params.get("mode", "from_selection")
        if mode == "from_selection":
            if document.active_selection_mask_id is None:
                raise ValueError("add_layer_mask from_selection requires an active selection")
            data = np.array(document.get_mask(document.active_selection_mask_id).data, copy=True)
        elif mode == "from_alpha":
            _require_pixel_layer(layer)
            data = np.array(layer.pixels[..., 3], copy=True)
        elif mode == "full":
            data = np.ones((document.canvas.height, document.canvas.width), dtype=np.float32)
        elif mode == "empty":
            data = np.zeros((document.canvas.height, document.canvas.width), dtype=np.float32)
        elif mode == "from_mask":
            data = np.array(document.get_mask(action.params["source_mask_id"]).data, copy=True)
        else:
            raise ValueError(f"unsupported layer mask mode {mode!r}")
        if action.params.get("invert", False):
            data = 1.0 - data
        mask = Mask(
            id=mask_id,
            name=action.params.get("name", f"{layer.name} mask"),
            data=np.clip(data, 0.0, 1.0).astype(np.float32),
            kind=MaskKind.LAYER_ALPHA,
            hard=bool(np.all((data == 0.0) | (data == 1.0))),
            source=action.id,
        )
        document.add_mask(mask)
        layer.mask_id = mask.id
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, created_mask_ids=[mask.id], changed_layer_ids=[layer.id])

    def _execute_apply_layer_mask(self, document: DocumentState, action: Action) -> ActionResult:
        """Bake a layer mask into the layer alpha channel."""
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        _require_pixel_layer(layer)
        if layer.mask_id is None:
            raise ValueError(f"layer {layer.id!r} has no layer mask")
        mask_id = layer.mask_id
        layer.pixels = np.array(layer.pixels, copy=True)
        layer.pixels[..., 3] *= document.get_mask(mask_id).data
        if action.params.get("remove_mask", True):
            layer.mask_id = None
            document.remove_mask(mask_id)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, changed_layer_ids=[layer.id], metadata={"applied_mask_id": mask_id})

    def _execute_remove_layer_mask(self, document: DocumentState, action: Action) -> ActionResult:
        """Detach a layer mask and optionally remove it from the document registry."""
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        if layer.mask_id is None:
            raise ValueError(f"layer {layer.id!r} has no layer mask")
        mask_id = layer.mask_id
        layer.mask_id = None
        if action.params.get("remove_mask", False):
            document.remove_mask(mask_id)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, changed_layer_ids=[layer.id], metadata={"removed_mask_id": mask_id})

    def _execute_select_rect(self, document: DocumentState, action: Action) -> ActionResult:
        """Create a rectangular selection mask and optionally make it active."""
        mask_id = _required_target(action.target.mask_id, "target.mask_id")
        data = _region_rect_mask(document.canvas.width, document.canvas.height, action.params["bbox_xyxy"])
        mask = Mask(
            id=mask_id,
            name=action.params.get("name", mask_id),
            data=data,
            kind=MaskKind.SELECTION,
            hard=True,
            source=action.id,
        )
        document.add_mask(mask)
        if action.params.get("set_active", True):
            document.set_active_selection(mask.id)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, created_mask_ids=[mask.id])

    def _execute_select_color_range(self, document: DocumentState, action: Action) -> ActionResult:
        """Create a mask from pixels close to a target color."""
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        if layer.pixels is None:
            raise ValueError(f"source layer {layer.id!r} has no pixel data")
        mask_id = _required_target(action.target.mask_id, "target.mask_id")
        data = _color_range_mask(
            layer.pixels,
            _parse_color(action.params["color"]),
            float(action.params["tolerance"]),
            float(action.params.get("alpha_min", 0.0)),
        )
        if "bbox_xyxy" in action.params:
            constrained = np.zeros_like(data)
            bbox = _region_bbox_from_xyxy(action.params["bbox_xyxy"], document.canvas.width, document.canvas.height)
            y_slice, x_slice = bbox.to_slices()
            constrained[y_slice, x_slice] = data[y_slice, x_slice]
            data = constrained
        mask = Mask(
            id=mask_id,
            name=action.params.get("name", mask_id),
            data=data,
            kind=MaskKind(action.params.get("kind", MaskKind.SELECTION.value)),
            hard=True,
            source=action.id,
        )
        document.add_mask(mask)
        if action.params.get("set_active", True):
            document.set_active_selection(mask.id)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, created_mask_ids=[mask.id])

    def _execute_select_ellipse(self, document: DocumentState, action: Action) -> ActionResult:
        """Create an elliptical selection mask and set it active if requested."""
        mask_id = _required_target(action.target.mask_id, "target.mask_id")
        data = _ellipse_mask(document.canvas.width, document.canvas.height, action.params["bbox_xyxy"])
        mask = Mask(
            id=mask_id,
            name=action.params.get("name", mask_id),
            data=data,
            kind=MaskKind.SELECTION,
            hard=True,
            source=action.id,
        )
        document.add_mask(mask)
        if action.params.get("set_active", True):
            document.set_active_selection(mask.id)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, created_mask_ids=[mask.id])

    def _execute_select_polygon(self, document: DocumentState, action: Action) -> ActionResult:
        """Create a polygon or freehand selection mask."""
        mask_id = _required_target(action.target.mask_id, "target.mask_id")
        data = _polygon_mask(document.canvas.width, document.canvas.height, action.params["points"], bool(action.params.get("closed", True)))
        mask = Mask(
            id=mask_id,
            name=action.params.get("name", mask_id),
            data=data,
            kind=MaskKind(action.params.get("kind", MaskKind.SELECTION.value)),
            hard=True,
            source=action.id,
        )
        document.add_mask(mask)
        if action.params.get("set_active", True):
            document.set_active_selection(mask.id)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, created_mask_ids=[mask.id])

    def _execute_select_from_alpha(self, document: DocumentState, action: Action) -> ActionResult:
        """Create a selection mask from a layer's alpha channel."""
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        _require_pixel_layer(layer)
        threshold = float(action.params.get("threshold", 0.01))
        data = (layer.pixels[..., 3] >= threshold).astype(np.float32)
        mask = Mask(
            id=_required_target(action.target.mask_id, "target.mask_id"),
            name=action.params.get("name", f"{layer.name} alpha"),
            data=data,
            kind=MaskKind(action.params.get("kind", MaskKind.SELECTION.value)),
            hard=True,
            source=action.id,
        )
        document.add_mask(mask)
        if action.params.get("set_active", True):
            document.set_active_selection(mask.id)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, created_mask_ids=[mask.id])

    def _execute_magic_wand_select(self, document: DocumentState, action: Action) -> ActionResult:
        """Create a contiguous color-based selection from a seed point."""
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        if layer.pixels is None:
            raise ValueError(f"source layer {layer.id!r} has no pixel data")
        mask_id = _required_target(action.target.mask_id, "target.mask_id")
        data = _magic_wand_mask(
            layer.pixels,
            action.params["seed_points"],
            float(action.params["tolerance"]),
            float(action.params.get("alpha_min", 0.0)),
            bool(action.params.get("diagonal", False)),
        )
        mask = Mask(
            id=mask_id,
            name=action.params.get("name", mask_id),
            data=data,
            kind=MaskKind(action.params.get("kind", MaskKind.SELECTION.value)),
            hard=True,
            source=action.id,
        )
        document.add_mask(mask)
        if action.params.get("set_active", True):
            document.set_active_selection(mask.id)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, created_mask_ids=[mask.id])

    def _execute_save_selection_as_mask(self, document: DocumentState, action: Action) -> ActionResult:
        """Copy the active or requested selection mask into a new reusable mask."""
        source_id = action.params.get("source_mask_id", document.active_selection_mask_id)
        if source_id is None:
            raise ValueError("save_selection_as_mask requires an active selection or params.source_mask_id")
        source = document.get_mask(source_id)
        output_id = _required_target(action.target.mask_id, "target.mask_id")
        mask = source.clone(output_id, action.params.get("name", output_id))
        mask.kind = MaskKind(action.params.get("kind", source.kind.value))
        mask.source = action.id
        document.add_mask(mask)
        if action.params.get("set_active", False):
            document.set_active_selection(mask.id)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, created_mask_ids=[mask.id])

    def _execute_create_mask_from_shape(self, document: DocumentState, action: Action) -> ActionResult:
        """Create a hard mask from a rectangle or ellipse shape."""
        mask_id = _required_target(action.target.mask_id, "target.mask_id")
        shape = action.params["shape"]
        data = _shape_mask(document.canvas.width, document.canvas.height, shape)
        mask = Mask(
            id=mask_id,
            name=action.params.get("name", mask_id),
            data=data,
            kind=MaskKind(action.params.get("kind", MaskKind.SELECTION.value)),
            hard=True,
            source=action.id,
        )
        document.add_mask(mask)
        if action.params.get("set_active", False):
            document.set_active_selection(mask.id)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, created_mask_ids=[mask.id])

    def _execute_grow_mask(self, document: DocumentState, action: Action) -> ActionResult:
        """Create a mask by growing another mask."""
        output_id = _required_target(action.target.mask_id, "target.mask_id")
        source = document.get_mask(action.params["source_mask_id"])
        grown = source.dilate(int(action.params["pixels"]), output_id, action.params.get("name", output_id))
        document.add_mask(grown)
        if action.params.get("set_active", False):
            document.set_active_selection(grown.id)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, created_mask_ids=[grown.id])

    def _execute_shrink_mask(self, document: DocumentState, action: Action) -> ActionResult:
        """Create a mask by shrinking another mask."""
        output_id = _required_target(action.target.mask_id, "target.mask_id")
        source = document.get_mask(action.params["source_mask_id"])
        shrunk = source.erode(int(action.params["pixels"]), output_id, action.params.get("name", output_id))
        document.add_mask(shrunk)
        if action.params.get("set_active", False):
            document.set_active_selection(shrunk.id)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, created_mask_ids=[shrunk.id])

    def _execute_invert_mask(self, document: DocumentState, action: Action) -> ActionResult:
        """Create an inverted copy of another mask."""
        output_id = _required_target(action.target.mask_id, "target.mask_id")
        source = document.get_mask(action.params["source_mask_id"])
        inverted = source.invert(output_id, action.params.get("name", output_id))
        document.add_mask(inverted)
        if action.params.get("set_active", False):
            document.set_active_selection(inverted.id)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, created_mask_ids=[inverted.id])

    def _execute_combine_masks(self, document: DocumentState, action: Action) -> ActionResult:
        """Union, intersect, or subtract masks and register the output mask."""
        mask_ids = action.params["mask_ids"]
        operation = action.params["operation"]
        output_id = _required_target(action.target.mask_id, "target.mask_id")
        output_name = action.params.get("name", output_id)

        result = document.get_mask(mask_ids[0]).clone(output_id, output_name)
        for mask_id in mask_ids[1:]:
            other = document.get_mask(mask_id)
            if operation == "union":
                result = result.union(other, output_id, output_name)
            elif operation == "intersect":
                result = result.intersect(other, output_id, output_name)
            elif operation == "subtract":
                result = result.subtract(other, output_id, output_name)
            else:
                raise ValueError(f"unsupported mask operation {operation!r}")

        document.add_mask(result)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, created_mask_ids=[result.id])

    def _execute_feather_mask(self, document: DocumentState, action: Action) -> ActionResult:
        """Create a softened copy of a mask."""
        output_id = _required_target(action.target.mask_id, "target.mask_id")
        source = document.get_mask(action.params["source_mask_id"])
        feathered = source.feather(float(action.params["radius"]), output_id, action.params.get("name", output_id))
        document.add_mask(feathered)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, created_mask_ids=[feathered.id])

    def _execute_refine_selection(self, document: DocumentState, action: Action) -> ActionResult:
        """Apply common cleanup operations to a selection mask."""
        source = document.get_mask(action.params["source_mask_id"])
        data = np.array(source.data, copy=True)
        if "threshold" in action.params:
            data = (data >= float(action.params["threshold"])).astype(np.float32)
        if int(action.params.get("grow_pixels", 0)) > 0:
            _require_scipy("refine_selection grow")
            data = _ndimage.binary_dilation(data > 0.0, structure=_disk_footprint(int(action.params["grow_pixels"]))).astype(np.float32)
        if int(action.params.get("shrink_pixels", 0)) > 0:
            _require_scipy("refine_selection shrink")
            data = _ndimage.binary_erosion(data > 0.0, structure=_disk_footprint(int(action.params["shrink_pixels"]))).astype(np.float32)
        if "min_area" in action.params:
            data = _remove_small_components(data, int(action.params["min_area"]))
        if float(action.params.get("feather_radius", 0.0)) > 0.0:
            _require_scipy("refine_selection feather")
            data = _ndimage.gaussian_filter(data, sigma=float(action.params["feather_radius"]), mode="nearest")
        mask = Mask(
            id=_required_target(action.target.mask_id, "target.mask_id"),
            name=action.params.get("name", f"{source.name} refined"),
            data=np.clip(data, 0.0, 1.0).astype(np.float32),
            kind=source.kind,
            hard=bool(np.all((data == 0.0) | (data == 1.0))),
            source=action.id,
        )
        document.add_mask(mask)
        if action.params.get("set_active", False):
            document.set_active_selection(mask.id)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, created_mask_ids=[mask.id])

    def _execute_remove_small_islands(self, document: DocumentState, action: Action) -> ActionResult:
        """Remove connected mask components below a minimum area."""
        source = document.get_mask(action.params["source_mask_id"])
        min_area = int(action.params.get("min_area", 1))
        data = _remove_small_components(source.data, min_area)
        mask = Mask(
            id=_required_target(action.target.mask_id, "target.mask_id"),
            name=action.params.get("name", f"{source.name} cleaned"),
            data=data,
            kind=source.kind,
            hard=True,
            source=action.id,
        )
        document.add_mask(mask)
        if action.params.get("set_active", False):
            document.set_active_selection(mask.id)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, created_mask_ids=[mask.id])

    def _execute_fill_mask_holes(self, document: DocumentState, action: Action) -> ActionResult:
        """Fill enclosed holes in a hard mask."""
        _require_scipy("fill_mask_holes")
        source = document.get_mask(action.params["source_mask_id"])
        data = _ndimage.binary_fill_holes(source.data > 0.0).astype(np.float32)
        mask = Mask(
            id=_required_target(action.target.mask_id, "target.mask_id"),
            name=action.params.get("name", f"{source.name} holes filled"),
            data=data,
            kind=source.kind,
            hard=True,
            source=action.id,
        )
        document.add_mask(mask)
        if action.params.get("set_active", False):
            document.set_active_selection(mask.id)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, created_mask_ids=[mask.id])

    def _execute_draw_shape(self, document: DocumentState, action: Action) -> ActionResult:
        """Rasterize a rectangle or ellipse onto a target layer."""
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        if layer.pixels is None:
            raise ValueError(f"target layer {layer.id!r} has no pixel data")
        shape = action.params["shape"]
        proposed = np.array(layer.pixels, copy=True)

        fill = action.params.get("fill")
        if fill is not None:
            fill_mask = _shape_mask(document.canvas.width, document.canvas.height, shape)
            _paint_rgba(proposed, fill_mask > 0.0, _parse_color(fill["color"]))

        stroke = action.params.get("stroke")
        if stroke is not None:
            stroke_mask = _stroke_mask(document.canvas.width, document.canvas.height, shape, stroke["width"])
            _paint_rgba(proposed, stroke_mask > 0.0, _parse_color(stroke["color"]))

        layer.pixels = self.apply_write_mask(layer.pixels, proposed, action.write_mask_id, document)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, changed_layer_ids=[layer.id])

    def _execute_draw_path(self, document: DocumentState, action: Action) -> ActionResult:
        """Stroke a polyline path onto a target layer."""
        return self._paint_path(document, action, erase=False)

    def _execute_brush_stroke(self, document: DocumentState, action: Action) -> ActionResult:
        """Paint a brush stroke along a sequence of points."""
        return self._paint_path(document, action, erase=False)

    def _execute_erase_stroke(self, document: DocumentState, action: Action) -> ActionResult:
        """Erase alpha along a sequence of points."""
        return self._paint_path(document, action, erase=True)

    def _paint_path(self, document: DocumentState, action: Action, erase: bool) -> ActionResult:
        """Paint or erase a stroked path through the action write mask."""
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        _require_pixel_layer(layer)
        proposed = np.array(layer.pixels, copy=True)
        stroke_mask = _path_stroke_mask(document.canvas.width, document.canvas.height, action.params["points"], float(action.params.get("width", 1.0)), bool(action.params.get("closed", False)))
        if erase:
            proposed[stroke_mask > 0.0, 3] = 0.0
            proposed[stroke_mask > 0.0, :3] = 0.0
        else:
            color = list(_parse_color(action.params["color"]))
            color[3] *= float(action.params.get("opacity", 1.0))
            mode = action.params.get("mode", "source_over")
            if mode == "replace_rgba":
                proposed[stroke_mask > 0.0, :] = np.asarray(color, dtype=np.float32)
            elif mode == "alpha_to_zero":
                proposed[stroke_mask > 0.0, 3] = 0.0
            else:
                _paint_rgba(proposed, stroke_mask > 0.0, tuple(color))
        layer.pixels = self.apply_write_mask(layer.pixels, proposed, action.write_mask_id, document)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, changed_layer_ids=[layer.id])

    def _execute_clear_region(self, document: DocumentState, action: Action) -> ActionResult:
        """Clear pixels inside the action's write mask on the target layer."""
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        if layer.pixels is None:
            raise ValueError(f"target layer {layer.id!r} has no pixel data")
        proposed = np.array(layer.pixels, copy=True)
        mode = action.params.get("mode", "alpha_to_zero")
        preserve_rgb = action.params.get("preserve_rgb", False)

        if mode == "alpha_to_zero":
            proposed[..., 3] = 0.0
            if not preserve_rgb:
                proposed[..., :3] = 0.0
        elif mode == "rgba_to_zero":
            proposed[..., :] = 0.0
        else:
            raise ValueError(f"unsupported clear mode {mode!r}")

        layer.pixels = self.apply_write_mask(layer.pixels, proposed, action.write_mask_id, document)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, changed_layer_ids=[layer.id])

    def _execute_paint_bucket_fill(self, document: DocumentState, action: Action) -> ActionResult:
        """Fill a contiguous or preselected region with a color/texture."""
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        if layer.pixels is None:
            raise ValueError(f"target layer {layer.id!r} has no pixel data")
        proposed = np.array(layer.pixels, copy=True)
        color = _parse_color(action.params["color"])
        mode = action.params.get("mode", "replace_rgb_preserve_alpha")

        if mode == "replace_rgb_preserve_alpha":
            proposed[..., :3] = color[:3]
        elif mode == "replace_rgba":
            proposed[..., :] = color
        elif mode == "source_over":
            _paint_rgba(proposed, np.ones(proposed.shape[:2], dtype=bool), color)
        else:
            raise ValueError(f"unsupported paint bucket mode {mode!r}")

        layer.pixels = self.apply_write_mask(layer.pixels, proposed, action.write_mask_id, document)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, changed_layer_ids=[layer.id])

    def _execute_gradient_fill(self, document: DocumentState, action: Action) -> ActionResult:
        """Fill the write mask with a linear or radial color gradient."""
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        _require_pixel_layer(layer)
        proposed = np.array(layer.pixels, copy=True)
        fill = _gradient_pixels(document.canvas.width, document.canvas.height, action.params)
        mode = action.params.get("mode", "replace_rgba")
        proposed = _apply_fill_mode(proposed, fill, mode)
        layer.pixels = self.apply_write_mask(layer.pixels, proposed, action.write_mask_id, document)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, changed_layer_ids=[layer.id])

    def _execute_pattern_fill(self, document: DocumentState, action: Action) -> ActionResult:
        """Fill the write mask with a simple repeating pattern."""
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        _require_pixel_layer(layer)
        proposed = np.array(layer.pixels, copy=True)
        fill = _pattern_pixels(document.canvas.width, document.canvas.height, action.params)
        mode = action.params.get("mode", "replace_rgba")
        proposed = _apply_fill_mode(proposed, fill, mode)
        layer.pixels = self.apply_write_mask(layer.pixels, proposed, action.write_mask_id, document)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, changed_layer_ids=[layer.id])

    def _execute_blur_region(self, document: DocumentState, action: Action) -> ActionResult:
        """Blur selected channels inside a write mask."""
        _require_scipy("blur_region")
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        if layer.pixels is None:
            raise ValueError(f"target layer {layer.id!r} has no pixel data")
        radius = float(action.params["radius"])
        channels = _channels(action.params.get("channels", "rgb"))
        edge_mode = action.params.get("edge_mode", "nearest")
        proposed = np.array(layer.pixels, copy=True)
        if radius > 0.0:
            blurred = np.empty_like(layer.pixels)
            for channel_index in range(4):
                blurred[..., channel_index] = _ndimage.gaussian_filter(
                    layer.pixels[..., channel_index],
                    sigma=radius,
                    mode=edge_mode,
                )
            for channel_name, channel_index in {"r": 0, "g": 1, "b": 2, "a": 3}.items():
                if channel_name in channels:
                    proposed[..., channel_index] = blurred[..., channel_index]
        layer.pixels = self.apply_write_mask(layer.pixels, proposed, action.write_mask_id, document)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, changed_layer_ids=[layer.id])

    def _execute_sharpen_region(self, document: DocumentState, action: Action) -> ActionResult:
        """Sharpen RGB channels inside a write mask."""
        _require_scipy("sharpen_region")
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        _require_pixel_layer(layer)
        radius = float(action.params.get("radius", 1.0))
        amount = float(action.params.get("amount", 1.0))
        proposed = np.array(layer.pixels, copy=True)
        blurred = np.empty_like(layer.pixels[..., :3])
        for channel_index in range(3):
            blurred[..., channel_index] = _ndimage.gaussian_filter(layer.pixels[..., channel_index], sigma=radius, mode="nearest")
        proposed[..., :3] = np.clip(layer.pixels[..., :3] + (layer.pixels[..., :3] - blurred) * amount, 0.0, 1.0)
        layer.pixels = self.apply_write_mask(layer.pixels, proposed, action.write_mask_id, document)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, changed_layer_ids=[layer.id])

    def _execute_noise_reduce(self, document: DocumentState, action: Action) -> ActionResult:
        """Apply a conservative median denoise to RGB channels."""
        _require_scipy("noise_reduce")
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        _require_pixel_layer(layer)
        radius = max(1, int(round(float(action.params.get("radius", 1.0)))))
        proposed = np.array(layer.pixels, copy=True)
        size = radius * 2 + 1
        for channel_index in range(3):
            proposed[..., channel_index] = _ndimage.median_filter(layer.pixels[..., channel_index], size=size, mode="nearest")
        layer.pixels = self.apply_write_mask(layer.pixels, proposed, action.write_mask_id, document)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, changed_layer_ids=[layer.id])

    def _execute_median_filter(self, document: DocumentState, action: Action) -> ActionResult:
        """Apply a median filter to selected channels."""
        _require_scipy("median_filter")
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        _require_pixel_layer(layer)
        radius = max(1, int(round(float(action.params.get("radius", 1.0)))))
        channels = _channels(action.params.get("channels", "rgb"))
        proposed = np.array(layer.pixels, copy=True)
        size = radius * 2 + 1
        for channel_name, channel_index in {"r": 0, "g": 1, "b": 2, "a": 3}.items():
            if channel_name in channels:
                proposed[..., channel_index] = _ndimage.median_filter(layer.pixels[..., channel_index], size=size, mode="nearest")
        layer.pixels = self.apply_write_mask(layer.pixels, proposed, action.write_mask_id, document)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, changed_layer_ids=[layer.id])

    def _execute_edge_detect(self, document: DocumentState, action: Action) -> ActionResult:
        """Detect luminance edges and write them into the selected region."""
        _require_scipy("edge_detect")
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        _require_pixel_layer(layer)
        luminance = _luminance(layer.pixels[..., :3])
        sx = _ndimage.sobel(luminance, axis=1, mode="nearest")
        sy = _ndimage.sobel(luminance, axis=0, mode="nearest")
        edges = np.clip(np.hypot(sx, sy), 0.0, 1.0)
        proposed = np.array(layer.pixels, copy=True)
        mode = action.params.get("mode", "luminance")
        if mode == "alpha":
            proposed[..., 3] = edges
        else:
            proposed[..., :3] = edges[..., np.newaxis]
            proposed[..., 3] = np.maximum(proposed[..., 3], edges)
        layer.pixels = self.apply_write_mask(layer.pixels, proposed, action.write_mask_id, document)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, changed_layer_ids=[layer.id])

    def _execute_stroke_selection(self, document: DocumentState, action: Action) -> ActionResult:
        """Paint an outline around a mask or the active selection."""
        _require_scipy("stroke_selection")
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        _require_pixel_layer(layer)
        source_id = action.params.get("source_mask_id", document.active_selection_mask_id)
        if source_id is None:
            raise ValueError("stroke_selection requires an active selection or params.source_mask_id")
        source = document.get_mask(source_id)
        radius = max(1, int(round(float(action.params.get("radius", 1.0)))))
        outer = _ndimage.binary_dilation(source.data > 0.0, structure=_disk_footprint(radius))
        inner = _ndimage.binary_erosion(source.data > 0.0, structure=_disk_footprint(max(radius - 1, 0))) if radius > 1 else source.data > 0.0
        stroke = outer & ~inner
        proposed = np.array(layer.pixels, copy=True)
        _paint_rgba(proposed, stroke, _parse_color(action.params.get("color", "#000000")))
        layer.pixels = self.apply_write_mask(layer.pixels, proposed, action.write_mask_id, document)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, changed_layer_ids=[layer.id])

    def _execute_drop_shadow(self, document: DocumentState, action: Action) -> ActionResult:
        """Create a blurred offset shadow below the target layer."""
        _require_scipy("drop_shadow")
        source = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        _require_pixel_layer(source)
        offset_x, offset_y = _point_to_float_pair(action.params.get("offset", [8, 8]))
        blur_radius = float(action.params.get("blur_radius", action.params.get("radius", 8.0)))
        color = _parse_color(action.params.get("color", "#00000080"))
        opacity = float(action.params.get("opacity", 1.0))
        alpha = source.pixels[..., 3]
        shifted = _translate_mask(alpha, int(round(offset_x)), int(round(offset_y)))
        if blur_radius > 0.0:
            shifted = _ndimage.gaussian_filter(shifted, sigma=blur_radius, mode="constant", cval=0.0)
        pixels = np.zeros_like(source.pixels)
        pixels[..., :3] = color[:3]
        pixels[..., 3] = np.clip(shifted * color[3] * opacity, 0.0, 1.0)
        output_id = action.target.output_layer_id or f"{source.id}_shadow"
        shadow = Layer(
            id=output_id,
            name=action.params.get("output_layer_name", f"{source.name} shadow"),
            kind=LayerKind.RASTER,
            pixels=pixels.astype(np.float32),
            opacity=1.0,
            blend_mode=BlendMode.NORMAL,
        )
        document.add_layer(shadow, _layer_index(document, source.id))
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, created_layer_ids=[shadow.id])

    def _execute_adjust_brightness_contrast(self, document: DocumentState, action: Action) -> ActionResult:
        """Adjust layer brightness and contrast in RGB."""
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        _require_pixel_layer(layer)
        brightness = float(action.params.get("brightness", 0.0))
        contrast = float(action.params.get("contrast", 1.0))
        proposed = np.array(layer.pixels, copy=True)
        proposed[..., :3] = np.clip((proposed[..., :3] - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)
        layer.pixels = self.apply_write_mask(layer.pixels, proposed, action.write_mask_id, document)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, changed_layer_ids=[layer.id])

    def _execute_adjust_hue_saturation(self, document: DocumentState, action: Action) -> ActionResult:
        """Adjust hue, saturation, and lightness in HSV space."""
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        _require_pixel_layer(layer)
        proposed = np.array(layer.pixels, copy=True)
        hsv = _rgb_to_hsv(proposed[..., :3])
        hsv[..., 0] = (hsv[..., 0] + float(action.params.get("hue_degrees", 0.0)) / 360.0) % 1.0
        hsv[..., 1] = np.clip(hsv[..., 1] * float(action.params.get("saturation", 1.0)), 0.0, 1.0)
        value = _hsv_to_rgb(hsv)
        lightness = float(action.params.get("lightness", 0.0))
        proposed[..., :3] = np.clip(value + lightness, 0.0, 1.0)
        layer.pixels = self.apply_write_mask(layer.pixels, proposed, action.write_mask_id, document)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, changed_layer_ids=[layer.id])

    def _execute_adjust_levels(self, document: DocumentState, action: Action) -> ActionResult:
        """Apply simple RGB levels remapping."""
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        _require_pixel_layer(layer)
        in_black = float(action.params.get("in_black", 0.0))
        in_white = float(action.params.get("in_white", 1.0))
        out_black = float(action.params.get("out_black", 0.0))
        out_white = float(action.params.get("out_white", 1.0))
        gamma = float(action.params.get("gamma", 1.0))
        if in_white <= in_black or gamma <= 0.0:
            raise ValueError("levels require in_white > in_black and gamma > 0")
        proposed = np.array(layer.pixels, copy=True)
        normalized = np.clip((proposed[..., :3] - in_black) / (in_white - in_black), 0.0, 1.0)
        normalized = normalized ** (1.0 / gamma)
        proposed[..., :3] = np.clip(out_black + normalized * (out_white - out_black), 0.0, 1.0)
        layer.pixels = self.apply_write_mask(layer.pixels, proposed, action.write_mask_id, document)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, changed_layer_ids=[layer.id])

    def _execute_adjust_curves(self, document: DocumentState, action: Action) -> ActionResult:
        """Apply a shared piecewise-linear curve to RGB channels."""
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        _require_pixel_layer(layer)
        points = sorted((_point_to_float_pair(point) for point in action.params["points"]), key=lambda item: item[0])
        xs = np.asarray([point[0] for point in points], dtype=np.float32)
        ys = np.asarray([point[1] for point in points], dtype=np.float32)
        proposed = np.array(layer.pixels, copy=True)
        proposed[..., :3] = np.interp(proposed[..., :3], xs, ys).astype(np.float32)
        proposed[..., :3] = np.clip(proposed[..., :3], 0.0, 1.0)
        layer.pixels = self.apply_write_mask(layer.pixels, proposed, action.write_mask_id, document)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, changed_layer_ids=[layer.id])

    def _execute_colorize(self, document: DocumentState, action: Action) -> ActionResult:
        """Colorize a layer while preserving luminance."""
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        _require_pixel_layer(layer)
        target = np.asarray(_parse_color(action.params.get("color", "#ff0000"))[:3], dtype=np.float32)
        amount = float(action.params.get("amount", 1.0))
        proposed = np.array(layer.pixels, copy=True)
        lum = _luminance(proposed[..., :3])[..., np.newaxis]
        colorized = lum * target
        proposed[..., :3] = np.clip(proposed[..., :3] * (1.0 - amount) + colorized * amount, 0.0, 1.0)
        layer.pixels = self.apply_write_mask(layer.pixels, proposed, action.write_mask_id, document)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, changed_layer_ids=[layer.id])

    def _execute_replace_color(self, document: DocumentState, action: Action) -> ActionResult:
        """Replace pixels near one color with another color."""
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        _require_pixel_layer(layer)
        source = np.asarray(_parse_color(action.params["source_color"])[:3], dtype=np.float32)
        target = np.asarray(_parse_color(action.params["target_color"])[:3], dtype=np.float32)
        tolerance = float(action.params.get("tolerance", 0.1))
        softness = max(float(action.params.get("softness", 0.0)), 1e-6)
        distance = np.linalg.norm(layer.pixels[..., :3] - source, axis=-1)
        weight = np.clip((tolerance + softness - distance) / softness, 0.0, 1.0)[..., np.newaxis]
        proposed = np.array(layer.pixels, copy=True)
        proposed[..., :3] = np.clip(proposed[..., :3] * (1.0 - weight) + target * weight, 0.0, 1.0)
        layer.pixels = self.apply_write_mask(layer.pixels, proposed, action.write_mask_id, document)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, changed_layer_ids=[layer.id])

    def _execute_desaturate(self, document: DocumentState, action: Action) -> ActionResult:
        """Convert RGB toward grayscale."""
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        _require_pixel_layer(layer)
        amount = float(action.params.get("amount", 1.0))
        proposed = np.array(layer.pixels, copy=True)
        gray = _luminance(proposed[..., :3])[..., np.newaxis]
        proposed[..., :3] = np.clip(proposed[..., :3] * (1.0 - amount) + gray * amount, 0.0, 1.0)
        layer.pixels = self.apply_write_mask(layer.pixels, proposed, action.write_mask_id, document)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, changed_layer_ids=[layer.id])

    def _execute_copy(self, document: DocumentState, action: Action) -> ActionResult:
        """Copy a masked region into the executor clipboard."""
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        _require_pixel_layer(layer)
        pixels, bbox = _copy_region_pixels(document, layer, action.params)
        self.context.metadata["clipboard"] = {
            "pixels": pixels,
            "bbox_xyxy": bbox,
            "source_layer_id": layer.id,
        }
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, metadata={"clipboard_bbox_xyxy": bbox})

    def _execute_cut(self, document: DocumentState, action: Action) -> ActionResult:
        """Copy a region to the clipboard and clear it from the source layer."""
        result = self._execute_copy(document, action)
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        mask = _region_mask(document, action.params)
        proposed = np.array(layer.pixels, copy=True)
        if action.params.get("clear_mode", "alpha_to_zero") == "rgba_to_zero":
            proposed[mask > 0.0, :] = 0.0
        else:
            proposed[mask > 0.0, 3] = 0.0
            if not action.params.get("preserve_rgb", False):
                proposed[mask > 0.0, :3] = 0.0
        layer.pixels = proposed
        result.changed_layer_ids.append(layer.id)
        return result

    def _execute_paste(self, document: DocumentState, action: Action) -> ActionResult:
        """Paste clipboard pixels as a new full-canvas layer."""
        clipboard = self.context.metadata.get("clipboard")
        if not isinstance(clipboard, dict) or not isinstance(clipboard.get("pixels"), np.ndarray):
            raise ValueError("paste requires clipboard pixels created by copy or cut")
        output_id = _required_target(action.target.output_layer_id, "target.output_layer_id")
        clip = clipboard["pixels"]
        x = int(action.params.get("x", clipboard["bbox_xyxy"][0]))
        y = int(action.params.get("y", clipboard["bbox_xyxy"][1]))
        pixels = _region_paste_crop(
            np.zeros((document.canvas.height, document.canvas.width, 4), dtype=np.float32),
            clip,
            x,
            y,
        )
        layer = Layer(
            id=output_id,
            name=action.params.get("name", "Pasted Layer"),
            kind=LayerKind.RASTER,
            pixels=pixels,
        )
        document.add_layer(layer)
        if action.params.get("set_active", True):
            document.set_active_layer(layer.id)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, created_layer_ids=[layer.id])

    def _execute_duplicate_region_to_layer(self, document: DocumentState, action: Action) -> ActionResult:
        """Copy a source region directly into a new layer."""
        self._execute_copy(document, action)
        return self._execute_paste(document, action)

    def _execute_inpaint_region(self, document: DocumentState, action: Action) -> ActionResult:
        """Call the diffusion backend for a masked region and composite the result."""
        return self._execute_diffusion_region(document, action, "inpaint")

    def _execute_img2img_to_layer(self, document: DocumentState, action: Action) -> ActionResult:
        """Call an image-to-image backend and place the result on a target layer."""
        return self._execute_diffusion_to_layer(document, action, "img2img")

    def _execute_txt2img_to_layer(self, document: DocumentState, action: Action) -> ActionResult:
        """Call a text-to-image backend and import the result as a new layer."""
        return self._execute_diffusion_to_layer(document, action, "txt2img")

    def _execute_outpaint_region(self, document: DocumentState, action: Action) -> ActionResult:
        """Call the diffusion backend for an outpainting region and composite the result."""
        return self._execute_diffusion_region(document, action, "inpaint")

    def _execute_create_text_layer(self, document: DocumentState, action: Action) -> ActionResult:
        """Create a rasterized text layer with editable text metadata."""
        output_id = _required_target(action.target.output_layer_id, "target.output_layer_id")
        pixels, metadata = _render_text_pixels(document.canvas.width, document.canvas.height, action.params)
        layer = Layer(
            id=output_id,
            name=action.params.get("name", "Text"),
            kind=LayerKind.TEXT,
            pixels=pixels,
            metadata=metadata,
        )
        document.add_layer(layer)
        if action.params.get("set_active", True):
            document.set_active_layer(layer.id)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, created_layer_ids=[layer.id])

    def _execute_edit_text_layer(self, document: DocumentState, action: Action) -> ActionResult:
        """Update text metadata and rerender the text layer."""
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        params = dict(layer.metadata.get("text", {}))
        params.update(action.params)
        pixels, metadata = _render_text_pixels(document.canvas.width, document.canvas.height, params)
        layer.kind = LayerKind.TEXT
        layer.pixels = pixels
        layer.metadata.update(metadata)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, changed_layer_ids=[layer.id])

    def _execute_rasterize_text_layer(self, document: DocumentState, action: Action) -> ActionResult:
        """Convert a text layer to an ordinary raster layer while preserving pixels."""
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        _require_pixel_layer(layer)
        layer.kind = LayerKind.RASTER
        layer.metadata["rasterized_from"] = "text"
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, changed_layer_ids=[layer.id])

    def _execute_detect_shape(self, document: DocumentState, action: Action) -> ActionResult:
        """Detect a coarse geometric shape from layer alpha."""
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        _require_pixel_layer(layer)
        threshold = float(action.params.get("threshold", action.params.get("alpha_min", 0.01)))
        observation = _shape_observation_from_mask(layer.pixels[..., 3] >= threshold)
        document.annotations.setdefault("observations", {})[action.id] = observation
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, metadata={"observation": observation})

    def _execute_detect_objects(self, document: DocumentState, action: Action) -> ActionResult:
        """Detect connected alpha components as coarse object observations."""
        _require_scipy("detect_objects")
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        _require_pixel_layer(layer)
        threshold = float(action.params.get("alpha_min", 0.01))
        min_area = int(action.params.get("min_area", 1))
        max_objects = int(action.params.get("max_objects", 32))
        observations = _connected_component_observations(layer.pixels[..., 3] >= threshold, min_area, max_objects)
        document.annotations.setdefault("observations", {})[action.id] = observations
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, metadata={"observations": observations})

    def _execute_segment_object(self, document: DocumentState, action: Action) -> ActionResult:
        """Create an object mask from seed points, alpha, or luminance."""
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        _require_pixel_layer(layer)
        mode = action.params.get("mode", "alpha")
        if "seed_points" in action.params:
            data = _magic_wand_mask(
                layer.pixels,
                action.params["seed_points"],
                float(action.params.get("tolerance", 0.1)),
                float(action.params.get("alpha_min", 0.01)),
                diagonal=True,
            )
        elif mode == "luminance":
            data = (_luminance(layer.pixels[..., :3]) >= float(action.params.get("threshold", 0.5))).astype(np.float32)
        else:
            data = (layer.pixels[..., 3] >= float(action.params.get("alpha_min", 0.01))).astype(np.float32)
        mask = Mask(
            id=_required_target(action.target.mask_id, "target.mask_id"),
            name=action.params.get("name", "segmented object"),
            data=data,
            kind=MaskKind.OBJECT,
            hard=True,
            source=action.id,
        )
        document.add_mask(mask)
        if action.params.get("set_active", True):
            document.set_active_selection(mask.id)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, created_mask_ids=[mask.id])

    def _execute_estimate_depth(self, document: DocumentState, action: Action) -> ActionResult:
        """Create a simple luminance-based depth proxy mask."""
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        _require_pixel_layer(layer)
        data = _luminance(layer.pixels[..., :3]).astype(np.float32)
        if action.params.get("mode", "luminance") == "alpha":
            data = layer.pixels[..., 3].astype(np.float32)
        mask = Mask(
            id=_required_target(action.target.mask_id, "target.mask_id"),
            name=action.params.get("name", "estimated depth"),
            data=np.clip(data, 0.0, 1.0),
            kind=MaskKind.OBJECT,
            hard=False,
            source=action.id,
        )
        document.add_mask(mask)
        if action.params.get("set_active", True):
            document.set_active_selection(mask.id)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, created_mask_ids=[mask.id])

    def _execute_extract_line_art(self, document: DocumentState, action: Action) -> ActionResult:
        """Create a mask from detected luminance edges."""
        _require_scipy("extract_line_art")
        layer = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        _require_pixel_layer(layer)
        luminance = _luminance(layer.pixels[..., :3])
        edges = np.hypot(_ndimage.sobel(luminance, axis=1, mode="nearest"), _ndimage.sobel(luminance, axis=0, mode="nearest"))
        threshold = float(action.params.get("threshold", 0.25))
        data = (edges >= threshold).astype(np.float32)
        mask = Mask(
            id=_required_target(action.target.mask_id, "target.mask_id"),
            name=action.params.get("name", "line art"),
            data=data,
            kind=MaskKind.LINE_ART_REGION,
            hard=True,
            source=action.id,
        )
        document.add_mask(mask)
        if action.params.get("set_active", True):
            document.set_active_selection(mask.id)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, created_mask_ids=[mask.id])

    def _execute_decompose_to_layers(self, document: DocumentState, action: Action) -> ActionResult:
        """Split connected alpha components into independent raster layers."""
        _require_scipy("decompose_to_layers")
        source = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        _require_pixel_layer(source)
        min_area = int(action.params.get("min_area", 1))
        max_objects = int(action.params.get("max_objects", 16))
        labels, count = _ndimage.label(source.pixels[..., 3] > float(action.params.get("alpha_min", 0.01)))
        created: list[str] = []
        for label in range(1, count + 1):
            region = labels == label
            if int(np.count_nonzero(region)) < min_area:
                continue
            if len(created) >= max_objects:
                break
            output_id = f"{action.target.output_layer_id}_{len(created) + 1}"
            pixels = np.zeros_like(source.pixels)
            pixels[region, :] = source.pixels[region, :]
            layer = Layer(
                id=output_id,
                name=f"{action.params.get('output_layer_name', source.name)} {len(created) + 1}",
                kind=LayerKind.RASTER,
                pixels=pixels,
            )
            document.add_layer(layer)
            created.append(output_id)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, created_layer_ids=created)

    def _execute_diffusion_to_layer(self, document: DocumentState, action: Action, method: str) -> ActionResult:
        """Call a configured diffusion backend and import returned pixels as a layer."""
        backend = self.context.diffusion_backend
        if backend is None:
            raise RuntimeError(f"{method} requires a configured diffusion_backend")
        job = self._diffusion_job(document, action)
        response = getattr(backend, method)(job)
        pixels = _pixels_from_backend_response(response)
        pixels = _fit_pixels_to_canvas(pixels, document.canvas.width, document.canvas.height)
        output_id = _required_target(action.target.output_layer_id, "target.output_layer_id")
        layer = Layer(
            id=output_id,
            name=action.params.get("output_layer_name", method),
            kind=LayerKind.RASTER,
            pixels=pixels,
        )
        document.add_layer(layer)
        document.set_active_layer(layer.id)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, created_layer_ids=[layer.id], output_assets=dict(response.get("assets", {})))

    def _execute_diffusion_region(self, document: DocumentState, action: Action, method: str) -> ActionResult:
        """Call a diffusion backend and composite the returned pixels through the write mask."""
        backend = self.context.diffusion_backend
        if backend is None:
            raise RuntimeError(f"{method} requires a configured diffusion_backend")
        target = document.get_layer(_required_target(action.target.layer_id, "target.layer_id"))
        _require_pixel_layer(target)
        job = self._diffusion_job(document, action)
        response = getattr(backend, method)(job)
        generated = _fit_pixels_to_canvas(_pixels_from_backend_response(response), document.canvas.width, document.canvas.height)
        output_id = _required_target(action.target.output_layer_id, "target.output_layer_id")
        if action.params.get("mode", "replace_region") == "new_layer":
            layer = Layer(id=output_id, name=action.params.get("output_layer_name", method), kind=LayerKind.RASTER, pixels=generated)
            document.add_layer(layer)
            return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, created_layer_ids=[layer.id], output_assets=dict(response.get("assets", {})))
        proposed = np.array(target.pixels, copy=True)
        proposed[..., :] = generated
        target.pixels = self.apply_write_mask(target.pixels, proposed, action.write_mask_id, document)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, changed_layer_ids=[target.id], output_assets=dict(response.get("assets", {})))

    def _diffusion_job(self, document: DocumentState, action: Action) -> dict[str, Any]:
        """Build a backend job payload with prompt, preview, and mask context."""
        job = dict(action.params.get("job", {}))
        job.update(
            {
                "prompt": action.params.get("prompt", ""),
                "negative_prompt": action.params.get("negative_prompt"),
                "seed": action.params.get("seed"),
                "denoise": action.params.get("denoise"),
                "document_id": document.id,
                "canvas": {"width": document.canvas.width, "height": document.canvas.height},
                "preview": document.flatten_preview(),
            }
        )
        if action.write_mask_id is not None:
            job["mask"] = document.get_mask(action.write_mask_id).data
        if action.target.layer_id is not None:
            layer = document.get_layer(action.target.layer_id)
            if layer.pixels is not None:
                job["target_pixels"] = layer.pixels
        return job

    def _execute_export_flat(self, document: DocumentState, action: Action) -> ActionResult:
        """Export a flattened preview to `.npy` or `.png`."""
        path = Path(action.params["path"])
        preview = document.flatten_preview()
        suffix = path.suffix.lower()
        path.parent.mkdir(parents=True, exist_ok=True)

        if suffix == ".npy":
            np.save(path, preview)
        elif suffix == ".png":
            try:
                from PIL import Image
            except ImportError as exc:
                raise RuntimeError("PNG export requires Pillow") from exc
            image = Image.fromarray(np.clip(preview * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGBA")
            image.save(path)
        else:
            raise ValueError("export_flat path must end in .npy or .png")

        return ActionResult(
            action_id=action.id,
            status=ActionStatus.EXECUTED,
            output_assets={"path": str(path), "format": suffix[1:]},
        )

    def _execute_export_layered_bundle(self, document: DocumentState, action: Action) -> ActionResult:
        """Export a directory bundle containing document, layer, mask, and preview assets."""
        root = Path(action.params["path"])
        overwrite = bool(action.params.get("overwrite", True))
        if root.exists() and any(root.iterdir()) and not overwrite:
            raise FileExistsError(f"layered bundle path {root} already exists and is not empty")
        root.mkdir(parents=True, exist_ok=True)
        layers_dir = root / "layers"
        masks_dir = root / "masks"
        layers_dir.mkdir(exist_ok=True)
        masks_dir.mkdir(exist_ok=True)

        include_hidden = bool(action.params.get("include_hidden", True))
        layer_entries: list[dict[str, Any]] = []
        for index, layer in enumerate(document.layers):
            entry = {
                "id": layer.id,
                "name": layer.name,
                "index": index,
                "kind": LayerKind(layer.kind).value,
                "visible": layer.visible,
                "opacity": float(layer.opacity),
                "blend_mode": BlendMode(layer.blend_mode).value,
                "mask_id": layer.mask_id,
                "pixels": None,
            }
            if layer.pixels is not None and (include_hidden or layer.visible):
                filename = f"layer_{index:04d}_{_safe_filename_segment(layer.id)}.png"
                _save_rgba_png(layer.pixels, layers_dir / filename)
                entry["pixels"] = {"path": f"layers/{filename}", "format": "png", "shape": list(layer.pixels.shape)}
            layer_entries.append(entry)

        mask_entries: list[dict[str, Any]] = []
        for mask in document.masks.values():
            filename = f"{_safe_filename_segment(mask.id)}.png"
            _save_mask_png(mask.data, masks_dir / filename)
            mask_entries.append(
                {
                    "id": mask.id,
                    "name": mask.name,
                    "kind": MaskKind(mask.kind).value,
                    "hard": mask.hard,
                    "source": mask.source,
                    "data": {"path": f"masks/{filename}", "format": "png", "shape": list(mask.data.shape)},
                }
            )

        preview_entry = None
        if action.params.get("include_preview", True):
            preview_path = root / "preview.png"
            _save_rgba_png(document.flatten_preview(include_hidden=include_hidden), preview_path)
            preview_entry = {"path": "preview.png", "format": "png"}

        snapshot_path = root / "document_snapshot.json"
        _write_json(snapshot_path, document.snapshot_summary())

        manifest = {
            "schema_version": "ai_edit_layered_bundle.v1",
            "document_id": document.id,
            "document_revision": document.revision,
            "canvas": document.snapshot_summary()["canvas"],
            "active_layer_id": document.active_layer_id,
            "active_selection_mask_id": document.active_selection_mask_id,
            "document_snapshot": "document_snapshot.json",
            "preview": preview_entry,
            "layers": layer_entries,
            "masks": mask_entries,
        }
        _write_json(root / "manifest.json", manifest)

        return ActionResult(
            action_id=action.id,
            status=ActionStatus.EXECUTED,
            output_assets={
                "path": str(root),
                "format": "ai_edit_layered_bundle.v1",
                "manifest": str(root / "manifest.json"),
                "document_snapshot": str(snapshot_path),
            },
            metadata={"layer_count": len(layer_entries), "mask_count": len(mask_entries)},
        )

    def _execute_no_op(self, document: DocumentState, action: Action) -> ActionResult:
        """Execute a no-op action."""
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, metadata={"no_op": True})

    def _execute_validate(self, document: DocumentState, action: Action) -> ActionResult:
        """Run document validation as an explicit non-mutating action."""
        report = self._validator().validate_document(document)
        status = ActionStatus.VALIDATED if not report.has_errors() else ActionStatus.FAILED
        error = None
        if report.has_errors():
            error = ActionError(code="validation.failed", message="document validation failed", action_id=action.id, details=report.to_json())
        return ActionResult(action_id=action.id, status=status, error=error, metadata={"validation": report.to_json()})

    def _merge_down(self, document: DocumentState, action: Action) -> ActionResult:
        """Merge the target layer into the layer immediately below it."""
        top_id = _required_target(action.target.layer_id, "target.layer_id")
        top_index = _layer_index(document, top_id)
        if top_index == 0:
            raise ValueError("merge down requires a layer below the target layer")
        lower = document.layers[top_index - 1]
        top = document.layers[top_index]
        _require_renderable_layer(lower)
        _require_renderable_layer(top)
        merged_pixels = _composite_layers_to_pixels(document, [lower, top])
        lower.pixels = merged_pixels
        lower.kind = LayerKind.RASTER
        lower.opacity = 1.0
        lower.visible = True
        lower.blend_mode = BlendMode.NORMAL
        lower.mask_id = None
        if "output_layer_name" in action.params:
            lower.name = action.params["output_layer_name"]
        document.layers.pop(top_index)
        _discard_group_references(document, {top.id})
        if document.active_layer_id == top.id:
            document.active_layer_id = lower.id
        return ActionResult(
            action_id=action.id,
            status=ActionStatus.EXECUTED,
            changed_layer_ids=[lower.id, top.id],
            metadata={"mode": "down", "removed_layer_ids": [top.id], "output_layer_id": lower.id},
        )

    def _merge_visible(self, document: DocumentState, action: Action) -> ActionResult:
        """Merge visible renderable layers into one output layer."""
        source_indices = [
            index
            for index, layer in enumerate(document.layers)
            if layer.visible and _is_renderable_layer(layer)
        ]
        if not source_indices:
            raise ValueError("merge visible requires at least one visible renderable layer")
        return self._replace_layers_with_merged(document, action, source_indices, "Merged Visible", mode="visible")

    def _merge_selected(self, document: DocumentState, action: Action) -> ActionResult:
        """Merge explicitly listed layers in current stack order."""
        layer_ids = action.params["layer_ids"]
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError("selected merge layer_ids must be unique")
        source_indices = sorted(_layer_index(document, layer_id) for layer_id in layer_ids)
        return self._replace_layers_with_merged(document, action, source_indices, "Merged Layers", mode="selected")

    def _flatten_image(self, document: DocumentState, action: Action) -> ActionResult:
        """Flatten visible layers into one layer and discard the previous stack."""
        source_layers = [layer for layer in document.layers if layer.visible and _is_renderable_layer(layer)]
        if not source_layers:
            raise ValueError("flatten requires at least one visible renderable layer")
        output_id = _required_target(action.target.output_layer_id, "target.output_layer_id")
        output_name = action.params.get("output_layer_name", "Flattened Image")
        merged_pixels = _composite_layers_to_pixels(document, source_layers, background=document.canvas.background_color_rgba)
        merged_pixels[..., 3] = 1.0
        old_layer_ids = [layer.id for layer in document.layers]
        document.layers = [
            Layer(
                id=output_id,
                name=output_name,
                kind=LayerKind.RASTER,
                pixels=merged_pixels,
                opacity=1.0,
                visible=True,
                blend_mode=BlendMode.NORMAL,
            )
        ]
        document.active_layer_id = output_id
        return ActionResult(
            action_id=action.id,
            status=ActionStatus.EXECUTED,
            created_layer_ids=[output_id],
            changed_layer_ids=old_layer_ids,
            metadata={"mode": "flatten", "removed_layer_ids": old_layer_ids, "output_layer_id": output_id},
        )

    def _replace_layers_with_merged(
        self,
        document: DocumentState,
        action: Action,
        source_indices: list[int],
        default_name: str,
        mode: str,
    ) -> ActionResult:
        """Composite source layers, remove them, and insert one merged layer."""
        output_id = _required_target(action.target.output_layer_id, "target.output_layer_id")
        source_layers = [document.layers[index] for index in source_indices]
        source_ids = [layer.id for layer in source_layers]
        if output_id in {layer.id for index, layer in enumerate(document.layers) if index not in source_indices}:
            raise ValueError(f"output layer id {output_id!r} already exists")
        for layer in source_layers:
            _require_renderable_layer(layer)
        merged_pixels = _composite_layers_to_pixels(document, source_layers)
        insert_index = min(source_indices)
        for index in sorted(source_indices, reverse=True):
            document.layers.pop(index)
        _discard_group_references(document, set(source_ids))
        output_layer = Layer(
            id=output_id,
            name=action.params.get("output_layer_name", default_name),
            kind=LayerKind.RASTER,
            pixels=merged_pixels,
            opacity=1.0,
            visible=True,
            blend_mode=BlendMode.NORMAL,
        )
        document.add_layer(output_layer, insert_index)
        document.active_layer_id = output_id
        return ActionResult(
            action_id=action.id,
            status=ActionStatus.EXECUTED,
            created_layer_ids=[output_id],
            changed_layer_ids=source_ids,
            metadata={"mode": mode, "removed_layer_ids": source_ids, "output_layer_id": output_id},
        )

    def _validator(self) -> Validator:
        """Return the configured validator or a default strict validator."""
        if self.context.validator is None:
            self.context.validator = Validator()
        return self.context.validator

    def _failed_result(
        self,
        action: Action,
        code: str,
        message: str,
        before_revision: int,
        details: Optional[dict[str, Any]] = None,
    ) -> ActionResult:
        """Create a failed action result."""
        return ActionResult(
            action_id=action.id,
            status=ActionStatus.FAILED,
            before_revision=before_revision,
            after_revision=before_revision,
            error=ActionError(
                code=code,
                message=message,
                action_id=action.id,
                details={} if details is None else details,
                recoverable=True,
            ),
        )

    def _unsupported(self, action: Action, document: DocumentState, name: str) -> ActionResult:
        """Return a structured unsupported-action failure."""
        return self._failed_result(
            action,
            "execution.unsupported_action",
            f"{name} is not implemented in the prototype executor",
            document.revision,
        )

    def _log_result(self, action: Action, result: ActionResult, document: DocumentState) -> None:
        """Log action results when a trace sink is configured."""
        if self.context.trace_sink is not None:
            self.context.trace_sink.log_action_result(action, result, document)


def _mutates_document(action: Action) -> bool:
    """Return whether a successful action should advance the document revision."""
    return ActionType(action.type) not in {
        ActionType.EXPORT_FLAT,
        ActionType.EXPORT_LAYERED_BUNDLE,
        ActionType.RASTERIZE_VECTOR_ASSET,
        ActionType.COPY,
        ActionType.NO_OP,
        ActionType.VALIDATE,
    }


def _required_target(value: Optional[str], field_name: str) -> str:
    """Return a required target ID or raise a clear error."""
    if value is None:
        raise ValueError(f"{field_name} is required")
    return value


def _resize_rgba_centered(
    pixels: np.ndarray,
    new_width: int,
    new_height: int,
    fill_color: tuple[float, float, float, float],
) -> np.ndarray:
    """Return a centered crop/pad copy of an RGBA pixel array."""
    if pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError("pixel arrays must have shape H x W x 4")
    output = np.zeros((new_height, new_width, 4), dtype=np.float32)
    output[..., :] = fill_color
    src_y, dst_y, copy_h = _centered_copy_axis(pixels.shape[0], new_height)
    src_x, dst_x, copy_w = _centered_copy_axis(pixels.shape[1], new_width)
    output[dst_y : dst_y + copy_h, dst_x : dst_x + copy_w, :] = pixels[src_y : src_y + copy_h, src_x : src_x + copy_w, :]
    return output


def _resize_mask_centered(data: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
    """Return a centered crop/pad copy of a mask array."""
    if data.ndim != 2:
        raise ValueError("mask arrays must be 2D")
    output = np.zeros((new_height, new_width), dtype=np.float32)
    src_y, dst_y, copy_h = _centered_copy_axis(data.shape[0], new_height)
    src_x, dst_x, copy_w = _centered_copy_axis(data.shape[1], new_width)
    output[dst_y : dst_y + copy_h, dst_x : dst_x + copy_w] = data[src_y : src_y + copy_h, src_x : src_x + copy_w]
    return output


def _centered_copy_axis(old_size: int, new_size: int) -> tuple[int, int, int]:
    """Return source start, destination start, and count for centered copy."""
    copy_size = min(old_size, new_size)
    source_start = max((old_size - new_size) // 2, 0)
    destination_start = max((new_size - old_size) // 2, 0)
    return source_start, destination_start, copy_size


def _load_rgba_image(path: Path) -> np.ndarray:
    """Load an image file as full-resolution straight-alpha RGBA floats."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("image import requires Pillow") from exc
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        return (np.asarray(rgba, dtype=np.float32) / 255.0).astype(np.float32)


def _require_pixel_layer(layer: Layer) -> None:
    """Require a layer with full-canvas RGBA pixel data."""
    if layer.pixels is None:
        raise ValueError(f"layer {layer.id!r} has no pixel data")
    if layer.pixels.ndim != 3 or layer.pixels.shape[2] != 4:
        raise ValueError(f"layer {layer.id!r} pixels must have shape H x W x 4")


def _content_bbox(pixels: np.ndarray, threshold: float = 0.0) -> Optional[tuple[int, int, int, int]]:
    """Return half-open bbox around alpha greater than threshold."""
    bbox = _region_content_bbox_rgba(pixels, threshold)
    return None if bbox is None else bbox.as_tuple()


def _anchor_point(value: Any, width: int, height: int) -> tuple[float, float]:
    """Return an absolute anchor point, defaulting to canvas center."""
    if value is None:
        return (width / 2.0, height / 2.0)
    if isinstance(value, str):
        lookup = {
            "center": (width / 2.0, height / 2.0),
            "top_left": (0.0, 0.0),
            "top_right": (float(width), 0.0),
            "bottom_left": (0.0, float(height)),
            "bottom_right": (float(width), float(height)),
        }
        if value not in lookup:
            raise ValueError(f"unsupported anchor {value!r}")
        return lookup[value]
    return _point_to_float_pair(value)


def _transform_matrix(params: dict[str, Any], operation: str, anchor: tuple[float, float]) -> np.ndarray:
    """Return a forward 3x3 affine matrix for layer transforms."""
    if operation == "affine" and "matrix" in params:
        a, b, c, d, e, f = [float(item) for item in params["matrix"]]
        return np.array([[a, c, e], [b, d, f], [0.0, 0.0, 1.0]], dtype=np.float64)
    ax, ay = anchor
    to_anchor = np.array([[1.0, 0.0, ax], [0.0, 1.0, ay], [0.0, 0.0, 1.0]], dtype=np.float64)
    from_anchor = np.array([[1.0, 0.0, -ax], [0.0, 1.0, -ay], [0.0, 0.0, 1.0]], dtype=np.float64)
    if operation == "translate":
        dx = float(params.get("dx", 0.0))
        dy = float(params.get("dy", 0.0))
        return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]], dtype=np.float64)
    if operation == "scale":
        sx = float(params.get("scale_x", 1.0))
        sy = float(params.get("scale_y", sx))
        local = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        return to_anchor @ local @ from_anchor
    if operation == "rotate":
        angle = np.deg2rad(float(params.get("angle_degrees", 0.0)))
        local = np.array(
            [[float(np.cos(angle)), -float(np.sin(angle)), 0.0], [float(np.sin(angle)), float(np.cos(angle)), 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return to_anchor @ local @ from_anchor
    if operation == "flip":
        sx = -1.0 if params.get("horizontal", True) else 1.0
        sy = -1.0 if params.get("vertical", False) else 1.0
        local = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        return to_anchor @ local @ from_anchor
    if operation == "affine":
        return np.eye(3, dtype=np.float64)
    raise ValueError(f"unsupported transform operation {operation!r}")


def _affine_transform_rgba(
    pixels: np.ndarray,
    matrix: np.ndarray,
    fill_color: tuple[float, float, float, float],
    resample: str,
) -> np.ndarray:
    """Apply a forward affine matrix to RGBA pixels using Pillow resampling."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("layer transforms require Pillow") from exc
    inverse = np.linalg.inv(matrix)
    coefficients = (inverse[0, 0], inverse[0, 1], inverse[0, 2], inverse[1, 0], inverse[1, 1], inverse[1, 2])
    resampling = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
    }[resample]
    image = Image.fromarray(np.clip(pixels * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGBA")
    fill = tuple(int(round(channel * 255.0)) for channel in fill_color)
    transformed = image.transform(image.size, Image.Transform.AFFINE, coefficients, resample=resampling, fillcolor=fill)
    return (np.asarray(transformed, dtype=np.float32) / 255.0).astype(np.float32)


def _polygon_mask(width: int, height: int, points: list[Any], closed: bool = True) -> np.ndarray:
    """Rasterize a polygon or freehand path mask."""
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("polygon selections require Pillow") from exc
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    coords = [tuple(_point_to_float_pair(point)) for point in points]
    if closed:
        draw.polygon(coords, fill=255)
    else:
        draw.line(coords, fill=255, width=1)
    return (np.asarray(image, dtype=np.float32) / 255.0).astype(np.float32)


def _path_stroke_mask(width: int, height: int, points: list[Any], stroke_width: float, closed: bool) -> np.ndarray:
    """Rasterize a stroked polyline into a soft-ish mask."""
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("path painting requires Pillow") from exc
    scale = 4 if max(width, height) <= 1024 else 2
    image = Image.new("L", (width * scale, height * scale), 0)
    draw = ImageDraw.Draw(image)
    coords = [(x * scale, y * scale) for x, y in (_point_to_float_pair(point) for point in points)]
    if closed and coords:
        coords = [*coords, coords[0]]
    draw.line(coords, fill=255, width=max(1, int(round(stroke_width * scale))), joint="curve")
    if scale != 1:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return (np.asarray(image, dtype=np.float32) / 255.0).astype(np.float32)


def _gradient_pixels(width: int, height: int, params: dict[str, Any]) -> np.ndarray:
    """Create a full-canvas RGBA gradient."""
    colors = np.asarray([_parse_color(color) for color in params["colors"]], dtype=np.float32)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    if params.get("type", "linear") == "radial":
        cx, cy = _point_to_float_pair(params["center"])
        radius = float(params["radius"])
        t = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(radius, 1e-6)
    else:
        sx, sy = _point_to_float_pair(params["start"])
        ex, ey = _point_to_float_pair(params["end"])
        vx = ex - sx
        vy = ey - sy
        denom = max(vx * vx + vy * vy, 1e-6)
        t = ((xx - sx) * vx + (yy - sy) * vy) / denom
    t = np.clip(t, 0.0, 1.0)
    if len(colors) == 2:
        return (colors[0] * (1.0 - t[..., np.newaxis]) + colors[1] * t[..., np.newaxis]).astype(np.float32)
    stops = np.linspace(0.0, 1.0, len(colors), dtype=np.float32)
    output = np.zeros((height, width, 4), dtype=np.float32)
    for channel in range(4):
        output[..., channel] = np.interp(t, stops, colors[:, channel])
    return output


def _pattern_pixels(width: int, height: int, params: dict[str, Any]) -> np.ndarray:
    """Create a simple repeating pattern image."""
    pattern = params.get("pattern", "checkerboard")
    if pattern == "image":
        tile = _load_rgba_image(Path(params["path"]))
        reps_y = int(np.ceil(height / tile.shape[0]))
        reps_x = int(np.ceil(width / tile.shape[1]))
        return np.tile(tile, (reps_y, reps_x, 1))[:height, :width, :].astype(np.float32)
    colors = [_parse_color(color) for color in params.get("colors", ["#000000", "#ffffff"])]
    yy, xx = np.mgrid[0:height, 0:width]
    cell = int(params.get("cell_size", 16))
    output = np.zeros((height, width, 4), dtype=np.float32)
    if pattern == "stripes":
        stripe = int(params.get("stripe_width", cell))
        index = ((xx // stripe) % len(colors)).astype(int)
    else:
        index = (((xx // cell) + (yy // cell)) % len(colors)).astype(int)
    for color_index, color in enumerate(colors):
        output[index == color_index] = color
    return output


def _apply_fill_mode(destination: np.ndarray, fill: np.ndarray, mode: str) -> np.ndarray:
    """Apply full-canvas fill pixels according to a simple paint mode."""
    proposed = np.array(destination, copy=True)
    if mode == "replace_rgb_preserve_alpha":
        proposed[..., :3] = fill[..., :3]
    elif mode == "source_over":
        proposed = _source_over_rgba(proposed, fill[..., :3], fill[..., 3:4])
    elif mode == "replace_rgba":
        proposed[..., :] = fill
    else:
        raise ValueError(f"unsupported fill mode {mode!r}")
    return np.clip(proposed, 0.0, 1.0).astype(np.float32)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    """Return sRGB-style luminance from RGB floats."""
    return (rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722).astype(np.float32)


def _disk_footprint(radius: int) -> np.ndarray:
    """Return a disk-shaped boolean footprint."""
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= radius * radius


def _remove_small_components(data: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components smaller than min_area."""
    _require_scipy("remove_small_components")
    labels, count = _ndimage.label(data > 0.0)
    output = np.zeros_like(data, dtype=np.float32)
    for label in range(1, count + 1):
        region = labels == label
        if int(np.count_nonzero(region)) >= min_area:
            output[region] = 1.0
    return output


def _translate_mask(data: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate a 2D mask by integer pixels."""
    output = np.zeros_like(data, dtype=np.float32)
    height, width = data.shape
    src_x0 = max(0, -dx)
    src_y0 = max(0, -dy)
    dst_x0 = max(0, dx)
    dst_y0 = max(0, dy)
    copy_w = min(width - src_x0, width - dst_x0)
    copy_h = min(height - src_y0, height - dst_y0)
    if copy_w > 0 and copy_h > 0:
        output[dst_y0 : dst_y0 + copy_h, dst_x0 : dst_x0 + copy_w] = data[src_y0 : src_y0 + copy_h, src_x0 : src_x0 + copy_w]
    return output


def _point_to_float_pair(point: Any) -> tuple[float, float]:
    """Validate a two-number point and return floats."""
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        raise TypeError("points must be two-number lists")
    if isinstance(point[0], bool) or isinstance(point[1], bool):
        raise TypeError("point entries must be numbers")
    return float(point[0]), float(point[1])


def _region_mask(document: DocumentState, params: dict[str, Any]) -> np.ndarray:
    """Return a full-canvas mask from params, active selection, or full canvas."""
    return _region_resolve_mask(
        document,
        mask_id=params.get("source_mask_id"),
        bbox=params.get("bbox_xyxy"),
        use_active_selection=True,
        default_full_canvas=True,
    )


def _copy_region_pixels(document: DocumentState, layer: Layer, params: dict[str, Any]) -> tuple[np.ndarray, list[int]]:
    """Copy a region from a layer into a cropped RGBA array."""
    mask = _region_mask(document, params)
    bbox = _region_bbox_from_mask(mask)
    if bbox is None:
        raise ValueError("cannot copy an empty region")
    pixels = _region_extract_rgba(layer.pixels, bbox)
    mask_crop = _region_extract_mask(mask, bbox)
    return _region_multiply_alpha_by_mask(pixels, mask_crop), bbox.as_list()


def _paste_pixels(destination: np.ndarray, source: np.ndarray, x: int, y: int) -> None:
    """Paste cropped source pixels into full-canvas destination at x/y."""
    destination[..., :] = _region_paste_crop(destination, source, x, y)


def _render_text_pixels(width: int, height: int, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Render text into a full-canvas RGBA layer and return text metadata."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("text rendering requires Pillow") from exc
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    font_size = int(params.get("font_size", 32))
    font_path = params.get("font_path")
    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    color = tuple(int(round(channel * 255.0)) for channel in _parse_color(params.get("color", "#000000")))
    draw.multiline_text(
        (int(params.get("x", 0)), int(params.get("y", 0))),
        params.get("text", ""),
        fill=color,
        font=font,
        anchor=params.get("anchor"),
        align=params.get("align", "left"),
        spacing=int(params.get("spacing", 0)),
    )
    metadata = {"text": dict(params), "rasterized": True}
    return (np.asarray(image, dtype=np.float32) / 255.0).astype(np.float32), metadata


def _shape_observation_from_mask(mask: np.ndarray) -> dict[str, Any]:
    """Return a coarse geometric observation for a binary mask."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return {"type": "empty", "bbox_xyxy": [0, 0, 0, 0], "area_pixels": 0}
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    width = x1 - x0
    height = y1 - y0
    area = int(xs.size)
    bbox_area = max(width * height, 1)
    shape_type = "ellipse" if area / bbox_area < 0.86 else "rectangle"
    return {"type": shape_type, "bbox_xyxy": [x0, y0, x1, y1], "area_pixels": area, "center": [float(xs.mean()), float(ys.mean())]}


def _connected_component_observations(mask: np.ndarray, min_area: int, max_objects: int) -> list[dict[str, Any]]:
    """Return connected component observations sorted by area descending."""
    labels, count = _ndimage.label(mask)
    observations = []
    for label in range(1, count + 1):
        region = labels == label
        area = int(np.count_nonzero(region))
        if area < min_area:
            continue
        observation = _shape_observation_from_mask(region)
        observation["id"] = f"object_{label:04d}"
        observations.append(observation)
    observations.sort(key=lambda item: int(item["area_pixels"]), reverse=True)
    return observations[:max_objects]


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorized RGB to HSV conversion for float arrays."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    v = maxc
    delta = maxc - minc
    s = np.zeros_like(maxc)
    np.divide(delta, maxc, out=s, where=maxc > 0.0)
    h = np.zeros_like(maxc)
    mask = delta > 0.0
    safe_delta = np.where(mask, delta, 1.0)
    h = np.where((maxc == r) & mask, ((g - b) / safe_delta) % 6.0, h)
    h = np.where((maxc == g) & mask, ((b - r) / safe_delta) + 2.0, h)
    h = np.where((maxc == b) & mask, ((r - g) / safe_delta) + 4.0, h)
    h = h / 6.0
    return np.stack([h, s, v], axis=-1).astype(np.float32)


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Vectorized HSV to RGB conversion for float arrays."""
    h = (hsv[..., 0] % 1.0) * 6.0
    s = np.clip(hsv[..., 1], 0.0, 1.0)
    v = np.clip(hsv[..., 2], 0.0, 1.0)
    i = np.floor(h).astype(int)
    f = h - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    choices = [
        np.stack([v, t, p], axis=-1),
        np.stack([q, v, p], axis=-1),
        np.stack([p, v, t], axis=-1),
        np.stack([p, q, v], axis=-1),
        np.stack([t, p, v], axis=-1),
        np.stack([v, p, q], axis=-1),
    ]
    output = np.zeros_like(choices[0])
    for index, choice in enumerate(choices):
        output[i % 6 == index] = choice[i % 6 == index]
    return output.astype(np.float32)


def _pixels_from_backend_response(response: dict[str, Any]) -> np.ndarray:
    """Read RGBA pixels from a diffusion backend response."""
    if not isinstance(response, dict):
        raise TypeError("diffusion backend response must be a dictionary")
    pixels = response.get("pixels")
    if isinstance(pixels, np.ndarray):
        if pixels.dtype != np.float32:
            pixels = pixels.astype(np.float32)
        if pixels.max(initial=0.0) > 1.0:
            pixels = pixels / 255.0
        if pixels.ndim == 3 and pixels.shape[2] == 4:
            return np.clip(pixels, 0.0, 1.0).astype(np.float32)
    path = response.get("path") or response.get("image_path")
    if path is not None:
        return _load_rgba_image(Path(path))
    raise ValueError("diffusion backend response must include RGBA pixels or an image path")


def _fit_pixels_to_canvas(pixels: np.ndarray, width: int, height: int) -> np.ndarray:
    """Center crop/pad pixels to the document canvas size."""
    if pixels.shape[:2] == (height, width):
        return pixels.astype(np.float32, copy=True)
    return _resize_rgba_centered(pixels, width, height, (0.0, 0.0, 0.0, 0.0))


def _rasterize_vector_asset(
    path: Path,
    width: Optional[int] = None,
    height: Optional[int] = None,
    background_color: Any = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rasterize a supported vector asset into straight-alpha RGBA floats."""
    suffix = path.suffix.lower()
    if suffix not in {".svg", ".svgz"}:
        raise ValueError("vector rasterization currently supports .svg and .svgz files")
    vector_bytes = _read_vector_bytes(path)

    rasterized = _rasterize_svg_with_cairosvg(vector_bytes, width, height)
    renderer = "cairosvg"
    if rasterized is None:
        rasterized = _rasterize_svg_builtin(vector_bytes, width, height)
        renderer = "builtin_svg_subset"

    if background_color is not None:
        rasterized = _composite_background(rasterized, _parse_color(background_color))

    image_height, image_width = rasterized.shape[:2]
    return rasterized, {"renderer": renderer, "source_path": str(path), "output_size": [image_width, image_height]}


def _read_vector_bytes(path: Path) -> bytes:
    """Read an SVG or compressed SVG file."""
    if not path.exists():
        raise FileNotFoundError(path)
    data = path.read_bytes()
    if path.suffix.lower() == ".svgz":
        return gzip.decompress(data)
    return data


def _rasterize_svg_with_cairosvg(vector_bytes: bytes, width: Optional[int], height: Optional[int]) -> Optional[np.ndarray]:
    """Rasterize SVG bytes through CairoSVG when the optional dependency exists."""
    try:
        import cairosvg  # type: ignore[import-not-found]
        from PIL import Image
    except ImportError:
        return None

    kwargs: dict[str, Any] = {"bytestring": vector_bytes}
    if width is not None:
        kwargs["output_width"] = int(width)
    if height is not None:
        kwargs["output_height"] = int(height)
    png_bytes = cairosvg.svg2png(**kwargs)
    with Image.open(BytesIO(png_bytes)) as image:
        rgba = image.convert("RGBA")
        return (np.asarray(rgba, dtype=np.float32) / 255.0).astype(np.float32)


def _rasterize_svg_builtin(vector_bytes: bytes, width: Optional[int], height: Optional[int]) -> np.ndarray:
    """Rasterize a conservative SVG subset without external dependencies."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("builtin SVG rasterization requires Pillow") from exc

    root = ET.fromstring(vector_bytes)
    if _local_name(root.tag) != "svg":
        raise ValueError("vector asset root must be an SVG document")
    output_width, output_height, view_box = _svg_output_geometry(root, width, height)
    antialias = _svg_antialias_scale(output_width, output_height)
    image = Image.new("RGBA", (output_width * antialias, output_height * antialias), (0, 0, 0, 0))

    view_x, view_y, view_width, view_height = view_box
    base_transform = (
        output_width * antialias / view_width,
        0.0,
        0.0,
        output_height * antialias / view_height,
        -view_x * output_width * antialias / view_width,
        -view_y * output_height * antialias / view_height,
    )
    _draw_svg_children(image, root, base_transform, _default_svg_style())
    if antialias != 1:
        image = image.resize((output_width, output_height), Image.Resampling.LANCZOS)
    return (np.asarray(image, dtype=np.float32) / 255.0).astype(np.float32)


def _svg_output_geometry(root: ET.Element, width: Optional[int], height: Optional[int]) -> tuple[int, int, tuple[float, float, float, float]]:
    """Return output size and viewBox for SVG rasterization."""
    view_box = _parse_view_box(root.get("viewBox"))
    intrinsic_width = _parse_svg_length(root.get("width"))
    intrinsic_height = _parse_svg_length(root.get("height"))
    if intrinsic_width is None and view_box is not None:
        intrinsic_width = view_box[2]
    if intrinsic_height is None and view_box is not None:
        intrinsic_height = view_box[3]
    if intrinsic_width is None or intrinsic_height is None:
        raise ValueError("SVG rasterization requires width/height or viewBox dimensions")

    if width is None and height is None:
        output_width = int(round(intrinsic_width))
        output_height = int(round(intrinsic_height))
    elif width is not None and height is None:
        output_width = int(width)
        output_height = int(round(output_width * intrinsic_height / intrinsic_width))
    elif height is not None and width is None:
        output_height = int(height)
        output_width = int(round(output_height * intrinsic_width / intrinsic_height))
    else:
        output_width = int(width)
        output_height = int(height)
    if output_width <= 0 or output_height <= 0:
        raise ValueError("rasterized SVG dimensions must be positive")
    if view_box is None:
        view_box = (0.0, 0.0, float(intrinsic_width), float(intrinsic_height))
    if view_box[2] <= 0.0 or view_box[3] <= 0.0:
        raise ValueError("SVG viewBox width and height must be positive")
    return output_width, output_height, view_box


def _svg_antialias_scale(width: int, height: int) -> int:
    """Return a bounded supersampling factor for the built-in SVG renderer."""
    max_dimension = max(width, height)
    if max_dimension <= 512:
        return 4
    if max_dimension <= 2048:
        return 2
    return 1


def _draw_svg_children(surface: Any, parent: ET.Element, transform: tuple[float, float, float, float, float, float], style: dict[str, Any]) -> None:
    """Render child SVG elements recursively."""
    for child in list(parent):
        _draw_svg_element(surface, child, transform, style)


def _draw_svg_element(surface: Any, element: ET.Element, parent_transform: tuple[float, float, float, float, float, float], parent_style: dict[str, Any]) -> None:
    """Render one supported SVG element."""
    tag = _local_name(element.tag)
    transform = _multiply_svg_transform(parent_transform, _parse_svg_transform(element.get("transform")))
    style = _svg_style(element, parent_style)

    if tag in {"defs", "title", "desc", "metadata", "style"}:
        return
    if tag in {"svg", "g", "symbol"}:
        _draw_svg_children(surface, element, transform, style)
        return
    if tag == "rect":
        _draw_svg_polygon(surface, _rect_points(element), transform, style, closed=True)
        return
    if tag == "circle":
        cx = _parse_svg_number(element.get("cx"), 0.0)
        cy = _parse_svg_number(element.get("cy"), 0.0)
        radius = _parse_svg_number(element.get("r"), 0.0)
        _draw_svg_polygon(surface, _ellipse_points(cx, cy, radius, radius), transform, style, closed=True)
        return
    if tag == "ellipse":
        cx = _parse_svg_number(element.get("cx"), 0.0)
        cy = _parse_svg_number(element.get("cy"), 0.0)
        rx = _parse_svg_number(element.get("rx"), 0.0)
        ry = _parse_svg_number(element.get("ry"), 0.0)
        _draw_svg_polygon(surface, _ellipse_points(cx, cy, rx, ry), transform, style, closed=True)
        return
    if tag == "line":
        points = [
            (_parse_svg_number(element.get("x1"), 0.0), _parse_svg_number(element.get("y1"), 0.0)),
            (_parse_svg_number(element.get("x2"), 0.0), _parse_svg_number(element.get("y2"), 0.0)),
        ]
        _draw_svg_polyline(surface, points, transform, style, closed=False)
        return
    if tag == "polygon":
        _draw_svg_polygon(surface, _parse_svg_points(element.get("points", "")), transform, style, closed=True)
        return
    if tag == "polyline":
        _draw_svg_polyline(surface, _parse_svg_points(element.get("points", "")), transform, style, closed=False)
        return
    if tag == "path":
        for subpath, closed in _parse_simple_svg_path(element.get("d", "")):
            if closed:
                _draw_svg_polygon(surface, subpath, transform, style, closed=True)
            else:
                _draw_svg_polyline(surface, subpath, transform, style, closed=False)
        return
    raise NotImplementedError(f"builtin SVG rasterizer does not support <{tag}> elements")


def _draw_svg_polygon(
    surface: Any,
    points: list[tuple[float, float]],
    transform: tuple[float, float, float, float, float, float],
    style: dict[str, Any],
    closed: bool,
) -> None:
    """Draw a filled and/or stroked polygon."""
    if len(points) < 2:
        return
    transformed = [_apply_svg_transform(transform, point) for point in points]
    fill = _svg_fill(style)
    stroke = _svg_stroke(style)
    stroke_width = _svg_stroke_width(style, transform)
    overlay = _svg_overlay(surface)
    draw = _svg_draw(overlay)
    if fill is not None and len(transformed) >= 3:
        draw.polygon(transformed, fill=fill)
    if stroke is not None and stroke_width > 0:
        line_points = [*transformed, transformed[0]] if closed else transformed
        draw.line(line_points, fill=stroke, width=stroke_width, joint="curve")
    surface.alpha_composite(overlay)


def _draw_svg_polyline(
    surface: Any,
    points: list[tuple[float, float]],
    transform: tuple[float, float, float, float, float, float],
    style: dict[str, Any],
    closed: bool,
) -> None:
    """Draw a stroked polyline."""
    if len(points) < 2:
        return
    transformed = [_apply_svg_transform(transform, point) for point in points]
    stroke = _svg_stroke(style)
    stroke_width = _svg_stroke_width(style, transform)
    if stroke is not None and stroke_width > 0:
        overlay = _svg_overlay(surface)
        draw = _svg_draw(overlay)
        line_points = [*transformed, transformed[0]] if closed else transformed
        draw.line(line_points, fill=stroke, width=stroke_width, joint="curve")
        surface.alpha_composite(overlay)


def _svg_overlay(surface: Any) -> Any:
    """Return a transparent drawing layer matching a PIL image."""
    from PIL import Image

    return Image.new("RGBA", surface.size, (0, 0, 0, 0))


def _svg_draw(surface: Any) -> Any:
    """Return an RGBA-aware PIL drawing context."""
    from PIL import ImageDraw

    return ImageDraw.Draw(surface)


def _rect_points(element: ET.Element) -> list[tuple[float, float]]:
    """Return rectangle corner points for a supported SVG rect."""
    x = _parse_svg_number(element.get("x"), 0.0)
    y = _parse_svg_number(element.get("y"), 0.0)
    width = _parse_svg_number(element.get("width"), 0.0)
    height = _parse_svg_number(element.get("height"), 0.0)
    return [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]


def _ellipse_points(cx: float, cy: float, rx: float, ry: float, steps: int = 96) -> list[tuple[float, float]]:
    """Approximate an ellipse with a polygon."""
    if rx <= 0.0 or ry <= 0.0:
        return []
    angles = np.linspace(0.0, np.pi * 2.0, steps, endpoint=False)
    return [(float(cx + np.cos(angle) * rx), float(cy + np.sin(angle) * ry)) for angle in angles]


def _parse_simple_svg_path(data: str) -> list[tuple[list[tuple[float, float]], bool]]:
    """Parse SVG paths containing M, L, H, V, and Z commands."""
    tokens = re.findall(r"[MmLlHhVvZz]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", data)
    subpaths: list[tuple[list[tuple[float, float]], bool]] = []
    current: list[tuple[float, float]] = []
    current_x = 0.0
    current_y = 0.0
    start_x = 0.0
    start_y = 0.0
    command: Optional[str] = None
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if re.fullmatch(r"[A-Za-z]", token):
            command = token
            index += 1
            if command in {"Z", "z"}:
                if current:
                    subpaths.append((current, True))
                    current = []
                current_x, current_y = start_x, start_y
                command = None
            continue
        if command is None:
            raise ValueError("SVG path data must start with a command")
        relative = command.islower()
        upper = command.upper()
        if upper in {"M", "L"}:
            x = _path_number(tokens, index)
            y = _path_number(tokens, index + 1)
            index += 2
            if relative:
                x += current_x
                y += current_y
            if upper == "M":
                if current:
                    subpaths.append((current, False))
                current = [(x, y)]
                start_x, start_y = x, y
                command = "l" if relative else "L"
            else:
                current.append((x, y))
            current_x, current_y = x, y
        elif upper == "H":
            x = _path_number(tokens, index)
            index += 1
            if relative:
                x += current_x
            current.append((x, current_y))
            current_x = x
        elif upper == "V":
            y = _path_number(tokens, index)
            index += 1
            if relative:
                y += current_y
            current.append((current_x, y))
            current_y = y
        else:
            raise NotImplementedError(f"builtin SVG rasterizer does not support path command {command!r}")
    if current:
        subpaths.append((current, False))
    return subpaths


def _path_number(tokens: list[str], index: int) -> float:
    """Return a path numeric token."""
    if index >= len(tokens) or re.fullmatch(r"[A-Za-z]", tokens[index]):
        raise ValueError("SVG path command is missing a numeric argument")
    return float(tokens[index])


def _parse_svg_points(points: str) -> list[tuple[float, float]]:
    """Parse an SVG point list."""
    numbers = [float(number) for number in re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", points)]
    if len(numbers) % 2 != 0:
        raise ValueError("SVG point list must contain x/y pairs")
    return [(numbers[index], numbers[index + 1]) for index in range(0, len(numbers), 2)]


def _default_svg_style() -> dict[str, Any]:
    """Return SVG default paint style."""
    return {
        "fill": "black",
        "stroke": "none",
        "stroke-width": "1",
        "opacity": "1",
        "fill-opacity": "1",
        "stroke-opacity": "1",
    }


def _svg_style(element: ET.Element, parent: dict[str, Any]) -> dict[str, Any]:
    """Return inherited SVG style values for an element."""
    style = dict(parent)
    css = element.get("style")
    if css:
        for part in css.split(";"):
            if ":" in part:
                key, value = part.split(":", 1)
                style[key.strip()] = value.strip()
    for key in ("fill", "stroke", "stroke-width", "opacity", "fill-opacity", "stroke-opacity"):
        if element.get(key) is not None:
            style[key] = element.get(key)
    return style


def _svg_fill(style: dict[str, Any]) -> Optional[tuple[int, int, int, int]]:
    """Return an RGBA fill color or None."""
    opacity = _parse_svg_number(style.get("opacity"), 1.0) * _parse_svg_number(style.get("fill-opacity"), 1.0)
    return _svg_color(style.get("fill", "black"), opacity)


def _svg_stroke(style: dict[str, Any]) -> Optional[tuple[int, int, int, int]]:
    """Return an RGBA stroke color or None."""
    opacity = _parse_svg_number(style.get("opacity"), 1.0) * _parse_svg_number(style.get("stroke-opacity"), 1.0)
    return _svg_color(style.get("stroke", "none"), opacity)


def _svg_stroke_width(style: dict[str, Any], transform: tuple[float, float, float, float, float, float]) -> int:
    """Return stroke width in rendered pixels."""
    width = _parse_svg_number(style.get("stroke-width"), 1.0)
    scale_x = float(np.hypot(transform[0], transform[1]))
    scale_y = float(np.hypot(transform[2], transform[3]))
    return max(1, int(round(width * (scale_x + scale_y) / 2.0)))


def _svg_color(value: Any, opacity: float) -> Optional[tuple[int, int, int, int]]:
    """Parse a small set of SVG color formats."""
    if value is None:
        return None
    color = str(value).strip().lower()
    if color in {"none", "transparent"}:
        return None
    named = {
        "black": "#000000",
        "white": "#ffffff",
        "red": "#ff0000",
        "green": "#008000",
        "blue": "#0000ff",
        "yellow": "#ffff00",
        "cyan": "#00ffff",
        "magenta": "#ff00ff",
        "purple": "#800080",
        "orange": "#ffa500",
        "gray": "#808080",
        "grey": "#808080",
    }
    if color in named:
        color = named[color]
    if color.startswith("#"):
        hex_value = color[1:]
        if len(hex_value) in {3, 4}:
            hex_value = "".join(char * 2 for char in hex_value)
        if len(hex_value) == 6:
            hex_value += "ff"
        if len(hex_value) != 8:
            raise ValueError(f"unsupported SVG color {value!r}")
        rgba = tuple(int(hex_value[index : index + 2], 16) for index in range(0, 8, 2))
        alpha = int(round(rgba[3] * np.clip(opacity, 0.0, 1.0)))
        return (rgba[0], rgba[1], rgba[2], alpha)
    match = re.fullmatch(r"rgba?\(([^)]+)\)", color)
    if match:
        parts = [part.strip() for part in match.group(1).split(",")]
        if len(parts) not in {3, 4}:
            raise ValueError(f"unsupported SVG color {value!r}")
        rgb = [_svg_color_channel(part) for part in parts[:3]]
        alpha = float(parts[3]) if len(parts) == 4 else 1.0
        alpha = int(round(255.0 * np.clip(alpha * opacity, 0.0, 1.0)))
        return (rgb[0], rgb[1], rgb[2], alpha)
    raise ValueError(f"unsupported SVG color {value!r}")


def _svg_color_channel(value: str) -> int:
    """Parse one SVG RGB channel."""
    stripped = value.strip()
    if stripped.endswith("%"):
        return int(round(np.clip(float(stripped[:-1]) / 100.0, 0.0, 1.0) * 255.0))
    return int(round(np.clip(float(stripped), 0.0, 255.0)))


def _parse_svg_transform(value: Optional[str]) -> tuple[float, float, float, float, float, float]:
    """Parse a small, useful subset of SVG transform syntax."""
    transform = _identity_svg_transform()
    if not value:
        return transform
    for name, raw_args in re.findall(r"([A-Za-z]+)\(([^)]*)\)", value):
        args = [float(number) for number in re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", raw_args)]
        lower_name = name.lower()
        if lower_name == "translate":
            tx = args[0] if args else 0.0
            ty = args[1] if len(args) > 1 else 0.0
            next_transform = (1.0, 0.0, 0.0, 1.0, tx, ty)
        elif lower_name == "scale":
            sx = args[0] if args else 1.0
            sy = args[1] if len(args) > 1 else sx
            next_transform = (sx, 0.0, 0.0, sy, 0.0, 0.0)
        elif lower_name == "rotate":
            angle = np.deg2rad(args[0] if args else 0.0)
            cos_theta = float(np.cos(angle))
            sin_theta = float(np.sin(angle))
            rotation = (cos_theta, sin_theta, -sin_theta, cos_theta, 0.0, 0.0)
            if len(args) >= 3:
                cx, cy = args[1], args[2]
                next_transform = _multiply_svg_transform(
                    _multiply_svg_transform((1.0, 0.0, 0.0, 1.0, cx, cy), rotation),
                    (1.0, 0.0, 0.0, 1.0, -cx, -cy),
                )
            else:
                next_transform = rotation
        elif lower_name == "matrix" and len(args) == 6:
            next_transform = (args[0], args[1], args[2], args[3], args[4], args[5])
        else:
            raise NotImplementedError(f"builtin SVG rasterizer does not support transform {name!r}")
        transform = _multiply_svg_transform(transform, next_transform)
    return transform


def _identity_svg_transform() -> tuple[float, float, float, float, float, float]:
    """Return the identity SVG transform."""
    return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def _multiply_svg_transform(
    first: tuple[float, float, float, float, float, float],
    second: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    """Return `first @ second` for SVG affine transforms."""
    a1, b1, c1, d1, e1, f1 = first
    a2, b2, c2, d2, e2, f2 = second
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def _apply_svg_transform(transform: tuple[float, float, float, float, float, float], point: tuple[float, float]) -> tuple[float, float]:
    """Apply an SVG affine transform to a point."""
    x, y = point
    a, b, c, d, e, f = transform
    return a * x + c * y + e, b * x + d * y + f


def _parse_view_box(value: Optional[str]) -> Optional[tuple[float, float, float, float]]:
    """Parse an SVG viewBox attribute."""
    if value is None:
        return None
    numbers = [float(number) for number in re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", value)]
    if len(numbers) != 4:
        raise ValueError("SVG viewBox must contain four numbers")
    return numbers[0], numbers[1], numbers[2], numbers[3]


def _parse_svg_length(value: Optional[str]) -> Optional[float]:
    """Parse a basic SVG length, accepting unitless values and px."""
    if value is None:
        return None
    stripped = value.strip()
    if stripped.endswith("%"):
        return None
    match = re.fullmatch(r"([-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?)(?:px)?", stripped)
    if not match:
        return None
    return float(match.group(1))


def _parse_svg_number(value: Any, default: float) -> float:
    """Parse a numeric SVG attribute."""
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    parsed = _parse_svg_length(str(value))
    if parsed is None:
        raise ValueError(f"unsupported SVG numeric value {value!r}")
    return parsed


def _local_name(tag: str) -> str:
    """Return an XML local name without namespace."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _composite_background(pixels: np.ndarray, background: tuple[float, float, float, float]) -> np.ndarray:
    """Composite RGBA pixels over a solid/transparent background."""
    base = np.zeros_like(pixels)
    base[..., :] = background
    return _source_over_rgba(base, pixels[..., :3], pixels[..., 3:4])


def _save_rgba_png(pixels: np.ndarray, path: Path) -> None:
    """Save an RGBA float array as a PNG."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("PNG export requires Pillow") from exc
    image = Image.fromarray(np.clip(pixels * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGBA")
    image.save(path)


def _save_mask_png(data: np.ndarray, path: Path) -> None:
    """Save a mask float array as an 8-bit grayscale PNG."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("mask PNG export requires Pillow") from exc
    image = Image.fromarray(np.clip(data * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
    image.save(path)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Write a JSON file with stable formatting."""
    path.write_text(json.dumps(_json_ready(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_ready(value: Any) -> Any:
    """Convert common NumPy values to JSON-compatible objects."""
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _safe_filename_segment(value: str) -> str:
    """Return a filesystem-safe filename segment for a layer or mask ID."""
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return safe or "item"


def _integer_coordinate(value: Any, field_name: str) -> int:
    """Validate an integer-valued coordinate."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number")
    if float(value) != int(value):
        raise ValueError(f"{field_name} must be an integer pixel coordinate")
    return int(value)


def _color_from_params(params: dict[str, Any], default: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Read a color from params, accepting hex or RGBA lists."""
    if "color_rgba" in params:
        return _parse_color(params["color_rgba"])
    if "color" in params:
        return _parse_color(params["color"])
    return default


def _require_scipy(operation: str) -> None:
    """Raise a clear error if a SciPy-backed operation is unavailable."""
    if _ndimage is None:
        raise RuntimeError(f"{operation} requires scipy.ndimage")


def _channels(value: Any) -> set[str]:
    """Return normalized RGBA channel names from schema-validated params."""
    aliases = {
        "rgb": {"r", "g", "b"},
        "alpha": {"a"},
        "rgba": {"r", "g", "b", "a"},
    }
    if isinstance(value, str):
        return set(aliases.get(value, {value}))
    return {str(item) for item in value}


def _layer_index(document: DocumentState, layer_id: str) -> int:
    """Return a layer index by ID."""
    for index, layer in enumerate(document.layers):
        if layer.id == layer_id:
            return index
    raise KeyError(f"layer id {layer_id!r} does not exist")


def _is_renderable_layer(layer: Layer) -> bool:
    """Return whether a layer can participate in raster compositing."""
    return LayerKind(layer.kind) is not LayerKind.GROUP and layer.pixels is not None


def _require_renderable_layer(layer: Layer) -> None:
    """Validate that a layer can be rendered by the prototype compositor."""
    if not _is_renderable_layer(layer):
        raise ValueError(f"layer {layer.id!r} has no renderable pixel data")
    if BlendMode(layer.blend_mode) is not BlendMode.NORMAL:
        raise NotImplementedError("prototype merge supports normal blend mode only")
    if not _is_identity_transform(layer):
        raise NotImplementedError("prototype merge does not render transformed layers")


def _composite_layers_to_pixels(
    document: DocumentState,
    layers: list[Layer],
    background: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> np.ndarray:
    """Composite layers into one straight-alpha full-canvas RGBA array."""
    output = np.zeros((document.canvas.height, document.canvas.width, 4), dtype=np.float32)
    output[..., :] = background
    for layer in layers:
        _require_renderable_layer(layer)
        if not layer.visible:
            continue
        layer_pixels = layer.pixels.astype(np.float32, copy=False)
        effective_alpha = layer_pixels[..., 3:4] * np.float32(layer.opacity)
        if layer.mask_id is not None:
            effective_alpha = effective_alpha * document.get_mask(layer.mask_id).data[..., np.newaxis]
        output = _source_over_rgba(output, layer_pixels[..., :3], effective_alpha)
    return output


def _source_over_rgba(destination: np.ndarray, source_rgb: np.ndarray, source_alpha: np.ndarray) -> np.ndarray:
    """Composite source RGB/effective alpha over a straight-alpha RGBA destination."""
    destination_rgb = destination[..., :3]
    destination_alpha = destination[..., 3:4]
    output_alpha = source_alpha + destination_alpha * (1.0 - source_alpha)
    output_premultiplied = (
        source_rgb * source_alpha
        + destination_rgb * destination_alpha * (1.0 - source_alpha)
    )
    output_rgb = np.zeros_like(destination_rgb)
    np.divide(output_premultiplied, output_alpha, out=output_rgb, where=output_alpha > 0.0)
    output = np.concatenate([output_rgb, output_alpha], axis=2)
    return np.clip(output, 0.0, 1.0).astype(np.float32)


def _is_identity_transform(layer: Layer) -> bool:
    """Return whether a layer transform is the default identity transform."""
    transform = layer.transform
    return (
        float(transform.x) == 0.0
        and float(transform.y) == 0.0
        and float(transform.scale_x) == 1.0
        and float(transform.scale_y) == 1.0
        and float(transform.rotation_degrees) == 0.0
    )


def _discard_group_references(document: DocumentState, removed_layer_ids: set[str]) -> None:
    """Remove stale group metadata references after batch layer replacement."""
    for layer in document.layers:
        layer.child_layer_ids = [layer_id for layer_id in layer.child_layer_ids if layer_id not in removed_layer_ids]
        if layer.parent_group_id in removed_layer_ids:
            layer.parent_group_id = None


def _parse_color(value: Any) -> tuple[float, float, float, float]:
    """Parse #RRGGBB, #RRGGBBAA, or a numeric RGBA sequence."""
    if isinstance(value, str):
        if len(value) not in {7, 9} or not value.startswith("#"):
            raise ValueError("colors must be #RRGGBB or #RRGGBBAA")
        red = int(value[1:3], 16) / 255.0
        green = int(value[3:5], 16) / 255.0
        blue = int(value[5:7], 16) / 255.0
        alpha = int(value[7:9], 16) / 255.0 if len(value) == 9 else 1.0
        return (red, green, blue, alpha)
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise TypeError("colors must be #RRGGBB, #RRGGBBAA, or a four-number RGBA sequence")
    rgba = tuple(float(channel) for channel in value)
    if any(channel < 0.0 or channel > 1.0 for channel in rgba):
        raise ValueError("RGBA color channels must be in [0, 1]")
    return rgba


def _bbox_to_ints(bbox: Any, width: int, height: int) -> tuple[int, int, int, int]:
    """Validate and convert half-open bbox coordinates to integers."""
    return _region_bbox_from_xyxy(bbox, width, height).as_tuple()


def _rect_mask(width: int, height: int, bbox: Any) -> np.ndarray:
    """Rasterize a hard rectangle mask."""
    return _region_rect_mask(width, height, bbox)


def _color_range_mask(
    pixels: np.ndarray,
    color: tuple[float, float, float, float],
    tolerance: float,
    alpha_min: float,
) -> np.ndarray:
    """Select pixels whose RGB values are within `tolerance` of `color`."""
    if pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError("pixel arrays must have shape H x W x 4")
    if tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative")
    if alpha_min < 0.0 or alpha_min > 1.0:
        raise ValueError("alpha_min must be in [0, 1]")
    target = np.asarray(color[:3], dtype=np.float32)
    distance = np.linalg.norm(pixels[..., :3] - target, axis=-1)
    data = (distance <= tolerance) & (pixels[..., 3] >= alpha_min)
    return data.astype(np.float32)


def _magic_wand_mask(
    pixels: np.ndarray,
    seed_points: list[Any],
    tolerance: float,
    alpha_min: float,
    diagonal: bool,
) -> np.ndarray:
    """Select contiguous regions close to the color under each seed point."""
    if pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError("pixel arrays must have shape H x W x 4")
    if tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative")
    if alpha_min < 0.0 or alpha_min > 1.0:
        raise ValueError("alpha_min must be in [0, 1]")
    height, width = pixels.shape[:2]
    selected = np.zeros((height, width), dtype=bool)
    neighbors = (
        [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        if diagonal
        else [(-1, 0), (1, 0), (0, -1), (0, 1)]
    )

    for point in seed_points:
        x, y = _point_to_ints(point, width, height)
        target = pixels[y, x, :3]
        distance = np.linalg.norm(pixels[..., :3] - target, axis=-1)
        candidate = (distance <= tolerance) & (pixels[..., 3] >= alpha_min)
        if not bool(candidate[y, x]) or bool(selected[y, x]):
            continue

        queue: deque[tuple[int, int]] = deque([(x, y)])
        selected[y, x] = True
        while queue:
            current_x, current_y = queue.popleft()
            for dx, dy in neighbors:
                next_x = current_x + dx
                next_y = current_y + dy
                if next_x < 0 or next_y < 0 or next_x >= width or next_y >= height:
                    continue
                if selected[next_y, next_x] or not candidate[next_y, next_x]:
                    continue
                selected[next_y, next_x] = True
                queue.append((next_x, next_y))

    return selected.astype(np.float32)


def _point_to_ints(point: Any, width: int, height: int) -> tuple[int, int]:
    """Validate and convert a seed point to integer canvas coordinates."""
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        raise TypeError("seed points must be two-number lists")
    x = _integer_coordinate(point[0], "seed_point.x")
    y = _integer_coordinate(point[1], "seed_point.y")
    if x < 0 or y < 0 or x >= width or y >= height:
        raise ValueError("seed points must be inside the canvas")
    return x, y


def _ellipse_mask(width: int, height: int, bbox: Any) -> np.ndarray:
    """Rasterize a hard ellipse mask inside bbox."""
    x0, y0, x1, y1 = _bbox_to_ints(bbox, width, height)
    data = np.zeros((height, width), dtype=np.float32)
    box_width = x1 - x0
    box_height = y1 - y0
    ys = np.arange(y0, y1, dtype=np.float32) + 0.5
    xs = np.arange(x0, x1, dtype=np.float32) + 0.5
    norm_x = ((xs - x0) / box_width - 0.5) * 2.0
    norm_y = ((ys - y0) / box_height - 0.5) * 2.0
    mask = norm_y[:, np.newaxis] ** 2 + norm_x[np.newaxis, :] ** 2 <= 1.0
    data[y0:y1, x0:x1] = mask.astype(np.float32)
    return data


def _shape_mask(width: int, height: int, shape: dict[str, Any]) -> np.ndarray:
    """Rasterize a supported shape into a hard mask."""
    if float(shape.get("corner_radius", 0.0)) != 0.0:
        raise NotImplementedError("rounded rectangle masks are not implemented yet")
    shape_type = shape["type"]
    if shape_type == "rectangle":
        return _rect_mask(width, height, shape["bbox_xyxy"])
    if shape_type == "ellipse":
        return _ellipse_mask(width, height, shape["bbox_xyxy"])
    raise ValueError(f"unsupported shape type {shape_type!r}")


def _stroke_mask(width: int, height: int, shape: dict[str, Any], stroke_width: Any) -> np.ndarray:
    """Rasterize the stroke band for a supported shape."""
    if isinstance(stroke_width, bool) or not isinstance(stroke_width, (int, float)):
        raise TypeError("stroke width must be a number")
    stroke_pixels = int(np.ceil(float(stroke_width)))
    if stroke_pixels <= 0:
        raise ValueError("stroke width must be greater than zero")

    outer = _shape_mask(width, height, shape)
    x0, y0, x1, y1 = _bbox_to_ints(shape["bbox_xyxy"], width, height)
    inner_bbox = [x0 + stroke_pixels, y0 + stroke_pixels, x1 - stroke_pixels, y1 - stroke_pixels]
    if inner_bbox[2] <= inner_bbox[0] or inner_bbox[3] <= inner_bbox[1]:
        return outer

    inner_shape = dict(shape)
    inner_shape["bbox_xyxy"] = inner_bbox
    inner = _shape_mask(width, height, inner_shape)
    return np.maximum(outer - inner, 0.0).astype(np.float32)


def _paint_rgba(pixels: np.ndarray, mask: np.ndarray, color: tuple[float, float, float, float]) -> None:
    """Source-over paint one RGBA color into pixels where mask is true."""
    if not bool(np.any(mask)):
        return
    source_rgb = np.asarray(color[:3], dtype=np.float32)
    source_alpha = np.float32(color[3])

    destination_rgb = pixels[mask, :3]
    destination_alpha = pixels[mask, 3:4]
    output_alpha = source_alpha + destination_alpha * (1.0 - source_alpha)
    output_premultiplied = source_rgb * source_alpha + destination_rgb * destination_alpha * (1.0 - source_alpha)

    output_rgb = np.zeros_like(destination_rgb)
    np.divide(output_premultiplied, output_alpha, out=output_rgb, where=output_alpha > 0.0)
    pixels[mask, :3] = output_rgb
    pixels[mask, 3] = output_alpha[:, 0]
    np.clip(pixels, 0.0, 1.0, out=pixels)
