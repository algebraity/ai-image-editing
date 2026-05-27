"""Action executor for the AI Editing Kernel.

The executor is the only runtime component that should mutate `DocumentState`.
Planners produce `Action` objects; validators check them; the executor applies
them; trace sinks record what happened.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np

from ai_edit_kernel.document.document_state import DocumentState
from ai_edit_kernel.document.layer import BlendMode, Layer, LayerKind
from ai_edit_kernel.document.mask import Mask, MaskKind
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
            ActionType.IMPORT_IMAGE_AS_LAYER: self._execute_import_image_as_layer,
            ActionType.CREATE_LAYER: self._execute_create_layer,
            ActionType.SET_ACTIVE_LAYER: self._execute_set_active_layer,
            ActionType.SELECT_RECT: self._execute_select_rect,
            ActionType.SELECT_COLOR_RANGE: self._execute_select_color_range,
            ActionType.MAGIC_WAND_SELECT: self._execute_magic_wand_select,
            ActionType.CREATE_MASK_FROM_SHAPE: self._execute_create_mask_from_shape,
            ActionType.COMBINE_MASKS: self._execute_combine_masks,
            ActionType.FEATHER_MASK: self._execute_feather_mask,
            ActionType.DRAW_SHAPE: self._execute_draw_shape,
            ActionType.PAINT_BUCKET_FILL: self._execute_paint_bucket_fill,
            ActionType.CLEAR_REGION: self._execute_clear_region,
            ActionType.EXPORT_FLAT: self._execute_export_flat,
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
        if not isinstance(before_pixels, np.ndarray) or not isinstance(proposed_pixels, np.ndarray):
            raise TypeError("before_pixels and proposed_pixels must be NumPy arrays")
        if before_pixels.shape != proposed_pixels.shape:
            raise ValueError("before_pixels and proposed_pixels must have the same shape")
        if before_pixels.ndim != 3 or before_pixels.shape[2] != 4:
            raise ValueError("pixel arrays must have shape H x W x 4")

        mask = document.get_mask(write_mask_id)
        if mask.data.shape != before_pixels.shape[:2]:
            raise ValueError("write mask shape must match pixel array dimensions")

        alpha = mask.data[..., np.newaxis].astype(np.float32, copy=False)
        blended = before_pixels * (1.0 - alpha) + proposed_pixels * alpha
        return np.clip(blended, 0.0, 1.0).astype(np.float32)

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

    def _execute_delete_layer(self, document: DocumentState, action: Action) -> ActionResult:
        """Remove a layer after lock and dependency checks."""
        return self._unsupported(action, document, "delete_layer")

    def _execute_duplicate_layer(self, document: DocumentState, action: Action) -> ActionResult:
        """Create a deep copy of a layer and insert it into the stack."""
        return self._unsupported(action, document, "duplicate_layer")

    def _execute_rename_layer(self, document: DocumentState, action: Action) -> ActionResult:
        """Rename a layer without changing its ID or pixels."""
        return self._unsupported(action, document, "rename_layer")

    def _execute_reorder_layer(self, document: DocumentState, action: Action) -> ActionResult:
        """Move a layer to a different stack index."""
        return self._unsupported(action, document, "reorder_layer")

    def _execute_set_active_layer(self, document: DocumentState, action: Action) -> ActionResult:
        """Set the document's active layer."""
        layer_id = _required_target(action.target.layer_id, "target.layer_id")
        document.set_active_layer(layer_id)
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, changed_layer_ids=[layer_id])

    def _execute_select_rect(self, document: DocumentState, action: Action) -> ActionResult:
        """Create a rectangular selection mask and optionally make it active."""
        mask_id = _required_target(action.target.mask_id, "target.mask_id")
        data = _rect_mask(document.canvas.width, document.canvas.height, action.params["bbox_xyxy"])
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
            x0, y0, x1, y1 = _bbox_to_ints(action.params["bbox_xyxy"], document.canvas.width, document.canvas.height)
            constrained[y0:y1, x0:x1] = data[y0:y1, x0:x1]
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
        return self._unsupported(action, document, "select_ellipse")

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


    def _execute_transform_layer(self, document: DocumentState, action: Action) -> ActionResult:
        """Move, scale, rotate, or align a layer according to action params."""
        return self._unsupported(action, document, "transform_layer")

    def _execute_inpaint_region(self, document: DocumentState, action: Action) -> ActionResult:
        """Call the diffusion backend for a masked region and composite the result."""
        return self._unsupported(action, document, "inpaint_region")

    def _execute_img2img_to_layer(self, document: DocumentState, action: Action) -> ActionResult:
        """Call an image-to-image backend and place the result on a target layer."""
        return self._unsupported(action, document, "img2img_to_layer")

    def _execute_txt2img_to_layer(self, document: DocumentState, action: Action) -> ActionResult:
        """Call a text-to-image backend and import the result as a new layer."""
        return self._unsupported(action, document, "txt2img_to_layer")

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

    def _execute_no_op(self, document: DocumentState, action: Action) -> ActionResult:
        """Execute a no-op action."""
        return ActionResult(action_id=action.id, status=ActionStatus.EXECUTED, metadata={"no_op": True})

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
    return ActionType(action.type) not in {ActionType.EXPORT_FLAT, ActionType.NO_OP, ActionType.VALIDATE}


def _required_target(value: Optional[str], field_name: str) -> str:
    """Return a required target ID or raise a clear error."""
    if value is None:
        raise ValueError(f"{field_name} is required")
    return value


def _load_rgba_image(path: Path) -> np.ndarray:
    """Load an image file as full-resolution straight-alpha RGBA floats."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("image import requires Pillow") from exc
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        return (np.asarray(rgba, dtype=np.float32) / 255.0).astype(np.float32)


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
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise TypeError("bbox_xyxy must be a four-number list")
    coords = []
    for value in bbox:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("bbox_xyxy entries must be numbers")
        if float(value) != int(value):
            raise ValueError("prototype bbox_xyxy entries must be integer pixel coordinates")
        coords.append(int(value))
    x0, y0, x1, y1 = coords
    if x1 <= x0 or y1 <= y0:
        raise ValueError("bbox_xyxy must satisfy x1 > x0 and y1 > y0")
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        raise ValueError("bbox_xyxy must be inside the canvas")
    return x0, y0, x1, y1


def _rect_mask(width: int, height: int, bbox: Any) -> np.ndarray:
    """Rasterize a hard rectangle mask."""
    x0, y0, x1, y1 = _bbox_to_ints(bbox, width, height)
    data = np.zeros((height, width), dtype=np.float32)
    data[y0:y1, x0:x1] = 1.0
    return data


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
