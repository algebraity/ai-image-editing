"""Typed action schema for the AI Editing Kernel.

Actions are the contract between planners and the editing runtime. A planner
outputs structured action JSON, not arbitrary Python. The executor validates and
applies those actions to `DocumentState`, and trace logging records the same JSON
for replay and training data.

The canonical action envelope is:

```
{
  "id": "action_001",
  "type": "draw_shape",
  "target": {"layer_id": "layer_border"},
  "write_mask_id": "mask_full_canvas",
  "params": {...},
  "preconditions": {...},
  "expected_result": {...},
  "description": "optional human-readable note",
  "created_by": "planner",
  "metadata": {}
}
```

Prototype geometry uses `bbox_xyxy`: four half-open pixel coordinates
`[x0, y0, x1, y1]`, where `x1` and `y1` are excluded just like Python slices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


JsonObject = dict[str, Any]
SCHEMA_VERSION = "ai_edit_actions.v1"


class ActionType(str, Enum):
    """Action vocabulary understood by the kernel schema."""

    # Document actions
    NEW_DOCUMENT = "new_document"
    IMPORT_IMAGE_AS_LAYER = "import_image_as_layer"
    IMPORT_VECTOR_AS_RASTER = "import_vector_as_raster"
    RASTERIZE_VECTOR_ASSET = "rasterize_vector_asset"
    EXPORT_FLAT = "export_flat"
    EXPORT_LAYERED_BUNDLE = "export_layered_bundle"
    RESIZE_CANVAS = "resize_canvas"
    CROP = "crop"

    # Layer actions
    CREATE_LAYER = "create_layer"
    DELETE_LAYER = "delete_layer"
    DUPLICATE_LAYER = "duplicate_layer"
    RENAME_LAYER = "rename_layer"
    REORDER_LAYER = "reorder_layer"
    SET_ACTIVE_LAYER = "set_active_layer"
    SET_LAYER_VISIBILITY = "set_layer_visibility"
    SET_LAYER_OPACITY = "set_layer_opacity"
    SET_BLEND_MODE = "set_blend_mode"
    MERGE_LAYERS = "merge_layers"

    # Selection and mask actions
    SELECT_RECT = "select_rect"
    SELECT_ELLIPSE = "select_ellipse"
    SELECT_POLYGON = "select_polygon"
    SELECT_FROM_ALPHA = "select_from_alpha"
    SELECT_COLOR_RANGE = "select_color_range"
    MAGIC_WAND_SELECT = "magic_wand_select"
    SAVE_SELECTION_AS_MASK = "save_selection_as_mask"
    CREATE_MASK_FROM_SHAPE = "create_mask_from_shape"
    GROW_MASK = "grow_mask"
    SHRINK_MASK = "shrink_mask"
    FEATHER_MASK = "feather_mask"
    INVERT_MASK = "invert_mask"
    COMBINE_MASKS = "combine_masks"

    # Drawing and pixel actions
    DRAW_SHAPE = "draw_shape"
    DRAW_PATH = "draw_path"
    BRUSH_STROKE = "brush_stroke"
    PAINT_BUCKET_FILL = "paint_bucket_fill"
    GRADIENT_FILL = "gradient_fill"
    BLUR_REGION = "blur_region"
    CLEAR_REGION = "clear_region"
    CUT = "cut"
    COPY = "copy"
    PASTE = "paste"
    TRANSFORM_LAYER = "transform_layer"
    ALIGN_LAYER = "align_layer"

    # Perception actions
    DETECT_SHAPE = "detect_shape"
    DETECT_OBJECTS = "detect_objects"
    EXTRACT_LINE_ART = "extract_line_art"
    DECOMPOSE_TO_LAYERS = "decompose_to_layers"

    # Diffusion actions
    TXT2IMG_TO_LAYER = "txt2img_to_layer"
    IMG2IMG_TO_LAYER = "img2img_to_layer"
    INPAINT_REGION = "inpaint_region"
    OUTPAINT_REGION = "outpaint_region"

    # Validation/meta actions
    VALIDATE = "validate"
    NO_OP = "no_op"


class ActionStatus(str, Enum):
    """Lifecycle state of an action result."""

    PLANNED = "planned"
    VALIDATED = "validated"
    EXECUTED = "executed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    SKIPPED = "skipped"


@dataclass(slots=True)
class ActionTarget:
    """Where an action operates or writes its output.

    `layer_id` and `mask_id` are stable ID references. Names may appear in params
    for display, but executor internals should use IDs.
    """

    document_id: Optional[str] = None
    layer_id: Optional[str] = None
    layer_name: Optional[str] = None
    mask_id: Optional[str] = None
    selection_id: Optional[str] = None
    output_layer_id: Optional[str] = None

    def to_json(self) -> JsonObject:
        return _drop_none(
            {
                "document_id": self.document_id,
                "layer_id": self.layer_id,
                "layer_name": self.layer_name,
                "mask_id": self.mask_id,
                "selection_id": self.selection_id,
                "output_layer_id": self.output_layer_id,
            }
        )

    @classmethod
    def from_json(cls, data: Optional[JsonObject]) -> "ActionTarget":
        if data is None:
            return cls()
        _require_mapping(data, "target")
        _reject_unknown_keys(
            data,
            "target",
            {"document_id", "layer_id", "layer_name", "mask_id", "selection_id", "output_layer_id"},
        )
        return cls(**{key: _optional_string(data.get(key), f"target.{key}") for key in data})


@dataclass(slots=True)
class ActionPreconditions:
    """Document conditions that must hold before execution."""

    required_layer_ids: list[str] = field(default_factory=list)
    required_mask_ids: list[str] = field(default_factory=list)
    require_active_layer: bool = False
    require_active_selection: bool = False
    require_unlocked_target_layer: bool = True
    require_write_mask: bool = True
    allow_hidden_layers: bool = False
    custom: JsonObject = field(default_factory=dict)

    def to_json(self) -> JsonObject:
        return {
            "required_layer_ids": list(self.required_layer_ids),
            "required_mask_ids": list(self.required_mask_ids),
            "require_active_layer": self.require_active_layer,
            "require_active_selection": self.require_active_selection,
            "require_unlocked_target_layer": self.require_unlocked_target_layer,
            "require_write_mask": self.require_write_mask,
            "allow_hidden_layers": self.allow_hidden_layers,
            "custom": _json_safe(self.custom),
        }

    @classmethod
    def from_json(cls, data: Optional[JsonObject]) -> "ActionPreconditions":
        if data is None:
            return cls()
        _require_mapping(data, "preconditions")
        _reject_unknown_keys(
            data,
            "preconditions",
            {
                "required_layer_ids",
                "required_mask_ids",
                "require_active_layer",
                "require_active_selection",
                "require_unlocked_target_layer",
                "require_write_mask",
                "allow_hidden_layers",
                "custom",
            },
        )
        return cls(
            required_layer_ids=_string_list(data.get("required_layer_ids", []), "preconditions.required_layer_ids"),
            required_mask_ids=_string_list(data.get("required_mask_ids", []), "preconditions.required_mask_ids"),
            require_active_layer=_bool_value(data.get("require_active_layer", False), "preconditions.require_active_layer"),
            require_active_selection=_bool_value(
                data.get("require_active_selection", False),
                "preconditions.require_active_selection",
            ),
            require_unlocked_target_layer=_bool_value(
                data.get("require_unlocked_target_layer", True),
                "preconditions.require_unlocked_target_layer",
            ),
            require_write_mask=_bool_value(data.get("require_write_mask", True), "preconditions.require_write_mask"),
            allow_hidden_layers=_bool_value(data.get("allow_hidden_layers", False), "preconditions.allow_hidden_layers"),
            custom=_mapping_value(data.get("custom", {}), "preconditions.custom"),
        )


@dataclass(slots=True)
class ExpectedResult:
    """What an action is expected to change or create."""

    changed_layer_ids: list[str] = field(default_factory=list)
    created_layer_names: list[str] = field(default_factory=list)
    created_mask_names: list[str] = field(default_factory=list)
    protected_mask_id: Optional[str] = None
    geometry_expectations: JsonObject = field(default_factory=dict)
    visual_expectations: JsonObject = field(default_factory=dict)
    custom: JsonObject = field(default_factory=dict)

    def to_json(self) -> JsonObject:
        return {
            "changed_layer_ids": list(self.changed_layer_ids),
            "created_layer_names": list(self.created_layer_names),
            "created_mask_names": list(self.created_mask_names),
            "protected_mask_id": self.protected_mask_id,
            "geometry_expectations": _json_safe(self.geometry_expectations),
            "visual_expectations": _json_safe(self.visual_expectations),
            "custom": _json_safe(self.custom),
        }

    @classmethod
    def from_json(cls, data: Optional[JsonObject]) -> "ExpectedResult":
        if data is None:
            return cls()
        _require_mapping(data, "expected_result")
        _reject_unknown_keys(
            data,
            "expected_result",
            {
                "changed_layer_ids",
                "created_layer_names",
                "created_mask_names",
                "protected_mask_id",
                "geometry_expectations",
                "visual_expectations",
                "custom",
            },
        )
        return cls(
            changed_layer_ids=_string_list(data.get("changed_layer_ids", []), "expected_result.changed_layer_ids"),
            created_layer_names=_string_list(data.get("created_layer_names", []), "expected_result.created_layer_names"),
            created_mask_names=_string_list(data.get("created_mask_names", []), "expected_result.created_mask_names"),
            protected_mask_id=_optional_string(data.get("protected_mask_id"), "expected_result.protected_mask_id"),
            geometry_expectations=_mapping_value(
                data.get("geometry_expectations", {}),
                "expected_result.geometry_expectations",
            ),
            visual_expectations=_mapping_value(
                data.get("visual_expectations", {}),
                "expected_result.visual_expectations",
            ),
            custom=_mapping_value(data.get("custom", {}), "expected_result.custom"),
        )


@dataclass(slots=True)
class Action:
    """A single structured operation requested by a planner or user."""

    id: str
    type: ActionType
    params: JsonObject = field(default_factory=dict)
    target: ActionTarget = field(default_factory=ActionTarget)
    preconditions: ActionPreconditions = field(default_factory=ActionPreconditions)
    expected_result: ExpectedResult = field(default_factory=ExpectedResult)
    write_mask_id: Optional[str] = None
    description: Optional[str] = None
    created_by: str = "planner"
    metadata: JsonObject = field(default_factory=dict)

    def validate_schema(self) -> None:
        """Validate the action envelope and action-specific params."""
        _validate_identifier(self.id, "action.id")
        self.type = ActionType(self.type)
        if not isinstance(self.params, dict):
            raise TypeError("action.params must be an object")
        if not isinstance(self.target, ActionTarget):
            raise TypeError("action.target must be an ActionTarget")
        if not isinstance(self.preconditions, ActionPreconditions):
            raise TypeError("action.preconditions must be an ActionPreconditions")
        if not isinstance(self.expected_result, ExpectedResult):
            raise TypeError("action.expected_result must be an ExpectedResult")
        if self.write_mask_id is not None:
            _validate_identifier(self.write_mask_id, "action.write_mask_id")
        if self.description is not None and not isinstance(self.description, str):
            raise TypeError("action.description must be a string or None")
        _validate_identifier(self.created_by, "action.created_by")
        if not isinstance(self.metadata, dict):
            raise TypeError("action.metadata must be an object")

        _validate_id_list(self.preconditions.required_layer_ids, "preconditions.required_layer_ids")
        _validate_id_list(self.preconditions.required_mask_ids, "preconditions.required_mask_ids")
        for field_name in (
            "require_active_layer",
            "require_active_selection",
            "require_unlocked_target_layer",
            "require_write_mask",
            "allow_hidden_layers",
        ):
            if not isinstance(getattr(self.preconditions, field_name), bool):
                raise TypeError(f"preconditions.{field_name} must be a bool")

        if self.requires_pixel_write() and self.preconditions.require_write_mask and self.write_mask_id is None:
            raise ValueError("pixel-writing actions must include write_mask_id")

        validator = _PARAM_VALIDATORS.get(self.type)
        if validator is None:
            raise ValueError(f"action type {self.type.value!r} is not part of the prototype action set")
        validator(self)

    def requires_pixel_write(self) -> bool:
        """Return whether this action can modify existing pixel or alpha data."""
        return self.type in {
            ActionType.DRAW_SHAPE,
            ActionType.PAINT_BUCKET_FILL,
            ActionType.BLUR_REGION,
            ActionType.CLEAR_REGION,
        }

    def to_json(self) -> JsonObject:
        """Serialize the action to canonical JSON-compatible form."""
        return _drop_none(
            {
                "id": self.id,
                "type": ActionType(self.type).value,
                "target": self.target.to_json(),
                "write_mask_id": self.write_mask_id,
                "params": _json_safe(self.params),
                "preconditions": self.preconditions.to_json(),
                "expected_result": self.expected_result.to_json(),
                "description": self.description,
                "created_by": self.created_by,
                "metadata": _json_safe(self.metadata),
            }
        )

    @classmethod
    def from_json(cls, data: JsonObject) -> "Action":
        """Deserialize an `Action` from canonical action JSON."""
        _require_mapping(data, "action")
        _reject_unknown_keys(
            data,
            "action",
            {
                "id",
                "type",
                "target",
                "write_mask_id",
                "params",
                "preconditions",
                "expected_result",
                "description",
                "created_by",
                "metadata",
            },
        )
        action = cls(
            id=_required_string(data, "id", "action.id"),
            type=ActionType(_required_string(data, "type", "action.type")),
            target=ActionTarget.from_json(data.get("target")),
            write_mask_id=_optional_string(data.get("write_mask_id"), "action.write_mask_id"),
            params=_mapping_value(data.get("params", {}), "action.params"),
            preconditions=ActionPreconditions.from_json(data.get("preconditions")),
            expected_result=ExpectedResult.from_json(data.get("expected_result")),
            description=_optional_string(data.get("description"), "action.description"),
            created_by=_required_string(data, "created_by", "action.created_by") if "created_by" in data else "planner",
            metadata=_mapping_value(data.get("metadata", {}), "action.metadata"),
        )
        action.validate_schema()
        return action


@dataclass(slots=True)
class ActionBatch:
    """A sequence of actions planned as one logical editing operation."""

    id: str
    actions: list[Action]
    user_prompt: Optional[str] = None
    description: Optional[str] = None
    stop_on_error: bool = True
    metadata: JsonObject = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def validate_schema(self) -> None:
        """Validate all actions and batch-level constraints."""
        _validate_identifier(self.id, "batch.id")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported action batch schema_version {self.schema_version!r}")
        if not isinstance(self.actions, list):
            raise TypeError("batch.actions must be a list")
        if self.user_prompt is not None and not isinstance(self.user_prompt, str):
            raise TypeError("batch.user_prompt must be a string or None")
        if self.description is not None and not isinstance(self.description, str):
            raise TypeError("batch.description must be a string or None")
        if not isinstance(self.stop_on_error, bool):
            raise TypeError("batch.stop_on_error must be a bool")
        if not isinstance(self.metadata, dict):
            raise TypeError("batch.metadata must be an object")

        seen_ids: set[str] = set()
        for index, action in enumerate(self.actions):
            if not isinstance(action, Action):
                raise TypeError(f"batch.actions[{index}] must be an Action")
            action.validate_schema()
            if action.id in seen_ids:
                raise ValueError(f"duplicate action id {action.id!r}")
            seen_ids.add(action.id)

    def to_json(self) -> JsonObject:
        """Serialize the batch to JSON-compatible form."""
        return _drop_none(
            {
                "schema_version": self.schema_version,
                "id": self.id,
                "user_prompt": self.user_prompt,
                "description": self.description,
                "stop_on_error": self.stop_on_error,
                "actions": [action.to_json() for action in self.actions],
                "metadata": _json_safe(self.metadata),
            }
        )

    @classmethod
    def from_json(cls, data: JsonObject) -> "ActionBatch":
        """Deserialize a batch from canonical batch JSON."""
        _require_mapping(data, "batch")
        _reject_unknown_keys(
            data,
            "batch",
            {"schema_version", "id", "user_prompt", "description", "stop_on_error", "actions", "metadata"},
        )
        actions_data = data.get("actions")
        if not isinstance(actions_data, list):
            raise TypeError("batch.actions must be a list")
        batch = cls(
            id=_required_string(data, "id", "batch.id"),
            actions=[Action.from_json(item) for item in actions_data],
            user_prompt=_optional_string(data.get("user_prompt"), "batch.user_prompt"),
            description=_optional_string(data.get("description"), "batch.description"),
            stop_on_error=_bool_value(data.get("stop_on_error", True), "batch.stop_on_error"),
            metadata=_mapping_value(data.get("metadata", {}), "batch.metadata"),
            schema_version=_required_string(data, "schema_version", "batch.schema_version")
            if "schema_version" in data
            else SCHEMA_VERSION,
        )
        batch.validate_schema()
        return batch


@dataclass(slots=True)
class ActionError:
    """Structured error returned when an action cannot be executed."""

    code: str
    message: str
    action_id: Optional[str] = None
    details: JsonObject = field(default_factory=dict)
    recoverable: bool = False

    def to_json(self) -> JsonObject:
        return _drop_none(
            {
                "code": self.code,
                "message": self.message,
                "action_id": self.action_id,
                "details": _json_safe(self.details),
                "recoverable": self.recoverable,
            }
        )

    @classmethod
    def from_json(cls, data: JsonObject) -> "ActionError":
        _require_mapping(data, "error")
        return cls(
            code=_required_string(data, "code", "error.code"),
            message=_required_string(data, "message", "error.message"),
            action_id=_optional_string(data.get("action_id"), "error.action_id"),
            details=_mapping_value(data.get("details", {}), "error.details"),
            recoverable=_bool_value(data.get("recoverable", False), "error.recoverable"),
        )


@dataclass(slots=True)
class ActionResult:
    """Result of attempting to execute one action."""

    action_id: str
    status: ActionStatus
    document_id: Optional[str] = None
    before_revision: Optional[int] = None
    after_revision: Optional[int] = None
    created_layer_ids: list[str] = field(default_factory=list)
    created_mask_ids: list[str] = field(default_factory=list)
    changed_layer_ids: list[str] = field(default_factory=list)
    output_assets: JsonObject = field(default_factory=dict)
    error: Optional[ActionError] = None
    metadata: JsonObject = field(default_factory=dict)

    def succeeded(self) -> bool:
        """Return True if the action executed successfully."""
        return self.status in {ActionStatus.EXECUTED, ActionStatus.VALIDATED}

    def to_json(self) -> JsonObject:
        """Serialize result to JSON-compatible form for traces."""
        return _drop_none(
            {
                "action_id": self.action_id,
                "status": ActionStatus(self.status).value,
                "document_id": self.document_id,
                "before_revision": self.before_revision,
                "after_revision": self.after_revision,
                "created_layer_ids": list(self.created_layer_ids),
                "created_mask_ids": list(self.created_mask_ids),
                "changed_layer_ids": list(self.changed_layer_ids),
                "output_assets": _json_safe(self.output_assets),
                "error": self.error.to_json() if self.error else None,
                "metadata": _json_safe(self.metadata),
            }
        )


@dataclass(slots=True)
class CreateLayerParams:
    name: str
    kind: str = "raster"
    width: Optional[int] = None
    height: Optional[int] = None
    opacity: float = 1.0
    blend_mode: str = "normal"
    insert_index: Optional[int] = None
    set_active: bool = True


@dataclass(slots=True)
class DrawShapeParams:
    shape_type: str
    bbox_xyxy: tuple[float, float, float, float]
    stroke_color_rgba: Optional[tuple[float, float, float, float]] = None
    stroke_width: float = 1.0
    fill_color_rgba: Optional[tuple[float, float, float, float]] = None
    corner_radius: float = 0.0


@dataclass(slots=True)
class ClearRegionParams:
    mode: str = "alpha_to_zero"
    preserve_rgb: bool = False


@dataclass(slots=True)
class InpaintRegionParams:
    prompt: str
    negative_prompt: Optional[str] = None
    source_context: str = "flattened_visible"
    denoise: float = 0.65
    seed: Optional[int] = None
    backend: Optional[str] = None
    output_layer_name: Optional[str] = None


def _validate_create_layer(action: Action) -> None:
    _reject_unknown_keys(
        action.params,
        "params",
        {"name", "kind", "width", "height", "opacity", "blend_mode", "insert_index", "set_active", "color", "color_rgba"},
    )
    _require_target_id(action.target.output_layer_id, "target.output_layer_id")
    _required_string(action.params, "name", "params.name")
    _optional_enum_string(action.params.get("kind", "raster"), {"raster", "group", "vector", "text", "adjustment"}, "params.kind")
    _optional_positive_int(action.params.get("width"), "params.width")
    _optional_positive_int(action.params.get("height"), "params.height")
    _optional_unit_number(action.params.get("opacity", 1.0), "params.opacity")
    _optional_enum_string(
        action.params.get("blend_mode", "normal"),
        {"normal", "multiply", "screen", "overlay", "add", "subtract"},
        "params.blend_mode",
    )
    _optional_nonnegative_int(action.params.get("insert_index"), "params.insert_index")
    _bool_value(action.params.get("set_active", True), "params.set_active")
    if "color" in action.params:
        _validate_color(action.params["color"], "params.color")
    if "color_rgba" in action.params:
        _rgba_sequence(action.params["color_rgba"], "params.color_rgba")


def _validate_resize_canvas(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", {"width", "height", "anchor", "fill_color"})
    _positive_int(action.params.get("width"), "params.width")
    _positive_int(action.params.get("height"), "params.height")
    anchor = action.params.get("anchor", "center")
    if anchor != "center":
        raise ValueError("params.anchor must be 'center'")
    if "fill_color" in action.params:
        _validate_color(action.params["fill_color"], "params.fill_color")


def _validate_crop(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", {"bbox_xyxy", "scope", "fill_color"})
    _bbox_xyxy(action.params.get("bbox_xyxy"), "params.bbox_xyxy")
    scope = action.params.get("scope", "document")
    if scope not in {"document", "layer", "mask"}:
        raise ValueError("params.scope must be 'document', 'layer', or 'mask'")
    if scope == "layer":
        _require_target_id(action.target.layer_id, "target.layer_id")
    if scope == "mask":
        _require_target_id(action.target.mask_id, "target.mask_id")
    if "fill_color" in action.params:
        _validate_color(action.params["fill_color"], "params.fill_color")


def _validate_import_image_as_layer(action: Action) -> None:
    _reject_unknown_keys(
        action.params,
        "params",
        {"path", "name", "x", "y", "opacity", "blend_mode", "set_active"},
    )
    _require_target_id(action.target.output_layer_id, "target.output_layer_id")
    _required_string(action.params, "path", "params.path")
    if "name" in action.params:
        _optional_string(action.params["name"], "params.name")
    _integer_number(action.params.get("x", 0), "params.x")
    _integer_number(action.params.get("y", 0), "params.y")
    _optional_unit_number(action.params.get("opacity", 1.0), "params.opacity")
    _optional_enum_string(
        action.params.get("blend_mode", "normal"),
        {"normal", "multiply", "screen", "overlay", "add", "subtract"},
        "params.blend_mode",
    )
    _bool_value(action.params.get("set_active", True), "params.set_active")


def _validate_import_vector_as_raster(action: Action) -> None:
    _reject_unknown_keys(
        action.params,
        "params",
        {"path", "name", "x", "y", "width", "height", "opacity", "blend_mode", "set_active", "background_color"},
    )
    _require_target_id(action.target.output_layer_id, "target.output_layer_id")
    _required_string(action.params, "path", "params.path")
    if "name" in action.params:
        _optional_string(action.params["name"], "params.name")
    _integer_number(action.params.get("x", 0), "params.x")
    _integer_number(action.params.get("y", 0), "params.y")
    _optional_positive_int(action.params.get("width"), "params.width")
    _optional_positive_int(action.params.get("height"), "params.height")
    _optional_unit_number(action.params.get("opacity", 1.0), "params.opacity")
    _optional_enum_string(
        action.params.get("blend_mode", "normal"),
        {"normal", "multiply", "screen", "overlay", "add", "subtract"},
        "params.blend_mode",
    )
    _bool_value(action.params.get("set_active", True), "params.set_active")
    if action.params.get("background_color") is not None:
        _validate_color(action.params["background_color"], "params.background_color")


def _validate_rasterize_vector_asset(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", {"path", "output_path", "width", "height", "background_color"})
    _required_string(action.params, "path", "params.path")
    _required_string(action.params, "output_path", "params.output_path")
    _optional_positive_int(action.params.get("width"), "params.width")
    _optional_positive_int(action.params.get("height"), "params.height")
    if action.params.get("background_color") is not None:
        _validate_color(action.params["background_color"], "params.background_color")


def _validate_set_active_layer(action: Action) -> None:
    _require_target_id(action.target.layer_id, "target.layer_id")


def _validate_delete_layer(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", set())
    _require_target_id(action.target.layer_id, "target.layer_id")


def _validate_duplicate_layer(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", {"name", "insert_index", "set_active"})
    _require_target_id(action.target.layer_id, "target.layer_id")
    _require_target_id(action.target.output_layer_id, "target.output_layer_id")
    if "name" in action.params:
        _optional_string(action.params["name"], "params.name")
    _optional_nonnegative_int(action.params.get("insert_index"), "params.insert_index")
    _bool_value(action.params.get("set_active", True), "params.set_active")


def _validate_rename_layer(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", {"name"})
    _require_target_id(action.target.layer_id, "target.layer_id")
    _required_string(action.params, "name", "params.name")


def _validate_reorder_layer(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", {"index"})
    _require_target_id(action.target.layer_id, "target.layer_id")
    _nonnegative_int(action.params.get("index"), "params.index")


def _validate_set_layer_visibility(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", {"visible"})
    _require_target_id(action.target.layer_id, "target.layer_id")
    _bool_value(action.params.get("visible"), "params.visible")


def _validate_set_layer_opacity(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", {"opacity"})
    _require_target_id(action.target.layer_id, "target.layer_id")
    _optional_unit_number(action.params.get("opacity"), "params.opacity")


def _validate_set_blend_mode(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", {"blend_mode"})
    _require_target_id(action.target.layer_id, "target.layer_id")
    _optional_enum_string(
        action.params.get("blend_mode"),
        {"normal", "multiply", "screen", "overlay", "add", "subtract"},
        "params.blend_mode",
    )


def _validate_merge_layers(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", {"mode", "layer_ids", "output_layer_name"})
    mode = action.params.get("mode", "down")
    if mode not in {"down", "visible", "selected", "flatten"}:
        raise ValueError("params.mode must be 'down', 'visible', 'selected', or 'flatten'")
    if mode == "down":
        _require_target_id(action.target.layer_id, "target.layer_id")
    if mode in {"visible", "selected", "flatten"}:
        _require_target_id(action.target.output_layer_id, "target.output_layer_id")
    if mode == "selected":
        layer_ids = _string_list(action.params.get("layer_ids"), "params.layer_ids")
        if len(layer_ids) < 2:
            raise ValueError("selected merge requires at least two layer_ids")
    elif "layer_ids" in action.params:
        raise ValueError("params.layer_ids is only valid for selected merge")
    if "output_layer_name" in action.params:
        _optional_string(action.params["output_layer_name"], "params.output_layer_name")


def _validate_select_rect(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", {"name", "bbox_xyxy", "set_active"})
    _require_target_id(action.target.mask_id, "target.mask_id")
    _bbox_xyxy(action.params.get("bbox_xyxy"), "params.bbox_xyxy")
    if "name" in action.params:
        _optional_string(action.params["name"], "params.name")
    _bool_value(action.params.get("set_active", True), "params.set_active")


def _validate_select_ellipse(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", {"name", "bbox_xyxy", "set_active"})
    _require_target_id(action.target.mask_id, "target.mask_id")
    _bbox_xyxy(action.params.get("bbox_xyxy"), "params.bbox_xyxy")
    if "name" in action.params:
        _optional_string(action.params["name"], "params.name")
    _bool_value(action.params.get("set_active", True), "params.set_active")


def _validate_select_color_range(action: Action) -> None:
    _reject_unknown_keys(
        action.params,
        "params",
        {"name", "color", "tolerance", "bbox_xyxy", "alpha_min", "kind", "set_active"},
    )
    _require_target_id(action.target.layer_id, "target.layer_id")
    _require_target_id(action.target.mask_id, "target.mask_id")
    _validate_color(action.params.get("color"), "params.color")
    _nonnegative_number(action.params.get("tolerance"), "params.tolerance")
    if "bbox_xyxy" in action.params:
        _bbox_xyxy(action.params["bbox_xyxy"], "params.bbox_xyxy")
    _optional_unit_number(action.params.get("alpha_min", 0.0), "params.alpha_min")
    if "name" in action.params:
        _optional_string(action.params["name"], "params.name")
    if "kind" in action.params:
        _optional_enum_string(
            action.params["kind"],
            {"selection", "write_guard", "layer_alpha", "object", "shape", "line_art_region", "diffusion"},
            "params.kind",
        )
    _bool_value(action.params.get("set_active", True), "params.set_active")


def _validate_magic_wand_select(action: Action) -> None:
    _reject_unknown_keys(
        action.params,
        "params",
        {"name", "seed_points", "tolerance", "alpha_min", "diagonal", "kind", "set_active"},
    )
    _require_target_id(action.target.layer_id, "target.layer_id")
    _require_target_id(action.target.mask_id, "target.mask_id")
    _point_list(action.params.get("seed_points"), "params.seed_points")
    _nonnegative_number(action.params.get("tolerance"), "params.tolerance")
    _optional_unit_number(action.params.get("alpha_min", 0.0), "params.alpha_min")
    _bool_value(action.params.get("diagonal", False), "params.diagonal")
    if "name" in action.params:
        _optional_string(action.params["name"], "params.name")
    if "kind" in action.params:
        _optional_enum_string(
            action.params["kind"],
            {"selection", "write_guard", "layer_alpha", "object", "shape", "line_art_region", "diffusion"},
            "params.kind",
        )
    _bool_value(action.params.get("set_active", True), "params.set_active")


def _validate_create_mask_from_shape(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", {"name", "kind", "shape", "set_active"})
    _require_target_id(action.target.mask_id, "target.mask_id")
    shape = _shape_object(action.params.get("shape"), "params.shape")
    _shape_type(shape, {"rectangle", "ellipse"})
    if "name" in action.params:
        _optional_string(action.params["name"], "params.name")
    if "kind" in action.params:
        _optional_enum_string(
            action.params["kind"],
            {"selection", "write_guard", "layer_alpha", "object", "shape", "line_art_region", "diffusion"},
            "params.kind",
        )
    _bool_value(action.params.get("set_active", False), "params.set_active")


def _validate_grow_mask(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", {"source_mask_id", "pixels", "name", "set_active"})
    _require_target_id(action.target.mask_id, "target.mask_id")
    _required_string(action.params, "source_mask_id", "params.source_mask_id")
    _nonnegative_int(action.params.get("pixels"), "params.pixels")
    if "name" in action.params:
        _optional_string(action.params["name"], "params.name")
    _bool_value(action.params.get("set_active", False), "params.set_active")


def _validate_shrink_mask(action: Action) -> None:
    _validate_grow_mask(action)


def _validate_invert_mask(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", {"source_mask_id", "name", "set_active"})
    _require_target_id(action.target.mask_id, "target.mask_id")
    _required_string(action.params, "source_mask_id", "params.source_mask_id")
    if "name" in action.params:
        _optional_string(action.params["name"], "params.name")
    _bool_value(action.params.get("set_active", False), "params.set_active")


def _validate_combine_masks(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", {"operation", "mask_ids", "name"})
    _require_target_id(action.target.mask_id, "target.mask_id")
    operation = _required_string(action.params, "operation", "params.operation")
    if operation not in {"union", "intersect", "subtract"}:
        raise ValueError("params.operation must be 'union', 'intersect', or 'subtract'")
    mask_ids = _string_list(action.params.get("mask_ids"), "params.mask_ids")
    if operation == "subtract" and len(mask_ids) != 2:
        raise ValueError("subtract combine_masks requires exactly two mask_ids")
    if operation in {"union", "intersect"} and len(mask_ids) < 2:
        raise ValueError("union and intersect combine_masks require at least two mask_ids")
    if "name" in action.params:
        _optional_string(action.params["name"], "params.name")


def _validate_feather_mask(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", {"source_mask_id", "radius", "name"})
    _require_target_id(action.target.mask_id, "target.mask_id")
    _required_string(action.params, "source_mask_id", "params.source_mask_id")
    _nonnegative_number(action.params.get("radius"), "params.radius")
    if "name" in action.params:
        _optional_string(action.params["name"], "params.name")


def _validate_draw_shape(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", {"shape", "stroke", "fill"})
    _require_target_id(action.target.layer_id, "target.layer_id")
    shape = _shape_object(action.params.get("shape"), "params.shape")
    _shape_type(shape, {"rectangle", "ellipse"})
    stroke = action.params.get("stroke")
    fill = action.params.get("fill")
    if stroke is None and fill is None:
        raise ValueError("draw_shape requires at least one of params.stroke or params.fill")
    if stroke is not None:
        _require_mapping(stroke, "params.stroke")
        _reject_unknown_keys(stroke, "params.stroke", {"color", "width"})
        _validate_color(stroke.get("color"), "params.stroke.color")
        _positive_number(stroke.get("width"), "params.stroke.width")
    if fill is not None:
        _require_mapping(fill, "params.fill")
        _reject_unknown_keys(fill, "params.fill", {"color"})
        _validate_color(fill.get("color"), "params.fill.color")


def _validate_paint_bucket_fill(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", {"color", "mode"})
    _require_target_id(action.target.layer_id, "target.layer_id")
    _validate_color(action.params.get("color"), "params.color")
    mode = action.params.get("mode", "replace_rgb_preserve_alpha")
    if mode not in {"replace_rgb_preserve_alpha", "replace_rgba", "source_over"}:
        raise ValueError("params.mode must be 'replace_rgb_preserve_alpha', 'replace_rgba', or 'source_over'")


def _validate_blur_region(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", {"radius", "channels", "edge_mode"})
    _require_target_id(action.target.layer_id, "target.layer_id")
    _nonnegative_number(action.params.get("radius"), "params.radius")
    _validate_channels(action.params.get("channels", "rgb"), "params.channels")
    edge_mode = action.params.get("edge_mode", "nearest")
    if edge_mode not in {"reflect", "constant", "nearest", "mirror", "wrap"}:
        raise ValueError("params.edge_mode must be one of 'reflect', 'constant', 'nearest', 'mirror', or 'wrap'")


def _validate_clear_region(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", {"mode", "preserve_rgb"})
    _require_target_id(action.target.layer_id, "target.layer_id")
    mode = action.params.get("mode", "alpha_to_zero")
    if mode not in {"alpha_to_zero", "rgba_to_zero"}:
        raise ValueError("params.mode must be 'alpha_to_zero' or 'rgba_to_zero'")
    _bool_value(action.params.get("preserve_rgb", False), "params.preserve_rgb")


def _validate_export_flat(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", {"path"})
    _required_string(action.params, "path", "params.path")


def _validate_no_op(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", set())


_PARAM_VALIDATORS = {
    ActionType.RESIZE_CANVAS: _validate_resize_canvas,
    ActionType.CROP: _validate_crop,
    ActionType.IMPORT_IMAGE_AS_LAYER: _validate_import_image_as_layer,
    ActionType.IMPORT_VECTOR_AS_RASTER: _validate_import_vector_as_raster,
    ActionType.RASTERIZE_VECTOR_ASSET: _validate_rasterize_vector_asset,
    ActionType.CREATE_LAYER: _validate_create_layer,
    ActionType.DELETE_LAYER: _validate_delete_layer,
    ActionType.DUPLICATE_LAYER: _validate_duplicate_layer,
    ActionType.RENAME_LAYER: _validate_rename_layer,
    ActionType.REORDER_LAYER: _validate_reorder_layer,
    ActionType.SET_ACTIVE_LAYER: _validate_set_active_layer,
    ActionType.SET_LAYER_VISIBILITY: _validate_set_layer_visibility,
    ActionType.SET_LAYER_OPACITY: _validate_set_layer_opacity,
    ActionType.SET_BLEND_MODE: _validate_set_blend_mode,
    ActionType.MERGE_LAYERS: _validate_merge_layers,
    ActionType.SELECT_RECT: _validate_select_rect,
    ActionType.SELECT_ELLIPSE: _validate_select_ellipse,
    ActionType.SELECT_COLOR_RANGE: _validate_select_color_range,
    ActionType.MAGIC_WAND_SELECT: _validate_magic_wand_select,
    ActionType.CREATE_MASK_FROM_SHAPE: _validate_create_mask_from_shape,
    ActionType.GROW_MASK: _validate_grow_mask,
    ActionType.SHRINK_MASK: _validate_shrink_mask,
    ActionType.INVERT_MASK: _validate_invert_mask,
    ActionType.COMBINE_MASKS: _validate_combine_masks,
    ActionType.FEATHER_MASK: _validate_feather_mask,
    ActionType.DRAW_SHAPE: _validate_draw_shape,
    ActionType.PAINT_BUCKET_FILL: _validate_paint_bucket_fill,
    ActionType.BLUR_REGION: _validate_blur_region,
    ActionType.CLEAR_REGION: _validate_clear_region,
    ActionType.EXPORT_FLAT: _validate_export_flat,
    ActionType.NO_OP: _validate_no_op,
}


def _shape_object(value: Any, field_name: str) -> JsonObject:
    shape = _mapping_value(value, field_name)
    _reject_unknown_keys(shape, field_name, {"type", "bbox_xyxy", "corner_radius"})
    _required_string(shape, "type", f"{field_name}.type")
    _bbox_xyxy(shape.get("bbox_xyxy"), f"{field_name}.bbox_xyxy")
    if "corner_radius" in shape:
        _nonnegative_number(shape["corner_radius"], f"{field_name}.corner_radius")
    return shape


def _shape_type(shape: JsonObject, allowed: set[str]) -> None:
    shape_type = shape["type"]
    if shape_type not in allowed:
        raise ValueError(f"shape.type must be one of {sorted(allowed)!r}")


def _bbox_xyxy(value: Any, field_name: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise TypeError(f"{field_name} must be a four-number list")
    coords = tuple(_number(item, f"{field_name} entry") for item in value)
    if coords[2] <= coords[0] or coords[3] <= coords[1]:
        raise ValueError(f"{field_name} must satisfy x1 > x0 and y1 > y0")
    return coords


def _point_list(value: Any, field_name: str) -> list[tuple[float, float]]:
    if not isinstance(value, list) or len(value) == 0:
        raise TypeError(f"{field_name} must be a non-empty list")
    points = []
    for index, item in enumerate(value):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise TypeError(f"{field_name}[{index}] must be a two-number list")
        points.append(
            (
                _number(item[0], f"{field_name}[{index}].x"),
                _number(item[1], f"{field_name}[{index}].y"),
            )
        )
    return points


def _validate_color(value: Any, field_name: str) -> None:
    if isinstance(value, str):
        if len(value) not in {7, 9} or not value.startswith("#"):
            raise ValueError(f"{field_name} must be #RRGGBB or #RRGGBBAA")
        try:
            int(value[1:], 16)
        except ValueError as exc:
            raise ValueError(f"{field_name} must contain hexadecimal digits") from exc
        return
    _rgba_sequence(value, field_name)


def _validate_channels(value: Any, field_name: str) -> set[str]:
    allowed = {"r", "g", "b", "a"}
    aliases = {
        "rgb": {"r", "g", "b"},
        "alpha": {"a"},
        "rgba": {"r", "g", "b", "a"},
    }
    if isinstance(value, str):
        if value in aliases:
            return aliases[value]
        if value in allowed:
            return {value}
        raise ValueError(f"{field_name} must be 'rgb', 'alpha', 'rgba', or a list of channels")
    if not isinstance(value, list) or len(value) == 0:
        raise TypeError(f"{field_name} must be a string or a non-empty list")
    channels: set[str] = set()
    for item in value:
        if item not in allowed:
            raise ValueError(f"{field_name} entries must be one of {sorted(allowed)!r}")
        channels.add(item)
    return channels


def _rgba_sequence(value: Any, field_name: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise TypeError(f"{field_name} must be #RRGGBB, #RRGGBBAA, or a four-number RGBA list")
    rgba = tuple(_number(item, f"{field_name} entry") for item in value)
    if any(channel < 0.0 or channel > 1.0 for channel in rgba):
        raise ValueError(f"{field_name} RGBA channels must be in [0, 1]")
    return rgba


def _drop_none(data: JsonObject) -> JsonObject:
    return {key: value for key, value in data.items() if value is not None}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _validate_identifier(value: Any, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if value == "":
        raise ValueError(f"{field_name} must not be empty")


def _require_target_id(value: Any, field_name: str) -> None:
    if value is None:
        raise ValueError(f"{field_name} is required")
    _validate_identifier(value, field_name)


def _require_mapping(value: Any, field_name: str) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be an object")


def _mapping_value(value: Any, field_name: str) -> JsonObject:
    _require_mapping(value, field_name)
    return dict(value)


def _reject_unknown_keys(data: JsonObject, field_name: str, allowed: set[str]) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"{field_name} contains unknown keys {unknown!r}")


def _required_string(data: JsonObject, key: str, field_name: str) -> str:
    if key not in data:
        raise ValueError(f"{field_name} is required")
    value = data[key]
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if value == "":
        raise ValueError(f"{field_name} must not be empty")
    return value


def _optional_string(value: Any, field_name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or None")
    if value == "":
        raise ValueError(f"{field_name} must not be empty")
    return value


def _bool_value(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool")
    return value


def _string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    for item in value:
        _validate_identifier(item, f"{field_name} entry")
    return list(value)


def _validate_id_list(value: Any, field_name: str) -> None:
    _string_list(value, field_name)


def _optional_enum_string(value: Any, allowed: set[str], field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if value not in allowed:
        raise ValueError(f"{field_name} must be one of {sorted(allowed)!r}")


def _optional_positive_int(value: Any, field_name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer or None")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero")


def _optional_nonnegative_int(value: Any, field_name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer or None")
    if value < 0:
        raise ValueError(f"{field_name} must not be negative")


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero")
    return value


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must not be negative")
    return value


def _number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number")
    return float(value)


def _integer_number(value: Any, field_name: str) -> int:
    number = _number(value, field_name)
    if number != int(number):
        raise ValueError(f"{field_name} must be an integer")
    return int(number)


def _positive_number(value: Any, field_name: str) -> float:
    number = _number(value, field_name)
    if number <= 0.0:
        raise ValueError(f"{field_name} must be greater than zero")
    return number


def _nonnegative_number(value: Any, field_name: str) -> float:
    number = _number(value, field_name)
    if number < 0.0:
        raise ValueError(f"{field_name} must not be negative")
    return number


def _optional_unit_number(value: Any, field_name: str) -> None:
    number = _number(value, field_name)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]")
