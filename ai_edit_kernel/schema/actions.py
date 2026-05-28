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
    MOVE_LAYER = "move_layer"
    SCALE_LAYER = "scale_layer"
    ROTATE_LAYER = "rotate_layer"
    FLIP_LAYER = "flip_layer"
    ADD_LAYER_MASK = "add_layer_mask"
    APPLY_LAYER_MASK = "apply_layer_mask"
    REMOVE_LAYER_MASK = "remove_layer_mask"

    # Selection and mask actions
    SELECT_RECT = "select_rect"
    SELECT_ELLIPSE = "select_ellipse"
    SELECT_POLYGON = "select_polygon"
    SELECT_FREEHAND = "select_freehand"
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
    REFINE_SELECTION = "refine_selection"
    REMOVE_SMALL_ISLANDS = "remove_small_islands"
    FILL_MASK_HOLES = "fill_mask_holes"

    # Drawing and pixel actions
    DRAW_SHAPE = "draw_shape"
    DRAW_PATH = "draw_path"
    BRUSH_STROKE = "brush_stroke"
    ERASE_STROKE = "erase_stroke"
    PAINT_BUCKET_FILL = "paint_bucket_fill"
    GRADIENT_FILL = "gradient_fill"
    PATTERN_FILL = "pattern_fill"
    BLUR_REGION = "blur_region"
    SHARPEN_REGION = "sharpen_region"
    NOISE_REDUCE = "noise_reduce"
    MEDIAN_FILTER = "median_filter"
    EDGE_DETECT = "edge_detect"
    DROP_SHADOW = "drop_shadow"
    STROKE_SELECTION = "stroke_selection"
    CLEAR_REGION = "clear_region"
    CUT = "cut"
    COPY = "copy"
    PASTE = "paste"
    PASTE_AS_NEW_LAYER = "paste_as_new_layer"
    DUPLICATE_REGION_TO_LAYER = "duplicate_region_to_layer"
    TRANSFORM_LAYER = "transform_layer"
    ALIGN_LAYER = "align_layer"
    ADJUST_BRIGHTNESS_CONTRAST = "adjust_brightness_contrast"
    ADJUST_HUE_SATURATION = "adjust_hue_saturation"
    ADJUST_LEVELS = "adjust_levels"
    ADJUST_CURVES = "adjust_curves"
    COLORIZE = "colorize"
    REPLACE_COLOR = "replace_color"
    DESATURATE = "desaturate"
    CREATE_TEXT_LAYER = "create_text_layer"
    EDIT_TEXT_LAYER = "edit_text_layer"
    RASTERIZE_TEXT_LAYER = "rasterize_text_layer"

    # Perception actions
    DETECT_SHAPE = "detect_shape"
    DETECT_OBJECTS = "detect_objects"
    SEGMENT_OBJECT = "segment_object"
    ESTIMATE_DEPTH = "estimate_depth"
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
            ActionType.DRAW_PATH,
            ActionType.BRUSH_STROKE,
            ActionType.ERASE_STROKE,
            ActionType.PAINT_BUCKET_FILL,
            ActionType.GRADIENT_FILL,
            ActionType.PATTERN_FILL,
            ActionType.BLUR_REGION,
            ActionType.SHARPEN_REGION,
            ActionType.NOISE_REDUCE,
            ActionType.MEDIAN_FILTER,
            ActionType.EDGE_DETECT,
            ActionType.STROKE_SELECTION,
            ActionType.CLEAR_REGION,
            ActionType.ADJUST_BRIGHTNESS_CONTRAST,
            ActionType.ADJUST_HUE_SATURATION,
            ActionType.ADJUST_LEVELS,
            ActionType.ADJUST_CURVES,
            ActionType.COLORIZE,
            ActionType.REPLACE_COLOR,
            ActionType.DESATURATE,
            ActionType.INPAINT_REGION,
            ActionType.OUTPAINT_REGION,
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


def _validate_new_document(action: Action) -> None:
    _reject_unknown_keys(
        action.params,
        "params",
        {"width", "height", "color_space", "background_color", "dpi", "title", "author", "source_file", "tags", "custom_metadata"},
    )
    _positive_int(action.params.get("width"), "params.width")
    _positive_int(action.params.get("height"), "params.height")
    if "color_space" in action.params:
        _optional_enum_string(action.params["color_space"], {"srgb", "linear_rgb", "display_p3"}, "params.color_space")
    if "background_color" in action.params:
        _validate_color(action.params["background_color"], "params.background_color")
    if action.params.get("dpi") is not None:
        _positive_number(action.params["dpi"], "params.dpi")
    for key in ("title", "author", "source_file"):
        if key in action.params:
            _optional_string(action.params[key], f"params.{key}")
    if "tags" in action.params:
        _string_list(action.params["tags"], "params.tags")
    if "custom_metadata" in action.params:
        _mapping_value(action.params["custom_metadata"], "params.custom_metadata")


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


def _validate_transform_layer(action: Action) -> None:
    _reject_unknown_keys(
        action.params,
        "params",
        {
            "operation",
            "dx",
            "dy",
            "scale_x",
            "scale_y",
            "angle_degrees",
            "horizontal",
            "vertical",
            "anchor",
            "matrix",
            "resample",
            "fill_color",
        },
    )
    _require_target_id(action.target.layer_id, "target.layer_id")
    operation = action.params.get("operation", "affine")
    if operation not in {"translate", "scale", "rotate", "flip", "affine"}:
        raise ValueError("params.operation must be 'translate', 'scale', 'rotate', 'flip', or 'affine'")
    for key in ("dx", "dy", "scale_x", "scale_y", "angle_degrees"):
        if key in action.params:
            _number(action.params[key], f"params.{key}")
    if "horizontal" in action.params:
        _bool_value(action.params["horizontal"], "params.horizontal")
    if "vertical" in action.params:
        _bool_value(action.params["vertical"], "params.vertical")
    if "anchor" in action.params:
        _point(action.params["anchor"], "params.anchor")
    if "matrix" in action.params:
        _number_list(action.params["matrix"], "params.matrix", length=6)
    if "resample" in action.params:
        _optional_enum_string(action.params["resample"], {"nearest", "bilinear", "bicubic"}, "params.resample")
    if "fill_color" in action.params:
        _validate_color(action.params["fill_color"], "params.fill_color")


def _validate_move_layer(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", {"dx", "dy", "resample", "fill_color"})
    _require_target_id(action.target.layer_id, "target.layer_id")
    _number(action.params.get("dx", 0.0), "params.dx")
    _number(action.params.get("dy", 0.0), "params.dy")
    if "resample" in action.params:
        _optional_enum_string(action.params["resample"], {"nearest", "bilinear", "bicubic"}, "params.resample")
    if "fill_color" in action.params:
        _validate_color(action.params["fill_color"], "params.fill_color")


def _validate_scale_layer(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", {"scale_x", "scale_y", "anchor", "resample", "fill_color"})
    _require_target_id(action.target.layer_id, "target.layer_id")
    _positive_number(action.params.get("scale_x", 1.0), "params.scale_x")
    _positive_number(action.params.get("scale_y", action.params.get("scale_x", 1.0)), "params.scale_y")
    if "anchor" in action.params:
        _point(action.params["anchor"], "params.anchor")
    if "resample" in action.params:
        _optional_enum_string(action.params["resample"], {"nearest", "bilinear", "bicubic"}, "params.resample")
    if "fill_color" in action.params:
        _validate_color(action.params["fill_color"], "params.fill_color")


def _validate_rotate_layer(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", {"angle_degrees", "anchor", "resample", "fill_color"})
    _require_target_id(action.target.layer_id, "target.layer_id")
    _number(action.params.get("angle_degrees"), "params.angle_degrees")
    if "anchor" in action.params:
        _point(action.params["anchor"], "params.anchor")
    if "resample" in action.params:
        _optional_enum_string(action.params["resample"], {"nearest", "bilinear", "bicubic"}, "params.resample")
    if "fill_color" in action.params:
        _validate_color(action.params["fill_color"], "params.fill_color")


def _validate_flip_layer(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", {"horizontal", "vertical", "anchor", "resample", "fill_color"})
    _require_target_id(action.target.layer_id, "target.layer_id")
    _bool_value(action.params.get("horizontal", True), "params.horizontal")
    _bool_value(action.params.get("vertical", False), "params.vertical")
    if "anchor" in action.params:
        _point(action.params["anchor"], "params.anchor")
    if "resample" in action.params:
        _optional_enum_string(action.params["resample"], {"nearest", "bilinear", "bicubic"}, "params.resample")
    if "fill_color" in action.params:
        _validate_color(action.params["fill_color"], "params.fill_color")


def _validate_align_layer(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", {"horizontal", "vertical", "margin", "fill_color"})
    _require_target_id(action.target.layer_id, "target.layer_id")
    if "horizontal" in action.params:
        _optional_enum_string(action.params["horizontal"], {"left", "center", "right", "none"}, "params.horizontal")
    if "vertical" in action.params:
        _optional_enum_string(action.params["vertical"], {"top", "center", "bottom", "none"}, "params.vertical")
    _nonnegative_int(action.params.get("margin", 0), "params.margin")
    if "fill_color" in action.params:
        _validate_color(action.params["fill_color"], "params.fill_color")


def _validate_layer_mask_action(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", {"mode", "source_mask_id", "name", "remove_mask", "invert"})
    _require_target_id(action.target.layer_id, "target.layer_id")
    if action.type == ActionType.ADD_LAYER_MASK:
        _require_target_id(action.target.mask_id, "target.mask_id")
        mode = action.params.get("mode", "from_selection")
        if mode not in {"from_selection", "from_alpha", "full", "empty", "from_mask"}:
            raise ValueError("params.mode must be 'from_selection', 'from_alpha', 'full', 'empty', or 'from_mask'")
        if mode == "from_mask":
            _required_string(action.params, "source_mask_id", "params.source_mask_id")
    if "name" in action.params:
        _optional_string(action.params["name"], "params.name")
    if "remove_mask" in action.params:
        _bool_value(action.params["remove_mask"], "params.remove_mask")
    if "invert" in action.params:
        _bool_value(action.params["invert"], "params.invert")


def _validate_select_polygon(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", {"name", "points", "closed", "kind", "set_active"})
    _require_target_id(action.target.mask_id, "target.mask_id")
    _point_list(action.params.get("points"), "params.points")
    _bool_value(action.params.get("closed", True), "params.closed")
    if "name" in action.params:
        _optional_string(action.params["name"], "params.name")
    if "kind" in action.params:
        _optional_enum_string(
            action.params["kind"],
            {"selection", "write_guard", "layer_alpha", "object", "shape", "line_art_region", "diffusion"},
            "params.kind",
        )
    _bool_value(action.params.get("set_active", True), "params.set_active")


def _validate_select_from_alpha(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", {"name", "threshold", "kind", "set_active"})
    _require_target_id(action.target.layer_id, "target.layer_id")
    _require_target_id(action.target.mask_id, "target.mask_id")
    _optional_unit_number(action.params.get("threshold", 0.01), "params.threshold")
    if "name" in action.params:
        _optional_string(action.params["name"], "params.name")
    if "kind" in action.params:
        _optional_enum_string(
            action.params["kind"],
            {"selection", "write_guard", "layer_alpha", "object", "shape", "line_art_region", "diffusion"},
            "params.kind",
        )
    _bool_value(action.params.get("set_active", True), "params.set_active")


def _validate_save_selection_as_mask(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", {"source_mask_id", "name", "kind", "set_active"})
    _require_target_id(action.target.mask_id, "target.mask_id")
    if "source_mask_id" in action.params:
        _required_string(action.params, "source_mask_id", "params.source_mask_id")
    if "name" in action.params:
        _optional_string(action.params["name"], "params.name")
    if "kind" in action.params:
        _optional_enum_string(
            action.params["kind"],
            {"selection", "write_guard", "layer_alpha", "object", "shape", "line_art_region", "diffusion"},
            "params.kind",
        )
    _bool_value(action.params.get("set_active", False), "params.set_active")


def _validate_mask_cleanup(action: Action) -> None:
    _reject_unknown_keys(
        action.params,
        "params",
        {"source_mask_id", "name", "threshold", "feather_radius", "grow_pixels", "shrink_pixels", "min_area", "set_active"},
    )
    _require_target_id(action.target.mask_id, "target.mask_id")
    _required_string(action.params, "source_mask_id", "params.source_mask_id")
    if "threshold" in action.params:
        _optional_unit_number(action.params["threshold"], "params.threshold")
    if "feather_radius" in action.params:
        _nonnegative_number(action.params["feather_radius"], "params.feather_radius")
    if "grow_pixels" in action.params:
        _nonnegative_int(action.params["grow_pixels"], "params.grow_pixels")
    if "shrink_pixels" in action.params:
        _nonnegative_int(action.params["shrink_pixels"], "params.shrink_pixels")
    if "min_area" in action.params:
        _nonnegative_int(action.params["min_area"], "params.min_area")
    if "name" in action.params:
        _optional_string(action.params["name"], "params.name")
    _bool_value(action.params.get("set_active", False), "params.set_active")


def _validate_path_paint(action: Action) -> None:
    _reject_unknown_keys(
        action.params,
        "params",
        {"points", "color", "width", "opacity", "mode", "hardness", "spacing", "closed"},
    )
    _require_target_id(action.target.layer_id, "target.layer_id")
    _point_list(action.params.get("points"), "params.points")
    if action.type != ActionType.ERASE_STROKE:
        _validate_color(action.params.get("color"), "params.color")
    _positive_number(action.params.get("width", 1.0), "params.width")
    _optional_unit_number(action.params.get("opacity", 1.0), "params.opacity")
    if "mode" in action.params:
        _optional_enum_string(action.params["mode"], {"source_over", "replace_rgba", "alpha_to_zero"}, "params.mode")
    if "hardness" in action.params:
        _optional_unit_number(action.params["hardness"], "params.hardness")
    if "spacing" in action.params:
        _positive_number(action.params["spacing"], "params.spacing")
    if "closed" in action.params:
        _bool_value(action.params["closed"], "params.closed")


def _validate_gradient_fill(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", {"type", "start", "end", "center", "radius", "colors", "mode"})
    _require_target_id(action.target.layer_id, "target.layer_id")
    gradient_type = action.params.get("type", "linear")
    if gradient_type not in {"linear", "radial"}:
        raise ValueError("params.type must be 'linear' or 'radial'")
    if gradient_type == "linear":
        _point(action.params.get("start"), "params.start")
        _point(action.params.get("end"), "params.end")
    else:
        _point(action.params.get("center"), "params.center")
        _positive_number(action.params.get("radius"), "params.radius")
    colors = action.params.get("colors")
    if not isinstance(colors, list) or len(colors) < 2:
        raise TypeError("params.colors must be a list of at least two colors")
    for index, item in enumerate(colors):
        _validate_color(item, f"params.colors[{index}]")
    if "mode" in action.params:
        _optional_enum_string(action.params["mode"], {"replace_rgb_preserve_alpha", "replace_rgba", "source_over"}, "params.mode")


def _validate_pattern_fill(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", {"pattern", "colors", "cell_size", "stripe_width", "angle_degrees", "mode", "path"})
    _require_target_id(action.target.layer_id, "target.layer_id")
    pattern = action.params.get("pattern", "checkerboard")
    if pattern not in {"checkerboard", "stripes", "image"}:
        raise ValueError("params.pattern must be 'checkerboard', 'stripes', or 'image'")
    if pattern == "image":
        _required_string(action.params, "path", "params.path")
    colors = action.params.get("colors", ["#000000", "#ffffff"])
    if not isinstance(colors, list) or len(colors) < 1:
        raise TypeError("params.colors must be a non-empty list")
    for index, item in enumerate(colors):
        _validate_color(item, f"params.colors[{index}]")
    _positive_int(action.params.get("cell_size", 16), "params.cell_size")
    _positive_int(action.params.get("stripe_width", action.params.get("cell_size", 16)), "params.stripe_width")
    if "angle_degrees" in action.params:
        _number(action.params["angle_degrees"], "params.angle_degrees")
    if "mode" in action.params:
        _optional_enum_string(action.params["mode"], {"replace_rgb_preserve_alpha", "replace_rgba", "source_over"}, "params.mode")


def _validate_color_adjustment(action: Action) -> None:
    allowed = {
        "brightness",
        "contrast",
        "hue_degrees",
        "saturation",
        "lightness",
        "gamma",
        "in_black",
        "in_white",
        "out_black",
        "out_white",
        "points",
        "color",
        "source_color",
        "target_color",
        "tolerance",
        "softness",
        "method",
        "amount",
    }
    _reject_unknown_keys(action.params, "params", allowed)
    _require_target_id(action.target.layer_id, "target.layer_id")
    for key in ("brightness", "contrast", "hue_degrees", "saturation", "lightness", "gamma", "in_black", "in_white", "out_black", "out_white", "tolerance", "softness", "amount"):
        if key in action.params:
            _number(action.params[key], f"params.{key}")
    for key in ("color", "source_color", "target_color"):
        if key in action.params:
            _validate_color(action.params[key], f"params.{key}")
    if action.type == ActionType.ADJUST_CURVES:
        points = action.params.get("points")
        if not isinstance(points, list) or len(points) < 2:
            raise TypeError("params.points must be a list of at least two [input, output] points")
        for index, point in enumerate(points):
            _point(point, f"params.points[{index}]")
    if "method" in action.params:
        _optional_enum_string(action.params["method"], {"luminance", "average", "lightness"}, "params.method")


def _validate_filter_action(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", {"radius", "amount", "threshold", "channels", "mode", "color", "offset", "blur_radius", "opacity", "source_mask_id", "output_layer_name"})
    _require_target_id(action.target.layer_id, "target.layer_id")
    for key in ("radius", "amount", "threshold", "blur_radius", "opacity"):
        if key in action.params:
            _nonnegative_number(action.params[key], f"params.{key}")
    if "channels" in action.params:
        _validate_channels(action.params["channels"], "params.channels")
    if "mode" in action.params:
        _optional_enum_string(action.params["mode"], {"replace_rgb_preserve_alpha", "replace_rgba", "source_over", "alpha", "luminance"}, "params.mode")
    if "color" in action.params:
        _validate_color(action.params["color"], "params.color")
    if "offset" in action.params:
        _point(action.params["offset"], "params.offset")
    if "source_mask_id" in action.params:
        _required_string(action.params, "source_mask_id", "params.source_mask_id")
    if "output_layer_name" in action.params:
        _optional_string(action.params["output_layer_name"], "params.output_layer_name")


def _validate_cut_copy_paste(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", {"source_mask_id", "bbox_xyxy", "x", "y", "name", "clear_mode", "preserve_rgb", "set_active"})
    if action.type in {ActionType.CUT, ActionType.COPY, ActionType.DUPLICATE_REGION_TO_LAYER}:
        _require_target_id(action.target.layer_id, "target.layer_id")
    if action.type in {ActionType.PASTE, ActionType.PASTE_AS_NEW_LAYER, ActionType.DUPLICATE_REGION_TO_LAYER}:
        _require_target_id(action.target.output_layer_id, "target.output_layer_id")
    if "source_mask_id" in action.params:
        _required_string(action.params, "source_mask_id", "params.source_mask_id")
    if "bbox_xyxy" in action.params:
        _bbox_xyxy(action.params["bbox_xyxy"], "params.bbox_xyxy")
    if "x" in action.params:
        _integer_number(action.params["x"], "params.x")
    if "y" in action.params:
        _integer_number(action.params["y"], "params.y")
    if "name" in action.params:
        _optional_string(action.params["name"], "params.name")
    if "clear_mode" in action.params:
        _optional_enum_string(action.params["clear_mode"], {"alpha_to_zero", "rgba_to_zero"}, "params.clear_mode")
    if "preserve_rgb" in action.params:
        _bool_value(action.params["preserve_rgb"], "params.preserve_rgb")
    _bool_value(action.params.get("set_active", True), "params.set_active")


def _validate_text_action(action: Action) -> None:
    _reject_unknown_keys(
        action.params,
        "params",
        {
            "text",
            "name",
            "x",
            "y",
            "font",
            "font_id",
            "font_path",
            "font_family",
            "font_style",
            "font_weight",
            "font_size",
            "style",
            "color",
            "outline",
            "outline_color",
            "outline_width",
            "stroke_color",
            "stroke_width",
            "layout",
            "anchor",
            "align",
            "spacing",
            "set_active",
        },
    )
    if action.type == ActionType.CREATE_TEXT_LAYER:
        _require_target_id(action.target.output_layer_id, "target.output_layer_id")
    else:
        _require_target_id(action.target.layer_id, "target.layer_id")
    if action.type == ActionType.CREATE_TEXT_LAYER:
        _required_string(action.params, "text", "params.text")
    elif action.type == ActionType.EDIT_TEXT_LAYER and "text" in action.params:
        _optional_string(action.params["text"], "params.text")
    if "name" in action.params:
        _optional_string(action.params["name"], "params.name")
    _integer_number(action.params.get("x", 0), "params.x")
    _integer_number(action.params.get("y", 0), "params.y")
    if "font" in action.params:
        _validate_text_font_object(action.params["font"], "params.font")
    for key in ("font_id", "font_path", "font_family", "font_style"):
        if key in action.params:
            _optional_string(action.params[key], f"params.{key}")
    if "font_weight" in action.params:
        _positive_int(action.params["font_weight"], "params.font_weight")
    if "font_path" in action.params:
        _optional_string(action.params["font_path"], "params.font_path")
    _positive_int(action.params.get("font_size", 32), "params.font_size")
    if "style" in action.params:
        _validate_text_style_object(action.params["style"], "params.style")
    if "color" in action.params:
        _validate_color(action.params["color"], "params.color")
    if "outline" in action.params:
        _validate_text_outline_object(action.params["outline"], "params.outline")
    for key in ("outline_color", "stroke_color"):
        if key in action.params:
            _validate_color(action.params[key], f"params.{key}")
    for key in ("outline_width", "stroke_width"):
        if key in action.params:
            _nonnegative_number(action.params[key], f"params.{key}")
    if "layout" in action.params:
        _validate_text_layout_object(action.params["layout"], "params.layout")
    if "anchor" in action.params:
        _optional_string(action.params["anchor"], "params.anchor")
    if "align" in action.params:
        _optional_enum_string(action.params["align"], {"left", "center", "right"}, "params.align")
    _nonnegative_int(action.params.get("spacing", 0), "params.spacing")
    _bool_value(action.params.get("set_active", True), "params.set_active")


def _validate_text_font_object(value: Any, field_name: str) -> None:
    font = _mapping_value(value, field_name)
    _reject_unknown_keys(font, field_name, {"id", "path", "family", "style", "weight", "size"})
    for key in ("id", "path", "family", "style"):
        if key in font:
            _optional_string(font[key], f"{field_name}.{key}")
    if "weight" in font:
        _positive_int(font["weight"], f"{field_name}.weight")
    if "size" in font:
        _positive_int(font["size"], f"{field_name}.size")


def _validate_text_style_object(value: Any, field_name: str) -> None:
    style = _mapping_value(value, field_name)
    _reject_unknown_keys(style, field_name, {"color", "color_rgba", "outline"})
    if "color" in style:
        _validate_color(style["color"], f"{field_name}.color")
    if "color_rgba" in style:
        _validate_color(style["color_rgba"], f"{field_name}.color_rgba")
    if "outline" in style:
        _validate_text_outline_object(style["outline"], f"{field_name}.outline")


def _validate_text_outline_object(value: Any, field_name: str) -> None:
    outline = _mapping_value(value, field_name)
    _reject_unknown_keys(outline, field_name, {"color", "color_rgba", "width"})
    if "color" in outline:
        _validate_color(outline["color"], f"{field_name}.color")
    if "color_rgba" in outline:
        _validate_color(outline["color_rgba"], f"{field_name}.color_rgba")
    if "width" in outline:
        _nonnegative_number(outline["width"], f"{field_name}.width")


def _validate_text_layout_object(value: Any, field_name: str) -> None:
    layout = _mapping_value(value, field_name)
    _reject_unknown_keys(layout, field_name, {"x", "y", "anchor", "align", "spacing"})
    if "x" in layout:
        _integer_number(layout["x"], f"{field_name}.x")
    if "y" in layout:
        _integer_number(layout["y"], f"{field_name}.y")
    if "anchor" in layout:
        _optional_string(layout["anchor"], f"{field_name}.anchor")
    if "align" in layout:
        _optional_enum_string(layout["align"], {"left", "center", "right"}, f"{field_name}.align")
    if "spacing" in layout:
        _nonnegative_int(layout["spacing"], f"{field_name}.spacing")


def _validate_perception_action(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", {"threshold", "alpha_min", "tolerance", "seed_points", "mode", "name", "set_active", "output_layer_name", "min_area", "max_objects"})
    if action.type in {ActionType.DETECT_SHAPE, ActionType.DETECT_OBJECTS, ActionType.SEGMENT_OBJECT, ActionType.ESTIMATE_DEPTH, ActionType.EXTRACT_LINE_ART, ActionType.DECOMPOSE_TO_LAYERS}:
        _require_target_id(action.target.layer_id, "target.layer_id")
    if action.type in {ActionType.SEGMENT_OBJECT, ActionType.ESTIMATE_DEPTH, ActionType.EXTRACT_LINE_ART}:
        _require_target_id(action.target.mask_id, "target.mask_id")
    if action.type == ActionType.DECOMPOSE_TO_LAYERS:
        _require_target_id(action.target.output_layer_id, "target.output_layer_id")
    for key in ("threshold", "alpha_min", "tolerance"):
        if key in action.params:
            _optional_unit_number(action.params[key], f"params.{key}")
    if "seed_points" in action.params:
        _point_list(action.params["seed_points"], "params.seed_points")
    if "mode" in action.params:
        _optional_enum_string(action.params["mode"], {"alpha", "luminance", "color", "edges"}, "params.mode")
    if "min_area" in action.params:
        _nonnegative_int(action.params["min_area"], "params.min_area")
    if "max_objects" in action.params:
        _positive_int(action.params["max_objects"], "params.max_objects")
    if "name" in action.params:
        _optional_string(action.params["name"], "params.name")
    if "output_layer_name" in action.params:
        _optional_string(action.params["output_layer_name"], "params.output_layer_name")
    _bool_value(action.params.get("set_active", True), "params.set_active")


def _validate_diffusion_action(action: Action) -> None:
    _reject_unknown_keys(
        action.params,
        "params",
        {"prompt", "negative_prompt", "seed", "denoise", "guidance_scale", "steps", "backend", "job", "output_layer_name", "mode", "padding"},
    )
    if action.type in {ActionType.INPAINT_REGION, ActionType.OUTPAINT_REGION, ActionType.IMG2IMG_TO_LAYER}:
        _require_target_id(action.target.layer_id, "target.layer_id")
    if action.type in {ActionType.TXT2IMG_TO_LAYER, ActionType.IMG2IMG_TO_LAYER, ActionType.INPAINT_REGION, ActionType.OUTPAINT_REGION}:
        _require_target_id(action.target.output_layer_id, "target.output_layer_id")
    if action.type in {ActionType.INPAINT_REGION, ActionType.OUTPAINT_REGION} and action.write_mask_id is None:
        raise ValueError("diffusion region actions require write_mask_id")
    if "prompt" in action.params:
        _optional_string(action.params["prompt"], "params.prompt")
    if "negative_prompt" in action.params:
        _optional_string(action.params["negative_prompt"], "params.negative_prompt")
    if "seed" in action.params and action.params["seed"] is not None:
        _integer_number(action.params["seed"], "params.seed")
    if "denoise" in action.params:
        _optional_unit_number(action.params["denoise"], "params.denoise")
    if "guidance_scale" in action.params:
        _nonnegative_number(action.params["guidance_scale"], "params.guidance_scale")
    if "steps" in action.params:
        _positive_int(action.params["steps"], "params.steps")
    if "backend" in action.params:
        _optional_string(action.params["backend"], "params.backend")
    if "job" in action.params:
        _mapping_value(action.params["job"], "params.job")
    if "output_layer_name" in action.params:
        _optional_string(action.params["output_layer_name"], "params.output_layer_name")
    if "mode" in action.params:
        _optional_enum_string(action.params["mode"], {"replace_region", "new_layer"}, "params.mode")
    if "padding" in action.params:
        _padding_value(action.params["padding"], "params.padding")


def _validate_export_flat(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", {"path"})
    _required_string(action.params, "path", "params.path")


def _validate_export_layered_bundle(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", {"path", "include_preview", "include_hidden", "overwrite"})
    _required_string(action.params, "path", "params.path")
    _bool_value(action.params.get("include_preview", True), "params.include_preview")
    _bool_value(action.params.get("include_hidden", True), "params.include_hidden")
    _bool_value(action.params.get("overwrite", True), "params.overwrite")


def _validate_no_op(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", set())


def _validate_validate(action: Action) -> None:
    _reject_unknown_keys(action.params, "params", set())


_PARAM_VALIDATORS = {
    ActionType.NEW_DOCUMENT: _validate_new_document,
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
    ActionType.MOVE_LAYER: _validate_move_layer,
    ActionType.SCALE_LAYER: _validate_scale_layer,
    ActionType.ROTATE_LAYER: _validate_rotate_layer,
    ActionType.FLIP_LAYER: _validate_flip_layer,
    ActionType.TRANSFORM_LAYER: _validate_transform_layer,
    ActionType.ALIGN_LAYER: _validate_align_layer,
    ActionType.ADD_LAYER_MASK: _validate_layer_mask_action,
    ActionType.APPLY_LAYER_MASK: _validate_layer_mask_action,
    ActionType.REMOVE_LAYER_MASK: _validate_layer_mask_action,
    ActionType.SELECT_RECT: _validate_select_rect,
    ActionType.SELECT_ELLIPSE: _validate_select_ellipse,
    ActionType.SELECT_POLYGON: _validate_select_polygon,
    ActionType.SELECT_FREEHAND: _validate_select_polygon,
    ActionType.SELECT_FROM_ALPHA: _validate_select_from_alpha,
    ActionType.SELECT_COLOR_RANGE: _validate_select_color_range,
    ActionType.MAGIC_WAND_SELECT: _validate_magic_wand_select,
    ActionType.SAVE_SELECTION_AS_MASK: _validate_save_selection_as_mask,
    ActionType.CREATE_MASK_FROM_SHAPE: _validate_create_mask_from_shape,
    ActionType.GROW_MASK: _validate_grow_mask,
    ActionType.SHRINK_MASK: _validate_shrink_mask,
    ActionType.INVERT_MASK: _validate_invert_mask,
    ActionType.COMBINE_MASKS: _validate_combine_masks,
    ActionType.FEATHER_MASK: _validate_feather_mask,
    ActionType.REFINE_SELECTION: _validate_mask_cleanup,
    ActionType.REMOVE_SMALL_ISLANDS: _validate_mask_cleanup,
    ActionType.FILL_MASK_HOLES: _validate_mask_cleanup,
    ActionType.DRAW_SHAPE: _validate_draw_shape,
    ActionType.DRAW_PATH: _validate_path_paint,
    ActionType.BRUSH_STROKE: _validate_path_paint,
    ActionType.ERASE_STROKE: _validate_path_paint,
    ActionType.PAINT_BUCKET_FILL: _validate_paint_bucket_fill,
    ActionType.GRADIENT_FILL: _validate_gradient_fill,
    ActionType.PATTERN_FILL: _validate_pattern_fill,
    ActionType.BLUR_REGION: _validate_blur_region,
    ActionType.SHARPEN_REGION: _validate_filter_action,
    ActionType.NOISE_REDUCE: _validate_filter_action,
    ActionType.MEDIAN_FILTER: _validate_filter_action,
    ActionType.EDGE_DETECT: _validate_filter_action,
    ActionType.DROP_SHADOW: _validate_filter_action,
    ActionType.STROKE_SELECTION: _validate_filter_action,
    ActionType.CLEAR_REGION: _validate_clear_region,
    ActionType.CUT: _validate_cut_copy_paste,
    ActionType.COPY: _validate_cut_copy_paste,
    ActionType.PASTE: _validate_cut_copy_paste,
    ActionType.PASTE_AS_NEW_LAYER: _validate_cut_copy_paste,
    ActionType.DUPLICATE_REGION_TO_LAYER: _validate_cut_copy_paste,
    ActionType.ADJUST_BRIGHTNESS_CONTRAST: _validate_color_adjustment,
    ActionType.ADJUST_HUE_SATURATION: _validate_color_adjustment,
    ActionType.ADJUST_LEVELS: _validate_color_adjustment,
    ActionType.ADJUST_CURVES: _validate_color_adjustment,
    ActionType.COLORIZE: _validate_color_adjustment,
    ActionType.REPLACE_COLOR: _validate_color_adjustment,
    ActionType.DESATURATE: _validate_color_adjustment,
    ActionType.CREATE_TEXT_LAYER: _validate_text_action,
    ActionType.EDIT_TEXT_LAYER: _validate_text_action,
    ActionType.RASTERIZE_TEXT_LAYER: _validate_text_action,
    ActionType.DETECT_SHAPE: _validate_perception_action,
    ActionType.DETECT_OBJECTS: _validate_perception_action,
    ActionType.SEGMENT_OBJECT: _validate_perception_action,
    ActionType.ESTIMATE_DEPTH: _validate_perception_action,
    ActionType.EXTRACT_LINE_ART: _validate_perception_action,
    ActionType.DECOMPOSE_TO_LAYERS: _validate_perception_action,
    ActionType.TXT2IMG_TO_LAYER: _validate_diffusion_action,
    ActionType.IMG2IMG_TO_LAYER: _validate_diffusion_action,
    ActionType.INPAINT_REGION: _validate_diffusion_action,
    ActionType.OUTPAINT_REGION: _validate_diffusion_action,
    ActionType.VALIDATE: _validate_validate,
    ActionType.EXPORT_FLAT: _validate_export_flat,
    ActionType.EXPORT_LAYERED_BUNDLE: _validate_export_layered_bundle,
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
        points.append(_point(item, f"{field_name}[{index}]"))
    return points


def _point(value: Any, field_name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError(f"{field_name} must be a two-number list")
    return (_number(value[0], f"{field_name}.x"), _number(value[1], f"{field_name}.y"))


def _number_list(value: Any, field_name: str, length: Optional[int] = None) -> list[float]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list of numbers")
    if length is not None and len(value) != length:
        raise ValueError(f"{field_name} must contain exactly {length} numbers")
    return [_number(item, f"{field_name} entry") for item in value]


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


def _padding_value(value: Any, field_name: str) -> None:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer or four-integer list")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{field_name} must not be negative")
        return
    if isinstance(value, (list, tuple)) and len(value) == 4:
        for index, item in enumerate(value):
            if isinstance(item, bool) or not isinstance(item, int):
                raise TypeError(f"{field_name}[{index}] must be an integer")
            if item < 0:
                raise ValueError(f"{field_name}[{index}] must not be negative")
        return
    raise TypeError(f"{field_name} must be an integer or four-integer list")
