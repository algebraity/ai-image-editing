"""Typed action schema for the AI Editing Kernel.

The action schema is the contract between the planner and the editing runtime.
The LLM or policy model should output Actions, not arbitrary Python. The Executor
then validates and applies those actions to DocumentState.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


JsonObject = dict[str, Any]


class ActionType(str, Enum):
    """Initial action vocabulary.

    This list should grow slowly. A smaller, well-tested action set is better
    than a large action set the planner cannot use consistently.
    """

    # Document actions
    NEW_DOCUMENT = "new_document"
    IMPORT_IMAGE_AS_LAYER = "import_image_as_layer"
    EXPORT_FLAT = "export_flat"
    EXPORT_LAYERED_BUNDLE = "export_layered_bundle"
    RESIZE_CANVAS = "resize_canvas"

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
    """Where an action is intended to operate.

    Not every field is required for every action. For example, CREATE_LAYER may
    not need a target_layer_id, while CLEAR_REGION should specify both a target
    layer and a write mask.
    """

    document_id: Optional[str] = None
    layer_id: Optional[str] = None
    layer_name: Optional[str] = None
    mask_id: Optional[str] = None
    selection_id: Optional[str] = None
    output_layer_id: Optional[str] = None


@dataclass(slots=True)
class ActionPreconditions:
    """Conditions that must hold before executing an action."""

    required_layer_ids: list[str] = field(default_factory=list)
    required_mask_ids: list[str] = field(default_factory=list)
    require_active_layer: bool = False
    require_active_selection: bool = False
    require_unlocked_target_layer: bool = True
    require_write_mask: bool = True
    allow_hidden_layers: bool = False
    custom: JsonObject = field(default_factory=dict)


@dataclass(slots=True)
class ExpectedResult:
    """What the action is expected to change or create.

    Validators use this to check that execution did what the planner intended.
    """

    changed_layer_ids: list[str] = field(default_factory=list)
    created_layer_names: list[str] = field(default_factory=list)
    created_mask_names: list[str] = field(default_factory=list)
    protected_mask_id: Optional[str] = None
    geometry_expectations: JsonObject = field(default_factory=dict)
    visual_expectations: JsonObject = field(default_factory=dict)
    custom: JsonObject = field(default_factory=dict)


@dataclass(slots=True)
class Action:
    """A single structured operation requested by a planner or user.

    `params` holds action-specific parameters. Later, individual action parameter
    dataclasses can be enforced per ActionType. V0 can use this generic field to
    iterate quickly while still logging a stable action envelope.
    """

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
        """Validate that required fields and params exist for this action type.

        This should be a lightweight schema check, not a document-state check.
        Document-specific checks belong in Validator.validate_preconditions().
        """
        raise NotImplementedError

    def requires_pixel_write(self) -> bool:
        """Return whether this action can change pixel or alpha data.

        Pixel-writing actions should always have a write mask unless they are
        explicitly full-document operations with a full-canvas write guard.
        """
        raise NotImplementedError

    def to_json(self) -> JsonObject:
        """Serialize the action to a JSON-compatible dictionary for traces."""
        raise NotImplementedError

    @classmethod
    def from_json(cls, data: JsonObject) -> "Action":
        """Deserialize an Action from trace or API JSON."""
        raise NotImplementedError


@dataclass(slots=True)
class ActionBatch:
    """A sequence of actions planned as one logical editing operation."""

    id: str
    actions: list[Action]
    user_prompt: Optional[str] = None
    description: Optional[str] = None
    stop_on_error: bool = True
    metadata: JsonObject = field(default_factory=dict)

    def validate_schema(self) -> None:
        """Validate all actions and batch-level constraints.

        Should reject duplicate action IDs, invalid action types, and inconsistent
        batch settings before anything is executed.
        """
        raise NotImplementedError

    def to_json(self) -> JsonObject:
        """Serialize the batch to JSON-compatible form."""
        raise NotImplementedError


@dataclass(slots=True)
class ActionError:
    """Structured error returned when an action cannot be executed."""

    code: str
    message: str
    action_id: Optional[str] = None
    details: JsonObject = field(default_factory=dict)
    recoverable: bool = False


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
        raise NotImplementedError

    def to_json(self) -> JsonObject:
        """Serialize result to JSON-compatible form for traces."""
        raise NotImplementedError


# Optional typed param skeletons for common early actions. These are not wired
# into Action yet, but they document the expected contents of Action.params.

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
    bbox: tuple[float, float, float, float]
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
