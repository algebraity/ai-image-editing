"""Machine-readable action catalog for AI planners.

The executor consumes canonical `ActionBatch` JSON. Models should not have to
produce that full envelope directly. This module exposes a stricter, smaller
planner-facing schema for each action: the model chooses an action type, target
references when they are semantically necessary, and params. The planner layer
then supplies action IDs, generated output IDs, preconditions, expected-result
metadata, and other kernel bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ai_edit_kernel.schema.actions import ActionType


PLANNER_CATALOG_VERSION = "planner_tools.v1"
PLANNER_OUTPUT_SCHEMA_VERSION = "ai_edit_planner_output.v1"


JsonObject = dict[str, Any]


class TargetFieldMode(str, Enum):
    """How a planner should treat a target field."""

    REQUIRED = "required"
    OPTIONAL = "optional"
    GENERATED = "generated"
    DEFAULT_ACTIVE_LAYER = "default_active_layer"
    DEFAULT_ACTIVE_SELECTION = "default_active_selection"


@dataclass(frozen=True, slots=True)
class ActionToolSpec:
    """Planner-facing contract for one executable action."""

    name: str
    category: str
    summary: str
    params_schema: JsonObject = field(default_factory=dict)
    target_fields: dict[str, TargetFieldMode] = field(default_factory=dict)
    write_mask: TargetFieldMode = TargetFieldMode.OPTIONAL
    kernel_filled_fields: tuple[str, ...] = (
        "id",
        "created_by",
        "preconditions",
        "expected_result",
    )
    notes: tuple[str, ...] = ()

    def to_json(self) -> JsonObject:
        """Return a JSON-compatible description suitable for planner prompts."""
        return {
            "name": self.name,
            "category": self.category,
            "summary": self.summary,
            "planner_schema": self.planner_schema(),
            "target_fields": {key: TargetFieldMode(value).value for key, value in self.target_fields.items()},
            "write_mask": TargetFieldMode(self.write_mask).value,
            "kernel_filled_fields": list(self.kernel_filled_fields),
            "notes": list(self.notes),
        }

    def planner_schema(self) -> JsonObject:
        """Return the compact action object schema expected from a model."""
        target_properties = {
            field_name: {
                "type": "string",
                "description": _target_field_description(field_name, mode),
            }
            for field_name, mode in self.target_fields.items()
        }
        target_required = [
            field_name
            for field_name, mode in self.target_fields.items()
            if TargetFieldMode(mode) == TargetFieldMode.REQUIRED
        ]

        properties: JsonObject = {
            "type": {"const": self.name},
            "params": self.params_schema,
            "target": {
                "type": "object",
                "additionalProperties": False,
                "required": target_required,
                "properties": target_properties,
            },
            "write_mask_id": {
                "type": "string",
                "description": _write_mask_description(self.write_mask),
            },
            "description": {
                "type": "string",
                "description": "Short human-readable reason for this action.",
            },
            "metadata": {
                "type": "object",
                "description": "Optional planner metadata. Avoid private reasoning.",
            },
        }
        required = ["type"]
        if _has_required_params(self.params_schema):
            required.append("params")
        if target_required:
            required.append("target")
        if TargetFieldMode(self.write_mask) == TargetFieldMode.REQUIRED:
            required.append("write_mask_id")

        return {
            "type": "object",
            "additionalProperties": False,
            "required": required,
            "properties": properties,
        }


def planner_output_schema() -> JsonObject:
    """Return the top-level schema expected from a planner backend."""
    return {
        "schema_version": PLANNER_OUTPUT_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "required": ["actions"],
        "properties": {
            "actions": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "description": "One planner action. Its exact params are defined by available_actions.",
                },
            },
            "description": {"type": "string"},
            "stop_on_error": {"type": "boolean", "default": True},
            "metadata": {"type": "object"},
        },
    }


def available_action_specs() -> list[JsonObject]:
    """Return all action specs as JSON-compatible objects."""
    return [spec.to_json() for spec in ACTION_TOOL_SPECS.values()]


def get_action_spec(action_type: ActionType | str) -> ActionToolSpec:
    """Return the spec for `action_type`, raising KeyError if it is unknown."""
    key = ActionType(action_type).value
    return ACTION_TOOL_SPECS[key]


def param_schema(
    properties: dict[str, JsonObject] | None = None,
    *,
    required: tuple[str, ...] = (),
    description: str | None = None,
) -> JsonObject:
    """Create a JSON-object schema for action params."""
    schema: JsonObject = {
        "type": "object",
        "additionalProperties": False,
        "required": list(required),
        "properties": {} if properties is None else properties,
    }
    if description is not None:
        schema["description"] = description
    return schema


def field(type_name: str | list[str], description: str, **extra: Any) -> JsonObject:
    """Create a JSON-schema property description."""
    data: JsonObject = {"type": type_name, "description": description}
    data.update(extra)
    return data


def enum_field(values: tuple[str, ...], description: str, **extra: Any) -> JsonObject:
    """Create a string enum property."""
    return field("string", description, enum=list(values), **extra)


def number_field(description: str, **extra: Any) -> JsonObject:
    """Create a numeric property."""
    return field("number", description, **extra)


def integer_field(description: str, **extra: Any) -> JsonObject:
    """Create an integer property."""
    return field("integer", description, **extra)


def bool_field(description: str, **extra: Any) -> JsonObject:
    """Create a boolean property."""
    return field("boolean", description, **extra)


def string_field(description: str, **extra: Any) -> JsonObject:
    """Create a string property."""
    return field("string", description, **extra)


def array_field(items: JsonObject, description: str, **extra: Any) -> JsonObject:
    """Create an array property."""
    return field("array", description, items=items, **extra)


def object_field(properties: dict[str, JsonObject], description: str, *, required: tuple[str, ...] = ()) -> JsonObject:
    """Create an object property."""
    return field(
        "object",
        description,
        additionalProperties=False,
        required=list(required),
        properties=properties,
    )


COLOR = field(
    ["string", "array"],
    "Color as #RRGGBB, #RRGGBBAA, or [r, g, b, a] floats in [0, 1].",
)
COLOR_LIST = array_field(COLOR, "Non-empty list of material colors.", minItems=1)
BBOX = array_field(number_field("Coordinate."), "Half-open [x0, y0, x1, y1] pixel bounds.", minItems=4, maxItems=4)
POINT = array_field(number_field("Coordinate."), "Point as [x, y].", minItems=2, maxItems=2)
POINTS = array_field(POINT, "Non-empty list of [x, y] points.", minItems=1)
SELECTION_CLICK = object_field(
    {
        "point": POINT,
        "operation": enum_field(("replace", "add", "subtract", "intersect"), "How this click combines with previous clicks.", default="add"),
        "threshold": number_field("Optional per-click GIMP-style threshold in 0-255 units.", minimum=0, maximum=255),
        "tolerance": number_field("Optional per-click normalized tolerance; legacy alias for threshold / 255.", minimum=0),
    },
    "One fuzzy-select click.",
    required=("point",),
)
SELECTION_CLICKS = array_field(SELECTION_CLICK, "One or more fuzzy-select clicks.", minItems=1)
SELECT_CRITERION = enum_field(
    ("composite", "rgb-red", "rgb-green", "rgb-blue", "hsv-hue", "hsv-saturation", "hsv-value", "alpha"),
    "GIMP-style color similarity criterion. Use composite for normal fuzzy select.",
    default="composite",
)
ID_LIST = array_field(string_field("Existing ID."), "List of existing IDs.", minItems=1)
CHANNELS = field(
    ["string", "array"],
    "Channels: 'rgb', 'rgba', 'alpha', one of 'r'/'g'/'b'/'a', or a list of channel letters.",
)
SHAPE = object_field(
    {
        "type": enum_field(("rectangle", "ellipse"), "Shape type."),
        "bbox_xyxy": BBOX,
        "corner_radius": number_field("Rounded-corner radius for rectangles. Currently accepted but not rasterized.", minimum=0),
    },
    "Deterministic shape geometry.",
    required=("type", "bbox_xyxy"),
)
STROKE = object_field(
    {
        "color": COLOR,
        "width": number_field("Stroke width in pixels.", exclusiveMinimum=0),
    },
    "Stroke style.",
    required=("color", "width"),
)
FILL = object_field({"color": COLOR}, "Fill style.", required=("color",))
MASK_KIND = enum_field(
    ("selection", "write_guard", "layer_alpha", "object", "shape", "line_art_region", "diffusion"),
    "Semantic role for a generated mask.",
)
BLEND_MODE = enum_field(("normal", "multiply", "screen", "overlay", "add", "subtract"), "Layer blend mode metadata.")
RESAMPLE = enum_field(("nearest", "bilinear", "bicubic"), "Resampling method.")


def _text_param_properties() -> dict[str, JsonObject]:
    """Return planner-facing text params shared by create/edit actions."""
    outline = object_field(
        {
            "color": COLOR,
            "color_rgba": COLOR,
            "width": number_field("Outline/stroke width in pixels.", minimum=0),
        },
        "Text outline settings.",
    )
    return {
        "text": string_field("Text content."),
        "name": string_field("Layer name."),
        "x": integer_field("Canvas x coordinate.", default=0),
        "y": integer_field("Canvas y coordinate.", default=0),
        "font": object_field(
            {
                "id": string_field("Stable font ID from the kernel font registry."),
                "path": string_field("Explicit font file path."),
                "family": string_field("Font family lookup string."),
                "style": string_field("Font style lookup string, such as Bold or Italic."),
                "weight": integer_field("Font weight, such as 400 or 700.", minimum=1),
                "size": integer_field("Font size in pixels.", minimum=1, default=32),
            },
            "Structured font request. Prefer id for reproducible plans.",
        ),
        "font_id": string_field("Stable font ID from the kernel font registry."),
        "font_path": string_field("Explicit font file path."),
        "font_family": string_field("Font family lookup string."),
        "font_style": string_field("Font style lookup string."),
        "font_weight": integer_field("Font weight, such as 400 or 700.", minimum=1),
        "font_size": integer_field("Font size in pixels.", minimum=1, default=32),
        "style": object_field(
            {
                "color": COLOR,
                "color_rgba": COLOR,
                "outline": outline,
            },
            "Structured text style settings.",
        ),
        "color": COLOR,
        "outline": outline,
        "outline_color": COLOR,
        "outline_width": number_field("Outline width in pixels.", minimum=0),
        "stroke_color": COLOR,
        "stroke_width": number_field("Outline width in pixels.", minimum=0),
        "layout": object_field(
            {
                "x": integer_field("Canvas x coordinate.", default=0),
                "y": integer_field("Canvas y coordinate.", default=0),
                "anchor": string_field("Pillow-style text anchor."),
                "align": enum_field(("left", "center", "right"), "Text alignment."),
                "spacing": integer_field("Line spacing.", minimum=0),
            },
            "Structured text placement settings.",
        ),
        "anchor": string_field("Pillow-style text anchor."),
        "align": enum_field(("left", "center", "right"), "Text alignment."),
        "spacing": integer_field("Line spacing.", minimum=0),
        "set_active": bool_field("Make layer active.", default=True),
    }


def spec(
    action_type: ActionType,
    category: str,
    summary: str,
    params: JsonObject | None = None,
    *,
    target_fields: dict[str, TargetFieldMode] | None = None,
    write_mask: TargetFieldMode = TargetFieldMode.OPTIONAL,
    notes: tuple[str, ...] = (),
) -> ActionToolSpec:
    """Create one action spec."""
    return ActionToolSpec(
        name=action_type.value,
        category=category,
        summary=summary,
        params_schema=param_schema() if params is None else params,
        target_fields={} if target_fields is None else target_fields,
        write_mask=write_mask,
        notes=notes,
    )


TF = TargetFieldMode


ACTION_TOOL_SPECS: dict[str, ActionToolSpec] = {
    item.name: item
    for item in [
        spec(
            ActionType.NEW_DOCUMENT,
            "document",
            "Replace the current document with a new empty canvas.",
            param_schema(
                {
                    "width": integer_field("Canvas width in pixels.", minimum=1),
                    "height": integer_field("Canvas height in pixels.", minimum=1),
                    "color_space": enum_field(("srgb", "linear_rgb", "display_p3"), "Document color space.", default="srgb"),
                    "background_color": COLOR,
                    "dpi": number_field("Optional document DPI.", exclusiveMinimum=0),
                    "title": string_field("Optional document title."),
                    "author": string_field("Optional document author."),
                    "source_file": string_field("Optional source path metadata."),
                    "tags": array_field(string_field("Tag."), "Document tags."),
                    "custom_metadata": field("object", "Arbitrary JSON-safe document metadata."),
                },
                required=("width", "height"),
            ),
            target_fields={"document_id": TF.GENERATED},
        ),
        spec(
            ActionType.IMPORT_IMAGE_AS_LAYER,
            "document",
            "Import a raster image file as a new full-canvas layer.",
            param_schema(
                {
                    "path": string_field("Filesystem path to an image readable by Pillow."),
                    "name": string_field("Layer name. Defaults to the filename stem."),
                    "x": integer_field("Canvas x offset.", default=0),
                    "y": integer_field("Canvas y offset.", default=0),
                    "opacity": number_field("Layer opacity.", minimum=0, maximum=1, default=1),
                    "blend_mode": BLEND_MODE,
                    "set_active": bool_field("Whether to make the imported layer active.", default=True),
                },
                required=("path",),
            ),
            target_fields={"output_layer_id": TF.GENERATED},
        ),
        spec(
            ActionType.IMPORT_VECTOR_AS_RASTER,
            "document",
            "Rasterize a vector asset and import it as a new layer.",
            param_schema(
                {
                    "path": string_field("Vector asset path, typically .svg or .svgz."),
                    "name": string_field("Layer name. Defaults to filename stem."),
                    "x": integer_field("Canvas x offset.", default=0),
                    "y": integer_field("Canvas y offset.", default=0),
                    "width": integer_field("Optional rasterized width.", minimum=1),
                    "height": integer_field("Optional rasterized height.", minimum=1),
                    "opacity": number_field("Layer opacity.", minimum=0, maximum=1, default=1),
                    "blend_mode": BLEND_MODE,
                    "set_active": bool_field("Whether to make the imported layer active.", default=True),
                    "background_color": COLOR,
                },
                required=("path",),
            ),
            target_fields={"output_layer_id": TF.GENERATED},
        ),
        spec(
            ActionType.RASTERIZE_VECTOR_ASSET,
            "document",
            "Rasterize a vector asset to a standalone .png or .npy file.",
            param_schema(
                {
                    "path": string_field("Vector asset path."),
                    "output_path": string_field("Destination .png or .npy path."),
                    "width": integer_field("Optional output width.", minimum=1),
                    "height": integer_field("Optional output height.", minimum=1),
                    "background_color": COLOR,
                },
                required=("path", "output_path"),
            ),
        ),
        spec(
            ActionType.EXPORT_FLAT,
            "document",
            "Export a flattened preview image.",
            param_schema({"path": string_field("Destination image path.")}, required=("path",)),
        ),
        spec(
            ActionType.EXPORT_LAYERED_BUNDLE,
            "document",
            "Export manifest, document summary, layer PNGs, mask PNGs, and preview.",
            param_schema(
                {
                    "path": string_field("Destination directory."),
                    "include_preview": bool_field("Whether to export preview.png.", default=True),
                    "include_hidden": bool_field("Whether to export hidden layers.", default=True),
                    "overwrite": bool_field("Whether to replace an existing bundle directory.", default=True),
                },
                required=("path",),
            ),
        ),
        spec(
            ActionType.RESIZE_CANVAS,
            "document",
            "Resize the canvas around its center.",
            param_schema(
                {
                    "width": integer_field("New canvas width.", minimum=1),
                    "height": integer_field("New canvas height.", minimum=1),
                    "anchor": enum_field(("center",), "Only center anchoring is currently supported.", default="center"),
                    "fill_color": COLOR,
                },
                required=("width", "height"),
            ),
        ),
        spec(
            ActionType.CROP,
            "document",
            "Crop the document, or clear outside a crop on one layer or mask.",
            param_schema(
                {
                    "bbox_xyxy": BBOX,
                    "scope": enum_field(("document", "layer", "mask"), "Crop scope.", default="document"),
                    "fill_color": COLOR,
                },
                required=("bbox_xyxy",),
            ),
            target_fields={"layer_id": TF.OPTIONAL, "mask_id": TF.OPTIONAL},
            notes=("Use target.layer_id when scope is 'layer'; use target.mask_id when scope is 'mask'.",),
        ),
        spec(
            ActionType.CREATE_LAYER,
            "layer",
            "Create a new full-canvas layer.",
            param_schema(
                {
                    "name": string_field("Human-readable layer name."),
                    "kind": enum_field(("raster", "group", "vector", "text", "adjustment"), "Layer kind.", default="raster"),
                    "width": integer_field("Optional layer width; defaults to canvas width.", minimum=1),
                    "height": integer_field("Optional layer height; defaults to canvas height.", minimum=1),
                    "opacity": number_field("Layer opacity.", minimum=0, maximum=1, default=1),
                    "blend_mode": BLEND_MODE,
                    "insert_index": integer_field("Optional stack insertion index.", minimum=0),
                    "set_active": bool_field("Whether the new layer becomes active.", default=True),
                    "color": COLOR,
                    "color_rgba": field("array", "Initial RGBA color as floats in [0, 1].", minItems=4, maxItems=4),
                },
                required=("name",),
            ),
            target_fields={"output_layer_id": TF.GENERATED},
        ),
        spec(ActionType.DELETE_LAYER, "layer", "Delete an existing layer.", target_fields={"layer_id": TF.REQUIRED}),
        spec(
            ActionType.DUPLICATE_LAYER,
            "layer",
            "Duplicate an existing layer.",
            param_schema(
                {
                    "name": string_field("Optional duplicate layer name."),
                    "insert_index": integer_field("Optional stack insertion index.", minimum=0),
                    "set_active": bool_field("Whether duplicate becomes active.", default=True),
                }
            ),
            target_fields={"layer_id": TF.REQUIRED, "output_layer_id": TF.GENERATED},
        ),
        spec(
            ActionType.RENAME_LAYER,
            "layer",
            "Rename an existing layer.",
            param_schema({"name": string_field("New layer name.")}, required=("name",)),
            target_fields={"layer_id": TF.REQUIRED},
        ),
        spec(
            ActionType.REORDER_LAYER,
            "layer",
            "Move a layer to a stack index.",
            param_schema({"index": integer_field("New bottom-to-top stack index.", minimum=0)}, required=("index",)),
            target_fields={"layer_id": TF.REQUIRED},
        ),
        spec(ActionType.SET_ACTIVE_LAYER, "layer", "Set the active layer.", target_fields={"layer_id": TF.REQUIRED}),
        spec(
            ActionType.SET_LAYER_VISIBILITY,
            "layer",
            "Show or hide a layer.",
            param_schema({"visible": bool_field("Layer visibility.")}, required=("visible",)),
            target_fields={"layer_id": TF.REQUIRED},
        ),
        spec(
            ActionType.SET_LAYER_OPACITY,
            "layer",
            "Set a layer's opacity.",
            param_schema({"opacity": number_field("Opacity in [0, 1].", minimum=0, maximum=1)}, required=("opacity",)),
            target_fields={"layer_id": TF.REQUIRED},
        ),
        spec(
            ActionType.SET_BLEND_MODE,
            "layer",
            "Set layer blend-mode metadata.",
            param_schema({"blend_mode": BLEND_MODE}, required=("blend_mode",)),
            target_fields={"layer_id": TF.REQUIRED},
        ),
        spec(
            ActionType.MERGE_LAYERS,
            "layer",
            "Merge layers using normal source-over compositing.",
            param_schema(
                {
                    "mode": enum_field(("down", "visible", "selected", "flatten"), "Merge mode.", default="down"),
                    "layer_ids": ID_LIST,
                    "output_layer_name": string_field("Optional name for generated merged layer."),
                }
            ),
            target_fields={"layer_id": TF.OPTIONAL, "output_layer_id": TF.GENERATED},
            notes=("mode='down' uses target.layer_id; modes visible, selected, and flatten use generated output_layer_id.",),
        ),
        spec(
            ActionType.MOVE_LAYER,
            "transform",
            "Translate a layer's raster pixels.",
            param_schema(
                {
                    "dx": number_field("Horizontal translation in pixels.", default=0),
                    "dy": number_field("Vertical translation in pixels.", default=0),
                    "resample": RESAMPLE,
                    "fill_color": COLOR,
                }
            ),
            target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER},
        ),
        spec(
            ActionType.SCALE_LAYER,
            "transform",
            "Scale a layer around an anchor.",
            param_schema(
                {
                    "scale_x": number_field("Horizontal scale factor.", exclusiveMinimum=0, default=1),
                    "scale_y": number_field("Vertical scale factor. Defaults to scale_x.", exclusiveMinimum=0),
                    "anchor": POINT,
                    "resample": RESAMPLE,
                    "fill_color": COLOR,
                }
            ),
            target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER},
        ),
        spec(
            ActionType.ROTATE_LAYER,
            "transform",
            "Rotate a layer around an anchor.",
            param_schema(
                {
                    "angle_degrees": number_field("Clockwise rotation angle in degrees."),
                    "anchor": POINT,
                    "resample": RESAMPLE,
                    "fill_color": COLOR,
                },
                required=("angle_degrees",),
            ),
            target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER},
        ),
        spec(
            ActionType.FLIP_LAYER,
            "transform",
            "Flip a layer horizontally and/or vertically.",
            param_schema(
                {
                    "horizontal": bool_field("Flip left-right.", default=True),
                    "vertical": bool_field("Flip top-bottom.", default=False),
                    "anchor": POINT,
                    "resample": RESAMPLE,
                    "fill_color": COLOR,
                }
            ),
            target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER},
        ),
        spec(
            ActionType.TRANSFORM_LAYER,
            "transform",
            "Apply a destructive affine transform to a layer.",
            param_schema(
                {
                    "operation": enum_field(("translate", "scale", "rotate", "flip", "affine"), "Transform operation.", default="affine"),
                    "dx": number_field("Translation x."),
                    "dy": number_field("Translation y."),
                    "scale_x": number_field("Scale x."),
                    "scale_y": number_field("Scale y."),
                    "angle_degrees": number_field("Rotation angle."),
                    "horizontal": bool_field("Flip left-right."),
                    "vertical": bool_field("Flip top-bottom."),
                    "anchor": POINT,
                    "matrix": array_field(number_field("Affine coefficient."), "Six-value affine matrix.", minItems=6, maxItems=6),
                    "resample": RESAMPLE,
                    "fill_color": COLOR,
                }
            ),
            target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER},
        ),
        spec(
            ActionType.ALIGN_LAYER,
            "transform",
            "Align nontransparent content within the canvas.",
            param_schema(
                {
                    "horizontal": enum_field(("left", "center", "right", "none"), "Horizontal alignment.", default="none"),
                    "vertical": enum_field(("top", "center", "bottom", "none"), "Vertical alignment.", default="none"),
                    "margin": integer_field("Margin in pixels.", minimum=0, default=0),
                    "fill_color": COLOR,
                }
            ),
            target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER},
        ),
        spec(
            ActionType.ADD_LAYER_MASK,
            "layer_mask",
            "Create and attach a layer-alpha mask.",
            param_schema(
                {
                    "mode": enum_field(("from_selection", "from_alpha", "full", "empty", "from_mask"), "Layer-mask source.", default="from_selection"),
                    "source_mask_id": string_field("Source mask ID when mode is from_mask."),
                    "name": string_field("Generated mask name."),
                    "remove_mask": bool_field("Accepted for API symmetry; ignored when adding.", default=False),
                    "invert": bool_field("Invert mask data before attaching.", default=False),
                }
            ),
            target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER, "mask_id": TF.GENERATED},
        ),
        spec(ActionType.APPLY_LAYER_MASK, "layer_mask", "Bake a layer mask into layer alpha.", target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER}),
        spec(ActionType.REMOVE_LAYER_MASK, "layer_mask", "Detach a layer mask.", target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER}),
        spec(
            ActionType.SELECT_RECT,
            "selection",
            "Create a rectangular selection mask.",
            param_schema(
                {"name": string_field("Mask name."), "bbox_xyxy": BBOX, "set_active": bool_field("Make active selection.", default=True)},
                required=("bbox_xyxy",),
            ),
            target_fields={"mask_id": TF.GENERATED},
        ),
        spec(
            ActionType.SELECT_ELLIPSE,
            "selection",
            "Create an elliptical selection mask.",
            param_schema(
                {"name": string_field("Mask name."), "bbox_xyxy": BBOX, "set_active": bool_field("Make active selection.", default=True)},
                required=("bbox_xyxy",),
            ),
            target_fields={"mask_id": TF.GENERATED},
        ),
        spec(
            ActionType.SELECT_POLYGON,
            "selection",
            "Create a polygon selection mask.",
            param_schema(
                {"name": string_field("Mask name."), "points": POINTS, "closed": bool_field("Close path.", default=True), "kind": MASK_KIND, "set_active": bool_field("Make active selection.", default=True)},
                required=("points",),
            ),
            target_fields={"mask_id": TF.GENERATED},
        ),
        spec(
            ActionType.SELECT_FREEHAND,
            "selection",
            "Create a freehand selection mask from points.",
            param_schema(
                {"name": string_field("Mask name."), "points": POINTS, "closed": bool_field("Close path.", default=True), "kind": MASK_KIND, "set_active": bool_field("Make active selection.", default=True)},
                required=("points",),
            ),
            target_fields={"mask_id": TF.GENERATED},
        ),
        spec(
            ActionType.SELECT_FROM_ALPHA,
            "selection",
            "Create a selection from layer alpha.",
            param_schema(
                {"name": string_field("Mask name."), "threshold": number_field("Alpha threshold.", minimum=0, maximum=1, default=0.01), "kind": MASK_KIND, "set_active": bool_field("Make active selection.", default=True)}
            ),
            target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER, "mask_id": TF.GENERATED},
        ),
        spec(
            ActionType.SELECT_COLOR_RANGE,
            "selection",
            "Create a non-contiguous selection from pixels close to a color or sampled seed colors.",
            param_schema(
                {
                    "name": string_field("Mask name."),
                    "color": COLOR,
                    "seed_points": POINTS,
                    "exclude_seed_points": POINTS,
                    "threshold": number_field("GIMP-style threshold in 0-255 units. Typical fuzzy-select default is 15.", minimum=0, maximum=255),
                    "tolerance": number_field("Legacy normalized tolerance. Use threshold for GIMP-like plans.", minimum=0),
                    "bbox_xyxy": BBOX,
                    "alpha_min": number_field("Minimum source alpha.", minimum=0, maximum=1, default=0),
                    "select_transparent": bool_field("Match GIMP's transparent-region behavior.", default=True),
                    "antialias": bool_field("Return a soft antialiased mask near the threshold boundary.", default=True),
                    "criterion": SELECT_CRITERION,
                    "color_space": enum_field(("rgb", "hsv"), "Color comparison space.", default="rgb"),
                    "hue_tolerance_degrees": number_field("HSV hue tolerance in degrees.", minimum=0, maximum=180),
                    "saturation_tolerance": number_field("HSV saturation tolerance.", minimum=0, maximum=1),
                    "value_tolerance": number_field("HSV value tolerance.", minimum=0, maximum=1),
                    "kind": MASK_KIND,
                    "set_active": bool_field("Make active selection.", default=True),
                },
                description="Use either color or seed_points. RGB/composite matching follows GIMP's max-channel difference, not Euclidean RGB distance.",
            ),
            target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER, "mask_id": TF.GENERATED},
        ),
        spec(
            ActionType.SELECT_BY_COLOR,
            "selection",
            "GIMP-like Select by Color: scan the layer for all pixels close to a clicked or explicit color.",
            param_schema(
                {
                    "name": string_field("Mask name."),
                    "color": COLOR,
                    "seed_points": POINTS,
                    "exclude_seed_points": POINTS,
                    "threshold": number_field("GIMP-style threshold in 0-255 units. Typical default is 15.", minimum=0, maximum=255),
                    "tolerance": number_field("Legacy normalized tolerance. Use threshold for GIMP-like plans.", minimum=0),
                    "bbox_xyxy": BBOX,
                    "alpha_min": number_field("Minimum source alpha.", minimum=0, maximum=1, default=0),
                    "select_transparent": bool_field("Match GIMP's transparent-region behavior.", default=True),
                    "antialias": bool_field("Return a soft antialiased mask near the threshold boundary.", default=True),
                    "criterion": SELECT_CRITERION,
                    "kind": MASK_KIND,
                    "set_active": bool_field("Make active selection.", default=True),
                },
                description="Use this for non-contiguous same-color selection. Use fuzzy_select for connected regions.",
            ),
            target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER, "mask_id": TF.GENERATED},
        ),
        spec(
            ActionType.MAGIC_WAND_SELECT,
            "selection",
            "Create a contiguous GIMP-like fuzzy selection from seed points.",
            param_schema(
                {
                    "name": string_field("Mask name."),
                    "seed_points": POINTS,
                    "clicks": SELECTION_CLICKS,
                    "exclude_seed_points": POINTS,
                    "threshold": number_field("GIMP-style threshold in 0-255 units. Typical default is 15.", minimum=0, maximum=255),
                    "tolerance": number_field("Legacy normalized tolerance. Use threshold for GIMP-like plans.", minimum=0),
                    "bbox_xyxy": BBOX,
                    "alpha_min": number_field("Minimum source alpha.", minimum=0, maximum=1, default=0),
                    "diagonal": bool_field("Allow diagonal connectivity.", default=False),
                    "select_transparent": bool_field("Match GIMP's transparent-region behavior.", default=True),
                    "antialias": bool_field("Return a soft antialiased mask near the threshold boundary.", default=True),
                    "criterion": SELECT_CRITERION,
                    "color_space": enum_field(("rgb", "hsv"), "Color comparison space.", default="rgb"),
                    "hue_tolerance_degrees": number_field("HSV hue tolerance in degrees.", minimum=0, maximum=180),
                    "saturation_tolerance": number_field("HSV saturation tolerance.", minimum=0, maximum=1),
                    "value_tolerance": number_field("HSV value tolerance.", minimum=0, maximum=1),
                    "kind": MASK_KIND,
                    "set_active": bool_field("Make active selection.", default=True),
                },
                description="This replaces edge-threshold heuristics with GIMP-style color similarity to the clicked seed color.",
            ),
            target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER, "mask_id": TF.GENERATED},
        ),
        spec(
            ActionType.FUZZY_SELECT,
            "selection",
            "GIMP-like Fuzzy Select: select connected regions from one or more meaningful clicks.",
            param_schema(
                {
                    "name": string_field("Mask name."),
                    "seed_points": POINTS,
                    "clicks": SELECTION_CLICKS,
                    "exclude_seed_points": POINTS,
                    "threshold": number_field("GIMP-style threshold in 0-255 units. Typical default is 15.", minimum=0, maximum=255),
                    "tolerance": number_field("Legacy normalized tolerance. Use threshold for GIMP-like plans.", minimum=0),
                    "bbox_xyxy": BBOX,
                    "alpha_min": number_field("Minimum source alpha.", minimum=0, maximum=1, default=0),
                    "diagonal": bool_field("Allow diagonal connectivity.", default=False),
                    "select_transparent": bool_field("Match GIMP's transparent-region behavior.", default=True),
                    "antialias": bool_field("Return a soft antialiased mask near the threshold boundary.", default=True),
                    "criterion": SELECT_CRITERION,
                    "kind": MASK_KIND,
                    "set_active": bool_field("Make active selection.", default=True),
                },
                description="Use this for tasks like: fuzzy select these clicked background regions with threshold 15, then clear.",
            ),
            target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER, "mask_id": TF.GENERATED},
        ),
        spec(
            ActionType.SAVE_SELECTION_AS_MASK,
            "selection",
            "Copy active selection or source mask to a reusable mask.",
            param_schema(
                {
                    "source_mask_id": string_field("Optional source mask. Defaults to active selection."),
                    "name": string_field("Output mask name."),
                    "kind": MASK_KIND,
                    "set_active": bool_field("Make output active.", default=False),
                }
            ),
            target_fields={"mask_id": TF.GENERATED},
        ),
        spec(
            ActionType.CREATE_MASK_FROM_SHAPE,
            "mask",
            "Create a mask from rectangle or ellipse geometry.",
            param_schema(
                {"name": string_field("Mask name."), "kind": MASK_KIND, "shape": SHAPE, "set_active": bool_field("Make active selection.", default=False)},
                required=("shape",),
            ),
            target_fields={"mask_id": TF.GENERATED},
        ),
        spec(
            ActionType.GROW_MASK,
            "mask",
            "Grow a mask by a pixel radius.",
            param_schema(
                {"source_mask_id": string_field("Existing source mask ID."), "pixels": integer_field("Growth radius.", minimum=0), "name": string_field("Output mask name."), "set_active": bool_field("Make output active.", default=False)},
                required=("source_mask_id", "pixels"),
            ),
            target_fields={"mask_id": TF.GENERATED},
        ),
        spec(
            ActionType.SHRINK_MASK,
            "mask",
            "Shrink a mask by a pixel radius.",
            param_schema(
                {"source_mask_id": string_field("Existing source mask ID."), "pixels": integer_field("Shrink radius.", minimum=0), "name": string_field("Output mask name."), "set_active": bool_field("Make output active.", default=False)},
                required=("source_mask_id", "pixels"),
            ),
            target_fields={"mask_id": TF.GENERATED},
        ),
        spec(
            ActionType.FEATHER_MASK,
            "mask",
            "Create a softened copy of a mask.",
            param_schema(
                {"source_mask_id": string_field("Existing source mask ID."), "radius": number_field("Gaussian blur radius.", minimum=0), "name": string_field("Output mask name.")},
                required=("source_mask_id", "radius"),
            ),
            target_fields={"mask_id": TF.GENERATED},
        ),
        spec(
            ActionType.INVERT_MASK,
            "mask",
            "Invert a mask.",
            param_schema(
                {"source_mask_id": string_field("Existing source mask ID."), "name": string_field("Output mask name."), "set_active": bool_field("Make output active.", default=False)},
                required=("source_mask_id",),
            ),
            target_fields={"mask_id": TF.GENERATED},
        ),
        spec(
            ActionType.COMBINE_MASKS,
            "mask",
            "Combine masks with union, intersect, or subtract.",
            param_schema(
                {"operation": enum_field(("union", "intersect", "subtract"), "Mask combine operation."), "mask_ids": ID_LIST, "name": string_field("Output mask name.")},
                required=("operation", "mask_ids"),
            ),
            target_fields={"mask_id": TF.GENERATED},
        ),
        spec(
            ActionType.REFINE_SELECTION,
            "mask",
            "Threshold, grow, shrink, fill holes, smooth, feather, or clean a mask.",
            param_schema(
                {
                    "source_mask_id": string_field("Existing source mask ID."),
                    "name": string_field("Output mask name."),
                    "threshold": number_field("Optional hard threshold.", minimum=0, maximum=1),
                    "feather_radius": number_field("Optional feather radius.", minimum=0),
                    "grow_pixels": integer_field("Optional grow radius.", minimum=0),
                    "shrink_pixels": integer_field("Optional shrink radius.", minimum=0),
                    "close_pixels": integer_field("Optional closing radius to bridge small gaps.", minimum=0),
                    "open_pixels": integer_field("Optional opening radius to remove thin noise.", minimum=0),
                    "min_area": integer_field("Optional minimum connected area.", minimum=0),
                    "fill_holes": bool_field("Fill enclosed holes before smoothing/feathering.", default=False),
                    "max_hole_area": integer_field("When filling holes, only fill holes up to this area.", minimum=0),
                    "smooth_radius": number_field("Optional hard-edge smoothing radius before feathering.", minimum=0),
                    "set_active": bool_field("Make output active.", default=False),
                },
                required=("source_mask_id",),
            ),
            target_fields={"mask_id": TF.GENERATED},
        ),
        spec(
            ActionType.REMOVE_SMALL_ISLANDS,
            "mask",
            "Remove small connected mask components.",
            param_schema(
                {"source_mask_id": string_field("Existing source mask ID."), "name": string_field("Output mask name."), "min_area": integer_field("Minimum component area.", minimum=0), "set_active": bool_field("Make output active.", default=False)},
                required=("source_mask_id", "min_area"),
            ),
            target_fields={"mask_id": TF.GENERATED},
        ),
        spec(
            ActionType.FILL_MASK_HOLES,
            "mask",
            "Fill enclosed holes in a mask.",
            param_schema(
                {"source_mask_id": string_field("Existing source mask ID."), "name": string_field("Output mask name."), "set_active": bool_field("Make output active.", default=False)},
                required=("source_mask_id",),
            ),
            target_fields={"mask_id": TF.GENERATED},
        ),
        spec(
            ActionType.CLEANUP_FRINGE,
            "mask",
            "Extend a mask into nearby old-material antialias/fringe pixels while preserving protected masks.",
            param_schema(
                {
                    "source_mask_id": string_field("Core object mask ID to extend."),
                    "name": string_field("Output mask name."),
                    "search_radius": integer_field("Pixel radius around the core mask to search for missed fringe.", minimum=0, default=2),
                    "old_colors": COLOR_LIST,
                    "seed_points": POINTS,
                    "protect_mask_ids": ID_LIST,
                    "source_threshold": number_field("Threshold for treating source mask pixels as core.", minimum=0, maximum=1, default=0.5),
                    "protect_threshold": number_field("Protected-mask values above this are excluded.", minimum=0, maximum=1, default=0.1),
                    "include_source_mask": bool_field("Union the fringe with the source mask.", default=True),
                    "bbox_xyxy": BBOX,
                    "alpha_min": number_field("Minimum source alpha.", minimum=0, maximum=1, default=0.0),
                    "color_space": enum_field(("rgb", "hsv"), "Color distance space.", default="hsv"),
                    "tolerance": number_field("RGB tolerance, or fallback HSV tolerance when explicit HSV tolerances are omitted.", minimum=0),
                    "hue_tolerance_degrees": number_field("HSV hue tolerance for old-material fringe matching.", minimum=0, maximum=180, default=35),
                    "saturation_tolerance": number_field("HSV saturation tolerance.", minimum=0, maximum=1, default=0.65),
                    "value_tolerance": number_field("HSV value tolerance.", minimum=0, maximum=1, default=0.85),
                    "feather_radius": number_field("Optional softening radius for the output mask.", minimum=0),
                    "set_active": bool_field("Make output active.", default=False),
                },
                required=("source_mask_id",),
            ),
            target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER, "mask_id": TF.GENERATED},
            notes=(
                "Use after color-based masks for flattened 2D art when old-color edge pixels were missed.",
                "Pass an extract_line_art output in protect_mask_ids before recoloring hair, irises, bows, or clothing near ink lines.",
            ),
        ),
        spec(
            ActionType.DRAW_SHAPE,
            "paint",
            "Draw a deterministic rectangle or ellipse.",
            param_schema({"shape": SHAPE, "stroke": STROKE, "fill": FILL}, required=("shape",)),
            target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER},
            write_mask=TF.GENERATED,
            notes=("At least one of params.stroke or params.fill is required.",),
        ),
        spec(
            ActionType.DRAW_PATH,
            "paint",
            "Stroke a path on a layer.",
            param_schema(
                {"points": POINTS, "color": COLOR, "width": number_field("Brush width.", exclusiveMinimum=0), "opacity": number_field("Opacity.", minimum=0, maximum=1), "mode": enum_field(("source_over", "replace_rgba", "alpha_to_zero"), "Paint mode."), "hardness": number_field("Brush hardness.", minimum=0, maximum=1), "spacing": number_field("Brush spacing.", exclusiveMinimum=0), "closed": bool_field("Close path.")},
                required=("points", "color"),
            ),
            target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER},
            write_mask=TF.GENERATED,
        ),
        spec(
            ActionType.BRUSH_STROKE,
            "paint",
            "Paint a brush stroke.",
            param_schema(
                {"points": POINTS, "color": COLOR, "width": number_field("Brush width.", exclusiveMinimum=0), "opacity": number_field("Opacity.", minimum=0, maximum=1), "mode": enum_field(("source_over", "replace_rgba", "alpha_to_zero"), "Paint mode."), "hardness": number_field("Brush hardness.", minimum=0, maximum=1), "spacing": number_field("Brush spacing.", exclusiveMinimum=0), "closed": bool_field("Close path.")},
                required=("points", "color"),
            ),
            target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER},
            write_mask=TF.GENERATED,
        ),
        spec(
            ActionType.ERASE_STROKE,
            "paint",
            "Erase alpha along a brush stroke.",
            param_schema(
                {"points": POINTS, "width": number_field("Brush width.", exclusiveMinimum=0), "opacity": number_field("Opacity.", minimum=0, maximum=1), "mode": enum_field(("source_over", "replace_rgba", "alpha_to_zero"), "Erase mode."), "hardness": number_field("Brush hardness.", minimum=0, maximum=1), "spacing": number_field("Brush spacing.", exclusiveMinimum=0), "closed": bool_field("Close path.")},
                required=("points",),
            ),
            target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER},
            write_mask=TF.GENERATED,
        ),
        spec(
            ActionType.PAINT_BUCKET_FILL,
            "paint",
            "Fill the write mask with a color.",
            param_schema(
                {"color": COLOR, "mode": enum_field(("replace_rgb_preserve_alpha", "replace_rgba", "source_over"), "Fill mode.", default="replace_rgb_preserve_alpha")},
                required=("color",),
            ),
            target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER},
            write_mask=TF.GENERATED,
        ),
        spec(
            ActionType.GRADIENT_FILL,
            "paint",
            "Fill the write mask with a linear or radial gradient.",
            param_schema(
                {
                    "type": enum_field(("linear", "radial"), "Gradient type.", default="linear"),
                    "start": POINT,
                    "end": POINT,
                    "center": POINT,
                    "radius": number_field("Radial gradient radius.", exclusiveMinimum=0),
                    "colors": array_field(COLOR, "Gradient colors.", minItems=2),
                    "mode": enum_field(("replace_rgb_preserve_alpha", "replace_rgba", "source_over"), "Fill mode."),
                },
                required=("colors",),
            ),
            target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER},
            write_mask=TF.GENERATED,
            notes=("Linear gradients require start and end. Radial gradients require center and radius.",),
        ),
        spec(
            ActionType.PATTERN_FILL,
            "paint",
            "Fill the write mask with a procedural or image pattern.",
            param_schema(
                {
                    "pattern": enum_field(("checkerboard", "stripes", "image"), "Pattern type.", default="checkerboard"),
                    "colors": array_field(COLOR, "Pattern colors.", minItems=1),
                    "cell_size": integer_field("Cell size.", minimum=1),
                    "stripe_width": integer_field("Stripe width.", minimum=1),
                    "angle_degrees": number_field("Stripe angle."),
                    "mode": enum_field(("replace_rgb_preserve_alpha", "replace_rgba", "source_over"), "Fill mode."),
                    "path": string_field("Image pattern path when pattern is image."),
                }
            ),
            target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER},
            write_mask=TF.GENERATED,
        ),
        spec(
            ActionType.BLUR_REGION,
            "filter",
            "Apply Gaussian blur through a write mask.",
            param_schema(
                {"radius": number_field("Blur radius.", minimum=0), "channels": CHANNELS, "edge_mode": enum_field(("reflect", "constant", "nearest", "mirror", "wrap"), "Boundary mode.", default="nearest")},
                required=("radius",),
            ),
            target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER},
            write_mask=TF.GENERATED,
        ),
        spec(ActionType.SHARPEN_REGION, "filter", "Sharpen RGB pixels.", target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER}, write_mask=TF.GENERATED),
        spec(ActionType.NOISE_REDUCE, "filter", "Denoise selected RGB pixels.", target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER}, write_mask=TF.GENERATED),
        spec(ActionType.MEDIAN_FILTER, "filter", "Apply a median filter.", target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER}, write_mask=TF.GENERATED),
        spec(ActionType.EDGE_DETECT, "filter", "Detect luminance edges.", target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER}, write_mask=TF.GENERATED),
        spec(ActionType.DROP_SHADOW, "filter", "Create a blurred offset shadow layer.", target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER}),
        spec(ActionType.STROKE_SELECTION, "filter", "Paint an outline around a selection mask.", target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER}, write_mask=TF.GENERATED),
        spec(
            ActionType.CLEAR_REGION,
            "paint",
            "Clear pixels or alpha inside the write mask.",
            param_schema({"mode": enum_field(("alpha_to_zero", "rgba_to_zero"), "Clear mode.", default="alpha_to_zero"), "preserve_rgb": bool_field("Accepted compatibility flag.", default=False)}),
            target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER},
            write_mask=TF.GENERATED,
        ),
        spec(ActionType.CUT, "clipboard", "Copy a masked region and clear it.", target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER}, write_mask=TF.GENERATED),
        spec(ActionType.COPY, "clipboard", "Copy a masked region.", target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER}, write_mask=TF.OPTIONAL),
        spec(ActionType.PASTE, "clipboard", "Paste clipboard pixels as a new layer.", target_fields={"output_layer_id": TF.GENERATED}),
        spec(ActionType.PASTE_AS_NEW_LAYER, "clipboard", "Paste clipboard pixels as a new layer.", target_fields={"output_layer_id": TF.GENERATED}),
        spec(
            ActionType.DUPLICATE_REGION_TO_LAYER,
            "clipboard",
            "Copy a source region directly into a new layer.",
            target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER, "output_layer_id": TF.GENERATED},
            write_mask=TF.OPTIONAL,
        ),
        spec(ActionType.ADJUST_BRIGHTNESS_CONTRAST, "color", "Adjust brightness and contrast.", target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER}, write_mask=TF.GENERATED),
        spec(ActionType.ADJUST_HUE_SATURATION, "color", "Adjust hue, saturation, and lightness.", target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER}, write_mask=TF.GENERATED),
        spec(ActionType.ADJUST_LEVELS, "color", "Apply simple RGB levels.", target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER}, write_mask=TF.GENERATED),
        spec(
            ActionType.ADJUST_CURVES,
            "color",
            "Apply a piecewise-linear RGB curve.",
            param_schema({"points": array_field(POINT, "Curve points as [input, output].", minItems=2)}, required=("points",)),
            target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER},
            write_mask=TF.GENERATED,
        ),
        spec(
            ActionType.COLORIZE,
            "color",
            "Colorize pixels with luminance or hue-preserving methods.",
            param_schema(
                {
                    "color": COLOR,
                    "amount": number_field("Colorize amount from 0 to 1.", minimum=0, maximum=1),
                    "method": enum_field(("gimp", "luminance", "set_hue_preserve_lightness", "set_hue_preserve_value", "material_hsl"), "Colorize method. Use gimp for GIMP-like Colorize.", default="gimp"),
                },
                required=("color",),
            ),
            target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER},
            write_mask=TF.GENERATED,
        ),
        spec(ActionType.REPLACE_COLOR, "color", "Replace pixels close to a source color.", target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER}, write_mask=TF.GENERATED),
        spec(ActionType.DESATURATE, "color", "Move RGB pixels toward grayscale.", target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER}, write_mask=TF.GENERATED),
        spec(
            ActionType.CREATE_TEXT_LAYER,
            "text",
            "Create a rasterized editable text layer.",
            param_schema(_text_param_properties(), required=("text",)),
            target_fields={"output_layer_id": TF.GENERATED},
        ),
        spec(ActionType.EDIT_TEXT_LAYER, "text", "Edit text layer metadata and rerender it.", target_fields={"layer_id": TF.REQUIRED}),
        spec(ActionType.RASTERIZE_TEXT_LAYER, "text", "Convert a text layer to a raster layer.", target_fields={"layer_id": TF.REQUIRED}),
        spec(ActionType.DETECT_SHAPE, "perception", "Detect coarse geometric shape from alpha.", target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER}),
        spec(ActionType.DETECT_OBJECTS, "perception", "Detect connected alpha components as objects.", target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER}),
        spec(ActionType.SEGMENT_OBJECT, "perception", "Create an object mask from alpha or seed points.", target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER, "mask_id": TF.GENERATED}),
        spec(ActionType.ESTIMATE_DEPTH, "perception", "Create a luminance-based depth proxy mask.", target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER, "mask_id": TF.GENERATED}),
        spec(
            ActionType.EXTRACT_LINE_ART,
            "perception",
            "Create a line-art mask from dark ink pixels or luminance edges.",
            param_schema(
                {
                    "mode": enum_field(("edges", "ink", "dark_pixels"), "Extraction mode. Use ink for anime/comic line art.", default="edges"),
                    "threshold": number_field("Edge magnitude threshold for edges mode, or maximum luminance for ink/dark-pixels mode. For anime ink, typical useful values are 0.12 to 0.18.", minimum=0, maximum=1),
                    "contrast_threshold": number_field("Optional local luminance contrast threshold for ink/dark-pixels mode.", minimum=0, maximum=1),
                    "alpha_min": number_field("Minimum source alpha.", minimum=0, maximum=1, default=0.0),
                    "source_mask_id": string_field("Optional mask limiting where line art can be detected."),
                    "bbox_xyxy": BBOX,
                    "grow_pixels": integer_field("Optional radius to expand detected line art for protection.", minimum=0),
                    "feather_radius": number_field("Optional softening radius for the line-art mask.", minimum=0),
                    "min_area": integer_field("Optional minimum connected line-art component area.", minimum=0),
                    "name": string_field("Output mask name."),
                    "set_active": bool_field("Make output active.", default=True),
                }
            ),
            target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER, "mask_id": TF.GENERATED},
            notes=(
                "For flattened anime-style recolors, use mode='ink' with a small grow_pixels value to protect black outlines before expanding or recoloring masks.",
            ),
        ),
        spec(ActionType.DECOMPOSE_TO_LAYERS, "perception", "Split alpha components into layers.", target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER, "output_layer_id": TF.GENERATED}),
        spec(ActionType.TXT2IMG_TO_LAYER, "diffusion", "Call txt2img backend and import result as a layer.", target_fields={"output_layer_id": TF.GENERATED}),
        spec(ActionType.IMG2IMG_TO_LAYER, "diffusion", "Call img2img backend using a source layer.", target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER, "output_layer_id": TF.GENERATED}),
        spec(ActionType.INPAINT_REGION, "diffusion", "Call inpainting backend for a masked region.", target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER, "output_layer_id": TF.GENERATED}, write_mask=TF.GENERATED),
        spec(ActionType.OUTPAINT_REGION, "diffusion", "Call outpainting backend for a masked region.", target_fields={"layer_id": TF.DEFAULT_ACTIVE_LAYER, "output_layer_id": TF.GENERATED}, write_mask=TF.GENERATED),
        spec(ActionType.VALIDATE, "meta", "Run document validation without mutation."),
        spec(ActionType.NO_OP, "meta", "Perform no document mutation."),
    ]
}


_FILTER_PARAMS = param_schema(
    {
        "radius": number_field("Filter radius.", minimum=0),
        "amount": number_field("Filter amount.", minimum=0),
        "threshold": number_field("Filter threshold.", minimum=0),
        "channels": CHANNELS,
        "mode": enum_field(("replace_rgb_preserve_alpha", "replace_rgba", "source_over", "alpha", "luminance"), "Filter mode."),
        "color": COLOR,
        "offset": POINT,
        "blur_radius": number_field("Shadow blur radius.", minimum=0),
        "opacity": number_field("Opacity.", minimum=0),
        "source_mask_id": string_field("Optional source mask ID."),
        "output_layer_name": string_field("Optional generated output layer name."),
    }
)
_COLOR_PARAMS = param_schema(
    {
        "brightness": number_field("Brightness adjustment."),
        "contrast": number_field("Contrast adjustment."),
        "hue_degrees": number_field("Hue rotation in degrees."),
        "saturation": number_field("Saturation multiplier or delta."),
        "lightness": number_field("Lightness adjustment."),
        "gamma": number_field("Gamma adjustment."),
        "in_black": number_field("Input black point."),
        "in_white": number_field("Input white point."),
        "out_black": number_field("Output black point."),
        "out_white": number_field("Output white point."),
        "color": COLOR,
        "source_color": COLOR,
        "target_color": COLOR,
        "tolerance": number_field("Normalized RGB color-distance tolerance. Typical useful range is 0.03 to 0.25; do not use 0-255 units.", minimum=0, maximum=1),
        "softness": number_field("Normalized softness around the color tolerance. Typical useful range is 0.01 to 0.2.", minimum=0, maximum=1),
        "method": enum_field(("gimp", "luminance", "average", "lightness", "set_hue_preserve_lightness", "set_hue_preserve_value", "material_hsl"), "Adjustment method."),
        "amount": number_field("Adjustment amount. For colorize/desaturate, use 0 to 1.", minimum=0, maximum=1),
    }
)
_CLIPBOARD_PARAMS = param_schema(
    {
        "source_mask_id": string_field("Optional source mask ID."),
        "bbox_xyxy": BBOX,
        "x": integer_field("Paste x coordinate."),
        "y": integer_field("Paste y coordinate."),
        "name": string_field("Generated layer name."),
        "clear_mode": enum_field(("alpha_to_zero", "rgba_to_zero"), "Cut clear mode."),
        "preserve_rgb": bool_field("Preserve RGB when clearing alpha."),
        "set_active": bool_field("Make output active.", default=True),
    }
)
_PERCEPTION_PARAMS = param_schema(
    {
        "threshold": number_field("Detection threshold.", minimum=0, maximum=1),
        "alpha_min": number_field("Minimum alpha.", minimum=0, maximum=1),
        "tolerance": number_field("Detection tolerance.", minimum=0, maximum=1),
        "seed_points": POINTS,
        "positive_seed_points": POINTS,
        "negative_seed_points": POINTS,
        "bbox_xyxy": BBOX,
        "diagonal": bool_field("Allow diagonal connectivity for seeded object selection.", default=True),
        "color_space": enum_field(("rgb", "hsv"), "Color comparison space for seeded object selection.", default="rgb"),
        "hue_tolerance_degrees": number_field("HSV hue tolerance in degrees.", minimum=0, maximum=180),
        "saturation_tolerance": number_field("HSV saturation tolerance.", minimum=0, maximum=1),
        "value_tolerance": number_field("HSV value tolerance.", minimum=0, maximum=1),
        "edge_stop_threshold": number_field("Optional normalized local RGB edge threshold for seeded object selection.", minimum=0),
        "negative_margin": number_field("Optional extra distance margin against negative seed colors.", minimum=0),
        "mode": enum_field(("alpha", "luminance", "color", "edges", "seeded_object"), "Perception mode."),
        "name": string_field("Generated mask name."),
        "set_active": bool_field("Make generated mask active.", default=True),
        "output_layer_name": string_field("Generated layer name."),
        "min_area": integer_field("Minimum component area.", minimum=0),
        "max_objects": integer_field("Maximum object count.", minimum=1),
    }
)
_DIFFUSION_PARAMS = param_schema(
    {
        "prompt": string_field("Generation prompt."),
        "negative_prompt": string_field("Negative prompt."),
        "seed": integer_field("Random seed."),
        "denoise": number_field("Denoise strength.", minimum=0, maximum=1),
        "guidance_scale": number_field("Guidance scale.", minimum=0),
        "steps": integer_field("Sampling step count.", minimum=1),
        "backend": string_field("Backend name."),
        "job": field("object", "Backend-specific job payload."),
        "output_layer_name": string_field("Generated layer name."),
        "mode": enum_field(("replace_region", "new_layer"), "How to integrate the generated image."),
        "padding": field("integer", "Context padding in pixels around the write-mask bbox."),
    }
)
_TEXT_EDIT_PARAMS = param_schema(
    _text_param_properties()
)


def _replace_params(specification: ActionToolSpec, params_schema: JsonObject) -> ActionToolSpec:
    """Return `specification` with a different params schema."""
    return ActionToolSpec(
        name=specification.name,
        category=specification.category,
        summary=specification.summary,
        params_schema=params_schema,
        target_fields=specification.target_fields,
        write_mask=specification.write_mask,
        kernel_filled_fields=specification.kernel_filled_fields,
        notes=specification.notes,
    )


for _name in (
    ActionType.SHARPEN_REGION.value,
    ActionType.NOISE_REDUCE.value,
    ActionType.MEDIAN_FILTER.value,
    ActionType.EDGE_DETECT.value,
    ActionType.DROP_SHADOW.value,
    ActionType.STROKE_SELECTION.value,
):
    ACTION_TOOL_SPECS[_name] = _replace_params(ACTION_TOOL_SPECS[_name], _FILTER_PARAMS)

for _name in (
    ActionType.ADJUST_BRIGHTNESS_CONTRAST.value,
    ActionType.ADJUST_HUE_SATURATION.value,
    ActionType.ADJUST_LEVELS.value,
    ActionType.COLORIZE.value,
    ActionType.REPLACE_COLOR.value,
    ActionType.DESATURATE.value,
):
    ACTION_TOOL_SPECS[_name] = _replace_params(ACTION_TOOL_SPECS[_name], _COLOR_PARAMS)

for _name in (
    ActionType.CUT.value,
    ActionType.COPY.value,
    ActionType.PASTE.value,
    ActionType.PASTE_AS_NEW_LAYER.value,
    ActionType.DUPLICATE_REGION_TO_LAYER.value,
):
    ACTION_TOOL_SPECS[_name] = _replace_params(ACTION_TOOL_SPECS[_name], _CLIPBOARD_PARAMS)

for _name in (
    ActionType.EDIT_TEXT_LAYER.value,
    ActionType.RASTERIZE_TEXT_LAYER.value,
):
    ACTION_TOOL_SPECS[_name] = _replace_params(ACTION_TOOL_SPECS[_name], _TEXT_EDIT_PARAMS)

for _name in (
    ActionType.DETECT_SHAPE.value,
    ActionType.DETECT_OBJECTS.value,
    ActionType.SEGMENT_OBJECT.value,
    ActionType.ESTIMATE_DEPTH.value,
    ActionType.DECOMPOSE_TO_LAYERS.value,
):
    ACTION_TOOL_SPECS[_name] = _replace_params(ACTION_TOOL_SPECS[_name], _PERCEPTION_PARAMS)

for _name in (
    ActionType.TXT2IMG_TO_LAYER.value,
    ActionType.IMG2IMG_TO_LAYER.value,
    ActionType.INPAINT_REGION.value,
    ActionType.OUTPAINT_REGION.value,
):
    ACTION_TOOL_SPECS[_name] = _replace_params(ACTION_TOOL_SPECS[_name], _DIFFUSION_PARAMS)


def _has_required_params(schema: JsonObject) -> bool:
    """Return whether a params object contains required model fields."""
    required = schema.get("required")
    return isinstance(required, list) and len(required) > 0


def _target_field_description(field_name: str, mode: TargetFieldMode) -> str:
    """Describe one target field for planner prompts."""
    descriptions = {
        "document_id": "Document ID. Usually generated or preserved by the kernel.",
        "layer_id": "Existing target layer ID.",
        "layer_name": "Human layer name. IDs are preferred when uniqueness matters.",
        "mask_id": "Output or target mask ID.",
        "selection_id": "Selection ID. Usually represented by mask_id in this prototype.",
        "output_layer_id": "Output layer ID.",
    }
    mode_text = {
        TargetFieldMode.REQUIRED: "Required from the planner.",
        TargetFieldMode.OPTIONAL: "Optional.",
        TargetFieldMode.GENERATED: "Optional; generated by the kernel when omitted.",
        TargetFieldMode.DEFAULT_ACTIVE_LAYER: "Optional; defaults to the active layer when omitted.",
        TargetFieldMode.DEFAULT_ACTIVE_SELECTION: "Optional; defaults to the active selection when omitted.",
    }[TargetFieldMode(mode)]
    return f"{descriptions.get(field_name, field_name)} {mode_text}"


def _write_mask_description(mode: TargetFieldMode) -> str:
    """Describe write-mask handling for planner prompts."""
    if TargetFieldMode(mode) == TargetFieldMode.REQUIRED:
        return "Required write mask ID limiting where pixels may change."
    if TargetFieldMode(mode) == TargetFieldMode.GENERATED:
        return "Optional; defaults to an existing or generated full-canvas write mask when omitted."
    return "Optional write mask ID."
