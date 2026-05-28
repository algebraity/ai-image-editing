"""Typed text rendering parameters.

Actions still carry JSON dictionaries, but text rendering needs a clearer
internal contract than a loose params bag. These dataclasses normalize both the
legacy flat action fields and the newer structured `font`, `style`, `layout`,
and `outline` objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


JsonObject = dict[str, Any]
RGBA = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class TextOutline:
    """Stroke drawn around glyphs before the fill is composited."""

    color: RGBA = (0.0, 0.0, 0.0, 1.0)
    width: float = 0.0

    def to_json(self) -> JsonObject:
        return {"color_rgba": list(self.color), "width": self.width}


@dataclass(frozen=True, slots=True)
class TextFontSpec:
    """Font request used by the renderer."""

    id: Optional[str] = None
    path: Optional[str] = None
    family: Optional[str] = None
    style: Optional[str] = None
    weight: Optional[int] = None
    size: int = 32

    def to_json(self) -> JsonObject:
        return _drop_none(
            {
                "id": self.id,
                "path": self.path,
                "family": self.family,
                "style": self.style,
                "weight": self.weight,
                "size": self.size,
            }
        )


@dataclass(frozen=True, slots=True)
class TextStyle:
    """Visual styling applied to text glyphs."""

    color: RGBA = (0.0, 0.0, 0.0, 1.0)
    outline: TextOutline = field(default_factory=TextOutline)

    def to_json(self) -> JsonObject:
        return {"color_rgba": list(self.color), "outline": self.outline.to_json()}


@dataclass(frozen=True, slots=True)
class TextLayout:
    """Canvas placement and multiline layout settings."""

    x: int = 0
    y: int = 0
    anchor: Optional[str] = None
    align: str = "left"
    spacing: int = 0

    def to_json(self) -> JsonObject:
        return _drop_none(
            {
                "x": self.x,
                "y": self.y,
                "anchor": self.anchor,
                "align": self.align,
                "spacing": self.spacing,
            }
        )


@dataclass(frozen=True, slots=True)
class TextRenderRequest:
    """Complete normalized input for text rendering."""

    text: str
    name: str = "Text"
    font: TextFontSpec = field(default_factory=TextFontSpec)
    style: TextStyle = field(default_factory=TextStyle)
    layout: TextLayout = field(default_factory=TextLayout)
    set_active: bool = True
    raw_params: JsonObject = field(default_factory=dict)

    @classmethod
    def from_params(cls, params: JsonObject) -> "TextRenderRequest":
        """Normalize legacy and structured action params."""
        if not isinstance(params, dict):
            raise TypeError("text params must be a dictionary")

        text = _string(params.get("text", ""), "text")
        name = _optional_string(params.get("name"), "name") or "Text"
        font = _font_spec(params)
        outline = _outline(params)
        style_data = params.get("style") if isinstance(params.get("style"), dict) else {}
        style_color = style_data.get("color", style_data.get("color_rgba", "#000000"))
        style = TextStyle(color=_color(params.get("color", style_color)), outline=outline)
        layout = _layout(params)
        return cls(
            text=text,
            name=name,
            font=font,
            style=style,
            layout=layout,
            set_active=bool(params.get("set_active", True)),
            raw_params=dict(params),
        )

    def to_json(self) -> JsonObject:
        """Return reproducible structured text metadata."""
        return {
            "text": self.text,
            "name": self.name,
            "font": self.font.to_json(),
            "style": self.style.to_json(),
            "layout": self.layout.to_json(),
            "set_active": self.set_active,
        }


@dataclass(frozen=True, slots=True)
class TextRenderResult:
    """Rendered pixels plus text metadata stored on the layer."""

    pixels: Any
    metadata: JsonObject


def merge_text_params(existing: JsonObject, updates: JsonObject) -> JsonObject:
    """Merge edit params into existing normalized text metadata.

    Flat compatibility fields such as `font_size` and `stroke_width` are mapped
    into the structured objects so partial text edits override the old values.
    """
    merged = _base_params(existing)
    _apply_update_mapping(merged, updates)
    return merged


def _base_params(existing: JsonObject) -> JsonObject:
    if not isinstance(existing, dict):
        return {}
    if "font" in existing or "style" in existing or "layout" in existing:
        return {
            "text": existing.get("text", ""),
            "name": existing.get("name", "Text"),
            "font": dict(existing.get("font", {})),
            "style": dict(existing.get("style", {})),
            "layout": dict(existing.get("layout", {})),
            "set_active": existing.get("set_active", True),
        }
    return dict(existing)


def _apply_update_mapping(target: JsonObject, updates: JsonObject) -> None:
    if not isinstance(updates, dict):
        raise TypeError("text update params must be a dictionary")
    target.setdefault("font", {})
    target.setdefault("style", {})
    target.setdefault("layout", {})
    for key, value in updates.items():
        if key == "font":
            target["font"].update(value)
        elif key == "style":
            target["style"].update(value)
        elif key == "layout":
            target["layout"].update(value)
        elif key in {"font_id", "font_path", "font_family", "font_style", "font_weight", "font_size"}:
            field_name = {
                "font_id": "id",
                "font_path": "path",
                "font_family": "family",
                "font_style": "style",
                "font_weight": "weight",
                "font_size": "size",
            }[key]
            target["font"][field_name] = value
            target[key] = value
        elif key in {"x", "y", "anchor", "align", "spacing"}:
            target["layout"][key] = value
            target[key] = value
        elif key in {"color", "outline", "stroke_color", "stroke_width", "outline_color", "outline_width"}:
            if key == "outline":
                target["style"]["outline"] = value
            elif key in {"stroke_color", "outline_color"}:
                outline = dict(target["style"].get("outline", {}))
                outline["color"] = value
                target["style"]["outline"] = outline
            elif key in {"stroke_width", "outline_width"}:
                outline = dict(target["style"].get("outline", {}))
                outline["width"] = value
                target["style"]["outline"] = outline
            else:
                target["style"][key] = value
            target[key] = value
        else:
            target[key] = value


def _font_spec(params: JsonObject) -> TextFontSpec:
    font = dict(params.get("font", {})) if isinstance(params.get("font"), dict) else {}
    return TextFontSpec(
        id=_optional_string(params.get("font_id", font.get("id")), "font.id"),
        path=_optional_string(params.get("font_path", font.get("path")), "font.path"),
        family=_optional_string(params.get("font_family", font.get("family")), "font.family"),
        style=_optional_string(params.get("font_style", font.get("style")), "font.style"),
        weight=_optional_int(params.get("font_weight", font.get("weight")), "font.weight"),
        size=_positive_int(params.get("font_size", font.get("size", 32)), "font.size"),
    )


def _outline(params: JsonObject) -> TextOutline:
    style = params.get("style") if isinstance(params.get("style"), dict) else {}
    outline = params.get("outline", style.get("outline", {}))
    outline = outline if isinstance(outline, dict) else {}
    color = params.get("stroke_color", params.get("outline_color", outline.get("color", outline.get("color_rgba", "#000000"))))
    width = params.get("stroke_width", params.get("outline_width", outline.get("width", 0.0)))
    return TextOutline(color=_color(color), width=_nonnegative_float(width, "outline.width"))


def _layout(params: JsonObject) -> TextLayout:
    layout = dict(params.get("layout", {})) if isinstance(params.get("layout"), dict) else {}
    return TextLayout(
        x=_integer(params.get("x", layout.get("x", 0)), "layout.x"),
        y=_integer(params.get("y", layout.get("y", 0)), "layout.y"),
        anchor=_optional_string(params.get("anchor", layout.get("anchor")), "layout.anchor"),
        align=_align(params.get("align", layout.get("align", "left"))),
        spacing=_nonnegative_int(params.get("spacing", layout.get("spacing", 0)), "layout.spacing"),
    )


def _nested(params: JsonObject, section: str, key: str, default: Any) -> Any:
    value = params.get(section)
    if isinstance(value, dict):
        return value.get(key, default)
    return default


def _color(value: Any) -> RGBA:
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


def _string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    return value


def _optional_string(value: Any, field_name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or None")
    return value


def _optional_int(value: Any, field_name: str) -> Optional[int]:
    if value is None:
        return None
    return _integer(value, field_name)


def _positive_int(value: Any, field_name: str) -> int:
    number = _integer(value, field_name)
    if number <= 0:
        raise ValueError(f"{field_name} must be positive")
    return number


def _nonnegative_int(value: Any, field_name: str) -> int:
    number = _integer(value, field_name)
    if number < 0:
        raise ValueError(f"{field_name} must be nonnegative")
    return number


def _integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    return int(value)


def _nonnegative_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number")
    number = float(value)
    if number < 0.0:
        raise ValueError(f"{field_name} must be nonnegative")
    return number


def _align(value: Any) -> str:
    if value not in {"left", "center", "right"}:
        raise ValueError("layout.align must be 'left', 'center', or 'right'")
    return str(value)


def _drop_none(data: JsonObject) -> JsonObject:
    return {key: value for key, value in data.items() if value is not None}
