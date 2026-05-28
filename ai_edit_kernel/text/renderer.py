"""Raster text rendering for text layers."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .fonts import FontFace, FontRegistry
from .params import TextRenderRequest, TextRenderResult


def render_text_pixels(
    width: int,
    height: int,
    params: dict[str, object],
    *,
    font_registry: Optional[FontRegistry] = None,
) -> TextRenderResult:
    """Render action params into a full-canvas RGBA text layer."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("text rendering requires Pillow") from exc

    request = TextRenderRequest.from_params(params)
    registry = FontRegistry.default() if font_registry is None else font_registry
    font_face = registry.resolve(
        font_id=request.font.id,
        font_path=request.font.path,
        family=request.font.family,
        style=request.font.style,
        weight=request.font.weight,
    )
    font = _load_font(ImageFont, font_face, request.font.size)

    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    outline_width = int(round(request.style.outline.width))
    outline_fill = _rgba255(request.style.outline.color) if outline_width > 0 else None
    draw.multiline_text(
        (request.layout.x, request.layout.y),
        request.text,
        fill=_rgba255(request.style.color),
        font=font,
        anchor=request.layout.anchor,
        align=request.layout.align,
        spacing=request.layout.spacing,
        stroke_width=outline_width,
        stroke_fill=outline_fill,
    )

    metadata = {
        "text": request.to_json(),
        "text_raw_params": dict(params),
        "rasterized": True,
        "font_face": font_face.to_json() if font_face is not None else None,
    }
    pixels = (np.asarray(image, dtype=np.float32) / 255.0).astype(np.float32)
    return TextRenderResult(pixels=pixels, metadata=metadata)


def _load_font(image_font_module: object, font_face: Optional[FontFace], size: int) -> object:
    """Load the requested font, falling back only when no explicit face exists."""
    if font_face is not None:
        return image_font_module.truetype(font_face.path, size)
    try:
        return image_font_module.truetype("DejaVuSans.ttf", size)
    except OSError:
        return image_font_module.load_default()


def _rgba255(rgba: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    return tuple(int(round(max(0.0, min(1.0, channel)) * 255.0)) for channel in rgba)
