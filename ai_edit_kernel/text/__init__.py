"""Text rendering support for editable text layers.

The public action schema remains JSON-oriented, but this package owns the typed
text model used internally by the executor. It resolves fonts, normalizes legacy
and structured text parameters, renders text pixels, and stores stable metadata
on text layers so future edits can be reproduced.
"""

from .fonts import FONT_CATALOG_SCHEMA_VERSION, FontFace, FontRegistry
from .params import TextFontSpec, TextLayout, TextOutline, TextRenderRequest, TextRenderResult, TextStyle, merge_text_params
from .renderer import render_text_pixels

__all__ = [
    "FontFace",
    "FontRegistry",
    "FONT_CATALOG_SCHEMA_VERSION",
    "TextFontSpec",
    "TextLayout",
    "TextOutline",
    "TextRenderRequest",
    "TextRenderResult",
    "TextStyle",
    "merge_text_params",
    "render_text_pixels",
]
