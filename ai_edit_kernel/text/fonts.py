"""Font discovery and stable font identifiers.

The planner should not need to guess filesystem paths. `FontRegistry` scans
well-known system and project font directories, extracts family/style metadata
with Pillow, and exposes stable, planner-friendly IDs such as
`dejavu_sans.book` or `noto_serif.bold`.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


FONT_SUFFIXES = {".ttf", ".otf", ".ttc"}
FONT_CATALOG_SCHEMA_VERSION = "ai_edit_font_catalog.v1"


@dataclass(frozen=True, slots=True)
class FontFace:
    """One loadable font face discovered on disk."""

    id: str
    family: str
    style: str
    path: str
    format: str
    weight: Optional[int] = None

    def to_json(self) -> dict[str, object]:
        """Return this font face as JSON-compatible metadata."""
        return {
            "id": self.id,
            "family": self.family,
            "style": self.style,
            "path": self.path,
            "format": self.format,
            "weight": self.weight,
        }


class FontRegistry:
    """Catalog of fonts addressable by stable IDs, family, or path."""

    _default: Optional["FontRegistry"] = None

    def __init__(self, faces: Iterable[FontFace] = ()) -> None:
        self._faces = sorted(list(faces), key=lambda face: (face.family.lower(), face.style.lower(), face.path))
        self._by_id = {face.id: face for face in self._faces}
        self._by_family: dict[str, list[FontFace]] = {}
        for face in self._faces:
            self._by_family.setdefault(face.family.lower(), []).append(face)

    @classmethod
    def default(cls) -> "FontRegistry":
        """Return a cached registry for the local machine."""
        if cls._default is None:
            cls._default = cls.discover()
        return cls._default

    @classmethod
    def discover(cls, search_dirs: Optional[Iterable[Path | str]] = None) -> "FontRegistry":
        """Scan directories and return a registry of readable font faces."""
        directories = _default_font_dirs() if search_dirs is None else [Path(item).expanduser() for item in search_dirs]
        candidates: list[Path] = []
        for directory in directories:
            if not directory.exists() or not directory.is_dir():
                continue
            try:
                candidates.extend(path for path in directory.rglob("*") if path.suffix.lower() in FONT_SUFFIXES)
            except OSError:
                continue

        raw_faces = [_load_font_face(path) for path in sorted(set(candidates))]
        faces = [face for face in raw_faces if face is not None]
        return cls(_assign_stable_ids(faces))

    def all_faces(self) -> list[FontFace]:
        """Return all known font faces sorted for deterministic display."""
        return list(self._faces)

    def catalog(
        self,
        *,
        limit: int = 80,
        include_paths: bool = False,
        query: Optional[str] = None,
    ) -> dict[str, object]:
        """Return a compact planner-facing font catalog.

        The full local font set can be very large. This method returns a
        representative, deterministic subset by default, while still reporting
        whether the catalog was truncated. Paths are omitted unless explicitly
        requested so planner prompts can use stable IDs without leaking local
        filesystem details.
        """
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer")
        if limit < 0:
            raise ValueError("limit must be nonnegative")
        faces = self._query_faces(query)
        selected = _representative_faces(faces, limit)
        return {
            "schema_version": FONT_CATALOG_SCHEMA_VERSION,
            "total_count": len(faces),
            "returned_count": len(selected),
            "truncated": len(selected) < len(faces),
            "selection_policy": "representative_by_family_style",
            "fonts": [_catalog_item(face, include_paths=include_paths) for face in selected],
        }

    def get(self, font_id: str) -> FontFace:
        """Return the face for `font_id`, raising `KeyError` when unknown."""
        return self._by_id[font_id]

    def _query_faces(self, query: Optional[str]) -> list[FontFace]:
        """Return faces whose ID, family, style, or tags match `query`."""
        if query is None or query.strip() == "":
            return list(self._faces)
        terms = [term for term in re.split(r"\s+", query.lower().strip()) if term]
        matches = []
        for face in self._faces:
            haystack = " ".join(
                [
                    face.id,
                    face.family,
                    face.style,
                    str(face.weight or ""),
                    " ".join(_tags_for_face(face)),
                ]
            ).lower()
            if all(term in haystack for term in terms):
                matches.append(face)
        return matches

    def find(
        self,
        *,
        family: Optional[str] = None,
        style: Optional[str] = None,
        weight: Optional[int] = None,
    ) -> Optional[FontFace]:
        """Find the closest face for the requested family/style/weight."""
        candidates = self._faces
        if family:
            family_key = family.lower()
            candidates = self._by_family.get(family_key, [])
            if not candidates:
                candidates = [face for face in self._faces if family_key in face.family.lower()]
        if not candidates:
            return None

        scored = sorted(
            candidates,
            key=lambda face: (
                _style_distance(face.style, style),
                _weight_distance(face.weight, weight),
                face.family.lower(),
                face.style.lower(),
                face.path,
            ),
        )
        return scored[0]

    def resolve(
        self,
        *,
        font_id: Optional[str] = None,
        font_path: Optional[str] = None,
        family: Optional[str] = None,
        style: Optional[str] = None,
        weight: Optional[int] = None,
    ) -> Optional[FontFace]:
        """Resolve explicit font fields to a font face.

        Explicit IDs and paths fail loudly when invalid. Family/style lookup is a
        best-effort convenience and returns `None` when no match exists, allowing
        callers to use their default rendering font.
        """
        if font_id:
            if font_id not in self._by_id:
                raise ValueError(f"unknown font_id {font_id!r}")
            return self._by_id[font_id]
        if font_path:
            path = Path(font_path).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"font_path does not exist: {font_path}")
            face = _load_font_face(path)
            if face is None:
                raise ValueError(f"font_path is not a readable font: {font_path}")
            return _assign_stable_ids([face])[0]
        if family or style or weight is not None:
            return self.find(family=family, style=style, weight=weight)
        return None


def _default_font_dirs() -> list[Path]:
    """Return conventional font search locations for common desktop systems."""
    home = Path.home()
    directories = [
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        home / ".local" / "share" / "fonts",
        home / ".fonts",
        Path("/Library/Fonts"),
        Path("/System/Library/Fonts"),
    ]
    windir = os.environ.get("WINDIR")
    if windir:
        directories.append(Path(windir) / "Fonts")
    return directories


def _load_font_face(path: Path) -> Optional[FontFace]:
    """Read family/style information from a font file."""
    try:
        from PIL import ImageFont

        font = ImageFont.truetype(str(path), 12)
        family, style = font.getname()
    except Exception:
        return None

    family = family or path.stem
    style = style or _style_from_name(path.stem)
    return FontFace(
        id="",
        family=str(family),
        style=str(style),
        path=str(path),
        format=path.suffix.lower().lstrip("."),
        weight=_weight_from_style(str(style), path.stem),
    )


def _assign_stable_ids(faces: list[FontFace]) -> list[FontFace]:
    """Assign deterministic IDs, disambiguating duplicate family/style pairs."""
    grouped: dict[str, list[FontFace]] = {}
    for face in faces:
        base = _font_id_base(face.family, face.style)
        grouped.setdefault(base, []).append(face)

    assigned: list[FontFace] = []
    for base, group in grouped.items():
        group = sorted(group, key=lambda face: face.path)
        for index, face in enumerate(group):
            font_id = base if index == 0 else f"{base}.{_short_hash(face.path)}"
            assigned.append(
                FontFace(
                    id=font_id,
                    family=face.family,
                    style=face.style,
                    path=face.path,
                    format=face.format,
                    weight=face.weight,
                )
            )
    return assigned


def _representative_faces(faces: list[FontFace], limit: int) -> list[FontFace]:
    """Select a stable, family-diverse subset for planner prompts."""
    if limit == 0:
        return []
    if len(faces) <= limit:
        return list(faces)

    by_family: dict[str, list[FontFace]] = {}
    for face in faces:
        by_family.setdefault(face.family.lower(), []).append(face)
    for family_faces in by_family.values():
        family_faces.sort(key=_font_priority_key)

    selected: list[FontFace] = []
    used: set[str] = set()
    max_family_depth = max(len(items) for items in by_family.values())
    for depth in range(max_family_depth):
        for family in sorted(by_family):
            family_faces = by_family[family]
            if depth >= len(family_faces):
                continue
            face = family_faces[depth]
            if face.id in used:
                continue
            selected.append(face)
            used.add(face.id)
            if len(selected) >= limit:
                return selected
    return selected


def _font_priority_key(face: FontFace) -> tuple[int, int, str, str]:
    """Prefer ordinary faces before decorative variants within a family."""
    style = face.style.lower()
    if style in {"regular", "book"}:
        style_rank = 0
    elif style == "bold":
        style_rank = 1
    elif style in {"italic", "oblique"}:
        style_rank = 2
    elif "bold" in style and ("italic" in style or "oblique" in style):
        style_rank = 3
    else:
        style_rank = 4
    weight = face.weight if face.weight is not None else 400
    return (style_rank, abs(weight - 400), style, face.path)


def _catalog_item(face: FontFace, *, include_paths: bool) -> dict[str, object]:
    """Return compact, planner-safe metadata for one font face."""
    item: dict[str, object] = {
        "id": face.id,
        "family": face.family,
        "style": face.style,
        "weight": face.weight,
        "format": face.format,
        "tags": _tags_for_face(face),
    }
    if include_paths:
        item["path"] = face.path
    return item


def _tags_for_face(face: FontFace) -> list[str]:
    """Infer simple searchable tags from font metadata."""
    text = f"{face.id} {face.family} {face.style} {Path(face.path).name}".lower()
    tags: set[str] = set()
    if "mono" in text or "code" in text:
        tags.add("monospace")
    if "serif" in text and "sans" not in text:
        tags.add("serif")
    if "sans" in text:
        tags.add("sans")
    if "bold" in text:
        tags.add("bold")
    if "italic" in text or "oblique" in text:
        tags.add("italic")
    if any(term in text for term in ("script", "hand", "brush", "calligraphy", "cursive", "nastaliq")):
        tags.add("handwritten")
    if any(term in text for term in ("display", "poster", "decorative", "black")):
        tags.add("display")
    if "emoji" in text:
        tags.add("emoji")
    if "symbol" in text or "icons" in text:
        tags.add("symbol")
    script_terms = {
        "arabic",
        "bengali",
        "cjk",
        "devanagari",
        "ethiopic",
        "hebrew",
        "japanese",
        "korean",
        "latin",
        "thai",
        "tibetan",
    }
    for term in script_terms:
        if term in text:
            tags.add(term)
    if not tags:
        tags.add("general")
    return sorted(tags)


def _font_id_base(family: str, style: str) -> str:
    family_slug = _slug(family)
    style_slug = _slug(style) if style else "regular"
    return f"{family_slug}.{style_slug}" if style_slug not in {"", "regular"} else family_slug


def _slug(value: str) -> str:
    safe = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return safe or "font"


def _short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]


def _style_from_name(name: str) -> str:
    lowered = name.lower()
    if "bold" in lowered and "italic" in lowered:
        return "Bold Italic"
    if "bold" in lowered:
        return "Bold"
    if "italic" in lowered or "oblique" in lowered:
        return "Italic"
    return "Regular"


def _weight_from_style(style: str, name: str) -> Optional[int]:
    lowered = f"{style} {name}".lower()
    weights = {
        "thin": 100,
        "extralight": 200,
        "extra light": 200,
        "light": 300,
        "regular": 400,
        "book": 400,
        "medium": 500,
        "semibold": 600,
        "semi bold": 600,
        "bold": 700,
        "extrabold": 800,
        "extra bold": 800,
        "black": 900,
    }
    for label, weight in weights.items():
        if label in lowered:
            return weight
    return None


def _style_distance(style: str, requested: Optional[str]) -> int:
    if not requested:
        return 0
    style_key = style.lower()
    requested_key = requested.lower()
    if style_key == requested_key:
        return 0
    if requested_key in style_key:
        return 1
    return 2


def _weight_distance(weight: Optional[int], requested: Optional[int]) -> int:
    if requested is None:
        return 0
    if weight is None:
        return 1000
    return abs(weight - requested)
