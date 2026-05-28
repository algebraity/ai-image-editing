"""Tests for text font resolution and structured text params."""

from __future__ import annotations

import unittest

import numpy as np

from ai_edit_kernel.document.document_state import CanvasSpec, DocumentState
from ai_edit_kernel.planning.planner import PlannerRequestBuilder
from ai_edit_kernel.runtime.executor import Executor
from ai_edit_kernel.schema.actions import Action, ActionStatus, ActionTarget, ActionType
from ai_edit_kernel.text import FONT_CATALOG_SCHEMA_VERSION, FontRegistry, render_text_pixels


class TextModuleTests(unittest.TestCase):
    """Exercise the text module independently and through text actions."""

    def test_font_registry_exposes_stable_ids(self) -> None:
        registry = FontRegistry.default()
        faces = registry.all_faces()
        if not faces:
            self.assertIsNone(registry.resolve())
            return

        face = faces[0]
        self.assertEqual(registry.get(face.id), face)
        self.assertEqual(registry.resolve(font_id=face.id), face)
        self.assertTrue(face.id)
        self.assertTrue(face.family)
        self.assertTrue(face.path)

    def test_font_catalog_is_compact_and_planner_safe(self) -> None:
        registry = FontRegistry.default()
        catalog = registry.catalog(limit=5)

        self.assertEqual(catalog["schema_version"], FONT_CATALOG_SCHEMA_VERSION)
        self.assertLessEqual(catalog["returned_count"], 5)
        self.assertEqual(catalog["returned_count"], len(catalog["fonts"]))
        for item in catalog["fonts"]:
            self.assertIn("id", item)
            self.assertIn("family", item)
            self.assertIn("style", item)
            self.assertIn("tags", item)
            self.assertNotIn("path", item)

    def test_planner_request_includes_font_catalog(self) -> None:
        document = DocumentState(id="doc_font_catalog", canvas=CanvasSpec(width=64, height=64))
        request = PlannerRequestBuilder(font_catalog_limit=3).build("Add a title.", document)

        self.assertIn("font_catalog", request)
        self.assertEqual(request["font_catalog"]["schema_version"], FONT_CATALOG_SCHEMA_VERSION)
        self.assertLessEqual(request["font_catalog"]["returned_count"], 3)
        self.assertTrue(any("font_id" in item for item in request["constraints"]))

    def test_render_structured_text_with_outline(self) -> None:
        font_id = self._optional_font_id()
        params = {
            "text": "AI",
            "layout": {"x": 12, "y": 12},
            "font": {"id": font_id, "size": 34} if font_id else {"size": 34},
            "style": {"color": "#ffffff", "outline": {"color": "#000000", "width": 3}},
        }

        result = render_text_pixels(120, 80, params)

        self.assertEqual(result.pixels.shape, (80, 120, 4))
        self.assertGreater(float(np.max(result.pixels[..., 3])), 0.0)
        self.assertEqual(result.metadata["text"]["font"]["size"], 34)
        self.assertEqual(result.metadata["text"]["style"]["outline"]["width"], 3.0)
        if font_id:
            self.assertEqual(result.metadata["text"]["font"]["id"], font_id)

    def test_create_and_edit_text_layer_with_partial_params(self) -> None:
        document = DocumentState(id="doc_text", canvas=CanvasSpec(width=160, height=96))
        executor = Executor()
        font_id = self._optional_font_id()
        font = {"id": font_id, "size": 26} if font_id else {"size": 26}

        create_result = executor.execute_action(
            document,
            Action(
                id="action_create_text",
                type=ActionType.CREATE_TEXT_LAYER,
                target=ActionTarget(output_layer_id="layer_text"),
                params={
                    "text": "Hello",
                    "font": font,
                    "layout": {"x": 12, "y": 12},
                    "style": {"color": "#2244ff"},
                    "set_active": False,
                },
            ),
        )
        edit_result = executor.execute_action(
            document,
            Action(
                id="action_edit_text",
                type=ActionType.EDIT_TEXT_LAYER,
                target=ActionTarget(layer_id="layer_text"),
                params={"font_size": 30, "stroke_color": "#ffcc00", "stroke_width": 2},
            ),
        )

        self.assertEqual(create_result.status, ActionStatus.EXECUTED)
        self.assertEqual(edit_result.status, ActionStatus.EXECUTED)
        layer = document.get_layer("layer_text")
        self.assertEqual(layer.kind.value, "text")
        self.assertEqual(layer.metadata["text"]["text"], "Hello")
        self.assertEqual(layer.metadata["text"]["font"]["size"], 30)
        self.assertEqual(layer.metadata["text"]["style"]["outline"]["width"], 2.0)
        self.assertGreater(float(np.max(layer.pixels[..., 3])), 0.0)

    def _optional_font_id(self) -> str | None:
        faces = FontRegistry.default().all_faces()
        return faces[0].id if faces else None


if __name__ == "__main__":
    unittest.main()
