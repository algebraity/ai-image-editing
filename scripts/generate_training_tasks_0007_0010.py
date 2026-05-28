#!/usr/bin/env python3
"""Generate local planner/executor training examples for tasks 0007-0010."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_edit_kernel.document.document_state import CanvasSpec, DocumentMetadata, DocumentState
from ai_edit_kernel.document.layer import Layer
from ai_edit_kernel.planning.planner import AIPlanner, PlannerExecutionResult, PlannerRequestBuilder, StaticPlannerBackend
from ai_edit_kernel.runtime.executor import ExecutionContext, Executor
from ai_edit_kernel.runtime.validator import Validator
from ai_edit_kernel.trace.trace_logger import TraceLogger, TraceSession


DATASET_SCHEMA_VERSION = "ai_edit_training_example.v1"
PROMPT_0007 = "add three red hearts of various sizes, slightly rotated to the left, in the upper left corner of the image"
PROMPT_0008 = "change the color of the bow on her chest to purple, while keeping the lighting / shading etc. the same / equally as pretty"
PROMPT_0009 = "mirror the image over the vertical axis in the center"
PROMPT_0010 = "make her heterochromatic, with a red left eye and orange / amber right eye, and make her hair darker, e.g. brown, while keeping the general contrast of the hair the same"


BOW_MAGIC_WAND_SEEDS = [
    [345, 185], [350, 185], [355, 185], [360, 185], [365, 185], [385, 185], [390, 185], [395, 185],
    [345, 190], [350, 190], [355, 190], [360, 190], [365, 190], [370, 190], [380, 190], [385, 190],
    [390, 190], [395, 190], [400, 190], [345, 195], [350, 195], [355, 195], [360, 195], [365, 195],
    [370, 195], [385, 195], [390, 195], [395, 195], [400, 195], [350, 200], [355, 200], [360, 200],
    [365, 200], [370, 200], [385, 200], [390, 200], [395, 200], [400, 200], [405, 200], [345, 205],
    [350, 205], [355, 205], [360, 205], [365, 205], [370, 205], [385, 205], [390, 205], [395, 205],
    [400, 205], [405, 205], [340, 210], [345, 210], [350, 210], [355, 210], [360, 210], [365, 210],
    [370, 210], [390, 210], [395, 210], [400, 210], [405, 210], [410, 210], [345, 215],
    [350, 215], [355, 215], [360, 215], [365, 215], [395, 215], [400, 215], [405, 215], [410, 215],
    [415, 215], [355, 220], [360, 220], [365, 220], [400, 220], [405, 220], [360, 225],
    [400, 225], [405, 225], [360, 230],
]

BOW_SHADOW_MAGIC_WAND_SEEDS = [[348, 204], [358, 212], [372, 196], [380, 210], [388, 196], [396, 220], [405, 215]]

KIRIRIN_LEFT_IRIS_SEEDS = [
    [150, 225], [158, 225], [158, 230], [166, 230], [158, 235], [166, 235], [150, 245], [158, 245],
    [166, 245], [174, 245], [150, 250], [158, 250], [174, 250], [150, 255], [158, 255], [174, 255],
    [150, 260], [158, 260], [166, 260], [174, 260], [150, 265], [158, 265], [166, 265], [174, 265],
    [158, 270], [166, 270], [158, 275],
]

KIRIRIN_RIGHT_IRIS_SEEDS = [
    [288, 200], [294, 200], [300, 200], [288, 205], [294, 205], [300, 205], [306, 205], [288, 210],
    [294, 210], [300, 210], [306, 210], [312, 215], [282, 222], [292, 222], [304, 224], [312, 232],
    [280, 238], [292, 242], [305, 242], [285, 252], [300, 252], [312, 247],
]

KIRIRIN_HAIR_BASE_SEEDS = [
    [221, 5], [239, 5], [257, 5], [167, 23], [185, 23], [203, 23], [221, 23], [239, 23], [257, 23],
    [275, 23], [293, 23], [311, 23], [131, 41], [149, 41], [167, 41], [185, 41], [203, 41], [221, 41],
    [239, 41], [257, 41], [275, 41], [293, 41], [311, 41], [329, 41], [113, 59], [131, 59], [149, 59],
    [167, 59], [185, 59], [203, 59], [221, 59], [239, 59], [257, 59], [275, 59], [293, 59], [311, 59],
    [329, 59], [347, 59], [95, 77], [113, 77], [131, 77], [149, 77], [167, 77], [185, 77], [203, 77],
    [221, 77], [239, 77], [257, 77], [275, 77], [293, 77], [311, 77], [329, 77], [347, 77], [77, 95],
    [95, 95], [113, 95], [131, 95], [149, 95], [167, 95], [185, 95], [203, 95], [221, 95], [239, 95],
    [257, 95], [275, 95], [293, 95], [311, 95], [95, 113], [113, 113], [131, 113], [149, 113], [167, 113],
    [185, 113], [203, 113], [221, 113], [239, 113], [257, 113], [311, 113], [347, 113], [365, 113],
    [383, 113], [59, 131], [77, 131], [95, 131], [113, 131], [167, 131], [185, 131], [203, 131], [221, 131],
    [239, 131], [257, 131], [275, 131], [293, 131], [329, 131], [347, 131], [365, 131], [383, 131],
    [59, 149], [77, 149], [95, 149], [113, 149], [167, 149], [221, 149], [239, 149], [257, 149],
    [275, 149], [311, 149], [329, 149], [347, 149], [365, 149], [383, 149], [59, 167], [77, 167],
    [95, 167], [113, 167], [185, 167], [239, 167], [257, 167], [293, 167], [311, 167], [347, 167],
    [365, 167], [383, 167], [59, 185], [77, 185], [95, 185], [113, 185], [185, 185], [293, 185],
    [311, 185], [329, 185], [347, 185], [365, 185], [383, 185], [59, 203], [77, 203], [347, 203],
    [365, 203], [383, 203], [59, 221], [77, 221], [347, 221], [365, 221], [59, 239], [77, 239],
    [347, 239], [365, 239], [77, 257], [95, 257], [347, 257], [365, 257], [401, 257], [77, 275],
    [95, 275], [347, 275], [365, 275], [383, 275], [95, 293], [113, 293], [77, 311], [113, 311],
    [131, 311], [113, 329], [131, 365], [365, 365], [131, 383], [365, 383], [131, 401], [365, 401],
    [131, 419], [365, 419], [131, 437], [365, 437], [131, 455], [365, 455], [131, 473], [365, 473],
    [131, 491], [365, 491], [437, 491], [131, 509], [365, 509], [437, 509], [131, 527], [365, 527],
    [131, 545], [347, 545], [131, 563], [347, 563],
]

KIRIRIN_HAIR_HIGHLIGHT_SEEDS = [
    [68, 106], [70, 110], [72, 122], [78, 118], [82, 108], [92, 112], [108, 96], [130, 84], [150, 74],
    [170, 65], [200, 55], [230, 50], [260, 48], [295, 54], [320, 62], [342, 44], [363, 85], [368, 100],
    [382, 114], [390, 158], [55, 146], [61, 134], [75, 129], [93, 120], [120, 100], [145, 88],
    [165, 78], [190, 70], [225, 64], [260, 63], [300, 69], [335, 82], [350, 96], [103, 445],
    [110, 460], [121, 478], [132, 500], [350, 410], [362, 430], [370, 455], [382, 478],
]

KIRIRIN_HAIR_SHADOW_SEEDS = [
    [56, 176], [64, 220], [88, 245], [116, 305], [126, 420], [126, 525], [354, 250], [390, 250],
    [368, 460], [365, 535], [330, 185], [288, 118], [280, 110], [305, 122], [320, 146], [345, 170],
]


@dataclass(slots=True)
class ExampleSpec:
    """Everything needed to run one local example through the planner path."""

    example_id: str
    prompt: str
    source_image: Path
    document: DocumentState
    planner_output: dict[str, Any]
    observations: list[dict[str, Any]]
    notes: list[str] | None = None


class TrainingRequestBuilder(PlannerRequestBuilder):
    """Keep new local examples close to the existing training0006 request shape."""

    def __init__(self) -> None:
        super().__init__(include_action_schemas=True, include_font_catalog=False)

    def build(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        data = super().build(*args, **kwargs)
        data.pop("font_catalog", None)
        data["constraints"] = [
            item for item in data["constraints"] if not item.startswith("When planning text,")
        ]
        return data


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)

    factories: dict[str, Callable[[Path, argparse.Namespace], ExampleSpec]] = {
        "training0007": build_example_0007,
        "training0008": build_example_0008,
        "training0009": build_example_0009,
        "training0010": build_example_0010,
    }
    selected = args.examples or list(factories)
    summaries = []
    for example_id in selected:
        if example_id not in factories:
            raise ValueError(f"unknown example id {example_id!r}")
        example = factories[example_id](dataset_root, args)
        summaries.append(run_example(example, dataset_root))

    success = all(item["succeeded"] and item["validation_passed"] for item in summaries)
    print(json.dumps({"success": success, "examples": summaries}, indent=2))
    return 0 if success else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate local deterministic AI editing examples 0007-0010.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/algebraity/Programming/aie-training-data"))
    parser.add_argument("--kirishita-image", type=Path, default=Path("/home/algebraity/Media/Pictures/kirishita.png"))
    parser.add_argument("--bow-image", type=Path, default=Path("/home/algebraity/Pictures/2026-05-27-000937_hyprshot.png"))
    parser.add_argument("--kiririn-image", type=Path, default=REPO_ROOT / "tests" / "kiririn.jpg")
    parser.add_argument("--kirikawaii-image", type=Path, default=Path("/home/algebraity/Media/Pictures/kirikawaii.jpg"))
    parser.add_argument("--examples", nargs="*", default=None, help="Optional subset, e.g. training0007 training0010.")
    return parser.parse_args()


def build_example_0007(dataset_root: Path, args: argparse.Namespace) -> ExampleSpec:
    source = args.kirishita_image.resolve()
    document, width, height = source_document("doc_training0007_initial", source, "Initial source document")
    output_dir = Path("data") / "training0007"
    heart = "\u2665"
    return ExampleSpec(
        example_id="training0007",
        prompt=PROMPT_0007,
        source_image=source,
        document=document,
        observations=[
            source_observation(width, height),
            {
                "type": "layout",
                "summary": "Three red heart text layers are placed in the upper-left background area, with different sizes and slight counter-clockwise rotation.",
                "details": {
                    "heart_positions_xy": [[55, 42], [112, 35], [76, 90]],
                    "font_sizes": [42, 31, 25],
                    "rotation_degrees": [-12, -10, -8],
                    "target_corner": "upper_left",
                },
            },
        ],
        planner_output={
            "schema_version": "ai_edit_planner_output.v1",
            "description": "Add three red heart overlays of varying sizes in the upper-left corner and rotate each slightly left.",
            "stop_on_error": True,
            "metadata": {"example_id": "training0007", "external_diffusion_used": False},
            "actions": [
                create_text_action("layer_heart_large", "large red heart", heart, 55, 42, 42, "#e60022", "Create the largest red heart in the upper-left corner."),
                rotate_action("layer_heart_large", -12, [55, 42], "Rotate the largest heart slightly to the left."),
                create_text_action("layer_heart_medium", "medium red heart", heart, 112, 35, 31, "#e60022", "Create a medium red heart near the top-left edge."),
                rotate_action("layer_heart_medium", -10, [112, 35], "Rotate the medium heart slightly to the left."),
                create_text_action("layer_heart_small", "small red heart", heart, 76, 90, 25, "#e60022", "Create a smaller red heart below the other two hearts."),
                rotate_action("layer_heart_small", -8, [76, 90], "Rotate the small heart slightly to the left."),
                validate_action("Validate that the image with heart overlays has no structural issues."),
                export_flat_action(output_dir, "Export the flattened image with the three red hearts."),
                export_bundle_action(output_dir, "Export layers, masks, preview, and document metadata for inspection."),
            ],
        },
    )


def build_example_0008(dataset_root: Path, args: argparse.Namespace) -> ExampleSpec:
    original_source = args.bow_image.resolve()
    crop_bbox = [246, 0, 467, 260]
    source = crop_source_image(dataset_root, original_source, "training0008", crop_bbox)
    document, width, height = source_document("doc_training0008_initial", source, "Initial source document")
    output_dir = Path("data") / "training0008"
    bow_bbox = [69, 180, 174, 236]
    bow_seeds = [[104, 190], [150, 200], [114, 220], [142, 196], [102, 205], [159, 215], [90, 213], [138, 198]]
    bow_exclude_seeds = [[74, 215], [132, 190], [136, 196]]
    return ExampleSpec(
        example_id="training0008",
        prompt=PROMPT_0008,
        source_image=source,
        document=document,
        observations=[
            source_observation(width, height),
            {
                "type": "preprocess",
                "summary": "The source image is cropped to the right character only, removing the left portrait and the black center divider before planning the bow recolor.",
                "details": {
                    "crop_bbox_xyxy_in_original": crop_bbox,
                    "cropped_canvas": {"width": width, "height": height},
                    "reason": "The requested edit concerns only the right-side character, so the training example starts from the relevant cropped image.",
                },
            },
            {
                "type": "region",
                "summary": "The bow on the cropped character's chest is selected with one bounded HSV seed-color selection, using bow seed pixels as inclusions and nearby skin/gem/knot pixels as exclusions.",
                "details": {
                    "bbox_xyxy": bow_bbox,
                    "seed_points": bow_seeds,
                    "exclude_seed_points": bow_exclude_seeds,
                    "color_space": "hsv",
                    "target_color": "#8A2BE2",
                },
            },
            {
                "type": "edit_strategy",
                "summary": "Clean and softly edge the selected bow pixels, then set their hue to purple while preserving the existing lightness/shading.",
                "details": {"amount": 1.0, "target_color": "#8A2BE2", "colorize_method": "set_hue_preserve_lightness"},
            },
        ],
        planner_output={
            "schema_version": "ai_edit_planner_output.v1",
            "description": "On the cropped right-character image, select the complete pink bow with a bounded HSV seed-color range and recolor it purple while preserving lightness.",
            "stop_on_error": True,
            "metadata": {"example_id": "training0008", "external_diffusion_used": False},
            "actions": [
                seeded_color_range_action(
                    "mask_bow_raw",
                    "Bounded HSV bow material selection",
                    bow_seeds,
                    bow_exclude_seeds,
                    bow_bbox,
                    "Use one bounded HSV seed-color selection to capture the disconnected pink bow material while excluding nearby skin, blue gem, and gold knot colors.",
                ),
                refine_action("mask_bow", "mask_bow_raw", "Cleaned bow write mask", 0.35, "Remove tiny selection islands, fill enclosed bow holes, smooth the hard selection, and feather the edge for clean antialiasing.", min_area=20, fill_holes=True, smooth_radius=0.35),
                colorize_action("mask_bow", "#8A2BE2", "Set only the bow hue to purple while preserving its existing lightness, line art, highlights, and shadows.", method="set_hue_preserve_lightness"),
                validate_action("Validate that the purple-bow edit has no structural issues."),
                export_flat_action(output_dir, "Export the flattened purple-bow result."),
                export_bundle_action(output_dir, "Export layers, masks, preview, and document metadata for inspection."),
            ],
        },
        notes=["The source image is intentionally cropped to the right-side character before planning."],
    )


def build_example_0009(dataset_root: Path, args: argparse.Namespace) -> ExampleSpec:
    source = dataset_root / "data" / "training0008" / "final_image.png"
    document, width, height = source_document("doc_training0009_initial", source, "Initial source document")
    output_dir = Path("data") / "training0009"
    return ExampleSpec(
        example_id="training0009",
        prompt=PROMPT_0009,
        source_image=source,
        document=document,
        observations=[
            source_observation(width, height),
            {
                "type": "transform_plan",
                "summary": "Mirror the single source layer horizontally around the exact vertical centerline of the canvas.",
                "details": {"canvas_width": width, "canvas_height": height, "flip_anchor_xy": [width / 2.0, height / 2.0]},
            },
        ],
        planner_output={
            "schema_version": "ai_edit_planner_output.v1",
            "description": "Flip the source layer horizontally around the canvas centerline.",
            "stop_on_error": True,
            "metadata": {"example_id": "training0009", "external_diffusion_used": False},
            "actions": [
                {
                    "type": "flip_layer",
                    "target": {"layer_id": "source_image"},
                    "params": {
                        "horizontal": True,
                        "vertical": False,
                        "anchor": [width / 2.0, height / 2.0],
                        "resample": "bicubic",
                        "fill_color": "#00000000",
                    },
                    "description": "Mirror the source image over the vertical axis through the canvas center.",
                },
                validate_action("Validate that the mirrored document has no structural issues."),
                export_flat_action(output_dir, "Export the flattened mirrored result."),
                export_bundle_action(output_dir, "Export layers, masks, preview, and document metadata for inspection."),
            ],
        },
    )


def build_example_0010(dataset_root: Path, args: argparse.Namespace) -> ExampleSpec:
    source = args.kirikawaii_image.resolve()
    document, width, height = source_document("doc_training0010_initial", source, "Initial source document")
    output_dir = Path("data") / "training0010"
    left_eye_bbox = [430, 270, 710, 575]
    right_eye_bbox = [930, 235, 1195, 500]
    left_reflection_bbox = [545, 450, 685, 535]
    right_reflection_bbox = [1000, 365, 1165, 465]
    hair_bbox = [80, 0, 1545, height]
    right_outer_tip_bbox = [1538, 945, 1588, 1010]
    dark_blue = "#294766"
    light_blue = "#7EABBB"
    reflection_blue = "#7EACBC"
    reflection_pale = "#CDDCE3"
    red = "#E00018"
    reflection_red = "#FF8FA0"
    amber = "#E69A00"
    reflection_amber = "#FFD36A"
    brown = "#6B3D1D"
    return ExampleSpec(
        example_id="training0010",
        prompt=PROMPT_0010,
        source_image=source,
        document=document,
        observations=[
            source_observation(width, height),
            {
                "type": "region",
                "summary": "The visible blue iris pixels are selected in two tonal bands per eye inside tight iris bounds, with the image-left iris recolored red and the image-right iris recolored amber.",
                "details": {
                    "left_eye_bbox": left_eye_bbox,
                    "right_eye_bbox": right_eye_bbox,
                    "left_reflection_bbox": left_reflection_bbox,
                    "right_reflection_bbox": right_reflection_bbox,
                    "source_dark_blue": dark_blue,
                    "source_light_blue": light_blue,
                    "source_reflection_colors": [reflection_blue, reflection_pale],
                    "left_eye_target_color": red,
                    "right_eye_target_color": amber,
                    "left_reflection_target_color": reflection_red,
                    "right_reflection_target_color": reflection_amber,
                },
            },
            {
                "type": "region",
                "summary": "The blonde hair is selected by its base, shadow, midtone, highlight, and deep-orange material colors inside the hair bounds, avoiding skin, background, clips, pupils, and line art.",
                "details": {
                    "hair_bbox": hair_bbox,
                    "source_tone_colors": ["#E8B566", "#D5885D", "#F2C360", "#FFF0A0", "#C77D4B"],
                    "target_color": brown,
                    "colorize_method": "material_hsl",
                },
            },
            {
                "type": "qa_correction",
                "summary": "After the main hair mask, inspect narrow extremal hair tips for disconnected old-blonde islands that were outside the broad hair bounds or not adjacent to the core mask. Add a tight supplemental color selection for each missed tip and union it into the final hair write mask.",
                "details": {
                    "missed_tip_bbox": right_outer_tip_bbox,
                    "why_needed": "The far-right lower hair tip is a small disconnected material island beyond the original broad hair bbox, so fringe cleanup cannot reach it.",
                    "how_to_spot": "Look for small yellow/blonde regions at the end of hair strands after the main recolor plan; these should be treated as hair material even when separated by line art or outside the first bbox.",
                    "source_tip_colors": ["#E8B566", "#EFB665", "#A78B64"],
                },
            },
        ],
        planner_output={
            "schema_version": "ai_edit_planner_output.v1",
            "description": "Make the image-left eye red, the image-right eye amber, and recolor the blonde hair brown while preserving line art, highlights, shadows, and contrast.",
            "stop_on_error": True,
            "metadata": {"example_id": "training0010", "external_diffusion_used": False},
            "actions": [
                color_range_action("mask_left_iris_dark", "Left iris dark blue", dark_blue, 0.18, left_eye_bbox, "Select the darker blue pixels in the viewer-left iris."),
                color_range_action("mask_left_iris_light", "Left iris light blue", light_blue, 0.20, left_eye_bbox, "Select the lighter blue pixels in the viewer-left iris."),
                color_range_action("mask_right_iris_dark", "Right iris dark blue", dark_blue, 0.18, right_eye_bbox, "Select the darker blue pixels in the viewer-right iris."),
                color_range_action("mask_right_iris_light", "Right iris light blue", light_blue, 0.20, right_eye_bbox, "Select the lighter blue pixels in the viewer-right iris."),
                color_range_action("mask_left_reflection_blue", "Left lower iris reflection blue", reflection_blue, 0.08, left_reflection_bbox, "Select the medium-blue lower reflection pixels in the viewer-left iris."),
                color_range_action("mask_left_reflection_pale", "Left lower iris reflection pale cyan", reflection_pale, 0.10, left_reflection_bbox, "Select the pale cyan lower reflection pixels in the viewer-left iris."),
                color_range_action("mask_right_reflection_blue", "Right lower iris reflection blue", reflection_blue, 0.08, right_reflection_bbox, "Select the medium-blue lower reflection pixels in the viewer-right iris."),
                color_range_action("mask_right_reflection_pale", "Right lower iris reflection pale cyan", reflection_pale, 0.10, right_reflection_bbox, "Select the pale cyan lower reflection pixels in the viewer-right iris."),
                combine_action("mask_left_iris_union", ["mask_left_iris_dark", "mask_left_iris_light"], "Combined left iris blue mask", "union", "Union the selected blue tone masks for the viewer-left iris."),
                combine_action("mask_right_iris_union", ["mask_right_iris_dark", "mask_right_iris_light"], "Combined right iris blue mask", "union", "Union the selected blue tone masks for the viewer-right iris."),
                combine_action("mask_left_reflection_union", ["mask_left_reflection_blue", "mask_left_reflection_pale"], "Combined left lower reflection mask", "union", "Union the lower reflection tone masks for the viewer-left iris."),
                combine_action("mask_right_reflection_union", ["mask_right_reflection_blue", "mask_right_reflection_pale"], "Combined right lower reflection mask", "union", "Union the lower reflection tone masks for the viewer-right iris."),
                refine_action("mask_left_iris_write", "mask_left_iris_union", "Soft left iris write mask", 0.65, "Remove tiny isolated selections and softly feather the viewer-left iris mask before recoloring.", min_area=40, smooth_radius=0.25),
                refine_action("mask_right_iris_write", "mask_right_iris_union", "Soft right iris write mask", 0.65, "Remove tiny isolated selections and softly feather the viewer-right iris mask before recoloring.", min_area=40, smooth_radius=0.25),
                refine_action("mask_left_reflection_write", "mask_left_reflection_union", "Soft left lower reflection mask", 0.35, "Clean and softly feather the viewer-left lower iris reflection mask before recoloring it as a lighter red reflection.", min_area=8, smooth_radius=0.15),
                refine_action("mask_right_reflection_write", "mask_right_reflection_union", "Soft right lower reflection mask", 0.35, "Clean and softly feather the viewer-right lower iris reflection mask before recoloring it as a lighter amber reflection.", min_area=8, smooth_radius=0.15),
                color_range_action("mask_hair_base", "Hair base gold", "#E8B566", 0.10, hair_bbox, "Select the base blonde hair material color while staying inside the hair bounds."),
                color_range_action("mask_hair_shadow", "Hair shadow orange", "#D5885D", 0.11, hair_bbox, "Select the orange-brown shadow hair material color while staying inside the hair bounds."),
                color_range_action("mask_hair_mid", "Hair mid gold", "#F2C360", 0.10, hair_bbox, "Select the mid/highlight blonde hair material color while staying inside the hair bounds."),
                color_range_action("mask_hair_highlight", "Hair pale highlight", "#FFF0A0", 0.08, hair_bbox, "Select the pale yellow hair highlight material color while staying inside the hair bounds."),
                color_range_action("mask_hair_deep", "Hair deep orange", "#C77D4B", 0.10, hair_bbox, "Select the deeper orange hair material color while staying inside the hair bounds."),
                combine_action("mask_hair_combined", ["mask_hair_base", "mask_hair_shadow", "mask_hair_mid", "mask_hair_highlight", "mask_hair_deep"], "Combined hair material mask", "union", "Union all selected blonde/orange hair tone masks into one hair recolor mask."),
                refine_action("mask_hair_core", "mask_hair_combined", "Soft hair core mask", 0.20, "Clean tiny islands and lightly smooth the selected interior hair material before local fringe cleanup.", min_area=20, smooth_radius=0.2),
                extract_line_art_action("mask_line_art", "Ink line protection", "Detect dark anime-style ink lines so fringe cleanup does not pull the recolor into line art."),
                cleanup_fringe_action(
                    "mask_hair_main_write",
                    "mask_hair_core",
                    "Hair with local blonde fringe",
                    ["#E8B566", "#D5885D", "#F2C360", "#FFF0A0", "#C77D4B"],
                    hair_bbox,
                    ["mask_line_art"],
                    "Recover old blonde antialias/fringe pixels immediately adjacent to the hair core while protecting ink lines.",
                ),
                seeded_color_range_action(
                    "mask_hair_tip_right",
                    "Right outer missed hair tip",
                    [[1548, 980], [1555, 985], [1562, 990], [1546, 970], [1568, 995]],
                    [[1582, 985], [1540, 955]],
                    right_outer_tip_bbox,
                    "Select the small disconnected far-right lower hair-tip island that the main hair mask and fringe cleanup missed; future planners should add this kind of tight supplemental selection when old-color pixels remain at hair-strand tips.",
                    hue_tolerance_degrees=18,
                    saturation_tolerance=0.45,
                    value_tolerance=0.50,
                ),
                combine_action("mask_hair_write", ["mask_hair_main_write", "mask_hair_tip_right"], "Hair mask including missed tips", "union", "Union the main hair mask with the supplemental missed-tip mask; this teaches planners to add bounded tip corrections when small old-color hair islands remain."),
                colorize_action("mask_hair_write", brown, "Recolor the selected blonde hair material to darker brown while preserving the general highlight and shadow contrast.", method="material_hsl", contrast=1.1),
                colorize_action("mask_left_iris_write", red, "Recolor only the viewer-left iris material to red while preserving existing shading, highlights, pupils, and antialiasing.", method="material_hsl", contrast=1.05),
                colorize_action("mask_right_iris_write", amber, "Recolor only the viewer-right iris material to amber while preserving existing shading, highlights, pupils, and antialiasing.", method="material_hsl", contrast=1.05),
                colorize_action("mask_left_reflection_write", reflection_red, "Recolor the viewer-left lower iris reflection to a lighter red/pink reflection while preserving its light structure.", method="material_hsl", contrast=1.0),
                colorize_action("mask_right_reflection_write", reflection_amber, "Recolor the viewer-right lower iris reflection to a lighter amber/yellow reflection while preserving its light structure.", method="material_hsl", contrast=1.0),
                validate_action("Validate that the heterochromia and brown-hair edit has no structural issues."),
                export_flat_action(output_dir, "Export the flattened heterochromia and brown-hair result."),
                export_bundle_action(output_dir, "Export layers, masks, preview, and document metadata for inspection."),
            ],
        },
    )

def source_document(document_id: str, source_image: Path, title: str) -> tuple[DocumentState, int, int]:
    if not source_image.exists():
        raise FileNotFoundError(f"source image does not exist: {source_image}")
    with Image.open(source_image) as image:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    height, width = rgba.shape[:2]
    document = DocumentState(
        id=document_id,
        canvas=CanvasSpec(width=width, height=height, background_color_rgba=(0.0, 0.0, 0.0, 0.0)),
        layers=[Layer(id="source_image", name="Source Image", pixels=rgba.astype(np.float32), metadata={"source_path": "initial_image.png", "import_offset_xy": [0, 0], "import_size": [width, height]})],
        active_layer_id="source_image",
        metadata=DocumentMetadata(title=title, source_file="initial_image.png"),
    )
    document.validate()
    return document, width, height


def crop_source_image(dataset_root: Path, source_image: Path, example_id: str, crop_bbox_xyxy: list[int]) -> Path:
    if not source_image.exists():
        raise FileNotFoundError(f"source image does not exist: {source_image}")
    output_dir = dataset_root / "generated_sources"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{example_id}_initial_image.png"
    with Image.open(source_image) as image:
        cropped = image.convert("RGBA").crop(tuple(crop_bbox_xyxy))
        cropped.save(output)
    return output


def source_observation(width: int, height: int) -> dict[str, Any]:
    return {
        "type": "source_image",
        "summary": f"User-provided image, {width}x{height} pixels.",
        "details": {"width": width, "height": height, "source_image_path": "initial_image.png"},
    }


def create_text_action(layer_id: str, name: str, text: str, x: int, y: int, font_size: int, color: str, description: str) -> dict[str, Any]:
    return {
        "type": "create_text_layer",
        "target": {"output_layer_id": layer_id},
        "params": {
            "name": name,
            "text": text,
            "x": x,
            "y": y,
            "font_size": font_size,
            "color": color,
            "anchor": "mm",
            "align": "center",
            "set_active": False,
        },
        "description": description,
    }


def rotate_action(layer_id: str, degrees: float, anchor: list[float], description: str) -> dict[str, Any]:
    return {
        "type": "rotate_layer",
        "target": {"layer_id": layer_id},
        "params": {"angle_degrees": degrees, "anchor": anchor, "resample": "bicubic", "fill_color": "#00000000"},
        "description": description,
    }


def magic_wand_action(
    mask_id: str,
    name: str,
    seed_points: list[list[int]],
    tolerance: float,
    description: str,
    *,
    bbox_xyxy: list[int] | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "name": name,
        "seed_points": seed_points,
        "tolerance": tolerance,
        "alpha_min": 0.5,
        "diagonal": True,
        "kind": "selection",
        "set_active": False,
    }
    if bbox_xyxy is not None:
        params["bbox_xyxy"] = bbox_xyxy
    return {
        "type": "magic_wand_select",
        "target": {"layer_id": "source_image", "mask_id": mask_id},
        "params": params,
        "description": description,
    }


def color_range_action(mask_id: str, name: str, color: str, tolerance: float, bbox_xyxy: list[int], description: str) -> dict[str, Any]:
    return {
        "type": "select_color_range",
        "target": {"layer_id": "source_image", "mask_id": mask_id},
        "params": {
            "name": name,
            "color": color,
            "tolerance": tolerance,
            "bbox_xyxy": bbox_xyxy,
            "alpha_min": 0.9,
            "kind": "selection",
            "set_active": False,
        },
        "description": description,
    }


def seeded_color_range_action(
    mask_id: str,
    name: str,
    seed_points: list[list[int]],
    exclude_seed_points: list[list[int]],
    bbox_xyxy: list[int],
    description: str,
    *,
    hue_tolerance_degrees: float = 30,
    saturation_tolerance: float = 0.55,
    value_tolerance: float = 0.60,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "name": name,
        "seed_points": seed_points,
        "bbox_xyxy": bbox_xyxy,
        "color_space": "hsv",
        "hue_tolerance_degrees": hue_tolerance_degrees,
        "saturation_tolerance": saturation_tolerance,
        "value_tolerance": value_tolerance,
        "alpha_min": 0.5,
        "kind": "selection",
        "set_active": False,
    }
    if exclude_seed_points:
        params["exclude_seed_points"] = exclude_seed_points
    return {
        "type": "select_color_range",
        "target": {"layer_id": "source_image", "mask_id": mask_id},
        "params": params,
        "description": description,
    }


def segment_object_action(
    mask_id: str,
    name: str,
    positive_seed_points: list[list[int]],
    negative_seed_points: list[list[int]],
    bbox_xyxy: list[int],
    description: str,
    *,
    tolerance: float,
    edge_stop_threshold: float,
    negative_margin: float,
) -> dict[str, Any]:
    return {
        "type": "segment_object",
        "target": {"layer_id": "source_image", "mask_id": mask_id},
        "params": {
            "name": name,
            "mode": "seeded_object",
            "positive_seed_points": positive_seed_points,
            "negative_seed_points": negative_seed_points,
            "bbox_xyxy": bbox_xyxy,
            "tolerance": tolerance,
            "edge_stop_threshold": edge_stop_threshold,
            "negative_margin": negative_margin,
            "alpha_min": 0.5,
            "diagonal": True,
            "set_active": False,
        },
        "description": description,
    }


def combine_action(mask_id: str, mask_ids: list[str], name: str, operation: str, description: str) -> dict[str, Any]:
    return {
        "type": "combine_masks",
        "target": {"mask_id": mask_id},
        "params": {"operation": operation, "mask_ids": mask_ids, "name": name},
        "description": description,
    }


def refine_action(
    mask_id: str,
    source_mask_id: str,
    name: str,
    feather_radius: float,
    description: str,
    *,
    min_area: int | None = None,
    fill_holes: bool = False,
    max_hole_area: int | None = None,
    close_pixels: int | None = None,
    open_pixels: int | None = None,
    smooth_radius: float | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"source_mask_id": source_mask_id, "name": name, "feather_radius": feather_radius, "set_active": False}
    if min_area is not None:
        params["min_area"] = min_area
    if fill_holes:
        params["fill_holes"] = True
    if max_hole_area is not None:
        params["max_hole_area"] = max_hole_area
    if close_pixels is not None:
        params["close_pixels"] = close_pixels
    if open_pixels is not None:
        params["open_pixels"] = open_pixels
    if smooth_radius is not None:
        params["smooth_radius"] = smooth_radius
    return {
        "type": "refine_selection",
        "target": {"mask_id": mask_id},
        "params": params,
        "description": description,
    }


def extract_line_art_action(mask_id: str, name: str, description: str) -> dict[str, Any]:
    return {
        "type": "extract_line_art",
        "target": {"layer_id": "source_image", "mask_id": mask_id},
        "params": {
            "mode": "ink",
            "threshold": 0.14,
            "name": name,
            "set_active": False,
        },
        "description": description,
    }


def cleanup_fringe_action(
    mask_id: str,
    source_mask_id: str,
    name: str,
    old_colors: list[str],
    bbox_xyxy: list[int],
    protect_mask_ids: list[str],
    description: str,
) -> dict[str, Any]:
    return {
        "type": "cleanup_fringe",
        "target": {"layer_id": "source_image", "mask_id": mask_id},
        "params": {
            "source_mask_id": source_mask_id,
            "name": name,
            "search_radius": 3,
            "old_colors": old_colors,
            "protect_mask_ids": protect_mask_ids,
            "bbox_xyxy": bbox_xyxy,
            "color_space": "hsv",
            "hue_tolerance_degrees": 34,
            "saturation_tolerance": 0.70,
            "value_tolerance": 0.90,
            "feather_radius": 0.25,
            "set_active": False,
        },
        "description": description,
    }


def colorize_action(mask_id: str, color: str, description: str, *, method: str = "luminance", contrast: float | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {"color": color, "amount": 1.0, "method": method}
    if contrast is not None:
        params["contrast"] = contrast
    return {
        "type": "colorize",
        "target": {"layer_id": "source_image"},
        "write_mask_id": mask_id,
        "params": params,
        "description": description,
    }


def validate_action(description: str) -> dict[str, Any]:
    return {"type": "validate", "params": {}, "description": description}


def export_flat_action(output_dir: Path, description: str) -> dict[str, Any]:
    return {"type": "export_flat", "params": {"path": "final_image.png"}, "description": description}


def export_bundle_action(output_dir: Path, description: str) -> dict[str, Any]:
    return {
        "type": "export_layered_bundle",
        "params": {"path": "layered_bundle", "include_preview": True, "include_hidden": True, "overwrite": True},
        "description": description,
    }


def run_example(example: ExampleSpec, dataset_root: Path) -> dict[str, Any]:
    example_rel = Path("data") / example.example_id
    example_dir = dataset_root / example_rel
    if example_dir.exists():
        shutil.rmtree(example_dir)
    example_dir.mkdir(parents=True)
    shutil.copy2(example.source_image, example_dir / "initial_image.png")

    old_cwd = Path.cwd()
    os.chdir(example_dir)
    try:
        logger = TraceLogger(
            Path("trace"),
            metadata={
                "task_type": "image_editing",
                "source": "interactive_controller_run",
                "split": "train",
                "planner": {"planner_type": "manual", "planner_model": None, "planner_version": None, "temperature": None},
                "rights": {
                    "source_image_license": "unknown",
                    "user_provided_content": False,
                    "allowed_for_training": True,
                    "contains_personal_data": False,
                },
                "artistic_constraints": ["preserve line art and shading", "deterministic kernel actions only"],
                "diffusion_backend": None,
                "example_number": example.example_id[-4:],
                "external_diffusion_allowed": False,
                "keep_unsuccessful_attempts": True,
            },
        )

        session = logger.start_session(example.prompt)
        logger.log_document_snapshot(example.document, "initial_source_document")
        logger.log_observation(example.document, {"observations": example.observations}, "manual_controller_observations")

        asset_refs = {
            "preview": f"previews/doc_rev_{example.document.revision:04d}_initial_source_document.png",
            "snapshot": f"snapshots/doc_rev_{example.document.revision:04d}_initial_source_document.json",
        }
        planner = AIPlanner(
            backend=StaticPlannerBackend([example.planner_output]),
            request_builder=TrainingRequestBuilder(),
            trace_sink=logger,
        )
        executor = Executor(ExecutionContext(trace_sink=logger))
        execution = planner.plan_and_execute(
            example.prompt,
            example.document,
            executor,
            observations=example.observations,
            asset_refs=asset_refs,
            metadata={"example_number": example.example_id[-4:], "source_image_path": "initial_image.png"},
        )

        report = Validator().validate_document(example.document)
        logger.log_validation_report(report, example.document)
        logger.log_document_snapshot(example.document, "final_document")

        succeeded = execution.succeeded() and not report.has_errors()
        if logger.session is None:
            raise RuntimeError("trace session was not started")
        logger.session.metadata["success"] = succeeded
        logger.session.metadata["status"] = "completed" if succeeded else "failed"
        session = logger.end_session()

        session_dir = Path("trace") / session.id
        write_json(Path("planner_request.json"), execution.planner_result.request)
        write_json(Path("planner_output.json"), execution.planner_result.raw_output)
        write_json(Path("action_batch.json"), execution.planner_result.action_batch.to_json())
        training_example = build_training_example(example, session, execution, report.has_errors(), Path("trace") / session.id)
        write_json(Path("training_example.json"), training_example)
        shutil.copy2(session_dir / "manifest.json", "manifest.json")
        shutil.copy2(session_dir / "events.jsonl", "events.jsonl")
        zip_directory_with_root(Path("layered_bundle"), Path("layered_bundle.zip"))
        zip_directory_with_root(session_dir, Path("trace_session.zip"))
        run_summary = build_run_summary(example, execution, succeeded, report.has_errors(), Path("trace") / session.id)
        write_json(Path("run_summary.json"), run_summary)
        validation = build_export_validation(example, example_dir, run_summary, training_example)
        write_json(Path("export_validation.json"), validation)
        validation = build_export_validation(example, example_dir, run_summary, training_example)
        write_json(Path("export_validation.json"), validation)
    finally:
        os.chdir(old_cwd)

    return {
        "example_id": example.example_id,
        "succeeded": succeeded,
        "final_image": str(example_rel / "final_image.png"),
        "trace": str(run_summary["trace_session_path"]),
        "validation_passed": validation["passed"],
    }


def build_training_example(
    example: ExampleSpec,
    session: TraceSession,
    execution: PlannerExecutionResult,
    validation_has_errors: bool,
    trace_path: Path,
) -> dict[str, Any]:
    request = execution.planner_result.request
    succeeded = execution.succeeded() and not validation_has_errors
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "example_id": example.example_id,
        "source_session_id": session.id,
        "example_type": "planner_imitation",
        "task_type": "image_editing",
        "split": "train",
        "input": {
            "user_prompt": example.prompt,
            "document_summary": request["document_summary"],
            "observations": request["observations"],
            "available_tools": request["available_actions"],
            "asset_refs": request["asset_refs"],
        },
        "target": {"planner_output": execution.planner_result.raw_output},
        "labels": {
            "success": succeeded,
            "validation_passed": not validation_has_errors,
            "human_accepted": None,
            "human_rating": None,
            "metrics": {},
        },
        "provenance": {"trace_path": str(trace_path), "source": "trace", "allowed_for_training": True},
    }


def build_run_summary(
    example: ExampleSpec,
    execution: PlannerExecutionResult,
    succeeded: bool,
    validation_has_errors: bool,
    trace_path: Path,
) -> dict[str, Any]:
    return {
        "user_prompt": example.prompt,
        "succeeded": succeeded,
        "validation_passed": not validation_has_errors,
        "validation_issues": [] if not validation_has_errors else ["document validation failed"],
        "source_image_provided": True,
        "diffusion_used": False,
        "actions_used": [action.type.value for action in execution.planner_result.action_batch.actions],
        "action_statuses": [result.status.value for result in execution.action_results],
        "planner_request_path": "planner_request.json",
        "planner_output_path": "planner_output.json",
        "action_batch_path": "action_batch.json",
        "manifest_path": "manifest.json",
        "events_path": "events.jsonl",
        "training_example_path": "training_example.json",
        "final_image_path": "final_image.png",
        "layered_bundle_path": "layered_bundle",
        "trace_session_path": str(trace_path),
        "notes": example.notes,
    }


def build_export_validation(
    example: ExampleSpec,
    example_dir: Path,
    run_summary: dict[str, Any],
    training_example: dict[str, Any],
) -> dict[str, Any]:
    dataset_root = example_dir.parents[1]
    returned = [
        f"data/{example.example_id}/initial_image.png",
        f"data/{example.example_id}/final_image.png",
        f"data/{example.example_id}/planner_request.json",
        f"data/{example.example_id}/planner_output.json",
        f"data/{example.example_id}/action_batch.json",
        f"data/{example.example_id}/manifest.json",
        f"data/{example.example_id}/events.jsonl",
        f"data/{example.example_id}/training_example.json",
        f"data/{example.example_id}/run_summary.json",
        f"data/{example.example_id}/export_validation.json",
        f"data/{example.example_id}/layered_bundle.zip",
        f"data/{example.example_id}/trace_session.zip",
    ]
    stale_hits = find_text_hits(example_dir, ["/mnt/data"])
    secret_hits = find_text_hits(example_dir, ["Bearer ", "YOUR_API_KEY", "venice-key"])
    other_example_hits = find_other_example_hits(example_dir, example.example_id)
    checks = {
        "does_not_reference_other_examples": not other_example_hits,
        "final_image_exists": (example_dir / "final_image.png").exists(),
        "initial_image_present_if_user_provided": (example_dir / "initial_image.png").exists(),
        "layered_bundle_exists": (example_dir / "layered_bundle" / "manifest.json").exists(),
        "manifest_success_true": manifest_success(example_dir / "manifest.json"),
        "no_api_keys_or_bearer_tokens": not secret_hits,
        "no_mnt_data_paths_remain": not stale_hits,
        "returned_artifacts_exist": all((dataset_root / path).exists() for path in returned),
        "trace_has_initial_preview": trace_asset_exists(example_dir, "previews", "doc_rev_0000_initial"),
        "trace_has_initial_snapshot": trace_asset_exists(example_dir, "snapshots", "doc_rev_0000_initial"),
        "trace_session_exists": any((example_dir / "trace").glob("session_*")),
        "training_example_target_has_no_action_batch": "action_batch" not in training_example["target"],
        "training_example_target_is_planner_output": list(training_example["target"]) == ["planner_output"],
        "training_example_uses_initial_input": training_example["input"]["document_summary"]["revision"] == 0,
        "validation_passed_true": bool(run_summary["validation_passed"]),
    }
    return {
        "schema_version": "ai_edit_export_validation.v1",
        "example_id": example.example_id,
        "passed": all(checks.values()),
        "checks": checks,
        "returned_artifacts": returned,
        "stale_path_hits": stale_hits,
        "secret_hits": secret_hits,
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def zip_directory_with_root(source_dir: Path, output_path: Path) -> None:
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for directory in sorted(path for path in source_dir.rglob("*") if path.is_dir()):
            archive.write(directory, directory.relative_to(source_dir.parent))
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source_dir.parent))


def manifest_success(path: Path) -> bool:
    if not path.exists():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("success") is True


def trace_asset_exists(example_dir: Path, subdir: str, prefix: str) -> bool:
    trace_root = example_dir / "trace"
    sessions = list(trace_root.glob("session_*"))
    return bool(sessions) and any((session / subdir).glob(f"{prefix}*") for session in sessions)


def find_text_hits(root: Path, needles: list[str]) -> list[str]:
    hits: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() in {".png", ".jpg", ".jpeg", ".zip"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for needle in needles:
            if needle in text:
                hits.append(str(path.relative_to(root)))
                break
    return hits


def find_other_example_hits(root: Path, current_example_id: str) -> list[str]:
    pattern = re.compile(r"training\d{4}")
    hits: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() in {".png", ".jpg", ".jpeg", ".zip"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        matches = {match.group(0) for match in pattern.finditer(text)}
        if matches - {current_example_id}:
            hits.append(str(path.relative_to(root)))
    return hits


if __name__ == "__main__":
    raise SystemExit(main())
