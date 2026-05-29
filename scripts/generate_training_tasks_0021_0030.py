#!/usr/bin/env python3
"""Generate local planner/executor training examples for tasks 0021-0030."""

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
    """Keep these local examples close to the existing compact request shape."""

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
        "training0021": build_example_0021,
        "training0022": build_example_0022,
        "training0023": build_example_0023,
        "training0024": build_example_0024,
        "training0025": build_example_0025,
        "training0026": build_example_0026,
        "training0027": build_example_0027,
        "training0028": build_example_0028,
        "training0029": build_example_0029,
        "training0030": build_example_0030,
    }
    selected = args.examples or list(factories)
    summaries = []
    for example_id in selected:
        if example_id not in factories:
            raise ValueError(f"unknown example id {example_id!r}")
        summaries.append(run_example(factories[example_id](dataset_root, args), dataset_root))

    success = all(item["succeeded"] and item["validation_passed"] for item in summaries)
    print(json.dumps({"success": success, "examples": summaries}, indent=2))
    return 0 if success else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate local deterministic AI editing examples 0021-0030.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/algebraity/Programming/aie-training-data"))
    parser.add_argument("--images-dir", type=Path, default=Path("/home/algebraity/Programming/aie-training-data/images"))
    parser.add_argument("--examples", nargs="*", default=None, help="Optional subset, e.g. training0021 training0030.")
    return parser.parse_args()


def build_example_0021(dataset_root: Path, args: argparse.Namespace) -> ExampleSpec:
    prompt = "color her scarf light blue while keeping the black line art and shading intact"
    source = args.images_dir / "anime_girl.jpg"
    document, width, height = source_document("doc_training0021_initial", source, "Initial source document")
    scarf_bbox = [90, 485, 850, 1202]
    return image_overlay_example(
        "training0021",
        prompt,
        source,
        document,
        width,
        height,
        [
            polygon_select_action("mask_scarf_neck_bounds", "Neck scarf bounds", [[250, 585], [340, 545], [505, 525], [605, 565], [580, 640], [450, 675], [300, 655]], "Bound the gray neck scarf below the mouth without including the face or jacket."),
            polygon_select_action("mask_scarf_left_bounds", "Left scarf tail bounds", [[215, 635], [302, 670], [355, 1201], [100, 1201], [150, 1130], [190, 1000], [180, 820], [195, 690]], "Bound the left dangling scarf tail, including the lower-left end while excluding nearby music notes."),
            polygon_select_action("mask_scarf_right_bounds", "Right scarf tail bounds", [[585, 535], [650, 590], [760, 780], [842, 1040], [850, 1201], [650, 1201], [690, 1030], [700, 880], [690, 760], [650, 660]], "Bound the right dangling scarf tail while excluding the decorative star and note cluster in the inner gap."),
            combine_action("mask_scarf_bounds", ["mask_scarf_neck_bounds", "mask_scarf_left_bounds", "mask_scarf_right_bounds"], "Combined scarf bounds", "union", "Union the hand-bounded scarf regions."),
            color_range_action("mask_scarf_dark", "Dark gray scarf tones", "#787878", 0.15, scarf_bbox, "Select darker gray scarf tones."),
            color_range_action("mask_scarf_mid", "Mid gray scarf tones", "#A2A2A2", 0.15, scarf_bbox, "Select mid-gray scarf tones."),
            color_range_action("mask_scarf_light", "Light gray scarf tones", "#D2D2D2", 0.10, scarf_bbox, "Select light gray scarf tones without selecting white background."),
            combine_action("mask_scarf_tones", ["mask_scarf_dark", "mask_scarf_mid", "mask_scarf_light"], "Combined scarf gray tones", "union", "Union the scarf tone selections."),
            combine_action("mask_scarf", ["mask_scarf_bounds", "mask_scarf_tones"], "Bounded scarf tone mask", "intersect", "Keep only gray scarf-tone pixels inside the bounded scarf regions."),
            shape_mask_action("mask_right_gap_upper_decor", "Upper right background decoration guard", {"type": "ellipse", "bbox_xyxy": [650, 610, 760, 700]}, "Protect the nearby background star and note outlines from the scarf recolor."),
            shape_mask_action("mask_right_gap_lower_decor", "Lower right background decoration guard", {"type": "ellipse", "bbox_xyxy": [710, 700, 805, 780]}, "Protect the nearby lower background star outlines from the scarf recolor."),
            combine_action("mask_scarf_background_decor", ["mask_right_gap_upper_decor", "mask_right_gap_lower_decor"], "Protected background decorations near scarf", "union", "Union the background decoration guard masks."),
            combine_action("mask_scarf_clean", ["mask_scarf", "mask_scarf_background_decor"], "Scarf mask without background decorations", "subtract", "Remove nearby background decorations from the scarf recolor mask."),
            colorize_action("mask_scarf_clean", "#6CB6FF", "Color the selected scarf tones light blue while preserving black line art, white highlights, and nearby gray decorations.", method="luminance"),
        ],
        [
            source_observation(width, height),
            region_observation("The scarf is selected by intersecting gray tone ranges with bounded scarf silhouettes so the lower-left tail is included and background decorations are excluded.", scarf_bbox),
        ],
        "Recolor the grayscale scarf to light blue while preserving line art.",
    )


def build_example_0022(dataset_root: Path, args: argparse.Namespace) -> ExampleSpec:
    prompt = "change the red hair bow to purple while preserving its highlights and shadows"
    source = args.images_dir / "asian_elf.jpg"
    document, width, height = source_document("doc_training0022_initial", source, "Initial source document")
    bow_bbox = [380, 0, 610, 130]
    bow_seeds = [[465, 28], [505, 58], [430, 70], [530, 38]]
    return image_overlay_example(
        "training0022",
        prompt,
        source,
        document,
        width,
        height,
        [
            seeded_color_range_action("mask_hair_bow", "Red hair bow material", bow_seeds, [], bow_bbox, "Select the red hair bow material inside a tight top-of-head bounding box."),
            colorize_action("mask_hair_bow", "#8A2BE2", "Recolor the red hair bow to purple while preserving its painted light and shadow.", method="set_hue_preserve_lightness"),
        ],
        [
            source_observation(width, height),
            region_observation("The red bow on top of the head is selected by seeded HSV color range and recolored purple.", bow_bbox),
        ],
        "Recolor the red hair bow to purple while preserving highlights and shadows.",
    )


def build_example_0023(dataset_root: Path, args: argparse.Namespace) -> ExampleSpec:
    prompt = "change the crescent moon in the sky to soft pink"
    source = args.images_dir / "bear.jpg"
    document, width, height = source_document("doc_training0023_initial", source, "Initial source document")
    moon_bbox = [780, 270, 920, 420]
    return image_overlay_example(
        "training0023",
        prompt,
        source,
        document,
        width,
        height,
        [
            color_range_action("mask_moon_yellow", "Moon yellow pixels", "#F8E797", 0.12, moon_bbox, "Select the yellow crescent moon pixels without selecting the blue sky."),
            colorize_action("mask_moon_yellow", "#FF8FCB", "Change the crescent moon to soft pink while keeping its soft shading.", method="set_hue_preserve_lightness"),
        ],
        [
            source_observation(width, height),
            region_observation("The crescent moon in the upper-right sky is selected by yellow color range and recolored soft pink.", moon_bbox),
        ],
        "Recolor the crescent moon to soft pink.",
    )


def build_example_0024(dataset_root: Path, args: argparse.Namespace) -> ExampleSpec:
    prompt = "cross out the word BEEF with a red slash and underline PORK? in red"
    source = args.images_dir / "beef_or_pork.png"
    document, width, height = source_document("doc_training0024_initial", source, "Initial source document")
    underline_bbox = [252, 68, 448, 77]
    return image_overlay_example(
        "training0024",
        prompt,
        source,
        document,
        width,
        height,
        [
            create_shape_layer_action("layer_red_marks", "red title marks", "Create a transparent overlay layer for the red title marks."),
            brush_stroke_action("layer_red_marks", [[18, 58], [182, 20]], "#e60022", 8, "Draw a centered red diagonal slash through the middle of the word BEEF."),
            draw_shape_action("layer_red_marks", {"type": "rectangle", "bbox_xyxy": underline_bbox, "corner_radius": 0}, None, {"color": "#e60022"}, "Draw a red underline beneath the word PORK?."),
        ],
        [
            source_observation(width, height),
            region_observation("The title is edited with a red slash through BEEF and a red underline under PORK?.", [25, 25, 446, 80]),
        ],
        "Add red shape marks that cross out BEEF and underline PORK?.",
    )


def build_example_0025(dataset_root: Path, args: argparse.Namespace) -> ExampleSpec:
    prompt = "change the bow at her collar to deep blue while preserving its folds"
    source = args.images_dir / "big_chest_girl.jpg"
    document, width, height = source_document("doc_training0025_initial", source, "Initial source document")
    bow_bbox = [220, 410, 445, 535]
    bow_seeds = [[350, 430], [360, 470], [280, 420], [305, 455], [325, 500]]
    return image_overlay_example(
        "training0025",
        prompt,
        source,
        document,
        width,
        height,
        [
            seeded_color_range_action("mask_collar_bow", "Collar bow red-brown material", bow_seeds, [[315, 415], [410, 430], [320, 535]], bow_bbox, "Select the red-brown collar bow material in the lower collar area while excluding skin and blouse pixels."),
            colorize_action("mask_collar_bow", "#1F5CFF", "Recolor the collar bow deep blue while preserving folds and darker line work.", method="set_hue_preserve_lightness"),
        ],
        [
            source_observation(width, height),
            region_observation("The red-brown collar bow is selected in a tight chest-area box and recolored blue.", bow_bbox),
        ],
        "Recolor the collar bow to deep blue while preserving fold shading.",
    )


def build_example_0026(dataset_root: Path, args: argparse.Namespace) -> ExampleSpec:
    prompt = "change the cow-print suit's black spots to brown while keeping the white fabric and line art the same"
    source = args.images_dir / "cow_woman.jpg"
    document, width, height = source_document("doc_training0026_initial", source, "Initial source document")
    suit_bbox = [285, 500, 765, 1181]
    return image_overlay_example(
        "training0026",
        prompt,
        source,
        document,
        width,
        height,
        [
            color_range_action("mask_cow_spots_gray", "Gray cow-print spot interiors", "#685A59", 0.18, suit_bbox, "Select gray interiors of the existing cow-print spots."),
            color_range_action("mask_cow_spots_dark", "Dark cow-print spot interiors", "#503C3D", 0.16, suit_bbox, "Select darker interiors of the existing cow-print spots."),
            combine_action("mask_cow_spots", ["mask_cow_spots_gray", "mask_cow_spots_dark"], "Combined cow-print spot mask", "union", "Combine the gray and dark cow-print spot interiors."),
            paint_bucket_fill_action("source_image", "mask_cow_spots", "#8A5833", "Fill the selected cow-print spots with visible warm brown."),
        ],
        [
            source_observation(width, height),
            region_observation("The existing gray and dark cow-print spot interiors on the suit and stockings are selected and recolored brown.", suit_bbox),
        ],
        "Recolor the cow-print spots from black to brown while preserving white fabric and line art.",
    )


def build_example_0027(dataset_root: Path, args: argparse.Namespace) -> ExampleSpec:
    prompt = "remove the two speech bubbles by painting them white"
    source = args.images_dir / "elf_girl.jpg"
    document, width, height = source_document("doc_training0027_initial", source, "Initial source document")
    left_bubble = [0, 20, 325, 305]
    right_bubble = [585, 240, 850, 485]
    return image_overlay_example(
        "training0027",
        prompt,
        source,
        document,
        width,
        height,
        [
            draw_shape_action("source_image", {"type": "rectangle", "bbox_xyxy": left_bubble, "corner_radius": 0}, None, {"color": "#FFFFFF"}, "Paint a white rectangle over the full left speech bubble and tail."),
            draw_shape_action("source_image", {"type": "rectangle", "bbox_xyxy": right_bubble, "corner_radius": 0}, None, {"color": "#FFFFFF"}, "Paint a white rectangle over the full right speech bubble and tail."),
        ],
        [
            source_observation(width, height),
            region_observation("Both speech bubbles are covered with white ellipse fills to remove the text and outlines.", [35, 55, 815, 455]),
        ],
        "Remove the speech bubbles by filling their regions with white.",
    )


def build_example_0028(dataset_root: Path, args: argparse.Namespace) -> ExampleSpec:
    prompt = "change the blue halo behind her head to gold while keeping its shape the same"
    source = args.images_dir / "halo_girl.jpg"
    document, width, height = source_document("doc_training0028_initial", source, "Initial source document")
    halo_bbox = [75, 0, 430, 175]
    return image_overlay_example(
        "training0028",
        prompt,
        source,
        document,
        width,
        height,
        [
            color_range_action("mask_halo_deep_blue", "Deep blue halo arcs", "#294A68", 0.18, halo_bbox, "Select the deepest blue halo arc segments behind the head."),
            color_range_action("mask_halo_mid_blue", "Mid blue halo arcs", "#3E7898", 0.20, halo_bbox, "Select the mid-blue antialiased halo arc pixels."),
            color_range_action("mask_halo_cyan", "Cyan halo ring", "#35DCE5", 0.16, halo_bbox, "Select the bright cyan inner halo ring."),
            combine_action("mask_halo", ["mask_halo_deep_blue", "mask_halo_mid_blue", "mask_halo_cyan"], "Combined blue halo mask", "union", "Combine the blue and cyan halo selections."),
            colorize_action("mask_halo", "#F0B400", "Recolor the halo gold while preserving the original ring shapes and opacity.", method="set_hue_preserve_lightness"),
        ],
        [
            source_observation(width, height),
            region_observation("The blue halo graphic behind the head is selected by color range and recolored gold without touching the head.", halo_bbox),
        ],
        "Recolor the blue halo behind the head to gold.",
    )


def build_example_0029(dataset_root: Path, args: argparse.Namespace) -> ExampleSpec:
    prompt = "change the central yellow sunflower in the bouquet to pink"
    source = args.images_dir / "horse_wife.jpg"
    document, width, height = source_document("doc_training0029_initial", source, "Initial source document")
    flower_bbox = [360, 885, 535, 1065]
    return image_overlay_example(
        "training0029",
        prompt,
        source,
        document,
        width,
        height,
        [
            color_range_action("mask_sunflower_yellow", "Sunflower yellow petals", "#F8E998", 0.16, flower_bbox, "Select the central sunflower's yellow petals."),
            color_range_action("mask_sunflower_gold", "Sunflower golden center", "#AD802B", 0.15, flower_bbox, "Select the central sunflower's golden center tones."),
            combine_action("mask_sunflower", ["mask_sunflower_yellow", "mask_sunflower_gold"], "Combined sunflower mask", "union", "Combine the sunflower petal and center masks."),
            colorize_action("mask_sunflower", "#FF7AB6", "Recolor the central sunflower to pink while preserving its light and dark detail.", method="set_hue_preserve_lightness"),
        ],
        [
            source_observation(width, height),
            region_observation("The central yellow sunflower in the bouquet is selected by color range and recolored pink.", flower_bbox),
        ],
        "Recolor the central yellow sunflower to pink.",
    )


def build_example_0030(dataset_root: Path, args: argparse.Namespace) -> ExampleSpec:
    prompt = "crop the image to focus on the large girl on the right, removing the smaller girl and desk on the left"
    source = args.images_dir / "imagination_girl.png"
    document, width, height = source_document("doc_training0030_initial", source, "Initial source document")
    crop_bbox = [360, 0, width, height]
    return image_overlay_example(
        "training0030",
        prompt,
        source,
        document,
        width,
        height,
        [
            crop_action(crop_bbox, "Crop the document to the right side so the large girl is the focus and the smaller girl and desk are removed."),
        ],
        [
            source_observation(width, height),
            region_observation("The crop keeps the large right-side character and removes the smaller left-side character and desk.", crop_bbox),
        ],
        "Crop the image to focus on the large girl on the right.",
    )


def image_overlay_example(
    example_id: str,
    prompt: str,
    source: Path,
    document: DocumentState,
    width: int,
    height: int,
    edit_actions: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    description: str,
) -> ExampleSpec:
    output_dir = Path("data") / example_id
    return ExampleSpec(
        example_id=example_id,
        prompt=prompt,
        source_image=source,
        document=document,
        observations=observations,
        planner_output={
            "schema_version": "ai_edit_planner_output.v1",
            "description": description,
            "stop_on_error": True,
            "metadata": {"example_id": example_id, "external_diffusion_used": False},
            "actions": [
                *edit_actions,
                validate_action("Validate that the edited image has no structural issues."),
                export_flat_action(output_dir, "Export the flattened edited result."),
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


def source_observation(width: int, height: int) -> dict[str, Any]:
    return {
        "type": "source_image",
        "summary": f"User-provided image, {width}x{height} pixels.",
        "details": {"width": width, "height": height, "source_image_path": "initial_image.png"},
    }


def layout_observation(summary: str, positions: list[list[int]], sizes: list[int]) -> dict[str, Any]:
    return {
        "type": "layout",
        "summary": summary,
        "details": {"positions_xy": positions, "font_sizes": sizes},
    }


def region_observation(summary: str, bbox_xyxy: list[int]) -> dict[str, Any]:
    return {
        "type": "region",
        "summary": summary,
        "details": {"bbox_xyxy": bbox_xyxy},
    }


def create_text_action(layer_id: str, name: str, text: str, center_xy: list[int], font_size: int, color: str, description: str) -> dict[str, Any]:
    return {
        "type": "create_text_layer",
        "target": {"output_layer_id": layer_id},
        "params": {
            "name": name,
            "text": text,
            "x": center_xy[0],
            "y": center_xy[1],
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


def create_shape_layer_action(layer_id: str, name: str, description: str) -> dict[str, Any]:
    return {
        "type": "create_layer",
        "target": {"output_layer_id": layer_id},
        "params": {"name": name, "kind": "raster", "color": "#00000000", "set_active": False},
        "description": description,
    }


def draw_shape_action(
    layer_id: str,
    shape: dict[str, Any],
    stroke: dict[str, Any] | None,
    fill: dict[str, Any] | None,
    description: str,
) -> dict[str, Any]:
    return {
        "type": "draw_shape",
        "target": {"layer_id": layer_id},
        "params": {"shape": shape, "stroke": stroke, "fill": fill},
        "description": description,
    }


def polygon_select_action(mask_id: str, name: str, points: list[list[int]], description: str) -> dict[str, Any]:
    return {
        "type": "select_polygon",
        "target": {"mask_id": mask_id},
        "params": {"name": name, "points": points, "closed": True, "kind": "selection", "set_active": False},
        "description": description,
    }


def shape_mask_action(mask_id: str, name: str, shape: dict[str, Any], description: str) -> dict[str, Any]:
    return {
        "type": "create_mask_from_shape",
        "target": {"mask_id": mask_id},
        "params": {"name": name, "shape": shape, "kind": "write_guard", "set_active": False},
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
    hue_tolerance_degrees: float = 32,
    saturation_tolerance: float = 0.55,
    value_tolerance: float = 0.60,
    edge_stop_threshold: float | None = None,
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
    if edge_stop_threshold is not None:
        params["edge_stop_threshold"] = edge_stop_threshold
    return {
        "type": "magic_wand_select" if edge_stop_threshold is not None else "select_color_range",
        "target": {"layer_id": "source_image", "mask_id": mask_id},
        "params": params,
        "description": description,
    }


def combine_action(mask_id: str, mask_ids: list[str], name: str, operation: str, description: str) -> dict[str, Any]:
    return {
        "type": "combine_masks",
        "target": {"mask_id": mask_id},
        "params": {"operation": operation, "mask_ids": mask_ids, "name": name},
        "description": description,
    }


def colorize_action(mask_id: str, color: str, description: str, *, method: str = "luminance") -> dict[str, Any]:
    return {
        "type": "colorize",
        "target": {"layer_id": "source_image"},
        "write_mask_id": mask_id,
        "params": {"color": color, "amount": 1.0, "method": method},
        "description": description,
    }


def paint_bucket_fill_action(layer_id: str, mask_id: str, color: str, description: str) -> dict[str, Any]:
    return {
        "type": "paint_bucket_fill",
        "target": {"layer_id": layer_id},
        "write_mask_id": mask_id,
        "params": {"color": color, "mode": "replace_rgb_preserve_alpha"},
        "description": description,
    }


def brush_stroke_action(
    layer_id: str,
    points: list[list[int]],
    color: str,
    width: float,
    description: str,
    *,
    mode: str = "source_over",
) -> dict[str, Any]:
    return {
        "type": "brush_stroke",
        "target": {"layer_id": layer_id},
        "params": {"points": points, "color": color, "width": width, "opacity": 1.0, "mode": mode},
        "description": description,
    }


def crop_action(bbox_xyxy: list[int], description: str) -> dict[str, Any]:
    return {
        "type": "crop",
        "params": {"bbox_xyxy": bbox_xyxy, "scope": "document"},
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
                "artistic_constraints": ["preserve existing image content unless the prompt asks for an overlay"],
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
        training_example = build_training_example(example, session, execution, report.has_errors(), session_dir)
        write_json(Path("training_example.json"), training_example)
        shutil.copy2(session_dir / "manifest.json", "manifest.json")
        shutil.copy2(session_dir / "events.jsonl", "events.jsonl")
        zip_directory_with_root(Path("layered_bundle"), Path("layered_bundle.zip"))
        zip_directory_with_root(session_dir, Path("trace_session.zip"))
        run_summary = build_run_summary(example, execution, succeeded, report.has_errors(), session_dir)
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
        "other_example_hits": other_example_hits,
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
