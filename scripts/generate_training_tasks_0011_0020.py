#!/usr/bin/env python3
"""Generate local planner/executor training examples for tasks 0011-0020."""

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
        "training0011": build_example_0011,
        "training0012": build_example_0012,
        "training0013": build_example_0013,
        "training0014": build_example_0014,
        "training0015": build_example_0015,
        "training0016": build_example_0016,
        "training0017": build_example_0017,
        "training0018": build_example_0018,
        "training0019": build_example_0019,
        "training0020": build_example_0020,
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
    parser = argparse.ArgumentParser(description="Generate local deterministic AI editing examples 0011-0020.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/algebraity/Programming/aie-training-data"))
    parser.add_argument("--images-dir", type=Path, default=Path("/home/algebraity/Programming/aie-training-data/images"))
    parser.add_argument("--examples", nargs="*", default=None, help="Optional subset, e.g. training0011 training0020.")
    return parser.parse_args()


def build_example_0011(dataset_root: Path, args: argparse.Namespace) -> ExampleSpec:
    prompt = "add three pink hearts around the girl's head"
    source = args.images_dir / "anime_girl.jpg"
    document, width, height = source_document("doc_training0011_initial", source, "Initial source document")
    positions = [[240, 160], [515, 95], [660, 185]]
    sizes = [58, 44, 50]
    return image_overlay_example(
        "training0011",
        prompt,
        source,
        document,
        width,
        height,
        [
            create_text_action("layer_heart_left", "left pink heart", "\u2665", positions[0], sizes[0], "#ff5aa5", "Create a pink heart to the upper-left of the girl's head."),
            rotate_action("layer_heart_left", -12, positions[0], "Rotate the left heart slightly counter-clockwise."),
            create_text_action("layer_heart_top", "top pink heart", "\u2665", positions[1], sizes[1], "#ff77bb", "Create a smaller pink heart above the girl's head."),
            rotate_action("layer_heart_top", 8, positions[1], "Rotate the top heart slightly clockwise."),
            create_text_action("layer_heart_right", "right pink heart", "\u2665", positions[2], sizes[2], "#ff5aa5", "Create a pink heart to the upper-right of the girl's head."),
            rotate_action("layer_heart_right", -6, positions[2], "Rotate the right heart slightly counter-clockwise."),
        ],
        [
            source_observation(width, height),
            layout_observation("Three pink heart stickers are placed around the girl's head without covering her face.", positions, sizes),
        ],
        "Add three pink heart overlays around the line-art girl's head.",
    )


def build_example_0012(dataset_root: Path, args: argparse.Namespace) -> ExampleSpec:
    prompt = "circle the red hair bow with a thin blue outline"
    source = args.images_dir / "asian_elf.jpg"
    document, width, height = source_document("doc_training0012_initial", source, "Initial source document")
    bow_bbox = [390, 0, 640, 145]
    return image_overlay_example(
        "training0012",
        prompt,
        source,
        document,
        width,
        height,
        [
            create_shape_layer_action("layer_bow_circle", "blue bow circle", "Create a transparent overlay layer for the bow outline."),
            draw_shape_action("layer_bow_circle", {"type": "ellipse", "bbox_xyxy": bow_bbox}, {"color": "#1e7cff", "width": 6}, None, "Draw a thin blue ellipse around the red hair bow."),
        ],
        [
            source_observation(width, height),
            region_observation("The red bow near the top-right of the head is circled with a blue ellipse.", bow_bbox),
        ],
        "Draw a blue ellipse around the character's red hair bow.",
    )


def build_example_0013(dataset_root: Path, args: argparse.Namespace) -> ExampleSpec:
    prompt = "add three yellow stars around the moon in the sky"
    source = args.images_dir / "bear.jpg"
    document, width, height = source_document("doc_training0013_initial", source, "Initial source document")
    positions = [[835, 265], [925, 395], [760, 470]]
    sizes = [62, 42, 38]
    return image_overlay_example(
        "training0013",
        prompt,
        source,
        document,
        width,
        height,
        [
            create_text_action("layer_star_large", "large yellow star", "\u2605", positions[0], sizes[0], "#fff35c", "Create a large yellow star near the moon."),
            rotate_action("layer_star_large", -10, positions[0], "Rotate the large star slightly left."),
            create_text_action("layer_star_medium", "medium yellow star", "\u2605", positions[1], sizes[1], "#ffeb4d", "Create a medium yellow star to the right of the moon."),
            rotate_action("layer_star_medium", 11, positions[1], "Rotate the medium star slightly right."),
            create_text_action("layer_star_small", "small yellow star", "\u2605", positions[2], sizes[2], "#fff58a", "Create a smaller yellow star below the moon."),
            rotate_action("layer_star_small", -5, positions[2], "Rotate the small star slightly left."),
        ],
        [
            source_observation(width, height),
            layout_observation("Three yellow star stickers are placed around the moon in the upper sky.", positions, sizes),
        ],
        "Add three yellow star overlays around the existing moon.",
    )


def build_example_0014(dataset_root: Path, args: argparse.Namespace) -> ExampleSpec:
    prompt = "underline the word PORK? in red"
    source = args.images_dir / "beef_or_pork.png"
    document, width, height = source_document("doc_training0014_initial", source, "Initial source document")
    underline_bbox = [250, 70, 446, 80]
    return image_overlay_example(
        "training0014",
        prompt,
        source,
        document,
        width,
        height,
        [
            create_shape_layer_action("layer_pork_underline", "red PORK underline", "Create a transparent overlay layer for the red underline."),
            draw_shape_action("layer_pork_underline", {"type": "rectangle", "bbox_xyxy": underline_bbox, "corner_radius": 0}, None, {"color": "#e60022"}, "Draw a red underline beneath the word PORK?."),
        ],
        [
            source_observation(width, height),
            region_observation("The word PORK? in the top title is underlined with a red rectangle.", underline_bbox),
        ],
        "Underline the word PORK? with a red bar.",
    )


def build_example_0015(dataset_root: Path, args: argparse.Namespace) -> ExampleSpec:
    prompt = "add a small golden star sticker near her raised hand"
    source = args.images_dir / "big_chest_girl.jpg"
    document, width, height = source_document("doc_training0015_initial", source, "Initial source document")
    position = [122, 275]
    return image_overlay_example(
        "training0015",
        prompt,
        source,
        document,
        width,
        height,
        [
            create_text_action("layer_hand_star", "gold hand star", "\u2605", position, 54, "#ffd24a", "Create a golden star near the raised hand."),
            rotate_action("layer_hand_star", -12, position, "Rotate the golden star slightly counter-clockwise."),
        ],
        [
            source_observation(width, height),
            layout_observation("A golden star sticker is placed near the raised hand without covering the face.", [position], [54]),
        ],
        "Add a golden star sticker beside the raised hand.",
    )


def build_example_0016(dataset_root: Path, args: argparse.Namespace) -> ExampleSpec:
    prompt = "add two pink hearts above the cow woman's head"
    source = args.images_dir / "cow_woman.jpg"
    document, width, height = source_document("doc_training0016_initial", source, "Initial source document")
    positions = [[410, 120], [535, 145]]
    sizes = [58, 46]
    return image_overlay_example(
        "training0016",
        prompt,
        source,
        document,
        width,
        height,
        [
            create_text_action("layer_cow_heart_left", "left pink heart", "\u2665", positions[0], sizes[0], "#ff5aa5", "Create a pink heart above the character's head."),
            rotate_action("layer_cow_heart_left", -10, positions[0], "Rotate the left heart slightly left."),
            create_text_action("layer_cow_heart_right", "right pink heart", "\u2665", positions[1], sizes[1], "#ff77bb", "Create a second pink heart above the character's head."),
            rotate_action("layer_cow_heart_right", 8, positions[1], "Rotate the right heart slightly right."),
        ],
        [
            source_observation(width, height),
            layout_observation("Two pink hearts are placed above the character's head in the empty white area.", positions, sizes),
        ],
        "Add two pink heart overlays above the character's head.",
    )


def build_example_0017(dataset_root: Path, args: argparse.Namespace) -> ExampleSpec:
    prompt = "add a thin blue border around the whole comic image"
    source = args.images_dir / "elf_girl.jpg"
    document, width, height = source_document("doc_training0017_initial", source, "Initial source document")
    border_bbox = [6, 6, width - 6, height - 6]
    return image_overlay_example(
        "training0017",
        prompt,
        source,
        document,
        width,
        height,
        [
            create_shape_layer_action("layer_blue_border", "blue comic border", "Create a transparent overlay layer for the full-image border."),
            draw_shape_action("layer_blue_border", {"type": "rectangle", "bbox_xyxy": border_bbox, "corner_radius": 0}, {"color": "#2f80ff", "width": 8}, None, "Draw a thin blue border around the whole image."),
        ],
        [
            source_observation(width, height),
            region_observation("A blue rectangular frame is drawn just inside the image edge.", border_bbox),
        ],
        "Add a thin blue rectangular border around the comic-style image.",
    )


def build_example_0018(dataset_root: Path, args: argparse.Namespace) -> ExampleSpec:
    prompt = "draw a thin golden oval around the blue halo behind her head"
    source = args.images_dir / "halo_girl.jpg"
    document, width, height = source_document("doc_training0018_initial", source, "Initial source document")
    halo_bbox = [110, 0, 425, 215]
    return image_overlay_example(
        "training0018",
        prompt,
        source,
        document,
        width,
        height,
        [
            create_shape_layer_action("layer_halo_outline", "gold halo outline", "Create a transparent overlay layer for the halo outline."),
            draw_shape_action("layer_halo_outline", {"type": "ellipse", "bbox_xyxy": halo_bbox}, {"color": "#ffd84d", "width": 8}, None, "Draw a golden oval around the existing halo."),
        ],
        [
            source_observation(width, height),
            region_observation("The existing blue halo graphic behind the character's head is emphasized with a golden oval.", halo_bbox),
        ],
        "Outline the existing blue halo behind the character's head with a thin golden ellipse.",
    )


def build_example_0019(dataset_root: Path, args: argparse.Namespace) -> ExampleSpec:
    prompt = "add three golden sparkles around her crown"
    source = args.images_dir / "horse_wife.jpg"
    document, width, height = source_document("doc_training0019_initial", source, "Initial source document")
    positions = [[335, 315], [440, 245], [535, 325]]
    sizes = [50, 42, 48]
    return image_overlay_example(
        "training0019",
        prompt,
        source,
        document,
        width,
        height,
        [
            create_text_action("layer_sparkle_left", "left gold sparkle", "\u2726", positions[0], sizes[0], "#ffd95a", "Create a golden sparkle to the left of the crown."),
            rotate_action("layer_sparkle_left", -8, positions[0], "Rotate the left sparkle slightly left."),
            create_text_action("layer_sparkle_top", "top gold sparkle", "\u2726", positions[1], sizes[1], "#ffe27a", "Create a golden sparkle above the crown."),
            rotate_action("layer_sparkle_top", 6, positions[1], "Rotate the top sparkle slightly right."),
            create_text_action("layer_sparkle_right", "right gold sparkle", "\u2726", positions[2], sizes[2], "#ffd95a", "Create a golden sparkle to the right of the crown."),
            rotate_action("layer_sparkle_right", 10, positions[2], "Rotate the right sparkle slightly right."),
        ],
        [
            source_observation(width, height),
            layout_observation("Three golden sparkle stickers are arranged around the crown area.", positions, sizes),
        ],
        "Add three golden sparkle overlays around the crown.",
    )


def build_example_0020(dataset_root: Path, args: argparse.Namespace) -> ExampleSpec:
    prompt = "add a red heart above the smaller girl in the lower-left corner"
    source = args.images_dir / "imagination_girl.png"
    document, width, height = source_document("doc_training0020_initial", source, "Initial source document")
    position = [195, 505]
    return image_overlay_example(
        "training0020",
        prompt,
        source,
        document,
        width,
        height,
        [
            create_text_action("layer_small_girl_heart", "red heart above small girl", "\u2665", position, 70, "#e60022", "Create a red heart above the smaller girl in the lower-left area."),
            rotate_action("layer_small_girl_heart", -8, position, "Rotate the red heart slightly counter-clockwise."),
        ],
        [
            source_observation(width, height),
            layout_observation("A red heart is placed above the smaller girl in the lower-left corner area.", [position], [70]),
        ],
        "Add a red heart above the smaller girl in the lower-left corner.",
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
