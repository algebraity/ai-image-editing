#!/usr/bin/env python3
"""Generate local planner/executor training examples for tasks 0002-0005."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_edit_kernel.document.document_state import CanvasSpec, DocumentMetadata, DocumentState
from ai_edit_kernel.document.layer import Layer
from ai_edit_kernel.planning.planner import AIPlanner, PlannerExecutionResult, StaticPlannerBackend
from ai_edit_kernel.runtime.executor import ExecutionContext, Executor
from ai_edit_kernel.runtime.validator import Validator
from ai_edit_kernel.trace.trace_logger import TraceLogger, TraceSession


DATASET_SCHEMA_VERSION = "ai_edit_training_example.v1"
PROMPT_0002 = "Cut a transparent ellipse out of the center of a 200x200 canvas, with height 100 and width 50. Give it a thin purple border."
PROMPT_0003 = "Draw red, green, blue, and yellow hearts, one in each corner of a filled-in shite square."
PROMPT_0004 = "Create a blank white canvas with the text 'work in progress' in the center, surrounded by a purple rectangle. Make the inside of the rectangle / background around the text pink."
PROMPT_0005 = "Take the provided image and change the color and cut out the white background to make it transparent. Make sure you don't cut the girls' hair or clothes out, since they are very white. Be careful when selecting."


@dataclass(slots=True)
class ExampleSpec:
    """Everything needed to run one local example through the planner path."""

    example_id: str
    prompt: str
    task_type: str
    document: DocumentState
    planner_output: dict[str, Any]
    observations: list[dict[str, Any]]
    source_image: Path | None = None
    notes: list[str] | None = None


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    source_image = args.source_image.resolve()

    dataset_root.mkdir(parents=True, exist_ok=True)
    os.chdir(dataset_root)

    examples = build_examples(source_image)
    selected = set(args.examples) if args.examples else {example.example_id for example in examples}
    summaries = []
    for example in examples:
        if example.example_id not in selected:
            continue
        summaries.append(run_example(example, dataset_root))

    print(json.dumps({"success": all(item["succeeded"] for item in summaries), "examples": summaries}, indent=2))
    return 0 if all(item["succeeded"] for item in summaries) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate local deterministic AI editing examples 0002-0005.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/algebraity/Programming/aie-training-data"))
    parser.add_argument(
        "--source-image",
        type=Path,
        default=Path("/home/algebraity/Media/Old Media :3/Pictures/vanilla nihongo.png"),
    )
    parser.add_argument("--examples", nargs="*", default=None, help="Optional subset, e.g. training0002 training0005.")
    return parser.parse_args()


def build_examples(source_image: Path) -> list[ExampleSpec]:
    return [
        example_0002(),
        example_0003(),
        example_0004(),
        example_0005(source_image),
    ]


def example_0002() -> ExampleSpec:
    document = blank_document("doc_training0002_seed", 200, 200, "training0002 empty seed")
    output_dir = Path("data") / "training0002"
    return ExampleSpec(
        example_id="training0002",
        prompt=PROMPT_0002,
        task_type="cutout_shape_with_border",
        document=document,
        observations=[
            {
                "id": "center_ellipse_cutout",
                "type": "ellipse_region",
                "bbox_xyxy": [75, 50, 125, 150],
                "width": 50,
                "height": 100,
                "note": "A 50x100 ellipse centered on the 200x200 canvas.",
            }
        ],
        planner_output={
            "schema_version": "ai_edit_planner_output.v1",
            "description": "Create a white 200x200 canvas, clear a centered transparent ellipse, and draw a thin purple border.",
            "stop_on_error": True,
            "metadata": {"planner_mode": "manual_static_planner", "training_example_id": "training0002"},
            "actions": [
                {
                    "type": "new_document",
                    "params": {
                        "width": 200,
                        "height": 200,
                        "background_color": "#00000000",
                        "title": "Transparent ellipse cutout",
                        "tags": ["shape", "cutout", "ellipse"],
                    },
                    "description": "Start with a transparent 200x200 document.",
                },
                {
                    "type": "create_layer",
                    "target": {"output_layer_id": "layer_white_canvas"},
                    "params": {"name": "white canvas", "kind": "raster", "color": "#ffffffff", "set_active": True},
                    "description": "Add the filled white canvas layer that will receive the cutout.",
                },
                {
                    "type": "create_mask_from_shape",
                    "target": {"mask_id": "mask_center_ellipse"},
                    "params": {
                        "name": "center ellipse cutout",
                        "kind": "write_guard",
                        "shape": {"type": "ellipse", "bbox_xyxy": [75, 50, 125, 150]},
                        "set_active": False,
                    },
                    "description": "Build the exact ellipse write mask.",
                },
                {
                    "type": "clear_region",
                    "target": {"layer_id": "layer_white_canvas"},
                    "write_mask_id": "mask_center_ellipse",
                    "params": {"mode": "alpha_to_zero", "preserve_rgb": False},
                    "description": "Clear the ellipse to transparent.",
                },
                {
                    "type": "create_layer",
                    "target": {"output_layer_id": "layer_purple_border"},
                    "params": {"name": "purple ellipse border", "kind": "raster", "color": "#00000000", "set_active": True},
                    "description": "Keep the border on a separate editable layer.",
                },
                {
                    "type": "draw_shape",
                    "target": {"layer_id": "layer_purple_border"},
                    "params": {
                        "shape": {"type": "ellipse", "bbox_xyxy": [75, 50, 125, 150]},
                        "stroke": {"color": "#800080", "width": 2},
                        "fill": None,
                    },
                    "description": "Draw the thin purple ellipse border.",
                },
                {"type": "validate", "params": {}, "description": "Validate the final document."},
                {
                    "type": "export_flat",
                    "params": {"path": str(output_dir / "final_image.png")},
                    "description": "Export the flattened PNG result.",
                },
                {
                    "type": "export_layered_bundle",
                    "params": {"path": str(output_dir / "layered_bundle"), "include_preview": True, "include_hidden": True, "overwrite": True},
                    "description": "Export layers, masks, preview, and document metadata.",
                },
            ],
        },
    )


def example_0003() -> ExampleSpec:
    document = blank_document("doc_training0003_seed", 200, 200, "training0003 empty seed")
    output_dir = Path("data") / "training0003"
    heart = "\\u2665"
    return ExampleSpec(
        example_id="training0003",
        prompt=PROMPT_0003,
        task_type="colored_symbols_on_square",
        document=document,
        observations=[
            {
                "id": "typo_interpretation",
                "type": "language_normalization",
                "input_word": "shite",
                "interpreted_as": "white",
                "note": "The prompt is interpreted as asking for a filled-in white square.",
            },
            {
                "id": "corner_heart_positions",
                "type": "layout_plan",
                "positions": {"top_left": [40, 40], "top_right": [160, 40], "bottom_left": [40, 160], "bottom_right": [160, 160]},
            },
        ],
        notes=["Interpreted 'shite square' as 'white square'."],
        planner_output={
            "schema_version": "ai_edit_planner_output.v1",
            "description": "Draw a filled white square with one colored heart in each corner.",
            "stop_on_error": True,
            "metadata": {
                "planner_mode": "manual_static_planner",
                "training_example_id": "training0003",
                "prompt_interpretation": "shite square means white square",
            },
            "actions": [
                {
                    "type": "new_document",
                    "params": {
                        "width": 200,
                        "height": 200,
                        "background_color": "#00000000",
                        "title": "Colored hearts in square",
                        "tags": ["text", "hearts", "layout"],
                    },
                    "description": "Start with a transparent 200x200 document.",
                },
                {
                    "type": "create_layer",
                    "target": {"output_layer_id": "layer_white_square"},
                    "params": {"name": "filled white square", "kind": "raster", "color": "#ffffffff", "set_active": True},
                    "description": "Use the whole canvas as the filled white square.",
                },
                text_action("layer_red_heart", "red heart", heart, 40, 40, 38, "#ff0000"),
                text_action("layer_green_heart", "green heart", heart, 160, 40, 38, "#008000"),
                text_action("layer_blue_heart", "blue heart", heart, 40, 160, 38, "#0000ff"),
                text_action("layer_yellow_heart", "yellow heart", heart, 160, 160, 38, "#ffd400"),
                {"type": "validate", "params": {}, "description": "Validate the final document."},
                {
                    "type": "export_flat",
                    "params": {"path": str(output_dir / "final_image.png")},
                    "description": "Export the flattened PNG result.",
                },
                {
                    "type": "export_layered_bundle",
                    "params": {"path": str(output_dir / "layered_bundle"), "include_preview": True, "include_hidden": True, "overwrite": True},
                    "description": "Export layers, masks, preview, and document metadata.",
                },
            ],
        },
    )


def example_0004() -> ExampleSpec:
    document = blank_document("doc_training0004_seed", 200, 200, "training0004 empty seed")
    output_dir = Path("data") / "training0004"
    return ExampleSpec(
        example_id="training0004",
        prompt=PROMPT_0004,
        task_type="text_inside_framed_panel",
        document=document,
        observations=[
            {
                "id": "pink_panel_rect",
                "type": "rectangle_region",
                "bbox_xyxy": [25, 55, 175, 145],
                "note": "The rectangle surrounds the centered text and its interior is filled pink.",
            }
        ],
        planner_output={
            "schema_version": "ai_edit_planner_output.v1",
            "description": "Create a white canvas, draw a pink rectangle with a purple border, and center the requested text.",
            "stop_on_error": True,
            "metadata": {"planner_mode": "manual_static_planner", "training_example_id": "training0004"},
            "actions": [
                {
                    "type": "new_document",
                    "params": {
                        "width": 200,
                        "height": 200,
                        "background_color": "#ffffffff",
                        "title": "Work in progress sign",
                        "tags": ["text", "rectangle", "layout"],
                    },
                    "description": "Start with a blank white 200x200 document.",
                },
                {
                    "type": "create_layer",
                    "target": {"output_layer_id": "layer_pink_panel"},
                    "params": {"name": "pink rectangle panel", "kind": "raster", "color": "#00000000", "set_active": True},
                    "description": "Create the editable panel layer.",
                },
                {
                    "type": "draw_shape",
                    "target": {"layer_id": "layer_pink_panel"},
                    "params": {
                        "shape": {"type": "rectangle", "bbox_xyxy": [25, 55, 175, 145], "corner_radius": 0},
                        "stroke": {"color": "#800080", "width": 3},
                        "fill": {"color": "#ffc0cb"},
                    },
                    "description": "Fill the rectangle pink and outline it in purple.",
                },
                text_action("layer_work_text", "work in progress text", "work in progress", 100, 100, 17, "#111111"),
                {"type": "validate", "params": {}, "description": "Validate the final document."},
                {
                    "type": "export_flat",
                    "params": {"path": str(output_dir / "final_image.png")},
                    "description": "Export the flattened PNG result.",
                },
                {
                    "type": "export_layered_bundle",
                    "params": {"path": str(output_dir / "layered_bundle"), "include_preview": True, "include_hidden": True, "overwrite": True},
                    "description": "Export layers, masks, preview, and document metadata.",
                },
            ],
        },
    )


def example_0005(source_image: Path) -> ExampleSpec:
    if not source_image.exists():
        raise FileNotFoundError(f"source image does not exist: {source_image}")

    output_dir = Path("data") / "training0005"
    copied_source = output_dir / "source_image.png"
    with Image.open(source_image) as image:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    height, width = rgba.shape[:2]
    document = DocumentState(
        id="doc_training0005_source",
        canvas=CanvasSpec(width=width, height=height, background_color_rgba=(0.0, 0.0, 0.0, 0.0)),
        layers=[Layer(id="layer_source", name="provided anime image", pixels=rgba.astype(np.float32))],
        active_layer_id="layer_source",
        metadata=DocumentMetadata(
            title="Provided image background removal",
            source_file=str(copied_source),
            tags=["background_removal", "color_change", "careful_selection"],
        ),
    )
    document.validate()

    return ExampleSpec(
        example_id="training0005",
        prompt=PROMPT_0005,
        task_type="background_removal_and_color_change",
        document=document,
        source_image=source_image,
        notes=[
            "The prompt did not specify which color to change or the target color.",
            "This run interprets the color change as changing the red circular backdrop to purple.",
        ],
        observations=[
            {
                "id": "source_image_geometry",
                "type": "image_metadata",
                "width": width,
                "height": height,
                "source_layer_id": "layer_source",
            },
            {
                "id": "white_background_strategy",
                "type": "mask_plan",
                "method": "magic_wand_from_border_seed_points",
                "tolerance": 0.025,
                "seed_points": [[1, 1], [1227, 1], [1, 600], [1227, 600], [100, 1100], [1100, 1100], [10, 1180], [1210, 1180]],
                "note": "Use a border-connected white selection so similarly white hair and clothing are not selected unless connected to the outer background.",
            },
            {
                "id": "color_change_interpretation",
                "type": "ambiguous_prompt_resolution",
                "changed_region": "red circular backdrop",
                "target_color": "#7a2cff",
                "note": "The prompt asks to change the color but does not name a source or destination color.",
            },
        ],
        planner_output={
            "schema_version": "ai_edit_planner_output.v1",
            "description": "Remove only the border-connected white background and recolor the contiguous red backdrop to purple.",
            "stop_on_error": True,
            "metadata": {
                "planner_mode": "manual_static_planner",
                "training_example_id": "training0005",
                "color_change_interpretation": "red circular backdrop to purple",
            },
            "actions": [
                {
                    "type": "magic_wand_select",
                    "target": {"layer_id": "layer_source", "mask_id": "mask_border_white_background"},
                    "params": {
                        "name": "border connected white background",
                        "seed_points": [[1, 1], [1227, 1], [1, 600], [1227, 600], [100, 1100], [1100, 1100], [10, 1180], [1210, 1180]],
                        "tolerance": 0.025,
                        "alpha_min": 0.95,
                        "diagonal": True,
                        "kind": "write_guard",
                        "set_active": False,
                    },
                    "description": "Select the white background from the image border, avoiding unconnected white hair and clothing.",
                },
                {
                    "type": "create_mask_from_shape",
                    "target": {"mask_id": "mask_bottom_edge_line"},
                    "params": {
                        "name": "bottom edge line",
                        "kind": "write_guard",
                        "shape": {"type": "rectangle", "bbox_xyxy": [0, 1195, 1229, 1198], "corner_radius": 0},
                        "set_active": False,
                    },
                    "description": "Include the thin bottom edge artifact in the removable outside-background area.",
                },
                {
                    "type": "combine_masks",
                    "target": {"mask_id": "mask_background_to_clear"},
                    "params": {
                        "operation": "union",
                        "mask_ids": ["mask_border_white_background", "mask_bottom_edge_line"],
                        "name": "background to clear",
                    },
                    "description": "Combine the selected white background with the bottom edge cleanup mask.",
                },
                {
                    "type": "clear_region",
                    "target": {"layer_id": "layer_source"},
                    "write_mask_id": "mask_background_to_clear",
                    "params": {"mode": "alpha_to_zero", "preserve_rgb": False},
                    "description": "Make the outside background transparent.",
                },
                {
                    "type": "magic_wand_select",
                    "target": {"layer_id": "layer_source", "mask_id": "mask_red_backdrop"},
                    "params": {
                        "name": "red circular backdrop",
                        "seed_points": [[300, 250], [900, 250], [250, 700], [1000, 700], [614, 100]],
                        "tolerance": 0.035,
                        "alpha_min": 0.95,
                        "diagonal": True,
                        "kind": "write_guard",
                        "set_active": False,
                    },
                    "description": "Select the contiguous red circular backdrop without targeting isolated red character details.",
                },
                {
                    "type": "paint_bucket_fill",
                    "target": {"layer_id": "layer_source"},
                    "write_mask_id": "mask_red_backdrop",
                    "params": {"color": "#7a2cff", "mode": "replace_rgb_preserve_alpha"},
                    "description": "Change the selected red backdrop to purple while preserving its alpha.",
                },
                {"type": "validate", "params": {}, "description": "Validate the final document."},
                {
                    "type": "export_flat",
                    "params": {"path": str(output_dir / "final_image.png")},
                    "description": "Export the flattened PNG result.",
                },
                {
                    "type": "export_layered_bundle",
                    "params": {"path": str(output_dir / "layered_bundle"), "include_preview": True, "include_hidden": True, "overwrite": True},
                    "description": "Export layers, masks, preview, and document metadata.",
                },
            ],
        },
    )


def text_action(layer_id: str, name: str, text: str, x: int, y: int, font_size: int, color: str) -> dict[str, Any]:
    return {
        "type": "create_text_layer",
        "target": {"output_layer_id": layer_id},
        "params": {
            "name": name,
            "text": text.encode("utf-8").decode("unicode_escape") if text.startswith("\\u") else text,
            "x": x,
            "y": y,
            "font_size": font_size,
            "color": color,
            "anchor": "mm",
            "align": "center",
            "set_active": False,
        },
        "description": f"Create the {name} layer.",
    }


def blank_document(document_id: str, width: int, height: int, title: str) -> DocumentState:
    document = DocumentState(
        id=document_id,
        canvas=CanvasSpec(width=width, height=height, background_color_rgba=(0.0, 0.0, 0.0, 0.0)),
        metadata=DocumentMetadata(title=title, tags=["empty_seed"]),
    )
    document.validate()
    return document


def run_example(example: ExampleSpec, dataset_root: Path) -> dict[str, Any]:
    example_rel = Path("data") / example.example_id
    example_dir = dataset_root / example_rel
    if example_dir.exists():
        shutil.rmtree(example_dir)
    example_dir.mkdir(parents=True)

    source_rel: Path | None = None
    if example.source_image is not None:
        source_rel = example_rel / "source_image.png"
        shutil.copy2(example.source_image, dataset_root / source_rel)
        save_preview_png(example.document, dataset_root / example_rel / "initial_image.png")

    trace_root = example_rel / "trace"
    logger = TraceLogger(
        trace_root,
        metadata={
            "task_type": example.task_type,
            "source": "manual_static_planner_local_generation",
            "split": "train",
            "planner": {
                "planner_type": "manual_static_planner",
                "planner_model": "human_controller",
                "planner_version": "training_tasks_0002_0005.v1",
                "temperature": 0.0,
            },
            "rights": {
                "source_image_license": "unknown",
                "user_provided_content": example.source_image is not None,
                "allowed_for_training": True,
                "contains_personal_data": False,
            },
            "notes": example.notes or [],
        },
    )

    session = logger.start_session(example.prompt)
    initial_label = "initial_source_document" if example.source_image is not None else "initial_empty_document"
    logger.log_document_snapshot(example.document, initial_label)
    logger.log_observation(example.document, {"observations": example.observations}, "manual_controller_observations")

    asset_refs = {
        "initial_snapshot": f"snapshots/doc_rev_{example.document.revision:04d}_{initial_label}.json",
        "initial_preview": f"previews/doc_rev_{example.document.revision:04d}_{initial_label}.png",
    }
    if source_rel is not None:
        asset_refs["source_image"] = str(source_rel)

    planner = AIPlanner(backend=StaticPlannerBackend([example.planner_output]), trace_sink=logger)
    executor = Executor(ExecutionContext(trace_sink=logger))
    execution = planner.plan_and_execute(
        example.prompt,
        example.document,
        executor,
        observations=example.observations,
        asset_refs=asset_refs,
        metadata={"controller": "manual model surrogate"},
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

    session_dir = dataset_root / trace_root / session.id
    write_json(example_dir / "planner_request.json", execution.planner_result.request)
    write_json(example_dir / "planner_output.json", execution.planner_result.raw_output)
    write_json(example_dir / "action_batch.json", execution.planner_result.action_batch.to_json())
    training_example = build_training_example(example, session, execution, report.has_errors(), trace_root / session.id)
    write_json(example_dir / "training_example.json", training_example)
    shutil.copy2(session_dir / "manifest.json", example_dir / "manifest.json")
    shutil.copy2(session_dir / "events.jsonl", example_dir / "events.jsonl")

    zip_directory(example_dir / "layered_bundle", example_dir / "layered_bundle.zip")
    zip_directory(session_dir, example_dir / "trace_session.zip")

    run_summary = build_run_summary(example, execution, succeeded, report.has_errors(), trace_root / session.id)
    write_json(example_dir / "run_summary.json", run_summary)
    validation = build_export_validation(example, example_dir, run_summary, training_example)
    write_json(example_dir / "export_validation.json", validation)

    return {
        "example_id": example.example_id,
        "succeeded": succeeded,
        "final_image": str(example_rel / "final_image.png"),
        "trace": str(trace_root / session.id),
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
        "example_id": f"ex_{session.id}_plan_001",
        "source_session_id": session.id,
        "example_type": "planner_imitation",
        "task_type": example.task_type,
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
        "provenance": {
            "trace_path": str(trace_path),
            "source": "manual_static_planner_local_generation",
            "allowed_for_training": True,
        },
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
        "source_image_provided": example.source_image is not None,
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
        "notes": example.notes or [],
    }


def build_export_validation(
    example: ExampleSpec,
    example_dir: Path,
    run_summary: dict[str, Any],
    training_example: dict[str, Any],
) -> dict[str, Any]:
    returned = [
        "final_image.png",
        "planner_request.json",
        "planner_output.json",
        "action_batch.json",
        "manifest.json",
        "events.jsonl",
        "training_example.json",
        "run_summary.json",
        "layered_bundle.zip",
        "trace_session.zip",
    ]
    if example.source_image is not None:
        returned.extend(["source_image.png", "initial_image.png"])

    stale_hits = find_text_hits(example_dir, ["/mnt/data"])
    secret_hits = find_text_hits(example_dir, ["Bearer ", "YOUR_API_KEY", "venice-key"])
    checks = {
        "final_image_exists": (example_dir / "final_image.png").exists(),
        "layered_bundle_exists": (example_dir / "layered_bundle" / "manifest.json").exists(),
        "trace_session_exists": any((example_dir / "trace").glob("session_*")),
        "manifest_success_true": manifest_success(example_dir / "manifest.json"),
        "trace_has_initial_preview": trace_asset_exists(example_dir, "previews", "doc_rev_0000_initial"),
        "trace_has_initial_snapshot": trace_asset_exists(example_dir, "snapshots", "doc_rev_0000_initial"),
        "training_example_target_is_planner_output": list(training_example["target"]) == ["planner_output"],
        "training_example_target_has_no_action_batch": "action_batch" not in training_example["target"],
        "training_example_uses_initial_input": training_example["input"]["document_summary"]["revision"] == 0,
        "source_image_present_if_user_provided": example.source_image is None or (example_dir / "source_image.png").exists(),
        "initial_image_present_if_user_provided": example.source_image is None or (example_dir / "initial_image.png").exists(),
        "returned_artifacts_exist": all((example_dir / path).exists() for path in returned),
        "validation_passed_true": bool(run_summary["validation_passed"]),
        "no_mnt_data_paths_remain": not stale_hits,
        "no_api_keys_or_bearer_tokens": not secret_hits,
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


def save_preview_png(document: DocumentState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = document.flatten_preview()
    image = Image.fromarray(np.clip(pixels * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGBA")
    image.save(path)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def zip_directory(source_dir: Path, output_path: Path) -> None:
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source_dir))


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


if __name__ == "__main__":
    raise SystemExit(main())
