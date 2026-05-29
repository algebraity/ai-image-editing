#!/usr/bin/env python3
"""Generate training examples 0031-0100 from the portrait candidate set."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_edit_kernel.diffusion import VeniceImageBackend
from ai_edit_kernel.planning.planner import AIPlanner, PlannerExecutionResult, StaticPlannerBackend
from ai_edit_kernel.runtime.executor import ExecutionContext, Executor
from ai_edit_kernel.runtime.validator import Validator
from ai_edit_kernel.trace.trace_logger import TraceLogger, TraceSession

from scripts import generate_training_tasks_0021_0030 as base


VENICE_KEY_PATH = REPO_ROOT / "keys" / "venice-key"


@dataclass(slots=True)
class ExampleSpec(base.ExampleSpec):
    """One generated example plus category metadata."""

    category: str = "image_editing"
    diffusion_used: bool = False


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)

    factories = build_factories()
    selected = normalize_examples(args.examples) if args.examples else list(factories)
    summaries = []
    for example_id in selected:
        if example_id not in factories:
            raise ValueError(f"unknown example id {example_id!r}")
        summaries.append(run_example(factories[example_id](dataset_root, args), dataset_root))

    success = all(item["succeeded"] and item["validation_passed"] for item in summaries)
    print(json.dumps({"success": success, "examples": summaries}, indent=2))
    return 0 if success else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AI editing examples 0031-0100.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/algebraity/Programming/aie-training-data"))
    parser.add_argument("--images-dir", type=Path, default=Path("/home/algebraity/Programming/aie-training-data/images/portraits"))
    parser.add_argument("--examples", nargs="*", default=None, help="Optional subset, e.g. training0031 42 0100.")
    return parser.parse_args()


def normalize_examples(values: list[str]) -> list[str]:
    examples = []
    for value in values:
        if re.fullmatch(r"training\d{4}", value):
            examples.append(value)
        elif re.fullmatch(r"\d{1,4}", value):
            examples.append(f"training{int(value):04d}")
        else:
            raise ValueError(f"invalid example id {value!r}")
    return examples


def build_factories() -> dict[str, Callable[[Path, argparse.Namespace], ExampleSpec]]:
    factories: dict[str, Callable[[Path, argparse.Namespace], ExampleSpec]] = {}
    groups = [
        (31, BACKGROUND_REMOVAL, build_background_removal_example),
        (41, BACKGROUND_GENERATION, build_background_generation_example),
        (51, EYE_RECOLOR, build_eye_recolor_example),
        (61, HAIR_RECOLOR, build_hair_recolor_example),
        (71, SMALL_PATTERNS, build_small_pattern_example),
        (81, CLEAN_REGION_GENERATION, build_clean_region_generation_example),
        (91, STICKERS, build_sticker_example),
    ]
    for start, specs, builder in groups:
        for offset, spec in enumerate(specs):
            example_id = f"training{start + offset:04d}"
            factories[example_id] = make_factory(example_id, spec, builder)
    return factories


def make_factory(
    example_id: str,
    spec: dict[str, Any],
    builder: Callable[[str, dict[str, Any], argparse.Namespace], ExampleSpec],
) -> Callable[[Path, argparse.Namespace], ExampleSpec]:
    def factory(dataset_root: Path, args: argparse.Namespace) -> ExampleSpec:
        return builder(example_id, spec, args)

    return factory


def build_background_removal_example(example_id: str, spec: dict[str, Any], args: argparse.Namespace) -> ExampleSpec:
    source = args.images_dir / spec["file"]
    document, width, height = base.source_document(f"doc_{example_id}_initial", source, "Initial source document")
    seeds = spec.get("seed_points", corner_seed_points(width, height))
    actions = [
        magic_wand_action(
            "mask_background",
            "contiguous plain background",
            seeds,
            spec.get("tolerance", 0.24),
            [0, 0, width, height],
            "Select the contiguous plain background from the image edges.",
            kind="write_guard",
            color_space="rgb",
            diagonal=True,
            edge_stop_threshold=0.06,
        ),
        cleanup_fringe_action(
            "mask_background_clean",
            "background plus antialiased edge fringe",
            "mask_background",
            seeds,
            [0, 0, width, height],
            "Extend the background mask into nearby old-background antialias pixels before clearing.",
            search_radius=1,
        ),
        clear_region_action("source_image", "mask_background_clean", "Clear the selected background to transparent pixels."),
    ]
    return make_example(
        example_id,
        spec["prompt"],
        source,
        document,
        actions,
        [
            base.source_observation(width, height),
            base.region_observation("The background is selected from multiple edge seeds and cleared to transparent.", [0, 0, width, height]),
        ],
        "background_removal_transparency",
        "Remove the edge-connected background and preserve the character.",
        diffusion_used=False,
    )


def build_background_generation_example(example_id: str, spec: dict[str, Any], args: argparse.Namespace) -> ExampleSpec:
    source = args.images_dir / spec["file"]
    document, width, height = base.source_document(f"doc_{example_id}_initial", source, "Initial source document")
    seeds = spec.get("seed_points", corner_seed_points(width, height))
    generation_prompt = spec["generation_prompt"]
    actions = [
        txt2img_to_layer_action(
            "layer_generated_background",
            "Generated Background",
            generation_prompt,
            "Generate a full-canvas replacement background layer with Venice diffusion.",
            seed=spec["seed"],
        ),
        reorder_layer_action("layer_generated_background", 0, "Move the generated background behind the source character layer."),
    ]
    if spec.get("selection_mode") == "gimp_fuzzy":
        actions.extend(
            [
                fuzzy_select_action(
                    "mask_background_raw",
                    "plain lavender dotted background",
                    spec["selection_clicks"],
                    spec.get("threshold", 15),
                    [0, 0, width, height],
                    "Fuzzy-select the lavender background with GIMP-style clicks and threshold 15, keeping the white hat and character unselected.",
                    kind="write_guard",
                    antialias=False,
                ),
                refine_selection_action(
                    "mask_background",
                    "background plus tiny white dot holes",
                    "mask_background_raw",
                    "Fill the small unselected white dot holes inside the selected lavender background before clearing it.",
                    threshold=0.5,
                    fill_holes=True,
                    max_hole_area=spec.get("fill_holes_max_area", 100),
                ),
                clear_region_action("source_image", "mask_background", "Clear the original lavender dotted background so the generated layer is visible."),
                *background_white_cleanup_actions(spec.get("white_cleanup_regions", [])),
                *remove_small_alpha_islands_actions(spec.get("remove_alpha_islands", {})),
            ]
        )
    else:
        actions.extend(
            [
                duplicate_layer_action("source_image", "layer_selection_guide", "Temporary Selection Guide", "Duplicate the source image to create a temporary selection guide layer."),
                *guard_stroke_actions(spec.get("guard_strokes", [])),
                magic_wand_action(
                    "mask_background",
                    "plain background behind character",
                    seeds,
                    spec.get("tolerance", 0.24),
                    [0, 0, width, height],
                    "Select the simple original background on the temporary guide layer.",
                    layer_id="layer_selection_guide",
                    kind="write_guard",
                    color_space="rgb",
                    diagonal=True,
                    edge_stop_threshold=0.06,
                ),
                clear_region_action("source_image", "mask_background", "Clear the old background on the source layer so the generated layer is visible."),
                *safe_background_clear_actions(spec.get("safe_clear_zones", [])),
                *background_cleanup_actions(spec.get("cleanup_regions", [])),
                set_layer_visibility_action("layer_selection_guide", False, "Hide the temporary selection guide layer so guard strokes never appear in the final flattened image."),
            ]
        )
    return make_example(
        example_id,
        spec["prompt"],
        source,
        document,
        actions,
        [
            base.source_observation(width, height),
            base.region_observation("The edge-connected background is opened so a generated layer behind the subject can show through.", [0, 0, width, height]),
        ],
        "background_generation_with_diffusion",
        "Generate a new background layer and clear the old simple background from the source layer.",
        diffusion_used=True,
    )


def build_eye_recolor_example(example_id: str, spec: dict[str, Any], args: argparse.Namespace) -> ExampleSpec:
    source = args.images_dir / spec["file"]
    document, width, height = base.source_document(f"doc_{example_id}_initial", source, "Initial source document")
    if len(spec["eye_ellipses"]) == 1:
        actions = [
            base.shape_mask_action(
                "mask_irises",
                "Visible iris shape",
                {"type": "ellipse", "bbox_xyxy": spec["eye_ellipses"][0]},
                "Create a tight elliptical mask around the visible iris.",
            )
        ]
    else:
        shape_ids = [f"mask_eye_shape_{index:02d}" for index, _bbox in enumerate(spec["eye_ellipses"], start=1)]
        actions = [
            *[
            base.shape_mask_action(
                shape_id,
                f"Iris shape guard {index}",
                {"type": "ellipse", "bbox_xyxy": ellipse},
                "Create a tight elliptical guard around one visible iris so nearby hair, skin, or accessories cannot be recolored.",
            )
            for index, (shape_id, ellipse) in enumerate(zip(shape_ids, spec["eye_ellipses"]), start=1)
            ],
            base.combine_action("mask_irises", shape_ids, "Combined iris masks", "union", "Combine the visible iris shape masks."),
        ]
    actions.extend([
        base.colorize_action(
            "mask_irises",
            spec["color"],
            "Change the iris hue while preserving the original highlights, pupils, and eye shading.",
            method="set_hue_preserve_lightness",
        ),
    ])
    return make_example(
        example_id,
        spec["prompt"],
        source,
        document,
        actions,
        [
            base.source_observation(width, height),
            base.region_observation("The visible irises are isolated with tight explicit ellipse masks before hue-preserving colorization.", spec["bbox"]),
        ],
        "eye_color_change",
        "Recolor the visible irises while preserving eye shading and line art.",
        diffusion_used=False,
    )


def build_hair_recolor_example(example_id: str, spec: dict[str, Any], args: argparse.Namespace) -> ExampleSpec:
    source = args.images_dir / spec["file"]
    document, width, height = base.source_document(f"doc_{example_id}_initial", source, "Initial source document")
    actions = [
        seeded_magic_wand_action(
            "mask_hair_or_object",
            "simple hair or object color",
            spec["seeds"],
            spec["bbox"],
            "Select the target hair or object material as contiguous seeded regions inside a constrained box.",
            hue_tolerance_degrees=spec.get("hue", 38),
            saturation_tolerance=spec.get("sat", 0.55),
            value_tolerance=spec.get("val", 0.72),
            edge_stop_threshold=spec.get("edge_stop", 0.14),
        ),
        cleanup_fringe_action(
            "mask_hair_or_object_clean",
            "target material plus edge fringe",
            "mask_hair_or_object",
            spec["seeds"],
            spec["bbox"],
            "Include nearby old-color edge pixels so the recolor does not leave a visible fringe.",
            search_radius=2,
        ),
        base.colorize_action(
            "mask_hair_or_object_clean",
            spec["color"],
            "Recolor the selected material while preserving line art, highlights, and shadows.",
            method=spec.get("method", "set_hue_preserve_lightness"),
        ),
    ]
    return make_example(
        example_id,
        spec["prompt"],
        source,
        document,
        actions,
        [
            base.source_observation(width, height),
            base.region_observation("The target material is selected by seeded color range and local fringe cleanup.", spec["bbox"]),
        ],
        "hair_or_object_recolor",
        "Recolor one simple hair or object material while preserving existing value structure.",
        diffusion_used=False,
    )


def build_small_pattern_example(example_id: str, spec: dict[str, Any], args: argparse.Namespace) -> ExampleSpec:
    source = args.images_dir / spec["file"]
    document, width, height = base.source_document(f"doc_{example_id}_initial", source, "Initial source document")
    actions = [base.create_shape_layer_action("layer_pattern_marks", "small pattern marks", "Create a transparent layer for the small decorative pattern.")]
    actions.extend(pattern_actions(spec["marks"]))
    return make_example(
        example_id,
        spec["prompt"],
        source,
        document,
        actions,
        [
            base.source_observation(width, height),
            base.layout_observation("Small decorative marks are placed on a localized accessory or clean visual area.", spec["observation_points"], spec["observation_sizes"]),
        ],
        "small_pattern_change",
        "Add a small localized pattern without disturbing the underlying drawing.",
        diffusion_used=False,
    )


def build_clean_region_generation_example(example_id: str, spec: dict[str, Any], args: argparse.Namespace) -> ExampleSpec:
    source = args.images_dir / spec["file"]
    document, width, height = base.source_document(f"doc_{example_id}_initial", source, "Initial source document")
    bbox = spec["bbox"]
    actions = [
        base.shape_mask_action(
            "mask_generated_card",
            "clean rectangular generation area",
            {"type": "rectangle", "bbox_xyxy": bbox, "corner_radius": 0},
            "Create a rectangular write mask in the clean region requested by the prompt.",
        ),
        inpaint_region_action(
            "source_image",
            "layer_generated_card_unused",
            "mask_generated_card",
            spec["generation_prompt"],
            "Use Venice diffusion to generate image content inside the clean rectangular region.",
            seed=spec["seed"],
            mode="replace_region",
            padding=0,
        ),
        base.create_shape_layer_action("layer_generated_card_frame", "generated card frame", "Create a transparent layer for a clean frame around the generated insert."),
        base.draw_shape_action(
            "layer_generated_card_frame",
            {"type": "rectangle", "bbox_xyxy": bbox, "corner_radius": 0},
            {"color": "#ffffff", "width": 5},
            None,
            "Draw a crisp white frame around the generated clean-region image.",
        ),
        base.draw_shape_action(
            "layer_generated_card_frame",
            {"type": "rectangle", "bbox_xyxy": bbox, "corner_radius": 0},
            {"color": "#2f2f2f", "width": 1},
            None,
            "Add a thin dark outline to define the generated insert.",
        ),
    ]
    return make_example(
        example_id,
        spec["prompt"],
        source,
        document,
        actions,
        [
            base.source_observation(width, height),
            base.region_observation("The generated image is restricted to a clean rectangular region in the existing artwork.", bbox),
        ],
        "generated_image_in_clean_region",
        "Generate a small picture-like insert in a clean region and frame it.",
        diffusion_used=True,
    )


def build_sticker_example(example_id: str, spec: dict[str, Any], args: argparse.Namespace) -> ExampleSpec:
    source = args.images_dir / spec["file"]
    document, width, height = base.source_document(f"doc_{example_id}_initial", source, "Initial source document")
    actions = []
    for index, sticker in enumerate(spec["stickers"], start=1):
        layer_id = f"layer_sticker_{index:02d}"
        actions.append(
            base.create_text_action(
                layer_id,
                sticker["name"],
                sticker["text"],
                sticker["pos"],
                sticker["size"],
                sticker["color"],
                f"Create {sticker['name']} at the requested sticker position.",
            )
        )
        if sticker.get("angle"):
            actions.append(base.rotate_action(layer_id, sticker["angle"], sticker["pos"], f"Rotate {sticker['name']} for a natural placed-sticker look."))
    return make_example(
        example_id,
        spec["prompt"],
        source,
        document,
        actions,
        [
            base.source_observation(width, height),
            base.layout_observation("Small sticker or shape overlays are placed in attractive empty areas without covering important facial features.", [s["pos"] for s in spec["stickers"]], [s["size"] for s in spec["stickers"]]),
        ],
        "stickers_shapes",
        "Add simple decorative stickers in visually sensible empty areas.",
        diffusion_used=False,
    )


def make_example(
    example_id: str,
    prompt: str,
    source: Path,
    document: Any,
    edit_actions: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    category: str,
    description: str,
    *,
    diffusion_used: bool,
) -> ExampleSpec:
    planner_output = {
        "schema_version": "ai_edit_planner_output.v1",
        "description": description,
        "stop_on_error": True,
        "metadata": {"example_id": example_id, "category": category, "external_diffusion_used": diffusion_used},
        "actions": [
            *edit_actions,
            base.validate_action("Validate that the edited image has no structural issues."),
            base.export_flat_action(Path("data") / example_id, "Export the flattened edited result."),
            base.export_bundle_action(Path("data") / example_id, "Export layers, masks, preview, and document metadata for inspection."),
        ],
    }
    return ExampleSpec(
        example_id=example_id,
        prompt=prompt,
        source_image=source,
        document=document,
        planner_output=planner_output,
        observations=observations,
        notes=[description],
        category=category,
        diffusion_used=diffusion_used,
    )


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
                "artistic_constraints": ["preserve existing image content unless the prompt asks for an overlay or generation"],
                "diffusion_backend": "venice" if example.diffusion_used else None,
                "example_number": example.example_id[-4:],
                "example_category": example.category,
                "external_diffusion_allowed": example.diffusion_used,
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
            request_builder=base.TrainingRequestBuilder(),
            trace_sink=logger,
        )
        diffusion_backend = VeniceImageBackend(api_key_path=VENICE_KEY_PATH) if example.diffusion_used else None
        executor = Executor(ExecutionContext(trace_sink=logger, diffusion_backend=diffusion_backend))
        execution = planner.plan_and_execute(
            example.prompt,
            example.document,
            executor,
            observations=example.observations,
            asset_refs=asset_refs,
            metadata={"example_number": example.example_id[-4:], "source_image_path": "initial_image.png", "category": example.category},
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
        base.write_json(Path("planner_request.json"), execution.planner_result.request)
        base.write_json(Path("planner_output.json"), execution.planner_result.raw_output)
        base.write_json(Path("action_batch.json"), execution.planner_result.action_batch.to_json())
        training_example = base.build_training_example(example, session, execution, report.has_errors(), session_dir)
        base.write_json(Path("training_example.json"), training_example)
        shutil.copy2(session_dir / "manifest.json", "manifest.json")
        shutil.copy2(session_dir / "events.jsonl", "events.jsonl")
        base.zip_directory_with_root(Path("layered_bundle"), Path("layered_bundle.zip"))
        base.zip_directory_with_root(session_dir, Path("trace_session.zip"))
        run_summary = build_run_summary(example, execution, succeeded, report.has_errors(), session_dir)
        base.write_json(Path("run_summary.json"), run_summary)
        validation = base.build_export_validation(example, example_dir, run_summary, training_example)
        base.write_json(Path("export_validation.json"), validation)
        validation = base.build_export_validation(example, example_dir, run_summary, training_example)
        base.write_json(Path("export_validation.json"), validation)
    finally:
        os.chdir(old_cwd)

    return {
        "example_id": example.example_id,
        "succeeded": succeeded,
        "final_image": str(example_rel / "final_image.png"),
        "trace": str(run_summary["trace_session_path"]),
        "validation_passed": validation["passed"],
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
        "diffusion_used": example.diffusion_used,
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


def edge_seed_points(width: int, height: int) -> list[list[int]]:
    margin = max(2, min(width, height) // 32)
    mid_x = width // 2
    mid_y = height // 2
    return [
        [margin, margin],
        [mid_x, margin],
        [width - margin - 1, margin],
        [margin, mid_y],
        [width - margin - 1, mid_y],
        [margin, height - margin - 1],
        [mid_x, height - margin - 1],
        [width - margin - 1, height - margin - 1],
    ]


def corner_seed_points(width: int, height: int) -> list[list[int]]:
    margin = max(2, min(width, height) // 32)
    return [
        [margin, margin],
        [width - margin - 1, margin],
        [margin, height - margin - 1],
        [width - margin - 1, height - margin - 1],
    ]


def magic_wand_action(
    mask_id: str,
    name: str,
    seed_points: list[list[int]],
    tolerance: float,
    bbox_xyxy: list[int],
    description: str,
    *,
    layer_id: str = "source_image",
    kind: str = "selection",
    color_space: str = "rgb",
    diagonal: bool = True,
    edge_stop_threshold: float | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "name": name,
        "seed_points": seed_points,
        "bbox_xyxy": bbox_xyxy,
        "color_space": color_space,
        "tolerance": tolerance,
        "alpha_min": 0.5,
        "diagonal": diagonal,
        "kind": kind,
        "set_active": False,
    }
    if edge_stop_threshold is not None:
        params["edge_stop_threshold"] = edge_stop_threshold
    return {
        "type": "magic_wand_select",
        "target": {"layer_id": layer_id, "mask_id": mask_id},
        "params": params,
        "description": description,
    }


def fuzzy_select_action(
    mask_id: str,
    name: str,
    clicks: list[dict[str, Any]],
    threshold: float,
    bbox_xyxy: list[int],
    description: str,
    *,
    layer_id: str = "source_image",
    kind: str = "selection",
    antialias: bool = False,
) -> dict[str, Any]:
    return {
        "type": "fuzzy_select",
        "target": {"layer_id": layer_id, "mask_id": mask_id},
        "params": {
            "name": name,
            "clicks": clicks,
            "threshold": threshold,
            "bbox_xyxy": bbox_xyxy,
            "alpha_min": 0.5,
            "diagonal": True,
            "criterion": "composite",
            "antialias": antialias,
            "kind": kind,
            "set_active": False,
        },
        "description": description,
    }


def refine_selection_action(
    mask_id: str,
    name: str,
    source_mask_id: str,
    description: str,
    *,
    threshold: float | None = None,
    min_area: int | None = None,
    fill_holes: bool = False,
    max_hole_area: int | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"source_mask_id": source_mask_id, "name": name, "set_active": False}
    if threshold is not None:
        params["threshold"] = threshold
    if min_area is not None:
        params["min_area"] = min_area
    if fill_holes:
        params["fill_holes"] = True
    if max_hole_area is not None:
        params["max_hole_area"] = max_hole_area
    return {
        "type": "refine_selection",
        "target": {"mask_id": mask_id},
        "params": params,
        "description": description,
    }


def select_by_color_action(
    mask_id: str,
    name: str,
    color: str,
    threshold: float,
    bbox_xyxy: list[int],
    description: str,
    *,
    layer_id: str = "source_image",
) -> dict[str, Any]:
    return {
        "type": "select_by_color",
        "target": {"layer_id": layer_id, "mask_id": mask_id},
        "params": {
            "name": name,
            "color": color,
            "threshold": threshold,
            "bbox_xyxy": bbox_xyxy,
            "alpha_min": 0.9,
            "criterion": "composite",
            "antialias": False,
            "kind": "selection",
            "set_active": False,
        },
        "description": description,
    }


def duplicate_layer_action(source_layer_id: str, output_layer_id: str, name: str, description: str) -> dict[str, Any]:
    return {
        "type": "duplicate_layer",
        "target": {"layer_id": source_layer_id, "output_layer_id": output_layer_id},
        "params": {"name": name, "set_active": False},
        "description": description,
    }


def delete_layer_action(layer_id: str, description: str) -> dict[str, Any]:
    return {
        "type": "delete_layer",
        "target": {"layer_id": layer_id},
        "params": {},
        "description": description,
    }


def set_layer_visibility_action(layer_id: str, visible: bool, description: str) -> dict[str, Any]:
    return {
        "type": "set_layer_visibility",
        "target": {"layer_id": layer_id},
        "params": {"visible": visible},
        "description": description,
    }


def guard_stroke_actions(strokes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        base.brush_stroke_action(
            "layer_selection_guide",
            stroke["points"],
            stroke.get("color", "#000000"),
            stroke.get("width", 5),
            stroke.get("description", "Draw a temporary dark guard stroke on the selection guide."),
            mode=stroke.get("mode", "replace_rgba"),
        )
        for stroke in strokes
    ]


def background_cleanup_actions(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for index, region in enumerate(regions, start=1):
        zone_ids = []
        for zone_index, bbox in enumerate(region["zones"], start=1):
            zone_id = f"mask_cleanup_{index:02d}_zone_{zone_index:02d}"
            zone_ids.append(zone_id)
            actions.append(
                base.shape_mask_action(
                    zone_id,
                    f"Background cleanup zone {index}.{zone_index}",
                    {"type": "rectangle", "bbox_xyxy": bbox, "corner_radius": 0},
                    region.get("zone_description", "Create a conservative safe cleanup zone in background-only space."),
                )
            )
        if len(zone_ids) == 1:
            combined_zone_id = zone_ids[0]
        else:
            combined_zone_id = f"mask_cleanup_{index:02d}_zones"
            actions.append(
                base.combine_action(
                    combined_zone_id,
                    zone_ids,
                    f"Combined cleanup zones {index}",
                    "union",
                    "Combine the conservative background-only cleanup zones.",
                )
            )
        color_mask_id = f"mask_cleanup_{index:02d}_color"
        actions.append(
            base.color_range_action(
                color_mask_id,
                region["name"],
                region["color"],
                region["tolerance"],
                region.get("bbox", [0, 0, 512, 512]),
                region["description"],
            )
        )
        final_mask_id = f"mask_cleanup_{index:02d}"
        actions.append(
            base.combine_action(
                final_mask_id,
                [color_mask_id, combined_zone_id],
                region["name"],
                "intersect",
                "Keep only old-background debris colors inside the conservative background cleanup zones.",
            )
        )
        actions.append(clear_region_action("source_image", final_mask_id, region["clear_description"]))
    return actions


def background_white_cleanup_actions(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for index, region in enumerate(regions, start=1):
        zone_ids = []
        for zone_index, bbox in enumerate(region["zones"], start=1):
            zone_id = f"mask_white_cleanup_{index:02d}_zone_{zone_index:02d}"
            zone_ids.append(zone_id)
            actions.append(
                base.shape_mask_action(
                    zone_id,
                    f"White cleanup zone {index}.{zone_index}",
                    {"type": "rectangle", "bbox_xyxy": bbox, "corner_radius": 0},
                    region.get("zone_description", "Create a conservative zone that contains only leftover old-background white flecks."),
                )
            )
        if len(zone_ids) == 1:
            combined_zone_id = zone_ids[0]
        else:
            combined_zone_id = f"mask_white_cleanup_{index:02d}_zones"
            actions.append(
                base.combine_action(
                    combined_zone_id,
                    zone_ids,
                    f"Combined white cleanup zones {index}",
                    "union",
                    "Combine the safe zones for leftover white old-background flecks.",
                )
            )
        color_mask_id = f"mask_white_cleanup_{index:02d}_color"
        actions.append(
            select_by_color_action(
                color_mask_id,
                region["name"],
                region.get("color", "#FFFFFF"),
                region.get("threshold", 20),
                region.get("bbox", [0, 0, 512, 512]),
                region["description"],
            )
        )
        final_mask_id = f"mask_white_cleanup_{index:02d}"
        actions.append(
            base.combine_action(
                final_mask_id,
                [color_mask_id, combined_zone_id],
                region["name"],
                "intersect",
                "Keep only near-white leftovers inside the conservative right-side background cleanup zones.",
            )
        )
        actions.append(clear_region_action("source_image", final_mask_id, region["clear_description"]))
    return actions


def remove_small_alpha_islands_actions(config: dict[str, Any]) -> list[dict[str, Any]]:
    if not config:
        return []
    min_area = int(config.get("min_area", 200))
    return [
        {
            "type": "select_from_alpha",
            "target": {"layer_id": "source_image", "mask_id": "mask_source_remaining_alpha"},
            "params": {
                "name": "remaining source alpha",
                "threshold": 0.01,
                "kind": "selection",
                "set_active": False,
            },
            "description": "Select all still-opaque source pixels after the background has been cleared.",
        },
        refine_selection_action(
            "mask_source_large_alpha_components",
            "large preserved source alpha components",
            "mask_source_remaining_alpha",
            f"Keep only connected alpha components at least {min_area} pixels large so tiny floating old-background islands can be isolated.",
            min_area=min_area,
        ),
        base.combine_action(
            "mask_small_alpha_islands",
            ["mask_source_remaining_alpha", "mask_source_large_alpha_components"],
            "tiny floating old-background islands",
            "subtract",
            "Subtract the large preserved character components from all remaining alpha to isolate tiny leftover background islands.",
        ),
        clear_region_action("source_image", "mask_small_alpha_islands", "Clear the tiny floating old-background islands left after color-based cleanup."),
    ]


def safe_background_clear_actions(zones: list[list[int]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for index, bbox in enumerate(zones, start=1):
        mask_id = f"mask_safe_background_clear_{index:02d}"
        actions.append(
            base.shape_mask_action(
                mask_id,
                f"Safe background clear zone {index}",
                {"type": "rectangle", "bbox_xyxy": bbox, "corner_radius": 0},
                "Create a conservative geometry-only clear zone that is known to contain background, not the character.",
            )
        )
        actions.append(clear_region_action("source_image", mask_id, "Clear residual old background pixels from a known-safe background-only zone."))
    return actions


def seeded_color_range_action(
    mask_id: str,
    name: str,
    seed_points: list[list[int]],
    bbox_xyxy: list[int],
    description: str,
    *,
    hue_tolerance_degrees: float,
    saturation_tolerance: float,
    value_tolerance: float,
) -> dict[str, Any]:
    return {
        "type": "select_color_range",
        "target": {"layer_id": "source_image", "mask_id": mask_id},
        "params": {
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
        },
        "description": description,
    }


def seeded_magic_wand_action(
    mask_id: str,
    name: str,
    seed_points: list[list[int]],
    bbox_xyxy: list[int],
    description: str,
    *,
    hue_tolerance_degrees: float,
    saturation_tolerance: float,
    value_tolerance: float,
    edge_stop_threshold: float,
) -> dict[str, Any]:
    return {
        "type": "magic_wand_select",
        "target": {"layer_id": "source_image", "mask_id": mask_id},
        "params": {
            "name": name,
            "seed_points": seed_points,
            "bbox_xyxy": bbox_xyxy,
            "color_space": "hsv",
            "hue_tolerance_degrees": hue_tolerance_degrees,
            "saturation_tolerance": saturation_tolerance,
            "value_tolerance": value_tolerance,
            "edge_stop_threshold": edge_stop_threshold,
            "alpha_min": 0.5,
            "diagonal": True,
            "kind": "selection",
            "set_active": False,
        },
        "description": description,
    }


def cleanup_fringe_action(
    mask_id: str,
    name: str,
    source_mask_id: str,
    seed_points: list[list[int]],
    bbox_xyxy: list[int],
    description: str,
    *,
    search_radius: int = 2,
) -> dict[str, Any]:
    return {
        "type": "cleanup_fringe",
        "target": {"layer_id": "source_image", "mask_id": mask_id},
        "params": {
            "name": name,
            "source_mask_id": source_mask_id,
            "seed_points": seed_points,
            "bbox_xyxy": bbox_xyxy,
            "search_radius": search_radius,
            "include_source_mask": True,
            "color_space": "hsv",
            "hue_tolerance_degrees": 36,
            "saturation_tolerance": 0.70,
            "value_tolerance": 0.85,
            "alpha_min": 0.5,
            "set_active": False,
        },
        "description": description,
    }


def clear_region_action(layer_id: str, mask_id: str, description: str) -> dict[str, Any]:
    return {
        "type": "clear_region",
        "target": {"layer_id": layer_id},
        "write_mask_id": mask_id,
        "params": {"mode": "alpha_to_zero", "preserve_rgb": False},
        "description": description,
    }


def txt2img_to_layer_action(layer_id: str, name: str, prompt: str, description: str, *, seed: int) -> dict[str, Any]:
    return {
        "type": "txt2img_to_layer",
        "target": {"output_layer_id": layer_id},
        "params": {
            "output_layer_name": name,
            "prompt": prompt,
            "negative_prompt": "text, watermark, logo, extra character, face, cropped subject, harsh contrast",
            "seed": seed,
            "job": {"model": "krea-v2-medium", "size": "1024x1024"},
        },
        "description": description,
    }


def inpaint_region_action(
    layer_id: str,
    output_layer_id: str,
    mask_id: str,
    prompt: str,
    description: str,
    *,
    seed: int,
    mode: str,
    padding: int,
) -> dict[str, Any]:
    return {
        "type": "inpaint_region",
        "target": {"layer_id": layer_id, "output_layer_id": output_layer_id},
        "write_mask_id": mask_id,
        "params": {
            "prompt": prompt,
            "negative_prompt": "text, watermark, logo, face, character, ugly border, cropped object",
            "seed": seed,
            "mode": mode,
            "padding": padding,
            "job": {"model": "krea-v2-medium", "size": "1024x1024"},
        },
        "description": description,
    }


def reorder_layer_action(layer_id: str, index: int, description: str) -> dict[str, Any]:
    return {
        "type": "reorder_layer",
        "target": {"layer_id": layer_id},
        "params": {"index": index},
        "description": description,
    }


def pattern_actions(marks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actions = []
    for mark in marks:
        if mark["type"] == "circle":
            x, y, r = mark["x"], mark["y"], mark["r"]
            actions.append(
                base.draw_shape_action(
                    "layer_pattern_marks",
                    {"type": "ellipse", "bbox_xyxy": [x - r, y - r, x + r, y + r]},
                    mark.get("stroke"),
                    {"color": mark["color"]},
                    mark["description"],
                )
            )
        elif mark["type"] == "rect":
            actions.append(
                base.draw_shape_action(
                    "layer_pattern_marks",
                    {"type": "rectangle", "bbox_xyxy": mark["bbox"], "corner_radius": mark.get("corner_radius", 0)},
                    mark.get("stroke"),
                    {"color": mark["color"]},
                    mark["description"],
                )
            )
        elif mark["type"] == "line":
            actions.append(base.brush_stroke_action("layer_pattern_marks", mark["points"], mark["color"], mark["width"], mark["description"]))
        else:
            raise ValueError(f"unknown mark type {mark['type']!r}")
    return actions


BACKGROUND_REMOVAL = [
    {"file": "11051710.jpg", "prompt": "remove the clean white background and keep the character transparent", "tolerance": 0.22},
    {"file": "10086430.jpg", "prompt": "remove the clean white background and keep the character transparent", "tolerance": 0.20},
    {"file": "10873771.jpg", "prompt": "remove the simple peach background and keep the character transparent", "tolerance": 0.23},
    {"file": "10186520.jpg", "prompt": "remove the pale green background and keep the character transparent", "tolerance": 0.22},
    {"file": "10716280.jpg", "prompt": "remove the clean white background and keep the character transparent", "tolerance": 0.18},
    {"file": "10935840.jpg", "prompt": "remove the pale pink background and keep the character transparent", "tolerance": 0.25},
    {"file": "10256980.jpg", "prompt": "remove the flat cyan background and keep the character transparent", "tolerance": 0.24},
    {"file": "10990480.jpg", "prompt": "remove the pale green background and keep the character transparent", "tolerance": 0.22},
    {"file": "10335770.jpg", "prompt": "remove the plain light background and keep the character transparent", "tolerance": 0.20},
    {"file": "11012870.jpg", "prompt": "remove the plain pink background and keep the character transparent", "tolerance": 0.24},
]


BACKGROUND_GENERATION = [
    {
        "file": "10887950.jpg",
        "prompt": "replace the simple dotted background with a generated soft pastel sky",
        "generation_prompt": "soft pastel blue and lavender sky background, gentle clouds, anime illustration background, no character, no text",
        "seed": 4101,
        "selection_mode": "gimp_fuzzy",
        "threshold": 15,
        "fill_holes_max_area": 100,
        "selection_clicks": [
            {"point": [492, 36], "operation": "replace"},
            {"point": [476, 476], "operation": "add"},
            {"point": [438, 330], "operation": "add"},
            {"point": [40, 20], "operation": "add"},
            {"point": [20, 20], "operation": "add"},
            {"point": [492, 220], "operation": "add"},
        ],
        "white_cleanup_regions": [
            {
                "name": "right-side white old-background leftovers",
                "color": "#FFFFFF",
                "threshold": 20,
                "zones": [[438, 0, 512, 512], [398, 260, 512, 512]],
                "description": "Select the near-white old-background flecks and halo fragments that remain on the right side after the lavender background is cleared.",
                "clear_description": "Clear the right-side white old-background leftovers so no pale speckles float over the generated sky.",
            }
        ],
        "remove_alpha_islands": {"min_area": 200},
    },
    {"file": "10161020.jpg", "prompt": "replace the simple peach background with a generated warm classroom background", "generation_prompt": "warm softly lit classroom background, anime visual novel style, subtle desks and window light, no character, no text", "seed": 4102, "tolerance": 0.14},
    {"file": "10271420.jpg", "prompt": "replace the simple pink background with a generated magical night sky", "generation_prompt": "magical night sky background with soft stars and violet gradient, anime illustration background, no character, no text", "seed": 4103, "tolerance": 0.13},
    {"file": "11028520.jpg", "prompt": "replace the plain background with generated sunset clouds", "generation_prompt": "gentle sunset cloud background, warm gold and pale rose colors, anime illustration background, no character, no text", "seed": 4104, "tolerance": 0.12},
    {"file": "10763370.jpg", "prompt": "replace the cream background with generated soft forest light", "generation_prompt": "soft forest light background with blurred green leaves and warm sunlight, anime illustration background, no character, no text", "seed": 4105, "tolerance": 0.12},
    {"file": "10055290.jpg", "prompt": "replace the yellow background with a generated starry pastel gradient", "generation_prompt": "starry pastel gradient background, pale yellow and blue, small glowing stars, anime style, no character, no text", "seed": 4106, "tolerance": 0.15},
    {"file": "10954220.jpg", "prompt": "replace the simple pink background with a generated flower garden", "generation_prompt": "soft flower garden background, pastel pink flowers and bright daylight, anime illustration background, no character, no text", "seed": 4107, "tolerance": 0.13},
    {"file": "10342970.jpg", "prompt": "replace the flat light background with a generated soft bedroom background", "generation_prompt": "soft bedroom background with gentle window light and pastel decor, anime visual novel style, no character, no text", "seed": 4108, "tolerance": 0.14},
    {"file": "10715850.jpg", "prompt": "replace the simple pink background with a generated city-at-dusk scene", "generation_prompt": "soft city at dusk background, warm window lights and pink evening sky, anime illustration background, no character, no text", "seed": 4109, "tolerance": 0.13},
    {"file": "10165650.jpg", "prompt": "replace the simple green background with a generated gentle sky", "generation_prompt": "gentle blue sky background with soft clouds and light bokeh, anime illustration background, no character, no text", "seed": 4110, "tolerance": 0.14},
]


EYE_RECOLOR = [
    {"file": "10069780.jpg", "prompt": "change the gold eyes to blue while keeping the highlights and line art the same", "seeds": [[175, 225], [290, 220]], "bbox": [110, 170, 340, 280], "eye_ellipses": [[142, 190, 205, 260], [258, 186, 322, 260]], "color": "#2F7DFF"},
    {"file": "10064830.jpg", "prompt": "change the brown eyes to green while keeping the highlights and line art the same", "seeds": [[162, 274], [305, 270]], "bbox": [110, 220, 360, 320], "eye_ellipses": [[130, 240, 205, 305], [270, 238, 345, 305]], "color": "#2FAE59"},
    {"file": "10065570.jpg", "prompt": "change the brown eyes to violet while keeping the highlights and line art the same", "seeds": [[180, 235], [310, 235]], "bbox": [100, 180, 370, 300], "eye_ellipses": [[134, 198, 218, 282], [270, 198, 350, 282]], "color": "#8A4DFF"},
    {"file": "10065220.jpg", "prompt": "change the amber eyes to blue while keeping the highlights and line art the same", "seeds": [[160, 245], [310, 215]], "bbox": [100, 170, 365, 290], "eye_ellipses": [[124, 214, 198, 282], [284, 184, 344, 248]], "color": "#2F7DFF"},
    {"file": "10064500.jpg", "prompt": "change the teal eyes to amber while keeping the highlights and line art the same", "seeds": [[170, 285], [300, 282]], "bbox": [110, 220, 360, 330], "eye_ellipses": [[136, 245, 204, 318], [270, 246, 332, 316]], "color": "#D99000"},
    {"file": "10416740.jpg", "prompt": "change the blue eyes to violet while keeping the highlights and line art the same", "seeds": [[250, 276], [350, 274]], "bbox": [190, 210, 400, 330], "eye_ellipses": [[212, 240, 288, 318], [326, 242, 382, 316]], "color": "#8A4DFF"},
    {"file": "10994680.jpg", "prompt": "change the pink eyes to gold while keeping the highlights and line art the same", "seeds": [[170, 285], [330, 285]], "bbox": [110, 230, 390, 330], "eye_ellipses": [[145, 262, 205, 318], [306, 262, 366, 318]], "color": "#D8A600"},
    {"file": "10545030.jpg", "prompt": "change the brown eyes to blue while keeping the highlights and line art the same", "seeds": [[170, 300], [332, 296]], "bbox": [105, 245, 395, 345], "eye_ellipses": [[136, 278, 204, 338], [286, 272, 348, 330]], "color": "#317EFF"},
    {"file": "10065020.jpg", "prompt": "change the red eyes to blue while keeping the highlights and line art the same", "seeds": [[160, 265], [310, 260]], "bbox": [100, 210, 360, 330], "eye_ellipses": [[130, 235, 198, 308], [272, 232, 340, 304]], "color": "#2F7DFF"},
    {"file": "10064382.jpg", "prompt": "change the amber eyes to teal while keeping the highlights and line art the same", "seeds": [[180, 270], [305, 300]], "bbox": [120, 220, 350, 340], "eye_ellipses": [[145, 235, 205, 305], [270, 260, 330, 330]], "color": "#2FB7B5"},
]


HAIR_RECOLOR = [
    {"file": "10323090.jpg", "prompt": "change the magenta hair to blue while preserving the painted shading", "seeds": [[145, 122], [228, 78], [315, 150]], "bbox": [0, 0, 430, 360], "color": "#2E72FF", "hue": 32, "sat": 0.55, "val": 0.70},
    {"file": "10568660.jpg", "prompt": "change the dark brown hair to silver while preserving the painted shading", "seeds": [[220, 120], [290, 125], [170, 190]], "bbox": [55, 20, 455, 355], "color": "#C9CED6", "hue": 45, "sat": 0.70, "val": 0.75, "method": "material_hsl"},
    {"file": "10653400.jpg", "prompt": "change the brown hair to auburn while preserving the highlights", "seeds": [[260, 112], [330, 170], [188, 198]], "bbox": [90, 10, 440, 360], "color": "#A24724", "hue": 36, "sat": 0.55, "val": 0.75},
    {"file": "10190400.jpg", "prompt": "change the dark green hair to purple while preserving the highlights", "seeds": [[230, 145], [292, 136], [182, 202]], "bbox": [80, 30, 430, 370], "color": "#6A45A8", "hue": 50, "sat": 0.60, "val": 0.80, "method": "material_hsl"},
    {"file": "10274430.jpg", "prompt": "change the dark hair to blue-black while preserving the line art", "seeds": [[244, 130], [315, 180], [184, 220]], "bbox": [75, 35, 425, 365], "color": "#162C58", "hue": 55, "sat": 0.70, "val": 0.80, "method": "material_hsl"},
    {"file": "10771300.jpg", "prompt": "change the green hair to lavender while preserving the highlights", "seeds": [[190, 175], [290, 135], [342, 245]], "bbox": [0, 45, 450, 405], "color": "#B78AF5", "hue": 35, "sat": 0.55, "val": 0.75},
    {"file": "10288100.jpg", "prompt": "change the green hair to teal while preserving the highlights", "seeds": [[250, 135], [165, 235], [370, 218]], "bbox": [40, 15, 470, 395], "color": "#19A7A6", "hue": 32, "sat": 0.55, "val": 0.72},
    {"file": "10059270.jpg", "prompt": "change the blonde hair to light pink while preserving the painted shading", "seeds": [[244, 130], [320, 170], [185, 210]], "bbox": [85, 35, 440, 385], "color": "#F5A1C8", "hue": 40, "sat": 0.55, "val": 0.78},
    {"file": "10025830.jpg", "prompt": "change the blonde hair to teal while preserving the highlights and shadows", "seeds": [[235, 105], [282, 190], [215, 290]], "bbox": [105, 10, 405, 445], "color": "#2EAAA3", "hue": 42, "sat": 0.58, "val": 0.80},
    {"file": "10103540.jpg", "prompt": "change the pale yellow hair strands to mint green while preserving the pink hair", "seeds": [[215, 185], [277, 128], [336, 200]], "bbox": [90, 20, 440, 390], "color": "#7CE3B1", "hue": 34, "sat": 0.60, "val": 0.78},
]


SMALL_PATTERNS = [
    {"file": "10071430.jpg", "prompt": "add three small white shine stripes to the blue bow loops", "marks": [{"type": "line", "points": [[145, 65], [174, 90]], "color": "#FFFFFF", "width": 5, "description": "Add a white shine stripe to the left blue bow loop."}, {"type": "line", "points": [[325, 62], [294, 92]], "color": "#FFFFFF", "width": 5, "description": "Add a white shine stripe to the right blue bow loop."}, {"type": "line", "points": [[248, 92], [270, 118]], "color": "#FFFFFF", "width": 4, "description": "Add a smaller white shine stripe on the center loop."}], "observation_points": [[145, 65], [325, 62], [248, 92]], "observation_sizes": [5, 5, 4]},
    {"file": "10214400.jpg", "prompt": "add tiny white dots to the bunny hood", "marks": [{"type": "circle", "x": 115, "y": 188, "r": 6, "color": "#FFFFFF", "description": "Add a small white dot to the left hood area."}, {"type": "circle", "x": 382, "y": 166, "r": 6, "color": "#FFFFFF", "description": "Add a small white dot to the right hood area."}, {"type": "circle", "x": 425, "y": 238, "r": 5, "color": "#FFFFFF", "description": "Add a small white dot to the lower right hood area."}], "observation_points": [[115, 188], [382, 166], [425, 238]], "observation_sizes": [12, 12, 10]},
    {"file": "10458120.jpg", "prompt": "add two gold dots to the frog hair ornament", "marks": [{"type": "circle", "x": 360, "y": 83, "r": 5, "color": "#FFD84D", "description": "Add a gold dot to the frog ornament."}, {"type": "circle", "x": 391, "y": 96, "r": 5, "color": "#FFD84D", "description": "Add a second gold dot to the frog ornament."}], "observation_points": [[360, 83], [391, 96]], "observation_sizes": [10, 10]},
    {"file": "10960520.jpg", "prompt": "add three white dots to the geometric hair clip", "marks": [{"type": "circle", "x": 353, "y": 80, "r": 4, "color": "#FFFFFF", "description": "Add a white dot to the hair clip."}, {"type": "circle", "x": 374, "y": 100, "r": 4, "color": "#FFFFFF", "description": "Add a second white dot to the hair clip."}, {"type": "circle", "x": 394, "y": 119, "r": 4, "color": "#FFFFFF", "description": "Add a third white dot to the hair clip."}], "observation_points": [[353, 80], [374, 100], [394, 119]], "observation_sizes": [8, 8, 8]},
    {"file": "10742120.jpg", "prompt": "add tiny stars near the crescent hair ornament", "marks": [{"type": "line", "points": [[370, 94], [386, 110]], "color": "#FFFFFF", "width": 3, "description": "Draw one stroke of a tiny star."}, {"type": "line", "points": [[386, 94], [370, 110]], "color": "#FFFFFF", "width": 3, "description": "Draw the crossing stroke of a tiny star."}, {"type": "line", "points": [[425, 72], [438, 85]], "color": "#FFFFFF", "width": 3, "description": "Draw one stroke of a second tiny star."}, {"type": "line", "points": [[438, 72], [425, 85]], "color": "#FFFFFF", "width": 3, "description": "Draw the crossing stroke of a second tiny star."}], "observation_points": [[378, 102], [431, 78]], "observation_sizes": [16, 13]},
    {"file": "10443230.jpg", "prompt": "add two pale stripes to the cat ears", "marks": [{"type": "line", "points": [[133, 92], [162, 124]], "color": "#F7DFFF", "width": 5, "description": "Add a pale stripe to the left cat ear."}, {"type": "line", "points": [[365, 82], [337, 118]], "color": "#F7DFFF", "width": 5, "description": "Add a pale stripe to the right cat ear."}], "observation_points": [[133, 92], [365, 82]], "observation_sizes": [5, 5]},
    {"file": "10985490.jpg", "prompt": "add pale stripes to the black cat ears", "marks": [{"type": "line", "points": [[145, 78], [171, 115]], "color": "#F5D5E6", "width": 5, "description": "Add a pale stripe to the left black cat ear."}, {"type": "line", "points": [[370, 72], [342, 118]], "color": "#F5D5E6", "width": 5, "description": "Add a pale stripe to the right black cat ear."}], "observation_points": [[145, 78], [370, 72]], "observation_sizes": [5, 5]},
    {"file": "11038260.jpg", "prompt": "add small blue stars to the maid headband", "marks": [{"type": "line", "points": [[211, 80], [225, 94]], "color": "#7DB7FF", "width": 3, "description": "Draw one stroke of a small blue star on the headband."}, {"type": "line", "points": [[225, 80], [211, 94]], "color": "#7DB7FF", "width": 3, "description": "Draw the crossing stroke of the first blue star."}, {"type": "line", "points": [[303, 82], [317, 96]], "color": "#7DB7FF", "width": 3, "description": "Draw one stroke of a second blue star on the headband."}, {"type": "line", "points": [[317, 82], [303, 96]], "color": "#7DB7FF", "width": 3, "description": "Draw the crossing stroke of the second blue star."}], "observation_points": [[218, 87], [310, 89]], "observation_sizes": [14, 14]},
    {"file": "10987360.jpg", "prompt": "add two pale diagonal stripes to the collar area", "marks": [{"type": "line", "points": [[280, 398], [320, 448]], "color": "#FFE7A8", "width": 7, "description": "Add a pale diagonal stripe to the collar area."}, {"type": "line", "points": [[326, 392], [368, 442]], "color": "#FFE7A8", "width": 7, "description": "Add a second pale diagonal stripe to the collar area."}], "observation_points": [[280, 398], [326, 392]], "observation_sizes": [7, 7]},
    {"file": "10802580.jpg", "prompt": "add tiny white dots to the glasses frame", "marks": [{"type": "circle", "x": 314, "y": 179, "r": 4, "color": "#FFFFFF", "description": "Add a small white dot to the left glasses rim."}, {"type": "circle", "x": 413, "y": 180, "r": 4, "color": "#FFFFFF", "description": "Add a small white dot to the right glasses rim."}, {"type": "circle", "x": 363, "y": 168, "r": 3, "color": "#FFFFFF", "description": "Add a small white dot to the bridge of the glasses."}], "observation_points": [[314, 179], [413, 180], [363, 168]], "observation_sizes": [8, 8, 6]},
]


CLEAN_REGION_GENERATION = [
    {"file": "10920460.jpg", "prompt": "generate a tiny landscape card in the blank right side", "bbox": [330, 42, 492, 152], "generation_prompt": "tiny framed landscape card, green hills and blue sky, painterly anime postcard, no text", "seed": 8101},
    {"file": "10480800.jpg", "prompt": "generate a small flower card in the blank upper right", "bbox": [340, 34, 492, 142], "generation_prompt": "small framed flower illustration, soft blue and white petals, anime postcard, no text", "seed": 8102},
    {"file": "10198530.jpg", "prompt": "generate a small moonlit window in the blank background", "bbox": [340, 38, 492, 148], "generation_prompt": "small moonlit window picture, glowing crescent moon and dark blue sky, anime postcard, no text", "seed": 8103},
    {"file": "10637080.jpg", "prompt": "generate a small poster image inside the clean red panel", "bbox": [338, 36, 492, 150], "generation_prompt": "small cute poster image of a pastel castle, anime postcard, no text", "seed": 8104},
    {"file": "10794130.jpg", "prompt": "generate a little framed picture in the blank upper left", "bbox": [26, 34, 166, 132], "generation_prompt": "little framed picture of a calm ocean sunset, anime postcard, no text", "seed": 8105},
    {"file": "10941030.jpg", "prompt": "generate a small starry card in the blank upper right", "bbox": [335, 42, 492, 150], "generation_prompt": "small starry sky card with gentle blue nebula, anime postcard, no text", "seed": 8106},
    {"file": "10862471.jpg", "prompt": "generate a small gold emblem in the blank side area", "bbox": [338, 38, 492, 146], "generation_prompt": "small ornate gold emblem on a pale card, clean fantasy icon, no text", "seed": 8107},
    {"file": "10064240.jpg", "prompt": "generate a tiny beach photo in the blank corner", "bbox": [338, 36, 492, 146], "generation_prompt": "tiny beach postcard, blue ocean, sand, bright sky, anime background card, no text", "seed": 8108},
    {"file": "10604290.jpg", "prompt": "generate a small flower photo in the blank right side", "bbox": [352, 36, 500, 146], "generation_prompt": "small framed flower photo, pale pink blossoms, soft daylight, no text", "seed": 8109},
    {"file": "11022560.jpg", "prompt": "generate a small sunset card in the clean background", "bbox": [26, 38, 174, 146], "generation_prompt": "small sunset card, orange clouds over calm water, anime postcard, no text", "seed": 8110},
]


STICKERS = [
    {"file": "10851340.jpg", "prompt": "add three small hearts in the upper left", "stickers": [{"name": "large peach heart", "text": "\u2665", "pos": [70, 65], "size": 36, "color": "#FF8CB3", "angle": -12}, {"name": "medium peach heart", "text": "\u2665", "pos": [120, 102], "size": 28, "color": "#FFB0CC", "angle": 9}, {"name": "small peach heart", "text": "\u2665", "pos": [44, 130], "size": 22, "color": "#FF7FA8", "angle": -6}]},
    {"file": "10554220.jpg", "prompt": "add two sparkles near the face", "stickers": [{"name": "left blue sparkle", "text": "\u2726", "pos": [96, 94], "size": 32, "color": "#B7F0FF", "angle": -10}, {"name": "right blue sparkle", "text": "\u2726", "pos": [430, 116], "size": 28, "color": "#CFF7FF", "angle": 8}]},
    {"file": "10554420.jpg", "prompt": "add small butterfly stickers around the flowers", "stickers": [{"name": "pink butterfly", "text": "\u2726", "pos": [92, 84], "size": 30, "color": "#FF9ED0", "angle": -15}, {"name": "blue butterfly", "text": "\u2726", "pos": [424, 108], "size": 28, "color": "#A5D8FF", "angle": 12}]},
    {"file": "10082880.jpg", "prompt": "add pastel stars around the character", "stickers": [{"name": "left pastel star", "text": "\u2605", "pos": [58, 95], "size": 28, "color": "#FFE680", "angle": -8}, {"name": "right pastel star", "text": "\u2605", "pos": [444, 98], "size": 30, "color": "#C9A7FF", "angle": 10}, {"name": "lower pastel star", "text": "\u2605", "pos": [405, 390], "size": 24, "color": "#A7F5FF", "angle": -5}]},
    {"file": "10438890.jpg", "prompt": "add two heart stickers near the bandana", "stickers": [{"name": "left pale heart", "text": "\u2665", "pos": [78, 92], "size": 30, "color": "#FF99C8", "angle": -10}, {"name": "right pale heart", "text": "\u2665", "pos": [436, 102], "size": 28, "color": "#FFC0D8", "angle": 10}]},
    {"file": "10122910.jpg", "prompt": "add soft sparkle stickers in the empty background", "stickers": [{"name": "soft sparkle one", "text": "\u2726", "pos": [412, 78], "size": 34, "color": "#EAD7FF", "angle": 8}, {"name": "soft sparkle two", "text": "\u2726", "pos": [455, 150], "size": 24, "color": "#FFFFFF", "angle": -12}]},
    {"file": "10439280.jpg", "prompt": "add small crescent moons around the hat", "stickers": [{"name": "left crescent", "text": "\u263E", "pos": [78, 72], "size": 30, "color": "#FFE680", "angle": -10}, {"name": "right crescent", "text": "\u263E", "pos": [442, 112], "size": 28, "color": "#FFEFA8", "angle": 12}]},
    {"file": "10199090.jpg", "prompt": "add cheerful star stickers beside the character", "stickers": [{"name": "left yellow star", "text": "\u2605", "pos": [62, 112], "size": 30, "color": "#FFE45C", "angle": -8}, {"name": "right yellow star", "text": "\u2605", "pos": [438, 122], "size": 28, "color": "#FFF08A", "angle": 10}]},
    {"file": "10109450.jpg", "prompt": "add small flower stickers in the top corners", "stickers": [{"name": "left flower sticker", "text": "\u273F", "pos": [70, 70], "size": 28, "color": "#FFC1E1", "angle": -8}, {"name": "right flower sticker", "text": "\u273F", "pos": [438, 72], "size": 28, "color": "#B6F2FF", "angle": 8}]},
    {"file": "10983410.jpg", "prompt": "add small heart stickers in the blank side area", "stickers": [{"name": "left gold heart", "text": "\u2665", "pos": [58, 84], "size": 28, "color": "#FFD76E", "angle": -10}, {"name": "right gold heart", "text": "\u2665", "pos": [454, 96], "size": 26, "color": "#FFE28A", "angle": 12}, {"name": "small gold heart", "text": "\u2665", "pos": [430, 174], "size": 20, "color": "#FFE8A8", "angle": -6}]},
]


if __name__ == "__main__":
    raise SystemExit(main())
