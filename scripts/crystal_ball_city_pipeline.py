"""Run the crystal-ball city diffusion pipeline through the planner and kernel."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_edit_kernel.diffusion import VeniceImageBackend
from ai_edit_kernel.document.document_state import CanvasSpec, DocumentMetadata, DocumentState
from ai_edit_kernel.document.layer import Layer
from ai_edit_kernel.planning import AIPlanner, StaticPlannerBackend
from ai_edit_kernel.runtime.executor import ExecutionContext, Executor
from ai_edit_kernel.trace.trace_logger import TraceLogger


USER_PROMPT = "create a hazy image of a mysterious fantasy city inside of the crystal ball"
VENICE_PROMPT = (
    "a mysterious fantasy city skyline inside a crystal ball, visible dark castle towers "
    "and glowing windows in atmospheric haze, distant rooftops seen through glass, "
    "dreamlike mist, soft cinematic light, painterly fantasy concept art, no text, no watermark"
)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_root = args.output_dir / "trace"

    document = load_document(args.source_image)
    initial_path = args.output_dir / "initial_image.jpg"
    final_path = args.output_dir / "final_image.png"
    bundle_path = args.output_dir / "layered_bundle"
    shutil.copy2(args.source_image, initial_path)

    logger = TraceLogger(
        trace_root,
        metadata={
            "task_type": "diffusion_insertion",
            "source": "manual_planner_venice_demo",
            "split": "test",
            "planner": {
                "planner_type": "manual_static_planner",
                "planner_model": "human_controller",
                "planner_version": "crystal_ball_city_demo.v1",
                "temperature": 0.0,
            },
            "rights": {
                "source_image_license": "unknown",
                "user_provided_content": True,
                "allowed_for_training": True,
                "contains_personal_data": False,
            },
        },
    )
    logger.start_session(USER_PROMPT)
    logger.log_document_snapshot(document, "initial")
    logger.log_observation(document, {"observations": observations()}, "manual_region_selection")

    planner_output = build_planner_output(final_path, bundle_path)
    planner = AIPlanner(backend=StaticPlannerBackend([planner_output]), trace_sink=logger)
    backend = VeniceImageBackend(
        api_key_path=args.key_path,
        model=args.model,
        default_size=args.size,
        timeout=args.timeout,
    )
    executor = Executor(ExecutionContext(diffusion_backend=backend, trace_sink=logger))
    execution = planner.plan_and_execute(
        USER_PROMPT,
        document,
        executor,
        observations=observations(),
        asset_refs={"source_image": str(initial_path), "final_export": str(final_path)},
        metadata={"controller": "manual model surrogate", "diffusion_backend": "venice"},
    )

    logger.log_document_snapshot(document, "final")
    succeeded = execution.succeeded()
    active_session = logger.session
    if active_session is None:
        raise RuntimeError("trace session was not started")
    active_session.metadata["success"] = succeeded
    active_session.metadata["status"] = "completed" if succeeded else "failed"
    session = logger.end_session()

    session_dir = trace_root / session.id
    write_json(args.output_dir / "planner_request.json", execution.planner_result.request)
    write_json(args.output_dir / "planner_output.json", execution.planner_result.raw_output)
    write_json(args.output_dir / "action_batch.json", execution.planner_result.action_batch.to_json())
    write_json(args.output_dir / "training_example.json", logger.export_training_example(session))
    shutil.copy2(session_dir / "manifest.json", args.output_dir / "manifest.json")
    shutil.copy2(session_dir / "events.jsonl", args.output_dir / "events.jsonl")

    summary = {
        "success": succeeded,
        "user_prompt": USER_PROMPT,
        "diffusion_prompt": VENICE_PROMPT,
        "source_image": str(args.source_image),
        "initial_image": str(initial_path),
        "final_image": str(final_path),
        "trace_dir": str(session_dir.resolve()),
        "manifest": str((args.output_dir / "manifest.json").resolve()),
        "events": str((args.output_dir / "events.jsonl").resolve()),
        "planner_request": str((args.output_dir / "planner_request.json").resolve()),
        "planner_output": str((args.output_dir / "planner_output.json").resolve()),
        "action_batch": str((args.output_dir / "action_batch.json").resolve()),
        "training_example": str((args.output_dir / "training_example.json").resolve()),
        "layered_bundle": str(bundle_path.resolve()),
        "action_results": [result.to_json() for result in execution.action_results],
        "final_revision": document.revision,
        "mask_stats": mask_stats(document),
    }
    write_json(args.output_dir / "run_summary.json", summary)

    print(json.dumps({"success": succeeded, "final_image": str(final_path), "trace_dir": str(session_dir)}, indent=2))
    return 0 if succeeded else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the crystal-ball city Venice demo through the planner.")
    parser.add_argument("--source-image", type=Path, default=Path("tests") / "crystal_ball.jpg")
    parser.add_argument("--output-dir", type=Path, default=Path("training") / "results" / "crystal_ball_city")
    parser.add_argument("--key-path", type=Path, default=Path("keys") / "venice-key")
    parser.add_argument("--model", default="krea-v2-medium")
    parser.add_argument("--size", default="1024x1024")
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def load_document(path: Path) -> DocumentState:
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    height, width = rgba.shape[:2]
    document = DocumentState(
        id="doc_crystal_ball_city",
        canvas=CanvasSpec(width=width, height=height),
        layers=[Layer(id="layer_source", name="source crystal ball", pixels=rgba.astype(np.float32))],
        active_layer_id="layer_source",
        metadata=DocumentMetadata(
            title="Crystal ball city demo",
            source_file=str(path),
            tags=["crystal_ball", "diffusion", "venice", "planner_demo"],
        ),
    )
    document.validate()
    return document


def observations() -> list[dict[str, Any]]:
    return [
        {
            "id": "crystal_ball_outer_ellipse",
            "type": "ellipse_region",
            "bbox_xyxy": [116, 26, 484, 394],
            "note": "Approximate visible glass sphere boundary in tests/crystal_ball.jpg.",
        },
        {
            "id": "crystal_ball_safe_interior",
            "type": "mask_plan",
            "source": "crystal_ball_outer_ellipse",
            "operation": "shrink_then_feather",
            "shrink_pixels": 18,
            "feather_radius": 3.0,
            "note": "Protects the glass rim and base while allowing city pixels inside the ball.",
        },
        {
            "id": "upper_left_glass_highlight",
            "type": "ellipse_region",
            "bbox_xyxy": [145, 42, 305, 142],
            "note": "Approximate highlight to preserve by clearing generated pixels in this area.",
        },
    ]


def build_planner_output(final_path: Path, bundle_path: Path) -> dict[str, Any]:
    return {
        "schema_version": "ai_edit_planner_output.v1",
        "description": "Generate a mysterious fantasy city as candidate pixels, then let the kernel mask it into the crystal ball.",
        "stop_on_error": True,
        "metadata": {
            "planner_mode": "manual_model_surrogate",
            "venice_model": "krea-v2-medium",
            "kernel_controls_selection_and_compositing": True,
        },
        "actions": [
            {
                "type": "create_mask_from_shape",
                "target": {"mask_id": "mask_crystal_ball_outer"},
                "params": {
                    "name": "crystal ball outer ellipse",
                    "kind": "diffusion",
                    "shape": {"type": "ellipse", "bbox_xyxy": [116, 26, 484, 394]},
                    "set_active": False,
                },
                "description": "Select the visible glass sphere with an ellipse.",
            },
            {
                "type": "shrink_mask",
                "target": {"mask_id": "mask_crystal_ball_interior"},
                "params": {
                    "source_mask_id": "mask_crystal_ball_outer",
                    "pixels": 18,
                    "name": "safe crystal ball interior",
                    "set_active": False,
                },
                "description": "Move the write area inward to protect the glass rim and base.",
            },
            {
                "type": "feather_mask",
                "target": {"mask_id": "mask_crystal_ball_write"},
                "params": {
                    "source_mask_id": "mask_crystal_ball_interior",
                    "radius": 3.0,
                    "name": "soft crystal ball write mask",
                },
                "description": "Soften the transition from generated city pixels to the original glass.",
            },
            {
                "type": "inpaint_region",
                "target": {"layer_id": "layer_source", "output_layer_id": "layer_generated_city"},
                "write_mask_id": "mask_crystal_ball_write",
                "params": {
                    "prompt": VENICE_PROMPT,
                    "negative_prompt": "text, watermark, border, frame, modern city, people, outside the crystal ball",
                    "mode": "new_layer",
                    "padding": 32,
                    "output_layer_name": "generated fantasy city",
                    "job": {"size": "1024x1024"},
                },
                "description": "Ask Venice only for a city image candidate; the kernel clips it into the mask.",
            },
            {
                "type": "create_mask_from_shape",
                "target": {"mask_id": "mask_glass_highlight"},
                "params": {
                    "name": "upper left glass highlight",
                    "kind": "write_guard",
                    "shape": {"type": "ellipse", "bbox_xyxy": [145, 42, 305, 142]},
                    "set_active": False,
                },
                "description": "Select the bright glass highlight that should stay visibly glassy.",
            },
            {
                "type": "feather_mask",
                "target": {"mask_id": "mask_glass_highlight_soft"},
                "params": {
                    "source_mask_id": "mask_glass_highlight",
                    "radius": 1.5,
                    "name": "soft glass highlight",
                },
                "description": "Soften the preserved highlight mask.",
            },
            {
                "type": "adjust_brightness_contrast",
                "target": {"layer_id": "layer_generated_city"},
                "write_mask_id": "mask_crystal_ball_write",
                "params": {"brightness": -0.04, "contrast": 1.38},
                "description": "Make the generated city silhouettes more readable while retaining haze.",
            },
            {
                "type": "clear_region",
                "target": {"layer_id": "layer_generated_city"},
                "write_mask_id": "mask_glass_highlight_soft",
                "params": {"mode": "alpha_to_zero", "preserve_rgb": False},
                "description": "Remove generated pixels from the main highlight so original glass remains visible.",
            },
            {
                "type": "set_layer_opacity",
                "target": {"layer_id": "layer_generated_city"},
                "params": {"opacity": 0.94},
                "description": "Blend the generated city through the pale crystal ball glass.",
            },
            {
                "type": "export_flat",
                "params": {"path": str(final_path)},
                "description": "Export the final flattened edit.",
            },
            {
                "type": "export_layered_bundle",
                "params": {"path": str(bundle_path), "include_preview": True, "include_hidden": True, "overwrite": True},
                "description": "Export layers, masks, preview, and document metadata for inspection.",
            },
        ],
    }


def mask_stats(document: DocumentState) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for mask_id, mask in document.masks.items():
        ys, xs = np.nonzero(mask.data > 0.0)
        bbox = None
        if xs.size:
            bbox = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
        stats[mask_id] = {
            "area_pixels": int(np.count_nonzero(mask.data > 0.0)),
            "soft_sum": float(mask.data.sum()),
            "bbox_xyxy": bbox,
        }
    return stats


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
