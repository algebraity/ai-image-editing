#!/usr/bin/env python3
"""Replay an existing local training example through the current kernel."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_edit_kernel.document.document_state import CanvasSpec, ColorSpace, DocumentMetadata, DocumentState
from ai_edit_kernel.document.layer import Layer, LayerKind
from ai_edit_kernel.planning.planner import AIPlanner, PlannerExecutionResult, PlannerRequestBuilder, StaticPlannerBackend
from ai_edit_kernel.runtime.executor import ExecutionContext, Executor
from ai_edit_kernel.runtime.validator import Validator
from ai_edit_kernel.trace.trace_logger import TraceLogger, TraceSession


DATASET_SCHEMA_VERSION = "ai_edit_training_example.v1"


@dataclass(slots=True)
class ReplaySpec:
    example_id: str
    prompt: str
    task_type: str
    document: DocumentState
    planner_output: dict[str, Any]
    observations: list[Any]
    asset_refs: dict[str, str]
    notes: list[str] | None
    source_image_provided: bool
    copied_files: list[str]
    snapshot_dir: Path


class TrainingRequestBuilder(PlannerRequestBuilder):
    """Match the compact planner request shape used by the local examples."""

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
    example_id = normalize_example_id(args.example)
    example_dir = dataset_root / "data" / example_id
    if not example_dir.exists():
        raise FileNotFoundError(f"example directory does not exist: {example_dir}")

    snapshot_dir = snapshot_example(example_dir)
    spec = build_replay_spec(example_id, snapshot_dir)
    summary = run_replay(spec, dataset_root, example_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["succeeded"] and summary["validation_passed"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay an existing training example through the current kernel.")
    parser.add_argument("example", help="Example id or number, such as training0006 or 6.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/algebraity/Programming/aie-training-data"))
    return parser.parse_args()


def normalize_example_id(value: str) -> str:
    if re.fullmatch(r"training\d{4}", value):
        return value
    if re.fullmatch(r"\d{1,4}", value):
        return f"training{int(value):04d}"
    raise ValueError(f"invalid example id {value!r}")


def snapshot_example(example_dir: Path) -> Path:
    temp_root = Path(tempfile.mkdtemp(prefix=f"{example_dir.name}_replay_"))
    snapshot = temp_root / example_dir.name
    shutil.copytree(example_dir, snapshot)
    return snapshot


def build_replay_spec(example_id: str, snapshot_dir: Path) -> ReplaySpec:
    training_example = read_json(snapshot_dir / "training_example.json")
    planner_output = read_json(snapshot_dir / "planner_output.json")
    run_summary = read_json(snapshot_dir / "run_summary.json")
    input_data = training_example["input"]
    prompt = input_data.get("user_prompt") or run_summary["user_prompt"]
    document = build_initial_document(snapshot_dir, input_data["document_summary"])
    copied_files = []
    if (snapshot_dir / "initial_image.png").exists():
        copied_files.append("initial_image.png")
    if (snapshot_dir / "source_image.png").exists():
        copied_files.append("source_image.png")
    return ReplaySpec(
        example_id=example_id,
        prompt=prompt,
        task_type=training_example.get("task_type", "image_editing"),
        document=document,
        planner_output=planner_output,
        observations=input_data.get("observations", []),
        asset_refs=input_data.get("asset_refs", {}),
        notes=run_summary.get("notes"),
        source_image_provided=bool(run_summary.get("source_image_provided", (snapshot_dir / "initial_image.png").exists())),
        copied_files=copied_files,
        snapshot_dir=snapshot_dir,
    )


def build_initial_document(snapshot_dir: Path, summary: dict[str, Any]) -> DocumentState:
    canvas_data = summary["canvas"]
    canvas = CanvasSpec(
        width=int(canvas_data["width"]),
        height=int(canvas_data["height"]),
        color_space=ColorSpace(canvas_data.get("color_space", "srgb")),
        background_color_rgba=tuple(float(value) for value in canvas_data.get("background_color_rgba", [0, 0, 0, 0])),
        dpi=canvas_data.get("dpi"),
    )
    metadata = metadata_from_summary(summary.get("metadata", {}))
    layers = []
    initial_image = snapshot_dir / "initial_image.png"
    if initial_image.exists():
        layer_summary = first_source_layer(summary)
        with Image.open(initial_image) as image:
            pixels = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
        layers.append(
            Layer(
                id=layer_summary.get("id", "source_image"),
                name=layer_summary.get("name", "Source Image"),
                kind=LayerKind(layer_summary.get("kind", "raster")),
                pixels=pixels.astype(np.float32),
                metadata=dict(layer_summary.get("metadata", {})),
            )
        )
    document = DocumentState(
        id=summary.get("id", f"doc_{snapshot_dir.name}_initial"),
        canvas=canvas,
        layers=layers,
        active_layer_id=summary.get("active_layer_id") if layers else None,
        metadata=metadata,
        annotations=dict(summary.get("annotations", {})),
    )
    document.validate()
    return document


def first_source_layer(summary: dict[str, Any]) -> dict[str, Any]:
    layers = summary.get("layers", [])
    if layers:
        return dict(layers[0])
    return {"id": "source_image", "name": "Source Image", "kind": "raster", "metadata": {"source_path": "initial_image.png"}}


def metadata_from_summary(data: dict[str, Any]) -> DocumentMetadata:
    return DocumentMetadata(
        title=data.get("title"),
        author=data.get("author"),
        created_at=data.get("created_at"),
        updated_at=data.get("updated_at"),
        source_file=data.get("source_file"),
        tags=list(data.get("tags", [])),
        custom=dict(data.get("custom", {})),
    )


def run_replay(spec: ReplaySpec, dataset_root: Path, example_dir: Path) -> dict[str, Any]:
    if example_dir.exists():
        shutil.rmtree(example_dir)
    example_dir.mkdir(parents=True)
    for filename in spec.copied_files:
        shutil.copy2(spec.snapshot_dir / filename, example_dir / filename)

    old_cwd = Path.cwd()
    os.chdir(example_dir)
    try:
        logger = TraceLogger(
            Path("trace"),
            metadata={
                "task_type": spec.task_type,
                "source": "existing_training_example_replay",
                "split": "train",
                "planner": {
                    "planner_type": "manual_static_planner",
                    "planner_model": None,
                    "planner_version": "replay_current_kernel.v1",
                    "temperature": 0.0,
                },
                "rights": {
                    "source_image_license": "unknown",
                    "user_provided_content": spec.source_image_provided,
                    "allowed_for_training": True,
                    "contains_personal_data": False,
                },
                "notes": spec.notes or [],
                "external_diffusion_allowed": False,
            },
        )

        label = "initial_source_document" if spec.source_image_provided else "initial_empty_document"
        session = logger.start_session(spec.prompt)
        logger.log_document_snapshot(spec.document, label)
        logger.log_observation(spec.document, {"observations": spec.observations}, "manual_controller_observations")

        asset_refs = dict(spec.asset_refs)
        if not asset_refs:
            asset_refs = {
                "preview": f"previews/doc_rev_{spec.document.revision:04d}_{label}.png",
                "snapshot": f"snapshots/doc_rev_{spec.document.revision:04d}_{label}.json",
            }

        planner = AIPlanner(
            backend=StaticPlannerBackend([spec.planner_output]),
            request_builder=TrainingRequestBuilder(),
            trace_sink=logger,
        )
        executor = Executor(ExecutionContext(trace_sink=logger))
        execution = planner.plan_and_execute(
            spec.prompt,
            spec.document,
            executor,
            observations=spec.observations,
            asset_refs=asset_refs,
            metadata={"example_number": spec.example_id[-4:], "replayed_from_existing_example": True},
        )

        report = Validator().validate_document(spec.document)
        logger.log_validation_report(report, spec.document)
        logger.log_document_snapshot(spec.document, "final_document")

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
        training_example = build_training_example(spec, session, execution, report.has_errors(), session_dir)
        write_json(Path("training_example.json"), training_example)
        shutil.copy2(session_dir / "manifest.json", "manifest.json")
        shutil.copy2(session_dir / "events.jsonl", "events.jsonl")
        zip_directory_with_root(Path("layered_bundle"), Path("layered_bundle.zip"))
        zip_directory_with_root(session_dir, Path("trace_session.zip"))
        run_summary = build_run_summary(spec, execution, succeeded, report.has_errors(), session_dir)
        write_json(Path("run_summary.json"), run_summary)
        validation = build_export_validation(spec, example_dir, run_summary, training_example)
        write_json(Path("export_validation.json"), validation)
        validation = build_export_validation(spec, example_dir, run_summary, training_example)
        write_json(Path("export_validation.json"), validation)
    finally:
        os.chdir(old_cwd)

    return {
        "example_id": spec.example_id,
        "succeeded": succeeded,
        "final_image": f"data/{spec.example_id}/final_image.png",
        "trace": str(run_summary["trace_session_path"]),
        "validation_passed": validation["passed"],
    }


def build_training_example(
    spec: ReplaySpec,
    session: TraceSession,
    execution: PlannerExecutionResult,
    validation_has_errors: bool,
    trace_path: Path,
) -> dict[str, Any]:
    request = execution.planner_result.request
    succeeded = execution.succeeded() and not validation_has_errors
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "example_id": spec.example_id,
        "source_session_id": session.id,
        "example_type": "planner_imitation",
        "task_type": spec.task_type,
        "split": "train",
        "input": {
            "user_prompt": spec.prompt,
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
        "provenance": {"trace_path": str(trace_path), "source": "existing_training_example_replay", "allowed_for_training": True},
    }


def build_run_summary(
    spec: ReplaySpec,
    execution: PlannerExecutionResult,
    succeeded: bool,
    validation_has_errors: bool,
    trace_path: Path,
) -> dict[str, Any]:
    summary = {
        "user_prompt": spec.prompt,
        "succeeded": succeeded,
        "validation_passed": not validation_has_errors,
        "validation_issues": [] if not validation_has_errors else ["document validation failed"],
        "source_image_provided": spec.source_image_provided,
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
        "notes": spec.notes,
    }
    if "initial_image.png" in spec.copied_files:
        summary["initial_image_path"] = "initial_image.png"
    return summary


def build_export_validation(
    spec: ReplaySpec,
    example_dir: Path,
    run_summary: dict[str, Any],
    training_example: dict[str, Any],
) -> dict[str, Any]:
    dataset_root = example_dir.parents[1]
    returned = [
        f"data/{spec.example_id}/final_image.png",
        f"data/{spec.example_id}/planner_request.json",
        f"data/{spec.example_id}/planner_output.json",
        f"data/{spec.example_id}/action_batch.json",
        f"data/{spec.example_id}/manifest.json",
        f"data/{spec.example_id}/events.jsonl",
        f"data/{spec.example_id}/training_example.json",
        f"data/{spec.example_id}/run_summary.json",
        f"data/{spec.example_id}/export_validation.json",
        f"data/{spec.example_id}/layered_bundle.zip",
        f"data/{spec.example_id}/trace_session.zip",
    ]
    if "initial_image.png" in spec.copied_files:
        returned.insert(0, f"data/{spec.example_id}/initial_image.png")
    if "source_image.png" in spec.copied_files:
        returned.insert(0, f"data/{spec.example_id}/source_image.png")
    stale_hits = find_text_hits(example_dir, ["/mnt/data"])
    secret_hits = find_text_hits(example_dir, ["Bearer ", "YOUR_API_KEY", "venice-key"])
    other_example_hits = find_other_example_hits(example_dir, spec.example_id)
    checks = {
        "does_not_reference_other_examples": not other_example_hits,
        "final_image_exists": (example_dir / "final_image.png").exists(),
        "initial_image_present_if_user_provided": not spec.source_image_provided or (example_dir / "initial_image.png").exists(),
        "layered_bundle_exists": (example_dir / "layered_bundle" / "manifest.json").exists(),
        "manifest_success_true": manifest_success(example_dir / "manifest.json"),
        "no_api_keys_or_bearer_tokens": not secret_hits,
        "no_initial_image_file_for_from_scratch": spec.source_image_provided or not (example_dir / "initial_image.png").exists(),
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
        "example_id": spec.example_id,
        "passed": all(checks.values()),
        "checks": checks,
        "returned_artifacts": returned,
        "stale_path_hits": stale_hits,
        "secret_hits": secret_hits,
        "other_example_hits": other_example_hits,
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
    data = read_json(path)
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
