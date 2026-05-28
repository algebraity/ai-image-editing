"""Run a local Venice-backed diffusion smoke test.

The script reads the Venice API key from `keys/venice-key` by default or from
`VENICE_API_KEY`, executes one kernel region-generation action, writes a trace,
and exports the final flattened image. It never prints the API key.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_edit_kernel.diffusion import VeniceImageBackend
from ai_edit_kernel.document.document_state import CanvasSpec, DocumentState
from ai_edit_kernel.document.layer import Layer
from ai_edit_kernel.runtime.executor import ExecutionContext, Executor
from ai_edit_kernel.schema.actions import ActionBatch, SCHEMA_VERSION
from ai_edit_kernel.trace.trace_logger import TraceLogger


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    document = make_document(args.canvas_width, args.canvas_height)
    logger = TraceLogger(output_dir / "traces", metadata={"task_type": "venice_diffusion_smoke"})
    logger.start_session(args.prompt)
    logger.log_document_snapshot(document, "initial")

    backend = VeniceImageBackend(
        api_key_path=args.key_path,
        model=args.model,
        default_size=args.size,
        timeout=args.timeout,
    )
    executor = Executor(ExecutionContext(diffusion_backend=backend, trace_sink=logger))
    batch = smoke_batch(args.prompt, args.canvas_width, args.canvas_height, output_dir / "final.png")

    results = executor.execute_batch(document, batch)
    logger.log_document_snapshot(document, "final")
    succeeded = all(result.succeeded() for result in results)
    active_session = logger.session
    if active_session is None:
        raise RuntimeError("trace session was not started")
    active_session.metadata["success"] = succeeded
    active_session.metadata["status"] = "completed" if succeeded else "failed"
    session = logger.end_session()

    summary = {
        "succeeded": succeeded,
        "output_image": str(output_dir / "final.png"),
        "trace_dir": str((output_dir / "traces" / session.id).resolve()),
        "results": [result.to_json() for result in results],
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"succeeded": summary["succeeded"], "output_image": summary["output_image"], "trace_dir": summary["trace_dir"]}, indent=2))
    return 0 if summary["succeeded"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Venice diffusion smoke test through the editing kernel.")
    parser.add_argument(
        "--prompt",
        default="a luminous miniature crystal garden, pastel lights, clean painterly texture, no text",
        help="Prompt sent to Venice.",
    )
    parser.add_argument("--model", default="krea-v2-medium", help="Venice image model.")
    parser.add_argument("--size", default="1024x1024", help="Venice output size string, for example 1024x1024.")
    parser.add_argument(
        "--key-path",
        type=Path,
        default=Path("keys") / "venice-key",
        help="Local Venice API key file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "venice_smoke",
        help="Artifact directory.",
    )
    parser.add_argument("--canvas-width", type=int, default=512, help="Smoke-test canvas width.")
    parser.add_argument("--canvas-height", type=int, default=512, help="Smoke-test canvas height.")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds.")
    return parser.parse_args()


def make_document(width: int, height: int) -> DocumentState:
    pixels = np.zeros((height, width, 4), dtype=np.float32)
    pixels[..., :] = [0.045, 0.05, 0.075, 1.0]
    document = DocumentState(
        id="doc_venice_smoke",
        canvas=CanvasSpec(width=width, height=height),
        layers=[Layer(id="layer_base", name="base", pixels=pixels)],
        active_layer_id="layer_base",
    )
    document.validate()
    return document


def smoke_batch(prompt: str, width: int, height: int, output_path: Path) -> ActionBatch:
    margin_x = max(width // 6, 1)
    margin_y = max(height // 6, 1)
    bbox = [margin_x, margin_y, width - margin_x, height - margin_y]
    return ActionBatch.from_json(
        {
            "schema_version": SCHEMA_VERSION,
            "id": "batch_venice_smoke",
            "actions": [
                {
                    "id": "action_001",
                    "type": "create_mask_from_shape",
                    "target": {"mask_id": "mask_generated_region"},
                    "params": {
                        "name": "generated region",
                        "kind": "diffusion",
                        "shape": {"type": "rectangle", "bbox_xyxy": bbox},
                        "set_active": False,
                    },
                },
                {
                    "id": "action_002",
                    "type": "inpaint_region",
                    "target": {"layer_id": "layer_base", "output_layer_id": "layer_generated"},
                    "write_mask_id": "mask_generated_region",
                    "params": {
                        "prompt": prompt,
                        "mode": "new_layer",
                        "padding": 32,
                        "output_layer_name": "Venice generated region",
                    },
                },
                {
                    "id": "action_003",
                    "type": "export_flat",
                    "target": {},
                    "params": {"path": str(output_path)},
                },
            ],
        }
    )


if __name__ == "__main__":
    raise SystemExit(main())
