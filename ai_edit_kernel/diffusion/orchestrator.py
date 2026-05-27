"""Diffusion orchestration over the kernel's region and document contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ai_edit_kernel.diffusion.adapters.base import backend_name, run_diffusion_backend
from ai_edit_kernel.diffusion.postprocess import composite_generated_crop, fit_rgba_to_size, generated_crop_to_layer
from ai_edit_kernel.diffusion.types import DiffusionJob, DiffusionOperation, DiffusionOptions, DiffusionRegion, DiffusionResult
from ai_edit_kernel.document.document_state import DocumentState
from ai_edit_kernel.document.layer import Layer, LayerKind
from ai_edit_kernel.region import bbox_from_mask, make_region_view
from ai_edit_kernel.schema.actions import Action, ActionResult, ActionStatus


@dataclass(slots=True)
class DiffusionOrchestrator:
    """Prepare, run, clip, composite, and trace diffusion jobs."""

    backend: Any
    trace_sink: Optional[Any] = None
    default_region_padding: int = 0

    def execute_to_layer(self, document: DocumentState, action: Action, operation: DiffusionOperation) -> ActionResult:
        """Run txt2img/img2img and import the result as a full-canvas layer."""
        options = self._options(action)
        target_pixels = None
        source_image = None
        if operation == "img2img":
            layer = document.get_layer(_required(action.target.layer_id, "target.layer_id"))
            _require_pixels(layer)
            target_pixels = np.array(layer.pixels, dtype=np.float32, copy=True)
            source_image = np.array(layer.pixels, dtype=np.float32, copy=True)
        job = DiffusionJob(
            job_id=_job_id(action, document, operation),
            operation=operation,
            options=options,
            canvas_width=document.canvas.width,
            canvas_height=document.canvas.height,
            source_image=source_image,
            target_pixels=target_pixels,
            metadata={"action_id": action.id, "document_revision": document.revision},
        )
        self._log_started(document, action, job, {})
        result = self._run_backend(document, action, job)
        if result.status != "completed" or result.pixels is None:
            self._log_result(document, action, result, {})
        pixels = self._result_pixels_or_raise(result)
        pixels = fit_rgba_to_size(pixels, document.canvas.width, document.canvas.height)
        output_id = _required(action.target.output_layer_id, "target.output_layer_id")
        layer = Layer(
            id=output_id,
            name=action.params.get("output_layer_name", operation),
            kind=LayerKind.RASTER,
            pixels=pixels,
        )
        document.add_layer(layer)
        document.set_active_layer(layer.id)
        result_assets = _assets_for_result(result)
        self._log_result(document, action, result, {"output": self._save_asset(result.job_id, "output", pixels)})
        return ActionResult(
            action_id=action.id,
            status=ActionStatus.EXECUTED,
            created_layer_ids=[layer.id],
            output_assets=result_assets,
            metadata={"diffusion": result.to_json()},
        )

    def execute_region(self, document: DocumentState, action: Action, operation: DiffusionOperation) -> ActionResult:
        """Run a crop-local region generation and composite it through the write mask."""
        target = document.get_layer(_required(action.target.layer_id, "target.layer_id"))
        _require_pixels(target)
        write_mask_id = _required(action.write_mask_id, "write_mask_id")
        options = self._options(action)
        mask_bbox = bbox_from_mask(document.get_mask(write_mask_id).data)
        if mask_bbox is None:
            raise ValueError(f"write mask {write_mask_id!r} is empty")

        view = make_region_view(
            document,
            mask_id=write_mask_id,
            source_layer_id=target.id,
            include_preview=True,
            include_layer_pixels=True,
            include_mask=True,
            default_full_canvas=False,
            padding=options.padding,
        )
        region = DiffusionRegion(
            canvas_width=document.canvas.width,
            canvas_height=document.canvas.height,
            mask_id=write_mask_id,
            mask_bbox_xyxy=mask_bbox,
            padded_bbox_xyxy=view.bbox,
            paste_bbox_xyxy=view.bbox,
        )
        job = DiffusionJob(
            job_id=_job_id(action, document, operation),
            operation=operation,
            options=options,
            canvas_width=document.canvas.width,
            canvas_height=document.canvas.height,
            source_image=view.preview,
            source_mask=view.mask,
            target_pixels=view.layer_pixels,
            region=region,
            metadata={"action_id": action.id, "document_revision": document.revision},
        )
        started_assets = {
            "source_crop": self._save_asset(job.job_id, "source_crop", view.preview),
            "mask_crop": self._save_asset(job.job_id, "mask_crop", view.mask),
            "target_crop": self._save_asset(job.job_id, "target_crop", view.layer_pixels),
        }
        self._log_started(document, action, job, started_assets)
        result = self._run_backend(document, action, job)
        if result.status != "completed" or result.pixels is None:
            self._log_result(document, action, result, {})
        generated = self._result_pixels_or_raise(result)
        generated = fit_rgba_to_size(generated, view.width, view.height)

        output_id = _required(action.target.output_layer_id, "target.output_layer_id")
        if options.mode == "new_layer":
            pixels = generated_crop_to_layer(generated, view.bbox, document.canvas.width, document.canvas.height, view.mask)
            layer = Layer(
                id=output_id,
                name=action.params.get("output_layer_name", operation),
                kind=LayerKind.RASTER,
                pixels=pixels,
            )
            document.add_layer(layer)
            document.set_active_layer(layer.id)
            action_result = ActionResult(
                action_id=action.id,
                status=ActionStatus.EXECUTED,
                created_layer_ids=[layer.id],
                output_assets=_assets_for_result(result),
            )
        else:
            target.pixels = composite_generated_crop(target.pixels, generated, view.bbox, view.mask)
            action_result = ActionResult(
                action_id=action.id,
                status=ActionStatus.EXECUTED,
                changed_layer_ids=[target.id],
                output_assets=_assets_for_result(result),
            )

        result_assets = {"generated_crop": self._save_asset(result.job_id, "generated_crop", generated)}
        self._log_result(document, action, result, result_assets)
        action_result.metadata["diffusion"] = {
            "job": job.to_json(),
            "result": result.to_json(),
            "integration_mode": options.mode,
        }
        return action_result

    def _options(self, action: Action) -> DiffusionOptions:
        options = DiffusionOptions.from_params(action.params)
        if "padding" not in action.params and self.default_region_padding:
            return DiffusionOptions(
                prompt=options.prompt,
                negative_prompt=options.negative_prompt,
                seed=options.seed,
                denoise=options.denoise,
                guidance_scale=options.guidance_scale,
                steps=options.steps,
                backend=options.backend,
                mode=options.mode,
                padding=self.default_region_padding,
                job=options.job,
            )
        return options

    def _result_pixels_or_raise(self, result: DiffusionResult) -> np.ndarray:
        if result.status != "completed" or result.pixels is None:
            raise RuntimeError(result.error or f"diffusion job {result.job_id!r} failed")
        return result.pixels

    def _run_backend(self, document: DocumentState, action: Action, job: DiffusionJob) -> DiffusionResult:
        try:
            return run_diffusion_backend(self.backend, job)
        except Exception as exc:
            result = DiffusionResult(
                job_id=job.job_id,
                status="failed",
                error=str(exc),
                metadata={"exception_type": type(exc).__name__},
            )
            self._log_result(document, action, result, {})
            raise

    def _log_started(self, document: DocumentState, action: Action, job: DiffusionJob, asset_refs: dict[str, Optional[str]]) -> None:
        if self.trace_sink is None or not hasattr(self.trace_sink, "log_diffusion_job_started"):
            return
        self.trace_sink.log_diffusion_job_started(
            document,
            action,
            job.job_id,
            job.operation,
            backend=job.options.backend or backend_name(self.backend),
            params=job.to_json(),
            asset_refs={key: value for key, value in asset_refs.items() if isinstance(value, str)},
            metadata={"orchestrator": "crop_local" if job.region is not None else "full_canvas"},
        )

    def _log_result(self, document: DocumentState, action: Action, result: DiffusionResult, asset_refs: dict[str, Optional[str]]) -> None:
        if self.trace_sink is None or not hasattr(self.trace_sink, "log_diffusion_job_result"):
            return
        self.trace_sink.log_diffusion_job_result(
            document,
            action,
            result.job_id,
            result.status,
            metrics=dict(result.metrics),
            asset_refs={key: value for key, value in asset_refs.items() if isinstance(value, str)},
            metadata=result.to_json(),
        )

    def _save_asset(self, job_id: str, name: str, array: Optional[np.ndarray]) -> Optional[str]:
        if array is None or self.trace_sink is None or not hasattr(self.trace_sink, "save_diffusion_asset"):
            return None
        return self.trace_sink.save_diffusion_asset(job_id, name, array)


def _job_id(action: Action, document: DocumentState, operation: str) -> str:
    return f"{action.id}_{operation}_rev_{document.revision:04d}"


def _assets_for_result(result: DiffusionResult) -> dict[str, Any]:
    return dict(result.assets)


def _required(value: Optional[str], field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} is required")
    return value


def _require_pixels(layer: Layer) -> None:
    if layer.pixels is None:
        raise ValueError(f"layer {layer.id!r} has no pixel data")
    if layer.pixels.ndim != 3 or layer.pixels.shape[2] != 4:
        raise ValueError(f"layer {layer.id!r} pixels must have shape H x W x 4")
