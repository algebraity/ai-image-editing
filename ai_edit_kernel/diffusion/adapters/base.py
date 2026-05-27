"""Backend protocol and compatibility helpers for diffusion adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ai_edit_kernel.diffusion.types import DiffusionJob, DiffusionResult


class DiffusionBackend(Protocol):
    """Backend-neutral adapter interface used by the diffusion orchestrator."""

    name: str

    def run(self, job: DiffusionJob) -> DiffusionResult:
        """Generate candidate pixels for `job`."""
        ...


def run_diffusion_backend(backend: Any, job: DiffusionJob) -> DiffusionResult:
    """Run a typed backend or an older loose-dict prototype backend."""
    if backend is None:
        raise RuntimeError(f"{job.operation} requires a configured diffusion_backend")

    if hasattr(backend, "run"):
        result = backend.run(job)
        if isinstance(result, DiffusionResult):
            return result
        if isinstance(result, dict):
            return diffusion_result_from_mapping(job.job_id, result)
        raise TypeError("diffusion backend run() must return DiffusionResult or dict")

    method_name = job.operation
    if method_name == "outpaint" and not hasattr(backend, "outpaint"):
        method_name = "inpaint"
    method = getattr(backend, method_name, None)
    if method is None:
        raise RuntimeError(f"diffusion backend does not implement {job.operation!r}")
    return diffusion_result_from_mapping(job.job_id, method(job.to_legacy_dict()))


def backend_name(backend: Any, fallback: str = "unknown") -> str:
    """Return a stable backend name for traces."""
    value = getattr(backend, "name", None)
    if isinstance(value, str) and value:
        return value
    return type(backend).__name__ if backend is not None else fallback


def diffusion_result_from_mapping(job_id: str, data: dict[str, Any]) -> DiffusionResult:
    """Convert a prototype response dict into `DiffusionResult`."""
    if not isinstance(data, dict):
        raise TypeError("diffusion backend response must be a dictionary")
    pixels = data.get("pixels")
    if pixels is None:
        path = data.get("path") or data.get("image_path")
        if path is not None:
            pixels = _load_rgba_image(Path(path))
    if pixels is not None:
        pixels = _coerce_rgba_pixels(pixels)
    status = data.get("status", "completed" if pixels is not None else "failed")
    if status not in {"completed", "failed"}:
        status = "failed"
    return DiffusionResult(
        job_id=str(data.get("job_id", job_id)),
        status=status,
        pixels=pixels,
        assets=dict(data.get("assets", {})),
        metrics=dict(data.get("metrics", {})),
        error=data.get("error"),
        metadata=dict(data.get("metadata", {})),
    )


def _coerce_rgba_pixels(value: Any) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("diffusion pixels must be a NumPy array")
    pixels = value.astype(np.float32, copy=False)
    if pixels.ndim != 3:
        raise ValueError("diffusion pixels must have shape H x W x C")
    if pixels.shape[2] == 3:
        alpha = np.ones((*pixels.shape[:2], 1), dtype=np.float32)
        pixels = np.concatenate([pixels, alpha], axis=2)
    if pixels.shape[2] != 4:
        raise ValueError("diffusion pixels must have 3 or 4 channels")
    if pixels.max(initial=0.0) > 1.0:
        pixels = pixels / 255.0
    return np.clip(pixels, 0.0, 1.0).astype(np.float32)


def _load_rgba_image(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on optional Pillow
        raise RuntimeError("image-path diffusion responses require Pillow") from exc
    with Image.open(path) as image:
        return (np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0).astype(np.float32)
