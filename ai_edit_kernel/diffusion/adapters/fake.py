"""Deterministic diffusion backend used by tests and local demos."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ai_edit_kernel.diffusion.types import DiffusionJob, DiffusionResult


@dataclass(slots=True)
class FakeDiffusionBackend:
    """Return deterministic RGBA crops without network or model dependencies."""

    name: str = "fake_diffusion"
    operation_colors: dict[str, tuple[float, float, float, float]] = field(
        default_factory=lambda: {
            "txt2img": (1.0, 0.0, 0.0, 1.0),
            "img2img": (0.0, 1.0, 0.0, 1.0),
            "inpaint": (0.0, 0.0, 1.0, 1.0),
            "outpaint": (0.0, 1.0, 1.0, 1.0),
        }
    )

    def run(self, job: DiffusionJob) -> DiffusionResult:
        """Return a solid crop for the requested operation."""
        height, width = _job_output_shape(job)
        pixels = np.zeros((height, width, 4), dtype=np.float32)
        pixels[..., :] = self.operation_colors.get(job.operation, (1.0, 0.0, 1.0, 1.0))
        return DiffusionResult(
            job_id=job.job_id,
            status="completed",
            pixels=pixels,
            assets={"backend": self.name},
            metrics={"width": width, "height": height, "deterministic": True},
            metadata={"fake": True},
        )


def _job_output_shape(job: DiffusionJob) -> tuple[int, int]:
    if job.source_image is not None:
        return int(job.source_image.shape[0]), int(job.source_image.shape[1])
    if job.target_pixels is not None:
        return int(job.target_pixels.shape[0]), int(job.target_pixels.shape[1])
    if job.region is not None:
        return job.region.padded_bbox_xyxy.height, job.region.padded_bbox_xyxy.width
    return int(job.canvas_height), int(job.canvas_width)
