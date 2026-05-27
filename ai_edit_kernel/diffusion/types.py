"""Typed diffusion job contracts for the editing kernel.

Diffusion backends are treated as candidate-pixel generators. They receive a
bounded, explicit job description and return pixels that the kernel will still
clip, composite, validate, and trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np

from ai_edit_kernel.region import BBoxXYXY


DiffusionOperation = Literal["txt2img", "img2img", "inpaint", "outpaint"]
DiffusionStatus = Literal["completed", "failed"]


@dataclass(frozen=True, slots=True)
class DiffusionRegion:
    """Canvas mapping for a crop-local diffusion operation."""

    canvas_width: int
    canvas_height: int
    mask_id: str
    mask_bbox_xyxy: BBoxXYXY
    padded_bbox_xyxy: BBoxXYXY
    paste_bbox_xyxy: BBoxXYXY

    def to_json(self) -> dict[str, Any]:
        """Return JSON-safe geometry metadata."""
        return {
            "canvas_size": [self.canvas_width, self.canvas_height],
            "mask_id": self.mask_id,
            "mask_bbox_xyxy": self.mask_bbox_xyxy.as_list(),
            "padded_bbox_xyxy": self.padded_bbox_xyxy.as_list(),
            "paste_bbox_xyxy": self.paste_bbox_xyxy.as_list(),
        }


@dataclass(frozen=True, slots=True)
class DiffusionOptions:
    """Planner/action parameters that control backend generation."""

    prompt: str = ""
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    denoise: Optional[float] = None
    guidance_scale: Optional[float] = None
    steps: Optional[int] = None
    backend: Optional[str] = None
    mode: str = "replace_region"
    padding: int | tuple[int, int, int, int] = 0
    job: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "DiffusionOptions":
        """Build options from an action params object."""
        if not isinstance(params, dict):
            raise TypeError("diffusion params must be a dictionary")
        return cls(
            prompt=str(params.get("prompt", "")),
            negative_prompt=_optional_string(params.get("negative_prompt")),
            seed=_optional_int(params.get("seed")),
            denoise=_optional_float(params.get("denoise")),
            guidance_scale=_optional_float(params.get("guidance_scale")),
            steps=_optional_int(params.get("steps")),
            backend=_optional_string(params.get("backend")),
            mode=str(params.get("mode", "replace_region")),
            padding=_padding_value(params.get("padding", 0)),
            job=dict(params.get("job", {})),
        )

    def to_json(self) -> dict[str, Any]:
        """Return JSON-safe generation options."""
        return {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
            "denoise": self.denoise,
            "guidance_scale": self.guidance_scale,
            "steps": self.steps,
            "backend": self.backend,
            "mode": self.mode,
            "padding": list(self.padding) if isinstance(self.padding, tuple) else self.padding,
            "job": _json_safe_dict(self.job),
        }


@dataclass(slots=True)
class DiffusionJob:
    """Backend-neutral request for generated candidate pixels."""

    job_id: str
    operation: DiffusionOperation
    options: DiffusionOptions = field(default_factory=DiffusionOptions)
    canvas_width: int = 0
    canvas_height: int = 0
    source_image: Optional[np.ndarray] = None
    source_mask: Optional[np.ndarray] = None
    target_pixels: Optional[np.ndarray] = None
    region: Optional[DiffusionRegion] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-safe summary without embedding raw arrays."""
        return {
            "job_id": self.job_id,
            "operation": self.operation,
            "options": self.options.to_json(),
            "canvas": {"width": self.canvas_width, "height": self.canvas_height},
            "region": None if self.region is None else self.region.to_json(),
            "arrays": {
                "source_image": _array_summary(self.source_image),
                "source_mask": _array_summary(self.source_mask),
                "target_pixels": _array_summary(self.target_pixels),
            },
            "metadata": _json_safe_dict(self.metadata),
        }

    def to_legacy_dict(self) -> dict[str, Any]:
        """Return the loose dict shape accepted by prototype backends."""
        payload = dict(self.options.job)
        payload.update(
            {
                "job_id": self.job_id,
                "operation": self.operation,
                "prompt": self.options.prompt,
                "negative_prompt": self.options.negative_prompt,
                "seed": self.options.seed,
                "denoise": self.options.denoise,
                "guidance_scale": self.options.guidance_scale,
                "steps": self.options.steps,
                "backend": self.options.backend,
                "canvas": {"width": self.canvas_width, "height": self.canvas_height},
                "metadata": dict(self.metadata),
            }
        )
        if self.region is not None:
            payload["region"] = self.region.to_json()
            payload["crop_bbox_xyxy"] = self.region.padded_bbox_xyxy.as_list()
            payload["paste_bbox_xyxy"] = self.region.paste_bbox_xyxy.as_list()
        if self.source_image is not None:
            payload["source_image"] = self.source_image
            payload["preview"] = self.source_image
        if self.source_mask is not None:
            payload["source_mask"] = self.source_mask
            payload["mask"] = self.source_mask
        if self.target_pixels is not None:
            payload["target_pixels"] = self.target_pixels
        return payload


@dataclass(slots=True)
class DiffusionResult:
    """Candidate pixels returned by a diffusion backend."""

    job_id: str
    status: DiffusionStatus
    pixels: Optional[np.ndarray] = None
    assets: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-safe result summary."""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "pixels": _array_summary(self.pixels),
            "assets": _json_safe_dict(self.assets),
            "metrics": _json_safe_dict(self.metrics),
            "error": self.error,
            "metadata": _json_safe_dict(self.metadata),
        }


def _array_summary(array: Optional[np.ndarray]) -> Optional[dict[str, Any]]:
    if array is None:
        return None
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(np.min(array)) if array.size else 0.0,
        "max": float(np.max(array)) if array.size else 0.0,
    }


def _optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("value must be a string or None")
    return value


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("value must be an integer or None")
    if int(value) != float(value):
        raise ValueError("value must be an integer")
    return int(value)


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("value must be a number or None")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("value must be finite")
    return number


def _padding_value(value: Any) -> int | tuple[int, int, int, int]:
    if isinstance(value, bool):
        raise TypeError("padding must be an integer or four-integer list")
    if isinstance(value, int):
        if value < 0:
            raise ValueError("padding must be nonnegative")
        return value
    if isinstance(value, (list, tuple)) and len(value) == 4:
        padding = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, int):
                raise TypeError("padding entries must be integers")
            if item < 0:
                raise ValueError("padding entries must be nonnegative")
            padding.append(item)
        return (padding[0], padding[1], padding[2], padding[3])
    raise TypeError("padding must be an integer or four-integer list")


def _json_safe_dict(value: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            continue
        if isinstance(item, np.ndarray):
            result[key] = _array_summary(item)
        elif isinstance(item, (str, int, float, bool)) or item is None:
            result[key] = item
        elif isinstance(item, dict):
            result[key] = _json_safe_dict(item)
        elif isinstance(item, (list, tuple)):
            result[key] = [
                _json_safe_dict(element) if isinstance(element, dict) else element
                for element in item
                if isinstance(element, (str, int, float, bool, dict)) or element is None
            ]
        else:
            result[key] = str(item)
    return result
