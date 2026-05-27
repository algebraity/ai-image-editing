"""Venice image-generation adapter for kernel-controlled diffusion actions.

This adapter uses Venice's image generation endpoint as a candidate-pixel
source. The kernel still decides where returned pixels may appear: region
actions are clipped through the action write mask by `DiffusionOrchestrator`.

The adapter intentionally keeps the request small and provider-specific options
explicit. Standard kernel fields such as `seed`, `steps`, and
`guidance_scale` are recorded in traces but are not sent to Venice unless the
caller places matching Venice parameters in `DiffusionOptions.job`.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from ai_edit_kernel.diffusion.types import DiffusionJob, DiffusionResult


DEFAULT_VENICE_ENDPOINT = "https://api.venice.ai/api/v1/images/generations"
DEFAULT_VENICE_MODEL = "krea-v2-medium"
DEFAULT_VENICE_KEY_PATH = Path("keys") / "venice-key"

UrlOpener = Callable[[urllib.request.Request, float], bytes]


@dataclass(slots=True)
class VeniceImageBackend:
    """Call Venice's image generation API and return decoded RGBA pixels."""

    name: str = "venice"
    api_key: Optional[str] = None
    api_key_path: str | Path | None = DEFAULT_VENICE_KEY_PATH
    model: str = DEFAULT_VENICE_MODEL
    endpoint: str = DEFAULT_VENICE_ENDPOINT
    default_size: str = "1024x1024"
    timeout: float = 120.0
    opener: Optional[UrlOpener] = None

    def run(self, job: DiffusionJob) -> DiffusionResult:
        """Generate candidate pixels for a typed diffusion job."""
        started = time.monotonic()
        try:
            api_key = self._api_key()
            payload = self._payload(job)
            request = urllib.request.Request(
                self.endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            response_data = self._open(request)
            response = json.loads(response_data.decode("utf-8"))
            pixels = _pixels_from_response(response)
            latency = time.monotonic() - started
            return DiffusionResult(
                job_id=job.job_id,
                status="completed",
                pixels=pixels,
                assets={"backend": self.name, "model": payload["model"], "endpoint": self.endpoint},
                metrics={
                    "latency_seconds": latency,
                    "width": int(pixels.shape[1]),
                    "height": int(pixels.shape[0]),
                },
                metadata={
                    "adapter_mode": self._adapter_mode(job),
                    "request": _sanitized_request(payload),
                    "response_keys": sorted(response.keys()) if isinstance(response, dict) else [],
                },
            )
        except urllib.error.HTTPError as exc:
            return self._failed_result(job, started, f"Venice API HTTP {exc.code}: {_safe_error_body(exc)}")
        except urllib.error.URLError as exc:
            return self._failed_result(job, started, f"Venice API request failed: {exc.reason}")
        except Exception as exc:
            return self._failed_result(job, started, str(exc), exception_type=type(exc).__name__)

    def _payload(self, job: DiffusionJob) -> dict[str, Any]:
        prompt = job.options.prompt.strip()
        if not prompt:
            raise ValueError("Venice image generation requires a non-empty prompt")

        provider_job = dict(job.options.job)
        _reject_secret_options(provider_job)

        model = str(provider_job.pop("model", self.model))
        size = str(provider_job.pop("size", self.default_size))

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "response_format": "b64_json",
        }
        payload.update(provider_job)
        payload["response_format"] = "b64_json"
        return payload

    def _api_key(self) -> str:
        if self.api_key is not None and self.api_key.strip():
            return self.api_key.strip()

        env_key = _env_api_key()
        if env_key is not None:
            return env_key

        if self.api_key_path is not None:
            path = Path(self.api_key_path)
            if path.exists():
                value = path.read_text(encoding="utf-8").strip()
                if value:
                    return value

        raise RuntimeError("Venice API key not found; set VENICE_API_KEY or provide api_key_path")

    def _open(self, request: urllib.request.Request) -> bytes:
        if self.opener is not None:
            return self.opener(request, self.timeout)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return response.read()

    def _adapter_mode(self, job: DiffusionJob) -> str:
        if job.operation == "txt2img":
            return "txt2img_generation"
        return "generation_endpoint_candidate_for_region"

    def _failed_result(
        self,
        job: DiffusionJob,
        started: float,
        error: str,
        *,
        exception_type: str | None = None,
    ) -> DiffusionResult:
        metadata: dict[str, Any] = {"adapter": self.name}
        if exception_type is not None:
            metadata["exception_type"] = exception_type
        return DiffusionResult(
            job_id=job.job_id,
            status="failed",
            error=error,
            metrics={"latency_seconds": time.monotonic() - started},
            metadata=metadata,
        )


def _env_api_key() -> str | None:
    try:
        import os
    except ImportError:  # pragma: no cover - os is always available on supported Python
        return None
    value = os.environ.get("VENICE_API_KEY")
    if value is None or not value.strip():
        return None
    return value.strip()


def _pixels_from_response(response: Any) -> np.ndarray:
    encoded = _b64_image_from_response(response)
    try:
        raw = base64.b64decode(_strip_data_url(encoded), validate=False)
    except Exception as exc:
        raise ValueError("Venice response contained invalid base64 image data") from exc

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on optional Pillow
        raise RuntimeError("Venice image responses require Pillow for decoding") from exc

    with Image.open(BytesIO(raw)) as image:
        pixels = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    return np.clip(pixels, 0.0, 1.0).astype(np.float32)


def _b64_image_from_response(response: Any) -> str:
    if not isinstance(response, dict):
        raise ValueError("Venice response must be a JSON object")
    direct = response.get("b64_json")
    if isinstance(direct, str) and direct:
        return direct

    for key in ("data", "images"):
        items = response.get(key)
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and isinstance(item.get("b64_json"), str) and item["b64_json"]:
                    return item["b64_json"]

    raise ValueError("Venice response did not include b64_json image data")


def _strip_data_url(value: str) -> str:
    if value.startswith("data:") and "," in value:
        return value.split(",", 1)[1]
    return value


def _safe_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", "replace")
    except Exception:
        return ""
    body = " ".join(body.split())
    if len(body) > 500:
        return f"{body[:500]}..."
    return body


def _sanitized_request(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if "key" not in key.lower() and "authorization" not in key.lower()
    }


def _reject_secret_options(payload: dict[str, Any]) -> None:
    forbidden = [
        key
        for key in payload
        if "key" in key.lower() or "authorization" in key.lower() or "token" in key.lower()
    ]
    if forbidden:
        raise ValueError(f"Venice job payload contains secret-like provider option(s): {', '.join(sorted(forbidden))}")
