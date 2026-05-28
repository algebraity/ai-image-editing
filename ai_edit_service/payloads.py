"""Conversion helpers between service payloads and kernel arrays."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Callable, Optional

import numpy as np
from PIL import Image

from ai_edit_service.models import ImagePayload, MaskPayload, PayloadEncoding


AssetResolver = Callable[[str], bytes]


def decode_image_payload(payload: ImagePayload, asset_resolver: Optional[AssetResolver] = None) -> np.ndarray:
    """Return a full-canvas `float32` RGBA array from a service image payload."""
    raw = _payload_bytes(payload.data_base64, payload.asset_id, asset_resolver)
    if payload.encoding == PayloadEncoding.PNG_BASE64:
        with Image.open(BytesIO(raw)) as image:
            rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
        return _validate_rgba(rgba, payload.width, payload.height)
    if payload.encoding == PayloadEncoding.RGBA_FLOAT32_BASE64:
        rgba = np.frombuffer(raw, dtype="<f4").reshape((payload.height, payload.width, 4))
        return _validate_rgba(rgba.astype(np.float32, copy=True), payload.width, payload.height)
    raise ValueError(f"image payload does not support encoding {payload.encoding.value!r}")


def decode_mask_payload(payload: MaskPayload, asset_resolver: Optional[AssetResolver] = None) -> np.ndarray:
    """Return a full-canvas `float32` mask array from a service mask payload."""
    raw = _payload_bytes(payload.data_base64, payload.asset_id, asset_resolver)
    if payload.encoding == PayloadEncoding.PNG_BASE64:
        with Image.open(BytesIO(raw)) as image:
            mask = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        return _validate_mask(mask, payload.width, payload.height)
    if payload.encoding == PayloadEncoding.MASK_FLOAT32_BASE64:
        mask = np.frombuffer(raw, dtype="<f4").reshape((payload.height, payload.width))
        return _validate_mask(mask.astype(np.float32, copy=True), payload.width, payload.height)
    raise ValueError(f"mask payload does not support encoding {payload.encoding.value!r}")


def encode_image_payload(
    pixels: np.ndarray,
    *,
    encoding: PayloadEncoding = PayloadEncoding.PNG_BASE64,
    color_space: str = "srgb",
    color_profile: Optional[str] = None,
) -> ImagePayload:
    """Return an inline service image payload from a kernel RGBA array."""
    rgba = _validate_rgba_array(pixels)
    height, width = rgba.shape[:2]
    if encoding == PayloadEncoding.PNG_BASE64:
        image = Image.fromarray(np.rint(np.clip(rgba, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGBA")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        data = base64.b64encode(buffer.getvalue()).decode("ascii")
    elif encoding == PayloadEncoding.RGBA_FLOAT32_BASE64:
        data = base64.b64encode(np.asarray(rgba, dtype="<f4").tobytes()).decode("ascii")
    else:
        raise ValueError(f"image payload does not support encoding {encoding.value!r}")
    return ImagePayload(
        width=width,
        height=height,
        encoding=encoding,
        data_base64=data,
        color_space=color_space,
        color_profile=color_profile,
    )


def encode_png_bytes(pixels: np.ndarray) -> bytes:
    """Return PNG bytes for a kernel RGBA array."""
    rgba = _validate_rgba_array(pixels)
    image = Image.fromarray(np.rint(np.clip(rgba, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGBA")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def encode_mask_payload(
    mask: np.ndarray,
    *,
    name: Optional[str] = None,
    encoding: PayloadEncoding = PayloadEncoding.PNG_BASE64,
) -> MaskPayload:
    """Return an inline service mask payload from a kernel mask array."""
    data = _validate_mask_array(mask)
    height, width = data.shape
    if encoding == PayloadEncoding.PNG_BASE64:
        image = Image.fromarray(np.rint(np.clip(data, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    elif encoding == PayloadEncoding.MASK_FLOAT32_BASE64:
        encoded = base64.b64encode(np.asarray(data, dtype="<f4").tobytes()).decode("ascii")
    else:
        raise ValueError(f"mask payload does not support encoding {encoding.value!r}")
    return MaskPayload(width=width, height=height, encoding=encoding, data_base64=encoded, name=name)


def _payload_bytes(data_base64: Optional[str], asset_id: Optional[str], asset_resolver: Optional[AssetResolver]) -> bytes:
    if data_base64 is not None:
        return base64.b64decode(data_base64.encode("ascii"))
    if asset_id is not None:
        if asset_resolver is None:
            raise ValueError("asset_id payload requires an asset resolver")
        return asset_resolver(asset_id)
    raise ValueError("payload requires data_base64 or asset_id")


def _validate_rgba(rgba: np.ndarray, width: int, height: int) -> np.ndarray:
    if rgba.shape != (height, width, 4):
        raise ValueError(f"RGBA payload shape {rgba.shape!r} does not match {(height, width, 4)!r}")
    return _validate_rgba_array(rgba)


def _validate_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape != (height, width):
        raise ValueError(f"mask payload shape {mask.shape!r} does not match {(height, width)!r}")
    return _validate_mask_array(mask)


def _validate_rgba_array(rgba: np.ndarray) -> np.ndarray:
    if not isinstance(rgba, np.ndarray):
        raise TypeError("pixels must be a NumPy array")
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("pixels must have shape height x width x 4")
    output = np.asarray(rgba, dtype=np.float32)
    if not np.all(np.isfinite(output)):
        raise ValueError("pixels must contain only finite values")
    return np.clip(output, 0.0, 1.0).astype(np.float32, copy=False)


def _validate_mask_array(mask: np.ndarray) -> np.ndarray:
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask must be a NumPy array")
    if mask.ndim != 2:
        raise ValueError("mask must have shape height x width")
    output = np.asarray(mask, dtype=np.float32)
    if not np.all(np.isfinite(output)):
        raise ValueError("mask must contain only finite values")
    return np.clip(output, 0.0, 1.0).astype(np.float32, copy=False)
