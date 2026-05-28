"""JSON contracts for host-neutral edit requests, jobs, and results."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


JsonObject = dict[str, Any]

EDIT_REQUEST_SCHEMA_VERSION = "ai_edit_service.edit_request.v1"
EDIT_RESULT_SCHEMA_VERSION = "ai_edit_service.edit_result.v1"
CAPABILITIES_SCHEMA_VERSION = "ai_edit_service.capabilities.v1"


class PayloadEncoding(str, Enum):
    """Transport encodings accepted by host adapters."""

    PNG_BASE64 = "png_base64"
    RGBA_FLOAT32_BASE64 = "rgba_float32_base64"
    MASK_FLOAT32_BASE64 = "mask_float32_base64"


class JobStatus(str, Enum):
    """Lifecycle state for asynchronous edit work."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass(slots=True)
class ServiceError:
    """JSON-safe error payload returned by the service."""

    code: str
    message: str
    details: JsonObject = field(default_factory=dict)

    def to_json(self) -> JsonObject:
        return {"code": self.code, "message": self.message, "details": dict(self.details)}


@dataclass(slots=True)
class HostContext:
    """Information about the calling editor or adapter."""

    host_name: str
    host_version: Optional[str] = None
    adapter_name: Optional[str] = None
    adapter_version: Optional[str] = None
    platform: Optional[str] = None
    document_id: Optional[str] = None
    document_name: Optional[str] = None
    color_profile: Optional[str] = None
    metadata: JsonObject = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: Optional[JsonObject]) -> "HostContext":
        payload = _mapping_or_empty(data, "host")
        return cls(
            host_name=_string_or_default(payload.get("host_name"), "unknown"),
            host_version=_optional_string(payload.get("host_version"), "host.host_version"),
            adapter_name=_optional_string(payload.get("adapter_name"), "host.adapter_name"),
            adapter_version=_optional_string(payload.get("adapter_version"), "host.adapter_version"),
            platform=_optional_string(payload.get("platform"), "host.platform"),
            document_id=_optional_string(payload.get("document_id"), "host.document_id"),
            document_name=_optional_string(payload.get("document_name"), "host.document_name"),
            color_profile=_optional_string(payload.get("color_profile"), "host.color_profile"),
            metadata=_mapping_or_empty(payload.get("metadata"), "host.metadata"),
        )

    def to_json(self) -> JsonObject:
        return _drop_none(
            {
                "host_name": self.host_name,
                "host_version": self.host_version,
                "adapter_name": self.adapter_name,
                "adapter_version": self.adapter_version,
                "platform": self.platform,
                "document_id": self.document_id,
                "document_name": self.document_name,
                "color_profile": self.color_profile,
                "metadata": dict(self.metadata),
            }
        )


@dataclass(slots=True)
class ImagePayload:
    """Raster image payload for source pixels, previews, and result layers."""

    width: int
    height: int
    encoding: PayloadEncoding
    data_base64: Optional[str] = None
    asset_id: Optional[str] = None
    color_space: str = "srgb"
    color_profile: Optional[str] = None
    metadata: JsonObject = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: JsonObject, field_name: str = "image") -> "ImagePayload":
        payload = _require_mapping(data, field_name)
        return cls(
            width=_positive_int(payload.get("width"), f"{field_name}.width"),
            height=_positive_int(payload.get("height"), f"{field_name}.height"),
            encoding=PayloadEncoding(_required_string(payload, "encoding", f"{field_name}.encoding")),
            data_base64=_optional_string(payload.get("data_base64"), f"{field_name}.data_base64"),
            asset_id=_optional_string(payload.get("asset_id"), f"{field_name}.asset_id"),
            color_space=_string_or_default(payload.get("color_space"), "srgb"),
            color_profile=_optional_string(payload.get("color_profile"), f"{field_name}.color_profile"),
            metadata=_mapping_or_empty(payload.get("metadata"), f"{field_name}.metadata"),
        )

    def to_json(self) -> JsonObject:
        return _drop_none(
            {
                "width": self.width,
                "height": self.height,
                "encoding": self.encoding.value,
                "data_base64": self.data_base64,
                "asset_id": self.asset_id,
                "color_space": self.color_space,
                "color_profile": self.color_profile,
                "metadata": dict(self.metadata),
            }
        )


@dataclass(slots=True)
class MaskPayload:
    """Single-channel mask payload for selections and returned layer masks."""

    width: int
    height: int
    encoding: PayloadEncoding
    data_base64: Optional[str] = None
    asset_id: Optional[str] = None
    name: Optional[str] = None
    metadata: JsonObject = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: Optional[JsonObject], field_name: str = "mask") -> Optional["MaskPayload"]:
        if data is None:
            return None
        payload = _require_mapping(data, field_name)
        return cls(
            width=_positive_int(payload.get("width"), f"{field_name}.width"),
            height=_positive_int(payload.get("height"), f"{field_name}.height"),
            encoding=PayloadEncoding(_required_string(payload, "encoding", f"{field_name}.encoding")),
            data_base64=_optional_string(payload.get("data_base64"), f"{field_name}.data_base64"),
            asset_id=_optional_string(payload.get("asset_id"), f"{field_name}.asset_id"),
            name=_optional_string(payload.get("name"), f"{field_name}.name"),
            metadata=_mapping_or_empty(payload.get("metadata"), f"{field_name}.metadata"),
        )

    def to_json(self) -> JsonObject:
        return _drop_none(
            {
                "width": self.width,
                "height": self.height,
                "encoding": self.encoding.value,
                "data_base64": self.data_base64,
                "asset_id": self.asset_id,
                "name": self.name,
                "metadata": dict(self.metadata),
            }
        )


@dataclass(slots=True)
class InputLayer:
    """One host-provided source layer normalized for the service."""

    id: str
    name: str
    image: ImagePayload
    kind: str = "raster"
    opacity: float = 1.0
    visible: bool = True
    blend_mode: str = "normal"
    mask: Optional[MaskPayload] = None
    metadata: JsonObject = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: JsonObject) -> "InputLayer":
        payload = _require_mapping(data, "layer")
        return cls(
            id=_required_string(payload, "id", "layer.id"),
            name=_string_or_default(payload.get("name"), payload.get("id", "layer")),
            image=ImagePayload.from_json(_require_mapping(payload.get("image"), "layer.image"), "layer.image"),
            kind=_string_or_default(payload.get("kind"), "raster"),
            opacity=_float_or_default(payload.get("opacity"), 1.0, "layer.opacity"),
            visible=_bool_or_default(payload.get("visible"), True, "layer.visible"),
            blend_mode=_string_or_default(payload.get("blend_mode"), "normal"),
            mask=MaskPayload.from_json(payload.get("mask"), "layer.mask"),
            metadata=_mapping_or_empty(payload.get("metadata"), "layer.metadata"),
        )

    def to_json(self) -> JsonObject:
        return _drop_none(
            {
                "id": self.id,
                "name": self.name,
                "image": self.image.to_json(),
                "kind": self.kind,
                "opacity": self.opacity,
                "visible": self.visible,
                "blend_mode": self.blend_mode,
                "mask": None if self.mask is None else self.mask.to_json(),
                "metadata": dict(self.metadata),
            }
        )


@dataclass(slots=True)
class EditRequest:
    """Host-neutral request for one natural-language image edit."""

    prompt: str
    canvas_width: int
    canvas_height: int
    layers: list[InputLayer]
    request_id: str = field(default_factory=lambda: f"req_{uuid.uuid4().hex}")
    schema_version: str = EDIT_REQUEST_SCHEMA_VERSION
    host: HostContext = field(default_factory=lambda: HostContext(host_name="unknown"))
    active_layer_id: Optional[str] = None
    selection_mask: Optional[MaskPayload] = None
    color_space: str = "srgb"
    observations: list[Any] = field(default_factory=list)
    asset_refs: dict[str, str] = field(default_factory=dict)
    options: JsonObject = field(default_factory=dict)
    metadata: JsonObject = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: JsonObject) -> "EditRequest":
        payload = _require_mapping(data, "edit request")
        schema_version = _string_or_default(payload.get("schema_version"), EDIT_REQUEST_SCHEMA_VERSION)
        if schema_version != EDIT_REQUEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported edit request schema_version {schema_version!r}")
        layers_data = payload.get("layers")
        if not isinstance(layers_data, list) or not layers_data:
            raise ValueError("edit request requires a non-empty layers list")
        request = cls(
            schema_version=schema_version,
            request_id=_string_or_default(payload.get("request_id"), f"req_{uuid.uuid4().hex}"),
            prompt=_required_string(payload, "prompt", "edit request.prompt"),
            host=HostContext.from_json(payload.get("host")),
            canvas_width=_positive_int(payload.get("canvas_width"), "edit request.canvas_width"),
            canvas_height=_positive_int(payload.get("canvas_height"), "edit request.canvas_height"),
            color_space=_string_or_default(payload.get("color_space"), "srgb"),
            layers=[InputLayer.from_json(item) for item in layers_data],
            active_layer_id=_optional_string(payload.get("active_layer_id"), "edit request.active_layer_id"),
            selection_mask=MaskPayload.from_json(payload.get("selection_mask"), "edit request.selection_mask"),
            observations=list(payload.get("observations") or []),
            asset_refs=_string_mapping(payload.get("asset_refs"), "edit request.asset_refs"),
            options=_mapping_or_empty(payload.get("options"), "edit request.options"),
            metadata=_mapping_or_empty(payload.get("metadata"), "edit request.metadata"),
        )
        request.validate()
        return request

    def validate(self) -> None:
        """Validate cross-field dimensions and active layer references."""
        layer_ids = {layer.id for layer in self.layers}
        if self.active_layer_id is not None and self.active_layer_id not in layer_ids:
            raise ValueError(f"active_layer_id {self.active_layer_id!r} does not match any layer")
        for layer in self.layers:
            if layer.image.width != self.canvas_width or layer.image.height != self.canvas_height:
                raise ValueError(f"layer {layer.id!r} image dimensions must match the canvas")
            if layer.mask is not None and (layer.mask.width != self.canvas_width or layer.mask.height != self.canvas_height):
                raise ValueError(f"layer {layer.id!r} mask dimensions must match the canvas")
        if self.selection_mask is not None and (
            self.selection_mask.width != self.canvas_width or self.selection_mask.height != self.canvas_height
        ):
            raise ValueError("selection_mask dimensions must match the canvas")

    def to_json(self) -> JsonObject:
        return _drop_none(
            {
                "schema_version": self.schema_version,
                "request_id": self.request_id,
                "prompt": self.prompt,
                "host": self.host.to_json(),
                "canvas_width": self.canvas_width,
                "canvas_height": self.canvas_height,
                "color_space": self.color_space,
                "layers": [layer.to_json() for layer in self.layers],
                "active_layer_id": self.active_layer_id,
                "selection_mask": None if self.selection_mask is None else self.selection_mask.to_json(),
                "observations": list(self.observations),
                "asset_refs": dict(self.asset_refs),
                "options": dict(self.options),
                "metadata": dict(self.metadata),
            }
        )


@dataclass(slots=True)
class ResultLayer:
    """One layer returned to the host for insertion or replacement."""

    id: str
    name: str
    image: ImagePayload
    opacity: float = 1.0
    blend_mode: str = "normal"
    mask: Optional[MaskPayload] = None
    metadata: JsonObject = field(default_factory=dict)

    def to_json(self) -> JsonObject:
        return _drop_none(
            {
                "id": self.id,
                "name": self.name,
                "image": self.image.to_json(),
                "opacity": self.opacity,
                "blend_mode": self.blend_mode,
                "mask": None if self.mask is None else self.mask.to_json(),
                "metadata": dict(self.metadata),
            }
        )


@dataclass(slots=True)
class TraceRef:
    """References to trace artifacts produced while executing an edit."""

    session_id: Optional[str] = None
    manifest_asset_id: Optional[str] = None
    events_asset_id: Optional[str] = None
    local_trace_dir: Optional[str] = None
    metadata: JsonObject = field(default_factory=dict)

    def to_json(self) -> JsonObject:
        return _drop_none(
            {
                "session_id": self.session_id,
                "manifest_asset_id": self.manifest_asset_id,
                "events_asset_id": self.events_asset_id,
                "local_trace_dir": self.local_trace_dir,
                "metadata": dict(self.metadata),
            }
        )


@dataclass(slots=True)
class EditResult:
    """Result returned after a completed edit job."""

    request_id: str
    layers: list[ResultLayer]
    schema_version: str = EDIT_RESULT_SCHEMA_VERSION
    status: JobStatus = JobStatus.SUCCEEDED
    preview: Optional[ImagePayload] = None
    masks: list[MaskPayload] = field(default_factory=list)
    trace: Optional[TraceRef] = None
    action_results: list[JsonObject] = field(default_factory=list)
    document_summary: JsonObject = field(default_factory=dict)
    metadata: JsonObject = field(default_factory=dict)

    def to_json(self) -> JsonObject:
        return _drop_none(
            {
                "schema_version": self.schema_version,
                "request_id": self.request_id,
                "status": self.status.value,
                "layers": [layer.to_json() for layer in self.layers],
                "masks": [mask.to_json() for mask in self.masks],
                "preview": None if self.preview is None else self.preview.to_json(),
                "trace": None if self.trace is None else self.trace.to_json(),
                "action_results": list(self.action_results),
                "document_summary": dict(self.document_summary),
                "metadata": dict(self.metadata),
            }
        )


def service_capabilities() -> JsonObject:
    """Return static capabilities advertised to host adapters."""
    return {
        "schema_version": CAPABILITIES_SCHEMA_VERSION,
        "transports": ["http_json", "inline_base64_assets", "service_asset_refs"],
        "request_schema": EDIT_REQUEST_SCHEMA_VERSION,
        "result_schema": EDIT_RESULT_SCHEMA_VERSION,
        "payload_encodings": [encoding.value for encoding in PayloadEncoding],
        "job_endpoints": ["/v1/edits", "/v1/jobs/{job_id}", "/v1/jobs/{job_id}/result", "/v1/assets/{asset_id}"],
        "host_requirements": {
            "layers": "full-canvas RGBA layers are preferred",
            "masks": "full-canvas single-channel masks in [0, 1]",
            "coordinates": "top-left pixel origin with half-open bbox_xyxy bounds",
        },
    }


def _drop_none(data: JsonObject) -> JsonObject:
    return {key: value for key, value in data.items() if value is not None}


def _require_mapping(value: Any, field_name: str) -> JsonObject:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be an object")
    return dict(value)


def _mapping_or_empty(value: Any, field_name: str) -> JsonObject:
    if value is None:
        return {}
    return _require_mapping(value, field_name)


def _required_string(data: JsonObject, key: str, field_name: str) -> str:
    if key not in data:
        raise ValueError(f"{field_name} is required")
    value = data[key]
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _optional_string(value: Any, field_name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or null")
    return value


def _string_or_default(value: Any, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise TypeError("value must be a string")
    return value


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero")
    return value


def _float_or_default(value: Any, default: float, field_name: str) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number")
    return float(value)


def _bool_or_default(value: Any, default: bool, field_name: str) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean")
    return value


def _string_mapping(value: Any, field_name: str) -> dict[str, str]:
    payload = _mapping_or_empty(value, field_name)
    output: dict[str, str] = {}
    for key, item in payload.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise TypeError(f"{field_name} must map strings to strings")
        output[key] = item
    return output
