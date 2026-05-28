"""Service-owned asset storage for uploads, results, and trace references."""

from __future__ import annotations

import json
import mimetypes
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass(slots=True)
class AssetRecord:
    """Metadata for one binary or JSON asset managed by the service."""

    id: str
    path: Path
    media_type: str
    size_bytes: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": str(self.path),
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class AssetStore:
    """Portable filesystem-backed asset store.

    Hosts can start with inline base64 payloads. Larger demos can move to
    service asset IDs without changing the edit request/result shape.
    """

    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def put_bytes(
        self,
        data: bytes,
        *,
        suffix: str = ".bin",
        media_type: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AssetRecord:
        """Store bytes and return an asset record."""
        if not suffix.startswith("."):
            suffix = f".{suffix}"
        asset_id = f"asset_{uuid.uuid4().hex}"
        path = self.root / f"{asset_id}{suffix}"
        path.write_bytes(data)
        guessed_type = media_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return AssetRecord(asset_id, path, guessed_type, len(data), dict(metadata or {}))

    def put_json(self, data: Any, *, metadata: Optional[dict[str, Any]] = None) -> AssetRecord:
        """Store JSON data and return an asset record."""
        encoded = json.dumps(data, indent=2, sort_keys=True).encode("utf-8")
        return self.put_bytes(encoded, suffix=".json", media_type="application/json", metadata=metadata)

    def get_bytes(self, asset_id: str) -> bytes:
        """Read a previously stored asset by ID."""
        return self.resolve(asset_id).read_bytes()

    def get_record(self, asset_id: str) -> AssetRecord:
        """Return filesystem metadata for a previously stored asset."""
        path = self.resolve(asset_id)
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return AssetRecord(asset_id, path, media_type, path.stat().st_size)

    def resolve(self, asset_id: str) -> Path:
        """Resolve an asset ID without allowing path traversal."""
        if "/" in asset_id or "\\" in asset_id or not asset_id.startswith("asset_"):
            raise ValueError("invalid asset_id")
        matches = list(self.root.glob(f"{asset_id}.*"))
        if not matches:
            raise FileNotFoundError(asset_id)
        if len(matches) > 1:
            raise RuntimeError(f"ambiguous asset_id {asset_id!r}")
        return matches[0]

    def asset_url_path(self, asset_id: str) -> str:
        """Return the HTTP path used to fetch this asset."""
        return f"/v1/assets/{asset_id}"
