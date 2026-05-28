"""Configuration for the host-neutral AI edit service."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass(slots=True)
class ServiceConfig:
    """Runtime settings shared by the HTTP service, jobs, assets, and runner."""

    host: str = "127.0.0.1"
    port: int = 8765
    workspace_dir: Path = Path("artifacts") / "ai_edit_service"
    assets_dir: Optional[Path] = None
    traces_dir: Optional[Path] = None
    max_request_bytes: int = 256 * 1024 * 1024
    enable_cors: bool = True
    allowed_origins: list[str] = field(default_factory=lambda: ["*"])
    planner_backend: str = "venice"
    planner_model: str = "openai-gpt-54"
    planner_endpoint: str = "https://api.venice.ai/api/v1/chat/completions"
    planner_api_key_path: Optional[Path] = Path("keys") / "venice-key"
    planner_temperature: float = 0.1
    planner_max_tokens: int = 16384
    planner_timeout: float = 360.0
    diffusion_backend: str = "venice"
    diffusion_model: str = "krea-v2-medium"
    diffusion_api_key_path: Optional[Path] = Path("keys") / "venice-key"
    diffusion_default_size: str = "1024x1024"
    diffusion_timeout: float = 120.0
    service_name: str = "ai-edit-service"
    service_version: str = "0.1.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, prefix: str = "AI_EDIT_SERVICE_") -> "ServiceConfig":
        """Build a config from environment variables with portable defaults."""
        defaults = cls()
        workspace = Path(os.environ.get(f"{prefix}WORKSPACE_DIR", str(defaults.workspace_dir)))
        return cls(
            host=os.environ.get(f"{prefix}HOST", defaults.host),
            port=_int_from_env(f"{prefix}PORT", defaults.port),
            workspace_dir=workspace,
            assets_dir=_optional_path_from_env(f"{prefix}ASSETS_DIR"),
            traces_dir=_optional_path_from_env(f"{prefix}TRACES_DIR"),
            max_request_bytes=_int_from_env(f"{prefix}MAX_REQUEST_BYTES", defaults.max_request_bytes),
            enable_cors=_bool_from_env(f"{prefix}ENABLE_CORS", True),
            allowed_origins=_list_from_env(f"{prefix}ALLOWED_ORIGINS", ["*"]),
            planner_backend=os.environ.get(f"{prefix}PLANNER_BACKEND", defaults.planner_backend),
            planner_model=os.environ.get(f"{prefix}PLANNER_MODEL", defaults.planner_model),
            planner_endpoint=os.environ.get(f"{prefix}PLANNER_ENDPOINT", defaults.planner_endpoint),
            planner_api_key_path=_path_from_env(f"{prefix}PLANNER_API_KEY_PATH", defaults.planner_api_key_path),
            planner_temperature=_float_from_env(f"{prefix}PLANNER_TEMPERATURE", defaults.planner_temperature),
            planner_max_tokens=_int_from_env(f"{prefix}PLANNER_MAX_TOKENS", defaults.planner_max_tokens),
            planner_timeout=_float_from_env(f"{prefix}PLANNER_TIMEOUT", defaults.planner_timeout),
            diffusion_backend=os.environ.get(f"{prefix}DIFFUSION_BACKEND", defaults.diffusion_backend),
            diffusion_model=os.environ.get(f"{prefix}DIFFUSION_MODEL", defaults.diffusion_model),
            diffusion_api_key_path=_path_from_env(f"{prefix}DIFFUSION_API_KEY_PATH", defaults.diffusion_api_key_path),
            diffusion_default_size=os.environ.get(f"{prefix}DIFFUSION_DEFAULT_SIZE", defaults.diffusion_default_size),
            diffusion_timeout=_float_from_env(f"{prefix}DIFFUSION_TIMEOUT", defaults.diffusion_timeout),
        )

    def resolved_assets_dir(self) -> Path:
        """Return the directory used for binary assets and result payloads."""
        return self.assets_dir if self.assets_dir is not None else self.workspace_dir / "assets"

    def resolved_traces_dir(self) -> Path:
        """Return the directory used for kernel trace sessions."""
        return self.traces_dir if self.traces_dir is not None else self.workspace_dir / "traces"

    def ensure_directories(self) -> None:
        """Create service-owned directories using platform-neutral paths."""
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.resolved_assets_dir().mkdir(parents=True, exist_ok=True)
        self.resolved_traces_dir().mkdir(parents=True, exist_ok=True)

    def bind_url(self) -> str:
        """Return the local URL hosts should call."""
        return f"http://{self.host}:{self.port}"

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-safe summary for diagnostics."""
        return {
            "host": self.host,
            "port": self.port,
            "workspace_dir": str(self.workspace_dir),
            "assets_dir": str(self.resolved_assets_dir()),
            "traces_dir": str(self.resolved_traces_dir()),
            "max_request_bytes": self.max_request_bytes,
            "enable_cors": self.enable_cors,
            "allowed_origins": list(self.allowed_origins),
            "planner_backend": self.planner_backend,
            "planner_model": self.planner_model,
            "planner_endpoint": self.planner_endpoint,
            "planner_api_key_path": None if self.planner_api_key_path is None else str(self.planner_api_key_path),
            "planner_temperature": self.planner_temperature,
            "planner_max_tokens": self.planner_max_tokens,
            "planner_timeout": self.planner_timeout,
            "diffusion_backend": self.diffusion_backend,
            "diffusion_model": self.diffusion_model,
            "diffusion_api_key_path": None if self.diffusion_api_key_path is None else str(self.diffusion_api_key_path),
            "diffusion_default_size": self.diffusion_default_size,
            "diffusion_timeout": self.diffusion_timeout,
            "service_name": self.service_name,
            "service_version": self.service_version,
            "metadata": dict(self.metadata),
        }


def _int_from_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def _bool_from_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _float_from_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    return float(value)


def _list_from_env(name: str, default: list[str]) -> list[str]:
    value = os.environ.get(name)
    if value is None:
        return list(default)
    return [part.strip() for part in value.split(",") if part.strip()]


def _path_from_env(name: str, default: Optional[Path]) -> Optional[Path]:
    value = os.environ.get(name)
    if value is None:
        return default
    if value == "":
        return None
    return Path(value)


def _optional_path_from_env(name: str) -> Optional[Path]:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return Path(value)
