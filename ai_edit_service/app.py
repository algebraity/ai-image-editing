"""Minimal HTTP entrypoint for the platform-neutral AI edit service."""

from __future__ import annotations

import json
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from ai_edit_kernel.diffusion import VeniceImageBackend
from ai_edit_service.assets import AssetStore
from ai_edit_service.config import ServiceConfig
from ai_edit_service.jobs import JobStore
from ai_edit_service.kernel_runner import KernelRunner
from ai_edit_service.models import EditRequest, ServiceError, service_capabilities
from ai_edit_service.planner_backends import VenicePlannerBackend


@dataclass(slots=True)
class AiEditService:
    """Composition root for HTTP routes, jobs, assets, and kernel execution."""

    config: ServiceConfig
    assets: AssetStore
    runner: KernelRunner
    jobs: JobStore

    @classmethod
    def create(cls, config: ServiceConfig | None = None, runner: KernelRunner | None = None) -> "AiEditService":
        """Create the service with default filesystem-backed dependencies."""
        resolved_config = config or ServiceConfig.from_env()
        resolved_config.ensure_directories()
        assets = AssetStore(resolved_config.resolved_assets_dir())
        planner_backend = _planner_backend_from_config(resolved_config, assets)
        diffusion_backend = _diffusion_backend_from_config(resolved_config)
        resolved_runner = runner or KernelRunner(
            planner_backend=planner_backend,
            diffusion_backend=diffusion_backend,
            asset_store=assets,
            trace_root=resolved_config.resolved_traces_dir(),
        )
        jobs = JobStore(resolved_runner.run_edit)
        return cls(config=resolved_config, assets=assets, runner=resolved_runner, jobs=jobs)

    def health(self) -> dict[str, Any]:
        """Return service health and version metadata."""
        return {
            "ok": True,
            "service": self.config.service_name,
            "version": self.config.service_version,
            "url": self.config.bind_url(),
        }


class ServiceHTTPServer(ThreadingHTTPServer):
    """HTTP server carrying the composed service object."""

    service: AiEditService

    def __init__(self, server_address: tuple[str, int], handler_class: type[BaseHTTPRequestHandler], service: AiEditService) -> None:
        super().__init__(server_address, handler_class)
        self.service = service


class AiEditRequestHandler(BaseHTTPRequestHandler):
    """HTTP JSON API for editor adapters."""

    server: ServiceHTTPServer

    def do_OPTIONS(self) -> None:
        self._send_json(200, {"ok": True})

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        try:
            if path == "/health":
                self._send_json(200, self.server.service.health())
            elif path == "/v1/capabilities":
                self._send_json(200, service_capabilities())
            elif path == "/v1/jobs":
                self._send_json(200, {"jobs": [job.to_json() for job in self.server.service.jobs.list_recent()]})
            elif path.startswith("/v1/jobs/"):
                self._handle_job_get(path)
            elif path.startswith("/v1/assets/"):
                self._handle_asset_get(path)
            else:
                self._send_error(404, "not_found", f"unknown route {path!r}")
        except Exception as exc:
            self._send_exception(exc)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = parse_qs(parsed.query)
        try:
            if path == "/v1/edits":
                request = EditRequest.from_json(self._read_json_body())
                job = self.server.service.jobs.submit(request)
                status = 202
                body: dict[str, Any] = {"job": job.to_json(), "result_url": f"/v1/jobs/{job.id}/result"}
                if query.get("wait") == ["true"]:
                    waited_job = self.server.service.jobs.wait(job.id)
                    body = {"job": waited_job.to_json(include_result=True)}
                    status = 200
                self._send_json(status, body)
            else:
                self._send_error(404, "not_found", f"unknown route {path!r}")
        except Exception as exc:
            self._send_exception(exc)

    def _handle_job_get(self, path: str) -> None:
        parts = path.split("/")
        if len(parts) == 4:
            job = self.server.service.jobs.get(parts[3])
            self._send_json(200, {"job": job.to_json()})
            return
        if len(parts) == 5 and parts[4] == "result":
            job = self.server.service.jobs.get(parts[3])
            if job.result is None:
                self._send_json(202, {"job": job.to_json()})
                return
            self._send_json(200, job.result.to_json())
            return
        self._send_error(404, "not_found", f"unknown job route {path!r}")

    def _handle_asset_get(self, path: str) -> None:
        asset_id = path.split("/")[-1]
        record = self.server.service.assets.get_record(asset_id)
        data = record.path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", record.media_type)
        self.send_header("Content-Length", str(len(data)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length > self.server.service.config.max_request_bytes:
            raise ValueError("request body exceeds max_request_bytes")
        raw = self.rfile.read(content_length)
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise TypeError("request body must be a JSON object")
        return data

    def _send_json(self, status: int, data: dict[str, Any]) -> None:
        encoded = json.dumps(data, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(encoded)

    def _send_error(self, status: int, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        self._send_json(status, {"error": ServiceError(code, message, details or {}).to_json()})

    def _send_exception(self, exc: Exception) -> None:
        status = 404 if isinstance(exc, (KeyError, FileNotFoundError)) else 400
        self._send_error(status, exc.__class__.__name__, str(exc))

    def _send_cors_headers(self) -> None:
        config = self.server.service.config
        if not config.enable_cors:
            return
        origin = self.headers.get("Origin")
        allowed = "*" if "*" in config.allowed_origins else (origin if origin in config.allowed_origins else config.allowed_origins[0])
        self.send_header("Access-Control-Allow-Origin", allowed)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format: str, *args: Any) -> None:
        """Keep the default server quiet for plugin-driven local calls."""
        return


def run_server(service: AiEditService | None = None) -> None:
    """Run the HTTP service until interrupted."""
    resolved_service = service or AiEditService.create()
    address = (resolved_service.config.host, resolved_service.config.port)
    httpd = ServiceHTTPServer(address, AiEditRequestHandler, resolved_service)
    print(f"ai-edit-service listening on {resolved_service.config.bind_url()}")
    httpd.serve_forever()


def _planner_backend_from_config(config: ServiceConfig, assets: AssetStore):
    if config.planner_backend == "none":
        return None
    if config.planner_backend == "venice":
        return VenicePlannerBackend(
            api_key_path=config.planner_api_key_path,
            model=config.planner_model,
            endpoint=config.planner_endpoint,
            temperature=config.planner_temperature,
            max_tokens=config.planner_max_tokens,
            timeout=config.planner_timeout,
            asset_store=assets,
        )
    raise ValueError(f"unsupported planner_backend {config.planner_backend!r}")


def _diffusion_backend_from_config(config: ServiceConfig):
    if config.diffusion_backend == "none":
        return None
    if config.diffusion_backend == "venice":
        return VeniceImageBackend(
            api_key_path=config.diffusion_api_key_path,
            model=config.diffusion_model,
            default_size=config.diffusion_default_size,
            timeout=config.diffusion_timeout,
        )
    raise ValueError(f"unsupported diffusion_backend {config.diffusion_backend!r}")


if __name__ == "__main__":
    run_server()
