"""Planner model backends for the AI edit service."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from base64 import b64encode
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ai_edit_kernel.diffusion.adapters.venice import DEFAULT_VENICE_KEY_PATH
from ai_edit_service.assets import AssetStore


DEFAULT_VENICE_CHAT_ENDPOINT = "https://api.venice.ai/api/v1/chat/completions"
DEFAULT_VENICE_PLANNER_MODEL = "openai-gpt-54"
DEFAULT_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parents[1] / "ai_edit_kernel" / "system_prompt.md"

UrlOpener = Callable[[urllib.request.Request, float], bytes]


@dataclass(slots=True)
class VenicePlannerBackend:
    """Use Venice's chat-completions API as a planner-only model backend."""

    api_key: Optional[str] = None
    api_key_path: str | Path | None = DEFAULT_VENICE_KEY_PATH
    model: str = DEFAULT_VENICE_PLANNER_MODEL
    endpoint: str = DEFAULT_VENICE_CHAT_ENDPOINT
    system_prompt_path: str | Path = DEFAULT_SYSTEM_PROMPT_PATH
    temperature: float = 0.1
    max_tokens: Optional[int] = 16384
    timeout: float = 360.0
    asset_store: Optional[AssetStore] = None
    opener: Optional[UrlOpener] = None
    extra_body: dict[str, Any] = field(default_factory=dict)

    def plan(self, request: dict[str, Any]) -> dict[str, Any] | str:
        """Return planner output JSON from the configured Venice chat model."""
        started = time.monotonic()
        payload = self._payload(request)
        http_request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            response_data = self._open(http_request)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Venice planner API HTTP {exc.code}: {_safe_error_body(exc)}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Venice planner API request failed: {exc.reason}") from exc

        response = json.loads(response_data.decode("utf-8"))
        content = _message_content(response)
        output = _extract_json_object(content)
        if isinstance(output, dict):
            output.setdefault("metadata", {})
            output["metadata"].setdefault(
                "planner_backend",
                {
                    "provider": "venice",
                    "model": self.model,
                    "latency_seconds": time.monotonic() - started,
                },
            )
        return output

    def _payload(self, request: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": self._user_content(request)},
            ],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        body.update(self.extra_body)
        _reject_secret_options(body)
        return body

    def _system_prompt(self) -> str:
        path = Path(self.system_prompt_path)
        return path.read_text(encoding="utf-8")

    def _user_content(self, request: dict[str, Any]) -> str | list[dict[str, Any]]:
        message = _planner_request_message(_request_text_payload(request))
        data_url = self._current_image_data_url(request)
        if data_url is None:
            return message
        return [
            {"type": "text", "text": message},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]

    def _current_image_data_url(self, request: dict[str, Any]) -> Optional[str]:
        asset_refs = request.get("asset_refs")
        if not isinstance(asset_refs, dict):
            return None
        ref = asset_refs.get("current_image")
        if not isinstance(ref, str) or not ref.startswith("asset:"):
            return None
        if self.asset_store is None:
            raise RuntimeError("current_image asset ref requires an AssetStore")
        asset_id = ref.split(":", 1)[1]
        record = self.asset_store.get_record(asset_id)
        data = self.asset_store.get_bytes(asset_id)
        media_type = record.media_type or "image/png"
        return f"data:{media_type};base64,{b64encode(data).decode('ascii')}"

    def _api_key(self) -> str:
        if self.api_key is not None and self.api_key.strip():
            return self.api_key.strip()

        env_key = os.environ.get("VENICE_API_KEY")
        if env_key is not None and env_key.strip():
            return env_key.strip()

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


def _planner_request_message(request: dict[str, Any]) -> str:
    return (
        "Return exactly one planner_output JSON object for this planner request. "
        "The current image is attached to this message as vision input when "
        "asset_refs.current_image is marked attached_vision_input. "
        "Do not include markdown, prose, code fences, or hidden reasoning.\n\n"
        f"{json.dumps(request, indent=2, sort_keys=True)}"
    )


def _request_text_payload(request: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(request))
    asset_refs = payload.get("asset_refs")
    if isinstance(asset_refs, dict) and isinstance(asset_refs.get("current_image"), str):
        asset_refs["current_image"] = "attached_to_this_message"
    return payload


def _message_content(response: Any) -> str:
    if not isinstance(response, dict):
        raise ValueError("Venice planner response must be a JSON object")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Venice planner response did not include choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("Venice planner response choice must be an object")
    message = first.get("message")
    if not isinstance(message, dict):
        raise ValueError("Venice planner response choice did not include message")
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        joined = "\n".join(part for part in parts if part.strip()).strip()
        if joined:
            return joined
    raise ValueError("Venice planner response message content was empty")


def _extract_json_object(content: str) -> dict[str, Any] | str:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        fenced = _strip_json_fence(content)
        if fenced != content:
            try:
                parsed = json.loads(fenced)
            except json.JSONDecodeError:
                parsed = _json_object_from_text(content)
        else:
            parsed = _json_object_from_text(content)
    if isinstance(parsed, dict):
        return parsed
    return content


def _strip_json_fence(content: str) -> str:
    value = content.strip()
    if not value.startswith("```") or not value.endswith("```"):
        return content
    lines = value.splitlines()
    if len(lines) < 3:
        return content
    return "\n".join(lines[1:-1]).strip()


def _json_object_from_text(content: str) -> dict[str, Any] | str:
    start = content.find("{")
    end = content.rfind("}")
    if start < 0 or end <= start:
        return content
    candidate = content[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return content
    return parsed if isinstance(parsed, dict) else content


def _safe_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", "replace")
    except Exception:
        return ""
    body = " ".join(body.split())
    if len(body) > 500:
        return f"{body[:500]}..."
    return body


def _reject_secret_options(payload: dict[str, Any]) -> None:
    forbidden = [key for key in payload if _looks_like_secret_option(key)]
    if forbidden:
        raise ValueError(f"Venice planner payload contains secret-like option(s): {', '.join(sorted(forbidden))}")


def _looks_like_secret_option(key: str) -> bool:
    value = key.lower()
    return (
        value in {"api_key", "apikey", "authorization", "token", "access_token", "bearer_token"}
        or value.endswith("_api_key")
        or value.endswith("_apikey")
        or value.endswith("_token")
        or value.startswith("authorization_")
    )
