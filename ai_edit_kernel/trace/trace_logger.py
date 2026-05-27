"""Trace logging for the AI Editing Kernel.

A trace is a replayable record of an editing session. The logger writes one
directory per session containing `manifest.json`, ordered `events.jsonl`, and
optional asset files such as document snapshots and flattened previews.

The trace format is intentionally useful in two directions: humans can inspect a
session after a prototype run, and dataset tooling can derive planner imitation
examples from the same event stream.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import numpy as np

from ai_edit_kernel.document.document_state import DocumentState
from ai_edit_kernel.runtime.validator import ValidationReport
from ai_edit_kernel.schema.actions import Action, ActionBatch, ActionResult, SCHEMA_VERSION as ACTION_SCHEMA_VERSION


TRACE_SCHEMA_VERSION = "ai_edit_trace.v1"
DOCUMENT_SCHEMA_VERSION = "ai_edit_document.v1"
TRAINING_SCHEMA_VERSION = "ai_edit_training_example.v1"
TOOL_CATALOG_VERSION = "tools.v1"
KERNEL_VERSION = "0.1.0"


class TraceEventType(str, Enum):
    """Kinds of events saved into a trace."""

    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    USER_PROMPT = "user_prompt"
    DOCUMENT_SNAPSHOT = "document_snapshot"
    OBSERVATION = "observation"
    PLANNER_INPUT = "planner_input"
    PLANNER_OUTPUT_RAW = "planner_output_raw"
    ACTION_PLANNED = "action_planned"
    ACTION_BATCH_PLANNED = "action_batch_planned"
    ACTION_STARTED = "action_started"
    ACTION_RESULT = "action_result"
    VALIDATION_REPORT = "validation_report"
    DIFFUSION_JOB_STARTED = "diffusion_job_started"
    DIFFUSION_JOB_RESULT = "diffusion_job_result"
    HUMAN_FEEDBACK = "human_feedback"
    ERROR = "error"
    NOTE = "note"


@dataclass(slots=True)
class TraceEvent:
    """One timestamped event in an editing session trace."""

    id: str
    type: TraceEventType
    timestamp: str
    document_id: Optional[str] = None
    document_revision_before: Optional[int] = None
    document_revision_after: Optional[int] = None
    action_id: Optional[str] = None
    payload: dict[str, Any] = field(default_factory=dict)
    asset_refs: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Serialize this event into the trace event envelope."""
        return {
            "event_id": self.id,
            "type": TraceEventType(self.type).value,
            "timestamp": self.timestamp,
            "document_id": self.document_id,
            "document_revision_before": self.document_revision_before,
            "document_revision_after": self.document_revision_after,
            "action_id": self.action_id,
            "payload": _json_safe(self.payload),
            "asset_refs": _json_safe(self.asset_refs),
            "metadata": _json_safe(self.metadata),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "TraceEvent":
        """Deserialize a trace event from JSON-compatible data."""
        if not isinstance(data, dict):
            raise TypeError("trace event data must be an object")
        event_id = data.get("event_id", data.get("id"))
        if not isinstance(event_id, str) or event_id == "":
            raise ValueError("trace event requires non-empty event_id")
        return cls(
            id=event_id,
            type=TraceEventType(data["type"]),
            timestamp=_required_string(data, "timestamp"),
            document_id=_optional_string(data.get("document_id"), "document_id"),
            document_revision_before=_optional_int(data.get("document_revision_before"), "document_revision_before"),
            document_revision_after=_optional_int(data.get("document_revision_after"), "document_revision_after"),
            action_id=_optional_string(data.get("action_id"), "action_id"),
            payload=_mapping_value(data.get("payload", {}), "payload"),
            asset_refs=_string_mapping(data.get("asset_refs", {}), "asset_refs"),
            metadata=_mapping_value(data.get("metadata", {}), "metadata"),
        )


@dataclass(slots=True)
class TraceSession:
    """A complete editing session consisting of many trace events."""

    id: str
    started_at: str
    ended_at: Optional[str] = None
    user_prompt: Optional[str] = None
    events: list[TraceEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_event(self, event: TraceEvent) -> None:
        """Append an event to this session."""
        if not isinstance(event, TraceEvent):
            raise TypeError("event must be a TraceEvent")
        self.events.append(event)

    def to_json(self) -> dict[str, Any]:
        """Serialize the full session, including events."""
        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            "session_id": self.id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "user_prompt": self.user_prompt,
            "events": [event.to_json() for event in self.events],
            "metadata": _json_safe(self.metadata),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "TraceSession":
        """Deserialize a session from JSON-compatible data."""
        if not isinstance(data, dict):
            raise TypeError("trace session data must be an object")
        session = cls(
            id=_required_string(data, "session_id"),
            started_at=_required_string(data, "started_at"),
            ended_at=_optional_string(data.get("ended_at"), "ended_at"),
            user_prompt=_optional_string(data.get("user_prompt"), "user_prompt"),
            events=[TraceEvent.from_json(event) for event in data.get("events", [])],
            metadata=_mapping_value(data.get("metadata", {}), "metadata"),
        )
        return session


@dataclass(slots=True)
class TraceLogger:
    """Record editing sessions, actions, results, validation, and snapshots."""

    output_dir: Path
    session: Optional[TraceSession] = None
    save_snapshots: bool = True
    save_preview_images: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def start_session(self, user_prompt: Optional[str] = None, metadata: Optional[dict[str, Any]] = None) -> TraceSession:
        """Create and activate a new trace session."""
        if user_prompt is not None and not isinstance(user_prompt, str):
            raise TypeError("user_prompt must be a string or None")
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary or None")

        session_id = _new_session_id()
        started_at = _utc_now()
        session_metadata = {
            **self.metadata,
            **({} if metadata is None else metadata),
            "status": "running",
            "success": None,
        }
        self.session = TraceSession(
            id=session_id,
            started_at=started_at,
            user_prompt=user_prompt,
            metadata=session_metadata,
        )
        self._ensure_session_dirs()
        self._reset_events_file()
        self._append_event(
            TraceEvent(
                id=self._next_event_id(),
                type=TraceEventType.SESSION_STARTED,
                timestamp=started_at,
                payload={"session_id": session_id, "schema_version": TRACE_SCHEMA_VERSION},
            )
        )
        if user_prompt is not None:
            self.log_user_prompt(user_prompt)
        self._write_manifest()
        return self.session

    def end_session(self) -> TraceSession:
        """End the active session and emit a `session_ended` event."""
        session = self._require_session()
        ended_at = _utc_now()
        session.ended_at = ended_at
        session.metadata["status"] = session.metadata.get("status") if session.metadata.get("status") != "running" else "completed"
        if session.metadata.get("success") is None:
            session.metadata["success"] = not _has_failed_action_or_error(session)
        last_doc_id, last_revision = _last_document_state(session.events)
        summary = self._summary()
        self._append_event(
            TraceEvent(
                id=self._next_event_id(),
                type=TraceEventType.SESSION_ENDED,
                timestamp=ended_at,
                document_id=last_doc_id,
                document_revision_before=last_revision,
                document_revision_after=last_revision,
                payload={
                    "session_id": session.id,
                    "status": session.metadata.get("status", "completed"),
                    "success": session.metadata.get("success"),
                    "summary": summary,
                },
            )
        )
        self._write_manifest()
        return session

    def log_user_prompt(self, prompt: str, document: Optional[DocumentState] = None) -> None:
        """Record the user's natural-language request."""
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        session = self._require_session()
        session.user_prompt = prompt
        self._append_event(
            TraceEvent(
                id=self._next_event_id(),
                type=TraceEventType.USER_PROMPT,
                timestamp=_utc_now(),
                document_id=document.id if document is not None else None,
                document_revision_after=document.revision if document is not None else None,
                payload={"prompt": prompt},
            )
        )
        self._write_manifest()

    def log_document_snapshot(self, document: DocumentState, label: str) -> None:
        """Record a JSON document summary and optional flattened preview."""
        self._require_session()
        if not isinstance(document, DocumentState):
            raise TypeError("document must be a DocumentState")
        if not isinstance(label, str):
            raise TypeError("label must be a string")

        summary = _document_summary(document)
        asset_refs: dict[str, str] = {}
        if self.save_snapshots:
            snapshot_rel = f"snapshots/doc_rev_{document.revision:04d}_{_safe_label(label)}.json"
            snapshot_path = self._session_dir() / snapshot_rel
            _write_json(snapshot_path, summary)
            asset_refs["snapshot"] = snapshot_rel
        if self.save_preview_images:
            preview_ref = self._save_preview(document, label)
            if preview_ref is not None:
                asset_refs["preview"] = preview_ref

        self._append_event(
            TraceEvent(
                id=self._next_event_id(),
                type=TraceEventType.DOCUMENT_SNAPSHOT,
                timestamp=_utc_now(),
                document_id=document.id,
                document_revision_before=document.revision,
                document_revision_after=document.revision,
                payload={"label": label, "document_summary": summary},
                asset_refs=asset_refs,
            )
        )
        self._write_manifest()

    def log_observation(self, document: DocumentState, observation: dict[str, Any], label: str) -> None:
        """Record perception or analyzer output."""
        self._require_session()
        if not isinstance(observation, dict):
            raise TypeError("observation must be a dictionary")
        if not isinstance(label, str):
            raise TypeError("label must be a string")
        self._append_event(
            TraceEvent(
                id=self._next_event_id(),
                type=TraceEventType.OBSERVATION,
                timestamp=_utc_now(),
                document_id=document.id,
                document_revision_before=document.revision,
                document_revision_after=document.revision,
                payload={"label": label, "observations": observation.get("observations", observation)},
            )
        )

    def log_action_batch_planned(self, batch: ActionBatch, document: Optional[DocumentState] = None) -> None:
        """Record a planned batch before execution."""
        self._require_session()
        batch.validate_schema()
        revision = document.revision if document is not None else None
        self._append_event(
            TraceEvent(
                id=self._next_event_id(),
                type=TraceEventType.ACTION_BATCH_PLANNED,
                timestamp=_utc_now(),
                document_id=document.id if document is not None else None,
                document_revision_before=revision,
                document_revision_after=revision,
                payload={"action_batch": batch.to_json()},
            )
        )
        self._write_manifest()

    def log_action_planned(self, action: Action, document: Optional[DocumentState] = None) -> None:
        """Record one planned action before execution."""
        self._require_session()
        action.validate_schema()
        revision = document.revision if document is not None else None
        self._append_event(
            TraceEvent(
                id=self._next_event_id(),
                type=TraceEventType.ACTION_PLANNED,
                timestamp=_utc_now(),
                document_id=document.id if document is not None else None,
                document_revision_before=revision,
                document_revision_after=revision,
                action_id=action.id,
                payload={"action": action.to_json()},
            )
        )

    def log_action_started(self, action: Action, document: DocumentState) -> None:
        """Record that the executor started an action."""
        self._require_session()
        self._append_event(
            TraceEvent(
                id=self._next_event_id(),
                type=TraceEventType.ACTION_STARTED,
                timestamp=_utc_now(),
                document_id=document.id,
                document_revision_before=document.revision,
                document_revision_after=document.revision,
                action_id=action.id,
                payload={"action": action.to_json()},
            )
        )

    def log_action_result(self, action: Action, result: ActionResult, document: DocumentState) -> None:
        """Record action result, including created layers/masks and errors."""
        self._require_session()
        self._append_event(
            TraceEvent(
                id=self._next_event_id(),
                type=TraceEventType.ACTION_RESULT,
                timestamp=_utc_now(),
                document_id=document.id,
                document_revision_before=result.before_revision,
                document_revision_after=result.after_revision,
                action_id=action.id,
                payload={"result": result.to_json()},
                asset_refs={key: str(value) for key, value in result.output_assets.items() if isinstance(value, str)},
            )
        )
        self._write_manifest()

    def log_validation_report(
        self,
        report: ValidationReport,
        document: Optional[DocumentState] = None,
        action: Optional[Action] = None,
    ) -> None:
        """Record validation output for an action or document."""
        self._require_session()
        revision = document.revision if document is not None else None
        self._append_event(
            TraceEvent(
                id=self._next_event_id(),
                type=TraceEventType.VALIDATION_REPORT,
                timestamp=_utc_now(),
                document_id=document.id if document is not None else None,
                document_revision_before=revision,
                document_revision_after=revision,
                action_id=action.id if action is not None else None,
                payload={"report": report.to_json()},
            )
        )
        self._write_manifest()

    def log_error(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        """Record an unexpected runtime or planning error."""
        if not isinstance(message, str):
            raise TypeError("message must be a string")
        if details is not None and not isinstance(details, dict):
            raise TypeError("details must be a dictionary or None")
        session = self._require_session()
        session.metadata["status"] = "failed"
        session.metadata["success"] = False
        last_doc_id, last_revision = _last_document_state(session.events)
        self._append_event(
            TraceEvent(
                id=self._next_event_id(),
                type=TraceEventType.ERROR,
                timestamp=_utc_now(),
                document_id=last_doc_id,
                document_revision_before=last_revision,
                document_revision_after=last_revision,
                payload={"code": "trace.error", "message": message, "details": {} if details is None else details},
            )
        )
        self._write_manifest()

    def log_note(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        """Record a human-readable note for debugging or review."""
        if not isinstance(message, str):
            raise TypeError("message must be a string")
        if details is not None and not isinstance(details, dict):
            raise TypeError("details must be a dictionary or None")
        session = self._require_session()
        last_doc_id, last_revision = _last_document_state(session.events)
        self._append_event(
            TraceEvent(
                id=self._next_event_id(),
                type=TraceEventType.NOTE,
                timestamp=_utc_now(),
                document_id=last_doc_id,
                document_revision_before=last_revision,
                document_revision_after=last_revision,
                payload={"message": message, "details": {} if details is None else details},
            )
        )

    def save_session(self, path: Optional[Path] = None) -> Path:
        """Write the active session to disk and return the session directory."""
        self._require_session()
        if path is not None:
            target = Path(path)
            if target != self._session_dir():
                raise ValueError("TraceLogger writes sessions to output_dir/session_id; use that directory as path")
        self._rewrite_events_file()
        self._write_manifest()
        return self._session_dir()

    def load_session(self, path: Path) -> TraceSession:
        """Load a trace session from a directory or manifest path."""
        base = Path(path)
        if base.name == "manifest.json":
            session_dir = base.parent
            manifest_path = base
        else:
            session_dir = base
            manifest_path = session_dir / "manifest.json"
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        events: list[TraceEvent] = []
        events_path = session_dir / "events.jsonl"
        if events_path.exists():
            with events_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if stripped:
                        events.append(TraceEvent.from_json(json.loads(stripped)))

        session = TraceSession(
            id=_required_string(manifest, "session_id"),
            started_at=_required_string(manifest, "created_at"),
            ended_at=_optional_string(manifest.get("ended_at"), "ended_at"),
            user_prompt=_optional_string(manifest.get("user_prompt"), "user_prompt"),
            events=events,
            metadata={
                "manifest": manifest,
                "status": manifest.get("status"),
                "success": manifest.get("success"),
                "session_dir": str(session_dir),
            },
        )
        self.output_dir = session_dir.parent
        self.session = session
        return session

    def export_training_example(self, session: TraceSession) -> dict[str, Any]:
        """Convert a trace session into a planner-imitation training example."""
        if not isinstance(session, TraceSession):
            raise TypeError("session must be a TraceSession")
        prompt = _last_payload_value(session.events, TraceEventType.USER_PROMPT, "prompt", session.user_prompt)
        document_summary = _last_document_summary(session.events)
        observations = _collect_observations(session.events)
        action_batch = _last_action_batch(session.events)
        human_feedback = _last_payload(session.events, TraceEventType.HUMAN_FEEDBACK)
        validation_passed = _all_validation_passed(session.events)
        success = bool(session.metadata.get("success", validation_passed))

        return {
            "schema_version": TRAINING_SCHEMA_VERSION,
            "example_id": f"ex_{session.id}_plan_001",
            "source_session_id": session.id,
            "example_type": "planner_imitation",
            "task_type": session.metadata.get("task_type"),
            "split": session.metadata.get("split", "train"),
            "input": {
                "user_prompt": prompt,
                "document_summary": document_summary,
                "observations": observations,
                "available_tools": _available_tools(),
                "asset_refs": _last_snapshot_asset_refs(session.events),
            },
            "target": {
                "action_batch": action_batch,
            },
            "labels": {
                "success": success,
                "validation_passed": validation_passed,
                "human_accepted": human_feedback.get("accepted") if human_feedback else None,
                "human_rating": human_feedback.get("rating") if human_feedback else None,
                "metrics": _collect_validation_metrics(session.events),
            },
            "provenance": {
                "trace_path": str(self._session_dir_for(session)),
                "source": session.metadata.get("source", "trace"),
                "allowed_for_training": _rights(session).get("allowed_for_training", False),
            },
        }

    def export_dataset(self, sessions: list[TraceSession], output_path: Path) -> Path:
        """Export many sessions into a JSONL dataset file."""
        if not isinstance(sessions, list):
            raise TypeError("sessions must be a list")
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for session in sessions:
                handle.write(json.dumps(self.export_training_example(session), sort_keys=True))
                handle.write("\n")
        return path

    def _append_event(self, event: TraceEvent) -> None:
        """Append an event to memory and to `events.jsonl`."""
        session = self._require_session()
        session.add_event(event)
        events_path = self._events_path()
        events_path.parent.mkdir(parents=True, exist_ok=True)
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_json(), sort_keys=True))
            handle.write("\n")

    def _require_session(self) -> TraceSession:
        """Return the active session or raise a clear error."""
        if self.session is None:
            raise RuntimeError("no active trace session")
        return self.session

    def _session_dir(self) -> Path:
        """Return the active session directory."""
        return self.output_dir / self._require_session().id

    def _session_dir_for(self, session: TraceSession) -> Path:
        """Return a best-effort session directory for a session object."""
        session_dir = session.metadata.get("session_dir")
        return Path(session_dir) if isinstance(session_dir, str) else self.output_dir / session.id

    def _events_path(self) -> Path:
        """Return the active events JSONL path."""
        return self._session_dir() / "events.jsonl"

    def _next_event_id(self) -> str:
        """Return the next sequential event ID for the active session."""
        return f"evt_{len(self._require_session().events) + 1:06d}"

    def _ensure_session_dirs(self) -> None:
        """Create the directory layout required by the trace spec."""
        base = self._session_dir()
        for relative in ("previews", "layers", "masks", "snapshots", "diffusion", "artifacts"):
            (base / relative).mkdir(parents=True, exist_ok=True)
        self._require_session().metadata["session_dir"] = str(base)

    def _reset_events_file(self) -> None:
        """Create an empty events file for a new session."""
        self._events_path().parent.mkdir(parents=True, exist_ok=True)
        self._events_path().write_text("", encoding="utf-8")

    def _rewrite_events_file(self) -> None:
        """Rewrite `events.jsonl` from the in-memory session."""
        with self._events_path().open("w", encoding="utf-8") as handle:
            for event in self._require_session().events:
                handle.write(json.dumps(event.to_json(), sort_keys=True))
                handle.write("\n")

    def _write_manifest(self) -> None:
        """Write `manifest.json` for the active session."""
        _write_json(self._session_dir() / "manifest.json", self._manifest())

    def _manifest(self) -> dict[str, Any]:
        """Build a manifest from session metadata and cached event summaries."""
        session = self._require_session()
        manifest_metadata = session.metadata.get("manifest") if isinstance(session.metadata.get("manifest"), dict) else {}
        rights = _rights(session)
        status = session.metadata.get("status", manifest_metadata.get("status", "running"))
        success = session.metadata.get("success", manifest_metadata.get("success"))
        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            "session_id": session.id,
            "created_at": session.started_at,
            "ended_at": session.ended_at,
            "user_prompt": session.user_prompt,
            "task_type": session.metadata.get("task_type", manifest_metadata.get("task_type")),
            "status": status,
            "success": success,
            "kernel": {
                "kernel_version": session.metadata.get("kernel_version", KERNEL_VERSION),
                "action_schema_version": ACTION_SCHEMA_VERSION,
                "document_schema_version": DOCUMENT_SCHEMA_VERSION,
                "tool_catalog_version": TOOL_CATALOG_VERSION,
            },
            "environment": session.metadata.get("environment", _environment()),
            "planner": session.metadata.get(
                "planner",
                {"planner_type": "manual", "planner_model": None, "planner_version": None, "temperature": None},
            ),
            "assets": {
                "root": ".",
                "previews_dir": "previews",
                "layers_dir": "layers",
                "masks_dir": "masks",
                "snapshots_dir": "snapshots",
                "diffusion_dir": "diffusion",
                "artifacts_dir": "artifacts",
            },
            "summary": self._summary(),
            "rights": rights,
            "metadata": _public_metadata(session.metadata),
        }

    def _summary(self) -> dict[str, Any]:
        """Return cached session summary fields for the manifest."""
        session = self._require_session()
        snapshots = [event for event in session.events if event.type == TraceEventType.DOCUMENT_SNAPSHOT]
        action_results = [event for event in session.events if event.type == TraceEventType.ACTION_RESULT]
        validation_reports = [event for event in session.events if event.type == TraceEventType.VALIDATION_REPORT]
        human_feedback = _last_payload(session.events, TraceEventType.HUMAN_FEEDBACK)
        initial_snapshot, initial_preview = _snapshot_refs(snapshots[0]) if snapshots else (None, None)
        final_snapshot, final_preview = _snapshot_refs(snapshots[-1]) if snapshots else (None, None)
        return {
            "initial_snapshot": initial_snapshot,
            "final_snapshot": final_snapshot,
            "initial_preview": initial_preview,
            "final_preview": final_preview,
            "action_count": len(action_results),
            "validation_passed": _reports_passed(validation_reports),
            "human_rating": human_feedback.get("rating") if human_feedback else None,
        }

    def _save_preview(self, document: DocumentState, label: str) -> Optional[str]:
        """Save a flattened preview as PNG when Pillow is available."""
        preview_rel = f"previews/doc_rev_{document.revision:04d}_{_safe_label(label)}.png"
        preview_path = self._session_dir() / preview_rel
        try:
            from PIL import Image
        except ImportError:
            npy_rel = f"previews/doc_rev_{document.revision:04d}_{_safe_label(label)}.npy"
            np.save(self._session_dir() / npy_rel, document.flatten_preview())
            return npy_rel
        image = Image.fromarray(np.clip(document.flatten_preview() * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGBA")
        image.save(preview_path)
        return preview_rel


def _new_session_id() -> str:
    """Return a stable, sortable session ID."""
    stamp = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H%M%S")
    return f"session_{stamp}_{uuid4().hex[:8]}"


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp with millisecond precision."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Write deterministic pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(data), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _document_summary(document: DocumentState) -> dict[str, Any]:
    """Return a document summary with an explicit schema version."""
    summary = document.snapshot_summary()
    summary["schema_version"] = DOCUMENT_SCHEMA_VERSION
    return summary


def _environment() -> dict[str, Any]:
    """Return minimal environment metadata."""
    return {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": platform.system().lower() or "local",
        "deterministic_mode": False,
        "random_seed": None,
    }


def _rights(session: TraceSession) -> dict[str, Any]:
    """Return rights metadata with conservative defaults."""
    rights = session.metadata.get("rights")
    if isinstance(rights, dict):
        return {
            "source_image_license": rights.get("source_image_license", "unknown"),
            "user_provided_content": bool(rights.get("user_provided_content", False)),
            "allowed_for_training": bool(rights.get("allowed_for_training", False)),
            "contains_personal_data": bool(rights.get("contains_personal_data", False)),
        }
    return {
        "source_image_license": "unknown",
        "user_provided_content": False,
        "allowed_for_training": False,
        "contains_personal_data": False,
    }


def _public_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return metadata safe for manifest top-level metadata."""
    hidden = {"manifest", "session_dir", "status", "success", "rights", "planner", "environment"}
    return {key: _json_safe(value) for key, value in metadata.items() if key not in hidden}


def _last_document_state(events: list[TraceEvent]) -> tuple[Optional[str], Optional[int]]:
    """Return the latest document ID and revision observed in events."""
    for event in reversed(events):
        if event.document_id is not None or event.document_revision_after is not None:
            return event.document_id, event.document_revision_after
    return None, None


def _snapshot_refs(event: TraceEvent) -> tuple[Optional[str], Optional[str]]:
    """Return snapshot and preview asset references from a snapshot event."""
    return event.asset_refs.get("snapshot"), event.asset_refs.get("preview")


def _reports_passed(events: list[TraceEvent]) -> bool:
    """Return whether all validation-report events passed."""
    if not events:
        return True
    return all(bool(event.payload.get("report", {}).get("passed", False)) for event in events)


def _all_validation_passed(events: list[TraceEvent]) -> bool:
    """Return whether validation reports passed and action results succeeded."""
    validation_events = [event for event in events if event.type == TraceEventType.VALIDATION_REPORT]
    action_events = [event for event in events if event.type == TraceEventType.ACTION_RESULT]
    reports_ok = _reports_passed(validation_events)
    actions_ok = all(event.payload.get("result", {}).get("status") in {"executed", "validated"} for event in action_events)
    return reports_ok and actions_ok


def _has_failed_action_or_error(session: TraceSession) -> bool:
    """Return whether a session contains failed action or error events."""
    for event in session.events:
        if event.type == TraceEventType.ERROR:
            return True
        if event.type == TraceEventType.ACTION_RESULT:
            if event.payload.get("result", {}).get("status") not in {"executed", "validated"}:
                return True
    return False


def _last_payload(events: list[TraceEvent], event_type: TraceEventType) -> dict[str, Any]:
    """Return the last payload for an event type."""
    for event in reversed(events):
        if event.type == event_type:
            return event.payload
    return {}


def _last_payload_value(events: list[TraceEvent], event_type: TraceEventType, key: str, default: Any) -> Any:
    """Return a key from the last payload for an event type."""
    payload = _last_payload(events, event_type)
    return payload.get(key, default)


def _last_document_summary(events: list[TraceEvent]) -> dict[str, Any]:
    """Return the last document summary before planning when available."""
    for event in reversed(events):
        if event.type == TraceEventType.DOCUMENT_SNAPSHOT:
            summary = event.payload.get("document_summary")
            if isinstance(summary, dict):
                return summary
    return {}


def _collect_observations(events: list[TraceEvent]) -> list[Any]:
    """Collect observation payloads for training export."""
    observations: list[Any] = []
    for event in events:
        if event.type == TraceEventType.OBSERVATION:
            value = event.payload.get("observations")
            if isinstance(value, list):
                observations.extend(value)
            elif value is not None:
                observations.append(value)
    return observations


def _last_action_batch(events: list[TraceEvent]) -> dict[str, Any]:
    """Return the last planned action batch, or synthesize one from action_started events."""
    for event in reversed(events):
        if event.type == TraceEventType.ACTION_BATCH_PLANNED:
            batch = event.payload.get("action_batch")
            if isinstance(batch, dict):
                return batch
    actions = []
    for event in events:
        if event.type == TraceEventType.ACTION_STARTED:
            action = event.payload.get("action")
            if isinstance(action, dict):
                actions.append(action)
    return {"schema_version": ACTION_SCHEMA_VERSION, "id": "batch_from_trace", "stop_on_error": True, "actions": actions}


def _last_snapshot_asset_refs(events: list[TraceEvent]) -> dict[str, str]:
    """Return asset refs from the last document snapshot."""
    for event in reversed(events):
        if event.type == TraceEventType.DOCUMENT_SNAPSHOT:
            return dict(event.asset_refs)
    return {}


def _collect_validation_metrics(events: list[TraceEvent]) -> dict[str, Any]:
    """Merge validation metrics from validation-report events."""
    metrics: dict[str, Any] = {}
    for event in events:
        if event.type == TraceEventType.VALIDATION_REPORT:
            report_metrics = event.payload.get("report", {}).get("metrics", {})
            if isinstance(report_metrics, dict):
                metrics.update(report_metrics)
    return metrics


def _available_tools() -> list[dict[str, str]]:
    """Return the prototype tool catalog used by training export."""
    return [
        {"name": "resize_canvas", "description": "Resize the canvas around its center."},
        {"name": "crop", "description": "Crop the document, or clear outside a crop on one layer or mask."},
        {"name": "import_image_as_layer", "description": "Import an image file into a full-canvas raster layer."},
        {"name": "import_vector_as_raster", "description": "Rasterize a vector asset and import it as a full-canvas raster layer."},
        {"name": "rasterize_vector_asset", "description": "Rasterize a vector asset to a standalone PNG or NPY artifact."},
        {"name": "create_layer", "description": "Create a new full-canvas layer."},
        {"name": "delete_layer", "description": "Remove a layer from the document stack."},
        {"name": "duplicate_layer", "description": "Create a deep copy of a layer."},
        {"name": "rename_layer", "description": "Rename a layer without changing its ID."},
        {"name": "reorder_layer", "description": "Move a layer to a new stack index."},
        {"name": "set_active_layer", "description": "Set the active layer."},
        {"name": "set_layer_visibility", "description": "Show or hide a layer."},
        {"name": "set_layer_opacity", "description": "Set a layer's opacity."},
        {"name": "set_blend_mode", "description": "Set a layer's blend mode metadata."},
        {"name": "merge_layers", "description": "Merge layers using normal source-over compositing."},
        {"name": "select_rect", "description": "Create a rectangular selection mask."},
        {"name": "select_ellipse", "description": "Create an elliptical selection mask."},
        {"name": "select_color_range", "description": "Create a mask from pixels close to a target color."},
        {"name": "magic_wand_select", "description": "Create a contiguous color-based selection from seed points."},
        {"name": "create_mask_from_shape", "description": "Create a mask from a deterministic geometric shape."},
        {"name": "grow_mask", "description": "Grow a mask by a pixel radius."},
        {"name": "shrink_mask", "description": "Shrink a mask by a pixel radius."},
        {"name": "invert_mask", "description": "Invert a mask."},
        {"name": "combine_masks", "description": "Combine masks with union, intersect, or subtract."},
        {"name": "feather_mask", "description": "Create a softened copy of a mask."},
        {"name": "draw_shape", "description": "Draw a deterministic geometric shape on a target layer."},
        {"name": "paint_bucket_fill", "description": "Fill the current write mask on a target layer with a color."},
        {"name": "blur_region", "description": "Apply Gaussian blur to selected channels through a write mask."},
        {"name": "clear_region", "description": "Clear pixels or alpha inside a write mask on a target layer."},
        {"name": "export_flat", "description": "Export a flattened preview image."},
        {"name": "no_op", "description": "Execute no document mutation."},
    ]


def _safe_label(label: str) -> str:
    """Return a filesystem-safe label segment."""
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label.strip().lower())
    return safe or "snapshot"


def _required_string(data: dict[str, Any], key: str) -> str:
    """Read a required non-empty string from a mapping."""
    value = data.get(key)
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_string(value: Any, field_name: str) -> Optional[str]:
    """Validate an optional string."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or None")
    return value


def _optional_int(value: Any, field_name: str) -> Optional[int]:
    """Validate an optional integer."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer or None")
    return value


def _mapping_value(value: Any, field_name: str) -> dict[str, Any]:
    """Validate a dictionary value."""
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be an object")
    return dict(value)


def _string_mapping(value: Any, field_name: str) -> dict[str, str]:
    """Validate a dictionary with string values."""
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be an object")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(item, str):
            raise TypeError(f"{field_name}.{key} must be a string")
        result[str(key)] = item
    return result


def _json_safe(value: Any) -> Any:
    """Convert common project objects into JSON-compatible structures."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else repr(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else repr(number)
    if isinstance(value, np.ndarray):
        return {"array": {"shape": list(value.shape), "dtype": str(value.dtype)}}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=repr)]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return {
            field_name: _json_safe(getattr(value, field_name))
            for field_name in value.__dataclass_fields__
        }
    return repr(value)
