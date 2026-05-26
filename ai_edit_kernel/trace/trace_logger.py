"""Trace logging for the AI Editing Kernel.

Traces are the bridge between today's prototype and tomorrow's trainable model.
Every user request, plan, action, validation result, and before/after snapshot
should eventually be logged in a replayable format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from ai_edit_kernel.document.document_state import DocumentState
from ai_edit_kernel.schema.actions import Action, ActionBatch, ActionResult
from ai_edit_kernel.runtime.validator import ValidationReport


class TraceEventType(str, Enum):
    """Kinds of events saved into a trace."""

    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    USER_PROMPT = "user_prompt"
    DOCUMENT_SNAPSHOT = "document_snapshot"
    OBSERVATION = "observation"
    ACTION_PLANNED = "action_planned"
    ACTION_BATCH_PLANNED = "action_batch_planned"
    ACTION_STARTED = "action_started"
    ACTION_RESULT = "action_result"
    VALIDATION_REPORT = "validation_report"
    ERROR = "error"
    NOTE = "note"


@dataclass(slots=True)
class TraceEvent:
    """One timestamped event in an editing session trace."""

    id: str
    type: TraceEventType
    timestamp: str
    document_id: Optional[str] = None
    document_revision: Optional[int] = None
    action_id: Optional[str] = None
    payload: dict[str, Any] = field(default_factory=dict)
    asset_refs: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Serialize this event into JSON-compatible form."""
        raise NotImplementedError

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "TraceEvent":
        """Deserialize a trace event from JSON-compatible data."""
        raise NotImplementedError


@dataclass(slots=True)
class TraceSession:
    """A complete editing session consisting of many TraceEvents."""

    id: str
    started_at: str
    ended_at: Optional[str] = None
    user_prompt: Optional[str] = None
    events: list[TraceEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_event(self, event: TraceEvent) -> None:
        """Append an event to this session."""
        raise NotImplementedError

    def to_json(self) -> dict[str, Any]:
        """Serialize the full session to JSON-compatible form."""
        raise NotImplementedError

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "TraceSession":
        """Deserialize a session from JSON-compatible data."""
        raise NotImplementedError


@dataclass(slots=True)
class TraceLogger:
    """Record editing sessions, actions, results, and snapshots.

    The trace logger should be lightweight enough to run during every prototype
    demo, but structured enough that traces can later become supervised training
    data for a tool-using model.
    """

    output_dir: Path
    session: Optional[TraceSession] = None
    save_snapshots: bool = True
    save_preview_images: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def start_session(self, user_prompt: Optional[str] = None, metadata: Optional[dict[str, Any]] = None) -> TraceSession:
        """Create and activate a new trace session.

        Must assign a unique session ID, capture start time, store optional prompt
        and metadata, and emit a SESSION_STARTED event.
        """
        raise NotImplementedError

    def end_session(self) -> TraceSession:
        """End the active session and emit a SESSION_ENDED event."""
        raise NotImplementedError

    def log_user_prompt(self, prompt: str, document: Optional[DocumentState] = None) -> None:
        """Record the user's natural-language request."""
        raise NotImplementedError

    def log_document_snapshot(self, document: DocumentState, label: str) -> None:
        """Record a snapshot or snapshot reference for the current document.

        V0 can save a JSON summary. Later versions should save layered document
        bundles and flattened preview images for replay and training.
        """
        raise NotImplementedError

    def log_observation(self, document: DocumentState, observation: dict[str, Any], label: str) -> None:
        """Record perception/analyzer output.

        Examples: detected circle geometry, line-art regions, object masks, or
        document summaries sent to the planner.
        """
        raise NotImplementedError

    def log_action_batch_planned(self, batch: ActionBatch, document: Optional[DocumentState] = None) -> None:
        """Record a planned batch before execution."""
        raise NotImplementedError

    def log_action_planned(self, action: Action, document: Optional[DocumentState] = None) -> None:
        """Record one planned action before execution."""
        raise NotImplementedError

    def log_action_started(self, action: Action, document: DocumentState) -> None:
        """Record that the executor started an action."""
        raise NotImplementedError

    def log_action_result(self, action: Action, result: ActionResult, document: DocumentState) -> None:
        """Record action result, including created layers/masks and errors."""
        raise NotImplementedError

    def log_validation_report(
        self,
        report: ValidationReport,
        document: Optional[DocumentState] = None,
        action: Optional[Action] = None,
    ) -> None:
        """Record validation output for an action or document."""
        raise NotImplementedError

    def log_error(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        """Record an unexpected runtime/planning error."""
        raise NotImplementedError

    def log_note(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        """Record a human-readable note for debugging or review."""
        raise NotImplementedError

    def save_session(self, path: Optional[Path] = None) -> Path:
        """Write the active session to disk and return the saved path."""
        raise NotImplementedError

    def load_session(self, path: Path) -> TraceSession:
        """Load a trace session from disk and make it active if desired."""
        raise NotImplementedError

    def export_training_example(self, session: TraceSession) -> dict[str, Any]:
        """Convert a session into a supervised training example.

        Should produce something like: prompt + document summary + observations →
        validated action sequence + final evaluation.
        """
        raise NotImplementedError

    def export_dataset(self, sessions: list[TraceSession], output_path: Path) -> Path:
        """Export many sessions into a dataset file for future model training."""
        raise NotImplementedError
