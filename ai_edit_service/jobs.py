"""Asynchronous job bookkeeping for local edit requests."""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

from ai_edit_service.models import EditRequest, EditResult, JobStatus, ServiceError


EditRunner = Callable[[EditRequest], EditResult]


@dataclass(slots=True)
class EditJob:
    """In-memory record for one edit request."""

    id: str
    request_id: str
    status: JobStatus
    created_at: float
    updated_at: float
    progress: float = 0.0
    message: str = ""
    result: Optional[EditResult] = None
    error: Optional[ServiceError] = None
    metadata: dict[str, object] = field(default_factory=dict)

    def to_json(self, *, include_result: bool = False) -> dict[str, object]:
        data: dict[str, object] = {
            "id": self.id,
            "request_id": self.request_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "progress": self.progress,
            "message": self.message,
            "metadata": dict(self.metadata),
        }
        if self.error is not None:
            data["error"] = self.error.to_json()
        if include_result and self.result is not None:
            data["result"] = self.result.to_json()
        return data


@dataclass(slots=True)
class JobStore:
    """Thread-backed job runner for one local service process."""

    runner: EditRunner
    _jobs: dict[str, EditJob] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _condition: threading.Condition = field(init=False)

    def __post_init__(self) -> None:
        self._condition = threading.Condition(self._lock)

    def submit(self, request: EditRequest) -> EditJob:
        """Queue an edit request and start execution in a background thread."""
        now = time.time()
        job = EditJob(
            id=f"job_{uuid.uuid4().hex}",
            request_id=request.request_id,
            status=JobStatus.QUEUED,
            created_at=now,
            updated_at=now,
            message="queued",
        )
        with self._lock:
            self._jobs[job.id] = job
        thread = threading.Thread(target=self._run_job, args=(job.id, request), daemon=True)
        thread.start()
        return job

    def get(self, job_id: str) -> EditJob:
        """Return a job by ID or raise `KeyError`."""
        with self._lock:
            return self._jobs[job_id]

    def result(self, job_id: str) -> EditResult:
        """Return the completed result for a job."""
        job = self.get(job_id)
        if job.result is None:
            raise RuntimeError(f"job {job_id!r} has no result")
        return job.result

    def wait(self, job_id: str, timeout: float = 30.0) -> EditJob:
        """Wait for a job to reach a terminal state or until timeout expires."""
        deadline = time.time() + timeout
        with self._condition:
            while self._jobs[job_id].status in {JobStatus.QUEUED, JobStatus.RUNNING}:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)
            return self._jobs[job_id]

    def list_recent(self, limit: int = 50) -> list[EditJob]:
        """Return recent jobs sorted newest first."""
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)
        return jobs[:limit]

    def _run_job(self, job_id: str, request: EditRequest) -> None:
        self._mark(job_id, JobStatus.RUNNING, progress=0.05, message="running")
        try:
            result = self.runner(request)
        except Exception as exc:  # pragma: no cover - exercised by integration callers
            self._fail(job_id, exc)
            return
        self._complete(job_id, result)

    def _mark(self, job_id: str, status: JobStatus, *, progress: float, message: str) -> None:
        with self._condition:
            job = self._jobs[job_id]
            job.status = status
            job.progress = progress
            job.message = message
            job.updated_at = time.time()
            self._condition.notify_all()

    def _complete(self, job_id: str, result: EditResult) -> None:
        with self._condition:
            job = self._jobs[job_id]
            job.status = JobStatus.SUCCEEDED
            job.progress = 1.0
            job.message = "completed"
            job.result = result
            job.updated_at = time.time()
            self._condition.notify_all()

    def _fail(self, job_id: str, exc: Exception) -> None:
        with self._condition:
            job = self._jobs[job_id]
            job.status = JobStatus.FAILED
            job.progress = 1.0
            job.message = "failed"
            job.error = ServiceError(
                code=exc.__class__.__name__,
                message=str(exc),
                details={"traceback": traceback.format_exc()},
            )
            job.updated_at = time.time()
            self._condition.notify_all()
