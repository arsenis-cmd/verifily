"""Async job queue for Verifily API.

In-memory job store with a single worker thread that executes jobs serially.
Optional persistence via VERIFILY_JOBS_PERSIST=1 (append-only JSONL, replay on boot).
Thread-safe.  No external dependencies (no Redis, no Celery).
"""

from __future__ import annotations

import enum
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from verifily_cli_v1.core.budget import BudgetCheckResult, BudgetMode, BudgetStore, budget_store
from verifily_cli_v1.core.api.usage_store import UsageStore

logger = logging.getLogger("verifily.api")


# ── Enums ────────────────────────────────────────────────────────

class JobStatus(str, enum.Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class JobType(str, enum.Enum):
    PIPELINE = "PIPELINE"
    REPORT = "REPORT"
    CONTAMINATION = "CONTAMINATION"
    CLASSIFY = "CLASSIFY"
    RETRAIN = "RETRAIN"


class BudgetExceededError(Exception):
    """Raised when a job submission exceeds budget limits."""
    
    def __init__(self, message: str, budget_result: BudgetCheckResult):
        super().__init__(message)
        self.budget_result = budget_result


# ── Job record ───────────────────────────────────────────────────

@dataclass
class JobRecord:
    id: str
    type: JobType
    status: JobStatus
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    request_id: Optional[str] = None
    api_key_id: str = "anonymous"
    project_id: str = "default"
    input_hash: Optional[str] = None
    result_path: Optional[str] = None
    error: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None

    def to_meta(self) -> Dict[str, Any]:
        """Return metadata dict (everything except payload and result)."""
        return {
            "job_id": self.id,
            "type": self.type.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "request_id": self.request_id,
            "api_key_id": self.api_key_id,
            "project_id": self.project_id,
            "input_hash": self.input_hash,
            "result_path": self.result_path,
            "error": self.error,
        }


# ── Executor registry ────────────────────────────────────────────

# Maps JobType -> callable(payload) -> result dict.
# Populated by server.py at app creation time.
_executors: Dict[JobType, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}


def register_executor(job_type: JobType, fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
    _executors[job_type] = fn


# ── Jobs store (singleton) ───────────────────────────────────────

class JobsStore:
    """Thread-safe in-memory job store with optional JSONL persistence."""

    def __init__(self, budget_store_instance: Optional[BudgetStore] = None) -> None:
        self._lock = threading.Lock()
        self._jobs: Dict[str, JobRecord] = {}
        self._queue: List[str] = []  # job ids waiting to run
        self._persist_path: Optional[str] = None
        self._worker: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._work_event = threading.Event()  # signal new work
        self._budget_store = budget_store_instance or budget_store

    # ── Configuration ─────────────────────────────────────────

    def configure_persistence(self, path: Optional[str]) -> None:
        with self._lock:
            self._persist_path = path
        if path:
            self._replay(path)

    def _replay(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            return
        try:
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    evt = json.loads(line)
                    jid = evt.get("job_id")
                    if not jid:
                        continue
                    action = evt.get("action")
                    with self._lock:
                        if action == "created":
                            rec = JobRecord(
                                id=jid,
                                type=JobType(evt["type"]),
                                status=JobStatus(evt["status"]),
                                created_at=evt["created_at"],
                                request_id=evt.get("request_id"),
                                api_key_id=evt.get("api_key_id", "anonymous"),
                                project_id=evt.get("project_id", "default"),
                                input_hash=evt.get("input_hash"),
                                payload=evt.get("payload", {}),
                            )
                            self._jobs[jid] = rec
                        elif action == "started":
                            if jid in self._jobs:
                                self._jobs[jid].status = JobStatus.RUNNING
                                self._jobs[jid].started_at = evt.get("started_at")
                        elif action == "succeeded":
                            if jid in self._jobs:
                                self._jobs[jid].status = JobStatus.SUCCEEDED
                                self._jobs[jid].finished_at = evt.get("finished_at")
                                self._jobs[jid].result_path = evt.get("result_path")
                        elif action == "failed":
                            if jid in self._jobs:
                                self._jobs[jid].status = JobStatus.FAILED
                                self._jobs[jid].finished_at = evt.get("finished_at")
                                self._jobs[jid].error = evt.get("error")
        except Exception:
            logger.warning("jobs_store: failed to replay %s", path, exc_info=True)

    def _persist_event(self, event: Dict[str, Any]) -> None:
        path = self._persist_path
        if not path:
            return
        try:
            with open(path, "a") as f:
                f.write(json.dumps(event, separators=(",", ":")) + "\n")
                f.flush()
        except Exception:
            logger.warning("jobs_store: persist failed", exc_info=True)

    # ── Worker lifecycle ──────────────────────────────────────

    def start_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop_event.clear()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def stop_worker(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        self._work_event.set()  # wake up if sleeping
        if self._worker is not None:
            self._worker.join(timeout=timeout)
            self._worker = None

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            job_id = self._dequeue()
            if job_id is None:
                self._work_event.wait(timeout=0.5)
                self._work_event.clear()
                continue
            self._execute(job_id)

    def _dequeue(self) -> Optional[str]:
        with self._lock:
            if not self._queue:
                return None
            return self._queue.pop(0)

    def _execute(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status != JobStatus.QUEUED:
                return
            job.status = JobStatus.RUNNING
            job.started_at = time.time()
            job_type = job.type
            payload = dict(job.payload)

        self._persist_event({
            "action": "started",
            "job_id": job_id,
            "started_at": job.started_at,
        })

        executor = _executors.get(job_type)
        if executor is None:
            with self._lock:
                job.status = JobStatus.FAILED
                job.finished_at = time.time()
                job.error = f"No executor registered for {job_type.value}"
            self._persist_event({
                "action": "failed",
                "job_id": job_id,
                "finished_at": job.finished_at,
                "error": job.error,
            })
            return

        try:
            result = executor(payload)
            with self._lock:
                job.status = JobStatus.SUCCEEDED
                job.finished_at = time.time()
                job.result = result
            self._persist_event({
                "action": "succeeded",
                "job_id": job_id,
                "finished_at": job.finished_at,
            })
        except Exception as exc:
            with self._lock:
                job.status = JobStatus.FAILED
                job.finished_at = time.time()
                job.error = str(exc)
            self._persist_event({
                "action": "failed",
                "job_id": job_id,
                "finished_at": job.finished_at,
                "error": job.error,
            })

    # ── Public API ────────────────────────────────────────────

    def submit(
        self,
        job_type: JobType,
        payload: Dict[str, Any],
        *,
        request_id: Optional[str] = None,
        api_key_id: str = "anonymous",
        project_id: str = "default",
        input_hash: Optional[str] = None,
        check_budget: bool = True,
    ) -> str:
        # Check budget if enabled
        if check_budget and self._budget_store:
            budget_result = self._budget_store.check_budget(project_id)
            if budget_result.mode == BudgetMode.BLOCK:
                raise BudgetExceededError(
                    f"Budget exceeded for project {project_id}: {budget_result.reason}",
                    budget_result,
                )
        
        job_id = uuid.uuid4().hex[:16]
        now = time.time()
        rec = JobRecord(
            id=job_id,
            type=job_type,
            status=JobStatus.QUEUED,
            created_at=now,
            request_id=request_id,
            api_key_id=api_key_id,
            project_id=project_id,
            input_hash=input_hash,
            payload=payload,
        )
        with self._lock:
            self._jobs[job_id] = rec
            self._queue.append(job_id)

        self._persist_event({
            "action": "created",
            "job_id": job_id,
            "type": job_type.value,
            "status": JobStatus.QUEUED.value,
            "created_at": now,
            "request_id": request_id,
            "api_key_id": api_key_id,
            "project_id": project_id,
            "input_hash": input_hash,
            "payload": payload,
        })

        self._work_event.set()
        return job_id

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def result(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.status == JobStatus.SUCCEEDED:
                return job.result
            if job.status == JobStatus.FAILED:
                return {"error": job.error}
            return None

    def list_jobs(
        self,
        *,
        status: Optional[str] = None,
        project_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())

        if status:
            jobs = [j for j in jobs if j.status.value == status]
        if project_id:
            jobs = [j for j in jobs if j.project_id == project_id]

        # Most recent first
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return [j.to_meta() for j in jobs[:limit]]

    def reset(self) -> None:
        """Clear all state (for test isolation)."""
        self.stop_worker()
        with self._lock:
            self._jobs.clear()
            self._queue.clear()
            self._persist_path = None

    def wait_for_job(self, job_id: str, timeout: float = 30.0, poll: float = 0.05) -> Optional[JobRecord]:
        """Block until job finishes or timeout.  For testing convenience."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.get(job_id)
            if job and job.status in (JobStatus.SUCCEEDED, JobStatus.FAILED):
                return job
            time.sleep(poll)
        return self.get(job_id)

    def drain(self) -> int:
        """Synchronously execute ALL queued jobs.  TEST-ONLY.

        Bypasses the worker thread entirely — no sleep, no polling.
        Returns the number of jobs executed.
        """
        executed = 0
        while True:
            job_id = self._dequeue()
            if job_id is None:
                break
            self._execute(job_id)
            executed += 1
        return executed


# Singleton
jobs_store = JobsStore()
