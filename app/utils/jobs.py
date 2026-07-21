"""Background job manager (Module 7).

Ingesting a full watch history (thousands of videos → LLM categorisation +
embeddings) takes minutes, which is far too long for a single HTTP request.
So the upload endpoint validates the file synchronously, then hands the slow
ingest to this manager, which runs it on a daemon thread and tracks progress.
The client polls a status endpoint until the job is `done` or `error`.

This is intentionally simple in-memory state (fine for a single-process,
single-user local app). A multi-worker or multi-user deployment would swap
this for a real task queue (Celery/RQ) + shared store — noted as future work.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)

# A worker receives a progress callback: progress_cb(done, total, phase).
ProgressCallback = Callable[[int, int, str], None]
JobWorker = Callable[[ProgressCallback], dict]


@dataclass
class Job:
    """Mutable status record for one background operation."""
    id: str
    status: str = "running"        # running | done | error
    phase: str = "starting"
    done: int = 0
    total: int = 0
    result: dict | None = None     # payload on success (e.g. parse+ingest stats)
    error: str | None = None       # message on failure

    def to_dict(self) -> dict:
        return {
            "job_id": self.id, "status": self.status, "phase": self.phase,
            "done": self.done, "total": self.total,
            "result": self.result, "error": self.error,
        }


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, total: int = 0) -> Job:
        job = Job(id=uuid.uuid4().hex, total=total)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def run(self, job: Job, worker: JobWorker) -> None:
        """Run `worker` on a daemon thread, updating `job` as it progresses."""

        def _progress(done: int, total: int, phase: str) -> None:
            job.done, job.total, job.phase = done, total, phase

        def _target() -> None:
            try:
                job.result = worker(_progress)
                job.status, job.phase = "done", "done"
            except Exception as exc:  # noqa: BLE001 — surface any failure via status
                job.error = str(exc)
                job.status = "error"
                logger.exception("Background job %s failed", job.id)

        threading.Thread(target=_target, name=f"job-{job.id}", daemon=True).start()


# Process-wide singleton.
job_manager = JobManager()
