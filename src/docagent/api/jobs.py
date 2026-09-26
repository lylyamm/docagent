"""Translation jobs: an in-memory registry mirrored to disk.

Each job has a folder ``<jobs_dir>/<job_id>/`` holding ``input.pdf``,
``output.pdf`` once done, and ``job.json`` (its status). Writing the status to
disk lets a restarted server still answer ``GET /v1/jobs/{id}`` for finished
jobs; jobs that were running when the server stopped are marked as failed.
"""

import threading
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class JobSummary(BaseModel):
    """What the rebuild report says, reduced to what an API client needs."""

    zones: int
    mean_font_scale: float
    overflow: int
    checks_ok: bool


class Job(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    filename: str
    source_lang: str
    target_lang: str
    status: JobStatus = JobStatus.QUEUED
    pages_total: int
    pages_done: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    duration_s: float | None = None
    error: str | None = None
    summary: JobSummary | None = None


class JobStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._load()

    def dir(self, job_id: str) -> Path:
        return self.root / job_id

    def input_path(self, job_id: str) -> Path:
        return self.dir(job_id) / "input.pdf"

    def output_path(self, job_id: str) -> Path:
        return self.dir(job_id) / "output.pdf"

    def create(self, job: Job, pdf: bytes) -> Job:
        self.dir(job.id).mkdir(parents=True)
        self.input_path(job.id).write_bytes(pdf)
        self.save(job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.model_copy() if job else None

    def update(self, job_id: str, **changes) -> Job:
        with self._lock:
            job = self._jobs[job_id].model_copy(update=changes)
        self.save(job)
        return job

    def save(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job
            path = self.dir(job.id) / "job.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(job.model_dump_json(indent=2), encoding="utf-8")
            tmp.replace(path)

    def _load(self) -> None:
        for path in self.root.glob("*/job.json"):
            try:
                job = Job.model_validate_json(path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            if job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
                job = job.model_copy(
                    update={"status": JobStatus.FAILED, "error": "server restarted during the job"}
                )
            self._jobs[job.id] = job
