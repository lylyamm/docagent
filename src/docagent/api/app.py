"""HTTP API: upload a PDF, follow the translation job, download the result.

    POST /v1/detect                 upload a PDF -> its language (metadata, else text)
    POST /v1/translate              upload a PDF -> 202 {"job_id": ...} (source "auto" by default)
    GET  /v1/jobs/{job_id}          status and progress
    GET  /v1/jobs/{job_id}/download translated PDF (when done)
    POST /v1/ask                    question -> answer from the indexed reports, with sources
    GET  /health                    liveness + configured LLM

Run locally:  uv run uvicorn docagent.api.app:serve --factory --reload
Interactive docs: http://localhost:8000/docs
"""

import logging
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated

import pymupdf
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..chat import ChatError
from ..config import Settings, get_settings
from ..pdf import rebuild_document
from ..pdf.language import SUPPORTED, DetectedLanguage, detect_language
from ..pdf.verify import verify_rebuild
from ..rag.answer import Answer, Answerer, build_answerer
from ..translate import Translator, build_translator
from .jobs import Job, JobStatus, JobStore, JobSummary
from .logging_config import configure_logging

logger = logging.getLogger("docagent.api")

LANGUAGES = set(SUPPORTED)


class JobCreated(BaseModel):
    job_id: str
    status: JobStatus
    pages: int
    source_lang: str
    detected: DetectedLanguage | None = None  # set when source_lang was "auto"


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    top_k: int | None = Field(default=None, ge=1, le=10, description="passages given to the LLM")


class Health(BaseModel):
    status: str
    translator: str
    llm: str | None


def create_app(
    settings: Settings | None = None,
    translator_factory: Callable[[], Translator] | None = None,
    answerer_factory: Callable[[], Answerer] | None = None,
    warm_up: bool = False,
) -> FastAPI:
    """Build the app. Tests inject settings, a fake translator and a fake answerer.

    ``warm_up``: load the search index and the embedding model in the background
    at start-up, so the first question does not wait for them (``serve`` does).
    """
    settings = settings or get_settings()
    store = JobStore(settings.jobs_dir)
    make_translator = translator_factory or (
        lambda: build_translator(settings, fake=settings.translator == "fake")
    )
    make_answerer = answerer_factory or (lambda: build_answerer(settings))
    # The search index (passages, embedding model, vectors) is loaded on the first
    # question, not at start-up: translation works without it. One instance is
    # shared; the lock stops two first questions from loading it twice.
    answerer: list[Answerer] = []
    answerer_lock = threading.Lock()

    def get_answerer() -> Answerer:
        with answerer_lock:
            if not answerer:
                answerer.append(make_answerer())
            return answerer[0]

    def warm() -> None:
        start = time.perf_counter()
        try:
            get_answerer().searcher.search("warm-up", 1, "hybrid")  # loads the model too
        except Exception as exc:  # noqa: BLE001 - no index yet: questions will say so
            logger.warning("search index not loaded at start-up: %s", exc)
            return
        duration = round(time.perf_counter() - start, 2)
        logger.info("search index ready", extra={"duration_s": duration})

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if warm_up:
            threading.Thread(target=warm, name="rag-warm-up", daemon=True).start()
        yield

    app = FastAPI(
        lifespan=lifespan,
        title="DocAgent",
        version="0.2.0",
        description="Layout-preserving PDF translation and question answering "
        "over climate & energy reports.",
    )

    @app.get("/health")
    def health() -> Health:
        llm = None
        if settings.translator == "llm":
            try:
                llm = f"{settings.llm_provider}:{settings.resolved_model}"
            except RuntimeError as exc:  # misconfiguration: report it, don't crash
                llm = f"misconfigured: {exc}"
        return Health(status="ok", translator=settings.translator, llm=llm)

    async def read_pdf(file: UploadFile) -> tuple[bytes, int]:
        """The uploaded bytes and page count, after every check that can reject them."""
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(415, "only .pdf files are accepted")
        limit = settings.max_upload_mb * 1024 * 1024
        data = await file.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, f"file larger than {settings.max_upload_mb} MB")
        if not data.startswith(b"%PDF"):
            raise HTTPException(415, "the file is not a PDF")
        try:
            with pymupdf.open(stream=data, filetype="pdf") as doc:
                pages = doc.page_count
                encrypted = doc.needs_pass
        except Exception as exc:  # noqa: BLE001 - any parsing failure is a bad upload
            raise HTTPException(422, "the PDF could not be opened") from exc
        if encrypted:
            raise HTTPException(422, "password-protected PDFs are not supported")
        if not 0 < pages <= settings.max_pages:
            raise HTTPException(413, f"PDF must have 1 to {settings.max_pages} pages")
        return data, pages

    def detect(data: bytes) -> DetectedLanguage:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            return detect_language(doc)

    @app.post("/v1/detect")
    async def detect_endpoint(
        file: Annotated[UploadFile, File(description="PDF whose language to detect")],
    ) -> DetectedLanguage:
        data, _pages = await read_pdf(file)
        return detect(data)

    @app.post("/v1/translate", status_code=202)
    async def translate(
        background: BackgroundTasks,
        file: Annotated[UploadFile, File(description="PDF to translate")],
        target_lang: Annotated[str, Form()] = "fr",
        source_lang: Annotated[str, Form(description='"auto" to detect it')] = "auto",
    ) -> JobCreated:
        if (source_lang != "auto" and source_lang not in LANGUAGES) or target_lang not in LANGUAGES:
            raise HTTPException(422, f"languages must be among {sorted(LANGUAGES)} (or auto)")
        if source_lang == target_lang:
            raise HTTPException(422, "source and target languages are the same")
        data, pages = await read_pdf(file)

        detected = None
        if source_lang == "auto":
            detected = detect(data)
            if detected.language is None:
                raise HTTPException(422, "could not detect the language: set source_lang")
            if detected.language == target_lang:
                raise HTTPException(422, f"the document is already in {target_lang}")
            source_lang = detected.language

        job = store.create(
            Job(
                filename=file.filename,
                source_lang=source_lang,
                target_lang=target_lang,
                pages_total=pages,
            ),
            data,
        )
        logger.info(
            "job created",
            extra={"job_id": job.id, "pages": pages, "source_lang": source_lang},
        )
        background.add_task(_run_job, store, job.id, make_translator)
        return JobCreated(
            job_id=job.id,
            status=job.status,
            pages=pages,
            source_lang=source_lang,
            detected=detected,
        )

    @app.get("/v1/jobs/{job_id}")
    def job_status(job_id: str) -> Job:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job")
        return job

    @app.get("/v1/jobs/{job_id}/download")
    def download(job_id: str) -> FileResponse:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job")
        if job.status != JobStatus.DONE:
            raise HTTPException(409, f"job is {job.status.value}, not done")
        stem = job.filename.rsplit(".", 1)[0]
        return FileResponse(
            store.output_path(job_id),
            media_type="application/pdf",
            filename=f"{stem}_{job.target_lang}.pdf",
        )

    @app.post("/v1/ask")
    def ask(request: AskRequest) -> Answer:
        try:
            qa = get_answerer()
        except FileNotFoundError as exc:
            raise HTTPException(503, f"search index not built: {exc}") from exc
        start = time.perf_counter()
        try:
            answer = qa.ask(request.question, request.top_k)
        except ChatError as exc:
            logger.warning("ask failed: %s", exc)
            raise HTTPException(502, "the language model did not give a valid answer") from exc
        logger.info(
            "question answered",
            extra={
                "answerable": answer.answerable,
                "sources": len(answer.sources),
                "duration_s": round(time.perf_counter() - start, 2),
                **answer.timings,
            },
        )
        return answer

    return app


def _run_job(store: JobStore, job_id: str, make_translator: Callable[[], Translator]) -> None:
    """Translate one job; any failure is recorded on the job, never raised."""
    job = store.update(job_id, status=JobStatus.RUNNING)
    start = time.perf_counter()
    try:
        report = rebuild_document(
            store.input_path(job_id),
            store.output_path(job_id),
            make_translator(),
            source_lang=job.source_lang,
            target_lang=job.target_lang,
            on_page=lambda done, _total: store.update(job_id, pages_done=done),
        )
        check = verify_rebuild(store.input_path(job_id), store.output_path(job_id), report)
        duration = round(time.perf_counter() - start, 2)
        store.update(
            job_id,
            status=JobStatus.DONE,
            finished_at=datetime.now(UTC),
            duration_s=duration,
            summary=JobSummary(
                zones=report.targets,
                mean_font_scale=report.mean_scale,
                overflow=report.overflow,
                checks_ok=check.ok,
            ),
        )
        logger.info(
            "job done",
            extra={
                "job_id": job_id,
                "pages": job.pages_total,
                "zones": report.targets,
                "duration_s": duration,
                "checks_ok": check.ok,
            },
        )
    except Exception as exc:  # noqa: BLE001 - the job records the error for the client
        duration = round(time.perf_counter() - start, 2)
        store.update(
            job_id,
            status=JobStatus.FAILED,
            finished_at=datetime.now(UTC),
            duration_s=duration,
            error=f"{type(exc).__name__}: {exc}",
        )
        logger.exception("job failed", extra={"job_id": job_id, "duration_s": duration})


def serve() -> FastAPI:
    """Entry point for uvicorn (``--factory``): JSON logs + settings from .env."""
    configure_logging()
    return create_app(warm_up=True)
