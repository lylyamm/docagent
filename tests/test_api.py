"""API tests: FastAPI TestClient + fake translator (no network, no key)."""

from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

from docagent.api import create_app
from docagent.config import Settings
from docagent.translate import FakeTranslator, Translator

TEXT = "The climate in 2024 was the warmest ever measured around the world"
FRENCH = "Le climat de la planète se réchauffe très vite"


def pdf_bytes(pages: int = 1, text: str = TEXT) -> bytes:
    doc = pymupdf.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_htmlbox(pymupdf.Rect(60, 60, 500, 120), f"<p>{text}, page {i}</p>")
    return doc.tobytes()


class Boom(Translator):
    name = "boom"

    def translate_blocks(self, texts, source, target):
        raise RuntimeError("LLM unavailable")


def make_client(tmp_path: Path, translator: Translator | None = None, **settings) -> TestClient:
    values = {"jobs_dir": tmp_path / "jobs", "translator": "fake", "max_pages": 3}
    values.update(settings)
    app = create_app(
        Settings(_env_file=None, **values),
        translator_factory=lambda: translator or FakeTranslator(),
    )
    return TestClient(app)


def upload(client: TestClient, data: bytes, name: str = "report.pdf", **form):
    return client.post("/v1/translate", files={"file": (name, data, "application/pdf")}, data=form)


def test_health(tmp_path):
    body = make_client(tmp_path).get("/health").json()
    assert body == {"status": "ok", "translator": "fake", "llm": None}


def test_translate_job_lifecycle(tmp_path):
    client = make_client(tmp_path)
    created = upload(client, pdf_bytes(2), target_lang="fr")
    assert created.status_code == 202
    job_id = created.json()["job_id"]
    assert created.json()["pages"] == 2

    # TestClient runs background tasks before returning: the job is finished.
    job = client.get(f"/v1/jobs/{job_id}").json()
    assert job["status"] == "done"
    assert job["pages_done"] == 2
    assert job["summary"]["zones"] == 2 and job["summary"]["checks_ok"]

    download = client.get(f"/v1/jobs/{job_id}/download")
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/pdf"
    assert 'filename="report_fr.pdf"' in download.headers["content-disposition"]
    with pymupdf.open(stream=download.content, filetype="pdf") as doc:
        assert "THE CLIMATE IN 2024" in doc[0].get_text()


def test_failed_job_reports_the_error(tmp_path):
    client = make_client(tmp_path, translator=Boom())
    job_id = upload(client, pdf_bytes()).json()["job_id"]
    job = client.get(f"/v1/jobs/{job_id}").json()
    assert job["status"] == "failed"
    assert "LLM unavailable" in job["error"]
    assert client.get(f"/v1/jobs/{job_id}/download").status_code == 409


@pytest.mark.parametrize(
    ("data", "name", "form", "status"),
    [
        (b"hello", "notes.txt", {}, 415),  # not a .pdf
        (b"not really a pdf", "fake.pdf", {}, 415),  # wrong magic bytes
        (b"%PDF-1.7 broken", "broken.pdf", {}, 422),  # cannot be opened
        (None, "big.pdf", {}, 413),  # too many pages (max_pages=3)
        (None, "x.pdf", {"target_lang": "xx"}, 422),  # unknown language
        (None, "x.pdf", {"source_lang": "fr", "target_lang": "fr"}, 422),
    ],
)
def test_upload_validation(tmp_path, data, name, form, status):
    client = make_client(tmp_path)
    if data is None:
        data = pdf_bytes(5 if name == "big.pdf" else 1)
    assert upload(client, data, name, **form).status_code == status


def test_upload_size_limit(tmp_path):
    client = make_client(tmp_path, max_upload_mb=0)
    assert upload(client, pdf_bytes()).status_code == 413


def test_unknown_job(tmp_path):
    client = make_client(tmp_path)
    assert client.get("/v1/jobs/nope").status_code == 404
    assert client.get("/v1/jobs/nope/download").status_code == 404


def test_jobs_survive_a_restart(tmp_path):
    client = make_client(tmp_path)
    job_id = upload(client, pdf_bytes()).json()["job_id"]
    restarted = make_client(tmp_path)
    assert restarted.get(f"/v1/jobs/{job_id}").json()["status"] == "done"


def test_detect_endpoint(tmp_path):
    client = make_client(tmp_path)
    files = {"file": ("r.pdf", pdf_bytes(text=FRENCH), "")}
    body = client.post("/v1/detect", files=files).json()
    assert body["language"] == "fr" and body["method"] == "text"
    assert client.post("/v1/detect", files={"file": ("x.txt", b"hi", "")}).status_code == 415


def test_source_language_is_detected_by_default(tmp_path):
    created = upload(make_client(tmp_path), pdf_bytes(), target_lang="fr").json()
    assert created["source_lang"] == "en"
    assert created["detected"]["language"] == "en"


def test_document_already_in_the_target_language(tmp_path):
    response = upload(make_client(tmp_path), pdf_bytes(text=FRENCH), target_lang="fr")
    assert response.status_code == 422 and "already in fr" in response.json()["detail"]


def test_undetectable_language_asks_for_source_lang(tmp_path):
    response = upload(make_client(tmp_path), pdf_bytes(text="12 34 56"), target_lang="fr")
    assert response.status_code == 422 and "source_lang" in response.json()["detail"]
