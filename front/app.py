"""Streamlit front end for the DocAgent API.

Upload a PDF, choose the languages, follow the progress, download the result
and compare original and translation page by page.

Run:  uv run --group front streamlit run front/app.py
(the API must be running; DOCAGENT_API_URL defaults to http://localhost:8000)
"""

import os
import time

import httpx
import pymupdf
import streamlit as st

API_URL = os.environ.get("DOCAGENT_API_URL", "http://localhost:8000").rstrip("/")
LANGUAGES = {
    "en": "English",
    "fr": "Français",
    "de": "Deutsch",
    "es": "Español",
    "it": "Italiano",
    "pt": "Português",
}
POLL_SECONDS = 1.5

st.set_page_config(page_title="DocAgent", page_icon="📄", layout="wide")
st.title("DocAgent · layout-preserving PDF translation")


def api_health() -> dict | None:
    try:
        return httpx.get(f"{API_URL}/health", timeout=5).json()
    except httpx.HTTPError:
        return None


def render_page(pdf: bytes, number: int, zoom: float = 1.4) -> bytes:
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        return doc[number].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("png")


def page_count(pdf: bytes) -> int:
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        return doc.page_count


def detect(name: str, pdf: bytes) -> dict | None:
    """The API's language detection for an uploaded PDF (once per file)."""
    key = f"detected:{name}:{len(pdf)}"
    if key not in st.session_state:
        try:
            response = httpx.post(
                f"{API_URL}/v1/detect", files={"file": (name, pdf, "application/pdf")}, timeout=30
            )
            st.session_state[key] = response.json() if response.status_code == 200 else None
        except httpx.HTTPError:
            st.session_state[key] = None
    return st.session_state[key]


with st.sidebar:
    health = api_health()
    if health is None:
        st.error(f"API unreachable at {API_URL}")
    else:
        st.success("API online")
        st.caption(f"Translator: {health['translator']} · LLM: {health['llm'] or '—'}")

uploaded = st.file_uploader("PDF to translate", type=["pdf"])

# Default languages: the detected source, and French (or English for a French document).
codes = list(LANGUAGES)
detected = detect(uploaded.name, uploaded.getvalue()) if uploaded and health else None
source_default = detected["language"] if detected and detected["language"] in LANGUAGES else "en"
target_default = "en" if source_default == "fr" else "fr"

with st.sidebar:
    source = st.selectbox(
        "Source language",
        codes,
        format_func=LANGUAGES.get,
        index=codes.index(source_default),
        key=f"source:{uploaded.name if uploaded else ''}",
    )
    if detected and detected["language"]:
        how = "PDF metadata" if detected["method"] == "metadata" else "text analysis"
        st.caption(
            f"Detected: {LANGUAGES[detected['language']]} ({how}, {detected['confidence']:.0%})"
        )
    elif uploaded and health:
        st.caption("Language not detected: choose it.")
    target = st.selectbox(
        "Target language",
        codes,
        format_func=LANGUAGES.get,
        index=codes.index(target_default),
        key=f"target:{uploaded.name if uploaded else ''}",
    )

if uploaded and st.button("Translate", type="primary", disabled=health is None):
    if source == target:
        st.warning("Choose two different languages.")
        st.stop()
    response = httpx.post(
        f"{API_URL}/v1/translate",
        files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
        data={"source_lang": source, "target_lang": target},
        timeout=60,
    )
    if response.status_code != 202:
        st.error(f"Upload refused ({response.status_code}): {response.json().get('detail')}")
        st.stop()
    job_id = response.json()["job_id"]

    progress = st.progress(0.0, text="Queued…")
    while True:
        job = httpx.get(f"{API_URL}/v1/jobs/{job_id}", timeout=10).json()
        done, total = job["pages_done"], job["pages_total"]
        progress.progress(done / total, text=f"{job['status']} · page {done}/{total}")
        if job["status"] in ("done", "failed"):
            break
        time.sleep(POLL_SECONDS)

    if job["status"] == "failed":
        st.error(f"Translation failed: {job['error']}")
        st.stop()
    result = httpx.get(f"{API_URL}/v1/jobs/{job_id}/download", timeout=60).content
    st.session_state["result"] = {
        "original": uploaded.getvalue(),
        "translated": result,
        "name": f"{uploaded.name.rsplit('.', 1)[0]}_{target}.pdf",
        "job": job,
    }

if "result" in st.session_state:
    res = st.session_state["result"]
    summary = res["job"]["summary"] or {}
    cols = st.columns(4)
    cols[0].metric("Pages", res["job"]["pages_total"])
    cols[1].metric("Zones rewritten", summary.get("zones", "—"))
    cols[2].metric("Mean font size", f"{summary.get('mean_font_scale', 0):.0%}")
    cols[3].metric("Duration", f"{res['job']['duration_s']} s")
    st.download_button(
        "Download translated PDF", res["translated"], file_name=res["name"], mime="application/pdf"
    )

    pages = page_count(res["original"])
    number = st.slider("Page", 1, pages, 1) - 1 if pages > 1 else 0
    left, right = st.columns(2)
    left.image(render_page(res["original"], number), caption="Original", width="stretch")
    right.image(render_page(res["translated"], number), caption="Translation", width="stretch")
