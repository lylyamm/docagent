"""Streamlit front end for the DocAgent API.

- Translate: upload a PDF, choose the languages, follow the progress, download
  the result and compare original and translation page by page.
- Ask: a question about the indexed reports, answered with its sources and the
  sentences that prove it (POST /v1/ask).

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
EXAMPLES = [
    "Quel est le premier secteur émetteur de gaz à effet de serre en France en 2024 ?",
    "How much did global mean sea level rise between 1901 and 2018?",
    "Combien d'argent sera investi dans les centres de données en 2025 ?",
    "Qui a gagné la Coupe du monde de football 2018 ?",
]

st.set_page_config(page_title="DocAgent", page_icon="📄", layout="wide")
st.title("DocAgent · climate & energy reports")


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


def translation_tab(health: dict | None) -> None:
    uploaded = st.file_uploader("PDF to translate", type=["pdf"])

    # Default languages: the detected source, and French (or English for a French document).
    codes = list(LANGUAGES)
    detected = detect(uploaded.name, uploaded.getvalue()) if uploaded and health else None
    found = detected["language"] if detected else None
    source_default = found if found in LANGUAGES else "en"
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
            return
        response = httpx.post(
            f"{API_URL}/v1/translate",
            files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
            data={"source_lang": source, "target_lang": target},
            timeout=60,
        )
        if response.status_code != 202:
            st.error(f"Upload refused ({response.status_code}): {response.json().get('detail')}")
            return
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
            return
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
            "Download translated PDF",
            res["translated"],
            file_name=res["name"],
            mime="application/pdf",
        )

        pages = page_count(res["original"])
        number = st.slider("Page", 1, pages, 1) - 1 if pages > 1 else 0
        left, right = st.columns(2)
        left.image(render_page(res["original"], number), caption="Original", width="stretch")
        right.image(render_page(res["translated"], number), caption="Translation", width="stretch")


def ask(question: str, top_k: int) -> dict:
    """POST /v1/ask. The first question loads the search index: up to a minute."""
    try:
        response = httpx.post(
            f"{API_URL}/v1/ask", json={"question": question, "top_k": top_k}, timeout=180
        )
    except httpx.HTTPError as exc:
        return {"error": f"API unreachable: {exc}"}
    if response.status_code != 200:
        return {"error": f"{response.status_code}: {response.json().get('detail')}"}
    return response.json()


def show_answer(result: dict) -> None:
    if "error" in result:
        st.error(result["error"])
        return
    if result["answerable"]:
        st.success(result["answer"])
    else:
        st.warning(result["answer"])
        st.caption(
            "No answer is shown unless the model quotes a sentence that the code finds "
            "in the reports, and every figure of the answer appears in the cited excerpts."
        )
    for source in result["sources"]:
        label = f"[{source['n']}] {source['title']}, p. {source['page']}"
        with st.expander(label, expanded=True):
            if source.get("section"):
                st.caption(source["section"])
            for quote in source["quotes"]:
                st.markdown(f"> {quote}")
            excerpt = source["excerpt"]
            short = excerpt if len(excerpt) <= 400 else excerpt[:400].rsplit(" ", 1)[0] + " …"
            st.caption(f"Excerpt given to the model: {short}")
    st.caption(
        f"{len(result['retrieved'])} passages given to {result['model']} "
        f"· prompt {result.get('prompt_version', '?')}"
    )


def questions_tab(health: dict | None) -> None:
    st.write(
        "Ask a question about the indexed reports (IPCC AR6 WGI, IEA World Energy Outlook "
        "2025, Haut Conseil pour le climat 2025), in French or English."
    )
    cols = st.columns(len(EXAMPLES))
    for col, example in zip(cols, EXAMPLES, strict=True):
        if col.button(example, width="stretch"):
            st.session_state["question"] = example
    with st.form("ask"):
        question = st.text_input("Question", key="question", max_chars=500)
        top_k = st.slider("Passages given to the model", 3, 10, 5)
        sent = st.form_submit_button("Ask", type="primary", disabled=health is None)
    if sent and len(question.strip()) >= 3:
        with st.spinner("Searching the reports and checking the answer…"):
            result = ask(question.strip(), top_k)
        st.session_state.setdefault("history", []).insert(0, result | {"question": question})
    history = st.session_state.get("history", [])
    if history:
        show_answer(history[0])
    if len(history) > 1:
        with st.expander("Previous questions"):
            for past in history[1:6]:
                st.markdown(f"**{past['question']}**  \n{past.get('answer', past.get('error'))}")


translate, questions = st.tabs(["Translate a PDF", "Ask the reports"])
with translate:
    translation_tab(health)
with questions:
    questions_tab(health)
