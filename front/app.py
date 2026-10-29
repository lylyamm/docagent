"""Streamlit front end for the DocAgent API.

- Translate: upload a PDF, choose the languages, follow the progress, download
  the result and compare original and translation page by page.
- Ask: a question about the indexed reports, answered with its sources and the
  sentences that prove it (POST /v1/ask).
- Assistant: any request to the agent (POST /v1/agent): its plan, each step as soon
  as it is done, its questions (yes / no), and the details of the run.

Run:  uv run --group front streamlit run front/app.py
(the API must be running; DOCAGENT_API_URL defaults to http://localhost:8000)

On Cloud Run the API is private: every call carries an identity token for it.
"""

import os
import time

import httpx
import pymupdf
import streamlit as st

from docagent.gcp import identity_token

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
# On Cloud Run the API stops when nobody uses it: the first call waits for it to start
# (the container, then the data and the model: about a minute).
ON_CLOUD_RUN = bool(os.environ.get("K_SERVICE"))
HEALTH_TIMEOUT_S = 90 if ON_CLOUD_RUN else 5
EXAMPLES = [
    "Quelles sont les émissions du secteur de la production d'énergie en 2024 ?",
    "How much did global mean sea level rise between 1901 and 2018?",
    "Combien d'argent sera investi dans les centres de données en 2025 ?",
    "Qui a gagné la Coupe du monde de football 2018 ?",
]

AGENT_EXAMPLES = {
    "Summary + comparison": (
        "Résume le rapport du HCC puis compare-le avec le GIEC sur les puits de carbone"
    ),
    "Translate then ask": (
        "Traduis le résumé pour décideurs de l'IPCC en français puis dis-moi de combien "
        "la température a augmenté depuis 1850"
    ),
    "Question": "Que dit le HCC sur les émissions des transports en 2024 ?",
    "Out of scope": "Quel temps fera-t-il demain ?",
}
ROUTES = {
    "question": "question",
    "summarize": "summary",
    "compare": "comparison",
    "translate": "translation",
    "clarify": "clarification",
    "out_of_scope": "out of scope",
}

st.set_page_config(page_title="DocAgent", page_icon="📄", layout="wide")
st.title("DocAgent · climate & energy reports")


@st.cache_data(ttl=45 * 60, show_spinner=False)  # a token is valid one hour
def cached_identity_token(audience: str) -> str:
    return identity_token(audience)


def api(method: str, path: str, **kwargs) -> httpx.Response:
    """One call to the API. On Cloud Run (K_SERVICE is set) the call carries an
    identity token for the API, signed for the front's service account; locally, none.
    Raises ``httpx.HTTPError``."""
    headers = {}
    if ON_CLOUD_RUN:
        headers["Authorization"] = f"Bearer {cached_identity_token(API_URL)}"
    return httpx.request(method, f"{API_URL}{path}", headers=headers, **kwargs)


def error_detail(response: httpx.Response) -> str:
    """The API's error message. Google's own refusals (403 before the API) are HTML."""
    try:
        return str(response.json().get("detail"))
    except ValueError:
        return response.reason_phrase or "error"


def api_health() -> tuple[dict | None, str]:
    """GET /health and, when it fails, why: unreachable, or refused (a 403 here means
    this front end may not call the API)."""
    try:
        response = api("GET", "/health", timeout=HEALTH_TIMEOUT_S)
    except httpx.HTTPError as exc:
        return None, f"unreachable ({type(exc).__name__})"
    if response.status_code != 200:
        return None, f"refused (HTTP {response.status_code}: {error_detail(response)})"
    return response.json(), ""


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
            response = api(
                "POST", "/v1/detect", files={"file": (name, pdf, "application/pdf")}, timeout=30
            )
            st.session_state[key] = response.json() if response.status_code == 200 else None
        except httpx.HTTPError:
            st.session_state[key] = None
    return st.session_state[key]


with st.sidebar:
    if ON_CLOUD_RUN:
        with st.spinner("Waking the API up: up to a minute after a quiet period…"):
            health, problem = api_health()
    else:
        health, problem = api_health()
    if health is None:
        st.error(f"API {problem} at {API_URL}")
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
        response = api(
            "POST",
            "/v1/translate",
            files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
            data={"source_lang": source, "target_lang": target},
            timeout=60,
        )
        if response.status_code != 202:
            st.error(f"Upload refused ({response.status_code}): {error_detail(response)}")
            return
        job_id = response.json()["job_id"]

        progress = st.progress(0.0, text="Queued…")
        while True:
            job = api("GET", f"/v1/jobs/{job_id}", timeout=10).json()
            done, total = job["pages_done"], job["pages_total"]
            progress.progress(done / total, text=f"{job['status']} · page {done}/{total}")
            if job["status"] in ("done", "failed"):
                break
            time.sleep(POLL_SECONDS)

        if job["status"] == "failed":
            st.error(f"Translation failed: {job['error']}")
            return
        result = api("GET", f"/v1/jobs/{job_id}/download", timeout=60).content
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
    """POST /v1/ask. The API loads the search index at start-up (about a minute)."""
    start = time.perf_counter()
    try:
        response = api("POST", "/v1/ask", json={"question": question, "top_k": top_k}, timeout=180)
    except httpx.HTTPError as exc:
        return {"error": f"API unreachable: {exc}"}
    if response.status_code != 200:
        return {"error": f"{response.status_code}: {error_detail(response)}"}
    return response.json() | {"elapsed_s": time.perf_counter() - start}


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
    t = result.get("timings", {})
    timing = f"search {t.get('search_s', 0):.1f} s · model {t.get('llm_s', 0):.1f} s"
    if t.get("llm_requests", 1) > 1:
        timing += f" ({int(t['llm_requests'])} requests: rate limit or invalid answer)"
    st.caption(
        f"{len(result['retrieved'])} passages given to {result['model']} "
        f"· prompt {result.get('prompt_version', '?')} · {timing} "
        f"· total {result.get('elapsed_s', 0):.1f} s"
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


def agent_call(method: str, path: str, **kwargs) -> dict:
    """An agent endpoint. A refused call comes back as {"api_error": ...}: a run has
    its own "error" field, set when the run itself failed."""
    try:
        response = api(method, path, timeout=30, **kwargs)
    except httpx.HTTPError as exc:
        return {"api_error": f"API unreachable: {exc}"}
    if response.status_code >= 400:
        return {"api_error": f"{response.status_code}: {error_detail(response)}"}
    return response.json()


def as_markdown(text: str) -> str:
    """The agent's plain text as Markdown: a line after a list ("1 point removed…")
    starts its own paragraph instead of joining the last item."""
    lines: list[str] = []
    for line in text.split("\n"):
        if lines and lines[-1].lstrip().startswith("- ") and not line.lstrip().startswith("- "):
            lines.append("")
        lines.append(line)
    return "\n".join(lines)


def plan_line(i: int, plan: dict, run: dict) -> str:
    """'1. summary · hcc-ra-2025 — done' (or running / waiting)."""
    what = ROUTES.get(plan["route"], plan["route"])
    about = ", ".join(plan["doc_ids"]) or plan.get("path") or plan.get("question") or ""
    done = len(run["sections"])
    if i < done:
        state = "✅ done"
    elif i == done and run["status"] in ("queued", "running"):
        state = "⏳ running…"
    elif i == done and run["status"] == "needs_confirmation":
        state = "❓ waiting for your answer"
    elif run["status"] == "done":
        state = "⏭️ not done"
    else:
        state = "· waiting"
    return f"{i + 1}. **{what}**" + (f" · {about}" if about else "") + f" — {state}"


def show_run(run: dict) -> None:
    plans = run["plans"]
    if plans:
        st.markdown("**Plan**  \n" + "  \n".join(plan_line(i, p, run) for i, p in enumerate(plans)))
    elif run["status"] in ("queued", "running"):
        st.info("Reading your request…")

    if run["status"] == "done" and (len(plans) <= 1 or not run["sections"]):
        st.markdown(as_markdown(run["message"]))  # one step, or the request itself failed
    else:
        for i, section in enumerate(run["sections"]):
            label = ROUTES.get(plans[i]["route"], "") if i < len(plans) else ""
            st.markdown(f"#### Step {i + 1}/{len(plans)}: {label}")
            st.markdown(as_markdown(section))
        if run["status"] == "done" and len(run["sections"]) < len(plans):
            st.markdown(run["message"].rsplit("\n\n", 1)[-1])  # why it stopped
    if run["status"] == "failed":
        st.error(f"The run failed: {run['error']}")

    with st.expander("Details (trajectory)"):
        cols = st.columns(4)
        cols[0].metric("Duration", f"{run['duration_s']:.0f} s")
        cols[1].metric("Supervisor calls", run["supervisor_calls"])
        cost = run["cost_usd"]
        cols[2].metric("Supervisor cost", "—" if cost is None else f"${cost:.4f}")
        cols[3].metric("Steps", f"{len(run['sections'])}/{len(plans) or '?'}")

        def cell(value) -> str:
            return "" if value is None else str(value)

        rows = []
        for n in run["nodes"]:
            not_reviewed = n["node"] == "judge" and n["judged"] is False
            rows.append(
                {
                    "node": n["node"],
                    "seconds": n["duration_s"],
                    "tool": n["tool"] or "",
                    "passages": cell(n["passages"]),
                    "points": cell(n["items"]),
                    "kept by judge": "not reviewed" if not_reviewed else cell(n["kept"]),
                    "removed by judge": "not reviewed" if not_reviewed else cell(n["removed"]),
                    "rejected by code": cell(len(n["rejected"] or []) or None),
                }
            )
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch")
        rejected = [r for n in run["nodes"] for r in (n["rejected"] or [])]
        if rejected:
            st.caption("Rejected by the code checks:  \n" + "  \n".join(f"- {r}" for r in rejected))
        for n in run["nodes"]:
            if n.get("refusal"):
                st.caption(f"Question not answered: {n['refusal']}")
        if run.get("trajectory_file"):
            st.caption(f"Trajectory saved: {run['trajectory_file']}")


def send_agent_request() -> None:
    request = st.session_state.get("agent_request", "").strip()
    if len(request) < 2:
        return
    run = agent_call("POST", "/v1/agent", json={"request": request})
    if "api_error" in run:
        st.session_state["agent_error"] = run["api_error"]
    else:
        st.session_state["agent_run"] = run["run_id"]


def assistant_tab(health: dict | None) -> None:
    st.write(
        "Ask the agent anything about the reports: a question, a summary, a comparison, a "
        "translation, or several in a row (« résume … puis compare … »). Each step is "
        "checked by code, then reviewed by the supervisor before it is shown."
    )
    cols = st.columns(len(AGENT_EXAMPLES))
    for col, (label, example) in zip(cols, AGENT_EXAMPLES.items(), strict=True):
        if col.button(label, width="stretch", key=f"agent-example:{label}"):
            st.session_state["agent_request"] = example
    with st.form("agent"):
        st.text_area("Request", key="agent_request", max_chars=1000, height=80)
        # Sent from the button's callback: it runs once per click. Read from the button's
        # return value instead, the request was sent twice, the page reloading itself
        # (to follow the run) while the click was still counted.
        st.form_submit_button(
            "Send", type="primary", disabled=health is None, on_click=send_agent_request
        )
    if error := st.session_state.pop("agent_error", None):
        st.error(error)

    run_id = st.session_state.get("agent_run")
    if not run_id:
        return
    run = agent_call("GET", f"/v1/agent/{run_id}")
    if "api_error" in run:
        st.error(run["api_error"])
        return
    show_run(run)

    if run["status"] == "needs_confirmation":
        st.warning(f"The agent asks: {run['question']}")
        yes, no = st.columns(2)
        for col, answer in ((yes, "oui"), (no, "non")):
            if col.button(answer.capitalize(), width="stretch", key=f"resume:{answer}"):
                resumed = agent_call("POST", f"/v1/agent/{run_id}/resume", json={"answer": answer})
                if "api_error" in resumed:
                    st.error(resumed["api_error"])
                else:
                    st.rerun()
    elif run["status"] in ("queued", "running"):
        time.sleep(POLL_SECONDS)  # follow the run: the page reloads until it ends or asks
        st.rerun()


translate, questions, assistant = st.tabs(["Translate a PDF", "Ask the reports", "Assistant"])
with translate:
    translation_tab(health)
with questions:
    questions_tab(health)
with assistant:
    assistant_tab(health)
