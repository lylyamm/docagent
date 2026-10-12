"""The Translation route of the agent, as a LangGraph graph.

    START ─► inspect ──(error)──────────────────────────────► finish ─► END
                │ ──(more than AGENT_CONFIRM_PAGES pages)──► confirm ──(no)──┘
                │                                              │ (yes)
                └──(small enough)────────────────────────────► translate ─► finish

- ``inspect``: page count and language of the PDF (``Toolbox.pdf_info``), before
  starting a translation that can take minutes. Refuses a missing or unreadable
  file, and a document already in the target language.
- ``confirm``: ``interrupt()``. The run stops and returns the question; the
  checkpointer keeps the state; ``Command(resume=True)`` continues from here.
- ``translate``: the layout-preserving translator of weeks 1-2, unchanged
  (``Toolbox.translate_pdf``).
- ``finish``: the message for the user (done, cancelled, or what went wrong).

The decisions (which branch) are taken by code, from the state: no model call
chooses the path. The only model calls are those of the translator itself.
"""

from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .tools import PdfInfo, Toolbox, ToolError, TranslationResult

YES = {"y", "yes", "o", "oui", "ok", "true", "1"}
LANGUAGE_NAMES_FR = {"fr": "français", "en": "anglais"}


class TranslationState(TypedDict, total=False):
    # input
    path: str  # the PDF to translate
    target_lang: str  # "fr" or "en"
    source_lang: str | None  # None: detected
    # filled by the nodes
    info: PdfInfo  # inspect
    confirmed: bool  # confirm (only for long PDFs)
    result: TranslationResult  # translate
    error: str  # any node: what went wrong, for the user
    message: str  # finish: what the user reads


def is_yes(answer: Any) -> bool:
    """The resume value: True/False, or a typed answer ("oui", "yes", "o")."""
    if isinstance(answer, bool):
        return answer
    return str(answer).strip().lower() in YES


def build_translation_graph(tools: Toolbox, checkpointer=None):
    """The compiled graph. A checkpointer is needed for the confirmation pause."""
    limit = tools.settings.agent_confirm_pages

    # -- nodes: each one reads the state and returns only what it changes --------

    def inspect(state: TranslationState) -> TranslationState:
        try:
            info = tools.pdf_info(state["path"])
        except ToolError as exc:
            return {"error": str(exc)}
        target = state.get("target_lang", "fr")
        source = state.get("source_lang") or info.language
        if source == target:
            name = LANGUAGE_NAMES_FR.get(target, target)
            return {"info": info, "error": f"{Path(info.path).name} est déjà en {name}"}
        return {"info": info}

    def confirm(state: TranslationState) -> TranslationState:
        info = state["info"]
        # Everything above this line runs again when the graph resumes: keep it
        # free of side effects. interrupt() returns the value given to resume.
        answer = interrupt(
            {
                "question": (
                    f"Ce PDF fait {info.pages} pages (plus de {limit}) : la traduction "
                    "peut prendre longtemps. Je la lance ?"
                ),
                "path": info.path,
                "pages": info.pages,
            }
        )
        return {"confirmed": is_yes(answer)}

    def translate(state: TranslationState) -> TranslationState:
        try:
            result = tools.translate_pdf(
                state["path"], state.get("target_lang", "fr"), state.get("source_lang")
            )
        except ToolError as exc:
            return {"error": str(exc)}
        return {"result": result}

    def finish(state: TranslationState) -> TranslationState:
        if state.get("error"):
            return {"message": f"Je n'ai pas pu traduire ce document : {state['error']}."}
        result = state.get("result")
        if result is None:  # the user said no
            return {"message": "Traduction annulée : rien n'a été traduit."}
        checks = (
            "mise en page vérifiée"
            if result.checks_ok
            else "attention, des contrôles de mise en page ont échoué"
        )
        overflow = f", {result.overflow} zone(s) trop longue(s)" if result.overflow else ""
        return {
            "message": (
                f"Traduction terminée : {result.output} ({result.pages} pages, "
                f"{result.source_lang} → {result.target_lang}, {result.duration_s} s, "
                f"{checks}{overflow})."
            )
        }

    # -- routing: plain functions of the state, chosen by code ----------------------

    def after_inspect(state: TranslationState) -> str:
        if state.get("error"):
            return "finish"
        if state["info"].pages > limit:
            return "confirm"
        return "translate"

    def after_confirm(state: TranslationState) -> str:
        return "translate" if state.get("confirmed") else "finish"

    graph = StateGraph(TranslationState)
    graph.add_node("inspect", inspect)
    graph.add_node("confirm", confirm)
    graph.add_node("translate", translate)
    graph.add_node("finish", finish)

    graph.add_edge(START, "inspect")
    graph.add_conditional_edges("inspect", after_inspect, ["confirm", "translate", "finish"])
    graph.add_conditional_edges("confirm", after_confirm, ["translate", "finish"])
    graph.add_edge("translate", "finish")
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer)
