"""The supervisor (agent/supervisor.py): scripted models, fake translator, no network.

Three fake models, as in the real agent: the supervisor's (router + judge), the
answerer's and the tools' (summary, comparison).
"""

import json
from pathlib import Path

from langgraph.types import Command
from test_agent_tools import DATA, SEA, make_pdf, settings
from test_answer import TRANSPORT, chat, searcher

from docagent.agent import Toolbox
from docagent.agent.checkpoint import memory_checkpointer
from docagent.agent.supervisor import LLMRoute, build_supervisor, check_plan
from docagent.rag import Answerer
from docagent.translate import FakeTranslator

CORPUS = [
    {"id": f"d{i}", "file": f"d{i}.pdf", "title": f"Report {i}", "org": "X", "year": 2025}
    for i in (1, 2, 3)
]


def agent(tmp_path: Path, router: dict, *, judge=(), answer=(), tool=()):
    """The supervisor with scripted replies; returns (graph, sent requests by model)."""
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps(CORPUS), encoding="utf-8")
    agent_chat, agent_sent = chat(router, *judge)
    answer_chat, answer_sent = chat(*answer)
    tool_chat, tool_sent = chat(*tool)
    tools = Toolbox(
        settings(corpus_path=corpus, agent_confirm_pages=2),
        answerer=Answerer(searcher(), answer_chat, top_k=1),
        make_translator=FakeTranslator,
        chat=tool_chat,
        output_dir=tmp_path / "out",
    )
    graph = build_supervisor(tools, agent_chat, checkpointer=memory_checkpointer())
    return graph, {"agent": agent_sent, "answer": answer_sent, "tool": tool_sent}


def ask(graph, request: str, thread: str = "t"):
    config = {"configurable": {"thread_id": thread}}
    return graph.invoke({"request": request}, config), config


ANSWER = {
    "answerable": True,
    "answer": "Les transports représentent 34 % des émissions en 2024 [1].",
    "evidence": [{"n": 1, "quote": TRANSPORT}],
}
FAITHFUL = {"items": [{"n": 1, "faithful": True, "problem": ""}]}


def test_a_question_is_answered_with_sources_then_reviewed(tmp_path):
    router = {"route": "question", "question": "Quelle part des émissions pour les transports ?"}
    graph, sent = agent(tmp_path, router, judge=[FAITHFUL], answer=[ANSWER])
    state, _ = ask(graph, "Quelle part des émissions pour les transports en 2024 ?")

    assert state["plan"].route == "question"
    assert state["message"].startswith("Les transports représentent 34 %")
    assert "Sources :\n[1] Report 1, p. 11" in state["message"]
    system = sent["agent"][0]["messages"][0]["content"]
    assert "- d2: Report 2 (X, 2025, ?)" in system  # the catalog given to the router
    judged = sent["agent"][1]["messages"][1]["content"]
    assert "[1] Statement: Les transports" in judged and "Quote: Report 1, p. 11" in judged
    assert "Passage: Report 1, p. 11: Le secteur des transports" in judged  # whole passage
    assert 'Write "problem" in French' in sent["agent"][1]["messages"][0]["content"]
    assert "never ask which" in system  # router-v2


def test_an_answer_the_judge_finds_unfaithful_is_not_shown(tmp_path):
    verdict = {"items": [{"n": 1, "faithful": False, "problem": "34 % concerne un autre secteur"}]}
    graph, _ = agent(tmp_path, {"route": "question"}, judge=[verdict], answer=[ANSWER])
    state, _ = ask(graph, "Quelle part des émissions pour les transports en 2024 ?")
    assert "Je préfère ne pas l'afficher" in state["message"]
    assert "34 % concerne un autre secteur" in state["message"]


def test_a_summary_keeps_only_the_points_the_judge_accepts(tmp_path):
    point = {"text": "La mer a monté de 0.20 m [1].", "evidence": [{"n": 1, "quote": SEA}]}
    other = {
        "text": "La hausse vient des glaciers, 0.20 m [1].",
        "evidence": [{"n": 1, "quote": SEA}],
    }
    verdicts = {
        "items": [
            {"n": 1, "faithful": True, "problem": ""},
            {"n": 2, "faithful": False, "problem": "la citation ne parle pas des glaciers"},
        ]
    }
    graph, sent = agent(
        tmp_path,
        {"route": "summarize", "doc_ids": ["d2"], "topic": "niveau de la mer"},
        judge=[verdicts],
        tool=[{"points": [point, other]}],
    )
    state, _ = ask(graph, "Résume le rapport 2 sur le niveau de la mer")
    message = state["message"]
    assert message.startswith("Résumé de Report 2 (à partir des 1 passages")
    assert "- La mer a monté de 0.20 m [1]. (Report 2, p. 12)" in message
    assert "glaciers" not in message.split("\n")[1]
    assert "1 point(s) retiré(s) à la relecture : la citation ne parle pas des glaciers" in message


def test_a_comparison_goes_through_the_same_review(tmp_path):
    side_a = {"report": "A", "says": "0.20 m [1].", "evidence": [{"n": 1, "quote": SEA}]}
    side_b = {"report": "B", "says": "580 milliards [2].", "evidence": [{"n": 2, "quote": DATA}]}
    reply = {"points": [{"kind": "difference", "aspect": "Chiffres", "sides": [side_a, side_b]}]}
    graph, _ = agent(
        tmp_path,
        {"route": "compare", "doc_ids": ["d2", "d3"], "topic": "chiffres clés"},
        judge=[FAITHFUL],
        tool=[reply],
    )
    state, _ = ask(graph, "Compare les rapports 2 et 3")
    assert state["message"].startswith("Comparaison sur « chiffres clés » :")
    assert "Report 2: 0.20 m [1]. Report 3: 580 milliards [2]." in state["message"]


def test_unknown_reports_and_off_topic_requests_make_no_tool_call(tmp_path):
    graph, sent = agent(tmp_path, {"route": "compare", "doc_ids": ["d2", "d9"]})
    state, _ = ask(graph, "Compare le rapport 2 et le rapport 9", thread="a")
    assert state["message"].startswith("Je ne connais pas ce rapport : d9.")
    assert "Rapports disponibles :\n- d1: Report 1" in state["message"]
    assert sent["tool"] == [] and len(sent["agent"]) == 1  # no tool, no judge

    graph, sent = agent(tmp_path, {"route": "out_of_scope"})
    state, _ = ask(graph, "Quel temps fera-t-il demain ?", thread="b")
    assert state["message"].startswith("Je peux traduire un PDF")
    assert sent["tool"] == [] and len(sent["agent"]) == 1


def test_a_translation_pause_pauses_the_whole_agent(tmp_path):
    path = str(make_pdf(tmp_path / "big.pdf", pages=3))
    graph, _ = agent(tmp_path, {"route": "translate", "path": path, "target_lang": "fr"})
    state, config = ask(graph, f"Traduis {path} en français")

    question = state["__interrupt__"][0].value
    assert question["pages"] == 3  # the sub-graph's question reaches the caller
    state = graph.invoke(Command(resume="oui"), config)
    assert state["message"].startswith("Traduction terminée")
    assert (tmp_path / "out" / "big_fr.pdf").exists()


def test_when_the_judge_fails_the_checked_points_stay_marked_not_reviewed(tmp_path):
    point = {"text": "La mer a monté de 0.20 m [1].", "evidence": [{"n": 1, "quote": SEA}]}
    graph, _ = agent(
        tmp_path,
        {"route": "summarize", "doc_ids": ["d2"]},
        judge=["not json", "still not", "nope"],  # 1 try + 2 retries
        tool=[{"points": [point]}],
    )
    state, _ = ask(graph, "Résume le rapport 2")
    assert "- La mer a monté de 0.20 m [1]." in state["message"]
    assert state["message"].endswith("(Réponse non relue : le superviseur n'a pas pu la vérifier.)")


def test_the_plan_is_checked_by_code():
    known = {"d1", "d2", "d3"}

    def plan(**raw):
        return check_plan(LLMRoute(**raw), "la demande", known, "fr", "m")

    assert plan(route="summarize", doc_ids=["d1", "d2"]).route == "clarify"
    assert (
        plan(route="compare", doc_ids=["d1"]).reply == "Quels rapports veux-tu comparer (2 ou 3) ?"
    )
    assert plan(route="translate").route == "clarify"
    assert plan(route="dance").route == "clarify"
    assert plan(route="question").question == "la demande"  # empty question: the request
    translate = plan(route="translate", path='"C:\\docs\\r.pdf"', target_lang="de")
    assert (translate.path, translate.target_lang) == ("C:\\docs\\r.pdf", "fr")
    assert LLMRoute.model_validate(
        {"route": "summarize", "doc_ids": "d1", "topic": None}
    ).doc_ids == ["d1"]


def test_the_language_of_short_requests():
    from docagent.agent.supervisor import request_language

    assert request_language("Résume le WEO") == "fr"
    assert request_language("Résume le rapport 2") == "fr"
    assert request_language("Summarise report 2") == "en"
