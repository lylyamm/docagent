"""The supervisor (agent/supervisor.py): scripted models, fake translator, no network.

Three fake models, as in the real agent: the supervisor's (router + judge), the
answerer's and the tools' (summary, comparison).
"""

import json
from pathlib import Path

import httpx
import pytest
from langgraph.types import Command
from test_agent_tools import DATA, SEA, make_pdf, settings
from test_answer import TRANSPORT, chat, searcher

from docagent.agent import Toolbox
from docagent.agent.checkpoint import memory_checkpointer
from docagent.agent.supervisor import LLMRoute, agent_chat_client, build_supervisor, check_plan
from docagent.chat import CallBudgetExceeded
from docagent.config import Settings
from docagent.rag import Answerer
from docagent.translate import FakeTranslator

CLIENTS: dict = {}
CORPUS = [
    {"id": f"d{i}", "file": f"d{i}.pdf", "title": f"Report {i}", "org": "X", "year": 2025}
    for i in (1, 2, 3)
]


def agent(tmp_path: Path, router: dict, *, judge=(), answer=(), tool=(), max_calls=None, **values):
    """The supervisor with scripted replies; returns (graph, sent requests by model)."""
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps(CORPUS), encoding="utf-8")
    agent_chat, agent_sent = chat(router, *judge)
    agent_chat.max_requests = max_calls
    CLIENTS["agent"] = agent_chat  # the last supervisor client, for test_trajectory
    answer_chat, answer_sent = chat(*answer)
    tool_chat, tool_sent = chat(*tool)
    tools = Toolbox(
        settings(corpus_path=corpus, agent_confirm_pages=2, rag_raw_dir=tmp_path, **values),
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
    judge_system = sent["agent"][1]["messages"][0]["content"]
    assert 'Write "problem" in French' in judge_system
    assert "never against what you know" in judge_system  # judge-v3: reports newer than it
    assert "never ask which" in system  # router-v2
    assert "at most 5" in system and '{"steps": [' in system  # router-v3
    assert "Never ask which version" in system  # router-v4: a report in two versions


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
    assert "- La mer a monté de 0.20 m. (Report 2, p. 12)" in message  # no [n]: unseen excerpts
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
    assert (  # one line per report, then the pages; no [n] markers (unseen excerpt numbers)
        "- **Chiffres**\n    - Report 2: 0.20 m.\n    - Report 3: 580 milliards.\n"
        "    - (Report 2, p. 12; Report 3, p. 13)"
    ) in state["message"]


def test_the_judge_reads_what_the_reports_say_not_the_note_on_what_is_missing(tmp_path):
    """An "only_in" point ends with a note written by code ("Rien sur ce point dans le
    passage consulté de Report 2."). Judged, it was removed as "in no passage"."""
    side_b = {"report": "B", "says": "580 milliards [2].", "evidence": [{"n": 2, "quote": DATA}]}
    reply = {"points": [{"kind": "only_in", "aspect": "Centres de données", "sides": [side_b]}]}
    graph, sent = agent(
        tmp_path,
        {"route": "compare", "doc_ids": ["d2", "d3"], "topic": "centres de données"},
        judge=[FAITHFUL],
        tool=[reply],
    )
    state, _ = ask(graph, "Compare les rapports 2 et 3")
    judged = sent["agent"][1]["messages"][1]["content"]
    assert "Statement: Centres de données. Report 3: 580 milliards [2]." in judged
    assert "Rien sur ce point" not in judged
    assert "    - Rien sur ce point dans le passage consulté de Report 2." in state["message"]


def test_a_page_cited_twice_is_shown_once():
    from docagent.agent.supervisor import _where

    class Source:
        def __init__(self, page):
            self.title, self.page = "HCC", page

    assert _where([Source(57), Source(57), Source(26)]) == "HCC, p. 57; HCC, p. 26"


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
    assert "- La mer a monté de 0.20 m. (Report 2, p. 12)" in state["message"]
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


def test_the_cost_guard_stops_before_a_request_it_could_not_review(tmp_path):
    router = {"route": "question"}
    graph, sent = agent(tmp_path, router, judge=[FAITHFUL], answer=[ANSWER], max_calls=3)
    state, _ = ask(graph, "Quelle part des émissions pour les transports en 2024 ?", thread="a")
    assert state["message"].startswith("Les transports représentent 34 %")  # 2 calls used

    state, _ = ask(graph, "Et pour l'industrie ?", thread="b")  # 2 + 2 > 3: not started
    assert state["message"].startswith("J'ai atteint la limite de 3 appels au superviseur")
    assert len(sent["agent"]) == 2 and len(sent["answer"]) == 1


def test_a_client_never_sends_more_than_its_limit():
    client, sent = chat({"route": "question"})
    client.max_requests = 0
    with pytest.raises(CallBudgetExceeded):
        client.complete_json([{"role": "user", "content": "x"}], LLMRoute)
    assert sent == []  # refused before sending, not retried


def test_vertex_endpoint_and_token():
    settings = Settings(
        _env_file=None,
        agent_provider="vertex",
        agent_model="google/gemini-3.8-flash",
        vertex_project="projetlilia",
        agent_max_calls=10,
    )
    root = "https://aiplatform.googleapis.com/v1/projects/projetlilia/locations/global"
    assert settings.vertex_base_url == f"{root}/endpoints/openapi"
    paris = settings.model_copy(update={"vertex_location": "europe-west9"})
    assert paris.vertex_base_url.startswith(
        "https://europe-west9-aiplatform.googleapis.com/v1/projects/projetlilia/locations/europe-west9/"
    )

    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), request.headers["authorization"], request.content))
        content = json.dumps({"route": "out_of_scope"})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    tokens = iter(["tok-1", "tok-2"])
    client = agent_chat_client(settings, token=lambda: next(tokens))  # no Mistral key needed
    assert client.client.timeout.read == settings.vertex_timeout_s  # not the translator's 120 s
    client.client = httpx.Client(transport=httpx.MockTransport(handler))
    client.sleep = lambda s: None
    assert (client.model, client.max_requests) == ("google/gemini-3.8-flash", 10)
    for _ in range(2):
        client.complete_json([{"role": "user", "content": "x"}], LLMRoute)
    assert [url for url, _, _ in seen] == [f"{root}/endpoints/openapi/chat/completions"] * 2
    assert [auth for _, auth, _ in seen] == ["Bearer tok-1", "Bearer tok-2"]  # asked each time
    assert json.loads(seen[0][2])["model"] == "google/gemini-3.8-flash"

    with pytest.raises(RuntimeError, match="AGENT_MODEL is required"):
        agent_chat_client(settings.model_copy(update={"agent_model": None}), token=lambda: "t")


def test_the_tools_model_can_be_on_vertex_too():
    """TOOLS_PROVIDER=vertex: answers, summaries and comparisons on TOOLS_MODEL (Gemini on
    Vertex AI); the translator keeps LLM_*. Default: the tools' client is LLM_*."""
    from docagent.chat import tools_chat_client

    settings = Settings(
        _env_file=None,
        mistral_api_key="k",
        llm_model="open-mistral-nemo",
        tools_provider="vertex",
        tools_model="google/gemini-3.5-flash-lite",
        vertex_project="projetlilia",
    )
    client = tools_chat_client(settings, token=lambda: "tok")
    assert client.model == "google/gemini-3.5-flash-lite"
    assert client.base_url.endswith("/projects/projetlilia/locations/global/endpoints/openapi")
    assert (client.temperature, client.max_requests) == (0.0, None)
    assert client.rate_limit_retries == settings.agent_rate_limit_retries  # busy model: wait
    assert client.client.timeout.read == settings.vertex_timeout_s

    same = tools_chat_client(settings.model_copy(update={"tools_provider": "same"}))
    assert (same.model, same.rate_limit_retries) == ("open-mistral-nemo", 0)
    assert same.client.timeout.read == settings.llm_timeout_s
    with pytest.raises(RuntimeError, match="TOOLS_MODEL is required"):
        tools_chat_client(settings.model_copy(update={"tools_model": None}), token=lambda: "t")


# --- several steps --------------------------------------------------------------------

POINT = {"text": "La mer a monté de 0.20 m [1].", "evidence": [{"n": 1, "quote": SEA}]}


def test_two_steps_run_in_order_each_one_reviewed(tmp_path):
    router = {
        "steps": [
            {"route": "summarize", "doc_ids": ["d2"]},
            {"route": "question", "question": "Quelle part des émissions pour les transports ?"},
        ]
    }
    graph, sent = agent(
        tmp_path,
        router,
        judge=[FAITHFUL, FAITHFUL],
        answer=[ANSWER],
        tool=[{"points": [POINT]}],
    )
    state, _ = ask(graph, "Résume le rapport 2 puis dis-moi la part des transports")

    assert [p.route for p in state["plans"]] == ["summarize", "question"]
    message = state["message"]
    assert message.startswith("Étape 1/2 : résumé\nRésumé de Report 2")
    assert "\n\nÉtape 2/2 : question\nLes transports représentent 34 %" in message
    assert len(sent["agent"]) == 3  # one plan, one review per step
    second_review = sent["agent"][2]["messages"][1]["content"]
    assert "Les transports" in second_review and "La mer" not in second_review  # clean slate


def test_a_step_that_fails_stops_the_chain(tmp_path):
    router = {"steps": [{"route": "summarize", "doc_ids": ["d9"]}, {"route": "question"}]}
    graph, sent = agent(tmp_path, router)
    state, _ = ask(graph, "Résume le rapport 9 puis dis-moi ce qu'il prévoit")
    message = state["message"]
    assert message.startswith("Étape 1/2 : précision\nJe ne connais pas ce rapport : d9.")
    assert message.endswith("Je me suis arrêté là : étape 2 (question) non faite(s).")
    assert sent["tool"] == [] and sent["answer"] == [] and len(sent["agent"]) == 1


def test_an_off_topic_step_after_a_step_that_worked(tmp_path):
    router = {"steps": [{"route": "summarize", "doc_ids": ["d2"]}, {"route": "out_of_scope"}]}
    graph, sent = agent(tmp_path, router, judge=[FAITHFUL], tool=[{"points": [POINT]}])
    state, _ = ask(graph, "Résume le rapport 2 et dis-moi quel temps il fera demain")
    message = state["message"]
    assert message.startswith("Étape 1/2 : résumé\nRésumé de Report 2")
    assert "\n\nÉtape 2/2 : hors sujet\nJe peux traduire un PDF" in message
    assert len(sent["agent"]) == 2  # plan + one review


def test_steps_beyond_the_limit_are_cut_by_code(tmp_path):
    step = {"route": "question", "question": "Quelle part des émissions pour les transports ?"}
    graph, sent = agent(
        tmp_path,
        {"steps": [step, step, step]},
        judge=[FAITHFUL, FAITHFUL],
        answer=[ANSWER, ANSWER],
        agent_max_steps=2,
    )
    state, _ = ask(graph, "Trois questions")
    assert len(state["plans"]) == 2 and len(sent["answer"]) == 2
    assert state["message"].endswith(
        "Je fais au plus 2 étapes par demande : 1 étape(s) en plus ignorée(s)."
    )


def test_translate_a_report_by_its_id_then_ask(tmp_path):
    make_pdf(tmp_path / "d1.pdf")  # the corpus PDF, in RAG_RAW_DIR
    router = {
        "steps": [
            {"route": "translate", "doc_ids": ["d1"], "target_lang": "fr"},
            {"route": "question", "question": "Quelle part des émissions pour les transports ?"},
        ]
    }
    graph, _ = agent(tmp_path, router, judge=[FAITHFUL], answer=[ANSWER])
    state, _ = ask(graph, "Traduis le rapport 1 puis dis-moi ce qu'il prévoit")
    assert state["plans"][0].path == str(tmp_path / "d1.pdf")
    assert "Étape 1/2 : traduction\nTraduction terminée" in state["message"]
    assert "Étape 2/2 : question\nLes transports" in state["message"]
    assert (tmp_path / "out" / "d1_fr.pdf").exists()


def test_a_cancelled_translation_stops_the_chain(tmp_path):
    path = str(make_pdf(tmp_path / "big.pdf", pages=3))  # more than AGENT_CONFIRM_PAGES
    router = {
        "steps": [
            {"route": "translate", "path": path},
            {"route": "question", "question": "Que dit ce rapport ?"},
        ]
    }
    graph, sent = agent(tmp_path, router)
    state, config = ask(graph, f"Traduis {path} puis résume-le")
    assert "__interrupt__" in state  # the pause inside step 1 pauses the whole chain
    state = graph.invoke(Command(resume="non"), config)
    assert "Traduction annulée" in state["message"]
    assert state["message"].endswith("Je me suis arrêté là : étape 2 (question) non faite(s).")
    assert sent["answer"] == []


def test_the_cost_guard_stops_between_steps(tmp_path):
    step = {"route": "question", "question": "Quelle part des émissions pour les transports ?"}
    graph, sent = agent(
        tmp_path, {"steps": [step, step]}, judge=[FAITHFUL], answer=[ANSWER], max_calls=2
    )
    state, _ = ask(graph, "Deux questions")
    assert len(sent["agent"]) == 2 and len(sent["answer"]) == 1  # step 2 not started
    assert "Je me suis arrêté là : étape 2 (question) non faite(s)." in state["message"]
    assert state["message"].endswith("Relance une session ou augmente AGENT_MAX_CALLS.")


def test_router_replies_are_read_leniently():
    from docagent.agent.supervisor import LLMSteps

    assert [s.route for s in LLMSteps.model_validate({"route": "question"}).steps] == ["question"]
    assert LLMSteps.model_validate({"steps": "nope"}).steps == []
    assert len(LLMSteps.model_validate({"steps": [{"route": "x"}, 3]}).steps) == 1
