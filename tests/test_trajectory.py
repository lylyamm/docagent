"""Trajectories (agent/trajectory.py): what the agent did, recorded from the stream."""

from langgraph.types import Command
from test_agent_tools import make_pdf, settings
from test_supervisor import ANSWER, CLIENTS, FAITHFUL, POINT, agent

from docagent.agent.trajectory import Recorder, load

TWO_STEPS = {
    "steps": [
        {"route": "summarize", "doc_ids": ["d2"]},
        {"route": "question", "question": "Quelle part des émissions pour les transports ?"},
    ]
}
ONE_REMOVED = {"items": [{"n": 1, "faithful": True}, {"n": 2, "faithful": False, "problem": "x"}]}


def test_a_two_step_request_is_recorded_node_by_node(tmp_path):
    graph, _ = agent(
        tmp_path,
        TWO_STEPS,
        judge=[ONE_REMOVED, FAITHFUL],
        answer=[ANSWER],
        tool=[{"points": [POINT, POINT]}],
    )
    recorder = Recorder(
        "Résume le rapport 2 puis dis-moi la part des transports",
        CLIENTS["agent"],
        settings(agent_price_in_usd=1.0, agent_price_out_usd=10.0),
    )
    state = recorder.run(graph, {"request": recorder.trajectory.request})
    t = recorder.finish(state)

    assert [n.node for n in t.nodes] == [
        "understand",
        "summarize",
        "judge",
        "next_step",
        "question",
        "judge",
        "next_step",
        "respond",
    ]
    summarize, judge1, question, judge2 = t.nodes[1], t.nodes[2], t.nodes[4], t.nodes[5]
    assert (summarize.tool, summarize.items) == ("summarize_document", 2)
    assert summarize.passages >= 1 and summarize.rejected == []  # both points passed the code
    assert question.passages == 1 and question.refusal is None
    assert {"search_s", "llm_s", "llm_requests", "other_s"} <= set(question.timings)
    assert (judge1.kept, judge1.removed, judge1.judged) == (1, 1, True)
    assert (question.tool, question.ok, judge2.kept, judge2.removed) == ("ask", True, 1, 0)
    assert [p["route"] for p in t.plans] == ["summarize", "question"]
    assert (t.steps_planned, t.steps_done, t.supervisor_calls, t.pauses) == (2, 2, 3, 0)
    assert t.cost_usd == 0.0  # the scripted replies report no token usage
    assert t.message.startswith("Étape 1/2 : résumé")
    assert all(n.duration_s >= 0 for n in t.nodes) and t.duration_s >= 0

    path = t.save(tmp_path / "trajectories")
    assert path.name.endswith(f"_{t.thread_id[:8]}.json")
    assert load(path) == t


def test_a_review_that_failed_is_not_counted_as_kept(tmp_path):
    point = {"points": [POINT]}
    graph, _ = agent(
        tmp_path, {"route": "summarize", "doc_ids": ["d2"]}, judge=["x", "y", "z"], tool=[point]
    )
    recorder = Recorder("Résume le rapport 2", CLIENTS["agent"], settings())
    t = recorder.finish(recorder.run(graph, {"request": "Résume le rapport 2"}))
    judge = next(n for n in t.nodes if n.node == "judge")
    assert (judge.judged, judge.kept, judge.removed) == (False, None, None)
    assert t.supervisor_calls == 4  # the plan + 3 answers that were not valid JSON (billed)


def test_a_pause_is_counted_and_the_wait_is_not(tmp_path):
    path = str(make_pdf(tmp_path / "big.pdf", pages=3))  # more than AGENT_CONFIRM_PAGES
    graph, _ = agent(tmp_path, {"route": "translate", "path": path})
    ticks = iter([0.0, 1.0, 1.5, 100.0, 101.0, 101.2, 101.4])  # 98.5 s waiting for the user
    recorder = Recorder(f"Traduis {path}", CLIENTS["agent"], settings(), clock=lambda: next(ticks))

    state = recorder.run(graph, {"request": recorder.trajectory.request})
    assert "__interrupt__" in state
    state = recorder.run(graph, Command(resume="non"))
    t = recorder.finish(state)

    assert t.pauses == 1
    assert [n.node for n in t.nodes] == ["understand", "translate", "next_step", "respond"]
    assert t.nodes[1].ok is False  # cancelled
    assert t.duration_s == 2.9  # 1.5 before the pause + 1.4 after it, not the 98.5 s wait
    assert t.message.startswith("Traduction annulée")


def test_a_refused_question_says_why():
    from docagent.agent.trajectory import refusal_reason
    from docagent.rag import Answer

    base = {"question": "q", "answer": "Non trouvé.", "sources": [], "model": "m"}
    answered = Answer(**{**base, "answer": "34 %"}, answerable=True, retrieved=["p1"])
    assert refusal_reason(answered) is None
    cases = [
        ({"retrieved": []}, "no passage found"),
        ({"model_answerable": False}, "the model said the passages do not answer"),
        (
            {"model_answerable": True, "unsupported_numbers": ["120"]},
            "figures not in the quotes: 120",
        ),
        (
            {"model_answerable": True, "rejected_quotes": ["une phrase inventée"]},
            "quotes not found in their passage: une phrase inventée",
        ),
    ]
    for diagnostics, reason in cases:
        answer = Answer(**base, answerable=False, **({"retrieved": ["p1"]} | diagnostics))
        assert refusal_reason(answer) == reason
