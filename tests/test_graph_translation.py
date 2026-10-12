"""The Translation route (agent/graph_translation.py): fake translator, no network."""

from pathlib import Path

from langgraph.types import Command
from test_agent_tools import make_pdf, settings  # same PDF and settings helpers

from docagent.agent import Toolbox
from docagent.agent.checkpoint import memory_checkpointer
from docagent.agent.graph_translation import build_translation_graph, is_yes
from docagent.translate import FakeTranslator


def graph(tmp_path: Path, confirm_pages: int = 2):
    tools = Toolbox(
        settings(agent_confirm_pages=confirm_pages),
        make_translator=FakeTranslator,
        output_dir=tmp_path / "out",
    )
    return build_translation_graph(tools, checkpointer=memory_checkpointer())


def run(g, request: dict, thread: str = "t1"):
    config = {"configurable": {"thread_id": thread}}
    return g.invoke(request, config), config


def test_a_short_pdf_is_translated_without_asking(tmp_path):
    g = graph(tmp_path)
    state, _ = run(g, {"path": str(make_pdf(tmp_path / "r.pdf", pages=2)), "target_lang": "fr"})
    assert "__interrupt__" not in state
    assert Path(state["result"].output) == tmp_path / "out" / "r_fr.pdf"
    assert Path(state["result"].output).exists()
    assert state["message"].startswith("Traduction terminée")
    assert "en → fr" in state["message"] and "mise en page vérifiée" in state["message"]


def test_a_long_pdf_waits_for_confirmation_then_resumes(tmp_path):
    g = graph(tmp_path)
    path = str(make_pdf(tmp_path / "big.pdf", pages=3))
    state, config = run(g, {"path": path, "target_lang": "fr"})

    # The run stopped at the pause: the question is returned, nothing translated yet.
    question = state["__interrupt__"][0].value
    assert question["pages"] == 3 and "plus de 2" in question["question"]
    assert not (tmp_path / "out" / "big_fr.pdf").exists()
    assert g.get_state(config).next == ("confirm",)  # kept by the checkpointer

    state = g.invoke(Command(resume="oui"), config)
    assert state["confirmed"] is True
    assert (tmp_path / "out" / "big_fr.pdf").exists()
    assert state["message"].startswith("Traduction terminée")


def test_saying_no_translates_nothing(tmp_path):
    g = graph(tmp_path)
    state, config = run(g, {"path": str(make_pdf(tmp_path / "big.pdf", pages=3))})
    state = g.invoke(Command(resume=False), config)
    assert state["confirmed"] is False and "result" not in state
    assert state["message"] == "Traduction annulée : rien n'a été traduit."
    assert not (tmp_path / "out").exists()


def test_problems_are_explained_without_asking(tmp_path):
    g = graph(tmp_path)
    missing, _ = run(g, {"path": str(tmp_path / "missing.pdf")}, thread="a")
    assert "__interrupt__" not in missing
    assert missing["message"].startswith("Je n'ai pas pu traduire ce document : missing.pdf")

    english = str(make_pdf(tmp_path / "en.pdf", pages=3))
    same, _ = run(g, {"path": english, "target_lang": "en"}, thread="b")
    assert "__interrupt__" not in same  # refused before the pause, even if long
    assert same["message"].endswith("en.pdf est déjà en anglais.")


def test_two_conversations_do_not_mix(tmp_path):
    """The checkpointer keeps one state per thread_id."""
    g = graph(tmp_path)
    big = str(make_pdf(tmp_path / "big.pdf", pages=3))
    small = str(make_pdf(tmp_path / "small.pdf", pages=1))
    _, waiting = run(g, {"path": big}, thread="x")
    done, _ = run(g, {"path": small}, thread="y")
    assert done["message"].startswith("Traduction terminée")
    assert g.get_state(waiting).next == ("confirm",)  # still waiting for its answer


def test_answers_accepted_as_yes():
    assert all(is_yes(a) for a in (True, "oui", "Oui ", "yes", "o", "ok"))
    assert not any(is_yes(a) for a in (False, "non", "no", "", "plus tard"))
