"""The scoring of the agent's evaluations, and the evaluation sets themselves."""

import json
from pathlib import Path

import pytest
from test_agent_tools import settings
from test_answer import chat
from test_supervisor import CORPUS

from docagent.agent import Toolbox
from docagent.agent.evaluation import (
    TrapError,
    judge_items,
    judge_summary,
    plan_matches,
    plan_signature,
    routing_summary,
    sentences,
    step_matches,
)
from docagent.agent.supervisor import ROUTES, Plan, plan_request
from docagent.rag.chunking import Passage

ROOT = Path(__file__).resolve().parents[1]
WEO = ["iea-weo-2025", "aie-weo-2025-resume"]
IPCC = ["giec-rid-fr", "ipcc-spm-en"]


def load(name: str) -> list[dict]:
    path = ROOT / "evals" / name
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


# -- routing -----------------------------------------------------------------------


def test_a_step_matches_its_route_reports_and_topic():
    plan = Plan(route="compare", doc_ids=["aie-weo-2025-resume", "ipcc-spm-en"], topic="le 1,5 °C")
    spec = {"route": "compare", "reports": [WEO, IPCC], "topic_any": ["1,5", "1.5"]}
    assert step_matches(plan, spec)
    assert step_matches(plan, spec | {"route": ["question", "compare"]})
    assert not step_matches(plan, spec | {"topic_any": ["charbon"]})
    assert not step_matches(plan, spec | {"route": "summarize"})


def test_two_versions_of_one_report_do_not_make_a_comparison_of_two_reports():
    both_weo = Plan(route="compare", doc_ids=WEO, topic="1,5")
    assert not step_matches(both_weo, {"route": "compare", "reports": [WEO, IPCC]})
    extra = Plan(route="compare", doc_ids=["iea-weo-2025", "ipcc-spm-en", "hcc-ra-2025"])
    assert not step_matches(extra, {"route": "compare", "reports": [WEO, IPCC]})


def test_a_translation_matches_on_its_file_and_language():
    plan = Plan(route="translate", path="data/raw/IPCC_AR6_WGI_SPM_English.pdf", target_lang="fr")
    spec = {"route": "translate", "path_contains": "ipcc_ar6_wgi_spm_english"}
    assert step_matches(plan, spec)
    assert not step_matches(plan, spec | {"target_lang": "en"})
    assert not step_matches(plan, spec | {"path_contains": "French"})


def test_a_plan_must_match_one_accepted_plan_step_by_step_in_order():
    summary = Plan(route="summarize", doc_ids=["hcc-ra-2025"])
    question = Plan(route="question", question="?")
    accepted = [[{"route": "summarize", "reports": [["hcc-ra-2025"]]}, {"route": "question"}]]
    assert plan_matches([summary, question], accepted)
    assert not plan_matches([question, summary], accepted)  # order
    assert not plan_matches([summary], accepted)  # a step missing
    assert plan_matches([question], accepted + [[{"route": "question"}]])  # an alternative


def test_routing_summary_counts_right_plans_and_stability():
    sig = plan_signature([Plan(route="summarize", doc_ids=["b", "a"]), Plan(route="question")])
    assert sig == "summarize(a,b) > question()"
    rows = [
        {"id": "q1", "category": "question", "run": 1, "ok": True, "signature": "question()"},
        {"id": "q1", "category": "question", "run": 2, "ok": True, "signature": "question()"},
        {
            "id": "k1",
            "category": "clarify",
            "run": 1,
            "ok": True,
            "signature": "clarify()",
            "fixed_by_code": True,
        },
        {"id": "k1", "category": "clarify", "run": 2, "ok": False, "signature": "summarize(x)"},
    ]
    summary = routing_summary(rows)
    assert (summary["right"], summary["total"]) == (3, 4)
    assert summary["by_category"] == {"question": (2, 2), "clarify": (1, 2)}
    assert (summary["stable"], summary["unstable"], summary["wrong"]) == (1, ["k1"], ["k1"])
    assert summary["fixed_by_code"] == ["k1"]


def test_plan_request_is_the_router_alone_with_the_checks_in_code(tmp_path):
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps(CORPUS), encoding="utf-8")
    router, sent = chat({"steps": [{"route": "summarize", "doc_ids": ["ademe"]}]})
    tools = Toolbox(settings(corpus_path=corpus))
    routing = plan_request(router, tools, "Résume le rapport de l'ADEME")
    assert routing.lang == "fr" and len(sent) == 1
    assert [s.route for s in routing.raw.steps] == ["summarize"]  # what the model said
    assert [p.route for p in routing.plans] == ["clarify"]  # what runs: unknown report
    assert "Report 1" in sent[0]["messages"][0]["content"]  # the catalogue is in the prompt


def test_the_routing_cases_name_only_reports_of_the_corpus():
    known = {d["id"] for d in json.loads((ROOT / "data" / "corpus.json").read_text("utf-8"))}
    cases = load("agent_routing.jsonl")
    assert len(cases) == 25 and len({c["id"] for c in cases}) == 25
    for case in cases:
        assert case["request"] and case["accept"], case["id"]
        for steps in case["accept"]:
            for step in steps:
                routes = step["route"] if isinstance(step["route"], list) else [step["route"]]
                assert set(routes) <= set(ROUTES), case["id"]
                for group in step.get("reports", []):
                    assert set(group) <= known, case["id"]


# -- judge -------------------------------------------------------------------------


def passage(text: str, page: int = 4) -> Passage:
    return Passage(id=f"r:p{page}:1", doc_id="r", title="Rapport", page=page, text=text)


def test_sentences_split_after_a_footnote_number_but_not_inside_a_figure():
    text = "It was 0.99 [0.84 to 1.10] °C higher than 1850–1900.9 It was 1.09 °C. Elle monte."
    assert sentences(text) == [
        "It was 0.99 [0.84 to 1.10] °C higher than 1850–1900.",  # the footnote "9" goes
        "It was 1.09 °C.",
        "Elle monte.",
    ]


def test_judge_items_quote_the_sentences_that_hold_the_words():
    text = "Le puits océanique baisse de 5,9 %. Le puits terrestre baisse de 27 %. Fin."
    case = {
        "id": "B",
        "doc_id": "r",
        "page": 4,
        "quotes": [["océanique"], ["terrestre", "27"]],
        "true": "vrai",
        "trap": "piège",
    }
    items = judge_items([passage("autre", page=5), passage(text)], case)
    assert [kind for kind, _ in items] == ["true", "trap"]  # no borderline in this case
    item = items[1][1]
    assert item.statement == "piège" and item.n == 1
    assert item.quotes == [
        "Rapport, p. 4: Le puits océanique baisse de 5,9 %.",
        "Rapport, p. 4: Le puits terrestre baisse de 27 %.",
    ]
    assert item.excerpts == [f"Rapport, p. 4: {text}"]


def test_a_case_that_finds_nothing_says_so():
    case = {"id": "X", "doc_id": "r", "quotes": [["introuvable"]], "true": "t"}
    with pytest.raises(TrapError, match="no passage"):
        judge_items([passage("Rien ici.")], case)
    split = {"id": "Y", "doc_id": "r", "quotes": [["un", "deux"]], "true": "t"}
    with pytest.raises(TrapError, match="no sentence"):
        judge_items([passage("un mot ici. Puis deux là.")], split)


def test_judge_summary_scores_traps_faithful_and_constancy():
    rows = [
        {"item": "A-trap", "kind": "trap", "judged": True, "removed": True},
        {"item": "A-trap", "kind": "trap", "judged": True, "removed": False},
        {"item": "A-true", "kind": "true", "judged": True, "removed": False},
        {"item": "A-true", "kind": "true", "judged": False, "removed": False},
        {"item": "B-borderline", "kind": "borderline", "judged": True, "removed": True},
    ]
    summary = judge_summary(rows)
    assert summary["traps_caught"] == (1, 2)
    assert summary["true_removed"] == (0, 1)  # the failed review is not counted
    assert summary["borderline_removed"] == (1, 1)
    assert (summary["constant"], summary["items"], summary["changing"]) == (2, 3, ["A-trap"])
    assert summary["not_judged"] == 1


def test_the_judge_cases_are_complete():
    cases = load("judge_traps.jsonl")
    assert len(cases) == 5
    for case in cases:
        assert case["true"] and case["trap"] and case["quotes"] and case["request"], case["id"]
