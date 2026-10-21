"""The agent's tools (docagent/agent/tools.py): small fakes, no network."""

from pathlib import Path

import pymupdf
import pytest
from test_answer import chat, searcher  # same fakes as the answer tests

from docagent.agent import Toolbox, ToolError, other_language, query_language
from docagent.config import Settings
from docagent.rag import Answerer
from docagent.translate import FakeTranslator

ENGLISH = "The climate in 2024 was the warmest ever measured around the world"


def make_pdf(path: Path, pages: int = 1, text: str = ENGLISH) -> Path:
    doc = pymupdf.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_htmlbox(pymupdf.Rect(60, 60, 500, 120), f"<p>{text}, page {i}</p>")
    doc.save(path)
    return path


def settings(**values) -> Settings:
    return Settings(_env_file=None, mistral_api_key="k", llm_model="m", **values)


def toolbox(tmp_path: Path, *replies, **values) -> tuple[Toolbox, list[dict]]:
    client, sent = chat(*replies)
    values.setdefault("corpus_path", tmp_path / "no_corpus.json")
    tools = Toolbox(
        settings(**values),
        answerer=Answerer(searcher(), client, top_k=3),
        make_translator=FakeTranslator,
        chat=client,
        output_dir=tmp_path / "out",
    )
    return tools, sent


def test_pdf_info_counts_pages_and_asks_for_confirmation_above_the_limit(tmp_path):
    tools, _ = toolbox(tmp_path, agent_confirm_pages=2)
    small = tools.pdf_info(make_pdf(tmp_path / "small.pdf", pages=2))
    big = tools.pdf_info(make_pdf(tmp_path / "big.pdf", pages=3))
    assert (small.pages, small.needs_confirmation) == (2, False)
    assert (big.pages, big.needs_confirmation) == (3, True)
    assert small.language == "en"


def test_pdf_info_refuses_what_is_not_a_readable_pdf(tmp_path):
    tools, _ = toolbox(tmp_path)
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / "broken.pdf").write_bytes(b"%PDF-1.7 not really")
    for name in ("missing.pdf", "notes.txt", "broken.pdf"):
        with pytest.raises(ToolError):
            tools.pdf_info(tmp_path / name)


def test_translate_pdf_writes_the_translation_and_its_checks(tmp_path):
    tools, _ = toolbox(tmp_path)
    pages = []
    result = tools.translate_pdf(
        make_pdf(tmp_path / "report.pdf", pages=2), on_page=lambda d, t: pages.append((d, t))
    )
    assert Path(result.output) == tmp_path / "out" / "report_fr.pdf"
    assert Path(result.output).exists()
    assert (result.source_lang, result.target_lang, result.pages) == ("en", "fr", 2)
    assert result.zones >= 2 and result.checks_ok
    assert pages == [(1, 2), (2, 2)]


def test_translate_pdf_refuses_a_document_already_in_the_target_language(tmp_path):
    tools, _ = toolbox(tmp_path)
    with pytest.raises(ToolError, match="already in French"):
        tools.translate_pdf(make_pdf(tmp_path / "r.pdf"), target_lang="fr", source_lang="fr")


def test_search_returns_short_references_with_who_found_them(tmp_path):
    tools, sent = toolbox(tmp_path)
    hits = tools.search_documents("transports émissions 2024", top_k=2)
    assert len(hits) == 2
    assert hits[0].doc_id == "d1" and hits[0].page == 11
    assert set(hits[0].found_by) <= {"dense", "bm25"} and hits[0].found_by
    assert sent == []  # no LLM call


def test_search_documents_can_be_limited_to_some_reports(tmp_path):
    tools, _ = toolbox(tmp_path)
    query = "transports émissions 2024"  # the best match is in d1
    hits = tools.search_documents(query, top_k=3, doc_ids=["d2", "d3"])
    assert hits and {h.doc_id for h in hits} <= {"d2", "d3"}
    with pytest.raises(ToolError, match="unknown document"):
        tools.search_documents(query, doc_ids=["d9"])


def test_ask_passes_refusals_through_as_normal_results(tmp_path):
    tools, _ = toolbox(tmp_path, {"answerable": False, "answer": "", "evidence": []})
    answer = tools.ask("Quel est le prix du pétrole en 2030 ?")
    assert answer.answerable is False and answer.sources == []


def test_translate_query_goes_to_the_other_language_and_keeps_the_rules(tmp_path):
    reply = {"translation": "Greenhouse gas emissions from transport in 2024"}
    tools, sent = toolbox(tmp_path, reply)
    result = tools.translate_query("Quelles sont les émissions de GES des transports en 2024 ?")
    assert (result.source_lang, result.target_lang) == ("fr", "en")
    assert result.translation == reply["translation"]
    system, user = sent[0]["messages"]
    assert "SSP1-2.6" in system["content"]  # scenario names are kept as they are
    assert user["content"].startswith("Target language: English")


def test_the_default_query_client_is_deterministic_and_fails_fast():
    client = Toolbox(settings(ask_max_retries=1)).chat
    assert (client.temperature, client.max_retries) == (0.0, 1)


def test_translate_query_retries_an_empty_translation(tmp_path):
    tools, sent = toolbox(tmp_path, {"translation": "  "}, {"translation": "sea level rise"})
    assert tools.translate_query("hausse du niveau de la mer").translation == "sea level rise"
    assert len(sent) == 2


def test_language_helpers():
    assert query_language("Quelle est la hausse du niveau de la mer depuis 1901 ?") == "fr"
    assert query_language("How much did sea level rise since 1901?") == "en"
    assert (other_language("fr"), other_language("en")) == ("en", "fr")


def test_the_index_is_loaded_only_when_a_question_needs_it(tmp_path):
    loads = []

    def make_answerer():
        loads.append(1)
        raise FileNotFoundError("no index")

    tools = Toolbox(
        settings(),
        make_answerer=make_answerer,
        make_translator=FakeTranslator,
        output_dir=tmp_path,
    )
    tools.translate_pdf(make_pdf(tmp_path / "r.pdf"))
    assert loads == []
    with pytest.raises(FileNotFoundError):
        tools.search_documents("anything")


# -- summarize_document ---------------------------------------------------------------

SEA = "Global mean sea level increased by 0.20 m between 1901 and 2018."


def test_summary_keeps_the_checked_points_and_says_why_others_were_dropped(tmp_path):
    reply = {
        "points": [
            {
                "text": "Le niveau moyen des mers a augmenté de 0,20 m entre 1901 et 2018 [1].",
                "evidence": [{"n": 1, "quote": SEA}],
            },
            {  # a figure that is in no excerpt
                "text": "Il pourrait monter de 0,35 m d'ici 2050 [1].",
                "evidence": [{"n": 1, "quote": SEA}],
            },
            {  # a quote that is not in the excerpt
                "text": "La hausse s'accélère [1].",
                "evidence": [{"n": 1, "quote": "The rise is accelerating fast everywhere."}],
            },
            "not a point",
        ]
    }
    tools, sent = toolbox(tmp_path, reply)
    summary = tools.summarize_document("d2", focus="niveau de la mer")

    assert summary.ok and len(summary.points) == 1
    point = summary.points[0]
    assert point.text.startswith("Le niveau moyen des mers") and point.text.endswith("[1].")
    assert (point.sources[0].doc_id, point.sources[0].page, point.sources[0].quotes) == (
        "d2",
        12,
        [SEA],
    )
    figure, fake = summary.rejected_points
    assert figure.quotes == [f"[1] {SEA}"] and figure.bad_quotes == []
    assert figure.found_elsewhere == {"0.35": [], "2050": []}  # in no excerpt at all
    assert fake.bad_quotes == ["[1] The rise is accelerating fast everywhere."]
    reasons = [r.reason for r in summary.rejected_points]
    assert reasons == [
        "figures not in the cited excerpts: 0.35, 2050",  # numbers normalised
        "no quote found in its excerpt",
    ]
    assert summary.retrieved == ["d2:p2:0"]  # only passages of that report
    assert (summary.passages_used, summary.passages_total) == (1, 1)

    system, user = sent[0]["messages"]
    assert "at most 5 points" in system["content"]
    assert "ONE fact per point" in system["content"]
    assert "Write the points in French" in system["content"]
    assert user["content"].startswith("Report: Report 2\nFocus: niveau de la mer")
    assert summary.method == "search"
    assert "[1] (Report 2, p. 12)" in user["content"]


def test_summary_without_focus_uses_the_default_one_in_the_chosen_language(tmp_path):
    tools, sent = toolbox(tmp_path, {"points": []})
    summary = tools.summarize_document("d3", lang="en")
    assert summary.focus == "main findings, key figures and conclusions of the report"
    assert "Write the points in English" in sent[0]["messages"][0]["content"]
    assert not summary.ok and summary.points == []  # nothing usable: no summary


def test_summary_of_an_unknown_report_or_in_an_unknown_language(tmp_path):
    tools, sent = toolbox(tmp_path)
    with pytest.raises(ToolError, match="unknown document"):
        tools.summarize_document("d9")
    with pytest.raises(ToolError, match="language"):
        tools.summarize_document("d1", lang="de")
    assert sent == []  # refused before any LLM call


def test_summary_keeps_at_most_max_points(tmp_path):
    point = {"text": "Le niveau a monté de 0.20 m [1].", "evidence": [{"n": 1, "quote": SEA}]}
    tools, _ = toolbox(tmp_path, {"points": [point] * 4})
    assert len(tools.summarize_document("d2", max_points=2).points) == 2


def test_without_a_focus_the_summary_starts_from_the_report_own_summary(tmp_path):
    corpus = tmp_path / "corpus.json"
    corpus.write_text(
        '[{"id": "d2", "file": "d2.pdf", "title": "Report 2", "org": "X",'
        ' "summary_pages": [12, 12]}]',
        encoding="utf-8",
    )
    point = {"text": "Le niveau a monté de 0.20 m [1].", "evidence": [{"n": 1, "quote": SEA}]}
    tools, sent = toolbox(tmp_path, {"points": [point]}, {"points": []}, corpus_path=corpus)

    summary = tools.summarize_document("d2")
    assert (summary.method, summary.pages, summary.retrieved) == (
        "author_summary",
        [12],
        ["d2:p2:0"],
    )
    assert "report's own summary (pages 12 to 12)" in sent[0]["messages"][1]["content"]
    assert summary.ok

    # A focus asks for something precise: back to search.
    assert tools.summarize_document("d2", focus="sea level").method == "search"


def test_spread_keeps_evenly_spaced_passages_within_the_budget():
    from docagent.agent.tools import spread
    from docagent.rag import Passage

    ps = [Passage(id=f"x:p{i}:0", doc_id="x", title="X", page=i, text="a" * 100) for i in range(10)]
    assert [p.page for p in spread(ps, 300)] == [0, 3, 6]
    assert spread(ps, 5000) == ps


def test_a_rejected_figure_says_which_excerpt_had_it():
    """The HCC 2025 case: a true figure, quoted from the wrong excerpt."""
    from docagent.agent.grounded import LLMPoints, check_points
    from docagent.rag import Passage

    ps = [
        Passage(id="h:p6:0", doc_id="h", title="HCC", page=6, text="Le rythme doit doubler."),
        Passage(
            id="h:p5:0",
            doc_id="h",
            title="HCC",
            page=5,
            text="Les émissions brutes sont inférieures de 32 % au niveau de 1990.",
        ),
    ]
    reply = LLMPoints.model_validate(
        {
            "points": [
                {
                    "text": "Les émissions ont baissé de 32 % et le rythme doit doubler [1].",
                    "evidence": [{"n": 1, "quote": "Le rythme doit doubler."}],
                }
            ]
        }
    )
    kept, rejected = check_points(reply, ps, "")
    assert kept == [] and rejected[0].found_elsewhere == {"32": [2]}


# -- compare_documents ----------------------------------------------------------------

DATA = "Investment in data centres is expected to reach USD 580 billion in 2025."


def side(report: str, says: str, *quotes: tuple[int, str]) -> dict:
    return {"report": report, "says": says, "evidence": [{"n": n, "quote": q} for n, q in quotes]}


def test_comparison_checks_who_says_what(tmp_path):
    sea, data = (1, SEA), (2, DATA)  # excerpt 1 belongs to A (d2), excerpt 2 to B (d3)
    reply = {
        "points": [
            {  # one side per report, each quoting its own report
                "kind": "difference",
                "aspect": "Chiffres clés",
                "sides": [
                    side("A", "0.20 m de hausse [1].", sea),
                    side("B", "580 milliards [2].", data),
                ],
            },
            {  # a difference with one side only
                "kind": "difference",
                "aspect": "Mer",
                "sides": [side("A", "0.20 m [1].", sea)],
            },
            {  # side A quotes report B's excerpt: wrong attribution
                "kind": "only_in",
                "aspect": "Data centres",
                "sides": [side("A", "580 milliards [2].", data)],
            },
            {  # only report B addresses it: kept, the note is written by code
                "kind": "only_in",
                "aspect": "Data centres",
                "sides": [side("b", "580 milliards USD en 2025 [2].", data)],
            },
            {  # only_in with two sides
                "kind": "only_in",
                "aspect": "x",
                "sides": [side("A", "0.20 m [1].", sea), side("B", "580 milliards [2].", data)],
            },
            {"kind": "opinion", "aspect": "x", "sides": [side("A", "x", sea)]},  # dropped
        ]
    }
    tools, sent = toolbox(tmp_path, reply)
    result = tools.compare_documents(["d2", "d3"], "niveau de la mer et énergie", per_doc=1)

    assert [p.kind for p in result.points] == ["difference", "only_in"]
    diff, only = result.points
    assert [(s.doc_id, s.title) for s in diff.sides] == [("d2", "Report 2"), ("d3", "Report 3")]
    assert (
        diff.text == "Chiffres clés. Report 2: 0.20 m de hausse [1]. Report 3: 580 milliards [2]."
    )
    assert diff.note is None
    assert only.sides[0].doc_id == "d3"  # label "b" accepted
    assert only.note == "Rien sur ce point dans le passage consulté de Report 2."
    assert only.text.endswith("[2]. Rien sur ce point dans le passage consulté de Report 2.")

    reasons = [r.reason for r in result.rejected_points]
    assert reasons == [
        "difference: needs one side per report (has 1)",
        "side A (Report 2) quotes excerpts [2] of another report: wrong attribution",
        "only_in: one side only (several reports: agreement or difference?)",
    ]
    assert result.retrieved == ["d2:p2:0", "d3:p3:0"]
    assert [(d.doc_id, d.passages_read) for d in result.documents] == [("d2", 1), ("d3", 1)]

    system, user = sent[0]["messages"]
    assert "Two unrelated facts are not a difference" in system["content"]
    assert "- A = Report 2: excerpts 1 to 1\n- B = Report 3: excerpts 2 to 2" in user["content"]


def test_a_side_with_an_unproven_figure_rejects_the_point(tmp_path):
    reply = {
        "points": [
            {
                "kind": "agreement",
                "aspect": "x",
                "sides": [side("A", "1,2 m [1].", (1, SEA)), side("B", "580 [2].", (2, DATA))],
            }
        ]
    }
    tools, _ = toolbox(tmp_path, reply)
    result = tools.compare_documents(["d2", "d3"], "x", per_doc=1)
    assert result.points == []
    assert result.rejected_points[0].reason.startswith("figures not in the cited excerpts: 1.2")
    assert result.rejected_points[0].text.endswith("(side A)")


def test_absence_note_in_english_and_for_several_passages():
    from docagent.agent.grounded import absence_note

    assert (
        absence_note("en", "IPCC SPM", 5)
        == "Nothing on this in the 5 passages of IPCC SPM that were read."
    )
    assert (
        absence_note("fr", "GIEC RID", 5)
        == "Rien sur ce point dans les 5 passages consultés de GIEC RID."
    )


@pytest.mark.parametrize(
    ("doc_ids", "topic", "lang", "error"),
    [
        (["d2"], "x", "fr", "2 or 3"),
        (["d2", "d2"], "x", "fr", "2 or 3"),
        (["d1", "d2", "d3", "d1x"], "x", "fr", "2 or 3"),
        (["d2", "d9"], "x", "fr", "unknown document"),
        (["d2", "d3"], "  ", "fr", "topic"),
        (["d2", "d3"], "x", "de", "language"),
    ],
)
def test_comparison_refuses_bad_requests_before_any_llm_call(tmp_path, doc_ids, topic, lang, error):
    tools, sent = toolbox(tmp_path)
    with pytest.raises(ToolError, match=error):
        tools.compare_documents(doc_ids, topic, lang=lang)
    assert sent == []


@pytest.mark.parametrize(
    "text",
    [
        # what the model wrote on the WEO / IPCC comparison (compare-v2)
        "Les scénarios SSP1-1.9 montrent une réduction drastique mais ne précisent pas "
        "explicitement une réduction de moitié d'ici 2035.",
        "Le rapport ne donne pas de chiffres précis sur les volumes de retrait de CO₂.",
        "Le rapport souligne la neutralité sans mentionner l'année 2035.",
        "Le rapport ne le mentionne pas.",
        "Il n’évoque aucun chiffre.",
        "Le GIEC reste muet sur ce point.",
        "The SPM does not provide figures for 2050.",
        "No specific figures are given.",
        # what slipped through on the IEA / HCC comparison in English (15 Oct.)
        "No specific percentage of energy efficiency improvement is mentioned.",
        "The 2035 target is not specified in the HCC report.",
        "L'objectif 2035 n'est pas mentionné.",
        "Aucun pourcentage d'amélioration n'est donné.",
        # what slipped through on the HCC / GIEC comparison (16 Oct.)
        "Diminution des puits, sans mention explicite de données sur les forêts avant 2100.",
    ],
)
def test_absence_claims_are_detected(text):
    from docagent.agent.grounded import absence_claim

    assert absence_claim(text)


@pytest.mark.parametrize(
    "text",
    [
        "Ces trajectoires ne sont pas compatibles avec 1,5 °C.",  # a fact the report states
        "Les émissions ne baissent pas assez vite.",
        "Il ne faut pas dépasser 1,5 °C.",
        "Global emissions do not fall fast enough.",
        "Le rapport précise que les émissions ont baissé de 32 %.",
        "The 1.5 °C limit is not reached in this scenario.",  # a passive about the world
        "Coal demand is not expected to fall before 2030.",
    ],
)
def test_negations_about_the_world_are_allowed(text):
    from docagent.agent.grounded import absence_claim

    assert absence_claim(text) is None


def test_a_side_with_an_absence_claim_rejects_the_point(tmp_path):
    says = "La mer a monté de 0.20 m [1], mais le rapport ne précise pas la cause."
    reply = {
        "points": [
            {
                "kind": "difference",
                "aspect": "mer",
                "sides": [side("A", says, (1, SEA)), side("B", "580 [2].", (2, DATA))],
            }
        ]
    }
    tools, sent = toolbox(tmp_path, reply)
    result = tools.compare_documents(["d2", "d3"], "x", per_doc=1)
    assert result.points == []
    assert result.rejected_points[0].reason == 'side A: absence claim ("ne précise pas")'
    assert 'never a "difference"' in sent[0]["messages"][0]["content"]


def test_a_bare_list_of_points_is_read_as_the_points(tmp_path):
    """nemo sometimes answers ``[...]`` instead of ``{"points": [...]}``: on one
    comparison, 3 times in a row, and the request failed although the points were fine."""
    from docagent.agent.grounded import LLMComparison, LLMPoints

    point = {"text": "La mer a monté de 0.20 m [1].", "evidence": [{"n": 1, "quote": SEA}]}
    side = {"report": "A", "says": "0.20 m [1].", "evidence": [{"n": 1, "quote": SEA}]}
    compared = {"kind": "only_in", "aspect": "Mer", "sides": [side]}
    client, sent = chat([compared], [point])
    message = [{"role": "user", "content": "x"}]
    assert client.complete_json(message, LLMComparison).points[0].aspect == "Mer"
    assert client.complete_json(message, LLMPoints).points[0].text.startswith("La mer")
    assert len(sent) == 2  # read at once, no retry
    assert LLMPoints.model_validate({"points": [point]}).points[0].evidence[0].n == 1
