"""Tests for document search: tokenizer, BM25, RRF, chunking, hybrid search, Mistral embeddings."""

import json
import math

import httpx
import pymupdf
import pytest

from docagent.config import Settings
from docagent.rag import (
    BM25Index,
    Chunker,
    HashEmbedder,
    HybridSearcher,
    MistralEmbedder,
    Passage,
    VectorStore,
    open_client,
    reciprocal_rank_fusion,
    tokenize,
)

# --- tokenizer and BM25 -------------------------------------------------------


def test_tokenize_keeps_codes_and_folds_accents_and_plurals():
    assert tokenize("Les émissions de CO2 dans le scénario SSP1-2.6 (1,5 °C)") == [
        "emission",
        "co2",
        "scenario",
        "ssp1-2.6",
        "1,5",
    ]


def test_bm25_prefers_rare_terms_and_short_passages():
    docs = ["climate warming"] * 5 + [
        "the SSP1-2.6 scenario limits warming",
        "climate " * 30 + "SSP1-2.6",
    ]
    index = BM25Index(docs)
    ranking = [i for i, _ in index.search("SSP1-2.6 climate")]
    # The rare code decides (2 passages of 7 vs 6 of 7 for "climate"); of the two
    # passages that have it, the long one repeating "climate" 30 times still loses.
    assert ranking[:2] == [5, 6]


def test_bm25_idf_formula():
    index = BM25Index(["a1 b1", "a1 c1", "d1"])
    # "a1" is in 2 documents out of 3.
    assert index.idf["a1"] == pytest.approx(math.log(1 + (3 - 2 + 0.5) / (2 + 0.5)))


def test_bm25_unknown_words_find_nothing():
    assert BM25Index(["carbon budget"]).search("photosynthese") == []


# --- reciprocal rank fusion ------------------------------------------------------


def test_rrf_scores_follow_the_formula():
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60))
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["a"] == pytest.approx(1 / 61)


def test_rrf_rewards_agreement_over_a_single_first_place():
    fused = [item for item, _ in reciprocal_rank_fusion([["x", "y"], ["z", "y"]])]
    assert fused[0] == "y"  # 2nd in both lists beats 1st in only one


# --- chunking ---------------------------------------------------------------------


def make_pdf(pages: list[list[tuple[str, float, bool]]]) -> pymupdf.Document:
    """Pages of (text, font size, bold) paragraphs, each drawn as its own block."""
    doc = pymupdf.open()
    for paragraphs in pages:
        page = doc.new_page(width=595, height=842)
        page.insert_text((60, 40), "Running header of the report", fontsize=8)
        y = 70
        for text, size, bold in paragraphs:
            rect = pymupdf.Rect(60, y, 535, y + 200)
            weight = "bold" if bold else "normal"
            html = f"<p style='font-size:{size}pt; font-weight:{weight}'>{text}</p>"
            spare, _ = page.insert_htmlbox(rect, html)
            y += 200 - spare + 20
    return doc


PARA = "Global surface temperature rose by 1.1 degrees since 1850, driven by emissions. " * 4


def test_chunker_keeps_pages_sections_and_drops_repeated_headers():
    pages = [
        [("A. The current state", 14, True), (PARA, 10, False), (PARA, 10, False)],
        [(PARA, 10, False), ("B. Possible futures", 14, True), (PARA, 10, False)],
        [(PARA, 10, False)],
    ]
    passages = Chunker(max_chars=700, overlap_chars=100).split_document(
        make_pdf(pages), "doc", "Test report"
    )
    assert {p.page for p in passages} == {1, 2, 3}
    assert all("Running header" not in p.text for p in passages)  # on every page: dropped
    assert all(len(p.text) <= 700 for p in passages)
    assert passages[0].section == "A. The current state"
    assert passages[-1].section == "B. Possible futures"  # carried over to page 3
    assert passages[0].embed_text.startswith("Test report — A. The current state\n")
    assert all(p.bboxes for p in passages)


def test_chunker_overlaps_consecutive_passages():
    sentences = [f"Sentence number {i} about carbon dioxide emissions." for i in range(40)]
    pages = [[(s, 10, False) for s in sentences[:14]]]
    passages = Chunker(max_chars=400, overlap_chars=120).split_document(make_pdf(pages), "doc", "T")
    assert len(passages) >= 2
    first, second = passages[0].text, passages[1].text
    assert first.split(". ")[-1] in second  # the last block is repeated


def test_chunker_puts_footnotes_after_the_body():
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_htmlbox(pymupdf.Rect(60, 700, 535, 760), "<p style='font-size:7pt'>1 A note.</p>")
    page.insert_htmlbox(pymupdf.Rect(60, 60, 535, 200), f"<p style='font-size:10pt'>{PARA}</p>")
    passages = Chunker(max_chars=2000).split_document(doc, "d", "T")
    assert passages[0].text.startswith("Global surface")


def test_chunker_rejects_overlap_larger_than_passages():
    with pytest.raises(ValueError):
        Chunker(max_chars=100, overlap_chars=100)


# --- hybrid search ---------------------------------------------------------------------


def passages() -> list[Passage]:
    texts = [
        "Remaining carbon budget for limiting warming to 1.5°C is 500 GtCO2.",
        "Le budget carbone résiduel pour limiter le réchauffement à 1,5 °C.",
        "Investment in data centres is expected to reach USD 580 billion in 2025.",
        "Arctic sea ice area reached its lowest level since at least 1850.",
        "The SSP1-2.6 scenario reaches net zero CO2 around 2070.",
    ]
    return [
        Passage(id=f"d:p{i}:{i}", doc_id="d", title="Doc", page=i, text=t)
        for i, t in enumerate(texts, start=1)
    ]


@pytest.fixture
def searcher() -> HybridSearcher:
    ps = passages()
    embedder = HashEmbedder()
    store = VectorStore(open_client(":memory:"), "t", embedder.dim, "hash")
    store.upsert(ps, embedder.embed_documents([p.embed_text for p in ps]))
    return HybridSearcher(ps, embedder, store, candidates=5)


def test_hybrid_search_fuses_both_rankings(searcher):
    hits = searcher.search("data centres investment 2025", top_k=3)
    assert hits[0].passage.page == 3
    assert hits[0].dense_rank == 1 and hits[0].bm25_rank == 1


def test_each_mode_can_run_alone(searcher):
    assert searcher.search("SSP1-2.6", 1, "bm25")[0].passage.page == 5
    dense = searcher.search("Arctic sea ice area", 1, "dense")[0]
    assert dense.passage.page == 4 and dense.bm25_rank is None


def test_bm25_only_search_needs_no_embedder():
    s = HybridSearcher(passages())
    assert s.search("budget carbone", 1, "bm25")[0].passage.page == 2
    with pytest.raises(RuntimeError):
        s.search("budget carbone", 1, "hybrid")


def test_store_only_embeds_what_is_missing():
    ps = passages()
    embedder = HashEmbedder()
    store = VectorStore(open_client(":memory:"), "t", embedder.dim, "hash")
    store.upsert(ps[:2], embedder.embed_documents([p.embed_text for p in ps[:2]]))
    assert [p.page for p in store.missing(ps)] == [3, 4, 5]


# --- Mistral embeddings (HTTP mocked) -----------------------------------------------------


def test_mistral_embedder_batches_normalises_and_retries():
    calls: list[dict] = []
    replies = [httpx.Response(429)]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if replies:
            return replies.pop()
        data = [{"index": i, "embedding": [3.0, 4.0]} for i in range(len(body["input"]))]
        return httpx.Response(200, json={"data": data, "usage": {"prompt_tokens": 7}})

    settings = Settings(_env_file=None, mistral_api_key="k", llm_max_retries=2)
    embedder = MistralEmbedder(
        settings,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda s: None,
        max_batch=2,
    )
    vectors = embedder.embed_documents(["a", "b", "c"])
    assert vectors == [[0.6, 0.8]] * 3  # unit length: dot product = cosine
    assert [len(c["input"]) for c in calls] == [2, 2, 1]  # 429, retried batch, last batch
    assert calls[0]["model"] == "mistral-embed"


# --- evaluation and paired tests ----------------------------------------------------------


def test_metrics_and_first_hit():
    from docagent.rag.evaluation import first_hit, summarise

    pages = [("a", 3), ("b", 7), ("a", 1)]
    assert first_hit(pages, {("a", 1), ("b", 7)}) == 2
    assert first_hit(pages, {("c", 1)}) is None
    s = summarise({"q1": 1, "q2": 4, "q3": None, "q4": 20})
    assert s == {"hit@1": 0.25, "hit@5": 0.5, "mrr@10": pytest.approx((1 + 1 / 4) / 4)}


def test_binomial_p_matches_known_values():
    from docagent.rag.evaluation import binomial_p

    assert binomial_p(8, 1) == pytest.approx(20 / 512)  # 2 * (C(9,0) + C(9,1)) / 2^9
    assert binomial_p(4, 0) == pytest.approx(0.125)
    assert binomial_p(0, 0) == 1.0
    assert binomial_p(5, 5) == 1.0


def test_compare_is_paired_and_counts_disagreements_only():
    from docagent.rag.evaluation import compare

    a = {"q1": 1, "q2": 2, "q3": None, "q4": 3}
    b = {"q1": 1, "q2": None, "q3": 7, "q4": 6}
    c = compare(a, b)
    assert (c.hit_wins, c.hit_losses) == (2, 0)  # q2 and q4 in A's top 5 only
    assert (c.rank_wins, c.rank_losses) == (2, 1)  # q3: 7 beats "not found" (11)


def test_local_embedder_adds_each_model_prompts(monkeypatch):
    import sys
    import types

    import numpy as np

    from docagent.rag import LocalEmbedder

    seen: list[str] = []

    class FakeModel:
        max_seq_length = 32768

        def __init__(self, repo):
            self.repo = repo

        def get_embedding_dimension(self):
            return 2

        def encode(self, texts, **kwargs):
            seen.extend(texts)
            return np.ones((len(texts), 2)) / np.sqrt(2)

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeModel)
    )
    qwen = LocalEmbedder("qwen3")
    qwen.embed_query("Quel secteur ?")
    qwen.embed_documents(["Les transports."])
    assert seen[0].startswith("Instruct: Given a question")
    assert seen[0].endswith("\nQuery:Quel secteur ?")
    assert seen[1] == "Les transports."  # no instruction on the passages
    assert qwen.model.max_seq_length == 512 and qwen.dim == 2

    e5 = LocalEmbedder("e5")
    e5.embed_query("q")
    e5.embed_documents(["d"])
    assert seen[2:] == ["query: q", "passage: d"]
