"""Question answering: prompt, citation checks, refusals, retries, /v1/ask (HTTP mocked)."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from docagent.api import create_app
from docagent.chat import ChatClient, ChatError
from docagent.config import Settings
from docagent.rag import (
    Answerer,
    HashEmbedder,
    HybridSearcher,
    Passage,
    VectorStore,
    open_client,
)
from docagent.rag.answer import NOT_FOUND, build_messages

TEXTS = [
    "Le secteur des transports reste le premier secteur émetteur (34 % des émissions en 2024).",
    "Global mean sea level increased by 0.20 m between 1901 and 2018.",
    "Investment in data centres is expected to reach USD 580 billion in 2025.",
]


def passages() -> list[Passage]:
    return [
        Passage(id=f"d{i}:p{i}:0", doc_id=f"d{i}", title=f"Report {i}", page=10 + i, text=t)
        for i, t in enumerate(TEXTS, start=1)
    ]


def searcher() -> HybridSearcher:
    ps = passages()
    embedder = HashEmbedder()
    store = VectorStore(open_client(":memory:"), "t", embedder.dim, "hash")
    store.upsert(ps, embedder.embed_documents([p.embed_text for p in ps]))
    return HybridSearcher(ps, embedder, store, candidates=3)


def chat(*replies) -> tuple[ChatClient, list[dict]]:
    """A ChatClient whose API answers with ``replies`` in turn (dicts become JSON)."""
    sent: list[dict] = []
    queue = list(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        reply = queue.pop(0)
        if isinstance(reply, httpx.Response):
            return reply
        content = reply if isinstance(reply, str) else json.dumps(reply)
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    settings = Settings(
        _env_file=None, mistral_api_key="k", llm_model="open-mistral-nemo", llm_max_retries=2
    )
    client = ChatClient(
        settings, client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda s: None
    )
    return client, sent


def test_prompt_numbers_the_excerpts_with_title_and_page():
    messages = build_messages("Quel secteur ?", passages()[:2])
    user = messages[1]["content"]
    assert user.startswith("Question: Quel secteur ?")
    assert "[1] (Report 1, p. 11)\nLe secteur des transports" in user
    assert "[2] (Report 2, p. 12)" in user
    assert "ONLY the numbered excerpts" in messages[0]["content"]


def test_answer_keeps_valid_citations_and_maps_them_to_pages():
    client, sent = chat(
        {
            "answerable": True,
            "answer": "Les transports, avec 34 % des émissions en 2024 [1][7].",
            "citations": [1, 7],
        }
    )
    answer = Answerer(searcher(), client, top_k=3).ask("Quel est le premier secteur émetteur ?")
    assert answer.answerable
    assert answer.answer == "Les transports, avec 34 % des émissions en 2024 [1]."  # [7] removed
    assert [(s.n, s.doc_id, s.page) for s in answer.sources] == [(1, "d1", 11)]
    assert len(answer.retrieved) == 3
    assert sent[0]["model"] == "open-mistral-nemo"
    assert sent[0]["response_format"] == {"type": "json_object"}


def test_refusal_gives_the_not_found_message_in_the_question_language():
    client, _ = chat({"answerable": False, "answer": "Not in the documents.", "citations": []})
    answer = Answerer(searcher(), client).ask("Qui a gagné la Coupe du monde 2018 ?")
    assert not answer.answerable and answer.sources == []
    assert answer.answer == NOT_FOUND["fr"]


def test_an_answer_citing_nothing_is_not_shown():
    client, _ = chat({"answerable": True, "answer": "Sea level rose by 0.20 m.", "citations": []})
    answer = Answerer(searcher(), client).ask("How much did sea level rise since 1901?")
    assert not answer.answerable
    assert answer.answer == NOT_FOUND["en"]


def test_invalid_json_and_rate_limits_are_retried():
    good = {"answerable": True, "answer": "USD 580 billion in 2025 [1].", "citations": [1]}
    client, sent = chat(httpx.Response(429), "not json", good)
    answer = Answerer(searcher(), client).ask("Investment in data centres in 2025?")
    assert answer.answerable and len(sent) == 3


def test_chat_gives_up_after_the_retries():
    client, _ = chat("x", "y", "z")
    with pytest.raises(ChatError):
        Answerer(searcher(), client).ask("Investment in data centres in 2025?")


# --- POST /v1/ask --------------------------------------------------------------------


def api(tmp_path, factory) -> TestClient:
    settings = Settings(_env_file=None, jobs_dir=tmp_path / "jobs", translator="fake")
    return TestClient(create_app(settings, answerer_factory=factory))


def test_ask_endpoint_returns_the_answer_and_its_sources(tmp_path):
    client, _ = chat({"answerable": True, "answer": "0.20 m [1].", "citations": [1]})
    built = []

    def factory():
        built.append(1)
        return Answerer(searcher(), client)

    http = api(tmp_path, factory)
    body = http.post("/v1/ask", json={"question": "How much did sea level rise?"}).json()
    assert body["answer"] == "0.20 m [1]." and body["sources"][0]["page"] >= 11
    assert http.post("/v1/ask", json={"question": "ab"}).status_code == 422  # too short
    assert built == [1]  # the index is loaded once, on the first question


def test_ask_endpoint_reports_a_missing_index_and_a_failing_llm(tmp_path):
    def no_index():
        raise FileNotFoundError("data/index/passages.jsonl not found")

    response = api(tmp_path, no_index).post("/v1/ask", json={"question": "Sea level?"})
    assert response.status_code == 503 and "index" in response.json()["detail"]

    client, _ = chat("x", "y", "z")
    failing = api(tmp_path, lambda: Answerer(searcher(), client))
    assert failing.post("/v1/ask", json={"question": "Sea level?"}).status_code == 502
