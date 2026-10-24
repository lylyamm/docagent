"""The pieces that connect the others, found untested by the coverage report (S5):
building the search index, opening it for answers, choosing the translator, and
the Google sign-in. No network: hash embedder, embedded Qdrant in a temporary
folder, scripted LLM replies, a fake Google credential."""

import json
from pathlib import Path

import pytest
from test_agent_tools import make_pdf, settings
from test_answer import chat

from docagent.gcp import google_token
from docagent.rag import build_answerer
from docagent.rag.chunking import Chunker
from docagent.rag.embeddings import HashEmbedder
from docagent.rag.index import (
    build_passages,
    index_vectors,
    load_corpus,
    load_passages,
    open_store,
    save_passages,
)
from docagent.translate import FakeTranslator
from docagent.translate.cache import CachedTranslator
from docagent.translate.factory import build_translator, llm_stats
from docagent.translate.llm import ChatTranslator

SEA = (
    "Global mean sea level increased by 0.20 m between 1901 and 2018. The average rate "
    "of sea level rise was 1.3 mm per year between 1901 and 1971"
)


def corpus_and_report(tmp_path: Path) -> tuple[Path, Path]:
    raw = tmp_path / "raw"
    raw.mkdir()
    make_pdf(raw / "sea.pdf", pages=2, text=SEA)
    corpus = tmp_path / "corpus.json"
    docs = [
        {"id": "sea", "file": "sea.pdf", "title": "Sea report", "org": "X", "lang": "en"},
        {"id": "gone", "file": "not_downloaded.pdf", "title": "Missing", "org": "X"},
    ]
    corpus.write_text(json.dumps(docs), encoding="utf-8")
    return raw, corpus


def test_the_index_is_built_once_then_answers_come_from_it(tmp_path):
    raw, corpus = corpus_and_report(tmp_path)
    config = settings(embedder="hash", rag_index_dir=tmp_path / "index", rag_top_k=2)

    seen = []
    passages = build_passages(
        load_corpus(corpus), raw, Chunker(), on_doc=lambda d, n: seen.append((d.id, n))
    )
    assert [doc for doc, _ in seen] == ["sea"]  # the report not downloaded is skipped
    assert passages and all(p.doc_id == "sea" and p.lang == "en" for p in passages)
    path = config.rag_index_dir / "passages.jsonl"
    save_passages(passages, path)
    assert load_passages(path) == passages

    store = open_store(config, HashEmbedder())
    batches = []
    assert index_vectors(
        store, HashEmbedder(), passages, batch=1, on_batch=lambda *b: batches.append(b)
    )
    assert batches[-1][0] == batches[-1][1] == len(passages)
    assert index_vectors(store, HashEmbedder(), passages) == 0  # already there: incremental

    answerer = build_answerer(config)
    reply = {
        "answerable": True,
        "answer": "It rose by 0.20 m [1].",
        "evidence": [{"n": 1, "quote": "Global mean sea level increased by 0.20 m"}],
    }
    answerer.chat = chat(reply)[0]
    answer = answerer.ask("How much did sea level rise since 1901?")
    assert answer.answerable and answer.sources[0].doc_id == "sea"
    assert {"search_s", "llm_s"} <= set(answer.timings)


def test_no_index_yet_says_which_script_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="rag_index.py"):
        build_answerer(settings(rag_index_dir=tmp_path / "nothing"))


def test_the_translator_is_chosen_by_the_settings(tmp_path):
    cache = tmp_path / "translations.json"
    assert isinstance(build_translator(settings(), fake=True), FakeTranslator)

    cached = build_translator(settings(translation_cache=cache))
    assert isinstance(cached, CachedTranslator) and isinstance(cached.inner, ChatTranslator)
    assert llm_stats(cached) is cached.inner.stats  # the stats of the model behind the cache

    plain = build_translator(settings(translation_cache=cache), use_cache=False)
    assert isinstance(plain, ChatTranslator) and llm_stats(plain) is plain.stats
    assert llm_stats(FakeTranslator()) is None


class FakeCredentials:
    """Like google.auth credentials: a token valid until refreshed again."""

    def __init__(self) -> None:
        self.valid, self.token, self.refreshes = False, None, 0

    def refresh(self, request) -> None:
        self.refreshes += 1
        self.valid, self.token = True, f"tok-{self.refreshes}"


def test_the_google_token_is_fetched_then_refreshed_only_when_expired(monkeypatch):
    google = pytest.importorskip("google.auth", reason="the vertex extra is not installed")

    credentials, asked = FakeCredentials(), {}

    def default(scopes, quota_project_id):
        asked.update(scopes=scopes, project=quota_project_id)
        return credentials, "projetlilia"

    monkeypatch.setattr(google, "default", default)
    token = google_token("projetlilia")
    assert asked == {
        "scopes": ["https://www.googleapis.com/auth/cloud-platform"],
        "project": "projetlilia",
    }
    assert token() == "tok-1" and token() == "tok-1"  # still valid: no new request
    credentials.valid = False  # about an hour later
    assert token() == "tok-2" and credentials.refreshes == 2


def test_no_google_sign_in_says_how_to_sign_in(monkeypatch):
    google = pytest.importorskip("google.auth", reason="the vertex extra is not installed")
    from google.auth.exceptions import DefaultCredentialsError

    def default(**_):
        raise DefaultCredentialsError("no credentials")

    monkeypatch.setattr(google, "default", default)
    with pytest.raises(RuntimeError, match="gcloud auth application-default login"):
        google_token("projetlilia")
