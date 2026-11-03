"""Tracing with Langfuse, with a fake client: what each request sends (no network)."""

import json
from contextlib import contextmanager

import httpx
import pytest
from fastapi.testclient import TestClient

from docagent import observability
from docagent.api import create_app
from docagent.chat import ChatClient, tools_chat_client
from docagent.config import Settings


class FakeSpan:
    def __init__(self, record: dict) -> None:
        self.record = record

    def update(self, **fields) -> None:
        self.record.update(fields)


class FakeLangfuse:
    """Records observations (name, type, fields) and scores, like the SDK would send."""

    def __init__(self) -> None:
        self.observations: list[dict] = []
        self.scores: list[tuple[str, float]] = []

    @contextmanager
    def start_as_current_observation(self, *, name, as_type="span", **fields):
        record = {"name": name, "type": as_type, **fields}
        self.observations.append(record)
        yield FakeSpan(record)

    def score_current_trace(self, *, name, value, comment=None) -> None:
        self.scores.append((name, value))


@pytest.fixture
def langfuse(monkeypatch):
    fake = FakeLangfuse()
    monkeypatch.setattr(observability, "enabled", lambda settings=None: True)
    monkeypatch.setattr(observability, "_client", fake)
    return fake


def settings(**more) -> Settings:
    return Settings(
        _env_file=None,
        mistral_api_key="k",
        langfuse_public_key="pk-lf-test",
        langfuse_secret_key="sk-lf-test",
        llm_min_interval_s=0,
        **more,
    )


def chat_reply(content: dict, tokens=(1000, 200), status: int = 200) -> httpx.Response:
    if status != 200:
        return httpx.Response(status, json={"message": "busy"})
    body = {
        "choices": [{"message": {"content": json.dumps(content)}}],
        "usage": {"prompt_tokens": tokens[0], "completion_tokens": tokens[1]},
    }
    return httpx.Response(200, json=body)


def test_without_keys_tracing_is_off_and_nothing_is_created():
    s = Settings(_env_file=None)
    assert observability.client(s) is None
    with observability.trace("ask", settings=s) as span:
        assert span is None
    assert observability.callbacks(s) == []


def test_each_llm_call_is_a_generation_with_its_tokens_and_cost(langfuse):
    from pydantic import BaseModel

    class Reply(BaseModel):
        ok: bool

    s = settings(tools_price_in_usd=0.30, tools_price_out_usd=2.50)
    replies = iter([chat_reply({}, status=503), chat_reply({"ok": True})])
    client = tools_chat_client(s)
    client.client = httpx.Client(transport=httpx.MockTransport(lambda r: next(replies)))
    client.sleep = lambda s: None
    client.complete_json([{"role": "user", "content": "hi"}], Reply)

    failed, answered = langfuse.observations
    assert failed["type"] == answered["type"] == "generation"
    assert failed["name"] == answered["name"] == "tools"
    assert failed["level"] == "WARNING" and "503" in failed["status_message"]  # retried
    assert answered["usage_details"] == {"input": 1000, "output": 200}
    assert answered["cost_details"]["total"] == pytest.approx(0.0003 + 0.0005)
    assert answered["input"] == [{"role": "user", "content": "hi"}]


def test_a_call_without_prices_has_tokens_but_no_cost(langfuse):
    from pydantic import BaseModel

    class Reply(BaseModel):
        ok: bool

    client = ChatClient(settings(), role="translator")
    client.client = httpx.Client(transport=httpx.MockTransport(lambda r: chat_reply({"ok": 1})))
    client.complete_json([{"role": "user", "content": "x"}], Reply)
    (gen,) = langfuse.observations
    assert gen["name"] == "translator" and gen["cost_details"] is None


def test_a_question_is_one_trace_scored_answered(langfuse, tmp_path):
    from docagent.rag import Answer

    class Answerer:
        def ask(self, question, top_k=None):
            return Answer(
                question=question,
                answer="0.20 m [1].",
                answerable=True,
                sources=[],
                retrieved=[],
                model="fake",
            )

    s = settings(jobs_dir=tmp_path / "jobs", translator="fake")
    http = TestClient(create_app(s, answerer_factory=Answerer))
    assert http.post("/v1/ask", json={"question": "Sea level?"}).status_code == 200
    (trace,) = langfuse.observations
    assert (trace["name"], trace["input"], trace["output"]) == ("ask", "Sea level?", "0.20 m [1].")
    assert langfuse.scores == [("answered", 1.0)]


def test_an_agent_run_is_scored_with_the_judges_verdicts(langfuse):
    from docagent.agent.trajectory import NodeRecord, Recorder

    s = settings()
    recorder = Recorder("Résume le HCC", ChatClient(s, role="supervisor"), s)
    recorder.trajectory.nodes = [
        NodeRecord(node="summarize", duration_s=1.0, items=4),
        NodeRecord(node="judge", duration_s=1.0, kept=3, removed=1),
    ]
    recorder._scores()
    assert langfuse.scores == [("judge_kept_ratio", 0.75), ("judge_removed", 1.0)]


def test_cost_needs_both_prices():
    assert observability.cost_usd(10, 10, None) is None
    assert observability.cost_usd(1_000_000, 0, (2.0, 5.0)) == {
        "input": 2.0,
        "output": 0.0,
        "total": 2.0,
    }


class BrokenLangfuse(FakeLangfuse):
    """An SDK that fails everywhere: tracing must step aside, never break a request."""

    def start_as_current_observation(self, **fields):
        raise RuntimeError("SDK broken")

    def score_current_trace(self, **fields):
        raise RuntimeError("SDK broken")


def test_a_broken_sdk_never_breaks_a_question_or_an_llm_call(monkeypatch, tmp_path):
    from pydantic import BaseModel

    from docagent.rag import Answer

    monkeypatch.setattr(observability, "enabled", lambda settings=None: True)
    monkeypatch.setattr(observability, "_client", BrokenLangfuse())

    class Reply(BaseModel):
        ok: bool

    client = ChatClient(settings(), role="tools")
    client.client = httpx.Client(transport=httpx.MockTransport(lambda r: chat_reply({"ok": 1})))
    assert client.complete_json([{"role": "user", "content": "x"}], Reply).ok

    class Answerer:
        def ask(self, question, top_k=None):
            return Answer(
                question=question, answer="ok", answerable=True, sources=[], retrieved=[], model="m"
            )

    s = settings(jobs_dir=tmp_path / "jobs", translator="fake")
    http = TestClient(create_app(s, answerer_factory=Answerer))
    assert http.post("/v1/ask", json={"question": "Sea level?"}).json()["answer"] == "ok"


def test_the_traced_codes_own_error_goes_through(langfuse):
    with pytest.raises(ValueError, match="mine"), observability.trace("ask", settings=settings()):
        raise ValueError("mine")


def test_the_real_sdk_nests_the_llm_calls_under_the_request(monkeypatch):
    """The real Langfuse SDK, its spans kept in memory instead of sent: one trace
    "ask", the LLM call as its child, with model and tokens."""
    from langfuse import Langfuse
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from pydantic import BaseModel

    class Reply(BaseModel):
        ok: bool

    exporter = InMemorySpanExporter()
    real = Langfuse(
        public_key="pk-lf-test",
        secret_key="sk-lf-test",
        host="http://127.0.0.1:9",  # nothing is sent: spans go to the exporter
        span_exporter=exporter,
        flush_at=1,
    )
    monkeypatch.setattr(observability, "enabled", lambda settings=None: True)
    monkeypatch.setattr(observability, "_client", real)

    s = settings(tools_price_in_usd=0.30, tools_price_out_usd=2.50)
    client = tools_chat_client(s)
    client.client = httpx.Client(transport=httpx.MockTransport(lambda r: chat_reply({"ok": 1})))
    with observability.trace("ask", input="Sea level?", settings=s) as span:
        client.complete_json([{"role": "user", "content": "x"}], Reply)
        span.update(output="0.20 m")
    real.flush()

    spans = {sp.name: sp for sp in exporter.get_finished_spans()}
    assert {"ask", "tools"} <= set(spans)
    assert spans["tools"].parent.span_id == spans["ask"].context.span_id
    attributes = dict(spans["tools"].attributes)
    assert attributes["langfuse.observation.type"] == "generation"
    assert attributes["langfuse.trace.name"] == "ask"
    assert json.loads(attributes["langfuse.observation.usage_details"]) == {
        "input": 1000,
        "output": 200,
    }
    cost = json.loads(attributes["langfuse.observation.cost_details"])
    assert cost["total"] == pytest.approx(0.0008)
