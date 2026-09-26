"""Tests for the translation layer. The LLM API is mocked: no network, no cost."""

import json

import httpx
import pytest

from docagent.config import Settings
from docagent.translate import (
    CachedTranslator,
    FakeTranslator,
    Glossary,
    Translator,
    fake_translate,
)
from docagent.translate.llm import ChatTranslator


def make_settings(**overrides) -> Settings:
    values = {
        "llm_provider": "mistral",
        "mistral_api_key": "test-key",
        "llm_max_retries": 3,
        "llm_min_interval_s": 0,
        "translation_cache": None,
        "glossary_path": None,
        "glossary_dir": None,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def answer(translations: list[str], status: int = 200, **headers) -> httpx.Response:
    content = json.dumps(
        {"translations": [{"id": i, "text": t} for i, t in enumerate(translations)]}
    )
    return httpx.Response(
        status,
        headers=headers,
        json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        },
    )


def translator_with(responses: list, **settings) -> tuple[ChatTranslator, list[dict]]:
    """A ChatTranslator whose HTTP calls return ``responses`` in order."""
    requests: list[dict] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    client = httpx.Client(transport=httpx.MockTransport(handler))
    translator = ChatTranslator(make_settings(**settings), client=client, sleep=lambda s: None)
    return translator, requests


# --- LLM translator ----------------------------------------------------------


def test_translates_a_page_in_one_request():
    translator, requests = translator_with([answer(["Le climat", "Le réchauffement"])])
    result = translator.translate_blocks(["The climate", "Warming"], "en", "fr")
    assert result == ["Le climat", "Le réchauffement"]
    assert len(requests) == 1
    body = requests[0]
    assert body["model"] == "mistral-small-latest"
    assert body["response_format"] == {"type": "json_object"}
    blocks = json.loads(body["messages"][1]["content"])["blocks"]
    assert blocks == [{"id": 0, "text": "The climate"}, {"id": 1, "text": "Warming"}]
    assert translator.stats.input_tokens == 100 and translator.stats.output_tokens == 50


def test_retries_on_rate_limit_and_server_error():
    translator, requests = translator_with(
        [
            httpx.Response(429, headers={"retry-after": "1"}),
            httpx.Response(503),
            answer(["Bonjour"]),
        ]
    )
    assert translator.translate_blocks(["Hello"], "en", "fr") == ["Bonjour"]
    assert len(requests) == 3 and translator.stats.retries == 2


def test_retries_on_invalid_json():
    bad_json = httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]})
    translator, requests = translator_with([bad_json, answer(["Un", "Deux"])])
    assert translator.translate_blocks(["One", "Two"], "en", "fr") == ["Un", "Deux"]
    assert len(requests) == 2


def test_only_the_skipped_blocks_are_asked_again():
    # The model stopped after id 0: only "Two" and "Three" are sent again.
    translator, requests = translator_with([answer(["Un"]), answer(["Deux", "Trois"])])
    assert translator.translate_blocks(["One", "Two", "Three"], "en", "fr") == [
        "Un",
        "Deux",
        "Trois",
    ]
    second = json.loads(requests[1]["messages"][1]["content"])["blocks"]
    assert [b["text"] for b in second] == ["Two", "Three"]
    assert translator.stats.missing_asked_again == 2 and translator.stats.retries == 0


def test_codes_and_exact_glossary_entries_never_reach_the_model(tmp_path):
    (tmp_path / "glossary_en_fr.json").write_text('{"SPM": "RID"}', encoding="utf-8")
    translator, requests = translator_with([answer(["Asie"])], glossary_dir=tmp_path)
    result = translator.translate_blocks(["MED", "SPM", "Asia", "SSP1-2.6"], "en", "fr")
    assert result == ["MED", "RID", "Asie", "SSP1-2.6"]
    sent = json.loads(requests[0]["messages"][1]["content"])["blocks"]
    assert [b["text"] for b in sent] == ["Asia"]
    assert translator.stats.kept_locally == 3


def test_long_lists_of_labels_are_split_by_count():
    translator, requests = translator_with([answer(["a", "b"]), answer(["c"])], llm_batch_items=2)
    assert translator.translate_blocks(["x", "y", "z"], "en", "fr") == ["a", "b", "c"]
    assert len(requests) == 2


def test_retries_on_network_error():
    translator, _ = translator_with([httpx.ConnectError("boom"), answer(["Oui"])])
    assert translator.translate_blocks(["Yes"], "en", "fr") == ["Oui"]


def test_gives_up_after_max_retries():
    translator, requests = translator_with([httpx.Response(503)] * 4, llm_max_retries=3)
    with pytest.raises(RuntimeError, match="after 4 attempts"):
        translator.translate_blocks(["Hello"], "en", "fr")
    assert len(requests) == 4


def test_does_not_retry_a_bad_key():
    translator, requests = translator_with([httpx.Response(401)])
    with pytest.raises(httpx.HTTPStatusError):
        translator.translate_blocks(["Hello"], "en", "fr")
    assert len(requests) == 1


def test_lost_tags_are_restored_locally_first():
    # The footnote call stays glued to the word: no second request needed.
    translator, requests = translator_with([answer(["La température à la surface8 a augmenté"])])
    result = translator.translate_blocks(["Surface temperature<s1>8</s1> increased"], "en", "fr")
    assert result == ["La température à la surface<s1>8</s1> a augmenté"]
    assert len(requests) == 1 and translator.stats.tags_restored == 1


def test_broken_style_tags_are_asked_again():
    translator, requests = translator_with(
        [
            answer(["<s1>Au cours de la décennie,</s1> le réchauffement", "Le réchauffement"]),
            answer(["Le réchauffement <s2>probable</s2>"]),  # only the broken block
        ]
    )
    result = translator.translate_blocks(
        ["<s1>Over the decade,</s1> warming", "<s2>Likely</s2> warming"], "en", "fr"
    )
    assert result[1] == "Le réchauffement <s2>probable</s2>"
    assert len(requests) == 2
    retry = requests[1]["messages"][1]["content"]
    assert json.loads(retry.split("\n")[0])["blocks"] == [
        {"id": 0, "text": "<s2>Likely</s2> warming"}
    ]
    assert "- id 0: <s2>…</s2> x1" in retry  # the tags it must contain are listed
    assert translator.stats.tag_repairs == 1 and translator.stats.tag_fallbacks == 0


def test_tags_still_broken_are_dropped_cleanly(tmp_path):
    cache = tmp_path / "translations.json"
    translator, _ = translator_with(
        [answer(["ok", "<s2>C'est cassé"]), answer(["<s2>toujours"])], translation_cache=cache
    )
    result = translator.translate_blocks(["ok", "It is <s2>badly</s2> broken"], "en", "fr")
    assert result == ["ok", "C'est cassé"]  # text kept, styles dropped
    assert translator.stats.tag_fallbacks == 1
    failure = json.loads((tmp_path / "tag_failures.jsonl").read_text(encoding="utf-8"))
    assert failure["first"] == "<s2>C'est cassé" and failure["retry"] == "<s2>toujours"


def test_long_pages_are_split_into_batches():
    translator, requests = translator_with([answer(["a", "b"]), answer(["c"])], llm_batch_chars=10)
    assert translator.translate_blocks(["xxxxx", "yyyyy", "zzzzz"], "en", "fr") == ["a", "b", "c"]
    assert len(requests) == 2


def test_glossary_terms_are_injected_only_when_relevant():
    glossary = Glossary({"carbon budget": "budget carbone", "net zero": "zéro émission nette"})
    translator, requests = translator_with([answer(["x"])])
    translator.glossary = glossary
    translator.translate_blocks(["The carbon budget was respected."], "en", "fr")
    system = requests[0]["messages"][0]["content"]
    assert "carbon budget -> budget carbone" in system
    assert "net zero" not in system


def test_glossary_of_the_language_pair_is_found_in_glossary_dir(tmp_path):
    (tmp_path / "glossary_en_fr.json").write_text('{"SPM": "RID"}', encoding="utf-8")
    translator, requests = translator_with([answer(["x"]), answer(["y"])], glossary_dir=tmp_path)
    translator.translate_blocks(["This SPM"], "en", "fr")
    translator.translate_blocks(["This SPM"], "en", "de")  # no file for en->de
    assert "SPM -> RID" in requests[0]["messages"][0]["content"]
    assert "SPM -> RID" not in requests[1]["messages"][0]["content"]


def test_glossary_matches_whole_words_and_acronyms_by_case():
    glossary = Glossary({"likely": "probable", "TS": "RT"})
    assert glossary.relevant(["unlikely results"]) == {}
    assert glossary.relevant(["It is likely (see TS.2)"]) == {"likely": "probable", "TS": "RT"}


def test_missing_api_key_is_a_clear_error():
    with pytest.raises(RuntimeError, match="MISTRAL_API_KEY"):
        ChatTranslator(make_settings(mistral_api_key=None))


@pytest.mark.parametrize(
    ("provider", "url", "model"),
    [
        ("mistral", "https://api.mistral.ai/v1", "mistral-small-latest"),
        ("groq", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
        ("ollama", "http://localhost:11434/v1", "qwen2.5:7b"),
    ],
)
def test_provider_presets(provider, url, model):
    settings = make_settings(llm_provider=provider, groq_api_key="k")
    assert settings.resolved_base_url == url
    assert settings.resolved_model == model


def test_ollama_needs_no_key_and_sends_no_auth_header():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["url"] = str(request.url)
        return answer(["Salut"])

    translator = ChatTranslator(
        make_settings(llm_provider="ollama", mistral_api_key=None),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert translator.translate_blocks(["Hi"], "en", "fr") == ["Salut"]
    assert seen["auth"] is None
    assert seen["url"] == "http://localhost:11434/v1/chat/completions"


def test_openrouter_requires_a_model():
    with pytest.raises(RuntimeError, match="LLM_MODEL"):
        ChatTranslator(make_settings(llm_provider="openrouter", openrouter_api_key="k"))


def test_model_override_and_legacy_mistral_model():
    assert make_settings(llm_model="open-mistral-nemo").resolved_model == "open-mistral-nemo"
    assert make_settings(mistral_model="mistral-medium-latest").resolved_model == (
        "mistral-medium-latest"
    )


def test_requests_are_spaced_out():
    """Free tiers allow about one request per second: the client waits between calls."""
    now = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    queue = [answer(["a"]), answer(["b"])]
    translator = ChatTranslator(
        make_settings(llm_min_interval_s=1.5),
        client=httpx.Client(transport=httpx.MockTransport(lambda r: queue.pop(0))),
        sleep=sleep,
        clock=lambda: now[0],
    )
    translator.translate_blocks(["x"], "en", "fr")
    translator.translate_blocks(["y"], "en", "fr")
    assert sleeps == [1.5]


def test_rate_limit_backoff_is_patient():
    translator, _ = translator_with([answer(["x"])])
    response = httpx.Response(429, request=httpx.Request("POST", "http://x"))
    from docagent.translate.llm import _RetryableHTTP

    delays = [translator._delay(a, _RetryableHTTP(response)) for a in range(5)]
    assert delays[0] >= 5 and delays[3] >= 40 and all(d <= 62 for d in delays)


# --- cache and glossary --------------------------------------------------------


class CountingTranslator(Translator):
    name = "counting"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def translate_blocks(self, texts, source, target):
        self.calls.append(list(texts))
        return [t.upper() for t in texts]


def test_cache_avoids_repeated_calls(tmp_path):
    inner = CountingTranslator()
    cache = CachedTranslator(inner, tmp_path / "cache.json")
    assert cache.translate_blocks(["a", "b"], "en", "fr") == ["A", "B"]
    assert cache.translate_blocks(["b", "c"], "en", "fr") == ["B", "C"]
    assert inner.calls == [["a", "b"], ["c"]]  # "b" came from the cache
    # A new instance reads the file written by the first one.
    again = CachedTranslator(CountingTranslator(), tmp_path / "cache.json")
    assert again.translate_blocks(["a"], "en", "fr") == ["A"] and again.hits == 1


def test_cache_separates_language_pairs(tmp_path):
    inner = CountingTranslator()
    cache = CachedTranslator(inner, tmp_path / "cache.json")
    cache.translate_blocks(["a"], "en", "fr")
    cache.translate_blocks(["a"], "en", "de")
    assert len(inner.calls) == 2


def test_glossary_file_is_validated(tmp_path):
    good = tmp_path / "g.json"
    good.write_text('{"IPCC": "GIEC"}', encoding="utf-8")
    assert Glossary.load(good).relevant(["The IPCC report"]) == {"IPCC": "GIEC"}
    bad = tmp_path / "bad.json"
    bad.write_text('["IPCC"]', encoding="utf-8")
    with pytest.raises(ValueError):
        Glossary.load(bad)


def test_fake_translator_matches_function():
    assert FakeTranslator().translate_blocks(["Le climat"], "fr", "en") == [
        fake_translate("Le climat")
    ]


def test_full_pipeline_with_mocked_llm(tmp_path):
    """PDF -> blocks -> one API call per page -> rebuilt PDF, without network."""
    from pathlib import Path

    import pymupdf

    from docagent.pdf import rebuild_document

    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        blocks = json.loads(json.loads(request.content)["messages"][1]["content"])["blocks"]
        calls.append(len(blocks))
        return answer([f"FR {b['text']}" for b in blocks])

    translator = ChatTranslator(
        make_settings(), client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    sample = Path(__file__).parent.parent / "data" / "samples" / "giec_spm_en.pdf"
    if not sample.exists():
        pytest.skip("GIEC sample missing")
    with pymupdf.open(sample) as src, pymupdf.open() as two_pages:
        two_pages.insert_pdf(src, from_page=0, to_page=1)
        two_pages.save(tmp_path / "in.pdf")
    report = rebuild_document(tmp_path / "in.pdf", tmp_path / "out.pdf", translator)
    assert len(calls) == 2  # one request per page
    # The side tabs "SPM" are codes: kept without asking the model.
    assert report.targets == sum(calls) + translator.stats.kept_locally == sum(calls) + 2
    assert "FR " in pymupdf.open(tmp_path / "out.pdf")[0].get_text()
