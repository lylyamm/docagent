"""Translation with any OpenAI-compatible chat completions API.

Mistral, Groq, OpenRouter, Ollama and vLLM all expose ``POST /chat/completions``
with the same request and response shape, so one class covers them; the
provider is a setting (see ``config.py``).

One request per page (split if the page is long): the model sees the blocks in
reading order, which gives it context, and returns JSON ``{"translations":
[{"id": 0, "text": "..."}, ...]}`` validated with Pydantic. Rate limits, server
errors, timeouts and invalid answers are retried with exponential backoff, and
requests are spaced out to stay under free-tier rate limits.
"""

import json
import logging
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, field

import httpx
from pydantic import BaseModel, ValidationError

from ..config import Settings, get_settings
from .base import Translator
from .glossary import Glossary
from .tags import prune_tags, restore_tags

logger = logging.getLogger(__name__)

PROMPT_VERSION = "v3"

LANGUAGES = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
}

SYSTEM_PROMPT = """You are a professional translator specialised in climate and energy \
reports (IPCC, IEA, national climate councils). You translate text blocks extracted \
from a PDF page, given in reading order.

Rules:
- Translate every block from {source} to {target}. Return exactly one translation per id.
- Blocks may contain style tags like <s1>...</s1> (colour, italics, superscript...). Keep \
every tag, unchanged, around the words that carry that style in the translation. Never \
add, remove or rename them, and never tag words that are untagged in the source (not even \
glossary terms or acronyms). Do not add HTML tags (<sub>, <b>, <i>...), except <sup> for \
French ordinal suffixes (XXIe siècle -> XXI<sup>e</sup> siècle).
- Keep numbers, units, years, citations such as {{2.3, TS.2.6}} and acronyms \
(IPCC, GHG, CO2, NDC...) unless a standard translation exists (IPCC -> GIEC).
- Headings stay short: do not add explanations. Do not merge or split blocks.
- If a block is a proper name, a code or already in {target}, copy it unchanged.
{glossary}
Answer with JSON only: {{"translations": [{{"id": <int>, "text": "<translation>"}}, ...]}}"""


class TranslationItem(BaseModel):
    id: int
    text: str


class TranslationBatch(BaseModel):
    translations: list[TranslationItem]


class InvalidAnswer(ValueError):
    """The model answered, but not with a usable translation batch."""


@dataclass
class UsageStats:
    requests: int = 0
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    kept_locally: int = 0  # codes and exact glossary entries, not sent to the model
    missing_asked_again: int = 0  # blocks the model skipped and that were requested again
    tags_restored: int = 0  # broken style tags fixed locally (no API call)
    tag_repairs: int = 0  # broken style tags fixed by asking the model again
    tag_fallbacks: int = 0  # translations whose style tags stayed broken and were dropped
    errors: Counter = field(default_factory=Counter)


_TAG_RE = re.compile(r"</?s\d+>")


def _tags(text: str) -> Counter:
    return Counter(_TAG_RE.findall(text))


def _strip(text: str) -> str:
    return _TAG_RE.sub("", text)


def _opening(text: str) -> list[tuple[str, int]]:
    """Opening tags of a text and how many times each appears: [("<s1>", 2)]."""
    counts = Counter(t for t in _TAG_RE.findall(text) if not t.startswith("</"))
    return sorted(counts.items())


# A code or acronym alone in its block ("MED", "NWN", "CO2", "SSP1-2.6"): kept as is.
_CODE_RE = re.compile(r"[A-Z][A-Z0-9]{1,5}(?:[-.][A-Z0-9.]{1,5})?")

# Requests for the blocks a model skipped, before giving up on a page.
MISSING_ROUNDS = 3


class ChatTranslator(Translator):
    def __init__(
        self,
        settings: Settings | None = None,
        glossary: Glossary | None = None,
        client: httpx.Client | None = None,
        sleep=time.sleep,
        clock=time.monotonic,
    ) -> None:
        self.settings = settings or get_settings()
        self.base_url = self.settings.resolved_base_url
        self.model = self.settings.resolved_model
        self._api_key = self.settings.resolved_api_key  # raises if a required key is missing
        self.name = f"{self.settings.llm_provider}:{self.model}:{PROMPT_VERSION}"
        # An explicit glossary wins; otherwise GLOSSARY_PATH, otherwise the file
        # for the language pair in GLOSSARY_DIR (data/glossary_en_fr.json).
        self.glossary = glossary
        self._glossaries: dict[tuple[str, str], Glossary] = {}
        self.client = client or httpx.Client(timeout=self.settings.llm_timeout_s)
        self.sleep = sleep
        self.clock = clock
        self._last_request = float("-inf")
        self.stats = UsageStats()

    # -- public API --------------------------------------------------------------

    def translate_blocks(self, texts: list[str], source: str, target: str) -> list[str]:
        """Translate a page's blocks; codes and exact glossary entries never reach the model.

        A map full of region codes ("MED", "SAH", "NWN") must keep them: the
        model "helpfully" expanded them ("Méditerranée", and even a wrong
        "Afrique de l'Est" for SAH), which also no longer fit their hexagon.
        """
        glossary = self.glossary_for(source, target).terms
        local: dict[int, str] = {}
        for i, text in enumerate(texts):
            if text.strip() in glossary:
                local[i] = glossary[text.strip()]
            elif _CODE_RE.fullmatch(text.strip()):
                local[i] = text.strip()
        todo = [i for i in range(len(texts)) if i not in local]
        self.stats.kept_locally += len(local)
        results = dict(local)
        translated: list[str] = []
        for batch in self._batches([texts[i] for i in todo]):
            translated.extend(self._translate_batch(batch, source, target))
        results.update(zip(todo, translated, strict=True))
        return [results[i] for i in range(len(texts))]

    def glossary_for(self, source: str, target: str) -> Glossary:
        if self.glossary is not None:
            return self.glossary
        if (source, target) not in self._glossaries:
            path = self.settings.glossary_path
            if path is None and self.settings.glossary_dir is not None:
                candidate = self.settings.glossary_dir / f"glossary_{source}_{target}.json"
                path = candidate if candidate.exists() else None
            self._glossaries[(source, target)] = Glossary.load(path)
        return self._glossaries[(source, target)]

    # -- internals ---------------------------------------------------------------

    def _batches(self, texts: list[str]) -> list[list[str]]:
        """Split a page's blocks into chunks (``llm_batch_chars``, ``llm_batch_items``)."""
        batches: list[list[str]] = []
        current: list[str] = []
        size = 0
        for text in texts:
            full = len(current) >= self.settings.llm_batch_items
            if current and (full or size + len(text) > self.settings.llm_batch_chars):
                batches.append(current)
                current, size = [], 0
            current.append(text)
            size += len(text)
        if current:
            batches.append(current)
        return batches

    def _messages(self, texts: list[str], source: str, target: str) -> list[dict]:
        terms = self.glossary_for(source, target).relevant(texts)
        glossary = ""
        if terms:
            lines = "\n".join(f"  - {src} -> {dst}" for src, dst in terms.items())
            glossary = f"- Use these imposed translations:\n{lines}\n"
        system = SYSTEM_PROMPT.format(
            source=LANGUAGES.get(source, source),
            target=LANGUAGES.get(target, target),
            glossary=glossary,
        )
        items = [{"id": i, "text": t} for i, t in enumerate(texts)]
        user = json.dumps({"blocks": items}, ensure_ascii=False)
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _translate_batch(self, texts: list[str], source: str, target: str) -> list[str]:
        results = self._translate_all(texts, source, target)
        broken = [i for i, src in enumerate(texts) if _tags(results[i]) != _tags(src)]
        if broken:
            results = self._repair_tags(results, broken, texts, source, target)
        return results

    def _translate_all(self, texts: list[str], source: str, target: str) -> list[str]:
        """Translations of every block; blocks the model skipped are asked again.

        On long pages (a chart with 70 labels) small models sometimes stop
        before the last ids. Retrying the whole page wastes the part that was
        fine, so only the missing blocks are sent again.
        """
        results: dict[int, str] = {}
        todo = list(range(len(texts)))
        for _ in range(MISSING_ROUNDS):
            subset = [texts[i] for i in todo]
            answer = self._request(self._messages(subset, source, target), len(subset))
            results.update({todo[k]: text for k, text in answer.items()})
            todo = [i for i in todo if i not in results]
            if not todo:
                return [results[i] for i in range(len(texts))]
            self.stats.missing_asked_again += len(todo)
            logger.warning("the model skipped %d blocks, asking again for them", len(todo))
        raise RuntimeError(f"blocks {todo[:5]} still untranslated after {MISSING_ROUNDS} requests")

    def _request(self, messages: list[dict], n: int) -> dict[int, str]:
        """One translation request, retried on errors; returns ``{id: translation}``."""
        attempts = self.settings.llm_max_retries + 1
        for attempt in range(attempts):
            try:
                return self._parse(self._call(messages), n)
            except (httpx.TransportError, InvalidAnswer, _RetryableHTTP) as exc:
                self.stats.errors[type(exc).__name__] += 1
                if attempt == attempts - 1:
                    raise RuntimeError(f"translation failed after {attempts} attempts") from exc
                delay = self._delay(attempt, exc)
                logger.warning(
                    "translation attempt %d failed (%s), retrying in %.1fs",
                    attempt + 1,
                    exc,
                    delay,
                )
                self.stats.retries += 1
                self.sleep(delay)
        raise AssertionError("unreachable")

    def _repair_tags(
        self, results: list[str], broken: list[int], texts: list[str], source: str, target: str
    ) -> list[str]:
        """Fix translations whose style tags came back broken.

        1. fix them locally (``tags.py``): drop the tags the model added, or put
           back the ones it lost when the runs can be found;
        2. otherwise ask the model once more for those blocks only, listing the
           tags each one must contain;
        3. what is still broken is kept without tags: the text matters more
           than its styles, and one bad block must not fail the page. Those
           cases are logged to ``tag_failures.jsonl`` next to the cache.
        """
        results = list(results)
        terms = self.glossary_for(source, target).relevant([texts[i] for i in broken])
        remaining = []
        for i in broken:
            restored = prune_tags(texts[i], results[i], terms) or restore_tags(
                texts[i], results[i], terms
            )
            if restored is not None:
                self.stats.tags_restored += 1
                results[i] = restored
            else:
                remaining.append(i)
        if not remaining:
            return results

        subset = [texts[i] for i in remaining]
        messages = self._messages(subset, source, target)
        required = "\n".join(
            f"- id {k}: " + ", ".join(f"{tag}…{tag.replace('<', '</')} x{n}" for tag, n in tags)
            for k, tags in enumerate(_opening(t) for t in subset)
        )
        messages[-1]["content"] += (
            "\nIn a previous answer the style tags of these blocks were lost or changed. "
            "Each translation must contain exactly these tags, around the matching words:\n"
            + required
        )
        try:
            answer = self._request(messages, len(subset))
        except RuntimeError:
            answer = {}
        for k, i in enumerate(remaining):
            retry = answer.get(k)
            if retry is not None and _tags(retry) != _tags(texts[i]):
                retry = prune_tags(texts[i], retry, terms) or restore_tags(texts[i], retry, terms)
            if retry is not None and _tags(retry) == _tags(texts[i]):
                self.stats.tag_repairs += 1
                results[i] = retry
            else:
                self.stats.tag_fallbacks += 1
                self._log_tag_failure(texts[i], results[i], answer.get(k))
                results[i] = _strip(results[i])
        return results

    def _log_tag_failure(self, source: str, first: str, retry: str | None) -> None:
        if not self.settings.translation_cache:
            return
        path = self.settings.translation_cache.parent / "tag_failures.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"model": self.model, "source": source, "first": first, "retry": retry}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _throttle(self) -> None:
        """Wait so that two requests are at least ``llm_min_interval_s`` apart."""
        wait = self._last_request + self.settings.llm_min_interval_s - self.clock()
        if wait > 0:
            self.sleep(wait)
        self._last_request = self.clock()

    def _call(self, messages: list[dict]) -> str:
        self._throttle()
        self.stats.requests += 1
        headers = {"Content-Type": "application/json"}
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key.get_secret_value()}"
        response = self.client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
            },
        )
        if response.status_code == 429 or response.status_code >= 500:
            raise _RetryableHTTP(response)
        if response.is_error:  # 4xx other than 429: a bug or a bad key, don't retry
            raise httpx.HTTPStatusError(
                f"HTTP {response.status_code}: {_error_message(response)}",
                request=response.request,
                response=response,
            )
        payload = response.json()
        usage = payload.get("usage") or {}
        self.stats.input_tokens += usage.get("prompt_tokens", 0)
        self.stats.output_tokens += usage.get("completion_tokens", 0)
        return payload["choices"][0]["message"]["content"]

    def _parse(self, content: str, n: int) -> dict[int, str]:
        """``{id: translation}`` for the ids 0..n-1 found in the answer.

        Invalid JSON, or an answer without a single expected id, is an error
        (retried); a partial answer is returned as is.
        """
        try:
            batch = TranslationBatch.model_validate_json(content)
        except ValidationError as exc:
            raise InvalidAnswer(f"invalid JSON answer: {exc.errors()[:1]}") from exc
        found = {item.id: item.text.strip() for item in batch.translations if 0 <= item.id < n}
        if not found:
            raise InvalidAnswer("no expected id in the answer")
        return found

    def _delay(self, attempt: int, exc: Exception) -> float:
        """Exponential backoff with jitter, honouring Retry-After when given.

        Rate limits (429) start at 5 s: free tiers count requests per minute,
        so retrying after 1-2 s only burns the next attempts.
        """
        if isinstance(exc, _RetryableHTTP):
            retry_after = exc.response.headers.get("retry-after")
            if retry_after:
                try:
                    return min(float(retry_after), 60.0)
                except ValueError:
                    pass
            if exc.response.status_code == 429:
                return min(5 * 2**attempt, 60) + random.uniform(0, 2)
        return min(2**attempt, 30) + random.uniform(0, 1)


class _RetryableHTTP(Exception):
    def __init__(self, response: httpx.Response) -> None:
        super().__init__(f"HTTP {response.status_code}: {_error_message(response)}")
        self.response = response


def _error_message(response: httpx.Response) -> str:
    """The API's own explanation (rate limit, capacity, plan...), shortened."""
    try:
        body = response.json()
        message = body.get("message") or body.get("detail") or body
    except ValueError:
        message = response.text
    return str(message)[:200]
