"""A small client for OpenAI-compatible chat completions, returning validated JSON.

The one client of the project: the answers (``rag/answer.py``), the agent's tools,
supervisor and judge, and the translator (``translate/llm.py``, which had its own
copy of this logic until the industrialisation week).

- one request = a list of messages -> a JSON answer validated by a Pydantic model;
- requests spaced by ``llm_min_interval_s`` (free tiers allow ~1 request/second);
- rate limits (429), server errors, timeouts and invalid JSON retried with
  exponential backoff; other 4xx errors (bad key, bad request) raised at once;
- ``check``: a test the caller adds on the parsed answer (the translator: at least
  one of the ids it asked for); an answer that fails it is retried like invalid JSON;
- counters for the cost and for monitoring: requests, answered, retries, errors by
  kind, tokens.
"""

import logging
import random
import time
from collections import Counter
from collections.abc import Callable
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .config import Settings, get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ChatError(RuntimeError):
    """The model could not give a valid answer after every retry."""


class CallBudgetExceeded(ChatError):
    """The client already sent its maximum number of requests (a cost guard)."""


def tools_chat_client(
    settings: Settings | None = None, token: Callable[[], str] | None = None
) -> "ChatClient":
    """The model that writes answers, summaries and comparisons (TOOLS_PROVIDER).

    Temperature 0 (the wording closest to the excerpts) and few retries: someone is
    waiting. On Vertex AI, a busy shared model (429) gets AGENT_RATE_LIMIT_RETRIES more
    tries, as the supervisor does. ``token``: for tests (else Google's sign-in).
    """
    settings = settings or get_settings()
    common: dict[str, Any] = {"temperature": 0.0, "max_retries": settings.ask_max_retries}
    if settings.tools_provider != "vertex":
        return ChatClient(settings, **common)
    if not settings.tools_model:
        raise RuntimeError(
            "TOOLS_MODEL is required with TOOLS_PROVIDER=vertex (e.g. google/gemini-3.5-flash-lite)"
        )
    if token is None:
        from .gcp import google_token

        token = google_token(settings.vertex_project)
    return ChatClient(
        settings,
        base_url=settings.vertex_base_url,
        model=settings.tools_model,
        token=token,
        rate_limit_retries=settings.agent_rate_limit_retries,
        **common,
    )


class _Retryable(Exception):
    def __init__(self, message: str, response: httpx.Response | None = None) -> None:
        super().__init__(message)
        self.response = response


def _kind(exc: Exception) -> str:
    """A failure's kind, for the counters: "HTTP 429", "invalid answer", "ConnectError"."""
    response = getattr(exc, "response", None)
    if response is not None:
        return f"HTTP {response.status_code}"
    if isinstance(exc, _Retryable):
        return "invalid answer"
    return type(exc).__name__


def _reason(response: httpx.Response, limit: int = 200) -> str:
    """The provider's own words for a refusal (a 429 can be a rate limit or a full
    service tier: only the body tells which)."""
    try:
        body = response.json()
    except ValueError:
        return response.text.strip()[:limit]
    if isinstance(body, list) and body:  # Google wraps some errors in a list
        body = body[0]
    if isinstance(body, dict):
        error = body.get("error")
        for value in (body.get("message"), body.get("detail"), error):
            if isinstance(value, dict):
                value = value.get("message")
            if isinstance(value, str) and value.strip():
                return value.strip()[:limit]
    return ""


class ChatClient:
    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.Client | None = None,
        sleep=time.sleep,
        clock=time.monotonic,
        temperature: float = 0.1,
        max_retries: int | None = None,
        *,
        base_url: str | None = None,
        model: str | None = None,
        token: Callable[[], str] | None = None,
        max_requests: int | None = None,
        rate_limit_retries: int | None = None,
    ) -> None:
        """``base_url``/``model``/``token`` override the settings for another provider;
        ``token`` returns a short-lived bearer token (e.g. Google's), asked before each
        request. ``max_requests``: at most that many answered requests (the ones billed;
        refused ones do not count), None = no limit. ``rate_limit_retries``: retries
        allowed for refusals for lack of capacity (429), on top of ``max_retries``: a
        shared model that is busy for a minute is not an error worth giving up on."""
        self.settings = settings or get_settings()
        self.base_url = (base_url or self.settings.resolved_base_url).rstrip("/")
        self.model = model or self.settings.resolved_model
        self._token = token
        # A fixed key, unless a token provider is given; raises if a required key is missing.
        self._api_key = None if token else self.settings.resolved_api_key
        self.max_requests = max_requests
        self.client = client or httpx.Client(timeout=self.settings.llm_timeout_s)
        self.sleep = sleep
        self.clock = clock
        self.temperature = temperature
        # Someone waiting for an answer cannot wait for 8 retries with growing pauses.
        self.max_retries = self.settings.llm_max_retries if max_retries is None else max_retries
        self._last_request = float("-inf")
        self.requests = 0  # every request sent, refused ones included
        self.answered = 0  # requests that got an answer (billed)
        self.rate_limit_retries = rate_limit_retries or 0
        self.retries = 0  # requests sent again after a failure
        self.errors: Counter[str] = Counter()  # failures by kind ("HTTP 429", "invalid answer")
        self.input_tokens = 0
        self.output_tokens = 0

    def complete_json(
        self,
        messages: list[dict],
        schema: type[T],
        check: Callable[[T], None] | None = None,
    ) -> T:
        """Send the messages; return the answer parsed into ``schema``.

        ``check`` raises ``ValueError`` on an answer that is valid JSON but unusable.
        """
        failures = rate_limited = 0  # other failures, and 429 refusals
        attempt = 0
        while True:
            attempt += 1
            try:
                content = self._call(messages)
                try:
                    parsed = schema.model_validate_json(content)
                except ValidationError as exc:
                    raise _Retryable(f"invalid JSON answer: {exc.errors()[:1]}") from exc
                if check is not None:
                    try:
                        check(parsed)
                    except ValueError as exc:
                        raise _Retryable(f"invalid answer: {exc}") from exc
                return parsed
            except (httpx.TransportError, _Retryable) as exc:
                response = getattr(exc, "response", None)
                if response is not None and response.status_code == 429:
                    rate_limited += 1
                else:
                    failures += 1
                self.errors[_kind(exc)] += 1
                extra = max(0, rate_limited - self.rate_limit_retries)  # beyond the 429 allowance
                if failures + extra > self.max_retries:
                    raise ChatError(f"no valid answer after {attempt} attempts: {exc}") from exc
                delay = self._delay(rate_limited - 1 if rate_limited else failures - 1, exc)
                logger.warning("chat attempt %d failed (%s), retry in %.1fs", attempt, exc, delay)
                self.retries += 1
                self.sleep(delay)

    def _call(self, messages: list[dict]) -> str:
        if self.max_requests is not None and self.answered >= self.max_requests:
            raise CallBudgetExceeded(f"request limit reached ({self.max_requests})")
        wait = self._last_request + self.settings.llm_min_interval_s - self.clock()
        if wait > 0:
            self.sleep(wait)
        self._last_request = self.clock()
        self.requests += 1
        headers = {"Content-Type": "application/json"}
        if self._token is not None:
            headers["Authorization"] = f"Bearer {self._token()}"
        elif self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key.get_secret_value()}"
        response = self.client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "response_format": {"type": "json_object"},
            },
        )
        if response.status_code == 429 or response.status_code >= 500:
            reason = _reason(response)
            status = f"HTTP {response.status_code}" + (f": {reason}" if reason else "")
            raise _Retryable(status, response)
        if response.is_error:  # other 4xx: a bad key or a bug, retrying won't help
            reason = _reason(response)
            raise httpx.HTTPStatusError(
                f"HTTP {response.status_code}" + (f": {reason}" if reason else ""),
                request=response.request,
                response=response,
            )
        self.answered += 1
        payload = response.json()
        usage = payload.get("usage") or {}
        self.input_tokens += usage.get("prompt_tokens", 0)
        self.output_tokens += usage.get("completion_tokens", 0)
        return payload["choices"][0]["message"]["content"]

    @staticmethod
    def _delay(attempt: int, exc: Exception) -> float:
        response = getattr(exc, "response", None)
        if response is not None:
            retry_after = response.headers.get("retry-after")
            if retry_after:
                try:
                    return min(float(retry_after), 60.0)
                except ValueError:
                    pass
            if response.status_code == 429:  # free tiers count per minute: wait longer
                return min(5 * 2**attempt, 60) + random.uniform(0, 2)
        return min(2**attempt, 30) + random.uniform(0, 1)
