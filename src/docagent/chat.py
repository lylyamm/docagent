"""A small client for OpenAI-compatible chat completions, returning validated JSON.

Used by the question answering (``rag/answer.py``). The translator
(``translate/llm.py``) has its own copy of this logic, written first; merging
the two is planned for the industrialisation week.

- one request = a list of messages -> a JSON answer validated by a Pydantic model;
- requests spaced by ``llm_min_interval_s`` (free tiers allow ~1 request/second);
- rate limits (429), server errors, timeouts and invalid JSON retried with
  exponential backoff; other 4xx errors (bad key, bad request) raised at once.
"""

import logging
import random
import time
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .config import Settings, get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ChatError(RuntimeError):
    """The model could not give a valid answer after every retry."""


class _Retryable(Exception):
    def __init__(self, message: str, response: httpx.Response | None = None) -> None:
        super().__init__(message)
        self.response = response


class ChatClient:
    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.Client | None = None,
        sleep=time.sleep,
        clock=time.monotonic,
        temperature: float = 0.1,
    ) -> None:
        self.settings = settings or get_settings()
        self.base_url = self.settings.resolved_base_url
        self.model = self.settings.resolved_model
        self._api_key = self.settings.resolved_api_key  # raises if a required key is missing
        self.client = client or httpx.Client(timeout=self.settings.llm_timeout_s)
        self.sleep = sleep
        self.clock = clock
        self.temperature = temperature
        self._last_request = float("-inf")
        self.requests = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def complete_json(self, messages: list[dict], schema: type[T]) -> T:
        """Send the messages; return the answer parsed into ``schema``."""
        attempts = self.settings.llm_max_retries + 1
        for attempt in range(attempts):
            try:
                content = self._call(messages)
                try:
                    return schema.model_validate_json(content)
                except ValidationError as exc:
                    raise _Retryable(f"invalid JSON answer: {exc.errors()[:1]}") from exc
            except (httpx.TransportError, _Retryable) as exc:
                if attempt == attempts - 1:
                    raise ChatError(f"no valid answer after {attempts} attempts: {exc}") from exc
                delay = self._delay(attempt, exc)
                logger.warning(
                    "chat attempt %d failed (%s), retry in %.1fs", attempt + 1, exc, delay
                )
                self.sleep(delay)
        raise AssertionError("unreachable")

    def _call(self, messages: list[dict]) -> str:
        wait = self._last_request + self.settings.llm_min_interval_s - self.clock()
        if wait > 0:
            self.sleep(wait)
        self._last_request = self.clock()
        self.requests += 1
        headers = {"Content-Type": "application/json"}
        if self._api_key is not None:
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
            raise _Retryable(f"HTTP {response.status_code}", response)
        response.raise_for_status()  # other 4xx: a bad key or a bug, retrying won't help
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
