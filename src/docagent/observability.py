"""Tracing with Langfuse: what each request did, how long it took, what it cost.

- one trace per request: a question (``/v1/ask``), an agent run, a translation job;
- inside it, one *generation* per LLM call (``ChatClient``): model, messages, answer,
  tokens and cost; failed calls are marked as errors or warnings (retries are visible);
- an agent run also gets one span per graph node (LangGraph callback) and scores:
  the share of points the judge kept, and how many it removed.

Off unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set: then every helper is a
no-op, so tests, the CI and a local run without keys behave exactly as before. The SDK
sends its data from a background thread, and a tracing failure never breaks a request.

The JSON trajectories of ``agent/trajectory.py`` stay: they are versioned with the
evaluations and replayable; Langfuse is for following the running service.
"""

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .config import Settings, get_settings

logger = logging.getLogger(__name__)

_client: Any = None
_lock = threading.Lock()


def enabled(settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)


def client(settings: Settings | None = None) -> Any:
    """The Langfuse client, created once; None when tracing is off or cannot start."""
    global _client
    settings = settings or get_settings()
    if not enabled(settings):
        return None
    with _lock:
        if _client is None:
            try:
                from langfuse import Langfuse

                assert settings.langfuse_public_key and settings.langfuse_secret_key
                _client = Langfuse(
                    public_key=settings.langfuse_public_key.get_secret_value(),
                    secret_key=settings.langfuse_secret_key.get_secret_value(),
                    host=settings.langfuse_host,
                    environment=settings.langfuse_environment,
                )
            except Exception as exc:  # noqa: BLE001 - tracing must never stop the app
                logger.warning("Langfuse tracing off: %s", exc)
                return None
    return _client


@contextmanager
def trace(
    name: str,
    *,
    input: Any = None,
    session_id: str | None = None,
    settings: Settings | None = None,
) -> Iterator[Any]:
    """A request: the root span of a trace; the LLM calls made inside become its
    children. Yields the span (``span.update(output=...)``), or None when off."""
    lf = client(settings)
    if lf is None:
        yield None
        return
    from langfuse import propagate_attributes

    with (
        propagate_attributes(trace_name=name, session_id=session_id),
        lf.start_as_current_observation(name=name, as_type="span", input=input) as span,
    ):
        yield span


@contextmanager
def generation(
    name: str,
    *,
    model: str,
    input: Any,
    model_parameters: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> Iterator[Any]:
    """One LLM call; ``ChatClient`` fills in the answer, tokens and cost."""
    lf = client(settings)
    if lf is None:
        yield None
        return
    with lf.start_as_current_observation(
        name=name,
        as_type="generation",
        model=model,
        input=input,
        model_parameters=model_parameters,
    ) as gen:
        yield gen


def score(name: str, value: float, comment: str | None = None, settings: Settings | None = None):
    """A score on the current trace (e.g. the share of points the judge kept)."""
    lf = client(settings)
    if lf is not None:
        try:
            lf.score_current_trace(name=name, value=value, comment=comment)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Langfuse score not sent: %s", exc)


def callbacks(settings: Settings | None = None) -> list[Any]:
    """LangChain callbacks for a LangGraph run: one span per node, under the current trace."""
    if client(settings) is None:
        return []
    try:
        from langfuse.langchain import CallbackHandler

        return [CallbackHandler()]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Langfuse LangGraph callback off: %s", exc)
        return []


def cost_usd(
    input_tokens: int, output_tokens: int, prices: tuple[float, float] | None
) -> dict[str, float] | None:
    """Cost of a call from prices per 1M tokens (input, output); None without prices."""
    if prices is None:
        return None
    cost_in = input_tokens * prices[0] / 1_000_000
    cost_out = output_tokens * prices[1] / 1_000_000
    return {"input": cost_in, "output": cost_out, "total": cost_in + cost_out}
