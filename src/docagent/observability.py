"""Tracing with Langfuse: what each request did, how long it took, what it cost.

- one trace per request: a question (``/v1/ask``), an agent run, a translation job;
- inside it, one *generation* per LLM call (``ChatClient``): model, messages, answer,
  tokens and cost; failed calls are marked as errors or warnings (retries are visible);
- an agent run also gets one span per graph node (LangGraph callback) and scores:
  the share of points the judge kept, and how many it removed.

Off unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set: then every helper is a
no-op, so tests, the CI and a local run without keys behave exactly as before. The SDK
sends its data from a background thread, and a tracing failure never breaks a request:
opening a span, updating it and closing it are guarded; only the traced code's own
errors go through.

The JSON trajectories of ``agent/trajectory.py`` stay: they are versioned with the
evaluations and replayable; Langfuse is for following the running service.
"""

import logging
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
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


class _Observation:
    """A span or generation whose ``update`` can never raise: a tracing failure must
    not turn into a failed request."""

    def __init__(self, obs: Any) -> None:
        self._obs = obs

    def update(self, **fields: Any) -> None:
        try:
            self._obs.update(**fields)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Langfuse update not sent: %s", exc)


@contextmanager
def _guarded(open_observation: Callable[[], Any]) -> Iterator[_Observation | None]:
    """Open an SDK context manager; on any SDK error, go on without tracing. An error
    of the traced code itself goes through unchanged (and is recorded on the span)."""
    try:
        cm = open_observation()
        obs = cm.__enter__()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Langfuse tracing skipped: %s", exc)
        yield None
        return
    exc_info: tuple[Any, Any, Any] = (None, None, None)
    try:
        yield _Observation(obs)
    except BaseException:
        exc_info = sys.exc_info()
        raise
    finally:
        try:
            cm.__exit__(*exc_info)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Langfuse tracing: %s", exc)


@contextmanager
def trace(
    name: str,
    *,
    input: Any = None,
    session_id: str | None = None,
    settings: Settings | None = None,
) -> Iterator[_Observation | None]:
    """A request: the root span of a trace; the LLM calls made inside become its
    children. Yields the span (``span.update(output=...)``), or None when off."""
    lf = client(settings)
    if lf is None:
        yield None
        return

    def open_trace() -> Any:
        from langfuse import propagate_attributes

        stack = ExitStack()
        stack.enter_context(propagate_attributes(trace_name=name, session_id=session_id))
        span = stack.enter_context(
            lf.start_as_current_observation(name=name, as_type="span", input=input)
        )
        return _Entered(stack, span)

    with _guarded(open_trace) as span:
        yield span


@contextmanager
def generation(
    name: str,
    *,
    model: str,
    input: Any,
    model_parameters: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> Iterator[_Observation | None]:
    """One LLM call; ``ChatClient`` fills in the answer, tokens and cost."""
    lf = client(settings)
    if lf is None:
        yield None
        return
    with _guarded(
        lambda: lf.start_as_current_observation(
            name=name,
            as_type="generation",
            model=model,
            input=input,
            model_parameters=model_parameters,
        )
    ) as gen:
        yield gen


class _Entered:
    """Two context managers already entered (attributes + span), closed together."""

    def __init__(self, stack: ExitStack, span: Any) -> None:
        self.stack, self.span = stack, span

    def __enter__(self) -> Any:
        return self.span

    def __exit__(self, *exc_info: Any) -> bool:
        return bool(self.stack.__exit__(*exc_info))


def score(name: str, value: float, comment: str | None = None, settings: Settings | None = None):
    """A score on the current trace (e.g. the share of points the judge kept)."""
    lf = client(settings)
    if lf is not None:
        try:
            lf.score_current_trace(name=name, value=value, comment=comment)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Langfuse score not sent: %s", exc)


def flush(settings: Settings | None = None) -> None:
    """Send the traces still waiting (the API calls it when the container stops: with
    scale to zero, an instance can stop a few seconds after its last request)."""
    lf = _client if enabled(settings) else None
    if lf is not None:
        try:
            lf.flush()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Langfuse flush failed: %s", exc)


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
