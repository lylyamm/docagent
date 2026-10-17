"""Agent runs for the API: start a request, follow it, answer its questions.

A request takes from a few seconds to several minutes (each supervisor call can
take 10-30 s, a translation much longer), too long to hold an HTTP request open.
So, like the translation jobs:

    start(request)        -> a run, status "queued", returned at once
    get(run_id)           -> where it is: plan, nodes done, parts of the message
                             already written, then the final message
    resume(run_id, "oui") -> the answer to the agent's question (long translation)

    queued ─► running ─┬─► done
                       ├─► needs_confirmation ─(resume)─► running ─► ...
                       └─► failed

- One worker thread: runs wait in line. The tools share one search index (an
  embedded Qdrant folder) and one model client; one run at a time keeps them safe.
- Each run has its own supervisor client, so AGENT_MAX_CALLS_PER_REQUEST is a
  limit per request, not per server.
- ``run_id`` is the graph's ``thread_id``: the checkpointer keeps the paused state.
- Everything lives in memory (one process): a restart forgets the runs. Their
  trajectories are saved in AGENT_TRAJECTORIES_DIR when they end.
"""

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from langgraph.types import Command
from pydantic import BaseModel, Field

from ..chat import ChatClient
from ..config import Settings
from .checkpoint import memory_checkpointer
from .supervisor import build_supervisor
from .tools import Toolbox
from .trajectory import NodeRecord, Recorder


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    NEEDS_CONFIRMATION = "needs_confirmation"
    DONE = "done"
    FAILED = "failed"


class AgentRun(BaseModel):
    """What a client sees of a run, at any time."""

    run_id: str
    status: RunStatus
    request: str
    created_at: datetime
    plans: list[dict[str, Any]] = Field(default_factory=list)
    nodes: list[NodeRecord] = Field(default_factory=list)  # nodes done so far
    sections: list[str] = Field(default_factory=list)  # message parts of the steps done
    question: str | None = None  # needs_confirmation: what the agent asks
    message: str | None = None  # done: the whole message
    error: str | None = None  # failed
    supervisor_calls: int = 0
    duration_s: float = 0.0
    cost_usd: float | None = None
    trajectory_file: str | None = None


class RunStateError(Exception):
    """The run cannot do that now (e.g. resume a run that asks nothing)."""


class _Entry:
    def __init__(self, request: str, chat: ChatClient, graph, recorder: Recorder) -> None:
        self.request = request
        self.chat = chat
        self.graph = graph
        self.recorder = recorder
        self.status = RunStatus.QUEUED
        self.created_at = datetime.now(UTC)
        self.question: str | None = None
        self.message: str | None = None
        self.error: str | None = None
        self.trajectory_file: str | None = None


class AgentRuns:
    def __init__(
        self,
        tools: Toolbox,
        settings: Settings,
        make_chat: Callable[[], ChatClient],
        workers: int = 1,
    ) -> None:
        self.tools = tools
        self.settings = settings
        self.make_chat = make_chat
        self.checkpointer = memory_checkpointer()  # shared: the runs' graphs differ by chat
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="agent")
        self._runs: dict[str, _Entry] = {}
        self._lock = threading.Lock()

    # -- API -------------------------------------------------------------------------

    def start(self, request: str) -> AgentRun:
        chat = self.make_chat()  # its own call counter: the cost guard is per request
        graph = build_supervisor(self.tools, chat, checkpointer=self.checkpointer)
        recorder = Recorder(request, chat, self.settings)
        entry = _Entry(request, chat, graph, recorder)
        with self._lock:
            self._runs[recorder.thread_id] = entry
        self.executor.submit(self._execute, entry, {"request": request})
        return self._view(recorder.thread_id, entry)

    def get(self, run_id: str) -> AgentRun | None:
        with self._lock:
            entry = self._runs.get(run_id)
        return None if entry is None else self._view(run_id, entry)

    def resume(self, run_id: str, answer: str) -> AgentRun:
        with self._lock:
            entry = self._runs.get(run_id)
            if entry is None:
                raise KeyError(run_id)
            if entry.status != RunStatus.NEEDS_CONFIRMATION:
                raise RunStateError(f"run is {entry.status.value}, it asks nothing")
            entry.status, entry.question = RunStatus.QUEUED, None
        self.executor.submit(self._execute, entry, Command(resume=answer))
        return self._view(run_id, entry)

    # -- work ------------------------------------------------------------------------

    def _execute(self, entry: _Entry, inputs) -> None:
        """Runs in the worker thread; every failure ends up on the run, never raised."""
        entry.status = RunStatus.RUNNING
        try:
            state = entry.recorder.run(entry.graph, inputs)
            if "__interrupt__" in state:
                entry.question = state["__interrupt__"][0].value["question"]
                entry.status = RunStatus.NEEDS_CONFIRMATION
                return
            trajectory = entry.recorder.finish(state)
            if self.settings.agent_trajectories_dir:
                path = trajectory.save(self.settings.agent_trajectories_dir)
                entry.trajectory_file = str(path)
            entry.message = state.get("message", "")
            entry.status = RunStatus.DONE
        except Exception as exc:  # noqa: BLE001 - the run records it for the client
            entry.error = f"{type(exc).__name__}: {exc}"
            entry.status = RunStatus.FAILED

    def _view(self, run_id: str, entry: _Entry) -> AgentRun:
        recorder, live = entry.recorder, entry.recorder.live
        plans = live.get("plans") or []
        trajectory = recorder.trajectory
        price_in = self.settings.agent_price_in_usd
        price_out = self.settings.agent_price_out_usd
        calls, spent_in, spent_out = recorder.usage()
        cost = None
        if price_in is not None and price_out is not None:
            cost = round((spent_in * price_in + spent_out * price_out) / 1_000_000, 6)
        return AgentRun(
            run_id=run_id,
            status=entry.status,
            request=entry.request,
            created_at=entry.created_at,
            plans=[p.model_dump(exclude={"model", "prompt_version"}) for p in plans],
            nodes=list(trajectory.nodes),
            sections=list(live.get("sections") or []),
            question=entry.question,
            message=entry.message,
            error=entry.error,
            supervisor_calls=calls,
            duration_s=trajectory.duration_s,
            cost_usd=cost,
            trajectory_file=entry.trajectory_file,
        )
