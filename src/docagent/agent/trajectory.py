"""Trajectories: what the agent did for one request, saved as one JSON file.

    request ─► plan(s) ─► nodes traversed (duration, tool, points kept/removed)
            ─► supervisor calls, tokens, cost, total duration ─► message

The recorder runs the graph with ``graph.stream(..., stream_mode="updates")``:
each node's update arrives when the node ends, so the time between two updates
is that node's duration (the graph runs one node at a time). Nothing changes in
the nodes themselves.

A pause (``interrupt()``, e.g. "translate 32 pages?") ends a run; the same
recorder continues on ``Command(resume=...)``, and the time spent waiting for the
user is not counted.

Used by ``scripts/agent_chat.py``, the API and the routing evaluation (which
compares the recorded plan with the expected route).
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..chat import ChatClient
from ..config import Settings
from ..rag import Answer

TOOLS = {
    "question": "ask",
    "summarize": "summarize_document",
    "compare": "compare_documents",
    "translate": "translate_pdf",
}


class NodeRecord(BaseModel):
    node: str
    duration_s: float
    tool: str | None = None
    items: int | None = None  # what the tool produced: points, or 1 answer
    passages: int | None = None  # passages the tool read
    rejected: list[str] | None = None  # points dropped by the code checks, with the reason
    kept: int | None = None  # judge: items kept
    removed: int | None = None  # judge: items removed
    judged: bool | None = None  # judge: False if the review failed (nothing removed)
    ok: bool | None = None  # question: answered; translate: done
    refusal: str | None = None  # question not answered: why (the model, or which check)
    # question: where the time went: search_s, llm_s, llm_requests (more than 1: retries),
    # other_s (the rest of the node: on the first request, loading the index and models)
    timings: dict[str, float] | None = None
    error: str | None = None


class Trajectory(BaseModel):
    request: str
    thread_id: str
    started_at: str
    supervisor_model: str
    plans: list[dict[str, Any]] = Field(default_factory=list)
    nodes: list[NodeRecord] = Field(default_factory=list)
    pauses: int = 0  # interrupts (questions to the user)
    steps_planned: int = 0
    steps_done: int = 0
    supervisor_calls: int = 0
    tokens: dict[str, int] = Field(default_factory=dict)
    cost_usd: float | None = None
    duration_s: float = 0.0  # time spent working, waits for the user excluded
    message: str = ""

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = self.started_at.replace("-", "").replace(":", "").replace("T", "-")
        path = directory / f"{stamp}_{self.thread_id[:8]}.json"
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path


class Recorder:
    """Records one request, across its runs (start, then each resume)."""

    def __init__(
        self,
        request: str,
        supervisor: ChatClient,
        settings: Settings,
        thread_id: str | None = None,
        clock=time.monotonic,
    ) -> None:
        self.thread_id = thread_id or uuid.uuid4().hex
        self.config = {"configurable": {"thread_id": self.thread_id}}
        self.supervisor = supervisor
        self.settings = settings
        self.clock = clock
        self._start = (supervisor.answered, supervisor.input_tokens, supervisor.output_tokens)
        self._items: int | None = None  # what the last tool produced, for the judge
        # The state as the nodes write it, updated live: the API shows the plan and
        # the parts of the message already written while the run goes on.
        self.live: dict[str, Any] = {}
        self.trajectory = Trajectory(
            request=request,
            thread_id=self.thread_id,
            started_at=datetime.now().isoformat(timespec="seconds"),
            supervisor_model=supervisor.model,
        )

    def run(self, graph, inputs) -> dict[str, Any]:
        """One run (``{"request": ...}`` or ``Command(resume=...)``); returns the state,
        with ``__interrupt__`` when the graph paused, like ``graph.invoke``."""
        last = self.clock()
        for chunk in graph.stream(inputs, self.config, stream_mode="updates"):
            now = self.clock()
            for node, update in chunk.items():
                if node == "__interrupt__":
                    self.trajectory.pauses += 1
                    continue
                self.live.update(update or {})
                self._record(node, update or {}, round(now - last, 3))
            self.trajectory.duration_s = round(self.trajectory.duration_s + now - last, 3)
            last = now
        snapshot = graph.get_state(self.config)
        state = dict(snapshot.values)
        if snapshot.interrupts:
            state["__interrupt__"] = list(snapshot.interrupts)
        return state

    def finish(self, state: dict[str, Any]) -> Trajectory:
        t = self.trajectory
        plans = state.get("plans") or []
        t.plans = [p.model_dump(exclude={"model", "prompt_version"}) for p in plans]
        t.steps_planned = len(plans)
        t.steps_done = len(state.get("sections") or [])
        t.message = state.get("message", "")
        calls, tokens_in, tokens_out = self.usage()
        t.supervisor_calls = calls
        t.tokens = {"input": tokens_in, "output": tokens_out}
        price_in = self.settings.agent_price_in_usd
        price_out = self.settings.agent_price_out_usd
        if price_in is not None and price_out is not None:
            t.cost_usd = round((tokens_in * price_in + tokens_out * price_out) / 1_000_000, 6)
        return t

    def usage(self) -> tuple[int, int, int]:
        """Supervisor calls, input and output tokens spent on this request so far."""
        return (
            self.supervisor.answered - self._start[0],
            self.supervisor.input_tokens - self._start[1],
            self.supervisor.output_tokens - self._start[2],
        )

    def _record(self, node: str, update: dict[str, Any], duration: float) -> None:
        record = NodeRecord(node=node, duration_s=duration, tool=TOOLS.get(node))
        if update.get("error"):
            record.error = update["error"]
        if node == "question" and (answer := update.get("answer")) is not None:
            record.ok = answer.answerable
            self._items = 1 if answer.answerable else 0
            record.passages = len(answer.retrieved)
            record.refusal = refusal_reason(answer)
            timings = dict(answer.timings)
            spent = timings.get("search_s", 0.0) + timings.get("llm_s", 0.0)
            timings["other_s"] = round(max(0.0, duration - spent), 3)
            record.timings = timings
        elif node in ("summarize", "compare"):
            result = update.get("summary") or update.get("comparison")
            self._items = len(result.points) if result is not None else 0
            record.items = self._items
            if result is not None:
                record.passages = len(result.retrieved)
                record.rejected = [f"{r.text[:120]} → {r.reason}" for r in result.rejected_points]
        elif node == "translate":
            record.ok = bool(update.get("translation_ok"))
        elif node == "judge":
            judgement = update.get("judgement")
            if judgement is not None:
                record.judged = judgement.judged
                if judgement.judged:  # not judged: nothing was reviewed, nothing removed
                    removed = sum(not v.faithful for v in judgement.verdicts.values())
                    record.removed = removed
                    if self._items is not None:
                        record.kept = self._items - removed
        if node == "question":
            record.items = self._items
        self.trajectory.nodes.append(record)


def refusal_reason(answer: Answer) -> str | None:
    """Why a question was not answered, from the answer's diagnostics; None if answered.

    Without it a trajectory only said "not answered": the same question was answered
    at 18:41 and refused at 23:42 (2026-10-21), and nothing told why.
    """
    if answer.answerable:
        return None
    if not answer.retrieved:
        return "no passage found"
    if answer.model_answerable is False:
        return "the model said the passages do not answer"
    if answer.unsupported_numbers:
        return "figures not in the quotes: " + ", ".join(answer.unsupported_numbers)
    if answer.rejected_quotes:
        return "quotes not found in their passage: " + " | ".join(
            q[:80] for q in answer.rejected_quotes
        )
    return "no verified quote"


def load(path: str | Path) -> Trajectory:
    return Trajectory.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
