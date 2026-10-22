"""Run the whole agent on 4 requests, and measure time and cost.

    uv run python scripts/agent_eval_runs.py
    uv run python scripts/agent_eval_runs.py --only 1,4

The requests below cover the routes that make claims (a question, a summary, a
comparison, then two steps in one request). Everything runs: router, tools, judge.
Measured per request: duration, supervisor calls and cost, the tools' cost (their
model is TOOLS_MODEL on TOOLS_PROVIDER, else LLM_MODEL; priced with
TOOLS_PRICE_*_USD), points kept and removed by the judge. To compare two tools'
models, run it once with each: everything else stays the same.

The agent's answers are not scored here (the routing and the judge have their own
evaluations): each one is saved as a trajectory in AGENT_TRAJECTORIES_DIR, to be read.
Results: a section appended to evals/agent_results.md. About 10 supervisor calls,
a few minutes (a summary or a comparison takes 30-60 s).
"""

import argparse
from datetime import datetime
from pathlib import Path

from langgraph.types import Command

from docagent.agent import Toolbox
from docagent.agent.checkpoint import memory_checkpointer
from docagent.agent.supervisor import agent_chat_client, build_supervisor
from docagent.agent.trajectory import Recorder
from docagent.config import get_settings

REQUESTS = [
    "Que dit le HCC sur les émissions des transports en 2024 ?",
    "Résume le rapport annuel 2025 du HCC",
    "Compare l'AIE et le GIEC sur les trajectoires compatibles avec 1,5 °C",
    "Résume le rapport du HCC puis compare-le avec le GIEC sur les puits de carbone",
]


def judged(trajectory) -> str:
    """Per step reviewed: "kept/total", or "not reviewed"."""
    parts = []
    for node in trajectory.nodes:
        if node.refusal:
            parts.append(f"not answered: {node.refusal}")
        if node.node != "judge" or node.judged is None:  # None: nothing to review
            continue
        if not node.judged:
            parts.append("not reviewed")
        elif node.kept is not None and node.removed is not None:
            parts.append(f"{node.kept}/{node.kept + node.removed}")
    if any(node.error for node in trajectory.nodes):
        parts.append("step failed")
    return ", ".join(parts) or "-"


def money(value: float | None) -> str:
    return "-" if value is None else f"${value:.4f}"


def cost(settings, tokens_in: int, tokens_out: int) -> float | None:
    """The tools' cost, with TOOLS_PRICE_*_USD (per 1M tokens); None if not set."""
    price_in, price_out = settings.tools_price_in_usd, settings.tools_price_out_usd
    if price_in is None or price_out is None:
        return None
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000


def points(trajectories) -> tuple[int, int]:
    """Points and answers kept by the review, out of those it read, over all requests."""
    kept = total = 0
    for t in trajectories:
        for node in t.nodes:
            if node.node == "judge" and node.judged and node.kept is not None:
                kept += node.kept
                total += node.kept + (node.removed or 0)
    return kept, total


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", default="", help="request numbers, comma-separated (1,4)")
    parser.add_argument("--results", type=Path, default=Path("evals/agent_results.md"))
    args = parser.parse_args()

    settings = get_settings()
    tools = Toolbox(settings)
    numbers = [int(n) for n in args.only.split(",")] if args.only else range(1, 5)
    rows = []
    model = ""
    for n in numbers:
        request = REQUESTS[n - 1]
        chat = agent_chat_client(settings)  # one per request, like the API
        model = chat.model
        graph = build_supervisor(tools, chat, checkpointer=memory_checkpointer())
        recorder = Recorder(request, chat, settings)
        print(f"\n> {request}")
        before = tools.usage()
        state = recorder.run(graph, {"request": request})
        while "__interrupt__" in state:  # not expected here: no translation
            state = recorder.run(graph, Command(resume="non"))
        trajectory = recorder.finish(state)
        path = trajectory.save(settings.agent_trajectories_dir)
        print(state.get("message", ""))
        for node in trajectory.nodes:
            if node.timings:  # a question: where its time went
                t = node.timings
                print(
                    f"[question: search {t.get('search_s', 0):.1f} s, model "
                    f"{t.get('llm_s', 0):.1f} s ({t.get('llm_requests', 0):.0f} request(s)), "
                    f"other {t.get('other_s', 0):.1f} s]"
                )
        tools_s = sum(r.duration_s for r in trajectory.nodes if r.tool is not None)
        spent_in, spent_out = (a - b for a, b in zip(tools.usage(), before, strict=True))
        tools_cost = cost(settings, spent_in, spent_out)
        rows.append((n, request, trajectory, tools_s, tools_cost, path))
        print(
            f"[{trajectory.duration_s:.0f} s, {trajectory.supervisor_calls} calls, "
            f"${trajectory.cost_usd or 0:.4f} + tools {money(tools_cost)}] {path}"
        )

    stamp = datetime.now()
    lines = [
        f"## {stamp:%Y-%m-%d %H:%M} · whole agent · supervisor {model} · tools {tools.chat.model}",
        "",
        "| # | request | steps | duration | of which tools | supervisor calls | "
        "supervisor cost | tools cost | points kept by the judge |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for n, request, t, tools_s, tools_cost, _ in rows:
        routes = " > ".join(p["route"] for p in t.plans)
        lines.append(
            f"| {n} | {request} | {routes} | {t.duration_s:.0f} s | {tools_s:.0f} s | "
            f"{t.supervisor_calls} | {money(t.cost_usd)} | {money(tools_cost)} | {judged(t)} |"
        )
    kept, total = points(row[2] for row in rows)
    lines += [
        "",
        f"Points and answers kept by the review: {kept} / {total}. Tools cost: with "
        'TOOLS_PRICE_IN_USD / TOOLS_PRICE_OUT_USD ("-": not set, or a free tier). '
        "Trajectories: " + ", ".join(p.name for *_, p in rows),
        "",
    ]
    text = "\n".join(lines)
    with args.results.open("a", encoding="utf-8") as f:
        f.write("\n" + text)
    print("\n" + text)


if __name__ == "__main__":
    main()
