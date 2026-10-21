"""Evaluate the agent's router: does each request get the right plan?

    uv run python scripts/agent_eval_routing.py              # 25 requests, 2 runs each
    uv run python scripts/agent_eval_routing.py --runs 1 --only q1,m2

Only the router runs (the ``understand`` node: one supervisor call, then the checks
in code); no tool is called, no report is read. The cases are in
evals/agent_routing.jsonl, with the plans each one accepts (see
docagent/agent/evaluation.py). Each request is played ``--runs`` times: a router
at temperature 0 can still change its mind, and the stability is measured.

Results: a section appended to evals/agent_results.md, and every plan (the
model's raw steps and the checked ones) in evals/agent_runs/routing_<date>.json.
About 50 supervisor calls (Gemini 2.5 Pro: a few cents, a few minutes).
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from docagent.agent import Toolbox
from docagent.agent.evaluation import plan_matches, plan_signature, routing_summary
from docagent.agent.supervisor import ROUTER_PROMPT_VERSION, agent_chat_client, plan_request
from docagent.chat import ChatError
from docagent.config import get_settings

CATEGORIES = {
    "question": "question",
    "summarize": "summary",
    "compare": "comparison",
    "translate": "translation",
    "multi": "several steps",
    "ambiguous": "ambiguous (2 plans accepted)",
    "clarify": "unclear → question back",
    "out_of_scope": "out of scope",
}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def cost(settings, tokens_in: int, tokens_out: int) -> float | None:
    price_in, price_out = settings.agent_price_in_usd, settings.agent_price_out_usd
    if price_in is None or price_out is None:
        return None
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000


def report(summary: dict, results: list[dict], meta: dict) -> str:
    pct = lambda a, b: f"{a}/{b} ({100 * a / b:.0f} %)" if b else "-"  # noqa: E731
    lines = [
        f"## {meta['date']} · routing {ROUTER_PROMPT_VERSION} · {meta['model']} · "
        f"{summary['cases']} requests × {meta['runs']} runs",
        "",
        "| requests | right plan |",
        "|---|---|",
    ]
    for key, label in CATEGORIES.items():
        if key in summary["by_category"]:
            right, total = summary["by_category"][key]
            lines.append(f"| {label} | {pct(right, total)} |")
    lines.append(f"| **all** | **{pct(summary['right'], summary['total'])}** |")
    lines.append("")
    if meta["runs"] > 1:
        lines.append(
            f"Same plan on every run: {pct(summary['stable'], summary['cases'])}"
            + (f" (changed: {', '.join(summary['unstable'])})" if summary["unstable"] else "")
            + "."
        )
    wrong = [r for r in results if not r["ok"]]
    lines.append("Wrong: " + ("none." if not wrong else ""))
    lines += [f"- {r['id']} (run {r['run']}): {r['signature'] or r['error']}" for r in wrong]
    if summary["fixed_by_code"]:
        lines.append(
            "Plans corrected by the checks in code (unknown report, missing file...): "
            + ", ".join(summary["fixed_by_code"])
            + "."
        )
    times = sorted(r["duration_s"] for r in results if r["error"] is None)
    if times:
        lines.append(f"Router call: median {times[len(times) // 2]:.1f} s, max {times[-1]:.1f} s.")
    usage = f"Supervisor: {meta['calls']} calls, {meta['tokens_in']} tokens in, "
    usage += f"{meta['tokens_out']} out"
    if meta["cost"] is not None:
        per = meta["cost"] / max(1, len(results))
        usage += f", about ${meta['cost']:.4f} (${per:.5f} per request)"
    lines += [usage + ".", f"Plans: {meta['details']}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cases", type=Path, default=Path("evals/agent_routing.jsonl"))
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--only", default="", help="case ids, comma-separated")
    parser.add_argument("--results", type=Path, default=Path("evals/agent_results.md"))
    args = parser.parse_args()

    settings = get_settings()
    # No call limit here: the evaluation knows how many calls it makes.
    chat = agent_chat_client(settings.model_copy(update={"agent_max_calls": 0}))
    tools = Toolbox(settings)  # only its corpus is used
    if not tools.corpus:
        raise SystemExit(f"No corpus at {settings.corpus_path}")
    cases = load(args.cases)
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    print(f"Router: {chat.model} · {len(cases)} requests × {args.runs} runs")

    results = []
    for run in range(1, args.runs + 1):
        for case in cases:
            start = time.monotonic()
            row = {"id": case["id"], "category": case["category"], "run": run}
            try:
                routing = plan_request(chat, tools, case["request"], settings.agent_max_steps)
            except ChatError as exc:
                row |= {"ok": False, "signature": "", "error": str(exc), "plans": []}
            else:
                raw_routes = [s.route for s in routing.raw.steps]
                routes = [p.route for p in routing.plans]
                row |= {
                    "ok": plan_matches(routing.plans, case["accept"]),
                    "signature": plan_signature(routing.plans),
                    "error": None,
                    "fixed_by_code": raw_routes != routes,
                    "raw": routing.raw.model_dump(),
                    "plans": [
                        p.model_dump(exclude={"model", "prompt_version"}) for p in routing.plans
                    ],
                }
            row["duration_s"] = round(time.monotonic() - start, 2)
            results.append(row)
            mark = "ok " if row["ok"] else "NO "
            print(
                f"{mark} {case['id']} (run {run}, {row['duration_s']:.1f} s): "
                f"{row['signature'] or row['error']}"
            )

    stamp = datetime.now()
    details = Path("evals/agent_runs") / f"routing_{stamp:%Y-%m-%d_%H%M}.json"
    details.parent.mkdir(parents=True, exist_ok=True)
    details.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = routing_summary(results)
    meta = {
        "date": f"{stamp:%Y-%m-%d %H:%M}",
        "model": chat.model,
        "runs": args.runs,
        "calls": chat.answered,
        "tokens_in": chat.input_tokens,
        "tokens_out": chat.output_tokens,
        "cost": cost(settings, chat.input_tokens, chat.output_tokens),
        "details": details.as_posix(),
    }
    text = report(summary, results, meta)
    with args.results.open("a", encoding="utf-8") as f:
        f.write("\n" + text)
    print("\n" + text)


if __name__ == "__main__":
    main()
