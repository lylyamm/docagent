"""Evaluate the agent's judge: does it remove wrong statements and keep right ones?

    uv run python scripts/agent_eval_judge.py --show     # the quotes found, no API call
    uv run python scripts/agent_eval_judge.py            # 11 statements, 3 runs each

evals/judge_traps.jsonl takes real passages of the reports (found in
data/index/passages.jsonl by report, page and a few words) and, for each, gives:
- "true": a faithful statement: it should be kept;
- "trap": the same figures, a wrong meaning (periods swapped, a rise called a
  fall...): it should be removed. The checks in code let it through (every figure
  is in the passage), only the judge can catch it;
- "borderline" (optional): a statement that leaves out a nuance; reported, not scored.

Each statement is judged alone (one judge call), as in the agent, ``--runs`` times:
the judge's answer can change from one run to the next.

Results: a section appended to evals/agent_results.md, and every verdict in
evals/agent_runs/judge_<date>.json. 33 supervisor calls by default.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from docagent.agent.evaluation import judge_items, judge_summary
from docagent.agent.judge import JUDGE_PROMPT_VERSION, judge
from docagent.agent.supervisor import agent_chat_client, request_language
from docagent.config import get_settings
from docagent.rag.index import load_passages


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def report(summary: dict, results: list[dict], meta: dict) -> str:
    def pct(pair: tuple[int, int]) -> str:
        a, b = pair
        return f"{a}/{b} ({100 * a / b:.0f} %)" if b else "-"

    lines = [
        f"## {meta['date']} · judge {JUDGE_PROMPT_VERSION} · {meta['model']} · "
        f"{summary['items']} statements × {meta['runs']} runs",
        "",
        "| statements | removed by the judge | expected |",
        "|---|---|---|",
        f"| traps (wrong meaning, right figures) | {pct(summary['traps_caught'])} | all |",
        f"| faithful | {pct(summary['true_removed'])} | none |",
    ]
    if summary["borderline_removed"][1]:
        lines.append(
            f"| borderline (a nuance left out) | {pct(summary['borderline_removed'])} | - |"
        )
    lines += [
        "",
        f"Same verdict on every run: {summary['constant']}/{summary['items']}"
        + (f" (changed: {', '.join(summary['changing'])})" if summary["changing"] else "")
        + f". Reviews that failed (nothing removed): {summary['not_judged']}.",
    ]
    missed = sorted({r["item"] for r in results if r["kind"] == "trap" and not r["removed"]})
    wrongly = sorted({r["item"] for r in results if r["kind"] == "true" and r["removed"]})
    if missed:
        lines.append(f"Traps kept at least once: {', '.join(missed)}.")
    if wrongly:
        lines.append(f"Faithful statements removed at least once: {', '.join(wrongly)}.")
    times = sorted(r["duration_s"] for r in results)
    usage = (
        f"Judge call: median {times[len(times) // 2]:.1f} s. Supervisor: {meta['calls']} calls, "
        f"{meta['tokens_in']} tokens in, {meta['tokens_out']} out"
    )
    if meta["cost"] is not None:
        usage += f", about ${meta['cost']:.4f}"
    lines += [usage + ".", f"Verdicts: {meta['details']}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cases", type=Path, default=Path("evals/judge_traps.jsonl"))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--only", default="", help="case ids, comma-separated (A,B)")
    parser.add_argument("--show", action="store_true", help="print the items, call nothing")
    parser.add_argument("--results", type=Path, default=Path("evals/agent_results.md"))
    args = parser.parse_args()

    settings = get_settings()
    passages = load_passages(settings.rag_index_dir / "passages.jsonl")
    cases = load(args.cases)
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    items = [(case, kind, item) for case in cases for kind, item in judge_items(passages, case)]

    if args.show:
        for case, kind, item in items:
            print(f"\n[{case['id']}-{kind}] {item.statement}")
            for quote in item.quotes:
                print(f"    Quote: {quote}")
        first = {id(c): item for c, _, item in items}
        for case in cases:
            print(f"\n[{case['id']}] passage: {first[id(case)].excerpts[0]}")
        return

    chat = agent_chat_client(settings.model_copy(update={"agent_max_calls": 0}))
    print(f"Judge: {chat.model} · {len(items)} statements × {args.runs} runs")
    results = []
    for run in range(1, args.runs + 1):
        for case, kind, item in items:
            start = time.monotonic()
            judgement = judge(chat, case["request"], [item], request_language(case["request"]))
            verdict = judgement.verdicts.get(1)
            removed = judgement.judged and not judgement.faithful(1)
            row = {
                "item": f"{case['id']}-{kind}",
                "kind": kind,
                "run": run,
                "judged": judgement.judged,
                "removed": removed,
                "problem": verdict.problem if verdict else "",
                "duration_s": round(time.monotonic() - start, 2),
            }
            results.append(row)
            expected = kind == "trap"
            mark = "   " if kind == "borderline" else ("ok " if removed == expected else "NO ")
            what = "removed" if removed else ("kept" if judgement.judged else "not judged")
            print(f"{mark} {row['item']} (run {run}): {what} {row['problem']}")

    stamp = datetime.now()
    details = Path("evals/agent_runs") / f"judge_{stamp:%Y-%m-%d_%H%M}.json"
    details.parent.mkdir(parents=True, exist_ok=True)
    details.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    price_in, price_out = settings.agent_price_in_usd, settings.agent_price_out_usd
    cost = None
    if price_in is not None and price_out is not None:
        cost = (chat.input_tokens * price_in + chat.output_tokens * price_out) / 1_000_000
    meta = {
        "date": f"{stamp:%Y-%m-%d %H:%M}",
        "model": chat.model,
        "runs": args.runs,
        "calls": chat.answered,
        "tokens_in": chat.input_tokens,
        "tokens_out": chat.output_tokens,
        "cost": cost,
        "details": details.as_posix(),
    }
    text = report(judge_summary(results), results, meta)
    with args.results.open("a", encoding="utf-8") as f:
        f.write("\n" + text)
    print("\n" + text)


if __name__ == "__main__":
    main()
