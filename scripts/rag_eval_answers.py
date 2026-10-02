"""Evaluate the answers (the "G" of RAG), not only the search.

    uv run python scripts/rag_eval_answers.py           # 24 questions + 6 without answer
    uv run python scripts/rag_eval_answers.py --limit 5 # a quick try

For each question of evals/rag_questions.jsonl (the answer is in the reports):
- found: the right page was among the passages given to the model (search);
- answered: the model gave an answer with at least one valid citation;
- cited right page: one of its citations is an expected page (grounding).
For each question of evals/rag_unanswerable.jsonl (not in the reports):
- refused: the model said the documents do not answer.

Whether an answer is CORRECT is not measured automatically: every answer is
saved to evals/answers/<date>.jsonl to be read and judged by hand.
One LLM request per question (~30 requests, about a minute on the free tier).
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from docagent.rag import build_answerer


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--questions", type=Path, default=Path("evals/rag_questions.jsonl"))
    parser.add_argument("--unanswerable", type=Path, default=Path("evals/rag_unanswerable.jsonl"))
    parser.add_argument("--results", type=Path, default=Path("evals/rag_results.md"))
    parser.add_argument("--limit", type=int, help="only the first N questions of each file")
    args = parser.parse_args()

    answerer = build_answerer()
    by_id = answerer.searcher.by_id
    answerable = load(args.questions)[: args.limit]
    unanswerable = load(args.unanswerable)[: args.limit]

    stamp = datetime.now()
    out_dir = Path("evals/answers")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stamp:%Y-%m-%d_%H%M}.jsonl"

    rows = []
    with out_path.open("w", encoding="utf-8") as out:
        for q in answerable + unanswerable:
            answer = answerer.ask(q["question"])
            expected = {(e["doc_id"], e["page"]) for e in q.get("expected", [])}
            given = {(by_id[pid].doc_id, by_id[pid].page) for pid in answer.retrieved}
            cited = {(s.doc_id, s.page) for s in answer.sources}
            row = {
                "id": q["id"],
                "type": q["type"],
                "has_answer": bool(expected),
                "found": bool(expected & given),
                "answered": answer.answerable,
                "cited_right_page": bool(expected & cited),
                "question": q["question"],
                "answer": answer.answer,
                "sources": [f"{s.doc_id} p.{s.page}" for s in answer.sources],
            }
            rows.append(row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            good = row["cited_right_page"] if expected else not row["answered"]
            mark = "ok " if good else "!! "
            print(f"{mark}{q['id']}  {answer.answer[:110]}")

    def share(items: list[dict], key: str) -> str:
        return f"{sum(r[key] for r in items)}/{len(items)}" if items else "-"

    with_answer = [r for r in rows if r["has_answer"]]
    without = [r for r in rows if not r["has_answer"]]
    refused = [{**r, "refused": not r["answered"]} for r in without]
    found = [r for r in with_answer if r["found"]]
    chat = answerer.chat
    lines = [
        f"\n## {stamp:%Y-%m-%d %H:%M} · answers · {chat.model} · embedder "
        f"{answerer.searcher.embedder.name} · top {answerer.top_k}\n",
        "| questions | right page given to the model | answered | cited the right page | "
        "cited it, when it was given |",
        "|---|---|---|---|---|",
        f"| with an answer ({len(with_answer)}) | {share(with_answer, 'found')} | "
        f"{share(with_answer, 'answered')} | {share(with_answer, 'cited_right_page')} | "
        f"{share(found, 'cited_right_page')} |",
        "",
        f"Without an answer in the reports: refused {share(refused, 'refused')}"
        f" (answered anyway: {', '.join(r['id'] for r in without if r['answered']) or 'none'}).",
        f"Missed: {', '.join(r['id'] for r in with_answer if not r['cited_right_page']) or 'none'}."
        f" Answers to read: {out_path.as_posix()}",
        f"LLM: {chat.requests} requests, {chat.input_tokens} tokens in, {chat.output_tokens} out.",
    ]
    report = "\n".join(lines)
    print(report)
    with args.results.open("a", encoding="utf-8") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
