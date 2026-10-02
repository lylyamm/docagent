"""Evaluate the search on evals/rag_questions.jsonl: hit rate@k and MRR.

    uv run python scripts/rag_eval.py                         # bm25, dense, hybrid (EMBEDDER)
    uv run python scripts/rag_eval.py --embedder e5
    uv run python scripts/rag_eval.py --embedder e5 qwen3 mistral   # compare models
    uv run python scripts/rag_eval.py --modes bm25            # no embeddings needed
    uv run python scripts/rag_eval.py --embedder e5 --details # why each question is missed

Metrics and paired tests are defined in docagent/rag/evaluation.py. With several
embedders, every pair is compared question by question (McNemar on hit@5, sign
test on the rank). Results are printed and appended to evals/rag_results.md.
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

from docagent.config import get_settings
from docagent.rag import HybridSearcher, build_embedder, load_passages, open_store
from docagent.rag.evaluation import Ranks, compare, first_hit, summarise


def expected_pages(question: dict) -> set[tuple[str, int]]:
    return {(e["doc_id"], e["page"]) for e in question["expected"]}


def evaluate(searcher: HybridSearcher, questions: list[dict], mode: str) -> Ranks:
    ranks: Ranks = {}
    for q in questions:
        hits = searcher.search(q["question"], 10, mode)
        pages = [(h.passage.doc_id, h.passage.page) for h in hits]
        ranks[q["id"]] = first_hit(pages, expected_pages(q))
    return ranks


def details(searcher: HybridSearcher, questions: list[dict], depth: int = 50) -> str:
    """Per question: rank of the first right page in each list (up to ``depth``),
    and what dense search put first."""
    by_id = searcher.by_id
    out = ["", "| question | type | bm25 rank | dense rank | dense top 3 (doc p. page) |"]
    out.append("|---|---|---|---|---|")
    for q in questions:
        ranks = []
        for hits in (searcher.lexical(q["question"], depth), searcher.dense(q["question"], depth)):
            pages = [(by_id[pid].doc_id, by_id[pid].page) for pid, _ in hits]
            rank = first_hit(pages, expected_pages(q))
            ranks.append(str(rank) if rank else f">{depth}")
        top = searcher.dense(q["question"], 3)
        top3 = ", ".join(f"{by_id[pid].doc_id} p.{by_id[pid].page}" for pid, _ in top)
        label = f"{q['id']} {q['question'][:50]}"
        out.append(f"| {label} | {q['type']} | {ranks[0]} | {ranks[1]} | {top3} |")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--questions", type=Path, default=Path("evals/rag_questions.jsonl"))
    parser.add_argument("--embedder", nargs="+", choices=["mistral", "e5", "qwen3", "hash"])
    parser.add_argument("--modes", nargs="+", default=["bm25", "dense", "hybrid"])
    parser.add_argument("--results", type=Path, default=Path("evals/rag_results.md"))
    parser.add_argument("--details", action="store_true", help="rank of the answer per question")
    args = parser.parse_args()

    settings = get_settings()
    passages = load_passages(settings.rag_index_dir / "passages.jsonl")
    questions = [json.loads(line) for line in args.questions.open(encoding="utf-8")]
    by_type: dict[str, list[str]] = defaultdict(list)
    for q in questions:
        by_type[q["type"]].append(q["id"])

    needs_vectors = any(m != "bm25" for m in args.modes)
    names = list(dict.fromkeys(args.embedder or [settings.embedder])) if needs_vectors else []
    lines = [
        f"\n## {datetime.now():%Y-%m-%d %H:%M} · embedders {', '.join(names) or '-'} · "
        f"{len(passages)} passages · {len(questions)} questions\n",
        "| mode | hit@1 | hit@5 | MRR@10 | " + " | ".join(f"hit@5 {t}" for t in by_type) + " |",
        "|---" * (4 + len(by_type)) + "|",
    ]
    runs: dict[str, Ranks] = {}  # "hybrid e5" -> ranks
    extra: list[str] = []

    def record(label: str, ranks: Ranks) -> None:
        runs[label] = ranks
        s = summarise(ranks)
        per_type = [summarise(ranks, ids)["hit@5"] for ids in by_type.values()]
        lines.append(
            f"| {label} | {s['hit@1']:.0%} | {s['hit@5']:.0%} | {s['mrr@10']:.2f} | "
            + " | ".join(f"{v:.0%}" for v in per_type)
            + " |"
        )

    if "bm25" in args.modes:
        record("bm25", evaluate(HybridSearcher(passages), questions, "bm25"))
    for name in names:
        embedder = build_embedder(settings, name)
        searcher = HybridSearcher(passages, embedder, open_store(settings, embedder))
        for mode in args.modes:
            if mode != "bm25":
                record(f"{mode} {name}", evaluate(searcher, questions, mode))
        if args.details:
            extra += [f"\n**Details, {name}**", details(searcher, questions)]

    lines.append("")
    for label, ranks in runs.items():
        missed = [qid for qid, r in ranks.items() if r is None or r > 5]
        lines.append(f"- missed in the top 5 by {label}: {', '.join(missed) or 'none'}")

    pairs = [
        (f"{mode} {a}", f"{mode} {b}")
        for mode in args.modes
        if mode != "bm25"
        for a, b in combinations(names, 2)
    ]
    if pairs:
        lines += [
            "",
            "Paired tests (A vs B, same questions): questions won by A / by B, two-sided p.",
            "",
            "| A vs B | hit@5 won | McNemar p | rank better | sign test p |",
            "|---|---|---|---|---|",
        ]
        for a, b in pairs:
            c = compare(runs[a], runs[b])
            lines.append(
                f"| {a} vs {b} | {c.hit_wins} / {c.hit_losses} | {c.hit_p:.3f} | "
                f"{c.rank_wins} / {c.rank_losses} | {c.rank_p:.3f} |"
            )

    report = "\n".join(lines + extra)
    print(report)
    with args.results.open("a", encoding="utf-8") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
