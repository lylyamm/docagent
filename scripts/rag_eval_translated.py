"""Does translating the question help the search? Measured before building the loop.

    uv run python scripts/rag_eval_translated.py --embedder e5
    (test set, once: add --questions evals/rag_test_questions.jsonl)

The Questions workflow of the agent plans a retry: when the answer is a
refusal, translate the question into the other language of the corpus and
search again. Before writing that loop, this script checks the idea on the
search alone (no answers, so ~1 LLM request per question, for the translation):

    original    hybrid(question)                          what runs today
    translated  hybrid(translation)                       the retry alone
    both        RRF(dense q, bm25 q, dense t, bm25 t)     search with both at once
    bm25 both   RRF(dense q, bm25 q, bm25 t)              translation for BM25 only
    bm25 x2     RRF(dense q, bm25 q, bm25 q)              control: no translation

"bm25 x2" is a control: "bm25 both" gives BM25 two votes out of three, so part
of any gain could come from weighting BM25 more, not from the translation.

Translations are saved in evals/translated_queries.jsonl (reused on the next
run, readable to check their quality). Results are appended to
evals/rag_results.md, with paired tests against "original".

Dev set first. The test set is run once, with the strategy chosen on dev.
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from docagent.agent import Toolbox
from docagent.agent.tools import QUERY_PROMPT_VERSION
from docagent.config import get_settings
from docagent.rag import HybridSearcher, build_embedder, load_passages, open_store
from docagent.rag.evaluation import Ranks, compare, first_hit, summarise
from docagent.rag.fusion import reciprocal_rank_fusion

CACHE = Path("evals/translated_queries.jsonl")


def load_cache(model: str) -> dict[str, dict]:
    """Saved translations made by the same model with the same prompt."""
    if not CACHE.exists():
        return {}
    rows = [json.loads(line) for line in CACHE.open(encoding="utf-8")]
    return {
        r["query"]: r
        for r in rows
        if r.get("model") == model and r.get("prompt_version") == QUERY_PROMPT_VERSION
    }


def translations(questions: list[dict], tools: Toolbox) -> dict[str, str]:
    """question -> translation, from the cache or the LLM (and then cached)."""
    cache = load_cache(tools.chat.model)
    missing = [q["question"] for q in questions if q["question"] not in cache]
    if missing:
        print(f"translating {len(missing)} questions with {tools.chat.model}...")
    with CACHE.open("a", encoding="utf-8") as f:
        for question in missing:
            result = tools.translate_query(question)
            row = result.model_dump() | {
                "model": tools.chat.model,
                "prompt_version": QUERY_PROMPT_VERSION,
            }
            cache[question] = row
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  {question}\n  -> {result.translation}")
    return {q["question"]: cache[q["question"]]["translation"] for q in questions}


def ranked_pages(searcher: HybridSearcher, lists: list[list[tuple[str, float]]]):
    fused = reciprocal_rank_fusion([[pid for pid, _ in hits] for hits in lists], k=searcher.rrf_k)
    by_id = searcher.by_id
    return [(by_id[pid].doc_id, by_id[pid].page) for pid, _ in fused[:10]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--questions", type=Path, default=Path("evals/rag_questions.jsonl"))
    parser.add_argument("--embedder", choices=["mistral", "e5", "qwen3", "hash"])
    parser.add_argument("--results", type=Path, default=Path("evals/rag_results.md"))
    args = parser.parse_args()

    settings = get_settings()
    passages = load_passages(settings.rag_index_dir / "passages.jsonl")
    embedder = build_embedder(settings, args.embedder or settings.embedder)
    store = open_store(settings, embedder)
    if store.missing(passages):
        raise SystemExit(f"the {embedder.name} index is incomplete: run scripts/rag_index.py")
    searcher = HybridSearcher(passages, embedder, store)
    questions = [json.loads(line) for line in args.questions.open(encoding="utf-8")]
    translated = translations(questions, Toolbox(settings))

    n = searcher.candidates
    runs: dict[str, Ranks] = defaultdict(dict)
    for q in questions:
        question, translation = q["question"], translated[q["question"]]
        expected = {(e["doc_id"], e["page"]) for e in q["expected"]}
        dq, bq = searcher.dense(question, n), searcher.lexical(question, n)
        dt, bt = searcher.dense(translation, n), searcher.lexical(translation, n)
        strategies = {
            "original": [dq, bq],
            "translated": [dt, bt],
            "both": [dq, bq, dt, bt],
            "bm25 both": [dq, bq, bt],
            "bm25 x2": [dq, bq, bq],
        }
        for name, lists in strategies.items():
            runs[name][q["id"]] = first_hit(ranked_pages(searcher, lists), expected)

    by_type: dict[str, list[str]] = defaultdict(list)
    for q in questions:
        by_type[q["type"]].append(q["id"])
    lines = [
        f"\n## {datetime.now():%Y-%m-%d %H:%M} · translated queries · {embedder.name} · "
        f"{len(questions)} questions ({args.questions.name})\n",
        "| strategy | hit@1 | hit@5 | MRR@10 | " + " | ".join(f"hit@5 {t}" for t in by_type) + " |",
        "|---" * (4 + len(by_type)) + "|",
    ]
    for name, ranks in runs.items():
        s = summarise(ranks)
        per_type = [summarise(ranks, ids)["hit@5"] for ids in by_type.values()]
        lines.append(
            f"| {name} | {s['hit@1']:.0%} | {s['hit@5']:.0%} | {s['mrr@10']:.2f} | "
            + " | ".join(f"{v:.0%}" for v in per_type)
            + " |"
        )
    lines += [
        "",
        "Paired tests against original: questions won / lost, two-sided p.",
        "",
        "| strategy | hit@5 won / lost | McNemar p | rank better / worse | sign test p |",
        "|---|---|---|---|---|",
    ]
    for name in ("translated", "both", "bm25 both", "bm25 x2"):
        c = compare(runs[name], runs["original"])
        lines.append(
            f"| {name} | {c.hit_wins} / {c.hit_losses} | {c.hit_p:.3f} | "
            f"{c.rank_wins} / {c.rank_losses} | {c.rank_p:.3f} |"
        )
    lines.append("")
    for name, ranks in runs.items():
        missed = [qid for qid, r in ranks.items() if r is None or r > 5]
        lines.append(f"- missed in the top 5 by {name}: {', '.join(missed) or 'none'}")

    report = "\n".join(lines)
    print(report)
    with args.results.open("a", encoding="utf-8") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
