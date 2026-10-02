"""Search the indexed reports from the command line.

uv run python scripts/rag_search.py "budget carbone restant pour 1,5 °C"
uv run python scripts/rag_search.py "SSP1-2.6" --mode bm25
uv run python scripts/rag_search.py "data centres investment" --embedder e5 -k 10
"""

import argparse

from docagent.config import get_settings
from docagent.rag import HybridSearcher, build_embedder, load_passages, open_store


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("query")
    parser.add_argument("--mode", choices=["hybrid", "dense", "bm25"], default="hybrid")
    parser.add_argument("--embedder", choices=["mistral", "e5", "qwen3", "hash"])
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    settings = get_settings()
    passages = load_passages(settings.rag_index_dir / "passages.jsonl")
    embedder = store = None
    if args.mode != "bm25":
        embedder = build_embedder(settings, args.embedder)
        store = open_store(settings, embedder)
    searcher = HybridSearcher(passages, embedder, store)

    for n, hit in enumerate(searcher.search(args.query, args.k, args.mode), start=1):
        p = hit.passage
        ranks = f"dense #{hit.dense_rank or '-'} · bm25 #{hit.bm25_rank or '-'}"
        print(f"\n{n}. [{p.title}, p. {p.page}]  score {hit.score}  ({ranks})")
        if p.section:
            print(f"   § {p.section}")
        print("   " + p.text[:300] + ("…" if len(p.text) > 300 else ""))


if __name__ == "__main__":
    main()
