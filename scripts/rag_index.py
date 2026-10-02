"""Build the search index: PDFs of data/corpus.json -> passages -> vectors.

    uv run python scripts/rag_index.py                      # Mistral embeddings (EMBEDDER in .env)
    uv run python scripts/rag_index.py --embedder e5        # local model (uv sync --extra local)
    uv run python scripts/rag_index.py --embedder qwen3     # Qwen3-Embedding-0.6B, slower
    uv run python scripts/rag_index.py --passages-only      # just cut the passages (BM25 works)
    uv run python scripts/rag_index.py --chunk-chars 500    # another passage size (rebuilds)

Only passages whose vector is missing are embedded: re-running is free, and a
run interrupted by a rate limit resumes where it stopped.
"""

import argparse
import logging
import time
from pathlib import Path

from docagent.config import get_settings
from docagent.rag import (
    Chunker,
    build_embedder,
    build_passages,
    index_vectors,
    load_corpus,
    open_store,
    save_passages,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus.json"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--embedder", choices=["mistral", "e5", "qwen3", "hash"])
    parser.add_argument("--chunk-chars", type=int)
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--passages-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    settings = get_settings()
    chunker = Chunker(
        args.chunk_chars or settings.rag_chunk_chars, args.overlap or settings.rag_chunk_overlap
    )
    start = time.perf_counter()
    passages = build_passages(
        load_corpus(args.corpus),
        args.raw,
        chunker,
        on_doc=lambda doc, n: print(f"  {doc.id:<22} {n:>5} passages"),
    )
    path = settings.rag_index_dir / "passages.jsonl"
    save_passages(passages, path)
    print(f"{len(passages)} passages -> {path} ({time.perf_counter() - start:.0f}s)")
    if args.passages_only:
        return

    embedder = build_embedder(settings, args.embedder)
    store = open_store(settings, embedder)
    start = time.perf_counter()
    done = index_vectors(
        store,
        embedder,
        passages,
        on_batch=lambda n, total: print(f"  embedded {n}/{total}", flush=True),
    )
    print(
        f"{done} passages embedded with {embedder.name} in {time.perf_counter() - start:.0f}s; "
        f"collection {store.collection}: {store.count()} vectors"
    )
    tokens = getattr(embedder, "tokens", 0)
    if tokens:
        print(f"{embedder.requests} requests, {tokens} tokens")


if __name__ == "__main__":
    main()
