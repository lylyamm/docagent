"""Ask a question to the indexed reports: an answer with its sources.

    uv run python scripts/rag_ask.py "Quel est le premier secteur émetteur en France en 2024 ?"
    uv run python scripts/rag_ask.py "How much did sea level rise since 1901?" -k 8
    uv run python scripts/rag_ask.py "Qui a gagné la Coupe du monde 2018 ?"   # not in the corpus

Uses EMBEDDER for the search and the LLM of .env (LLM_PROVIDER, LLM_MODEL) for
the answer: one LLM request per question.
"""

import argparse
import time

from docagent.rag import build_answerer


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("question")
    parser.add_argument("-k", type=int, help="passages given to the LLM (RAG_TOP_K, default 5)")
    parser.add_argument("--retrieved", action="store_true", help="list every passage given")
    args = parser.parse_args()

    answerer = build_answerer()
    start = time.perf_counter()
    answer = answerer.ask(args.question, args.k)
    elapsed = time.perf_counter() - start

    print(f"\n{answer.answer}\n")
    for source in answer.sources:
        print(f"[{source.n}] {source.title}, p. {source.page}")
        if source.section:
            print(f"    § {source.section}")
        for quote in source.quotes:  # the sentences checked against the excerpt
            print(f"    « {quote} »")
    if args.retrieved:
        print("\nGiven to the model: " + ", ".join(answer.retrieved))
    chat = answerer.chat
    print(
        f"\n{answer.model} · {elapsed:.1f}s · {chat.input_tokens} tokens in, "
        f"{chat.output_tokens} out"
    )


if __name__ == "__main__":
    main()
