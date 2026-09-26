"""Translate PDFs with an LLM, keeping the layout.

The provider comes from .env (LLM_PROVIDER: mistral, groq, openrouter, ollama,
openai_compatible); --provider and --model override it for one run.

For each PDF, writes to data/output/:
- ``translated_<name>_<lang>.pdf``: the translated PDF
- ``compare_translated_<name>_<lang>.pdf``: original (left) and translation (right)

Translations are cached in .cache/translations.json: re-running on the same
PDF costs no API calls. Only send public documents (free tiers may use the
data to improve their models).

Usage:
    uv run python scripts/translate.py data/samples/giec_spm_en.pdf --pages 1 2
    uv run python scripts/translate.py data/samples/hcc_2025_2col.pdf --source fr --target en
    uv run python scripts/translate.py data/samples/giec_spm_en.pdf --provider groq
"""

import argparse
import logging
import tempfile
import time
from pathlib import Path

import pymupdf

from docagent.config import get_settings
from docagent.pdf import rebuild_document, side_by_side
from docagent.pdf.verify import verify_rebuild
from docagent.translate import CachedTranslator, build_translator, llm_stats

OUTPUT_DIR = Path("data/output")


def keep_pages(path: Path, pages: list[int] | None, tmp: Path) -> Path:
    """Copy of ``path`` limited to some 1-based pages (to test cheaply)."""
    if not pages:
        return path
    out = tmp / path.name
    with pymupdf.open(path) as src, pymupdf.open() as doc:
        for number in pages:
            doc.insert_pdf(src, from_page=number - 1, to_page=number - 1)
        doc.save(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pdfs", type=Path, nargs="+")
    parser.add_argument("--source", default="en", help="source language code (en)")
    parser.add_argument("--target", default="fr", help="target language code (fr)")
    parser.add_argument("--pages", type=int, nargs="*", help="only these 1-based pages")
    parser.add_argument("--provider", help="override LLM_PROVIDER")
    parser.add_argument("--model", help="override LLM_MODEL")
    parser.add_argument("--glossary", type=Path, help="JSON glossary (overrides GLOSSARY_PATH)")
    parser.add_argument("--no-cache", action="store_true", help="always call the API")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    settings = get_settings()
    overrides = {"llm_provider": args.provider, "llm_model": args.model}
    settings = settings.model_copy(update={k: v for k, v in overrides.items() if v})
    translator = build_translator(settings, glossary=args.glossary, use_cache=not args.no_cache)
    print(f"LLM: {settings.llm_provider} / {settings.resolved_model}")

    with tempfile.TemporaryDirectory() as tmp:
        for path in args.pdfs:
            source = keep_pages(path, args.pages, Path(tmp))
            name = f"{path.stem}_{args.target}"
            output = OUTPUT_DIR / f"translated_{name}.pdf"
            start = time.perf_counter()
            report = rebuild_document(
                source, output, translator, source_lang=args.source, target_lang=args.target
            )
            side_by_side(source, output, OUTPUT_DIR / f"compare_translated_{name}.pdf")
            check = verify_rebuild(source, output, report)
            print(
                f"{path.name}: {report.targets} zones, mean size {report.mean_scale:.0%}, "
                f"overflow {report.overflow}, check {'OK' if check.ok else 'FAIL'}, "
                f"{time.perf_counter() - start:.1f}s -> {output}"
            )
            if not check.ok:
                print(f"  checks: {check.model_dump(exclude={'file', 'lost_numbers'})}")

    s = llm_stats(translator)
    if s:
        print(
            f"\n{s.requests} requests, {s.retries} retries, "
            f"{s.input_tokens} input + {s.output_tokens} output tokens, "
            f"codes/glossary kept without the model {s.kept_locally}, "
            f"skipped blocks asked again {s.missing_asked_again}\n"
            f"style tags: restored locally {s.tags_restored}, repaired by the model "
            f"{s.tag_repairs}, lost {s.tag_fallbacks}"
        )
    if isinstance(translator, CachedTranslator):
        print(f"Cache: {translator.hits} hits, {translator.misses} misses")


if __name__ == "__main__":
    main()
