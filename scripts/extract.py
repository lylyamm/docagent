"""Run extract_blocks on PDFs and save the results for inspection.

For each PDF, writes to data/output/:
- ``<name>_blocks.json``: every TextBlock as JSON
- ``extract_<name>.pdf``: the PDF with each block outlined by category
  (green = translatable, grey = numeric, orange = rotated)

Usage:
    uv run python scripts/extract.py                      # all files in data/samples
    uv run python scripts/extract.py data/samples/giec_spm_en.pdf
"""

import argparse
import json
from pathlib import Path

import pymupdf

from docagent.pdf import TextBlock, extract_blocks

SAMPLES_DIR = Path("data/samples")
OUTPUT_DIR = Path("data/output")

GREEN = (0.0, 0.6, 0.25)
GREY = (0.55, 0.55, 0.55)
ORANGE = (1.0, 0.5, 0.0)


def block_color(block: TextBlock) -> tuple[float, float, float]:
    if not block.horizontal:
        return ORANGE
    if block.numeric:
        return GREY
    return GREEN


def process(path: Path) -> dict:
    """Extract blocks, write the JSON dump and the annotated PDF, return stats."""
    doc = pymupdf.open(path)
    blocks: list[TextBlock] = []
    for page in doc:
        page_blocks = extract_blocks(page)
        blocks.extend(page_blocks)
        for block in page_blocks:
            rect = pymupdf.Rect(block.bbox)
            color = block_color(block)
            page.draw_rect(rect, color=color, width=0.8)
            page.insert_text(rect.tl + (1, 6), str(block.index), fontsize=6, color=color)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT_DIR / f"extract_{path.stem}.pdf", garbage=3, deflate=True)
    dump = [block.model_dump() for block in blocks]
    (OUTPUT_DIR / f"{path.stem}_blocks.json").write_text(
        json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "file": path.name,
        "pages": doc.page_count,
        "blocks": len(blocks),
        "translatable": sum(b.translatable for b in blocks),
        "numeric": sum(b.numeric for b in blocks),
        "rotated": sum(not b.horizontal for b in blocks),
        "chars_to_translate": sum(len(b.text) for b in blocks if b.translatable),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "pdfs", type=Path, nargs="*", help="PDF files (default: data/samples/*.pdf)"
    )
    args = parser.parse_args()
    paths = args.pdfs or sorted(SAMPLES_DIR.glob("*.pdf"))

    header = (
        f"{'file':<24}{'pages':>6}{'blocks':>8}{'transl.':>9}"
        f"{'numeric':>9}{'rotated':>9}{'chars':>9}"
    )
    print(header)
    print("-" * len(header))
    for path in paths:
        s = process(path)
        print(
            f"{s['file']:<24}{s['pages']:>6}{s['blocks']:>8}{s['translatable']:>9}"
            f"{s['numeric']:>9}{s['rotated']:>9}{s['chars_to_translate']:>9}"
        )
    print(f"\nJSON dumps and annotated PDFs written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
