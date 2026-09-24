"""Explore how PyMuPDF structures the text of a PDF page.

For each page, prints the block -> line -> span hierarchy returned by
``page.get_text("dict")`` and writes a debug PDF where every text block is
outlined in red (lines in blue) and numbered in reading order.

Usage:
    uv run python scripts/explore.py data/samples/hcc_2025_2col.pdf
    uv run python scripts/explore.py data/samples/arxiv_ieee_2col.pdf --pages 1 3
    uv run python scripts/explore.py data/samples/aie_weo_en.pdf --summary
"""

import argparse
from pathlib import Path

import pymupdf

OUTPUT_DIR = Path("data/output")
RED = (1, 0, 0)
BLUE = (0, 0, 1)

# Bit flags stored in span["flags"] (see PyMuPDF docs, "Text extraction flags").
FLAG_ITALIC = 2
FLAG_BOLD = 16


def describe_span(span: dict) -> str:
    """One-line description of a span: position, font, size, style and text."""
    x0, y0, x1, y1 = span["bbox"]
    style = ""
    if span["flags"] & FLAG_BOLD:
        style += "B"
    if span["flags"] & FLAG_ITALIC:
        style += "I"
    text = span["text"].replace("\n", " ")
    if len(text) > 40:
        text = text[:40] + "…"
    return (
        f"({x0:6.1f},{y0:6.1f},{x1:6.1f},{y1:6.1f}) "
        f"{span['font'][:22]:<22} {span['size']:5.1f}pt {style:<2} | {text!r}"
    )


def print_page_tree(page: pymupdf.Page) -> None:
    """Print every text block, its lines and their spans."""
    blocks = page.get_text("dict")["blocks"]
    for b_idx, block in enumerate(blocks):
        if block["type"] != 0:  # 1 = image block
            print(f"[block {b_idx}] IMAGE bbox={tuple(round(v, 1) for v in block['bbox'])}")
            continue
        bbox = tuple(round(v, 1) for v in block["bbox"])
        print(f"[block {b_idx}] TEXT bbox={bbox} lines={len(block['lines'])}")
        for l_idx, line in enumerate(block["lines"]):
            direction = "" if line["dir"] == (1.0, 0.0) else f" dir={line['dir']}"
            print(f"    line {l_idx}{direction}")
            for span in line["spans"]:
                print(f"        {describe_span(span)}")


def page_summary(page: pymupdf.Page) -> dict:
    """Aggregate statistics to compare pages and documents quickly."""
    blocks = page.get_text("dict")["blocks"]
    text_blocks = [b for b in blocks if b["type"] == 0]
    lines = [line for b in text_blocks for line in b["lines"]]
    spans = [s for line in lines for s in line["spans"]]
    return {
        "text_blocks": len(text_blocks),
        "image_blocks": len(blocks) - len(text_blocks),
        "lines": len(lines),
        "spans": len(spans),
        "fonts": len({s["font"] for s in spans}),
        "sizes": sorted({round(s["size"], 1) for s in spans}),
        "non_horizontal_lines": sum(1 for line in lines if line["dir"] != (1.0, 0.0)),
        "vector_drawings": len(page.get_drawings()),
    }


def draw_debug(page: pymupdf.Page) -> None:
    """Outline text blocks (red, numbered) and their lines (blue) on the page."""
    blocks = page.get_text("dict")["blocks"]
    for b_idx, block in enumerate(blocks):
        if block["type"] != 0:
            continue
        rect = pymupdf.Rect(block["bbox"])
        page.draw_rect(rect, color=RED, width=0.8)
        page.insert_text(rect.tl + (1, 7), str(b_idx), fontsize=7, color=RED)
        for line in block["lines"]:
            page.draw_rect(pymupdf.Rect(line["bbox"]), color=BLUE, width=0.3)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pdf", type=Path, help="PDF file to explore")
    parser.add_argument("--pages", type=int, nargs="*", help="1-based page numbers (default: all)")
    parser.add_argument("--summary", action="store_true", help="print per-page statistics only")
    args = parser.parse_args()

    doc = pymupdf.open(args.pdf)
    pages = args.pages or range(1, doc.page_count + 1)

    for number in pages:
        page = doc[number - 1]
        size = f"{page.rect.width:.0f}x{page.rect.height:.0f}pt"
        print(f"\n===== {args.pdf.name} · page {number}/{doc.page_count} · {size} =====")
        if args.summary:
            print(page_summary(page))
        else:
            print_page_tree(page)

    # The debug PDF is drawn on a fresh copy so the tree above reflects the original.
    debug_doc = pymupdf.open(args.pdf)
    for number in pages:
        draw_debug(debug_doc[number - 1])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"debug_{args.pdf.stem}.pdf"
    debug_doc.save(out_path, garbage=3, deflate=True)
    print(f"\nDebug PDF written to {out_path}")


if __name__ == "__main__":
    main()
