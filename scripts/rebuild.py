"""Rebuild PDFs with a fake translation (UPPERCASE, +20% length).

For each PDF, writes to data/output/:
- ``rebuilt_<name>.pdf``: the PDF with the fake translation in place
- ``compare_<name>.pdf``: original (left) and rebuilt (right) side by side

Usage:
    uv run python scripts/rebuild.py                      # all files in data/samples
    uv run python scripts/rebuild.py data/samples/giec_spm_en.pdf --expansion 0.3
"""

import argparse
import time
from functools import partial
from pathlib import Path

from docagent.pdf import rebuild_document, side_by_side
from docagent.pdf.verify import verify_rebuild
from docagent.translate import fake_translate

SAMPLES_DIR = Path("data/samples")
OUTPUT_DIR = Path("data/output")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pdfs", type=Path, nargs="*", help="default: data/samples/*.pdf")
    parser.add_argument("--expansion", type=float, default=0.2, help="length increase (0.2)")
    parser.add_argument("--min-scale", type=float, default=0.5, help="smallest font scale")
    parser.add_argument(
        "--keep-case", action="store_true", help="don't uppercase (closer to a real translation)"
    )
    args = parser.parse_args()
    paths = args.pdfs or sorted(SAMPLES_DIR.glob("*.pdf"))
    translate = partial(fake_translate, expansion=args.expansion, uppercase=not args.keep_case)

    header = (
        f"{'file':<24}{'targets':>8}{'orig font':>10}{'mean size':>10}{'scaled':>8}"
        f"{'overflow':>9}{'min scale':>10}"
        f"{'numbers kept':>14}{'overlaps':>10}{'check':>7}{'time':>7}"
    )
    print(header)
    print("-" * len(header))
    for path in paths:
        start = time.perf_counter()
        rebuilt = OUTPUT_DIR / f"rebuilt_{path.stem}.pdf"
        report = rebuild_document(path, rebuilt, translate, min_scale=args.min_scale)
        side_by_side(path, rebuilt, OUTPUT_DIR / f"compare_{path.stem}.pdf")
        check = verify_rebuild(path, rebuilt, translate)
        elapsed = time.perf_counter() - start
        numbers = f"{check.numbers_kept}/{check.numbers_total}"
        overlaps = f"{check.overlaps_before}->{check.overlaps_after}"
        print(
            f"{report.file:<24}{report.targets:>8}{report.original_font:>10}"
            f"{report.mean_scale:>9.0%} {report.scaled:>8}"
            f"{report.overflow:>9}"
            f"{report.min_scale:>10.2f}{numbers:>14}{overlaps:>10}{'OK' if check.ok else 'FAIL':>7}"
            f"{elapsed:>6.1f}s"
        )
        for problem in check.lost_numbers:
            print(f"    lost number {problem}")
    print(f"\nRebuilt and side-by-side PDFs written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
