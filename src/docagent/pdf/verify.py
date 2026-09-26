"""Automatic checks comparing a source PDF with its rebuilt version."""

import re
from pathlib import Path

import pymupdf
from pydantic import BaseModel

from .extract import extract_blocks
from .rebuild import DocumentReport
from .text import is_numeric, strip_tags


class VerifyReport(BaseModel):
    file: str
    pages: int
    pages_same_images: int  # pages whose image count is unchanged
    drawings_before: int  # vector graphics visible on the pages
    drawings_after: int
    residual_source: int  # rewritten zones where the source text is still readable
    zones_checked: int
    overlaps_before: int  # pairs of words drawn on top of each other
    overlaps_after: int
    numbers_kept: int  # numeric blocks still readable at the same place
    numbers_total: int
    lost_numbers: list[str]
    markup_leaks: int = 0  # style or HTML tags printed as text ("CO<sub>2</sub>")
    empty_zones: int = 0  # rewritten zones where no text at all was written

    @property
    def ok(self) -> bool:
        return (
            self.pages_same_images == self.pages
            and self.drawings_after == self.drawings_before
            and self.residual_source == 0
            and self.numbers_kept == self.numbers_total
            and self.overlaps_after <= self.overlaps_before
            and self.markup_leaks == 0
            and self.empty_zones == 0
        )


def _visible_drawings(page: pymupdf.Page) -> int:
    """Vector drawings intersecting the page (ignores off-page artefacts)."""
    return sum(1 for d in page.get_drawings() if d["rect"].intersects(page.rect))


_MARKUP_RE = re.compile(r"</?(?:s\d+|sub|sup|b|i|em|strong|span)\b[^>]*>", re.I)


def _compact(text: str) -> str:
    return "".join(text.split())


def _text_in(words: list[tuple], rect: pymupdf.Rect) -> str:
    """Text of the words whose center lies in ``rect`` (one get_text call per page)."""
    return " ".join(
        w[4] for w in words if rect.contains(pymupdf.Point((w[0] + w[2]) / 2, (w[1] + w[3]) / 2))
    )


def count_overlaps(page: pymupdf.Page) -> int:
    """Pairs of words whose boxes overlap by more than 30% of the smaller one.

    Words are bucketed in a 20 pt grid so only neighbours are compared.
    """
    words = [pymupdf.Rect(w[:4]) for w in page.get_text("words")]
    cells: dict[tuple[int, int], list[int]] = {}
    for i, r in enumerate(words):
        for cx in range(int(r.x0) // 20, int(r.x1) // 20 + 1):
            for cy in range(int(r.y0) // 20, int(r.y1) // 20 + 1):
                cells.setdefault((cx, cy), []).append(i)
    pairs: set[tuple[int, int]] = set()
    for members in cells.values():
        for n, i in enumerate(members):
            for j in members[n + 1 :]:
                a, b = words[i], words[j]
                inter = a & b
                smaller = min(a.width * a.height, b.width * b.height)
                if (
                    smaller > 1
                    and not inter.is_empty
                    and inter.width * inter.height > 0.3 * smaller
                ):
                    pairs.add((min(i, j), max(i, j)))
    return len(pairs)


def verify_rebuild(source: str | Path, rebuilt: str | Path, report: DocumentReport) -> VerifyReport:
    """Check that graphics survived, source text is gone and numbers are untouched."""
    residual = checked = kept = total = 0
    same_images = drawings_before = drawings_after = 0
    overlaps_before = overlaps_after = leaks = empty = 0
    lost: list[str] = []
    with pymupdf.open(source) as src, pymupdf.open(rebuilt) as out:
        for before, after in zip(src, out, strict=True):
            same_images += len(before.get_images()) == len(after.get_images())
            drawings_before += _visible_drawings(before)
            drawings_after += _visible_drawings(after)
            overlaps_before += count_overlaps(before)
            overlaps_after += count_overlaps(after)
            tags_after = len(_MARKUP_RE.findall(after.get_text()))
            leaks += max(tags_after - len(_MARKUP_RE.findall(before.get_text())), 0)

            blocks = extract_blocks(before)
            words_after = after.get_text("words")
            for target in report.pages[before.number].zones:
                zone = pymupdf.Rect(target.bbox) + (-2, -2, 2, 2)
                if strip_tags(target.translation).strip() and not _text_in(words_after, zone):
                    empty += 1
                # Source words that the translation itself doesn't contain (the
                # fake translation's padding repeats source words in lowercase).
                translated = strip_tags(target.translation)
                words = [
                    w
                    for w in strip_tags(target.source).split()
                    if len(w) > 4 and w != w.upper() and w not in translated
                ]
                if not words:
                    continue
                checked += 1
                found = _text_in(words_after, pymupdf.Rect(target.bbox))
                if sum(w in found for w in words) > len(words) / 2:
                    residual += 1

            for block in blocks:
                if not block.horizontal:
                    continue
                for line, bbox in zip(block.lines, block.line_bboxes, strict=True):
                    if not is_numeric(line):
                        continue
                    total += 1
                    zone = pymupdf.Rect(bbox) + (-2, -2, 2, 2)
                    if _compact(line) in _compact(_text_in(words_after, zone)):
                        kept += 1
                    else:
                        lost.append(f"p{before.number + 1}: {line}")

        return VerifyReport(
            file=Path(source).name,
            pages=src.page_count,
            pages_same_images=same_images,
            drawings_before=drawings_before,
            drawings_after=drawings_after,
            residual_source=residual,
            overlaps_before=overlaps_before,
            overlaps_after=overlaps_after,
            zones_checked=checked,
            numbers_kept=kept,
            numbers_total=total,
            lost_numbers=lost,
            markup_leaks=leaks,
            empty_zones=empty,
        )
