"""Cut documents into passages: the unit that is indexed, retrieved and cited.

Built on the PDF extraction of the translator (``extract_blocks``): a passage is
a run of consecutive text blocks of ONE page, so every search result can cite
its page and even highlight its blocks (``bboxes``).

Rules, in order:
- running headers/footers (a short line repeated on many pages) are dropped;
- numbers alone, rotated text, word-less fragments and table-of-contents lines
  (dot leaders) are dropped;
- footnotes (text clearly smaller than the body) are packed after the body
  text of their page, so they don't cut a paragraph in two;
- a heading (bold or larger than body text, short, no final period) starts a new
  section: its text stays in the passage, and the section name is remembered
  for the next passages (it goes into the text that is embedded, see
  ``Passage.embed_text``);
- blocks are packed up to ``max_chars``; the next passage repeats the last
  blocks of the previous one up to ``overlap_chars``, so a sentence cut at a
  boundary is still whole in one of them; a block longer than ``max_chars`` is
  split at sentence ends.
"""

import re
from collections import Counter
from dataclasses import dataclass
from statistics import median

import pymupdf
from pydantic import BaseModel

from ..pdf.extract import extract_blocks
from ..pdf.models import BBox, TextBlock
from ..pdf.text import has_words

_SENTENCE_END_RE = re.compile(r"(?<=[.!?;])\s+")
_SPACES_RE = re.compile(r"\s+")
# Table-of-contents lines: "1.4 What does the Age of Electricity mean? ........ 47"
_DOT_LEADER_RE = re.compile(r"(\.\s?){6,}")


class Passage(BaseModel):
    id: str  # "<doc_id>:p<page>:<n>", stable while the document and settings don't change
    doc_id: str
    title: str
    page: int  # 1-based, as printed in citations
    section: str | None = None
    text: str
    bboxes: list[BBox] = []
    lang: str | None = None

    @property
    def embed_text(self) -> str:
        """What is embedded: the passage prefixed with its document and section.

        A passage like "It rose by 1.1°C since 1850" says nothing about what
        "it" is; the header ("IPCC AR6 SPM — A. The Current State of the
        Climate") gives the vector that context.
        """
        header = f"{self.title} — {self.section}" if self.section else self.title
        return f"{header}\n{self.text}"


@dataclass
class _Unit:
    text: str
    bbox: BBox
    heading: bool
    note: bool = False


def _normalise(text: str) -> str:
    return _SPACES_RE.sub(" ", text).strip()


def boilerplate_lines(pages: list[list[TextBlock]], min_share: float = 0.2) -> set[str]:
    """Short texts repeated on many pages: running headers, footers, page labels."""
    counts: Counter = Counter()
    for blocks in pages:
        counts.update({_normalise(b.text).lower() for b in blocks if len(b.text) < 80})
    threshold = max(3, min_share * len(pages))
    return {text for text, n in counts.items() if n >= threshold}


def body_font_size(blocks: list[TextBlock]) -> float:
    """Most common text size, weighted by characters (the body text)."""
    sizes = [b.size for b in blocks for _ in range(len(b.text))]
    return median(sizes) if sizes else 10.0


def _is_heading(block: TextBlock, body: float) -> bool:
    text = block.text.strip()
    return (
        3 <= len(text) <= 120
        and not text.endswith((".", ",", ";", ":"))
        and (block.bold or block.size >= 1.15 * body)
    )


def _units(blocks: list[TextBlock], body: float, skip: set[str]) -> list[_Unit]:
    units = []
    for block in blocks:
        text = _normalise(block.text)
        if (
            not block.horizontal
            or block.numeric
            or not has_words(text)
            or text.lower() in skip
            or _DOT_LEADER_RE.search(text)
        ):
            continue
        note = block.size < 0.85 * body
        units.append(_Unit(text, block.bbox, _is_heading(block, body) and not note, note))
    # PDFs often store footnotes before the body text: read the body first.
    return [u for u in units if not u.note] + [u for u in units if u.note]


def _split_long(unit: _Unit, max_chars: int) -> list[_Unit]:
    """A block longer than a passage, cut at sentence ends (then anywhere)."""
    if len(unit.text) <= max_chars:
        return [unit]
    pieces, current = [], ""
    for sentence in _SENTENCE_END_RE.split(unit.text):
        while len(sentence) > max_chars:  # one enormous "sentence": hard cut
            pieces.append((current + " " + sentence[:max_chars]).strip())
            sentence, current = sentence[max_chars:], ""
        if current and len(current) + 1 + len(sentence) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return [_Unit(p, unit.bbox, False, unit.note) for p in pieces]


class Chunker:
    def __init__(self, max_chars: int = 1000, overlap_chars: int = 150, min_chars: int = 80):
        if overlap_chars >= max_chars:
            raise ValueError("overlap_chars must be smaller than max_chars")
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.min_chars = min_chars

    def split_document(
        self, doc: pymupdf.Document, doc_id: str, title: str, lang: str | None = None
    ) -> list[Passage]:
        pages = [extract_blocks(page) for page in doc]
        body = body_font_size([b for blocks in pages for b in blocks])
        skip = boilerplate_lines(pages)
        passages: list[Passage] = []
        section: str | None = None
        for number, blocks in enumerate(pages, start=1):
            units = [
                piece
                for unit in _units(blocks, body, skip)
                for piece in _split_long(unit, self.max_chars)
            ]
            for texts, bboxes, heading in self._pack(units):
                section = heading or section
                text = " ".join(texts)
                if len(text) < self.min_chars:
                    continue
                passages.append(
                    Passage(
                        id=f"{doc_id}:p{number}:{len(passages)}",
                        doc_id=doc_id,
                        title=title,
                        page=number,
                        section=section,
                        text=text,
                        bboxes=bboxes,
                        lang=lang,
                    )
                )
        return passages

    def _pack(self, units: list[_Unit]):
        """Greedy packing with overlap: yields (texts, bboxes, last heading seen)."""
        current: list[_Unit] = []
        size = 0
        heading: str | None = None
        for unit in units:
            starts_notes = unit.note and current and not current[-1].note
            if (unit.heading or starts_notes) and current and not all(u.heading for u in current):
                # A new section starts: close the passage before its heading.
                yield [u.text for u in current], [u.bbox for u in current], heading
                current, size = [], 0
            if current and size + 1 + len(unit.text) > self.max_chars:
                yield [u.text for u in current], [u.bbox for u in current], heading
                current = self._overlap(current)
                size = sum(len(u.text) + 1 for u in current)
            if unit.heading:
                heading = unit.text
            current.append(unit)
            size += len(unit.text) + 1
        if current:
            yield [u.text for u in current], [u.bbox for u in current], heading

    def _overlap(self, units: list[_Unit]) -> list[_Unit]:
        """Last units of a passage, up to ``overlap_chars``, to start the next one."""
        carry: list[_Unit] = []
        size = 0
        for unit in reversed(units):
            if size + len(unit.text) > self.overlap_chars:
                break
            carry.insert(0, unit)
            size += len(unit.text) + 1
        return carry
