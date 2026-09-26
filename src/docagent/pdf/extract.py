"""Extract text blocks from PDF pages.

Built on ``page.get_text("rawdict")``, which returns a block -> line -> span ->
char hierarchy. Blocks are the translation unit: spans are too fine-grained
(LaTeX PDFs emit one span per word and per space) and are only used to infer
style. Characters are needed to cut leading bullets out of the text.
"""

import re
from collections import Counter
from pathlib import Path

import pymupdf

from .models import BBox, InlineStyle, TextBlock
from .text import font_style, is_numeric, is_symbol, join_lines

# Default flags, minus ligature preservation: "ﬃ" becomes "ffi".
EXTRACT_FLAGS = pymupdf.TEXTFLAGS_RAWDICT & ~pymupdf.TEXT_PRESERVE_LIGATURES

# A footnote or item number set apart from its text: "10   Since AR5, ...".
# The gap after it is wider than a normal space (as a fraction of the font size).
LABEL_MAX_CHARS = 3
LABEL_MIN_GAP = 0.6


def _union(boxes: list[BBox]) -> BBox:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _round(box) -> BBox:
    return tuple(round(v, 2) for v in box)


def _number_label(chars: list[tuple[dict, dict]], start: int) -> int:
    """Length of a number label at ``start`` ("10" in "10   Since AR5"), or 0.

    Only digits followed by a clearly wider gap than a space count: in running
    text ("10 countries") the gap is a normal space and nothing is cut.
    """
    end = start
    while end < len(chars) and chars[end][0]["c"].isdigit():
        end += 1
    if not 0 < end - start <= LABEL_MAX_CHARS:
        return 0
    rest = end
    while rest < len(chars) and chars[rest][0]["c"].isspace():
        rest += 1
    if rest == len(chars) or not chars[rest][0]["c"].isalpha():
        return 0
    size = chars[start][1]["size"]
    gap = chars[rest][0]["bbox"][0] - chars[end - 1][0]["bbox"][2]
    return rest - start if gap > LABEL_MIN_GAP * size else 0


def _split_line(line: dict) -> tuple[list[tuple[str, dict]], BBox | None, list[BBox]]:
    """Characters of a line (with their span), bbox of its text, and leading bullets.

    Leading bullets, and a number label set apart from the text ("10   Since"),
    are cut out of the text and of the bbox, so they are neither erased nor
    rewritten. Symbols elsewhere in the line are simply dropped.
    """
    chars = [(c, span) for span in line["spans"] for c in span["chars"]]
    bullets: list[BBox] = []
    i = 0
    while i < len(chars) and (chars[i][0]["c"].isspace() or is_symbol(chars[i][0]["c"])):
        if is_symbol(chars[i][0]["c"]):
            bullets.append(_round(chars[i][0]["bbox"]))
        i += 1
    label = _number_label(chars, i)
    if label:
        digits = [c["bbox"] for c, _ in chars[i : i + label] if not c["c"].isspace()]
        bullets.append(_round(_union(digits)))
        i += label
    body = [(c, span) for c, span in chars[i:] if not is_symbol(c["c"])]
    if bullets and body:
        # A bullet's glyph box may overlap the first letter; clamp it so that
        # protecting the bullet never protects part of the text.
        first_x = body[0][0]["bbox"][0]
        bullets = [(b[0], b[1], min(b[2], first_x), b[3]) for b in bullets]
        bullets = [b for b in bullets if b[2] - b[0] > 0.5]
    visible = [c["bbox"] for c, _ in body if not c["c"].isspace()]
    bbox = (
        _round((_union(visible)[0], line["bbox"][1], _union(visible)[2], line["bbox"][3]))
        if visible
        else None
    )
    return [(c["c"], span) for c, span in body], bbox, bullets


def _dominant_style(spans: list[dict]) -> tuple[str, float, int, bool, bool]:
    """Font, size, color, bold and italic most used in the block (by characters)."""
    weights: Counter = Counter()
    for span in spans:
        n_chars = sum(1 for c in span["chars"] if not c["c"].isspace() and not is_symbol(c["c"]))
        if n_chars:
            key = (
                span["font"],
                round(span["size"], 1),
                span["color"],
                *font_style(span["font"], span["flags"]),
            )
            weights[key] += n_chars
    return weights.most_common(1)[0][0]


def _position(span: dict, size: float, baseline: float) -> str | None:
    """ "sup"/"sub" for small text raised or lowered from the line's baseline."""
    if span["size"] >= 0.8 * size:
        return None
    shift = span["origin"][1] - baseline
    if shift < -0.1 * size:
        return "sup"
    if shift > 0.1 * size:
        return "sub"
    return None


def _markup(
    char_lines: list[list[tuple[str, dict]]],
    dominant: tuple[str, float, int, bool, bool],
) -> tuple[list[str], dict[str, InlineStyle]]:
    """Markup of each line: runs whose style differs from the dominant one get a tag.

    Only visible differences count (color, bold, italic, sub/superscript); a
    different font name alone is ignored. Identical styles share one tag.
    """
    font, size, color, bold, italic = dominant
    styles: dict[str, InlineStyle] = {}
    keys: dict[tuple, str] = {}
    markups = []
    for chars in char_lines:
        spans = [span for _, span in chars]
        baseline = max(
            (s["origin"][1] for s in spans if s["size"] >= 0.8 * size),
            default=spans[0]["origin"][1] if spans else 0,
        )
        out, current = [], None
        for char, span in chars:
            bold_, italic_ = font_style(span["font"], span["flags"])
            style = InlineStyle(
                font=span["font"],
                color=span["color"],
                bold=bold_,
                italic=italic_,
                position=_position(span, size, baseline),
            )
            differs = (style.color, style.bold, style.italic, style.position) != (
                color,
                bold,
                italic,
                None,
            )
            tag = None
            if differs and not char.isspace():
                key = (style.font, style.color, style.bold, style.italic, style.position)
                if key not in keys:
                    keys[key] = f"s{len(keys) + 1}"
                    styles[keys[key]] = style
                tag = keys[key]
            elif char.isspace():
                tag = current  # a space inside a styled run stays in the run
            if tag != current:
                if current:
                    out.append(f"</{current}>")
                if tag:
                    out.append(f"<{tag}>")
                current = tag
            out.append(char)
        if current:
            out.append(f"</{current}>")
        markups.append("".join(out).strip())
    return markups, styles


def _tidy_markup(markup: str) -> str:
    """Move spaces out of closing tags and collapse whitespace."""
    markup = re.sub(r"\s+(</s\d+>)", r"\1 ", markup)
    return re.sub(r"\s+", " ", markup).strip()


def _join_markup(line_markups: list[str]) -> str:
    """Join line markups like join_lines, then merge a run split across two lines."""
    joined = join_lines([_tidy_markup(m) for m in line_markups])
    return _tidy_markup(re.sub(r"</(s\d+)>(\s?)<\1>", r"\2", joined))


def _leading_label(boxes: list[BBox], lines: list[str]) -> bool:
    """True if the first line is a short label left of the text ("A.1.7", "(a)", "1.").

    Such labels share the band of the first text line, which would otherwise
    make the whole paragraph look like a table row.
    """
    if len(boxes) < 2 or len(lines[0]) > 10:
        return False
    first = boxes[0]
    beside = [
        b
        for b in boxes[1:]
        if min(first[3], b[3]) - max(first[1], b[1]) > 0.5 * (first[3] - first[1])
    ]
    return len(beside) == 1 and first[2] <= beside[0][0]


def _make_block(page, index, char_lines, line_bboxes, bullets, raw_lines) -> TextBlock:
    spans = [span for line in raw_lines for span in line["spans"]]
    dominant = _dominant_style(spans)
    font, size, color, bold, italic = dominant
    lines = ["".join(c for c, _ in chars).strip() for chars in char_lines]
    line_markups, styles = _markup(char_lines, dominant)
    text = join_lines(lines)
    horizontal = all(
        abs(line["dir"][0] - 1) < 1e-3 and abs(line["dir"][1]) < 1e-3 for line in raw_lines
    )
    return TextBlock(
        page=page.number,
        index=index,
        bbox=_union(line_bboxes),
        text=text,
        lines=lines,
        markup=_join_markup(line_markups) if styles else text,
        line_markups=[_tidy_markup(m) for m in line_markups],
        styles=styles,
        line_bboxes=line_bboxes,
        bullet_bboxes=bullets,
        font=font,
        size=size,
        color=color,
        bold=bold,
        italic=italic,
        horizontal=horizontal,
        numeric=is_numeric(text),
    )


def extract_blocks(page: pymupdf.Page) -> list[TextBlock]:
    """Return the non-empty text blocks of a page, in PyMuPDF reading order.

    A short label in front of a paragraph ("A.1.7") becomes its own block.
    """
    blocks: list[TextBlock] = []
    for index, block in enumerate(page.get_text("rawdict", flags=EXTRACT_FLAGS)["blocks"]):
        if block["type"] != 0:  # image block
            continue
        char_lines: list[list[tuple[str, dict]]] = []
        line_bboxes: list[BBox] = []
        raw_lines: list[dict] = []
        bullets: list[BBox] = []
        for line in block["lines"]:
            chars, bbox, line_bullets = _split_line(line)
            bullets.extend(line_bullets)
            if bbox is not None and "".join(c for c, _ in chars).strip():
                char_lines.append(chars)
                line_bboxes.append(bbox)
                raw_lines.append(line)
        lines = ["".join(c for c, _ in chars).strip() for chars in char_lines]
        if not join_lines(lines):
            continue
        if _leading_label(line_bboxes, lines):
            blocks.append(
                _make_block(page, index, char_lines[:1], line_bboxes[:1], [], raw_lines[:1])
            )
            char_lines, line_bboxes, raw_lines = char_lines[1:], line_bboxes[1:], raw_lines[1:]
        blocks.append(_make_block(page, index, char_lines, line_bboxes, bullets, raw_lines))
    return blocks


def extract_document(path: str | Path) -> list[TextBlock]:
    """Extract the text blocks of every page of a PDF."""
    with pymupdf.open(path) as doc:
        return [block for page in doc for block in extract_blocks(page)]
