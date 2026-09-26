"""Rebuild a PDF with translated text written back in place.

Steps for each page:
1. pick the blocks to translate (``TextBlock.translatable``); table rows are
   handled line by line so each cell keeps its own position;
2. erase the original text with redactions, leaving images and vector
   graphics (charts, table backgrounds) untouched;
3. write the translation with ``insert_htmlbox``, in the original font when
   the embedded font has all the needed glyphs; the box may grow into free
   space below it before the font is shrunk (down to ``min_scale``).
"""

import re
from collections.abc import Callable
from html import escape
from pathlib import Path

import pymupdf
from pydantic import BaseModel

from ..translate.base import Translator, as_translator
from .extract import extract_blocks
from .models import BBox, TextBlock
from .text import TAG_RE, has_words, is_math_font, is_numeric, strip_tags

# Shrink redaction rectangles slightly so they don't bite into neighbouring text.
REDACT_MARGIN = 0.5


class Target(BaseModel):
    """A piece of text to erase and rewrite at a given position.

    ``source`` and ``translation`` are markup: inline style tags ("<s1>…</s1>")
    refer to ``block.styles`` and are turned back into colours, bold, italic,
    sub/superscripts when the text is written.
    """

    bbox: BBox
    source: str
    translation: str = ""
    block: TextBlock

    @property
    def plain_translation(self) -> str:
        return strip_tags(self.translation)


class Zone(BaseModel):
    """What was written where: kept in the report for checks and debugging."""

    bbox: BBox
    source: str
    translation: str


class PageReport(BaseModel):
    page: int
    targets: int
    mean_scale: float  # average font scale over rewritten zones (1.0 = original size)
    scaled: int  # font had to be reduced to fit
    overflow: int  # did not fit even at min_scale
    original_font: int  # rewritten with the document's own embedded font
    min_scale: float
    zones: list[Zone] = []


class DocumentReport(BaseModel):
    file: str
    pages: list[PageReport]

    @property
    def targets(self) -> int:
        return sum(p.targets for p in self.pages)

    @property
    def scaled(self) -> int:
        return sum(p.scaled for p in self.pages)

    @property
    def overflow(self) -> int:
        return sum(p.overflow for p in self.pages)

    @property
    def mean_scale(self) -> float:
        weighted = sum(p.mean_scale * p.targets for p in self.pages)
        return round(weighted / self.targets, 3) if self.targets else 1.0

    @property
    def original_font(self) -> int:
        return sum(p.original_font for p in self.pages)

    @property
    def min_scale(self) -> float:
        return min((p.min_scale for p in self.pages), default=1.0)


def _font_family(font: str) -> str:
    """Generic CSS family closest to an embedded font, used when it can't be reused."""
    name = font.lower()
    if any(k in name for k in ("mono", "courier", "consolas")):
        return "monospace"
    if "sans" in name:
        return "sans-serif"
    if any(
        k in name for k in ("times", "roman", "nimbusrom", "serif", "georgia", "garamond", "cmr")
    ):
        return "serif"
    return "sans-serif"


def _base_name(font: str) -> str:
    """'ABCDEF+Lato-Bold' -> 'Lato-Bold' (drop the subset prefix)."""
    return font.split("+", 1)[-1]


class FontCatalog:
    """Fonts embedded in a document, reusable to rewrite text in its original font.

    Embedded fonts are usually subsets: they only contain the glyphs the source
    text needed. A font is reused only if it has every character of the new text;
    otherwise the whole text falls back to a generic family (mixing fonts inside
    a word looks worse than a consistent substitute).
    """

    def __init__(self, doc: pymupdf.Document) -> None:
        self.archive = pymupdf.Archive()
        self._fonts: dict[str, list[tuple[str, pymupdf.Font]]] = {}
        self._ink: dict[tuple[str, str], bool] = {}
        seen: set[int] = set()
        for page in doc:
            for entry in page.get_fonts(full=True):
                xref, basefont = entry[0], entry[3]
                if xref in seen or xref == 0:
                    continue
                seen.add(xref)
                _, ext, _, buffer = doc.extract_font(xref)
                if not buffer or ext in ("n/a", ""):
                    continue
                try:
                    font = pymupdf.Font(fontbuffer=buffer)
                except Exception:  # noqa: BLE001 - unreadable font program: just skip it
                    continue
                filename = f"font{xref}.{ext}"
                self.archive.add(buffer, filename)
                self._fonts.setdefault(_base_name(basefont), []).append((filename, font))

    def find(self, font_name: str, text: str) -> str | None:
        """File name of an embedded font able to draw ``text``, or None."""
        needed = {c for c in text if not c.isspace()}
        for filename, font in self._fonts.get(_base_name(font_name), []):
            if all(self._can_draw(filename, font, c) for c in needed):
                return filename
        return None

    def _can_draw(self, filename: str, font: pymupdf.Font, char: str) -> bool:
        """True if the font really draws ``char``.

        ``has_glyph`` is not enough: subsetting often keeps the glyph slot and
        its advance width but empties the outline, so the letter would come out
        blank. The glyph is rendered once and checked for ink (result cached).
        """
        key = (filename, char)
        if key not in self._ink:
            ok = bool(font.has_glyph(ord(char)))
            if ok:
                page = pymupdf.open().new_page(width=24, height=24)
                writer = pymupdf.TextWriter(page.rect)
                writer.append((2, 20), char, font=font, fontsize=20)
                writer.write_text(page)
                pix = page.get_pixmap(dpi=36, colorspace=pymupdf.csGRAY)
                ok = min(pix.samples) < 128
            self._ink[key] = ok
        return self._ink[key]


# HTML tags a translator may add on its own ("XXI<sup>e</sup>", "CO<sub>2</sub>").
_HTML_CSS = {
    "sub": "vertical-align: sub; font-size: 70%",
    "sup": "vertical-align: super; font-size: 70%",
    "b": "font-weight: bold",
    "strong": "font-weight: bold",
    "i": "font-style: italic",
    "em": "font-style: italic",
}
_PART_RE = re.compile(r"(</?s\d+>|</?(?:sub|sup|b|i|em|strong)>|<br\s*/?>)", re.I)


def markup_to_html(target: Target, fonts: "FontCatalog | None") -> tuple[str, str]:
    """HTML for a target's translation, and the @font-face rules its runs need.

    Each tagged run becomes a <span> with its own colour, weight, slant and
    vertical position, in its original font when that font can draw the run.
    HTML tags the translator added (``<sup>``, ``<sub>``, ``<b>``, ``<i>``) are
    rendered too. Unknown or unbalanced tags are dropped, never shown as text.
    """
    styles = target.block.styles
    faces: list[str] = []
    html: list[str] = []
    stack: list[str] = []  # open spans, by tag name ("s1", "sup"...)
    for part in _PART_RE.split(target.translation):
        if not part:
            continue
        if not _PART_RE.fullmatch(part):
            html.append(escape(part))
            continue
        name = part.strip("</>").lower()
        if name.startswith("br"):
            html.append(" ")
            continue
        if part.startswith("</"):
            if name in stack:  # close it, and anything left open inside it
                while stack:
                    html.append("</span>")
                    if stack.pop() == name:
                        break
            continue
        if name in _HTML_CSS:
            html.append(f'<span style="{_HTML_CSS[name]}">')
            stack.append(name)
            continue
        if name not in styles or any(TAG_RE.fullmatch(f"<{n}>") for n in stack):
            continue  # unknown style, or nested in another style run
        html.append(f'<span style="{_run_css(target, name, fonts, faces)}">')
        stack.append(name)
    html.extend("</span>" for _ in stack)
    return "".join(html), " ".join(faces)


def _run_css(target: Target, name: str, fonts: "FontCatalog | None", faces: list[str]) -> str:
    """CSS of a style run; adds the @font-face it needs to ``faces``."""
    style = target.block.styles[name]
    run_text = strip_tags(_run_text(target.translation, name))
    css = [f"color: #{style.color:06x}"]
    font_file = fonts.find(style.font, run_text) if fonts else None
    if font_file:
        family = f"run{len(faces)}"
        faces.append(f"@font-face {{ font-family: {family}; src: url({font_file}); }}")
        css += [f"font-family: {family}", "font-weight: normal", "font-style: normal"]
    else:
        css += [
            f"font-weight: {'bold' if style.bold else 'normal'}",
            f"font-style: {'italic' if style.italic else 'normal'}",
        ]
        if fonts and style.font != target.block.font:
            css.append(f"font-family: {_font_family(style.font)}")
    if style.position:
        css += [f"vertical-align: {'super' if style.position == 'sup' else 'sub'}"]
        css += ["font-size: 70%"]
    return "; ".join(css)


def _run_text(markup: str, name: str) -> str:
    """Concatenated text of every run tagged ``name``."""
    return "".join(re.findall(rf"<{name}>(.*?)</{name}>", markup, flags=re.S))


def _line_height(block: TextBlock) -> float:
    """Line spacing of the original block, as a multiple of the font size."""
    tops = [b[1] for b in block.line_bboxes]
    if len(tops) < 2 or block.rewrite_by_line:
        return 1.15
    pitch = (tops[-1] - tops[0]) / (len(tops) - 1)
    return min(max(pitch / block.size, 1.0), 1.6)


def block_css(
    block: TextBlock,
    font_file: str | None = None,
    scale: float = 1.0,
    faces: str = "",
    align: str | None = None,
) -> str:
    """CSS reproducing the dominant style of a block, optionally scaled down.

    With ``font_file`` (an embedded font), bold/italic are already in the font
    program, so no synthetic weight or slant is added on top.
    """
    if font_file:
        face = f"@font-face {{ font-family: orig; src: url({font_file}); }} "
        family, weight, style = "orig", "normal", "normal"
    else:
        face = ""
        family = _font_family(block.font)
        weight = "bold" if block.bold else "normal"
        style = "italic" if block.italic else "normal"
    if align is None:
        align = "left" if block.rewrite_by_line else (block.alignment or "left")
    # First line starting further right (indent, or a bullet kept in place).
    indent = 0.0 if block.rewrite_by_line else block.line_bboxes[0][0] - block.bbox[0]
    return (
        f"{face}{faces} * {{"
        f" font-family: {family};"
        f" font-size: {block.size * scale:.2f}pt;"
        f" line-height: {_line_height(block):.2f};"
        f" color: #{block.color:06x};"
        f" font-weight: {weight};"
        f" font-style: {style};"
        f" text-align: {align};"
        f" text-indent: {indent if indent > 1 else 0:.1f}pt;"
        " margin: 0; padding: 0; }"
    )


def stacked_lines(blocks: list[TextBlock]) -> set[BBox]:
    """Line boxes drawn on top of another line: a symbol with its sub/superscripts.

    LaTeX nomenclatures ("C^curt_{s/w}") come out as tiny overlapping lines,
    often in different blocks. Rewriting each one in its own box would pile the
    texts up, so they are left untouched (formulas are not translated anyway).
    Lines are bucketed in a 20 pt grid so only neighbours are compared.
    """
    boxes = [b for block in blocks for b in block.line_bboxes]
    cells: dict[tuple[int, int], list[int]] = {}
    for i, b in enumerate(boxes):
        for cx in range(int(b[0]) // 20, int(b[2]) // 20 + 1):
            for cy in range(int(b[1]) // 20, int(b[3]) // 20 + 1):
                cells.setdefault((cx, cy), []).append(i)
    stacked: set[BBox] = set()
    for members in cells.values():
        for n, i in enumerate(members):
            for j in members[n + 1 :]:
                a, b = boxes[i], boxes[j]
                w = min(a[2], b[2]) - max(a[0], b[0])
                h = min(a[3], b[3]) - max(a[1], b[1])
                smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
                if w > 0 and h > 0 and w * h > 0.4 * smaller:
                    stacked.update((a, b))
    return stacked


def plan_targets(blocks: list[TextBlock]) -> list[Target]:
    """Decide what to rewrite: whole paragraphs, or single lines (cells, chart labels)."""
    targets: list[Target] = []
    stacked = stacked_lines(blocks)
    for block in blocks:
        if not block.translatable:
            continue
        if len(block.line_bboxes) == 1 and block.line_bboxes[0] in stacked:
            continue
        if block.rewrite_by_line:
            markups = block.line_markups or block.lines
            for text, markup, bbox in zip(block.lines, markups, block.line_bboxes, strict=True):
                if bbox not in stacked and has_words(text) and not is_numeric(text):
                    targets.append(Target(bbox=bbox, source=markup, block=block))
        else:
            markup = block.markup or block.text
            targets.append(Target(bbox=block.bbox, source=markup, block=block))
    return targets


def inline_fragments(blocks: list[TextBlock], targets: list[Target]) -> list[BBox]:
    """Word-less fragments (LaTeX math like "b", "t") sitting inside a paragraph.

    They belong to the sentence being rewritten: the new text reflows, so they
    can't stay at their old position. They are erased with the paragraph (and
    lost for now: see DECISIONS.md).
    """
    paragraphs = [pymupdf.Rect(t.bbox) for t in targets if not t.block.rewrite_by_line]
    fragments = []
    for block in blocks:
        if block.numeric or (has_words(block.text) and not is_math_font(block.font)):
            continue
        rect = pymupdf.Rect(block.bbox)
        area = rect.width * rect.height
        if any((rect & p).width * (rect & p).height > 0.5 * area for p in paragraphs):
            fragments.append(block.bbox)
    return fragments


def protected_boxes(
    blocks: list[TextBlock], targets: list[Target], erased: list[BBox] | None = None
) -> list[pymupdf.Rect]:
    """Boxes that must survive: numbers, rotated labels, bullets, untranslated lines."""
    skip = {tuple(t.bbox) for t in targets} | {tuple(b) for b in erased or []}
    boxes = [pymupdf.Rect(b) for block in blocks for b in block.bullet_bboxes]
    for block in blocks:
        if tuple(block.bbox) in skip:
            continue
        boxes.extend(pymupdf.Rect(b) for b in block.line_bboxes if tuple(b) not in skip)
    return boxes


def redaction_rects(target: Target, protected: list[pymupdf.Rect]) -> list[pymupdf.Rect]:
    """Rectangles to erase for a target: one per line, trimmed around kept text.

    Redaction removes every glyph that touches the rectangle. Each line is
    erased separately so that a bullet before the first line only trims that
    line. A rectangle is cut above/below a protected line it overlaps (line
    boxes of big fonts overlap the next line: "-37,4" above "Mt éqCO2"), or
    left/right of it when they share the same line (a bullet before its text).
    """
    lines = [target.bbox] if target.block.rewrite_by_line else target.block.line_bboxes
    rects = []
    for bbox in lines:
        rect = pymupdf.Rect(bbox) + (REDACT_MARGIN, REDACT_MARGIN, -REDACT_MARGIN, -REDACT_MARGIN)
        for box in protected:
            if not rect.intersects(box):
                continue
            same_line = min(rect.y1, box.y1) - max(rect.y0, box.y0) > 0.5 * box.height
            if same_line:
                if (box.x0 + box.x1) / 2 < (rect.x0 + rect.x1) / 2:
                    rect.x0 = max(rect.x0, box.x1 + REDACT_MARGIN)
                else:
                    rect.x1 = min(rect.x1, box.x0 - REDACT_MARGIN)
            elif (box.y0 + box.y1) / 2 < (rect.y0 + rect.y1) / 2:
                rect.y0 = max(rect.y0, box.y1 + REDACT_MARGIN)
            else:
                rect.y1 = min(rect.y1, box.y0 - REDACT_MARGIN)
        if not rect.is_empty:
            rects.append(rect)
    return rects


class FreeSpace:
    """Coarse occupancy map of a page (1 pt cells) to let text boxes grow downwards.

    Obstacles are all original text lines, bullets, images and vector drawings,
    except large drawings that contain text (coloured panels, frames): those
    are boundaries the text must stay inside.
    """

    def __init__(self, page: pymupdf.Page, blocks: list[TextBlock]) -> None:
        self.width = int(page.rect.width) + 1
        self.height = int(page.rect.height) + 1
        self.grid = bytearray(self.width * self.height)
        self.panels: list[pymupdf.Rect] = []
        texts = [pymupdf.Rect(b) for block in blocks for b in block.line_bboxes]
        for rect in texts:
            self._mark(rect)
        for block in blocks:
            for b in block.bullet_bboxes:
                self._mark(pymupdf.Rect(b))
        for info in page.get_image_info():
            self._mark(pymupdf.Rect(info["bbox"]), skip_if_contains=texts)
        for drawing in page.get_drawings():
            self._mark(drawing["rect"], skip_if_contains=texts)

    def _mark(self, rect: pymupdf.Rect, skip_if_contains: list[pymupdf.Rect] | None = None) -> None:
        large = rect.width * rect.height > 400
        if skip_if_contains and large and any(rect.contains(t) for t in skip_if_contains):
            self.panels.append(rect)  # background panel: not an obstacle, but a boundary
            return
        x0, x1 = max(int(rect.x0), 0), min(int(rect.x1) + 1, self.width)
        y0, y1 = max(int(rect.y0), 0), min(int(rect.y1) + 1, self.height)
        if x0 >= x1:
            return
        filled = b"\x01" * (x1 - x0)
        for y in range(y0, y1):
            self.grid[y * self.width + x0 : y * self.width + x1] = filled

    def room_below(self, bbox: BBox, limit: float, keep_gap: float = 0.0) -> float:
        """Free height (pt) directly below ``bbox``, up to ``limit``.

        ``keep_gap`` is the share of the original gap to the next obstacle that
        must stay empty: paragraphs keep a visible space between them instead
        of the translation filling it. Text never leaves its coloured panel.
        """
        rect = pymupdf.Rect(bbox)
        for panel in self.panels:
            if panel.contains(rect):
                limit = min(limit, panel.y1 - bbox[3] - 2)
        x0, x1 = max(int(bbox[0]) + 1, 0), min(int(bbox[2]), self.width)
        start = int(bbox[3]) + 2
        # Look further than the limit: the gap is measured to the next obstacle.
        horizon = limit / (1 - keep_gap) if keep_gap < 1 else limit
        for y in range(start, min(int(bbox[3] + horizon) + 3, self.height)):
            if any(self.grid[y * self.width + x0 : y * self.width + x1]):
                gap = y - bbox[3]
                return max(min(limit, gap * (1 - keep_gap) - 2), 0)
        return max(min(limit, self.height - 1 - bbox[3] - 2), 0)

    def gap_below(self, bbox: BBox, horizon: float) -> float:
        """Distance (pt) from ``bbox`` down to the first obstacle, up to ``horizon``."""
        x0, x1 = max(int(bbox[0]) + 1, 0), min(int(bbox[2]), self.width)
        for y in range(int(bbox[3]) + 1, min(int(bbox[3] + horizon) + 1, self.height)):
            if any(self.grid[y * self.width + x0 : y * self.width + x1]):
                return max(y - bbox[3], 0)
        return horizon

    def room_right(self, rect: pymupdf.Rect, limit: float) -> float:
        """Free width (pt) directly right of ``rect``, up to ``limit``."""
        for panel in self.panels:
            if panel.contains(rect):
                limit = min(limit, panel.x1 - rect.x1 - 2)
        y0, y1 = max(int(rect.y0) + 1, 0), min(int(rect.y1), self.height)
        start = int(rect.x1) + 2
        for x in range(start, min(int(rect.x1 + limit) + 1, self.width)):
            if any(self.grid[y * self.width + x] for y in range(y0, y1)):
                return max(x - rect.x1 - 2, 0)
        return max(min(limit, self.width - 1 - rect.x1 - 2), 0)

    def room_left(self, rect: pymupdf.Rect, limit: float) -> float:
        """Free width (pt) directly left of ``rect``, up to ``limit``."""
        for panel in self.panels:
            if panel.contains(rect):
                limit = min(limit, rect.x0 - panel.x0 - 2)
        y0, y1 = max(int(rect.y0) + 1, 0), min(int(rect.y1), self.height)
        start = int(rect.x0) - 2
        for x in range(start, max(int(rect.x0 - limit) - 1, -1), -1):
            if any(self.grid[y * self.width + x] for y in range(y0, y1)):
                return max(rect.x0 - x - 2, 0)
        return max(min(limit, rect.x0 - 2), 0)


# Share of the space between two paragraphs that the translation may not fill.
KEEP_GAP = 0.6


def insert_rect(target: Target, space: FreeSpace) -> pymupdf.Rect:
    """Where to write a target: its bbox, extended into free space.

    - Down: capped at the original height plus one line, so text doesn't drift
      far from where it was, and never closer than ``KEEP_GAP`` of the original
      spacing to the text below (paragraphs stay visibly separated).
    - Right: half a font size of slack when free. The bbox is the exact extent
      of the source glyphs, so a word that the source fitted on the line could
      miss it by less than a point here and wrap, leaving a visibly short line.
    """
    x0, y0, x1, y1 = target.bbox
    size = target.block.size
    room = space.room_below(target.bbox, (y1 - y0) + size * 1.2, keep_gap=KEEP_GAP)
    rect = pymupdf.Rect(x0, y0, x1, y1 + room)
    return rect + (0, 0, space.room_right(rect, 0.5 * size), 0)


def single_line_rect(
    target: Target, space: FreeSpace, column: tuple[float, float], page_width: float
) -> tuple[pymupdf.Rect, str]:
    """Box and alignment for a one-line text that should stay on one line.

    A heading or running header wraps badly ("Résumé à l'intention des /
    décideurs"): the box is widened sideways, within the page's text column and
    the free space, on the side the line is anchored to.
    """
    x0, y0, x1, y1 = target.bbox
    rect = pymupdf.Rect(target.bbox)
    left_edge, right_edge = column
    center = (x0 + x1) / 2
    if x1 > right_edge - 2 and x0 - left_edge > 0.25 * (right_edge - left_edge):
        return rect - (space.room_left(rect, x0 - left_edge), 0, 0, 0), "right"
    if abs(center - page_width / 2) < 2 and x0 - left_edge > 20:
        side = min(space.room_left(rect, x0 - left_edge), space.room_right(rect, right_edge - x1))
        return rect + (-side, 0, side, 0), "center"
    return rect + (0, 0, space.room_right(rect, max(right_edge - x1, 0)), 0), "left"


def body_boxes(blocks: list[TextBlock]) -> list[BBox]:
    """Boxes of the page's paragraphs (multi-line blocks), which define its columns."""
    body = [b.bbox for b in blocks if b.horizontal and len(b.line_bboxes) > 1]
    return body or [b.bbox for b in blocks if b.horizontal]


def text_column(bodies: list[BBox], bbox: BBox) -> tuple[float, float]:
    """Left and right edges of the text column a line belongs to.

    The column is made of the paragraphs spanning the line's left edge, so in a
    two-column layout a line of the left column never widens into the right
    one. Headers or side tabs outside every paragraph use the whole body width.
    """
    x0 = bbox[0]
    same = [b for b in bodies if b[0] - 2 <= x0 < b[2]] or bodies
    if not same:
        return bbox[0], bbox[2]
    return min(b[0] for b in same), max(b[2] for b in same)


# Below this scale, a one-line text may wrap onto a second line instead.
SINGLE_LINE_MIN_SCALE = 0.85


def _style_key(block: TextBlock) -> tuple:
    return (block.font, round(block.size), block.bold, block.italic, block.rewrite_by_line)


def _fit_scale(
    page: pymupdf.Page,
    rect: pymupdf.Rect,
    html: str,
    css: str,
    fonts: FontCatalog | None,
    min_scale: float,
) -> float:
    """Scale insert_htmlbox would need to fit ``html`` in ``rect`` (dry run)."""
    scratch = pymupdf.open().new_page(width=page.rect.width, height=page.rect.height)
    spare, scale = scratch.insert_htmlbox(
        rect,
        html,
        css=css,
        archive=fonts.archive if fonts else None,
        scale_low=min_scale,
    )
    return scale if spare >= 0 else min_scale


def _group_scales(plans: list[tuple]) -> dict[tuple, float]:
    """One starting scale per style: the median of what its blocks need.

    Using the minimum would let one cramped block shrink every similar
    paragraph; blocks that need more than the group scale are shrunk further
    individually. Measured on the samples: no harmonisation gives 87% mean size,
    median 84%, 20th percentile 82% (fake translation keeping case, +20%).
    """
    needs: dict[tuple, list[float]] = {}
    for target, _font, _rect, needed, *_ in plans:
        needs.setdefault(_style_key(target.block), []).append(needed)
    return {key: sorted(v)[(len(v) - 1) // 2] for key, v in needs.items()}


def rebuild_page(
    page: pymupdf.Page,
    translator: "Translator | Callable[[str], str]",
    min_scale: float = 0.5,
    fonts: FontCatalog | None = None,
    source_lang: str = "en",
    target_lang: str = "fr",
) -> PageReport:
    """Replace the translatable text of a page in place.

    All the page's texts go to the translator in one call, in reading order,
    so an LLM sees each block in context.
    """
    blocks = extract_blocks(page)
    targets = plan_targets(blocks)
    if targets:
        translations = as_translator(translator).translate_blocks(
            [t.source for t in targets], source_lang, target_lang
        )
        for target, translation in zip(targets, translations, strict=True):
            target.translation = translation
    fragments = inline_fragments(blocks, targets)
    protected = protected_boxes(blocks, targets, erased=fragments)
    space = FreeSpace(page, blocks)  # computed before erasing anything

    # 1. Erase: text only; keep images and line art (charts, table fills).
    for target in targets:
        for rect in redaction_rects(target, protected):
            page.add_redact_annot(rect, fill=False)
    for bbox in fragments:
        page.add_redact_annot(pymupdf.Rect(bbox), fill=False)
    page.apply_redactions(
        images=pymupdf.PDF_REDACT_IMAGE_NONE,
        graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
    )

    # 2. Rewrite in the original font when possible, growing the box into free
    #    space first and shrinking the text only if it still doesn't fit.
    #    Blocks sharing a style get the same scale, so similar paragraphs keep
    #    the same size instead of each one shrinking by a different amount.
    bodies = body_boxes(blocks)
    plans = []
    for target in targets:
        font_file = fonts.find(target.block.font, target.plain_translation) if fonts else None
        html, faces = markup_to_html(target, fonts)
        css = block_css(target.block, font_file, faces=faces)
        rect = insert_rect(target, space)
        needed = _fit_scale(page, rect, html, css, fonts, min_scale)
        if needed < 1:
            # Still too big: allow a few points to the right when that space is
            # free. The box is the source text's exact width, so a word missing
            # the line by 1 pt used to wrap and leave a visibly short line.
            limit = min(target.block.size, 0.04 * rect.width)
            wider = rect + (0, 0, space.room_right(rect, limit), 0)
            wider_needed = _fit_scale(page, wider, html, css, fonts, min_scale)
            if wider_needed > needed:
                rect, needed = wider, wider_needed
        align = None
        one_line = len(target.block.line_bboxes) == 1 or target.block.rewrite_by_line
        if one_line and space.gap_below(target.bbox, target.block.size) > 0.3 * target.block.size:
            # One line in the source: keep one line, widened sideways, unless
            # that needs a much smaller font than wrapping does. Not for a line
            # glued to the text below it (a caption split into several blocks).
            column = text_column(bodies, target.bbox)
            line_rect, line_align = single_line_rect(target, space, column, page.rect.width)
            line_css = block_css(target.block, font_file, faces=faces, align=line_align)
            line_needed = _fit_scale(page, line_rect, html, line_css, fonts, min_scale)
            if line_needed >= min(needed, SINGLE_LINE_MIN_SCALE):
                rect, needed, align = line_rect, line_needed, line_align
        plans.append((target, font_file, rect, needed, html, faces, align))
    group_scale = _group_scales(plans)

    scaled = overflow = reused = 0
    smallest = 1.0
    scales: list[float] = []
    for target, font_file, rect, _needed, html, faces, align in plans:
        reused += font_file is not None
        start = group_scale[_style_key(target.block)]
        spare_height, scale = page.insert_htmlbox(
            rect,
            html,
            css=block_css(target.block, font_file, scale=start, faces=faces, align=align),
            archive=fonts.archive if fonts else None,
            scale_low=min_scale / start,
        )
        if spare_height < 0:
            # Too long even at min_scale: insert_htmlbox wrote nothing. Write it
            # anyway, as small as needed, rather than lose the text (counted).
            spare_height, scale = page.insert_htmlbox(
                rect,
                html,
                css=block_css(target.block, font_file, scale=start, faces=faces, align=align),
                archive=fonts.archive if fonts else None,
                scale_low=0,
            )
            spare_height = -1.0
        scale *= start
        scales.append(scale)
        if spare_height < 0:
            overflow += 1
        elif scale < 0.999:
            scaled += 1
            smallest = min(smallest, scale)

    return PageReport(
        page=page.number,
        targets=len(targets),
        mean_scale=round(sum(scales) / len(scales), 3) if scales else 1.0,
        scaled=scaled,
        overflow=overflow,
        original_font=reused,
        min_scale=round(smallest, 3),
        zones=[Zone(bbox=t.bbox, source=t.source, translation=t.translation) for t in targets],
    )


def rebuild_document(
    source: str | Path,
    output: str | Path,
    translator: "Translator | Callable[[str], str]",
    min_scale: float = 0.5,
    source_lang: str = "en",
    target_lang: str = "fr",
    on_page: Callable[[int, int], None] | None = None,
) -> DocumentReport:
    """Translate every page of ``source`` and save the result to ``output``.

    ``on_page(done, total)`` is called after each page (progress reporting).
    """
    source, output = Path(source), Path(output)
    translator = as_translator(translator)
    with pymupdf.open(source) as doc:
        fonts = FontCatalog(doc)
        pages = []
        for page in doc:
            pages.append(rebuild_page(page, translator, min_scale, fonts, source_lang, target_lang))
            if on_page:
                on_page(len(pages), doc.page_count)
        output.parent.mkdir(parents=True, exist_ok=True)
        # insert_htmlbox embeds a font copy per call: subset the fonts and let
        # garbage=4 merge duplicate objects, otherwise the file is ~10x bigger.
        doc.subset_fonts()
        doc.save(output, garbage=4, deflate=True)
    return DocumentReport(file=source.name, pages=pages)


def side_by_side(original: str | Path, rebuilt: str | Path, output: str | Path) -> None:
    """Write a PDF showing each original page (left) next to its rebuilt version (right)."""
    with pymupdf.open(original) as left, pymupdf.open(rebuilt) as right, pymupdf.open() as out:
        for i in range(left.page_count):
            r = left[i].rect
            page = out.new_page(width=2 * r.width + 20, height=r.height)
            page.show_pdf_page(pymupdf.Rect(0, 0, r.width, r.height), left, i)
            page.show_pdf_page(pymupdf.Rect(r.width + 20, 0, 2 * r.width + 20, r.height), right, i)
            page.draw_line((r.width + 10, 0), (r.width + 10, r.height), color=(0.6, 0.6, 0.6))
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        out.save(output, garbage=4, deflate=True)
