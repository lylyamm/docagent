"""Data models for the PDF pipeline."""

from pydantic import BaseModel, Field

from .text import has_words, is_math_font, is_numeric

BBox = tuple[float, float, float, float]


class InlineStyle(BaseModel):
    """Style of a run of text that differs from its block's dominant style."""

    font: str
    color: int
    bold: bool
    italic: bool
    position: str | None = Field(default=None, description='"sup", "sub" or None')


class TextBlock(BaseModel):
    """A block of text extracted from a PDF page, ready to be translated.

    A block usually maps to a paragraph, a heading, a table cell or a chart label.
    ``text`` is cleaned for translation (lines joined, hyphenation and ligatures
    resolved); ``lines`` keeps the raw text of each visual line.
    """

    page: int = Field(ge=0, description="0-based page index")
    index: int = Field(ge=0, description="position of the block in PyMuPDF reading order")
    bbox: BBox = Field(description="(x0, y0, x1, y1) in points, origin at the top-left corner")
    text: str
    lines: list[str]
    markup: str = Field(
        default="",
        description='text with inline style tags: "<s1>Au cours de…,</s1> le réchauffement"',
    )
    line_markups: list[str] = Field(default_factory=list, description="markup of each line")
    styles: dict[str, InlineStyle] = Field(default_factory=dict, description="tag -> style")
    line_bboxes: list[BBox] = Field(description="bbox of each entry of `lines`")
    bullet_bboxes: list[BBox] = Field(
        default_factory=list, description="bullets cut out of the lines (kept as is)"
    )

    # Dominant style, weighted by number of characters.
    font: str
    size: float
    color: int = Field(description="sRGB color as an integer, e.g. 0x000000 for black")
    bold: bool
    italic: bool

    # Hints for later stages.
    horizontal: bool = Field(description="False for rotated text (axis labels, margins)")
    numeric: bool = Field(description="True if the text holds only numbers and symbols")

    @property
    def is_table_row(self) -> bool:
        """True when some lines sit side by side (table cells, chart legends).

        In a paragraph, lines are stacked; here at least two lines share the same
        vertical band, so each line must be rewritten in its own bbox.
        """
        boxes = self.line_bboxes
        for i, a in enumerate(boxes):
            for b in boxes[i + 1 :]:
                overlap = min(a[3], b[3]) - max(a[1], b[1])
                if overlap > 0.5 * min(a[3] - a[1], b[3] - b[1]):
                    return True
        return False

    @property
    def alignment(self) -> str | None:
        """Paragraph alignment guessed from line edges, or None if lines are scattered.

        The first line may be indented and the last line may be short, so they
        are ignored where relevant. One-line blocks are "left".
        """
        boxes = self.line_bboxes
        if len(boxes) < 2:
            return "left"
        tol = max(self.size * 0.4, 1.5)

        def same(values: list[float]) -> bool:
            return max(values) - min(values) < tol

        lefts = [b[0] for b in boxes]
        rights = [b[2] for b in boxes]
        centers = [(b[0] + b[2]) / 2 for b in boxes]
        body_lefts = lefts[1:] if len(boxes) > 2 else lefts
        if same(body_lefts) and len(boxes) > 2 and same(rights[:-1]):
            return "justify"
        if same(body_lefts):
            return "left"
        if same(rights):
            return "right"
        if same(centers):
            return "center"
        return None

    @property
    def rewrite_by_line(self) -> bool:
        """Rewrite each line in its own bbox instead of the block as one paragraph.

        Needed for table rows, for blocks mixing text with numbers (an axis tick
        grouped with a chart label) and for lines that do not form an aligned
        paragraph (chart and infographic labels grouped by PyMuPDF).
        """
        if len(self.line_bboxes) < 2:
            return False
        if self.is_table_row or any(is_numeric(line) for line in self.lines):
            return True
        return self.alignment is None

    @property
    def translatable(self) -> bool:
        """Whether the block should be sent to the translator."""
        return (
            self.horizontal
            and not self.numeric
            and not is_math_font(self.font)
            and has_words(self.text)
        )
