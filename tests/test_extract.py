"""Tests for text block extraction."""

from pathlib import Path

import pymupdf
import pytest

from docagent.pdf import TextBlock, extract_blocks, extract_document
from docagent.pdf.extract import is_numeric, join_lines
from docagent.pdf.text import has_words, is_symbol

SAMPLES = Path(__file__).parent.parent / "data" / "samples"


@pytest.fixture
def simple_page() -> pymupdf.Page:
    """A one-page PDF with a heading, a paragraph with bold words and a number."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_htmlbox(pymupdf.Rect(60, 60, 535, 100), "<h1>Le climat en 2024</h1>")
    page.insert_htmlbox(
        pymupdf.Rect(60, 120, 535, 200),
        "<p>L'année 2024 a été la <b>plus chaude jamais mesurée</b> dans le monde.</p>",
    )
    page.insert_text((60, 300), "1,52", fontsize=11)
    return page


# --- unit tests on text helpers ---------------------------------------------


def test_join_lines_removes_hyphenation():
    assert join_lines(["des éco-", "systèmes fragiles"]) == "des écosystèmes fragiles"


def test_join_lines_keeps_meaningful_hyphens():
    assert join_lines(["période 1850-", "1900"]) == "période 1850-1900"
    assert join_lines(["vent du Nord-", "Est"]) == "vent du Nord-Est"


def test_join_lines_normalises_spaces():
    assert join_lines(["  a  b ", "", "c"]) == "a b c"


@pytest.mark.parametrize("text", ["2,0", "1850-1900", "+1,52 °C", "2.0 ºC", "ºC", "(12.5%)", "414"])
def test_is_numeric_true(text):
    assert is_numeric(text)


@pytest.mark.parametrize("text", ["Solar PV", "2024 record", "", "CO2"])
def test_is_numeric_false(text):
    assert not is_numeric(text)


# --- extract_blocks on a synthetic page -------------------------------------


def test_extract_blocks_structure(simple_page):
    blocks = extract_blocks(simple_page)
    assert [b.text for b in blocks] == [
        "Le climat en 2024",
        "L'année 2024 a été la plus chaude jamais mesurée dans le monde.",
        "1,52",
    ]
    assert all(isinstance(b, TextBlock) for b in blocks)


def test_extract_blocks_bbox_inside_page(simple_page):
    for block in extract_blocks(simple_page):
        x0, y0, x1, y1 = block.bbox
        assert 0 <= x0 < x1 <= 595
        assert 0 <= y0 < y1 <= 842


def test_extract_blocks_style_and_flags(simple_page):
    heading, paragraph, number = extract_blocks(simple_page)
    assert heading.bold and heading.size > paragraph.size
    assert not paragraph.bold  # bold words are a minority of the characters
    assert number.numeric and not number.translatable
    assert paragraph.translatable


# --- extraction on the real samples -----------------------------------------

SAMPLE_FILES = sorted(SAMPLES.glob("*.pdf"))


@pytest.mark.skipif(not SAMPLE_FILES, reason="no sample PDFs in data/samples")
@pytest.mark.parametrize("path", SAMPLE_FILES, ids=lambda p: p.stem)
def test_samples_extraction_not_empty(path):
    blocks = extract_document(path)
    with pymupdf.open(path) as doc:
        pages_with_blocks = {b.page for b in blocks}
        assert len(pages_with_blocks) == doc.page_count
    assert sum(b.translatable for b in blocks) > 0


@pytest.mark.skipif(not (SAMPLES / "hcc_2025_2col.pdf").exists(), reason="HCC sample missing")
def test_ligatures_are_expanded():
    text = " ".join(b.text for b in extract_document(SAMPLES / "hcc_2025_2col.pdf"))
    assert "ﬃ" not in text and "ﬁ" not in text
    assert "affichent" in text


# --- layout decisions on hand-made blocks -----------------------------------


def _block(lines: list[str], boxes: list[tuple]) -> TextBlock:
    xs0, ys0, xs1, ys1 = zip(*boxes, strict=True)
    return TextBlock(
        page=0,
        index=0,
        bbox=(min(xs0), min(ys0), max(xs1), max(ys1)),
        text=join_lines(lines),
        lines=lines,
        line_bboxes=boxes,
        font="Helvetica",
        size=10,
        color=0,
        bold=False,
        italic=False,
        horizontal=True,
        numeric=is_numeric(join_lines(lines)),
    )


def test_paragraph_is_rewritten_as_a_whole():
    block = _block(
        ["first line of text", "second line"], [(50, 100, 300, 112), (50, 113, 200, 125)]
    )
    assert not block.rewrite_by_line


def test_table_row_is_rewritten_by_line():
    block = _block(
        ["Solar", "35", "47"], [(50, 100, 90, 110), (200, 100, 215, 110), (250, 100, 265, 110)]
    )
    assert block.is_table_row and block.rewrite_by_line


def test_label_grouped_with_axis_tick_is_rewritten_by_line():
    # IPCC figure: PyMuPDF groups the label "observed" with the tick "1.0" far away.
    block = _block(["observed", "1.0"], [(231, 324, 260, 334), (66, 315, 76, 325)])
    assert not block.numeric and block.rewrite_by_line


def test_scattered_labels_are_rewritten_by_line():
    # HCC infographic: labels next to bars of different lengths, no common edge.
    boxes = [(408, 217, 502, 227), (335, 229, 440, 239), (331, 240, 430, 250), (295, 251, 426, 262)]
    block = _block(["Voitures", "Poids lourds", "Utilitaires", "Aérien domestique"], boxes)
    assert block.alignment is None and block.rewrite_by_line


def test_centered_paragraph_is_detected():
    boxes = [(100, 100, 200, 110), (80, 111, 220, 121), (120, 122, 180, 132)]
    block = _block(["pour limiter", "le réchauffement et faire", "face aux"], boxes)
    assert block.alignment == "center" and not block.rewrite_by_line


def test_bullets_and_math_symbols():
    assert is_symbol("\x07") and is_symbol("•") and is_symbol("\uf0a7")
    assert not is_symbol("a") and not is_symbol("-")
    assert has_words("τair,t") and not has_words("b") and not has_words("x1")


@pytest.mark.skipif(not (SAMPLES / "hcc_grand_public.pdf").exists(), reason="HCC sample missing")
def test_bullets_are_cut_out_of_the_text():
    blocks = extract_document(SAMPLES / "hcc_grand_public.pdf")
    assert all("\x07" not in b.text for b in blocks)
    assert sum(len(b.bullet_bboxes) for b in blocks) > 0


@pytest.mark.skipif(not (SAMPLES / "hcc_grand_public.pdf").exists(), reason="HCC sample missing")
def test_inline_styles_become_markup():
    blocks = extract_document(SAMPLES / "hcc_grand_public.pdf")
    block = next(b for b in blocks if b.text.startswith("Au cours de la dernière"))
    assert block.markup.startswith("<s1>Au cours de la dernière décennie,</s1> le réchauffement")
    assert block.styles["s1"].color == 0xB92B20  # red run inside a dark blue block
    assert "<" not in block.text  # plain text stays tag-free


@pytest.mark.skipif(not (SAMPLES / "hcc_2025_2col.pdf").exists(), reason="HCC sample missing")
def test_footnote_marker_is_superscript():
    blocks = extract_document(SAMPLES / "hcc_2025_2col.pdf")
    block = next(b for b in blocks if b.text.startswith("Les Français affichent"))
    assert "climatique<s1>I</s1>," in block.markup
    assert block.styles["s1"].position == "sup"
