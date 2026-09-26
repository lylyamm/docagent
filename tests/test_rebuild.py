"""Tests for fake translation and PDF reconstruction."""

from pathlib import Path

import pymupdf
import pytest

from docagent.pdf import extract_document
from docagent.pdf.rebuild import rebuild_document, side_by_side
from docagent.pdf.verify import verify_rebuild
from docagent.translate import fake_translate

SAMPLES = Path(__file__).parent.parent / "data" / "samples"
SAMPLE_FILES = sorted(SAMPLES.glob("*.pdf"))

# --- fake_translate ----------------------------------------------------------


def test_fake_translate_is_uppercase_and_longer():
    source = "L'année 2024 a été la plus chaude jamais mesurée."
    result = fake_translate(source)
    assert result.startswith(source.upper())
    assert len(result) >= 1.19 * len(source)
    # The padding is lowercase, so it can't be mistaken for duplicated text.
    padding = result[len(source) :].strip()
    assert padding and padding == padding.lower()


def test_fake_translate_keeps_style_tags():
    result = fake_translate("<s1>Au cours de la décennie,</s1> le réchauffement")
    assert result.startswith("<s1>AU COURS DE LA DÉCENNIE,</s1> LE RÉCHAUFFEMENT")
    assert result.count("<s1>") == 1 and result.count("</s1>") == 1


def test_fake_translate_without_expansion():
    assert fake_translate("Solar PV", expansion=0) == "SOLAR PV"


def test_fake_translate_rejects_negative_expansion():
    with pytest.raises(ValueError):
        fake_translate("text", expansion=-0.1)


# --- rebuild on a synthetic page --------------------------------------------


@pytest.fixture
def source_pdf(tmp_path) -> Path:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.draw_rect(pymupdf.Rect(60, 400, 300, 500), color=(1, 0, 0), fill=(1, 0.9, 0.9))
    page.insert_htmlbox(pymupdf.Rect(60, 60, 535, 100), "<h1>Le climat en 2024</h1>")
    page.insert_htmlbox(
        pymupdf.Rect(60, 120, 535, 200),
        "<p>L'année 2024 a été la plus chaude jamais mesurée dans le monde.</p>",
    )
    page.insert_text((60, 300), "1,52", fontsize=11)
    path = tmp_path / "source.pdf"
    doc.save(path)
    return path


def test_rebuild_replaces_text_in_place(source_pdf, tmp_path):
    output = tmp_path / "rebuilt.pdf"
    report = rebuild_document(source_pdf, output, fake_translate)

    assert report.targets == 2  # heading + paragraph, not the number
    assert report.overflow == 0
    text = " ".join(pymupdf.open(output)[0].get_text().split())  # lines may wrap differently
    assert "LE CLIMAT EN 2024" in text
    assert "Le climat en 2024" not in text.replace("le climat en 2024", "")
    assert "1,52" in text  # numbers are left untouched


def test_rebuild_keeps_graphics(source_pdf, tmp_path):
    output = tmp_path / "rebuilt.pdf"
    rebuild_document(source_pdf, output, fake_translate)
    before = len(pymupdf.open(source_pdf)[0].get_drawings())
    after = len(pymupdf.open(output)[0].get_drawings())
    assert after == before


def test_rebuilt_blocks_stay_in_original_area(source_pdf, tmp_path):
    output = tmp_path / "rebuilt.pdf"
    rebuild_document(source_pdf, output, fake_translate)
    original = {b.index: b for b in extract_document(source_pdf)}
    for block in extract_document(output):
        if block.text.startswith("LE CLIMAT"):
            heading = original[0]
            assert abs(block.bbox[0] - heading.bbox[0]) < 5
            assert abs(block.bbox[1] - heading.bbox[1]) < 10


# --- rebuild on the real samples --------------------------------------------


@pytest.mark.skipif(not SAMPLE_FILES, reason="no sample PDFs in data/samples")
@pytest.mark.parametrize("path", SAMPLE_FILES, ids=lambda p: p.stem)
def test_samples_rebuild(path, tmp_path):
    output = tmp_path / f"rebuilt_{path.name}"
    report = rebuild_document(path, output, fake_translate)

    with pymupdf.open(path) as src, pymupdf.open(output) as out:
        assert out.page_count == src.page_count  # output opens, same length
        for p_src, p_out in zip(src, out, strict=True):
            assert len(p_out.get_images()) == len(p_src.get_images())
    assert report.targets > 0
    check = verify_rebuild(path, output, report)
    assert check.drawings_after == check.drawings_before  # charts, table fills intact
    assert check.residual_source == 0  # no source text left under the translation
    assert check.numbers_kept == check.numbers_total, check.lost_numbers
    if path.stem.startswith("arxiv"):
        # Known limitation: drop cap and inline LaTeX math (see DECISIONS.md).
        assert check.overlaps_after <= 1.2 * check.overlaps_before
    else:
        assert check.overlaps_after <= check.overlaps_before  # no text drawn over text
    # A handful of overflows is tolerated (dense LaTeX nomenclature), not more.
    assert report.overflow <= 0.05 * report.targets


@pytest.mark.skipif(not SAMPLE_FILES, reason="no sample PDFs in data/samples")
def test_side_by_side(tmp_path):
    source = SAMPLE_FILES[0]
    rebuilt = tmp_path / "rebuilt.pdf"
    rebuild_document(source, rebuilt, fake_translate)
    compare = tmp_path / "compare.pdf"
    side_by_side(source, rebuilt, compare)
    with pymupdf.open(source) as src, pymupdf.open(compare) as cmp:
        assert cmp.page_count == src.page_count
        assert cmp[0].rect.width > 2 * src[0].rect.width


def test_fake_translate_can_keep_case():
    assert fake_translate("Solar PV", expansion=0, uppercase=False) == "Solar PV"


@pytest.mark.skipif(not (SAMPLES / "hcc_2025_2col.pdf").exists(), reason="HCC sample missing")
def test_original_font_is_reused(tmp_path):
    """Text is rewritten in the document's embedded font when it has the glyphs."""
    output = tmp_path / "rebuilt.pdf"
    report = rebuild_document(SAMPLES / "hcc_2025_2col.pdf", output, fake_translate)
    assert report.original_font > 0.7 * report.targets
    fonts = {f[3] for page in pymupdf.open(output) for f in page.get_fonts()}
    assert any("EuclidCircularA" in name for name in fonts)


@pytest.mark.skipif(not (SAMPLES / "hcc_grand_public.pdf").exists(), reason="HCC sample missing")
def test_inline_colour_survives_rebuild(tmp_path):
    """HCC page 1: "Au cours de la dernière décennie," is red inside a blue block."""
    output = tmp_path / "rebuilt.pdf"
    rebuild_document(SAMPLES / "hcc_grand_public.pdf", output, fake_translate)
    spans = [
        s
        for b in pymupdf.open(output)[0].get_text("dict")["blocks"]
        if b["type"] == 0
        for line in b["lines"]
        for s in line["spans"]
    ]
    red = [s["text"] for s in spans if s["color"] == 0xB92B20]
    assert any("AU COURS DE LA" in t for t in red)
    assert not any("RÉCHAUFFEMENT MONDIAL" in t for t in red)


def test_markup_to_html_drops_unknown_tags():
    from docagent.pdf.models import InlineStyle, TextBlock
    from docagent.pdf.rebuild import Target, markup_to_html

    block = TextBlock(
        page=0,
        index=0,
        bbox=(0, 0, 100, 10),
        text="x",
        lines=["x"],
        line_bboxes=[(0, 0, 100, 10)],
        styles={"s1": InlineStyle(font="F", color=0xFF0000, bold=True, italic=False)},
        font="F",
        size=10,
        color=0,
        bold=False,
        italic=False,
        horizontal=True,
        numeric=False,
    )
    target = Target(
        bbox=(0, 0, 100, 10), source="", translation="<s1>A & B</s1> <s9>c</s9>", block=block
    )
    html, _ = markup_to_html(target, None)
    assert (
        html
        == '<span style="color: #ff0000; font-weight: bold; font-style: normal">A &amp; B</span> c'
    )


def test_markup_to_html_renders_html_tags_added_by_the_translator():
    from docagent.pdf.models import TextBlock
    from docagent.pdf.rebuild import Target, markup_to_html

    block = TextBlock(
        page=0,
        index=0,
        bbox=(0, 0, 100, 10),
        text="x",
        lines=["x"],
        line_bboxes=[(0, 0, 100, 10)],
        font="F",
        size=10,
        color=0,
        bold=False,
        italic=False,
        horizontal=True,
        numeric=False,
    )
    target = Target(
        bbox=(0, 0, 100, 10), source="", translation="XXI<sup>e</sup> siècle, CO<sub>2", block=block
    )
    html, _ = markup_to_html(target, None)
    assert "<sup>" not in html and "&lt;" not in html  # never printed as text
    assert '<span style="vertical-align: super; font-size: 70%">e</span>' in html
    assert html.endswith('CO<span style="vertical-align: sub; font-size: 70%">2</span>')


def _lines_of(path: Path, page: int = 0) -> list[tuple[str, pymupdf.Rect]]:
    with pymupdf.open(path) as doc:
        return [
            ("".join(s["text"] for s in line["spans"]), pymupdf.Rect(line["bbox"]))
            for block in doc[page].get_text("dict")["blocks"]
            if block["type"] == 0
            for line in block["lines"]
        ]


def test_one_line_header_stays_on_one_line(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((60, 40), "Summary for Policymakers", fontsize=9)
    body = "Global surface temperature was higher in 2011-2020 than in 1850-1900. " * 4
    page.insert_textbox(pymupdf.Rect(60, 60, 535, 140), body, fontsize=10)
    source, output = tmp_path / "s.pdf", tmp_path / "o.pdf"
    doc.save(source)

    rebuild_document(source, output, lambda t: "Résumé à l'intention des décideurs")
    header = [(t, r) for t, r in _lines_of(output) if r.y1 < 55]
    assert len(header) == 1  # widened sideways, not wrapped
    assert header[0][0] == "Résumé à l'intention des décideurs"


def test_paragraphs_keep_a_visible_gap(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    text = "The report builds upon the contribution of Working Group I. " * 5
    page.insert_textbox(pymupdf.Rect(60, 60, 535, 110), text, fontsize=10)
    page.insert_textbox(pymupdf.Rect(60, 125, 535, 175), text, fontsize=10)
    source, output = tmp_path / "s.pdf", tmp_path / "o.pdf"
    doc.save(source)
    before = extract_document(source)
    gap = before[1].bbox[1] - before[0].bbox[3]

    rebuild_document(source, output, lambda t: fake_translate(t, expansion=0.3))
    lines = _lines_of(output)
    first = max(r.y1 for _, r in lines if r.y0 < before[1].bbox[1] - 1)
    second = min(r.y0 for _, r in lines if r.y0 >= before[1].bbox[1] - 1)
    assert second - first >= 0.4 * gap  # the translation did not fill the gap


def test_text_that_cannot_fit_is_still_written(tmp_path):
    # A label boxed in on every side (like a code in a hexagon of a map).
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((100, 100), "Med", fontsize=6)
    for rect in [(80, 85, 98, 105), (118, 85, 140, 105), (95, 80, 125, 92), (95, 101, 125, 115)]:
        page.draw_rect(pymupdf.Rect(rect), color=(0, 0, 0), fill=(0.8, 0.5, 0.5))
    source, output = tmp_path / "s.pdf", tmp_path / "o.pdf"
    doc.save(source)

    long_text = "Méditerranée et régions voisines du bassin méditerranéen"
    report = rebuild_document(source, output, lambda t: long_text)
    assert report.overflow == 1  # reported...
    with pymupdf.open(output) as out:
        assert "Méditerranée" in out[0].get_text()  # ...but never silently dropped
    assert verify_rebuild(source, output, report).empty_zones == 0
