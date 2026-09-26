"""Tests for PDF language detection (metadata first, text otherwise)."""

from pathlib import Path

import pymupdf
import pytest

from docagent.pdf.language import detect_language

SAMPLES = Path(__file__).parent.parent / "data" / "samples"
ENGLISH = "Global surface temperature was higher in the last decade than in any period since 1850"
FRENCH = "La température à la surface du globe a été plus élevée que jamais au cours de la décennie"


def make_pdf(text: str, lang: str | None = None) -> pymupdf.Document:
    doc = pymupdf.open()
    doc.new_page().insert_htmlbox(pymupdf.Rect(50, 50, 550, 200), f"<p>{text}</p>")
    if lang:
        doc.xref_set_key(doc.pdf_catalog(), "Lang", f"({lang})")
    return doc


def test_text_is_used_when_there_is_no_metadata():
    detected = detect_language(make_pdf(ENGLISH))
    assert (detected.language, detected.method) == ("en", "text")
    assert detected.confidence > 0.95


def test_metadata_confirmed_by_the_text():
    detected = detect_language(make_pdf(FRENCH, "fr-FR"))
    assert (detected.language, detected.method, detected.confidence) == ("fr", "metadata", 1.0)
    assert detected.metadata == "fr-FR"


def test_metadata_alone_when_there_is_too_little_text():
    detected = detect_language(make_pdf("12 34", "de"))
    assert (detected.language, detected.method, detected.confidence) == ("de", "metadata", 0.9)


def test_clearly_contradicted_metadata_is_overridden():
    # A French translation that kept the English original's /Lang.
    detected = detect_language(make_pdf(FRENCH, "en-GB"))
    assert (detected.language, detected.method) == ("fr", "text")


def test_unknown_language():
    detected = detect_language(make_pdf("2024 1850 42"))
    assert (detected.language, detected.method) == (None, "none")


@pytest.mark.parametrize(
    ("name", "expected"),
    [("giec_spm_en.pdf", "en"), ("giec_spm_fr.pdf", "fr"), ("hcc_grand_public.pdf", "fr")],
)
def test_samples(name, expected):
    if not (SAMPLES / name).exists():
        pytest.skip(f"{name} missing")
    assert detect_language(SAMPLES / name).language == expected
