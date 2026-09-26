"""Detect the language of a PDF: its metadata first, its text otherwise.

- Metadata: the catalog's ``/Lang`` entry ("en-GB"), set by the author's tool.
  Free and instant, but often missing (none of the 7 sample PDFs has one) and
  sometimes wrong (a translation that kept the original's settings).
- Text: a few thousand characters of the first pages, classified by py3langid
  (a small n-gram model, deterministic), restricted to the supported languages.

The metadata is trusted unless the text clearly says otherwise.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

import pymupdf
from pydantic import BaseModel

SUPPORTED = ("en", "fr", "de", "es", "it", "pt")
SAMPLE_PAGES = 3
SAMPLE_CHARS = 5000
# Below this many letters the text says too little to classify.
MIN_LETTERS = 20
# The text overrides the metadata only when it is this sure.
OVERRIDE_CONFIDENCE = 0.95
# Confidence reported for metadata that the text can't confirm (too little text).
METADATA_ONLY = 0.9


class DetectedLanguage(BaseModel):
    language: str | None  # ISO 639-1 code, None if unknown
    confidence: float  # 0..1 (1.0 when metadata and text agree)
    method: Literal["metadata", "text", "none"]
    metadata: str | None = None  # the raw /Lang value, for transparency


@lru_cache(maxsize=1)
def _identifier():
    """The py3langid model, loaded once and limited to the supported languages."""
    from py3langid.langid import MODEL_FILE, LanguageIdentifier

    identifier = LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)
    identifier.set_languages(list(SUPPORTED))
    return identifier


def metadata_language(doc: pymupdf.Document) -> str | None:
    """Language code from the catalog's /Lang ("en-GB" -> "en"), if supported."""
    kind, value = doc.xref_get_key(doc.pdf_catalog(), "Lang")
    if kind != "string" or not value:
        return None
    code = value.strip().lower().replace("_", "-").split("-")[0]
    return code if code in SUPPORTED else None


def text_sample(doc: pymupdf.Document) -> str:
    """Text of the first pages, up to ``SAMPLE_CHARS`` characters."""
    parts: list[str] = []
    size = 0
    for page in doc.pages(0, min(SAMPLE_PAGES, doc.page_count)):
        text = " ".join(page.get_text().split())
        parts.append(text)
        size += len(text)
        if size >= SAMPLE_CHARS:
            break
    return " ".join(parts)[:SAMPLE_CHARS]


def classify_text(text: str) -> tuple[str | None, float]:
    """(language, confidence) of a text, or (None, 0.0) if it has too few letters."""
    if sum(c.isalpha() for c in text) < MIN_LETTERS:
        return None, 0.0
    language, confidence = _identifier().classify(text)
    return language, round(float(confidence), 3)


def detect_language(source: str | Path | pymupdf.Document) -> DetectedLanguage:
    """Language of a PDF (a path or an open document)."""
    if isinstance(source, pymupdf.Document):
        return _detect(source)
    with pymupdf.open(source) as doc:
        return _detect(doc)


def _detect(doc: pymupdf.Document) -> DetectedLanguage:
    meta = metadata_language(doc)
    kind, raw = doc.xref_get_key(doc.pdf_catalog(), "Lang")
    raw_meta = raw if kind == "string" else None
    language, confidence = classify_text(text_sample(doc))

    if meta and (language in (None, meta) or confidence < OVERRIDE_CONFIDENCE):
        if language == meta:
            score = 1.0  # both sources agree
        elif language is None:
            score = METADATA_ONLY  # nothing to check it against
        else:
            score = round(1 - confidence, 3)  # the text leaned (weakly) the other way
        return DetectedLanguage(
            language=meta, confidence=score, method="metadata", metadata=raw_meta
        )
    if language:
        return DetectedLanguage(
            language=language, confidence=confidence, method="text", metadata=raw_meta
        )
    return DetectedLanguage(language=None, confidence=0.0, method="none", metadata=raw_meta)
