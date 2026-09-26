"""PDF core: text block extraction and reconstruction."""

from .extract import extract_blocks, extract_document
from .language import DetectedLanguage, detect_language
from .models import TextBlock
from .rebuild import DocumentReport, rebuild_document, rebuild_page, side_by_side

__all__ = [
    "DetectedLanguage",
    "DocumentReport",
    "TextBlock",
    "detect_language",
    "extract_blocks",
    "extract_document",
    "rebuild_document",
    "rebuild_page",
    "side_by_side",
]
