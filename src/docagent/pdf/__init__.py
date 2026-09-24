"""PDF core: text block extraction and reconstruction."""

from .extract import extract_blocks, extract_document
from .models import TextBlock
from .rebuild import DocumentReport, rebuild_document, rebuild_page, side_by_side

__all__ = [
    "DocumentReport",
    "TextBlock",
    "extract_blocks",
    "extract_document",
    "rebuild_document",
    "rebuild_page",
    "side_by_side",
]
