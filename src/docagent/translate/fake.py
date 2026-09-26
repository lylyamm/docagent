"""Fake translation, used to test PDF reconstruction without any LLM.

The output is visibly different from the source (UPPERCASE) and longer, because
real translations change length (French is typically 15-25% longer than English).
Reconstruction must cope with that expansion.
"""

import math
import re

from .base import Translator

# Inline style tags ("<s1>…</s1>") must survive translation untouched.
_TAG_RE = re.compile(r"(</?s\d+>)")


def fake_translate(text: str, expansion: float = 0.2, uppercase: bool = True) -> str:
    """Return ``text`` in uppercase, lengthened by ``expansion`` (0.2 = +20%).

    Uppercase glyphs are ~20-30% wider than lowercase ones, so the default is a
    harsher test than a real translation; ``uppercase=False`` keeps the case.

    The extra length is the text itself, repeated and cut at the target length.
    With ``uppercase`` the padding stays in lowercase so it can't be mistaken
    for a duplicated sentence: "LE CLIMAT EN 2024 le c". (Brackets were tried
    first, but embedded font subsets rarely contain "[" and "]", which forced a
    fallback font.) Style tags are kept as they are.
    """
    if expansion < 0:
        raise ValueError("expansion must be >= 0")
    parts = _TAG_RE.split(text)
    converted = "".join(
        part if _TAG_RE.fullmatch(part) else (part.upper() if uppercase else part) for part in parts
    )
    plain = _TAG_RE.sub("", text).strip()
    if not plain or expansion == 0:
        return converted

    extra = math.ceil(len(plain) * expansion) - 1  # minus the separating space
    if extra < 1:
        return converted
    repeats = math.ceil(extra / len(plain)) + 1
    padding = " ".join([plain] * repeats)[:extra].rstrip()
    if uppercase:
        padding = padding.lower()
    return f"{converted.rstrip()} {padding}"


class FakeTranslator(Translator):
    """Translator wrapper around :func:`fake_translate` (no network, no cost)."""

    name = "fake"

    def __init__(self, expansion: float = 0.2, uppercase: bool = True) -> None:
        self.expansion = expansion
        self.uppercase = uppercase

    def translate_blocks(self, texts: list[str], source: str, target: str) -> list[str]:
        return [fake_translate(t, self.expansion, self.uppercase) for t in texts]
