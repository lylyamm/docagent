"""Translator interface shared by every backend (fake, Mistral, later others)."""

from abc import ABC, abstractmethod
from collections.abc import Callable


class Translator(ABC):
    """Translates a batch of texts (usually all the blocks of one page).

    Texts are markup: inline style tags such as ``<s1>…</s1>`` must come back
    unchanged around the corresponding words. Translating a page at once gives
    the model context (a heading and the paragraph below it).
    """

    name: str = "translator"

    @abstractmethod
    def translate_blocks(self, texts: list[str], source: str, target: str) -> list[str]:
        """Return one translation per input text, in the same order."""


class FunctionTranslator(Translator):
    """Adapter turning a ``str -> str`` function into a Translator."""

    def __init__(self, func: Callable[[str], str], name: str = "function") -> None:
        self.func = func
        self.name = name

    def translate_blocks(self, texts: list[str], source: str, target: str) -> list[str]:
        return [self.func(text) for text in texts]


def as_translator(translator: "Translator | Callable[[str], str]") -> Translator:
    """Accept either a Translator or a plain function (handy in tests)."""
    if isinstance(translator, Translator):
        return translator
    return FunctionTranslator(translator)
