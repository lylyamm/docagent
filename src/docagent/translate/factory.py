"""Build the translator described by the settings (used by the CLI and the API)."""

from pathlib import Path

from ..config import Settings, get_settings
from .base import Translator
from .cache import CachedTranslator
from .fake import FakeTranslator
from .glossary import Glossary
from .llm import PROMPT_VERSION, ChatTranslator


def build_translator(
    settings: Settings | None = None,
    *,
    fake: bool = False,
    glossary: Path | None = None,
    use_cache: bool = True,
) -> Translator:
    """The LLM translator (cached), or the fake one for layout tests."""
    settings = settings or get_settings()
    if fake:
        return FakeTranslator()
    translator: Translator = ChatTranslator(
        settings, glossary=Glossary.load(glossary) if glossary else None
    )
    if use_cache and settings.translation_cache:
        translator = CachedTranslator(translator, settings.translation_cache, PROMPT_VERSION)
    return translator


def llm_stats(translator: Translator):
    """Usage stats of the LLM behind a (possibly cached) translator, if any."""
    inner = translator.inner if isinstance(translator, CachedTranslator) else translator
    return getattr(inner, "stats", None)
