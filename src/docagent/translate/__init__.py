"""Translation backends: a fake one for layout tests, any OpenAI-compatible LLM for real ones."""

from .base import FunctionTranslator, Translator, as_translator
from .cache import CachedTranslator
from .factory import build_translator, llm_stats
from .fake import FakeTranslator, fake_translate
from .glossary import Glossary
from .llm import ChatTranslator

__all__ = [
    "CachedTranslator",
    "ChatTranslator",
    "FakeTranslator",
    "FunctionTranslator",
    "Glossary",
    "Translator",
    "as_translator",
    "build_translator",
    "fake_translate",
    "llm_stats",
]
