"""Translation backends. Phase 1 only has a fake translator; the LLM comes in phase 2."""

from .fake import fake_translate

__all__ = ["fake_translate"]
