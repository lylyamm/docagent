"""On-disk cache of translations, so re-running a rebuild costs no API calls.

Keys hash the model, the language pair and the source text: changing the
model or the prompt version gives new entries instead of stale ones.
"""

import hashlib
import json
from pathlib import Path

from .base import Translator


class CachedTranslator(Translator):
    def __init__(self, inner: Translator, path: str | Path, version: str = "") -> None:
        self.inner = inner
        self.name = f"cached({inner.name})"
        self.path = Path(path)
        self.version = version
        self.hits = self.misses = 0
        try:
            self._data: dict[str, str] = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            self._data = {}

    def _key(self, text: str, source: str, target: str) -> str:
        raw = f"{self.version}|{self.inner.name}|{source}|{target}|{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def translate_blocks(self, texts: list[str], source: str, target: str) -> list[str]:
        keys = [self._key(t, source, target) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self._data]
        self.hits += len(texts) - len(missing)
        self.misses += len(missing)
        if missing:
            fresh = self.inner.translate_blocks([texts[i] for i in missing], source, target)
            for i, translation in zip(missing, fresh, strict=True):
                self._data[keys[i]] = translation
            self._save()
        return [self._data[k] for k in keys]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=False, indent=0), encoding="utf-8")
        tmp.replace(self.path)  # atomic: an interrupted run never leaves a broken cache
