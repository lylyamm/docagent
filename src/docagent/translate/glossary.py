"""Optional glossary: terms whose translation is imposed.

A JSON file mapping source terms to target terms, e.g.
``{"carbon budget": "budget carbone", "net zero": "neutralité carbone"}``.
Only the entries that appear in the texts being translated are sent to the
model, so a large glossary doesn't inflate every prompt. Terms match whole
words only ("likely" not in "unlikely"); acronyms ("SPM", "TS") are matched
case-sensitively so "ts" in "results" doesn't count.
"""

import json
import re
from pathlib import Path


class Glossary:
    def __init__(self, terms: dict[str, str] | None = None) -> None:
        self.terms = terms or {}

    @classmethod
    def load(cls, path: str | Path | None) -> "Glossary":
        if not path:
            return cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in data.items()
        ):
            raise ValueError(f"{path}: expected a JSON object of string -> string")
        return cls(data)

    def relevant(self, texts: list[str]) -> dict[str, str]:
        """Entries whose source term occurs as whole words in the texts."""
        joined = " ".join(texts)
        return {src: dst for src, dst in self.terms.items() if _pattern(src).search(joined)}


def _pattern(term: str) -> re.Pattern:
    flags = 0 if term.isupper() else re.IGNORECASE
    return re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", flags)
