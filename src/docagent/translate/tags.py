"""Fix the style tags of a translation without calling the LLM again.

Two ways small models break the ``<s1>…</s1>`` tags of a block:

1. They add tags of their own, typically around glossary terms ("le <s1>principal
   facteur</s1>") or acronyms. ``prune_tags`` keeps the runs that match the
   source's and turns the others back into plain text.
2. They drop tags in long paragraphs. ``restore_tags`` finds the runs again:

- footnote calls and sub/superscripts ("temperature<s1>8</s1>", "CO<s2>2</s2>"):
  a short number glued to the previous word, which the translation keeps glued
  ("globe8", "CO2");
- italic calibrated language ("<s1>likely</s1>"): the glossary says what it
  becomes ("probable"), possibly inflected ("très probablement", "improbables");
- words kept as is (names, acronyms, numbers);
- a bold lead-in at the start of a caption ("<s1>Panel (a) Observed warming</s1>
  (increase…"): it ends at the same punctuation in the translation.

Every run must be placed, in order, or nothing is: a half-restored block would
put a style on the wrong words, which is worse than no style.
"""

import re

_RUN_RE = re.compile(r"<(s\d+)>(.*?)</\1>", re.S)
_TAG_RE = re.compile(r"</?s\d+>")
# A footnote call or an exponent: "8", "11", "–1", "2".
_MARKER_RE = re.compile(r"[\d–\-−+.,]{1,5}")


def prune_tags(source: str, translation: str, terms: dict[str, str]) -> str | None:
    """``translation`` keeping only the runs that match the source's, or None.

    Source runs are matched in order with translation runs of the same tag
    whose content fits (same footnote number, glossary translation, same
    position for a lead-in). Unmatched translation runs lose their tags.
    """
    if not _simple(source) or not _simple(translation):
        return None
    source_runs = list(_RUN_RE.finditer(source))
    runs = list(_RUN_RE.finditer(translation))
    lowered = {k.lower(): v for k, v in terms.items()}
    kept: set[int] = set()
    k = 0
    for run in source_runs:
        while k < len(runs) and not (
            runs[k].group(1) == run.group(1)
            and _same_run(source, run, translation, runs[k], lowered)
        ):
            k += 1
        if k == len(runs):
            return None
        kept.add(k)
        k += 1
    parts, pos = [], 0
    for i, run in enumerate(runs):
        parts += [translation[pos : run.start()], run.group(0) if i in kept else run.group(2)]
        pos = run.end()
    return "".join([*parts, translation[pos:]])


def _simple(markup: str) -> bool:
    """True if every tag belongs to a simple, non-nested ``<sN>…</sN>`` pair."""
    return _TAG_RE.search(_RUN_RE.sub("", markup)) is None


def _same_run(
    source: str, run: re.Match, translation: str, candidate: re.Match, terms: dict[str, str]
) -> bool:
    """Does the translation run ``candidate`` render the source run ``run``?"""
    text, other = run.group(2).strip(), candidate.group(2).strip()
    before = _TAG_RE.sub("", source[: run.start()])
    if _MARKER_RE.fullmatch(text):
        return other == text
    if not before.strip():  # a lead-in: it opens the translation too
        return not _TAG_RE.sub("", translation[: candidate.start()]).strip()
    return _place(other, 0, text, "x", "", terms) is not None


def restore_tags(source: str, translation: str, terms: dict[str, str]) -> str | None:
    """``translation`` with the source's style runs tagged again, or None."""
    runs = list(_RUN_RE.finditer(source))
    if not runs or not _simple(source):
        return None  # no runs, or tags that are not simple balanced pairs
    plain = _TAG_RE.sub("", translation)
    lowered_terms = {k.lower(): v for k, v in terms.items()}
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for run in runs:
        text = run.group(2)
        before = _TAG_RE.sub("", source[: run.start()])
        after = _TAG_RE.sub("", source[run.end() :])
        span = _place(plain, cursor, text, before, after, lowered_terms)
        if span is None:
            return None
        spans.append((span[0], span[1], run.group(1)))
        cursor = span[1]
    for start, end, name in reversed(spans):
        plain = f"{plain[:start]}<{name}>{plain[start:end]}</{name}>{plain[end:]}"
    return plain


def _place(
    plain: str, cursor: int, text: str, before: str, after: str, terms: dict[str, str]
) -> tuple[int, int] | None:
    """Where the run ``text`` is in the translation, searching from ``cursor``."""
    stripped = text.strip()
    if not stripped:
        return None
    if not before.strip() and not after.strip():  # the whole block is one run
        return 0, len(plain)
    if _MARKER_RE.fullmatch(stripped) and before and not before[-1].isspace():
        return _place_marker(plain, cursor, stripped, before)
    candidates = [c for c in (terms.get(stripped.lower()), stripped) if c]
    for pattern in [rf"{re.escape(c)}(?!\w)" for c in candidates] + [
        # Same word with another ending: "probable" -> "probablement", "observed" -> "observé".
        rf"{re.escape(c[:-2])}\w*"
        for c in candidates
        if len(c.split()[-1]) >= 5
    ]:
        match = re.compile(rf"(?<!\w){pattern}", re.I).search(plain, cursor)
        if match:
            return match.span()
    if not before.strip():
        return _place_lead_in(plain, stripped, after)
    return None


def _place_marker(plain: str, cursor: int, marker: str, before: str) -> tuple[int, int] | None:
    """A number glued to the text before it: "2010–2019" + "11", "globe" + "8"."""
    anchor = before.split()[-1] if before.split() else ""
    # The anchor survives translation when it is a number or a formula ("2019", "CO").
    if anchor and not anchor.isalpha():
        index = plain.find(anchor + marker, cursor)
        if index >= 0:
            start = index + len(anchor)
            return start, start + len(marker)
    # Otherwise: the marker glued to any word, and not part of a longer number.
    match = re.compile(rf"(?<=[^\s\d]){re.escape(marker)}(?![\d])").search(plain, cursor)
    return match.span() if match else None


def _place_lead_in(plain: str, text: str, after: str) -> tuple[int, int] | None:
    """A run opening the block ("Panel (a) …") ends at the same punctuation."""
    if text[-1] in ".:":
        mark, count = text[-1], text.count(text[-1])
        end_offset = 1  # the punctuation belongs to the run
    elif after.lstrip()[:1] in ("(", ":", ",", ";", "."):
        mark, count = after.lstrip()[0], text.count(after.lstrip()[0]) + 1
        end_offset = 0
    else:
        return None
    index = -1
    for _ in range(count):
        index = plain.find(mark, index + 1)
        if index < 0:
            return None
    end = index + end_offset
    while end > 0 and plain[end - 1].isspace():
        end -= 1
    return (0, end) if end > 0 else None
