"""Answer a question from the reports, with citations (the "G" of RAG).

    question ──► hybrid search ──► 5 passages, numbered [1]..[5]
                                        │
                                        ▼
              LLM: "answer ONLY from these excerpts, cite them as [n],
                    say so when they do not contain the answer"
                                        │
                                        ▼
              JSON {answerable, answer, evidence: [{n, quote}]} ──► checked in code

The model is never trusted blindly. For each fact it must copy, word for word,
the sentence of the excerpt that states it (the "evidence"), and the code checks
that this quote really is in excerpt n:
- every number of the quote must appear in the excerpt;
- its words must appear in the excerpt in the same order, give or take one
  letter (the PDF extraction damaged some words, "cete" for "cette", and the
  model may repair them) and a few skipped words (a dropped parenthesis). A
  swapped word ("bâtiment" for "transports") is rejected, even when the
  excerpt contains "bâtiment" elsewhere.
A quote that fails is dropped with its [n] markers. Then every number of the
answer must appear in an excerpt with a verified quote (or in the question): a
figure the model computed, converted or took from nowhere is not shown. An answer that
fails, or has no verified quote, is replaced by the "not found" message.

This catches invented quotes and figures, not every wrong answer: a true
sentence from the wrong excerpt still passes (see evals/answers/). What was
rejected is kept in the answer (model_answerable, rejected_quotes,
unsupported_numbers) to understand each refusal.
"""

import logging
import re
import time

from pydantic import BaseModel, Field, field_validator

from ..chat import ChatClient
from ..config import Settings, get_settings
from ..pdf.language import classify_text
from .bm25 import tokenize
from .chunking import Passage
from .embeddings import build_embedder
from .index import load_passages, open_store
from .search import HybridSearcher

logger = logging.getLogger(__name__)

PROMPT_VERSION = "ask-v3.4"  # + answer language stated explicitly

SYSTEM_PROMPT = """You answer questions about climate and energy reports (IPCC, IEA, \
French High Council on Climate) using ONLY the numbered excerpts you are given.

Rules:
- Answer only if an excerpt states the answer directly. If you would have to infer, \
combine, compute or rank figures yourself (for example decide which sector is "the \
first"), or use outside knowledge, set "answerable" to false.
- Check that each figure refers to exactly the period, place and scenario asked.
- For each fact, put in "evidence" the sentence of the excerpt that states it, copied \
word for word in the excerpt's language, with the excerpt number n.
- Write the answer in the language of the question, in 1 to 4 sentences, without \
markdown. Copy numbers and units exactly as the excerpt writes them (Mt éqCO2, GtCO2, \
ppm, °C, USD billion): never convert, round or translate a unit.
- Cite the excerpt after each fact as [n], e.g. [2] or [1][3].
- Excerpts may be in English or French, whatever the language of the question.
- If "answerable" is false, say in one sentence that the documents provided do not \
answer the question, with no evidence.

Answer with JSON only:
{"answerable": true, "answer": "<answer with [n] citations>",
 "evidence": [{"n": <excerpt number>, "quote": "<exact sentence from that excerpt>"}]}"""

NOT_FOUND = {
    "fr": "Je n'ai pas trouvé la réponse dans les rapports indexés.",
    "en": "I could not find the answer in the indexed reports.",
}

LANGUAGE_NAMES = {"fr": "French", "en": "English"}
_CITATION_RE = re.compile(r"\[(\d{1,2})\]")
# A number standing alone: not the digits of CO2, CH4, SSP1 or a footnote call glued
# to a word ("emissions23").
_NUMBER_RE = re.compile(r"(?<![^\W\d])(?<![\d.])\d+(?:[.,]\d+)*")
_GLUED_DIGITS_RE = re.compile(r"(?<=[^\W\d])\d+(?:[.,]\d+)*")  # "éqCO2146,147"
_ELLIPSIS_RE = re.compile(r"\s*(?:\[\.\.\.\]|\[…\]|\.\.\.|…)\s*")
MAX_QUOTE_WORDS = 60  # a sentence, not a page of figure labels


class Evidence(BaseModel):
    n: int  # excerpt number
    quote: str  # sentence copied from that excerpt


class LLMAnswer(BaseModel):
    """What the model must return (validated, retried when invalid).

    Only "answerable" is required. A refusal often comes back with "evidence":
    null, "answer": null or no evidence field at all: rejecting it would trigger
    retries with growing waits (slow "not found" answers) for nothing, since a
    refusal needs no evidence. Malformed evidence items are dropped, not fatal.
    """

    answerable: bool
    answer: str = ""
    evidence: list[Evidence] = Field(default_factory=list)

    @field_validator("answer", mode="before")
    @classmethod
    def _no_null_answer(cls, value: object) -> object:
        return "" if value is None else value

    @field_validator("evidence", mode="before")
    @classmethod
    def _usable_evidence(cls, value: object) -> list:
        if not isinstance(value, list):
            return []
        return [
            item
            for item in value
            if isinstance(item, dict)
            and isinstance(item.get("n"), int)
            and isinstance(item.get("quote"), str)
            and item["quote"].strip()
        ]


class Source(BaseModel):
    n: int  # the [n] used in the answer
    passage_id: str
    doc_id: str
    title: str
    page: int
    section: str | None = None
    excerpt: str  # the passage text, to show next to the answer
    quotes: list[str]  # the verified sentences of this passage the answer relies on


class Answer(BaseModel):
    question: str
    answer: str
    answerable: bool
    sources: list[Source]  # the cited passages only, in [n] order
    retrieved: list[str]  # ids of every passage given to the model, for evaluation
    model: str
    prompt_version: str = PROMPT_VERSION
    # Diagnostics, to understand a refusal: what the model said and what failed.
    model_answerable: bool | None = None
    rejected_quotes: list[str] = Field(default_factory=list)
    unsupported_numbers: list[str] = Field(default_factory=list)
    # Where the time goes: search_s, llm_s (waits and retries included), llm_requests.
    timings: dict[str, float] = Field(default_factory=dict)


def question_language(question: str) -> str:
    """Language of the "not found" message, French or English.

    The model answers in the language of the question on its own; this is only
    for the message written in code. Too short to tell ("SSP1-1.9?"): English.
    """
    language, _confidence = classify_text(question)
    return "fr" if language == "fr" else "en"


def build_messages(question: str, passages: list[Passage]) -> list[dict]:
    excerpts = []
    for n, p in enumerate(passages, start=1):
        where = f"{p.title}, p. {p.page}" + (f" — {p.section}" if p.section else "")
        excerpts.append(f"[{n}] ({where})\n{p.text}")
    user = f"Question: {question}\n\nExcerpts:\n\n" + "\n\n".join(excerpts)
    language, _confidence = classify_text(question)
    if language in LANGUAGE_NAMES:
        # Said explicitly: with "the language of the question" only, a small model
        # answering from English excerpts mixes languages ("0.20 mètres ... entre").
        name = LANGUAGE_NAMES[language]
        user += f"\n\nWrite the whole answer in {name}, even where the excerpts are not."
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def _normalise_numbers(text: str) -> str:
    """Write numbers one way, "2 390" -> "2390" and "1,09" -> "1.09", so both match."""
    text = re.sub(r"(?<=\d)[ \u00a0\u202f](?=\d{3}\b)", "", text)
    return re.sub(r"(?<=\d),(?=\d)", ".", text)


def _undouble(word: str) -> str:
    """Fold double letters, "cette" -> "cete", to match the damaged spellings."""
    return re.sub(r"([a-z])\1", r"\1", word)


def _one_edit(a: str, b: str) -> bool:
    """True if a and b differ by one letter replaced, added or removed."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b, strict=True)) == 1
    short, long = sorted((a, b), key=len)
    return any(long[:i] + long[i + 1 :] == short for i in range(len(long)))


def _same_word(quoted: str, found: str) -> bool:
    """Equal; or one letter apart for words of 5 letters or more (damaged spelling);
    or the excerpt's word is the quoted one plus a glued footnote call of one or
    two letters ("éqCO2I" for "éqCO2" + note I)."""
    if quoted == found:
        return True
    if min(len(quoted), len(found)) >= 5 and _one_edit(quoted, found):
        return True
    return len(quoted) >= 3 and found.startswith(quoted) and len(found) - len(quoted) <= 2


def _in_order(quote: list[str], passage: list[str], max_skipped: int) -> bool:
    """Are the quote's words found in the passage, in order, skipping at most
    ``max_skipped`` passage words in between? (a subsequence match)"""
    for start, word in enumerate(passage):
        if not _same_word(quote[0], word):
            continue
        position, skipped = start + 1, 0
        for wanted in quote[1:]:
            while position < len(passage) and not _same_word(wanted, passage[position]):
                position += 1
                skipped += 1
            if position >= len(passage) or skipped > max_skipped:
                break
            position += 1
        else:
            return True
    return False


def _words(text: str) -> list[str]:
    """Words compared between a quote and an excerpt, numbers aside (they are
    checked on their own): accents, plurals, stop words, double letters and
    digits glued to words ("emissions23", "CO2") left out."""
    tokens = tokenize(_GLUED_DIGITS_RE.sub("", text))
    return [_undouble(w) for w in tokens if not w[0].isdigit()]


def quote_in_passage(quote: str, passage: str) -> bool:
    """Is the quote really taken from the passage? (numbers exact, words nearly)

    The model may shorten a quote with "..." : each part is checked on its own.
    A number only has to appear where a number starts in the passage: the
    extraction glues footnote calls to numbers ("1850-19009" for "1850-1900" +
    note 9), and the model rightly drops them when quoting.
    """
    quote, passage = _normalise_numbers(quote), _normalise_numbers(passage)
    raw = [part for part in _ELLIPSIS_RE.split(quote) if re.search(r"\w", part)]
    parts = [_words(part) for part in raw] or [[]]
    if sum(map(len, parts)) > MAX_QUOTE_WORDS or min(map(len, parts)) < 3:
        # A chunk of text rather than a sentence, or a part too short to prove
        # anything ("... 1.0": a lone figure picked from a chart's labels).
        return False
    for number in _NUMBER_RE.findall(quote):
        # Must start where a number starts: "20" is not found inside "2020".
        if not re.search(rf"(?<![\d.]){re.escape(number)}", passage):
            return False
    text = _words(passage)
    return all(_in_order(words, text, max_skipped=max(3, len(words) // 3)) for words in parts)


def verified_evidence(
    reply: LLMAnswer, passages: list[Passage]
) -> tuple[dict[int, list[str]], list[str]]:
    """({excerpt number: its verified quotes}, the quotes that failed the check)."""
    verified: dict[int, list[str]] = {}
    rejected: list[str] = []
    for item in reply.evidence:
        if 1 <= item.n <= len(passages) and quote_in_passage(item.quote, passages[item.n - 1].text):
            verified.setdefault(item.n, []).append(item.quote.strip())
        else:
            rejected.append(f"[{item.n}] {item.quote}")
    return dict(sorted(verified.items())), rejected


def unsupported_numbers(answer: str, excerpts: list[str], question: str) -> list[str]:
    """Numbers of the answer found neither in the excerpts it relies on (those with
    a verified quote) nor in the question: computed, converted or invented figures.

    Whole numbers only: "20" is not supported by "2018", nor "1.2" by "1.20".
    """
    support = _normalise_numbers(" ".join([*excerpts, question]))
    numbers = _NUMBER_RE.findall(_normalise_numbers(_CITATION_RE.sub("", answer)))
    return [
        n
        for n in dict.fromkeys(numbers)
        if not re.search(rf"(?<![\d.]){re.escape(n)}(?![\d]|\.\d)", support)
    ]


def keep_markers(text: str, valid: list[int]) -> str:
    """Remove the [n] markers whose excerpt has no verified quote."""
    text = _CITATION_RE.sub(lambda m: m.group(0) if int(m.group(1)) in valid else "", text)
    text = text.replace("**", "")  # markdown bold, despite the instruction
    return re.sub(r"\s+([.,;])", r"\1", text).strip()


class Answerer:
    def __init__(self, searcher: HybridSearcher, chat: ChatClient, top_k: int = 5) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.searcher = searcher
        self.chat = chat
        self.top_k = top_k

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        start = time.perf_counter()
        hits = self.searcher.search(question, top_k or self.top_k, "hybrid")
        passages = [h.passage for h in hits]
        not_found = NOT_FOUND[question_language(question)]
        timings = {"search_s": round(time.perf_counter() - start, 3)}
        base = {
            "question": question,
            "retrieved": [p.id for p in passages],
            "model": self.chat.model,
            "timings": timings,
        }
        if not passages:
            return Answer(answer=not_found, answerable=False, sources=[], **base)

        requests, llm_start = self.chat.requests, time.perf_counter()
        reply = self.chat.complete_json(build_messages(question, passages), LLMAnswer)
        timings["llm_s"] = round(time.perf_counter() - llm_start, 3)
        timings["llm_requests"] = self.chat.requests - requests  # more than 1: retries
        base["model_answerable"] = reply.answerable
        if not reply.answerable:
            return Answer(answer=not_found, answerable=False, sources=[], **base)
        evidence, base["rejected_quotes"] = verified_evidence(reply, passages)
        cited = [passages[n - 1].text for n in evidence]
        base["unsupported_numbers"] = unsupported_numbers(reply.answer, cited, question)
        if not evidence or base["unsupported_numbers"]:
            # Nothing the user could verify, or a figure that comes from nowhere.
            return Answer(answer=not_found, answerable=False, sources=[], **base)

        sources = []
        for n, quotes in evidence.items():
            p = passages[n - 1]
            sources.append(
                Source(
                    n=n,
                    passage_id=p.id,
                    doc_id=p.doc_id,
                    title=p.title,
                    page=p.page,
                    section=p.section,
                    excerpt=p.text,
                    quotes=quotes,
                )
            )
        text = keep_markers(reply.answer, list(evidence))
        return Answer(answer=text, answerable=True, sources=sources, **base)


def build_answerer(settings: Settings | None = None) -> Answerer:
    """The answerer described by the settings: passages + vectors on disk, the LLM.

    Raises ``FileNotFoundError`` when the index has not been built yet.
    """
    settings = settings or get_settings()
    path = settings.rag_index_dir / "passages.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run scripts/rag_index.py first")
    embedder = build_embedder(settings)
    passages, store = load_passages(path), open_store(settings, embedder)
    missing = len(store.missing(passages))
    if missing:
        # Dense search would only see part of the corpus: answers stay possible
        # (BM25 sees everything) but are worse, so say it loudly.
        logger.warning(
            "%d of %d passages have no %s vector in %s: run scripts/rag_index.py or "
            "scripts/rag_copy_vectors.py",
            missing,
            len(passages),
            embedder.name,
            store.collection,
        )
    searcher = HybridSearcher(passages, embedder, store)
    # Temperature 0: the most likely wording, closest to the excerpts, and repeatable.
    chat = ChatClient(settings, temperature=0.0, max_retries=settings.ask_max_retries)
    return Answerer(searcher, chat, top_k=settings.rag_top_k)
