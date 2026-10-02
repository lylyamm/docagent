"""Answer a question from the reports, with citations (the "G" of RAG).

    question ──► hybrid search ──► 5 passages, numbered [1]..[5]
                                        │
                                        ▼
              LLM: "answer ONLY from these excerpts, cite them as [n],
                    say so when they do not contain the answer"
                                        │
                                        ▼
              JSON {answerable, answer, citations} ──► checked in code

The model is never trusted blindly:
- a citation must point to one of the excerpts it was given (no invented [7]);
- an answer without any valid citation is not shown: it would be unverifiable,
  so the user gets the standard "not found" message instead;
- the sources returned are the cited passages with their document and page,
  so every claim can be checked in the PDF.
"""

import re

from pydantic import BaseModel, Field

from ..chat import ChatClient
from ..config import Settings, get_settings
from ..pdf.language import classify_text
from .chunking import Passage
from .embeddings import build_embedder
from .index import load_passages, open_store
from .search import HybridSearcher

PROMPT_VERSION = "ask-v1"

SYSTEM_PROMPT = """You answer questions about climate and energy reports (IPCC, IEA, \
French High Council on Climate) using ONLY the numbered excerpts you are given.

Rules:
- Answer in the language of the question, in 1 to 5 sentences.
- Use only facts stated in the excerpts. No outside knowledge, no guesses.
- After each sentence that states a fact, cite its excerpt(s) as [n], e.g. [2] or [1][3].
- Copy numbers, units, years and scenario names exactly as the excerpts give them, \
and say which year or scenario a figure refers to.
- Excerpts may be in English or French, whatever the language of the question.
- If the excerpts do not contain the answer, set "answerable" to false and say in one \
sentence that the documents provided do not answer the question.

Answer with JSON only:
{"answerable": true, "answer": "<answer with [n] citations>", "citations": [<n>, ...]}"""

NOT_FOUND = {
    "fr": "Je n'ai pas trouvé la réponse dans les rapports indexés.",
    "en": "I could not find the answer in the indexed reports.",
}

_CITATION_RE = re.compile(r"\[(\d{1,2})\]")


class LLMAnswer(BaseModel):
    """What the model must return (validated, retried when invalid)."""

    answerable: bool
    answer: str
    citations: list[int] = Field(default_factory=list)


class Source(BaseModel):
    n: int  # the [n] used in the answer
    passage_id: str
    doc_id: str
    title: str
    page: int
    section: str | None = None
    excerpt: str  # the passage text, to show next to the answer


class Answer(BaseModel):
    question: str
    answer: str
    answerable: bool
    sources: list[Source]  # the cited passages only, in [n] order
    retrieved: list[str]  # ids of every passage given to the model, for evaluation
    model: str


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
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def cited_numbers(reply: LLMAnswer, available: int) -> list[int]:
    """Citations that point to a given excerpt: from the list and from the text."""
    numbers = set(reply.citations) | {int(m) for m in _CITATION_RE.findall(reply.answer)}
    return sorted(n for n in numbers if 1 <= n <= available)


def strip_invalid_markers(text: str, valid: list[int]) -> str:
    """Remove [n] markers that point to no excerpt (the model invented them)."""
    return _CITATION_RE.sub(lambda m: m.group(0) if int(m.group(1)) in valid else "", text).strip()


class Answerer:
    def __init__(self, searcher: HybridSearcher, chat: ChatClient, top_k: int = 5) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.searcher = searcher
        self.chat = chat
        self.top_k = top_k

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        hits = self.searcher.search(question, top_k or self.top_k, "hybrid")
        passages = [h.passage for h in hits]
        not_found = NOT_FOUND[question_language(question)]
        base = {
            "question": question,
            "retrieved": [p.id for p in passages],
            "model": self.chat.model,
        }
        if not passages:
            return Answer(answer=not_found, answerable=False, sources=[], **base)

        reply = self.chat.complete_json(build_messages(question, passages), LLMAnswer)
        cited = cited_numbers(reply, len(passages))
        if not reply.answerable or not cited:
            # Refusal, or an answer that cites nothing: nothing the user could verify.
            return Answer(answer=not_found, answerable=False, sources=[], **base)

        sources = [
            Source(
                n=n,
                passage_id=passages[n - 1].id,
                doc_id=passages[n - 1].doc_id,
                title=passages[n - 1].title,
                page=passages[n - 1].page,
                section=passages[n - 1].section,
                excerpt=passages[n - 1].text,
            )
            for n in cited
        ]
        text = strip_invalid_markers(reply.answer, cited)
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
    searcher = HybridSearcher(load_passages(path), embedder, open_store(settings, embedder))
    return Answerer(searcher, ChatClient(settings), top_k=settings.rag_top_k)
