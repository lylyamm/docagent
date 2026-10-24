"""The agent's tools: what DocAgent already does, behind small typed functions.

The graph (``docagent/agent/graph.py``, written in week 4) never touches the
PDF code, the index or the LLM client directly: each node calls one of these
tools and stores the result in the graph state.

    tool              what it wraps                       LLM call?
    ─────────────────────────────────────────────────────────────────
    pdf_info          page count + language detection     no
    translate_pdf     pdf.rebuild_document + verify       yes (per page)
    search_documents  rag.HybridSearcher                  no
    ask               rag.Answerer (verified quotes)      yes (1)
    summarize_document  its own summary (or search) -> points  yes (1)
    compare_documents   5 passages per report -> points    yes (1)
    translate_query   a short JSON request                yes (1)

Why plain functions and not LangChain ``@tool`` objects: the Translation and
Questions workflows call their tools in a fixed order, so no model has to pick
them. Only the ingestion agent (an LLM choosing its next step) needs tools
described to a model, and those come from the MCP servers (Fetch, Filesystem).

Every result is a Pydantic model: it can be put in the graph state, logged as
a trajectory step and returned by the API as it is.
"""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pymupdf
from pydantic import BaseModel, field_validator

from ..chat import ChatClient, tools_chat_client
from ..config import Settings, get_settings
from ..pdf import detect_language, rebuild_document
from ..pdf.language import classify_text
from ..pdf.verify import verify_rebuild
from ..rag import Answer, Answerer, CorpusDoc, HybridSearcher, Passage, build_answerer, load_corpus
from ..translate import Translator, build_translator
from .grounded import (
    COMPARE_PROMPT_VERSION,
    COMPARE_SYSTEM_PROMPT,
    SUMMARY_PROMPT_VERSION,
    SUMMARY_SYSTEM_PROMPT,
    ComparisonPoint,
    LLMComparison,
    LLMPoints,
    Point,
    RejectedPoint,
    Report,
    check_comparison,
    check_points,
    excerpts_message,
)

LANGUAGES = {"fr": "French", "en": "English"}
QUERY_PROMPT_VERSION = "query-v1"
# Characters of the report's own summary sent in one request (~15k tokens): the
# WEO's executive summary (27k) fits whole, the IPCC SPM (120k) is sampled.
SUMMARY_MAX_CHARS = 60_000
# Without a focus, the summary is built around what a reader looks for first.
DEFAULT_FOCUS = {
    "fr": "principaux constats, chiffres clés et conclusions du rapport",
    "en": "main findings, key figures and conclusions of the report",
}

QUERY_SYSTEM_PROMPT = """You translate search queries about climate and energy reports \
(IPCC, IEA, Haut Conseil pour le climat).

Rules:
- Translate the meaning, as a native speaker would phrase the same question.
- Use the usual terms of the field in the target language ("émissions de gaz à effet de \
serre" / "greenhouse gas emissions", "neutralité carbone" / "net zero").
- Keep unchanged: numbers, years, units, acronyms and scenario names (SSP1-2.6, NZE, GES, \
CO2, COP28).
- Do not answer the question, do not add words.

Reply with JSON only: {"translation": "<the translated query>"}"""


class ToolError(RuntimeError):
    """A tool could not do its job; the message is meant for the user."""


# -- results --------------------------------------------------------------------


class PdfInfo(BaseModel):
    path: str
    pages: int
    language: str | None  # ISO 639-1, None if unknown
    language_confidence: float
    needs_confirmation: bool  # more pages than ``agent_confirm_pages``: ask first


class TranslationResult(BaseModel):
    output: str  # path of the translated PDF
    pages: int
    source_lang: str
    target_lang: str
    zones: int  # text zones rewritten
    mean_font_scale: float
    overflow: int  # zones whose text did not fit, even at the smallest size
    checks_ok: bool
    duration_s: float


class PassageRef(BaseModel):
    """One search hit, short enough to show in a trajectory."""

    passage_id: str
    doc_id: str
    title: str
    page: int
    score: float
    found_by: list[str]  # "dense", "bm25" or both
    text: str


class DocumentSummary(BaseModel):
    """A summary built from the most relevant passages of ONE report (not all of it).

    ``passages_used`` / ``passages_total`` say how much of the report was read.
    """

    doc_id: str
    title: str
    focus: str
    lang: str
    # "author_summary": from the report's own summary pages; "search": the
    # passages most relevant to the focus (or no summary pages are known)
    method: Literal["author_summary", "search"]
    pages: list[int]  # pages the excerpts come from
    points: list[Point]  # each with its sources (page, verified quote)
    passages_used: int
    passages_total: int
    retrieved: list[str]  # passage ids given to the model
    rejected_points: list[RejectedPoint]  # points that failed a check, and why
    model: str
    prompt_version: str = SUMMARY_PROMPT_VERSION

    @property
    def ok(self) -> bool:
        return bool(self.points)


class ComparedDocument(BaseModel):
    doc_id: str
    title: str
    passages_read: int  # passages of this report given to the model


class DocumentComparison(BaseModel):
    """What 2 or 3 reports say about one topic: agreements, differences, and what only
    one of them addresses (in the passages read: absence is never claimed further)."""

    topic: str
    lang: str
    documents: list[ComparedDocument]
    points: list[ComparisonPoint]
    retrieved: list[str]
    rejected_points: list[RejectedPoint]
    model: str
    prompt_version: str = COMPARE_PROMPT_VERSION

    @property
    def ok(self) -> bool:
        return bool(self.points)


class QueryTranslation(BaseModel):
    query: str
    translation: str
    source_lang: str
    target_lang: str


class _LLMQuery(BaseModel):
    translation: str

    @field_validator("translation")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("empty translation")
        return value


# -- helpers ------------------------------------------------------------------------


def spread(passages: list[Passage], max_chars: int) -> list[Passage]:
    """Passages evenly spaced through the list, within ``max_chars``.

    Keeping every k-th passage covers the whole summary (each section keeps a
    share), where "the first ones" would stop half-way.
    """
    total = sum(len(p.text) for p in passages)
    if total <= max_chars:
        return passages
    keep = max(1, int(len(passages) * max_chars / total))
    step = len(passages) / keep
    return [passages[int(i * step)] for i in range(keep)]


def query_language(text: str) -> str:
    """Return "fr" or "en" (too short to tell: English, like the answerer)."""
    language, _confidence = classify_text(text)
    return "fr" if language == "fr" else "en"


def other_language(language: str) -> str:
    """The corpus has French and English reports: the other one of the pair."""
    return "en" if language == "fr" else "fr"


# -- the toolbox ------------------------------------------------------------------


class Toolbox:
    """The tools, sharing the heavy objects (index, model, LLM clients).

    Everything is injected so the tests can pass small fakes; ``from_settings``
    builds the real thing, lazily: a translation request never loads the index.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        answerer: Answerer | None = None,
        make_answerer: Callable[[], Answerer] | None = None,
        make_translator: Callable[[], Translator] | None = None,
        chat: ChatClient | None = None,
        output_dir: Path | None = None,
    ) -> None:
        self.settings = settings
        self._answerer = answerer
        self._make_answerer = make_answerer or (lambda: build_answerer(settings))
        self._make_translator = make_translator or (
            lambda: build_translator(settings, fake=settings.translator == "fake")
        )
        self._chat = chat
        self._corpus: dict[str, CorpusDoc] | None = None
        self.output_dir = output_dir or Path("data/output")

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "Toolbox":
        return cls(settings or get_settings())

    @property
    def answerer(self) -> Answerer:
        if self._answerer is None:
            self._answerer = self._make_answerer()  # FileNotFoundError: no index yet
        return self._answerer

    @property
    def corpus(self) -> dict[str, CorpusDoc]:
        """corpus.json by document id (titles, summary pages); empty if missing."""
        if self._corpus is None:
            path = Path(self.settings.corpus_path)
            self._corpus = {d.id: d for d in load_corpus(path)} if path.exists() else {}
        return self._corpus

    @property
    def searcher(self) -> HybridSearcher:
        return self.answerer.searcher

    @property
    def chat(self) -> ChatClient:
        if self._chat is None:
            # The same model as the answers (TOOLS_PROVIDER), at temperature 0.
            self._chat = tools_chat_client(self.settings)
        return self._chat

    def usage(self) -> tuple[int, int]:
        """Input and output tokens spent so far by the tools' models (summaries and
        comparisons, answers): what the supervisor's count leaves out."""
        clients = [self._chat, self._answerer.chat if self._answerer else None]
        seen = {id(c): c for c in clients if c is not None}.values()  # may be one client
        return sum(c.input_tokens for c in seen), sum(c.output_tokens for c in seen)

    # -- Translation workflow ---------------------------------------------------

    def pdf_info(self, path: str | Path) -> PdfInfo:
        """Page count and language of a PDF; refuses what cannot be translated."""
        path = Path(path)
        if path.suffix.lower() != ".pdf" or not path.is_file():
            raise ToolError(f"{path.name}: no such PDF file")
        try:
            with pymupdf.open(path) as doc:
                if doc.needs_pass:
                    raise ToolError(f"{path.name}: password-protected PDFs are not supported")
                pages = doc.page_count
                detected = detect_language(doc)
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001 - any parsing failure is a bad file
            raise ToolError(f"{path.name}: the PDF could not be opened") from exc
        if pages == 0:
            raise ToolError(f"{path.name}: the PDF has no pages")
        return PdfInfo(
            path=str(path),
            pages=pages,
            language=detected.language,
            language_confidence=detected.confidence,
            needs_confirmation=pages > self.settings.agent_confirm_pages,
        )

    def translate_pdf(
        self,
        path: str | Path,
        target_lang: str = "fr",
        source_lang: str | None = None,
        on_page: Callable[[int, int], None] | None = None,
    ) -> TranslationResult:
        """Translate a PDF, keeping its layout; the result goes to ``output_dir``.

        The size check and the user's confirmation happen before, in the graph
        (``pdf_info`` then ``interrupt()``): this tool does what it is told.
        """
        info = self.pdf_info(path)
        source_lang = source_lang or info.language
        if not source_lang:
            raise ToolError(f"{Path(path).name}: language unknown, say which one it is")
        if source_lang == target_lang:
            name = LANGUAGES.get(target_lang, target_lang)
            raise ToolError(f"{Path(path).name} is already in {name}")
        source = Path(path)
        output = self.output_dir / f"{source.stem}_{target_lang}.pdf"
        start = time.perf_counter()
        report = rebuild_document(
            source,
            output,
            self._make_translator(),
            source_lang=source_lang,
            target_lang=target_lang,
            on_page=on_page,
        )
        check = verify_rebuild(source, output, report)
        return TranslationResult(
            output=str(output),
            pages=info.pages,
            source_lang=source_lang,
            target_lang=target_lang,
            zones=report.targets,
            mean_font_scale=report.mean_scale,
            overflow=report.overflow,
            checks_ok=check.ok,
            duration_s=round(time.perf_counter() - start, 2),
        )

    # -- Questions workflow -----------------------------------------------------

    def search_documents(
        self, query: str, top_k: int | None = None, doc_ids: list[str] | None = None
    ) -> list[PassageRef]:
        """Hybrid search only: what the corpus says, without the LLM.

        ``doc_ids`` limits the search to some reports (to summarize one, or to
        compare two: each must bring its own passages).
        """
        if doc_ids:
            known = {p.doc_id for p in self.searcher.passages}
            unknown = [d for d in doc_ids if d not in known]
            if unknown:
                raise ToolError(
                    f"unknown document(s): {', '.join(unknown)}; "
                    f"indexed: {', '.join(sorted(known))}"
                )
        top_k = top_k or self.settings.rag_top_k
        hits = self.searcher.search(query, top_k, "hybrid", doc_ids)
        return [
            PassageRef(
                passage_id=h.passage.id,
                doc_id=h.passage.doc_id,
                title=h.passage.title,
                page=h.passage.page,
                score=h.score,
                found_by=[
                    name
                    for name, rank in (("dense", h.dense_rank), ("bm25", h.bm25_rank))
                    if rank is not None
                ],
                text=h.passage.text,
            )
            for h in hits
        ]

    def summarize_document(
        self,
        doc_id: str,
        focus: str | None = None,
        lang: str = "fr",
        passages: int = 8,
        max_points: int = 5,
    ) -> DocumentSummary:
        """The key points of one report, each one checked against its page.

        A report is too long to send whole (the WEO: ~500 pages), and summarising
        every part then the summaries (map-reduce) costs ~80 model calls. So, in
        one call:

        - no focus: the report's own summary (``summary_pages`` in corpus.json),
          whole or evenly sampled. Searching for "main findings" does not work:
          it finds passages that use those words, not the main messages;
        - a focus ("the NZE scenario"): the ``passages`` most relevant to it in
          that report.

        Every point is checked like an answer; points that fail are dropped one
        by one, and no point left means the excerpts did not allow a summary.
        """
        if lang not in LANGUAGES:
            raise ToolError(f"summary language must be one of {', '.join(LANGUAGES)}")
        in_doc = [p for p in self.searcher.passages if p.doc_id == doc_id]
        if not in_doc:
            known = sorted({p.doc_id for p in self.searcher.passages})
            raise ToolError(f"unknown document: {doc_id}; indexed: {', '.join(known)}")
        doc = self.corpus.get(doc_id)
        if not (focus or "").strip() and doc and doc.summary_pages:
            # The authors already summarised their report: start from that.
            first, last = doc.summary_pages
            method: Literal["author_summary", "search"] = "author_summary"
            focus = DEFAULT_FOCUS[lang]
            found = spread([p for p in in_doc if first <= p.page <= last], SUMMARY_MAX_CHARS)
            where = f"Excerpts of the report's own summary (pages {first} to {last})"
        else:
            method = "search"
            focus = (focus or "").strip() or DEFAULT_FOCUS[lang]
            found = [h.passage for h in self.searcher.search(focus, passages, "hybrid", [doc_id])]
            where = "Excerpts of the report most relevant to the focus"
        if not found:
            raise ToolError(f"{doc_id}: no passage found for this focus")
        system = SUMMARY_SYSTEM_PROMPT.format(max_points=max_points, language=LANGUAGES[lang])
        user = f"Report: {found[0].title}\nFocus: {focus}\n\n{where}:\n\n{excerpts_message(found)}"
        reply = self.chat.complete_json(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            LLMPoints,
        )
        points, rejected = check_points(reply, found, focus)
        return DocumentSummary(
            doc_id=doc_id,
            title=found[0].title,
            focus=focus,
            lang=lang,
            method=method,
            pages=sorted({p.page for p in found}),
            points=points[:max_points],
            passages_used=len(found),
            passages_total=len(in_doc),
            retrieved=[p.id for p in found],
            rejected_points=rejected,
            model=self.chat.model,
        )

    def compare_documents(
        self,
        doc_ids: list[str],
        topic: str,
        lang: str = "fr",
        per_doc: int = 5,
        max_points: int = 5,
    ) -> DocumentComparison:
        """Compare what 2 or 3 reports say about ``topic``, each point checked.

        Each report is searched on its own (``per_doc`` passages each): searched
        together, the WEO (1,937 passages) could leave nothing to the IPCC SPM.
        One model call. Each point is made of sides, one per report (labelled
        A, B, C in the prompt); on top of the summary checks, a side labelled A
        may only quote excerpts of report A, and report names are written by
        code. An agreement or a difference has 2+ sides, "only_in" exactly one,
        and the code (not the model) adds what was read of the other reports.
        """
        ids = list(dict.fromkeys(doc_ids))
        if not 2 <= len(ids) <= 3:
            raise ToolError("compare 2 or 3 different reports")
        if lang not in LANGUAGES:
            raise ToolError(f"language must be one of {', '.join(LANGUAGES)}")
        topic = topic.strip()
        if not topic:
            raise ToolError("say what to compare (the topic)")
        known = {p.doc_id for p in self.searcher.passages}
        unknown = [d for d in ids if d not in known]
        if unknown:
            raise ToolError(
                f"unknown document(s): {', '.join(unknown)}; indexed: {', '.join(sorted(known))}"
            )

        found: list[Passage] = []
        documents: list[ComparedDocument] = []
        reports: list[Report] = []
        for label, doc_id in zip("ABC", ids, strict=False):
            hits = self.searcher.search(topic, per_doc, "hybrid", [doc_id])
            if not hits:
                raise ToolError(f"{doc_id}: no passage found on this topic")
            title = hits[0].passage.title
            first = len(found) + 1
            found += [h.passage for h in hits]
            documents.append(ComparedDocument(doc_id=doc_id, title=title, passages_read=len(hits)))
            reports.append(
                Report(label=label, doc_id=doc_id, title=title, first=first, last=len(found))
            )

        listing = [f"- {r.label} = {r.title}: excerpts {r.first} to {r.last}" for r in reports]
        system = COMPARE_SYSTEM_PROMPT.format(max_points=max_points, language=LANGUAGES[lang])
        user = (
            f"Topic: {topic}\n\nReports compared:\n"
            + "\n".join(listing)
            + f"\n\nExcerpts:\n\n{excerpts_message(found)}"
        )
        reply = self.chat.complete_json(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            LLMComparison,
        )
        points, rejected = check_comparison(reply, found, topic, reports, lang)
        return DocumentComparison(
            topic=topic,
            lang=lang,
            documents=documents,
            points=points[:max_points],
            retrieved=[p.id for p in found],
            rejected_points=rejected,
            model=self.chat.model,
        )

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        """Search, then an answer whose quotes and figures are checked by code.

        A refusal (``answerable=False``) is a normal result, not an error: the
        graph decides what to do next (translate the question and try again).
        """
        return self.answerer.ask(question, top_k)

    def translate_query(self, query: str, target_lang: str | None = None) -> QueryTranslation:
        """The same question in the other language of the corpus.

        A French question on an English report shares few words with it: BM25
        misses it (0/4 cross-lingual questions in the test set), dense search
        less often. Asking again in the report's language is the retry.
        """
        source_lang = query_language(query)
        target_lang = target_lang or other_language(source_lang)
        if target_lang not in LANGUAGES:
            raise ToolError(f"cannot translate a query into {target_lang!r}")
        messages = [
            {"role": "system", "content": QUERY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Target language: {LANGUAGES[target_lang]}\n\nQuery: {query}",
            },
        ]
        reply = self.chat.complete_json(messages, _LLMQuery)
        return QueryTranslation(
            query=query,
            translation=reply.translation,
            source_lang=source_lang,
            target_lang=target_lang,
        )


__all__ = [
    "PassageRef",
    "PdfInfo",
    "QueryTranslation",
    "ToolError",
    "Toolbox",
    "TranslationResult",
    "other_language",
    "query_language",
]
