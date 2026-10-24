"""Texts written from excerpts, each point checked by code: summaries (and comparisons).

Same contract as the answers (``rag/answer.py``): the model writes from
numbered excerpts and copies, for each point, the sentence that supports it.
Code then checks every point on its own:

    the quote is in the excerpt it cites     (quote_in_passage)
    every number of the point is in the cited excerpts or the request
                                              (unsupported_numbers)

A point that fails is dropped and reported; the others are kept. One invented
figure costs one point, not the whole summary.
"""

import re

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

from ..rag.answer import (
    Evidence,
    LLMAnswer,
    Source,
    keep_markers,
    unsupported_numbers,
    verified_evidence,
)
from ..rag.chunking import Passage

# v2: one fact per point, every figure inside the quote that proves it. With v1,
# points packed 3-4 facts with one quote, and a true figure taken from another
# excerpt got the whole point rejected (HCC 2025: 3 of 5 points lost).
SUMMARY_PROMPT_VERSION = "summary-v2"

SUMMARY_SYSTEM_PROMPT = """You summarise a climate or energy report (IPCC, IEA, French \
High Council on Climate) using ONLY the numbered excerpts you are given, all taken \
from that report.

Rules:
- Write at most {max_points} points: the most important findings about the focus, \
the most important first.
- ONE fact per point, in one sentence. Do not pack several facts or figures from \
different places into the same point: write separate points instead.
- For each point, put in "evidence" the sentence that states the fact, copied word for \
word from the excerpt (in the excerpt's language), with its excerpt number n.
- Every figure of the point must appear in one of its quoted sentences. If the point \
uses two excerpts, give one quote from each.
- Copy numbers and units exactly as the excerpt writes them (Mt éqCO2, GtCO2, ppm, °C, \
USD billion): never convert, round, add up or compare figures yourself.
- Cite the excerpt after the fact as [n].
- Do not use outside knowledge. If the excerpts say nothing useful about the focus, \
return no points.
- Write the points in {language}, even where the excerpts are not, without markdown.

Answer with JSON only:
{{"points": [{{"text": "<one-fact point with [n] citations>", \
"evidence": [{{"n": <excerpt number>, "quote": "<exact sentence from that excerpt>"}}]}}]}}"""


class LLMPoint(BaseModel):
    text: str
    evidence: list[Evidence] = Field(default_factory=list)


def _wrap_bare_list(value: object) -> object:
    """``[{...}, {...}]`` read as ``{"points": [...]}``: nemo sometimes leaves out the
    wrapping object (3 times in a row on one comparison), and the points were fine."""
    return {"points": value} if isinstance(value, list) else value


class LLMPoints(BaseModel):
    """What the model must return. Malformed points are dropped, not fatal."""

    points: list[LLMPoint] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _bare_list(cls, value: object) -> object:
        return _wrap_bare_list(value)

    @field_validator("points", mode="before")
    @classmethod
    def _usable_points(cls, value: object) -> list:
        if not isinstance(value, list):
            return []
        kept = []
        for item in value:
            try:
                point = LLMPoint.model_validate(item)
            except ValueError:
                continue
            if point.text.strip():
                kept.append(point)
        return kept


class Point(BaseModel):
    text: str  # with [n] markers, only those of verified excerpts
    sources: list[Source]  # the cited passages, with their verified quotes


class RejectedPoint(BaseModel):
    """A point that failed a check, with what is needed to see why."""

    text: str
    reason: str  # "no quote found in its excerpt", "figures not in the cited excerpts: ..."
    quotes: list[str] = Field(default_factory=list)  # as proposed: "[n] quote"
    bad_quotes: list[str] = Field(default_factory=list)  # those not found in their excerpt
    # figure -> numbers of the excerpts that do contain it (not cited by the point)
    found_elsewhere: dict[str, list[int]] = Field(default_factory=dict)


def excerpts_message(passages: list[Passage]) -> str:
    blocks = []
    for n, p in enumerate(passages, start=1):
        where = f"{p.title}, p. {p.page}" + (f" — {p.section}" if p.section else "")
        blocks.append(f"[{n}] ({where})\n{p.text}")
    return "\n\n".join(blocks)


def _where_found(numbers: list[str], passages: list[Passage]) -> dict[str, list[int]]:
    """For each figure, the excerpts that contain it: was it a wrong citation?"""
    return {
        number: [
            n
            for n, p in enumerate(passages, start=1)
            if not unsupported_numbers(number, [p.text], "")
        ]
        for number in numbers
    }


def check_point(
    point: "LLMPoint", passages: list[Passage], request: str
) -> "Point | RejectedPoint":
    """One point: kept with its verified sources, or rejected with the reason."""
    evidence_items = [e.model_dump() for e in point.evidence]  # its validator wants dicts
    as_answer = LLMAnswer(answerable=True, answer=point.text, evidence=evidence_items)  # type: ignore[arg-type]
    evidence, bad_quotes = verified_evidence(as_answer, passages)
    proposed = [f"[{e.n}] {e.quote}" for e in point.evidence]
    if not evidence:
        return RejectedPoint(
            text=point.text,
            reason="no quote found in its excerpt",
            quotes=proposed,
            bad_quotes=bad_quotes,
        )
    cited = [passages[n - 1].text for n in evidence]
    numbers = unsupported_numbers(point.text, cited, request)
    if numbers:
        return RejectedPoint(
            text=point.text,
            reason=f"figures not in the cited excerpts: {', '.join(numbers)}",
            quotes=proposed,
            bad_quotes=bad_quotes,
            found_elsewhere=_where_found(numbers, passages),
        )
    sources = [
        Source(
            n=n,
            passage_id=passages[n - 1].id,
            doc_id=passages[n - 1].doc_id,
            title=passages[n - 1].title,
            page=passages[n - 1].page,
            section=passages[n - 1].section,
            excerpt=passages[n - 1].text,
            quotes=quotes,
        )
        for n, quotes in evidence.items()
    ]
    return Point(text=keep_markers(point.text, list(evidence)), sources=sources)


def check_points(
    reply: LLMPoints, passages: list[Passage], request: str
) -> tuple[list[Point], list[RejectedPoint]]:
    """The points that pass the checks, and those that don't (with the reason)."""
    kept: list[Point] = []
    rejected: list[RejectedPoint] = []
    for point in reply.points:
        result = check_point(point, passages, request)
        if isinstance(result, Point):
            kept.append(result)
        else:
            rejected.append(result)
    return kept, rejected


# -- comparisons --------------------------------------------------------------------

# v2: each point is made of "sides", one per report, and the code checks that a
# side labelled A only quotes excerpts of report A. With v1 (free text), the model
# wrote "The IPCC states..." over a quote taken from the IEA's WEO, and two
# unrelated facts were labelled a "difference".
# v3: a side may not say what its report does NOT contain (absence claim, checked by
# code), and an aspect that one report alone addresses is "only_in", not a difference.
# v4: one short sentence per side; the agent asks for 3 points (a comparison is read in
# a chat window: 4-5 points of 5-8 lines each were too long to read).
COMPARE_PROMPT_VERSION = "compare-v4"

COMPARE_SYSTEM_PROMPT = """You compare what climate and energy reports (IPCC, IEA, \
French High Council on Climate) say about a topic, using ONLY the numbered excerpts you \
are given. The reports are labelled A, B (and C); each one has its own excerpt numbers.

Write at most {max_points} points, the most important first. Each point has:
- "kind":
  - "agreement": the reports say the same thing about the same aspect;
  - "difference": the reports answer the SAME question differently (other figures, \
dates, scenarios). Two unrelated facts are not a difference;
  - "only_in": something that only the excerpts of ONE report address. If only one \
report has something on an aspect, it is "only_in", never a "difference".
- "aspect": the question or aspect compared, in a few words (e.g. "emissions in 2035").
- "sides": what each report says about it, one side per report: {{"report": "A", \
"says": "...", "evidence": [...]}}. "says" is ONE short sentence (at most 30 words) with \
the key fact. An agreement or a difference has one side per report (at least two); \
"only_in" has exactly one side. A side quotes ONLY excerpts of \
its own report. Never write what a report does not say: the code adds that, carefully.

Rules:
- In "evidence", copy word for word the sentences that state what the side says (in \
the excerpt's language), with their excerpt numbers n.
- Every figure of a side must appear in one of its quoted sentences. Copy numbers and \
units exactly; never convert, round or compute.
- Do not use outside knowledge. Write "aspect" and "says" in {language}, without markdown.

Answer with JSON only:
{{"points": [{{"kind": "agreement" | "difference" | "only_in", "aspect": "...", \
"sides": [{{"report": "A", "says": "...", \
"evidence": [{{"n": <excerpt number>, "quote": "<exact sentence from that excerpt>"}}]}}]}}]}}"""

KINDS = ("agreement", "difference", "only_in")

# Written by code, never by the model: what was checked, not "the report is silent".
ABSENCE_NOTE = {
    "fr": {
        1: "Rien sur ce point dans le passage consulté de {title}.",
        "n": "Rien sur ce point dans les {n} passages consultés de {title}.",
    },
    "en": {
        1: "Nothing on this in the passage of {title} that was read.",
        "n": "Nothing on this in the {n} passages of {title} that were read.",
    },
}


# A side may only say what its report says. Phrases about what a report does not
# contain ("does not specify", "ne précise pas") cannot be proven from the few
# passages read: the code rejects them. A negation about the world stays allowed
# ("these pathways are not compatible with 1.5 °C" is something the report says).
_REPORT_VERBS_FR = (
    r"mention\w*|pr[ée]cis\w*|fourni\w*|donn\w*|[ée]voqu\w*|abord\w*|indiqu\w*"
    r"|chiffr\w*|trait\w*|parl\w*|d[ée]taill\w*|quantifi\w*|cit\w*"
)
_ABSENCE_RE = re.compile(
    "|".join(
        [
            # "ne précise pas", "ne le mentionne pas", "n'évoque aucun chiffre"
            rf"\bn(?:e\b|['’])\s*(?:\w+\s+){{0,2}}?(?:{_REPORT_VERBS_FR})\s+(?:\w+\s+)?(?:pas|aucun\w*|jamais|rien)\b",
            rf"\bsans\s+(?:les?\s+|la\s+|l['’]\s*)?(?:{_REPORT_VERBS_FR})",
            r"\baucune?\s+(?:chiffre|donn[ée]e|mention|pr[ée]cision|valeur|pourcentage|estimation)",
            r"\bpas\s+de\s+(?:chiffres?|donn[ée]es|mention|pr[ée]cision|pourcentages?)",
            # passive: "n'est pas mentionné", "ne sont pas précisés"
            rf"\bn(?:e\b|['’])\s*(?:est|sont)\s+pas\s+(?:\w+\s+)?(?:{_REPORT_VERBS_FR})",
            r"\b(?:reste|est|demeure)\s+(?:muet|silencieux)|\bpasse\s+sous\s+silence",
            r"\b(?:does|do|did)\s*(?:n['’]t|\s+not)\s+(?:\w+\s+)?(?:mention|specify|provide|give"
            r"|discuss|address|quantify|state|detail|cover|include)",
            r"\bwithout\s+(?:mentioning|specifying|giving|providing|quantifying|stating)",
            r"\bno\s+(?:specific\s+|precise\s+|explicit\s+)?(?:figures?|mention|data|numbers?"
            r"|values?|percentages?|estimates?|details?)\b",
            # passive: "is not mentioned", "are not quantified"
            r"\b(?:is|are|was|were)\s*(?:n['’]t|\s+not)\s+(?:\w+\s+)?(?:mentioned|specified"
            r"|provided|given|discussed|addressed|quantified|stated|detailed|covered)\b",
            r"\bis\s+silent\b|\bsilent\s+on\b",
        ]
    ),
    re.IGNORECASE,
)


def absence_claim(text: str) -> str | None:
    """The phrase saying what a report does not contain, if there is one."""
    match = _ABSENCE_RE.search(text)
    return match.group(0) if match else None


class LLMSide(BaseModel):
    report: str
    says: str
    evidence: list[Evidence] = Field(default_factory=list)


class LLMComparePoint(BaseModel):
    kind: str
    aspect: str = ""
    sides: list[LLMSide] = Field(default_factory=list)


class LLMComparison(BaseModel):
    """What the model must return. Malformed points or unknown kinds are dropped."""

    points: list[LLMComparePoint] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _bare_list(cls, value: object) -> object:
        return _wrap_bare_list(value)

    @field_validator("points", mode="before")
    @classmethod
    def _usable_points(cls, value: object) -> list:
        if not isinstance(value, list):
            return []
        kept = []
        for item in value:
            try:
                point = LLMComparePoint.model_validate(item)
            except ValueError:
                continue
            if point.kind in KINDS and point.sides:
                kept.append(point)
        return kept


class Report(BaseModel):
    """A compared report: its label in the prompt and its excerpt numbers."""

    label: str  # "A", "B", "C"
    doc_id: str
    title: str
    first: int  # excerpt numbers first..last belong to this report
    last: int

    @property
    def read(self) -> int:
        return self.last - self.first + 1


class ComparisonSide(BaseModel):
    doc_id: str
    title: str
    says: str  # with [n] markers of verified excerpts
    sources: list[Source]


class ComparisonPoint(BaseModel):
    kind: str  # agreement, difference, only_in
    aspect: str
    sides: list[ComparisonSide]  # the report of each side is set by code
    note: str | None = None  # only_in: what was read of the other reports (by code)

    @property
    def claims(self) -> str:
        """What the reports say, as the model wrote it: what the judge reads. The
        note ("nothing on this in the N passages read of X") is not a claim of the
        reports but what the code read: judged, it was removed as "in no passage"."""
        head = f"{self.aspect}. " if self.aspect else ""
        return head + " ".join(f"{s.title}: {s.says}" for s in self.sides)

    @computed_field  # type: ignore[prop-decorator]  # pydantic's documented pattern
    @property
    def text(self) -> str:
        """The point as shown: report names written by code, not by the model."""
        return self.claims + (f" {self.note}" if self.note else "")


def absence_note(lang: str, title: str, n: int) -> str:
    templates = ABSENCE_NOTE[lang]
    return (templates[1] if n == 1 else templates["n"]).format(title=title, n=n)


def _rejected_text(point: LLMComparePoint) -> str:
    sides = " | ".join(f"{s.report}: {s.says}" for s in point.sides)
    return f"[{point.kind}] {point.aspect} — {sides}"


def check_comparison(
    reply: LLMComparison,
    passages: list[Passage],
    request: str,
    reports: list[Report],
    lang: str,
) -> tuple[list[ComparisonPoint], list[RejectedPoint]]:
    """Each side checked like a summary point, plus who says what:

    - a side labelled A quotes only excerpts of report A (else: wrong attribution);
    - one side per report; an agreement or a difference has 2+ sides, only_in one.
    """
    by_label = {r.label: r for r in reports}
    kept: list[ComparisonPoint] = []
    rejected: list[RejectedPoint] = []
    for point in reply.points:
        text = _rejected_text(point)

        def reject(reason: str, **details) -> None:
            rejected.append(RejectedPoint(text=text, reason=reason, **details))  # noqa: B023

        labels = [s.report.strip().upper() for s in point.sides]
        if any(label not in by_label for label in labels):
            reject(f"unknown report label in {labels}")
            continue
        if len(set(labels)) != len(labels):
            reject("two sides for the same report")
            continue
        if point.kind in ("agreement", "difference") and len(labels) < 2:
            reject(f"{point.kind}: needs one side per report (has {len(labels)})")
            continue
        if point.kind == "only_in" and len(labels) != 1:
            reject("only_in: one side only (several reports: agreement or difference?)")
            continue
        sides: list[ComparisonSide] = []
        for side, label in zip(point.sides, labels, strict=True):
            report = by_label[label]
            claim = absence_claim(side.says)
            if claim:
                reject(f'side {label}: absence claim ("{claim}")')
                break
            outside = [e.n for e in side.evidence if not report.first <= e.n <= report.last]
            if outside:
                reject(
                    f"side {label} ({report.title}) quotes excerpts {outside} "
                    "of another report: wrong attribution",
                    quotes=[f"[{e.n}] {e.quote}" for e in side.evidence],
                )
                break
            result = check_point(
                LLMPoint(text=side.says, evidence=side.evidence),
                passages,
                f"{request} {point.aspect}",
            )
            if isinstance(result, RejectedPoint):
                rejected.append(result.model_copy(update={"text": f"{text} (side {label})"}))
                break
            sides.append(
                ComparisonSide(
                    doc_id=report.doc_id,
                    title=report.title,
                    says=result.text,
                    sources=result.sources,
                )
            )
        else:
            note = None
            if point.kind == "only_in":
                note = " ".join(
                    absence_note(lang, r.title, r.read) for r in reports if r.label not in labels
                )
            kept.append(
                ComparisonPoint(
                    kind=point.kind, aspect=point.aspect.strip(), sides=sides, note=note
                )
            )
    return kept, rejected
