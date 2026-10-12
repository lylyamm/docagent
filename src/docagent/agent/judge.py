"""The supervisor's second call: is each point faithful to its own quotes?

The tools already check, in code, that every quote is in its excerpt and that
every figure of a point appears in the cited excerpts. What code cannot check is
meaning: "rainfall rose by 15 %" passes when the quote says that *heatwaves*
rose by 15 %. The judge reads each point next to its quotes and says whether
the point says the same thing.

The judge can only take away, never add: a point it finds unfaithful is
removed, a refused answer stays refused, and nothing it writes is shown as a
fact. If the judge itself fails (no valid answer), the points are kept and
marked "not judged": the code checks still hold.
"""

from pydantic import BaseModel, Field, field_validator

from ..chat import ChatClient, ChatError

# v2: judges only real errors (omissions and missing nuance are not), sees the whole
# cited passages, answers in the user's language. v1 (nemo, HCC 2025 summary)
# removed 4 points of 5, for omissions ("does not mention SSP5-8.5") or for a figure
# present in the passage but not in the quoted sentence.
JUDGE_PROMPT_VERSION = "judge-v2"

JUDGE_SYSTEM_PROMPT = """You review statements written from climate and energy reports.

You get a request and numbered items. Each item has a statement, the quotes chosen to \
support it, and the whole passages those quotes come from.

An item is NOT faithful only if you can point to one of these errors:
- a figure, date, place or scenario attached to something else than in the passages \
(e.g. "rainfall +15 %" when the passage says "heatwaves +15 %");
- a fact or a figure that is in none of the passages;
- a meaning the passages contradict (a rise called a fall, "possible" called "certain");
- something attributed to the wrong report.
These are NOT errors: leaving out details, context or scenarios; paraphrasing; \
summarising; translating; a figure taken from the passage rather than from the quote.
When you cannot name the error precisely, the item is faithful. Judge each item alone, \
and do not judge whether the reports are right.

Write "problem" in {language}, in one sentence, naming the error.

Answer with JSON only:
{{"items": [{{"n": <item number>, "faithful": true | false, "problem": "<empty if faithful>"}}]}}"""

LANGUAGES = {"fr": "French", "en": "English"}


class Item(BaseModel):
    """What the judge reads: one statement and the quotes that support it."""

    n: int
    statement: str
    quotes: list[str]  # "Title, p. 12: <quote>"
    excerpts: list[str] = Field(default_factory=list)  # the whole cited passages


class ItemVerdict(BaseModel):
    n: int
    faithful: bool
    problem: str = ""


class LLMVerdicts(BaseModel):
    items: list[ItemVerdict] = Field(default_factory=list)

    @field_validator("items", mode="before")
    @classmethod
    def _usable(cls, value: object) -> list:
        if not isinstance(value, list):
            return []
        kept = []
        for item in value:
            try:
                kept.append(ItemVerdict.model_validate(item))
            except ValueError:
                continue
        return kept


class Judgement(BaseModel):
    verdicts: dict[int, ItemVerdict]  # by item number; missing = not judged
    judged: bool  # False: the judge failed, nothing was removed
    model: str
    prompt_version: str = JUDGE_PROMPT_VERSION

    def faithful(self, n: int) -> bool:
        """Kept unless the judge said it is not faithful."""
        verdict = self.verdicts.get(n)
        return verdict is None or verdict.faithful


EXCERPT_CHARS = 1500  # per passage: enough context, a bounded prompt


def items_message(request: str, items: list[Item]) -> str:
    blocks = [f"Request: {request}", ""]
    for item in items:
        blocks.append(f"[{item.n}] Statement: {item.statement}")
        blocks += [f"    Quote: {q}" for q in item.quotes]
        blocks += [f"    Passage: {e[:EXCERPT_CHARS]}" for e in item.excerpts]
        blocks.append("")
    return "\n".join(blocks).rstrip()


def judge(chat: ChatClient, request: str, items: list[Item], lang: str = "fr") -> Judgement:
    if not items:
        return Judgement(verdicts={}, judged=True, model=chat.model)
    system = JUDGE_SYSTEM_PROMPT.format(language=LANGUAGES.get(lang, "French"))
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": items_message(request, items)},
    ]
    try:
        reply = chat.complete_json(messages, LLMVerdicts)
    except ChatError:
        return Judgement(verdicts={}, judged=False, model=chat.model)
    known = {item.n for item in items}
    verdicts = {v.n: v for v in reply.items if v.n in known}
    return Judgement(verdicts=verdicts, judged=True, model=chat.model)
