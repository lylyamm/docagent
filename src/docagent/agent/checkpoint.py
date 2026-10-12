"""The checkpointer: where the graph keeps its state between two calls.

A pause (``interrupt()``) ends the call; the state is saved here under the
conversation's ``thread_id``, and ``Command(resume=...)`` reloads it.

LangGraph stores the state with msgpack and, for safety, only rebuilds the
Python classes it was told about: the Pydantic results of our tools are listed
here (otherwise a warning today, an error in a future version).
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# Every Pydantic class that can sit in a graph state.
STATE_TYPES = [
    ("docagent.agent.tools", "PdfInfo"),
    ("docagent.agent.tools", "TranslationResult"),
    ("docagent.agent.tools", "DocumentSummary"),
    ("docagent.agent.tools", "DocumentComparison"),
    ("docagent.agent.tools", "ComparedDocument"),
    ("docagent.agent.grounded", "Point"),
    ("docagent.agent.grounded", "RejectedPoint"),
    ("docagent.agent.grounded", "ComparisonPoint"),
    ("docagent.agent.grounded", "ComparisonSide"),
    ("docagent.agent.judge", "Judgement"),
    ("docagent.agent.judge", "ItemVerdict"),
    ("docagent.agent.supervisor", "Plan"),
    ("docagent.rag.answer", "Answer"),
    ("docagent.rag.answer", "Source"),
]


def serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=STATE_TYPES)


def memory_checkpointer() -> InMemorySaver:
    """In memory: enough for one process (script, tests, one API worker).

    The state is lost when the process stops; a database checkpointer
    (SQLite, Postgres) would keep it, for later.
    """
    return InMemorySaver(serde=serializer())
