"""Try the agent's Translation route from the terminal.

    uv run python scripts/agent_translate.py data/samples/giec_spm_en.pdf
    uv run python scripts/agent_translate.py data/samples/giec_spm_en.pdf --confirm-pages 5
    uv run python scripts/agent_translate.py data/samples/giec_spm_en.pdf --fake

The graph (docagent/agent/graph_translation.py) inspects the PDF, pauses to ask
you when it has more than --confirm-pages pages (default: AGENT_CONFIRM_PAGES,
50), then runs the layout-preserving translator. --fake uses the fake
translator (no LLM call, no key): handy to see the pause and the flow for free.
The translated PDF goes to data/output/.
"""

import argparse
import uuid

from langgraph.types import Command

from docagent.agent import Toolbox
from docagent.agent.checkpoint import memory_checkpointer
from docagent.agent.graph_translation import build_translation_graph
from docagent.config import get_settings
from docagent.translate import FakeTranslator


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pdf")
    parser.add_argument("--to", default="fr", choices=["fr", "en"], help="target language")
    parser.add_argument("--confirm-pages", type=int, help="ask above this many pages")
    parser.add_argument("--fake", action="store_true", help="fake translator, no LLM")
    args = parser.parse_args()

    settings = get_settings()
    if args.confirm_pages is not None:
        settings = settings.model_copy(update={"agent_confirm_pages": args.confirm_pages})
    tools = Toolbox(settings, make_translator=FakeTranslator) if args.fake else Toolbox(settings)
    graph = build_translation_graph(tools, checkpointer=memory_checkpointer())
    # One thread = one conversation: the checkpointer keeps its state between calls.
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}

    state = graph.invoke({"path": args.pdf, "target_lang": args.to}, config)
    while "__interrupt__" in state:  # paused in the "confirm" node
        question = state["__interrupt__"][0].value["question"]
        answer = input(f"{question} [oui/non] ")
        state = graph.invoke(Command(resume=answer), config)
    print(state["message"])


if __name__ == "__main__":
    main()
