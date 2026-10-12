"""Talk to the agent in the terminal: one request per line, empty line to quit.

    uv run python scripts/agent_chat.py
    uv run python scripts/agent_chat.py --show-plan      # also print the route chosen
    uv run python scripts/agent_chat.py --fake-translator

Examples:
    Que dit le HCC sur les émissions des transports en 2024 ?
    Résume le rapport annuel 2025 du HCC
    Compare l'AIE et le GIEC sur les trajectoires compatibles avec 1,5 °C
    Traduis data/samples/giec_spm_en.pdf en français

The supervisor's model is AGENT_MODEL (default: LLM_MODEL); the tools use LLM_MODEL.
"""

import argparse
import uuid

from langgraph.types import Command

from docagent.agent import Toolbox
from docagent.agent.checkpoint import memory_checkpointer
from docagent.agent.supervisor import agent_chat_client, build_supervisor
from docagent.config import get_settings
from docagent.translate import FakeTranslator


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--show-plan", action="store_true", help="print the route and parameters")
    parser.add_argument("--fake-translator", action="store_true", help="translate without LLM")
    args = parser.parse_args()

    settings = get_settings()
    tools = (
        Toolbox(settings, make_translator=FakeTranslator)
        if args.fake_translator
        else Toolbox(settings)
    )
    supervisor_chat = agent_chat_client(settings)
    graph = build_supervisor(tools, supervisor_chat, checkpointer=memory_checkpointer())
    print(f"Supervisor: {supervisor_chat.model} · tools: {tools.chat.model}. Empty line to quit.")

    while request := input("\n> ").strip():
        config = {"configurable": {"thread_id": uuid.uuid4().hex}}  # one request, one thread
        state = graph.invoke({"request": request}, config)
        while "__interrupt__" in state:
            question = state["__interrupt__"][0].value["question"]
            state = graph.invoke(Command(resume=input(f"{question} [oui/non] ")), config)
        if args.show_plan and state.get("plan"):
            print(f"[plan] {state['plan'].model_dump(exclude={'model', 'prompt_version'})}")
        print(state["message"])


if __name__ == "__main__":
    main()
