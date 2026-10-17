"""Talk to the agent in the terminal: one request per line, empty line to quit.

    uv run python scripts/agent_chat.py
    uv run python scripts/agent_chat.py --show-plan      # also print the route chosen
    uv run python scripts/agent_chat.py --fake-translator

Examples:
    Que dit le HCC sur les émissions des transports en 2024 ?
    Résume le rapport du HCC puis compare-le avec le GIEC sur les puits de carbone
    Résume le rapport annuel 2025 du HCC
    Compare l'AIE et le GIEC sur les trajectoires compatibles avec 1,5 °C
    Traduis data/samples/giec_spm_en.pdf en français

The supervisor's model is AGENT_MODEL (default: LLM_MODEL), on AGENT_PROVIDER (same as
the tools, or vertex); the tools use LLM_MODEL. At the end: the supervisor's calls,
tokens and, with AGENT_PRICE_IN_USD / AGENT_PRICE_OUT_USD, an estimated cost.
Each request is saved as a trajectory (AGENT_TRAJECTORIES_DIR, data/trajectories/).
"""

import argparse

from langgraph.types import Command

from docagent.agent import Toolbox
from docagent.agent.checkpoint import memory_checkpointer
from docagent.agent.supervisor import agent_chat_client, build_supervisor
from docagent.agent.trajectory import Recorder
from docagent.chat import ChatClient
from docagent.config import Settings, get_settings
from docagent.translate import FakeTranslator


def usage(chat: ChatClient, settings: Settings) -> str:
    line = (
        f"Supervisor: {chat.answered} call(s) ({chat.requests - chat.answered} refused), "
        f"{chat.input_tokens} input / "
        f"{chat.output_tokens} output tokens"
    )
    price_in, price_out = settings.agent_price_in_usd, settings.agent_price_out_usd
    if price_in is not None and price_out is not None:
        cost = (chat.input_tokens * price_in + chat.output_tokens * price_out) / 1_000_000
        line += f", about ${cost:.4f}"
    return line


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

    try:
        while request := input("\n> ").strip():
            recorder = Recorder(request, supervisor_chat, settings)  # one request, one thread
            try:
                state = recorder.run(graph, {"request": request})
                while "__interrupt__" in state:
                    question = state["__interrupt__"][0].value["question"]
                    state = recorder.run(graph, Command(resume=input(f"{question} [oui/non] ")))
            except Exception as exc:  # a bug in one request must not end the session
                print(f"[error] {type(exc).__name__}: {exc}")
                continue
            if args.show_plan:
                for i, plan in enumerate(state.get("plans") or [], start=1):
                    shown = plan.model_dump(exclude={"model", "prompt_version"})
                    print(f"[plan {i}] {shown}")
            print(state["message"])
            trajectory = recorder.finish(state)
            if settings.agent_trajectories_dir:
                path = trajectory.save(settings.agent_trajectories_dir)
                print(
                    f"[trajectory] {path} · {trajectory.duration_s} s · "
                    f"{trajectory.supervisor_calls} supervisor call(s)"
                )
    finally:
        print(usage(supervisor_chat, settings))


if __name__ == "__main__":
    main()
