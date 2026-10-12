"""Can the model choose tools? A 2-minute check before writing the agent.

    uv run python scripts/agent_check_tools.py
    uv run python scripts/agent_check_tools.py --model mistral-small-latest
    uv run python scripts/agent_check_tools.py --raw     # print the JSON exchanged

"Function calling": we describe tools to the model (name, purpose, JSON schema
of the arguments) and, instead of text, it may answer with a *tool call*:

    request   messages + tools=[{name, description, parameters}, ...]
    response  {"role": "assistant", "tool_calls": [{"function": {"name": "search_reports",
               "arguments": "{\\"query\\": \\"...\\"}"}}]}

The code runs the tool, sends its result back as a {"role": "tool"} message,
and the model writes the final answer (or calls another tool: that loop is the
ReAct agent). Only the ingestion agent of DocAgent works this way; the
Translation and Questions workflows call their tools in a fixed order.

Each case below says which tool a good model should pick. Six cases are not a
benchmark: the point is to see the mechanism and spot a model that cannot do it.
Uses LLM_PROVIDER / LLM_MODEL from .env (the key is read by the settings, never
printed). About 8 requests.
"""

import argparse
import json
import time

import httpx

from docagent.config import get_settings

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_reports",
            "description": (
                "Search the indexed climate and energy reports (IPCC AR6 WG1 summary, IEA "
                "World Energy Outlook 2025, Haut Conseil pour le climat 2025) and return the "
                "most relevant passages with their page."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "translate_pdf",
            "description": "Translate a PDF file, keeping its layout.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path of the PDF file"},
                    "target_lang": {"type": "string", "enum": ["fr", "en"]},
                },
                "required": ["path", "target_lang"],
            },
        },
    },
]

SYSTEM = (
    "You are DocAgent's assistant. Use the tools when the request needs them; answer "
    "directly when it does not. Never invent the content of a report."
)

# (request, expected first tool or None for a direct answer, expected arguments subset)
CASES = [
    ("Quelle hausse du niveau de la mer entre 1901 et 2018 selon le GIEC ?", "search_reports", {}),
    ("What does the IEA expect for data centre investment in 2025?", "search_reports", {}),
    ("Traduis data/raw/HCC_RA_2025.pdf en anglais", "translate_pdf", {"target_lang": "en"}),
    ("Bonjour ! Tu sais faire quoi ?", None, {}),
    ("Quelle est la capitale de l'Australie ?", None, {}),
    (
        "Traduis data/raw/WEO2025.pdf en français puis dis-moi ce qu'il prévoit pour le solaire",
        "translate_pdf",
        {"target_lang": "fr"},
    ),
]

# A fake search result, to see the second half of the loop (tool result -> answer).
FAKE_RESULT = (
    "[1] (IPCC AR6 WG1 SPM, p. 5) Global mean sea level increased by 0.20 [0.15 to 0.25] m "
    "between 1901 and 2018."
)


def post(client: httpx.Client, url: str, headers: dict, body: dict, raw: bool) -> dict:
    for attempt in range(4):
        response = client.post(url, headers=headers, json=body)
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(5 * 2**attempt)  # free tier: requests per minute are limited
            continue
        response.raise_for_status()
        data = response.json()
        if raw:
            print(json.dumps(data["choices"][0]["message"], ensure_ascii=False, indent=2))
        time.sleep(1.5)
        return data["choices"][0]["message"]
    raise RuntimeError(f"HTTP {response.status_code} after 4 attempts")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", help="default: LLM_MODEL from .env")
    parser.add_argument("--raw", action="store_true", help="print the raw assistant messages")
    args = parser.parse_args()

    settings = get_settings()
    model = args.model or settings.resolved_model
    url = f"{settings.resolved_base_url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    key = settings.resolved_api_key
    if key is not None:
        headers["Authorization"] = f"Bearer {key.get_secret_value()}"
    client = httpx.Client(timeout=settings.llm_timeout_s)
    print(f"model: {model}\n")

    good = 0
    for request, expected, expected_args in CASES:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": request}]
        body = {"model": model, "messages": messages, "tools": TOOLS, "tool_choice": "auto"}
        reply = post(client, url, headers, {**body, "temperature": 0}, args.raw)
        calls = reply.get("tool_calls") or []
        first = calls[0]["function"] if calls else None
        name = first["name"] if first else None
        try:
            arguments = json.loads(first["arguments"]) if first else {}
        except json.JSONDecodeError:
            arguments = {"<invalid JSON>": first["arguments"]}
        ok = name == expected and all(arguments.get(k) == v for k, v in expected_args.items())
        good += ok
        print(f"{'OK ' if ok else 'NO '} {request}")
        wanted = expected or "direct answer"
        print(f"     expected: {wanted} {expected_args or ''}")
        if calls:
            shown = ", ".join(
                f"{c['function']['name']}({c['function']['arguments']})" for c in calls
            )
            print(f"     got:      {shown}")
        else:
            print(f"     got:      direct answer: {(reply.get('content') or '')[:120]!r}")
        print()

    print(f"{good}/{len(CASES)} as expected\n")

    # Second half of the loop: give the tool result back, the model answers.
    print("Full loop on the first case (tool call -> fake result -> answer):")
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": CASES[0][0]}]
    body = {"model": model, "messages": messages, "tools": TOOLS, "temperature": 0}
    reply = post(client, url, headers, body, args.raw)
    calls = reply.get("tool_calls") or []
    if not calls:
        print("  no tool call, loop not tested")
        return
    messages.append(
        {"role": "assistant", "content": reply.get("content") or "", "tool_calls": calls}
    )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": calls[0]["id"],
            "name": calls[0]["function"]["name"],
            "content": FAKE_RESULT,
        }
    )
    final = post(client, url, headers, {**body, "messages": messages}, args.raw)
    print(f"  answer: {final.get('content')}")


if __name__ == "__main__":
    main()
