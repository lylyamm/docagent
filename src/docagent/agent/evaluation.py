"""Scoring for the agent's evaluations (scripts/agent_eval_*.py).

Routing: is the plan right? Each case of evals/agent_routing.jsonl lists the plans
it accepts; a plan is right if it matches one of them, step by step:

    {"route": "compare",                               route, or a list of routes
     "reports": [["iea-weo-2025", "aie-weo-2025-resume"],   one group per report:
                 ["giec-rid-fr", "ipcc-spm-en"]],           any id of the group
     "topic_any": ["1,5", "1.5"],                       the topic names one of these
     "path_contains": "IPCC_AR6_WGI_SPM_English",       translation: the file
     "target_lang": "fr"}

The score is on the plan that runs (after the checks in code), not on the
model's raw answer: an unknown report the code turns into a question counts as a
question.

Judge: is a wrong statement removed, a right one kept? evals/judge_traps.jsonl
gives, for real passages of the reports, a faithful statement and a trap (figures
swapped, a rise called a fall...): every figure of a trap is in the passage, so
the checks in code let it through, and only the judge can catch it.
"""

import re
from collections import Counter
from typing import Any

from ..rag.chunking import Passage
from .judge import Item
from .supervisor import Plan

# -- Routing --------------------------------------------------------------------


def step_matches(plan: Plan, spec: dict[str, Any]) -> bool:
    routes = spec["route"] if isinstance(spec["route"], list) else [spec["route"]]
    if plan.route not in routes:
        return False
    groups = spec.get("reports")
    if groups is not None:
        ids = set(plan.doc_ids)
        if len(ids) != len(groups) or not all(ids & set(group) for group in groups):
            return False
    if (words := spec.get("topic_any")) and not any(w.lower() in plan.topic.lower() for w in words):
        return False
    if (part := spec.get("path_contains")) and part.lower() not in plan.path.lower():
        return False
    lang = spec.get("target_lang")
    return not lang or plan.target_lang == lang


def plan_matches(plans: list[Plan], accepted: list[list[dict[str, Any]]]) -> bool:
    """Right if the steps match one of the accepted plans, in order."""
    return any(
        len(plans) == len(steps)
        and all(step_matches(p, s) for p, s in zip(plans, steps, strict=True))
        for steps in accepted
    )


def plan_signature(plans: list[Plan]) -> str:
    """What the plan does, to compare two runs: "summarize(hcc) > compare(giec,hcc)"."""
    parts = []
    for p in plans:
        args = ",".join(sorted(p.doc_ids)) or (p.path.rsplit("/", 1)[-1] if p.path else "")
        parts.append(f"{p.route}({args})")
    return " > ".join(parts)


def routing_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """results: one per (case, run): {"id", "category", "run", "ok", "signature", ...}."""
    by_category: dict[str, list[bool]] = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r["ok"])
    signatures: dict[str, set[str]] = {}
    for r in results:
        signatures.setdefault(r["id"], set()).add(r["signature"])
    runs = len({r["run"] for r in results})
    return {
        "right": sum(r["ok"] for r in results),
        "total": len(results),
        "by_category": {c: (sum(v), len(v)) for c, v in by_category.items()},
        "stable": sum(len(s) == 1 for s in signatures.values()),
        "cases": len(signatures),
        "unstable": sorted(i for i, s in signatures.items() if len(s) > 1) if runs > 1 else [],
        "wrong": sorted({r["id"] for r in results if not r["ok"]}),
        "fixed_by_code": sorted({r["id"] for r in results if r.get("fixed_by_code")}),
    }


# -- Judge ----------------------------------------------------------------------

# A sentence ends with . ! or ? (and maybe a footnote number: "1850–1900.9 Global"),
# then a capital letter starts the next one. "1.09 [" or "0,99 °C" do not split.
SENTENCE_END = re.compile(r"(?<=[.!?])\d{0,2}\s+(?=[A-ZÀ-ÖØ-Þ«(])")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(text) if s.strip()]


class TrapError(ValueError):
    """A case of judge_traps.jsonl does not find its passage or its quote."""


def find_passage(passages: list[Passage], case: dict[str, Any]) -> Passage:
    """The passage of ``doc_id`` (and ``page``) holding every word of every quote."""
    words = [w for group in case["quotes"] for w in group]
    found = [
        p
        for p in passages
        if p.doc_id == case["doc_id"]
        and (case.get("page") is None or p.page == case["page"])
        and all(w in p.text for w in words)
    ]
    if not found:
        raise TrapError(f"{case['id']}: no passage of {case['doc_id']} has {words}")
    return found[0]


def find_quotes(passage: Passage, case: dict[str, Any]) -> list[str]:
    """One quote per group of words: the first sentence that has them all."""
    quotes = []
    for group in case["quotes"]:
        match = next((s for s in sentences(passage.text) if all(w in s for w in group)), None)
        if match is None:
            raise TrapError(f"{case['id']}: no sentence of {passage.id} has all of {group}")
        quotes.append(match)
    return quotes


KINDS = ("true", "trap", "borderline")  # borderline: reported, not scored


def judge_items(passages: list[Passage], case: dict[str, Any]) -> list[tuple[str, Item]]:
    """The items of one case, as the supervisor gives them to the judge."""
    passage = find_passage(passages, case)
    where = f"{passage.title}, p. {passage.page}"
    quotes = [f"{where}: {q}" for q in find_quotes(passage, case)]
    excerpt = f"{where}: {passage.text}"
    return [
        (kind, Item(n=1, statement=case[kind], quotes=quotes, excerpts=[excerpt]))
        for kind in KINDS
        if case.get(kind)
    ]


def judge_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """results: one per (item, run): {"item", "kind", "judged", "removed"}."""
    judged = [r for r in results if r["judged"]]

    def removed(kind: str) -> tuple[int, int]:
        rows = [r for r in judged if r["kind"] == kind]
        return sum(r["removed"] for r in rows), len(rows)

    verdicts: dict[str, Counter] = {}
    for r in judged:
        verdicts.setdefault(r["item"], Counter())[r["removed"]] += 1
    return {
        "traps_caught": removed("trap"),
        "true_removed": removed("true"),
        "borderline_removed": removed("borderline"),
        "constant": sum(len(c) == 1 for c in verdicts.values()),
        "items": len(verdicts),
        "not_judged": len(results) - len(judged),
        "changing": sorted(i for i, c in verdicts.items() if len(c) > 1),
    }
