# Evaluation sets

| File | Role |
|---|---|
| `rag_questions.jsonl` (24) + `rag_unanswerable.jsonl` (6) | **Development set.** Read, analysed and used to fix the search and the answer checks (week 3). Its scores are optimistic. |
| `rag_test_questions.jsonl` (15) + `rag_test_unanswerable.jsonl` (4) | **Test set, kept aside.** Written after the development work, answer pages found by searching the reports for the answer text (every page that states it). Measured once per version; its failures are not used to tune anything. |
| `rag_results.md` | Every run, appended. |
| `answers/` | Every answer of every answer run, to be read. |
| `agent_routing.jsonl` (25) | **Agent routing.** Requests and the plans each one accepts (route, reports, topic, file), in 8 kinds: question, summary, comparison, translation, several steps, ambiguous (2 plans accepted), unclear, out of scope. Written once router-v3 was done; q1 and o1 are also examples of the router's prompt. **Development set since router-v4**: its v3 run (s3, c1, m3: which version of the WEO) was used to write v4's rule, so its v4 scores are optimistic. |
| `judge_traps.jsonl` (5 passages) | **Agent judge.** For real passages of the reports: a faithful statement, a trap (same figures, wrong meaning) and, for one, a borderline statement. |
| `agent_results.md` | Every agent evaluation run, appended. |
| `agent_runs/` | Every plan and every verdict of these runs. |

Test set run:

    uv run python scripts/rag_eval.py --embedder e5 --questions evals/rag_test_questions.jsonl
    uv run python scripts/rag_eval_answers.py --questions evals/rag_test_questions.jsonl --unanswerable evals/rag_test_unanswerable.jsonl

Agent runs (supervisor: AGENT_MODEL on AGENT_PROVIDER):

    uv run python scripts/agent_eval_routing.py     # router alone, 25 requests × 2
    uv run python scripts/agent_eval_judge.py --show  # check the quotes found, no API call
    uv run python scripts/agent_eval_judge.py       # judge alone, 11 statements × 3
    uv run python scripts/agent_eval_runs.py        # whole agent, 4 requests
