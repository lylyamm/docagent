# Evaluation sets

| File | Role |
|---|---|
| `rag_questions.jsonl` (24) + `rag_unanswerable.jsonl` (6) | **Development set.** Read, analysed and used to fix the search and the answer checks (week 3). Its scores are optimistic. |
| `rag_test_questions.jsonl` (15) + `rag_test_unanswerable.jsonl` (4) | **Test set, kept aside.** Written after the development work, answer pages found by searching the reports for the answer text (every page that states it). Measured once per version; its failures are not used to tune anything. |
| `rag_results.md` | Every run, appended. |
| `answers/` | Every answer of every answer run, to be read. |

Test set run:

    uv run python scripts/rag_eval.py --embedder e5 --questions evals/rag_test_questions.jsonl
    uv run python scripts/rag_eval_answers.py --questions evals/rag_test_questions.jsonl --unanswerable evals/rag_test_unanswerable.jsonl
