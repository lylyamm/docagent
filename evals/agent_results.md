# Agent evaluation results

Every run of the agent's evaluations, appended by the scripts:

- `scripts/agent_eval_routing.py`: the router alone, on `agent_routing.jsonl` (25 requests);
- `scripts/agent_eval_judge.py`: the judge alone, on `judge_traps.jsonl` (faithful statements and traps);
- `scripts/agent_eval_runs.py`: the whole agent on 4 requests (duration, cost).

Details of each run (every plan, every verdict) in `agent_runs/`.

## 2026-10-21 18:35 · routing router-v3 · google/gemini-2.5-pro · 25 requests × 2 runs

| requests | right plan |
|---|---|
| question | 10/10 (100 %) |
| summary | 5/6 (83 %) |
| comparison | 6/6 (100 %) |
| translation | 4/4 (100 %) |
| several steps | 8/8 (100 %) |
| ambiguous (2 plans accepted) | 6/6 (100 %) |
| unclear → question back | 4/4 (100 %) |
| out of scope | 6/6 (100 %) |
| **all** | **49/50 (98 %)** |

Same plan on every run: 21/25 (84 %) (changed: a1, c1, m3, s3).
Wrong: 
- s3 (run 2): clarify()
Router call: median 9.9 s, max 180.8 s.
Supervisor: 50 calls, 43520 tokens in, 4006 out, about $0.0945 ($0.00189 per request).
Plans: evals/agent_runs/routing_2026-10-21_1835.json

## 2026-10-21 18:41 · judge judge-v2 · google/gemini-2.5-pro · 11 statements × 3 runs

| statements | removed by the judge | expected |
|---|---|---|
| traps (wrong meaning, right figures) | 15/15 (100 %) | all |
| faithful | 0/15 (0 %) | none |
| borderline (a nuance left out) | 2/3 (67 %) | - |

Same verdict on every run: 10/11 (changed: B-borderline). Reviews that failed (nothing removed): 0.
Judge call: median 9.0 s. Supervisor: 33 calls, 21561 tokens in, 1831 out, about $0.0453.
Verdicts: evals/agent_runs/judge_2026-10-21_1841.json

## 2026-10-21 18:45 · whole agent · supervisor google/gemini-2.5-pro · tools open-mistral-nemo

| # | request | steps | duration | of which tools | supervisor calls | cost | points kept by the judge |
|---|---|---|---|---|---|---|---|
| 1 | Que dit le HCC sur les émissions des transports en 2024 ? | question | 69 s | 53 s | 2 | $0.0035 | 1/1 |
| 2 | Résume le rapport annuel 2025 du HCC | summarize | 45 s | 10 s | 2 | $0.0066 | 4/5 |
| 3 | Compare l'AIE et le GIEC sur les trajectoires compatibles avec 1,5 °C | compare | 54 s | 15 s | 2 | $0.0056 | 0/2 |
| 4 | Résume le rapport du HCC puis compare-le avec le GIEC sur les puits de carbone | summarize > compare | 84 s | 19 s | 3 | $0.0099 | 2/4, 1/2 |

Cost: the supervisor only (the tools' model is billed apart). Trajectories: 20261021-184124_64a3c9ba.json, 20261021-184234_ecc6bb21.json, 20261021-184321_75a49a94.json, 20261021-184416_4fd4c3ee.json

## 2026-10-21 19:32 · whole agent · supervisor google/gemini-2.5-pro · tools open-mistral-nemo

| # | request | steps | duration | of which tools | supervisor calls | cost | points kept by the judge |
|---|---|---|---|---|---|---|---|
| 3 | Compare l'AIE et le GIEC sur les trajectoires compatibles avec 1,5 °C | compare | 103 s | 97 s | 1 | $0.0019 | not reviewed |

Cost: the supervisor only (the tools' model is billed apart). Trajectories: 20261021-193058_7a0854d8.json

## 2026-10-21 21:34 · whole agent · supervisor google/gemini-2.5-pro · tools open-mistral-nemo

| # | request | steps | duration | of which tools | supervisor calls | cost | points kept by the judge |
|---|---|---|---|---|---|---|---|
| 3 | Compare l'AIE et le GIEC sur les trajectoires compatibles avec 1,5 °C | compare | 76 s | 30 s | 2 | $0.0062 | 2/3 |

Cost: the supervisor only (the tools' model is billed apart). Trajectories: 20261021-213318_9d164cd1.json

## 2026-10-21 22:15 · routing router-v3 · google/gemini-2.5-pro · 25 requests × 2 runs

| requests | right plan |
|---|---|
| question | 10/10 (100 %) |
| summary | 5/6 (83 %) |
| comparison | 6/6 (100 %) |
| translation | 4/4 (100 %) |
| several steps | 8/8 (100 %) |
| ambiguous (2 plans accepted) | 6/6 (100 %) |
| unclear → question back | 4/4 (100 %) |
| out of scope | 6/6 (100 %) |
| **all** | **49/50 (98 %)** |

Same plan on every run: 22/25 (88 %) (changed: a1, c1, s3).
Wrong: 
- s3 (run 1): clarify()
Router call: median 7.6 s, max 35.0 s.
Supervisor: 50 calls, 43520 tokens in, 4321 out, about $0.0976 ($0.00195 per request).
Plans: evals/agent_runs/routing_2026-10-21_2215.json

## 2026-10-21 23:35 · routing router-v4 · google/gemini-2.5-pro · 25 requests × 2 runs

| requests | right plan |
|---|---|
| question | 10/10 (100 %) |
| summary | 6/6 (100 %) |
| comparison | 6/6 (100 %) |
| translation | 4/4 (100 %) |
| several steps | 8/8 (100 %) |
| ambiguous (2 plans accepted) | 6/6 (100 %) |
| unclear → question back | 4/4 (100 %) |
| out of scope | 6/6 (100 %) |
| **all** | **50/50 (100 %)** |

Same plan on every run: 25/25 (100 %).
Wrong: none.
Router call: median 5.9 s, max 29.4 s.
Supervisor: 50 calls, 46920 tokens in, 4040 out, about $0.0990 ($0.00198 per request).
Plans: evals/agent_runs/routing_2026-10-21_2335.json

## 2026-10-21 23:46 · whole agent · supervisor google/gemini-2.5-pro · tools open-mistral-nemo

| # | request | steps | duration | of which tools | supervisor calls | cost | points kept by the judge |
|---|---|---|---|---|---|---|---|
| 1 | Que dit le HCC sur les émissions des transports en 2024 ? | question | 73 s | 64 s | 1 | $0.0020 | - |
| 2 | Résume le rapport annuel 2025 du HCC | summarize | 55 s | 10 s | 2 | $0.0059 | 3/4 |
| 3 | Compare l'AIE et le GIEC sur les trajectoires compatibles avec 1,5 °C | compare | 41 s | 7 s | 2 | $0.0049 | 1/2 |
| 4 | Résume le rapport du HCC puis compare-le avec le GIEC sur les puits de carbone | summarize > compare | 86 s | 22 s | 3 | $0.0092 | 1/3, 1/2 |

Cost: the supervisor only (the tools' model is billed apart). Trajectories: 20261021-234225_7d3790bb.json, 20261021-234339_ed0e6bcf.json, 20261021-234435_2ce7907a.json, 20261021-234517_cc3cb569.json

## 2026-10-21 23:56 · whole agent · supervisor google/gemini-2.5-pro · tools open-mistral-nemo

| # | request | steps | duration | of which tools | supervisor calls | cost | points kept by the judge |
|---|---|---|---|---|---|---|---|
| 1 | Que dit le HCC sur les émissions des transports en 2024 ? | question | 55 s | 17 s | 2 | $0.0042 | 0/1 |

Cost: the supervisor only (the tools' model is billed apart). Trajectories: 20261021-235551_1d74d151.json

## 2026-10-22 10:31 · whole agent · supervisor google/gemini-2.5-pro · tools open-mistral-nemo

| # | request | steps | duration | of which tools | supervisor calls | cost | points kept by the judge |
|---|---|---|---|---|---|---|---|
| 1 | Que dit le HCC sur les émissions des transports en 2024 ? | question | 139 s | 133 s | 1 | $0.0020 | not answered: figures not in the quotes: 1.6, 2050 |
| 2 | Résume le rapport annuel 2025 du HCC | summarize | 98 s | 12 s | 2 | $0.0064 | 1/4 |
| 3 | Compare l'AIE et le GIEC sur les trajectoires compatibles avec 1,5 °C | compare | 64 s | 8 s | 2 | $0.0055 | 0/2 |
| 4 | Résume le rapport du HCC puis compare-le avec le GIEC sur les puits de carbone | summarize > compare | 491 s | 26 s | 3 | $0.0088 | 2/3, 0/2 |

Cost: the supervisor only (the tools' model is billed apart). Trajectories: 20261022-101840_524a5696.json, 20261022-102101_53592b6d.json, 20261022-102241_22359420.json, 20261022-102346_05012fd6.json

## 2026-10-22 12:04 · whole agent · supervisor google/gemini-2.5-pro · tools google/gemini-2.5-flash

| # | request | steps | duration | of which tools | supervisor calls | supervisor cost | tools cost | points kept by the judge |
|---|---|---|---|---|---|---|---|---|
| 1 | Que dit le HCC sur les émissions des transports en 2024 ? | question | 90 s | 63 s | 2 | $0.0039 | $0.0022 | 1/1 |
| 2 | Résume le rapport annuel 2025 du HCC | summarize | 49 s | 20 s | 2 | $0.0070 | $0.0083 | 5/5 |
| 3 | Compare l'AIE et le GIEC sur les trajectoires compatibles avec 1,5 °C | compare | 61 s | 18 s | 2 | $0.0049 | $0.0035 | 1/2 |
| 4 | Résume le rapport du HCC puis compare-le avec le GIEC sur les puits de carbone | summarize > compare | 92 s | 47 s | 3 | $0.0141 | $0.0131 | 4/5, 3/3 |

Points and answers kept by the review: 14 / 16. Tools cost: with TOOLS_PRICE_IN_USD / TOOLS_PRICE_OUT_USD ("-": not set, or a free tier). Trajectories: 20261022-115927_48d76a7e.json, 20261022-120058_31bacdd0.json, 20261022-120147_17afa4f6.json, 20261022-120250_85bbc131.json

## 2026-10-22 12:50 · judge judge-v3 · google/gemini-2.5-pro · 11 statements × 3 runs

| statements | removed by the judge | expected |
|---|---|---|
| traps (wrong meaning, right figures) | 15/15 (100 %) | all |
| faithful | 1/15 (7 %) | none |
| borderline (a nuance left out) | 3/3 (100 %) | - |

Same verdict on every run: 10/11 (changed: B-true). Reviews that failed (nothing removed): 0.
Faithful statements removed at least once: B-true.
Judge call: median 9.8 s. Supervisor: 33 calls, 22815 tokens in, 1979 out, about $0.0483.
Verdicts: evals/agent_runs/judge_2026-10-22_1250.json

## 2026-10-22 12:55 · whole agent · supervisor google/gemini-2.5-pro · tools google/gemini-2.5-flash

| # | request | steps | duration | of which tools | supervisor calls | supervisor cost | tools cost | points kept by the judge |
|---|---|---|---|---|---|---|---|---|
| 1 | Que dit le HCC sur les émissions des transports en 2024 ? | question | 68 s | 33 s | 2 | $0.0043 | $0.0022 | 1/1 |
| 2 | Résume le rapport annuel 2025 du HCC | summarize | 56 s | 24 s | 2 | $0.0084 | $0.0089 | 4/5 |
| 3 | Compare l'AIE et le GIEC sur les trajectoires compatibles avec 1,5 °C | compare | 44 s | 23 s | 2 | $0.0047 | $0.0035 | 2/2 |
| 4 | Résume le rapport du HCC puis compare-le avec le GIEC sur les puits de carbone | summarize > compare | 113 s | 48 s | 3 | $0.0149 | $0.0133 | 4/5, 3/3 |

Points and answers kept by the review: 14 / 16. Tools cost: with TOOLS_PRICE_IN_USD / TOOLS_PRICE_OUT_USD ("-": not set, or a free tier). Trajectories: 20261022-125029_08bfcd33.json, 20261022-125138_6f1f4e68.json, 20261022-125235_2b6bc41e.json, 20261022-125319_52712e3f.json

## 2026-10-22 13:11 · judge judge-v3 · google/gemini-2.5-pro · 11 statements × 3 runs

| statements | removed by the judge | expected |
|---|---|---|
| traps (wrong meaning, right figures) | 15/15 (100 %) | all |
| faithful | 0/15 (0 %) | none |
| borderline (a nuance left out) | 3/3 (100 %) | - |

Same verdict on every run: 11/11. Reviews that failed (nothing removed): 0.
Judge call: median 8.7 s. Supervisor: 33 calls, 22821 tokens in, 1955 out, about $0.0481.
Verdicts: evals/agent_runs/judge_2026-10-22_1311.json
