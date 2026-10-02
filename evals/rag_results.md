
## 2026-09-30 17:27 · embedder - · 4970 passages · 24 questions

| mode | hit@1 | hit@5 | MRR@10 | hit@5 cross-lingual | hit@5 paraphrase | hit@5 exact |
|---|---|---|---|---|---|---|
| bm25 | 33% | 83% | 0.53 | 75% | 89% | 82% |

- missed in the top 5 by bm25: q08, q17, q23, q24

## 2026-09-30 17:28 · embedder hash · 4970 passages · 24 questions

| mode | hit@1 | hit@5 | MRR@10 | hit@5 cross-lingual | hit@5 paraphrase | hit@5 exact |
|---|---|---|---|---|---|---|
| bm25 | 33% | 83% | 0.53 | 75% | 89% | 82% |
| dense | 29% | 54% | 0.40 | 50% | 56% | 55% |
| hybrid | 46% | 75% | 0.58 | 75% | 67% | 82% |

- missed in the top 5 by bm25: q08, q17, q23, q24
- missed in the top 5 by dense: q03, q04, q05, q07, q10, q14, q16, q17, q19, q23, q24
- missed in the top 5 by hybrid: q07, q10, q14, q17, q19, q24

## 2026-09-30 18:23 · embedder mistral · 4970 passages · 24 questions

| mode | hit@1 | hit@5 | MRR@10 | hit@5 cross-lingual | hit@5 paraphrase | hit@5 exact |
|---|---|---|---|---|---|---|
| bm25 | 33% | 83% | 0.53 | 75% | 89% | 82% |
| dense | 25% | 62% | 0.39 | 75% | 67% | 55% |
| hybrid | 46% | 75% | 0.60 | 75% | 89% | 64% |

- missed in the top 5 by bm25: q08, q17, q23, q24
- missed in the top 5 by dense: q03, q06, q07, q11, q17, q18, q21, q22, q23
- missed in the top 5 by hybrid: q01, q03, q07, q17, q21, q23

## 2026-09-30 18:33 · embedder mistral · 4970 passages · 24 questions

| mode | hit@1 | hit@5 | MRR@10 | hit@5 cross-lingual | hit@5 paraphrase | hit@5 exact |
|---|---|---|---|---|---|---|
| bm25 | 33% | 83% | 0.53 | 75% | 89% | 82% |
| dense | 25% | 62% | 0.39 | 75% | 67% | 55% |
| hybrid | 46% | 75% | 0.60 | 75% | 89% | 64% |

- missed in the top 5 by bm25: q08, q17, q23, q24
- missed in the top 5 by dense: q03, q06, q07, q11, q17, q18, q21, q22, q23
- missed in the top 5 by hybrid: q01, q03, q07, q17, q21, q23

## 2026-09-30 18:46 · embedder mistral · 4970 passages · 24 questions

| mode | hit@1 | hit@5 | MRR@10 | hit@5 cross-lingual | hit@5 paraphrase | hit@5 exact |
|---|---|---|---|---|---|---|
| bm25 | 33% | 83% | 0.53 | 75% | 89% | 82% |
| dense | 25% | 62% | 0.39 | 75% | 67% | 55% |
| hybrid | 46% | 75% | 0.60 | 75% | 89% | 64% |

- missed in the top 5 by bm25: q08, q17, q23, q24
- missed in the top 5 by dense: q03, q06, q07, q11, q17, q18, q21, q22, q23
- missed in the top 5 by hybrid: q01, q03, q07, q17, q21, q23

| question | type | bm25 rank | dense rank | dense top 3 (doc p. page) |
|---|---|---|---|---|
| q01 De combien la température à la surface du globe a- | cross-lingual | 2 | 5 | giec-rid-fr p.10, hcc-ra-2025 p.4, giec-rid-fr p.12 |
| q02 How much did global mean sea level rise between 19 | paraphrase | 1 | 2 | ipcc-spm-en p.21, ipcc-spm-en p.5, ipcc-spm-en p.21 |
| q03 Quelle était la concentration de CO2 dans l'atmosp | exact | 2 | 7 | hcc-ra-2025 p.103, giec-rid-fr p.38, hcc-ra-2025 p.5 |
| q04 What were the atmospheric concentrations of methan | exact | 1 | 1 | ipcc-spm-en p.4, ipcc-spm-en p.4, iea-weo-2025 p.343 |
| q05 How much has Arctic sea ice shrunk since the late  | paraphrase | 1 | 2 | ipcc-spm-en p.16, ipcc-spm-en p.5, ipcc-spm-en p.8 |
| q06 Les vagues de chaleur sont-elles devenues plus fré | paraphrase | 2 | 19 | hcc-ra-2025 p.51, hcc-ra-2025 p.53, hcc-ra-2025 p.4 |
| q07 Combien de CO2 d'origine humaine a été émis entre  | exact | 5 | 21 | giec-rid-fr p.37, giec-rid-fr p.38, giec-rid-fr p.37 |
| q08 What does the SSP1-1.9 scenario assume about futur | exact | 12 | 4 | ipcc-spm-en p.13, ipcc-spm-en p.13, ipcc-spm-en p.20 |
| q09 By how much did AR6 revise the remaining carbon bu | exact | 1 | 1 | ipcc-spm-en p.29, ipcc-spm-en p.29, ipcc-spm-en p.28 |
| q10 Qu'appelle-t-on le budget carbone résiduel ? | paraphrase | 3 | 1 | giec-rid-fr p.35, hcc-ra-2025 p.95, hcc-ra-2025 p.95 |
| q11 Combien d'argent sera investi dans les centres de  | paraphrase | 1 | 13 | hcc-ra-2025 p.361, hcc-ra-2025 p.191, hcc-ra-2025 p.126 |
| q12 How much investment is expected in data centres in | exact | 1 | 5 | iea-weo-2025 p.51, iea-weo-2025 p.53, iea-weo-2025 p.52 |
| q13 What are the main scenarios of the World Energy Ou | paraphrase | 3 | 5 | iea-weo-2025 p.448, iea-weo-2025 p.18, iea-weo-2025 p.104 |
| q14 Où iront les nouveaux volumes de gaz naturel liqué | paraphrase | 3 | 1 | aie-weo-2025-resume p.10, aie-weo-2025-resume p.10, aie-weo-2025-resume p.10 |
| q15 Why is the supply of critical minerals a threat to | paraphrase | 2 | 1 | iea-weo-2025 p.18, iea-weo-2025 p.62, iea-weo-2025 p.104 |
| q16 Nucléaire : les petits réacteurs modulaires peuven | cross-lingual | 2 | 2 | aie-weo-2025-resume p.8, aie-weo-2025-resume p.9, aie-weo-2025-resume p.7 |
| q17 Quel est le premier secteur émetteur de gaz à effe | paraphrase | 34 | 12 | hcc-ra-2025 p.84, hcc-ra-2025 p.145, hcc-ra-2025 p.168 |
| q18 De combien les émissions de l'industrie ont-elles  | exact | 1 | 16 | hcc-ra-2025 p.83, hcc-ra-2025 p.81, hcc-ra-2025 p.168 |
| q19 Quelles ont été les émissions du transport routier | exact | 4 | 2 | hcc-ra-2025 p.87, hcc-ra-2025 p.83, hcc-ra-2025 p.115 |
| q20 Le deuxième budget carbone 2019-2023 a-t-il été re | exact | 1 | 3 | hcc-ra-2025 p.91, hcc-ra-2025 p.78, hcc-ra-2025 p.5 |
| q21 Quand le troisième plan national d'adaptation (PNA | exact | 2 | 27 | hcc-ra-2025 p.301, hcc-ra-2025 p.299, hcc-ra-2025 p.374 |
| q22 How much faster must French emissions fall to meet | cross-lingual | 4 | 9 | hcc-ra-2025 p.293, hcc-ra-2025 p.95, hcc-ra-2025 p.287 |
| q23 Quelles sont les émissions du secteur de la produc | exact | 14 | 14 | hcc-ra-2025 p.88, hcc-ra-2025 p.145, hcc-ra-2025 p.202 |
| q24 What is the composition of France's greenhouse gas | cross-lingual | >50 | 1 | hcc-ra-2025 p.83, hcc-ra-2025 p.5, hcc-ra-2025 p.85 |

## 2026-09-30 19:41 · embedder e5 · 4970 passages · 24 questions

| mode | hit@1 | hit@5 | MRR@10 | hit@5 cross-lingual | hit@5 paraphrase | hit@5 exact |
|---|---|---|---|---|---|---|
| bm25 | 33% | 83% | 0.53 | 75% | 89% | 82% |
| dense | 67% | 92% | 0.76 | 75% | 100% | 91% |
| hybrid | 67% | 92% | 0.79 | 100% | 89% | 91% |

- missed in the top 5 by bm25: q08, q17, q23, q24
- missed in the top 5 by dense: q20, q22
- missed in the top 5 by hybrid: q17, q23

| question | type | bm25 rank | dense rank | dense top 3 (doc p. page) |
|---|---|---|---|---|
| q01 De combien la température à la surface du globe a- | cross-lingual | 2 | 1 | giec-rid-fr p.9, giec-rid-fr p.12, giec-rid-fr p.10 |
| q02 How much did global mean sea level rise between 19 | paraphrase | 1 | 1 | ipcc-spm-en p.5, ipcc-spm-en p.25, ipcc-spm-en p.23 |
| q03 Quelle était la concentration de CO2 dans l'atmosp | exact | 2 | 1 | giec-rid-fr p.8, giec-rid-fr p.12, giec-rid-fr p.38 |
| q04 What were the atmospheric concentrations of methan | exact | 1 | 2 | iea-weo-2025 p.343, ipcc-spm-en p.4, ipcc-spm-en p.4 |
| q05 How much has Arctic sea ice shrunk since the late  | paraphrase | 1 | 1 | ipcc-spm-en p.5, ipcc-spm-en p.8, ipcc-spm-en p.16 |
| q06 Les vagues de chaleur sont-elles devenues plus fré | paraphrase | 2 | 1 | giec-rid-fr p.12, hcc-ra-2025 p.51, hcc-ra-2025 p.45 |
| q07 Combien de CO2 d'origine humaine a été émis entre  | exact | 5 | 4 | giec-rid-fr p.37, giec-rid-fr p.38, giec-rid-fr p.37 |
| q08 What does the SSP1-1.9 scenario assume about futur | exact | 12 | 3 | ipcc-spm-en p.20, ipcc-spm-en p.31, ipcc-spm-en p.12 |
| q09 By how much did AR6 revise the remaining carbon bu | exact | 1 | 1 | ipcc-spm-en p.29, ipcc-spm-en p.29, ipcc-spm-en p.27 |
| q10 Qu'appelle-t-on le budget carbone résiduel ? | paraphrase | 3 | 1 | giec-rid-fr p.35, hcc-ra-2025 p.360, giec-rid-fr p.37 |
| q11 Combien d'argent sera investi dans les centres de  | paraphrase | 1 | 1 | aie-weo-2025-resume p.7, iea-weo-2025 p.20, iea-weo-2025 p.51 |
| q12 How much investment is expected in data centres in | exact | 1 | 1 | iea-weo-2025 p.20, iea-weo-2025 p.51, iea-weo-2025 p.52 |
| q13 What are the main scenarios of the World Energy Ou | paraphrase | 3 | 1 | iea-weo-2025 p.17, iea-weo-2025 p.25, iea-weo-2025 p.281 |
| q14 Où iront les nouveaux volumes de gaz naturel liqué | paraphrase | 3 | 1 | aie-weo-2025-resume p.10, hcc-ra-2025 p.383, hcc-ra-2025 p.132 |
| q15 Why is the supply of critical minerals a threat to | paraphrase | 2 | 2 | iea-weo-2025 p.62, iea-weo-2025 p.18, iea-weo-2025 p.27 |
| q16 Nucléaire : les petits réacteurs modulaires peuven | cross-lingual | 2 | 1 | aie-weo-2025-resume p.9, aie-weo-2025-resume p.8, iea-weo-2025 p.366 |
| q17 Quel est le premier secteur émetteur de gaz à effe | paraphrase | 34 | 1 | hcc-ra-2025 p.6, hcc-ra-2025 p.81, hcc-ra-2025 p.78 |
| q18 De combien les émissions de l'industrie ont-elles  | exact | 1 | 1 | hcc-ra-2025 p.7, hcc-ra-2025 p.83, hcc-ra-2025 p.81 |
| q19 Quelles ont été les émissions du transport routier | exact | 4 | 1 | hcc-ra-2025 p.83, hcc-ra-2025 p.112, hcc-ra-2025 p.9 |
| q20 Le deuxième budget carbone 2019-2023 a-t-il été re | exact | 1 | 8 | hcc-ra-2025 p.91, hcc-ra-2025 p.78, hcc-ra-2025 p.118 |
| q21 Quand le troisième plan national d'adaptation (PNA | exact | 2 | 3 | hcc-ra-2025 p.299, hcc-ra-2025 p.43, hcc-ra-2025 p.12 |
| q22 How much faster must French emissions fall to meet | cross-lingual | 4 | 10 | hcc-ra-2025 p.12, hcc-ra-2025 p.272, hcc-ra-2025 p.108 |
| q23 Quelles sont les émissions du secteur de la produc | exact | 14 | 5 | hcc-ra-2025 p.205, hcc-ra-2025 p.81, hcc-ra-2025 p.6 |
| q24 What is the composition of France's greenhouse gas | cross-lingual | >50 | 1 | hcc-ra-2025 p.83, hcc-ra-2025 p.5, hcc-ra-2025 p.80 |

## 2026-10-02 22:35 · embedders e5, qwen3, mistral · 4970 passages · 24 questions

| mode | hit@1 | hit@5 | MRR@10 | hit@5 cross-lingual | hit@5 paraphrase | hit@5 exact |
|---|---|---|---|---|---|---|
| bm25 | 33% | 88% | 0.54 | 75% | 89% | 91% |
| dense e5 | 71% | 92% | 0.80 | 75% | 100% | 91% |
| hybrid e5 | 67% | 96% | 0.80 | 100% | 89% | 100% |
| dense qwen3 | 50% | 67% | 0.56 | 50% | 89% | 55% |
| hybrid qwen3 | 50% | 79% | 0.63 | 50% | 89% | 82% |
| dense mistral | 25% | 67% | 0.41 | 75% | 67% | 64% |
| hybrid mistral | 46% | 79% | 0.60 | 75% | 89% | 73% |

- missed in the top 5 by bm25: q08, q17, q24
- missed in the top 5 by dense e5: q20, q22
- missed in the top 5 by hybrid e5: q17
- missed in the top 5 by dense qwen3: q17, q18, q19, q20, q21, q22, q23, q24
- missed in the top 5 by hybrid qwen3: q17, q19, q22, q23, q24
- missed in the top 5 by dense mistral: q03, q06, q07, q11, q17, q18, q21, q22
- missed in the top 5 by hybrid mistral: q01, q03, q07, q17, q21

Paired tests (A vs B, same questions): questions won by A / by B, two-sided p.

| A vs B | hit@5 won | McNemar p | rank better | sign test p |
|---|---|---|---|---|
| dense e5 vs dense qwen3 | 6 / 0 | 0.031 | 12 / 3 | 0.035 |
| dense e5 vs dense mistral | 7 / 1 | 0.070 | 16 / 4 | 0.012 |
| dense qwen3 vs dense mistral | 4 / 4 | 1.000 | 9 / 6 | 0.607 |
| hybrid e5 vs hybrid qwen3 | 4 / 0 | 0.125 | 10 / 3 | 0.092 |
| hybrid e5 vs hybrid mistral | 4 / 0 | 0.125 | 13 / 4 | 0.049 |
| hybrid qwen3 vs hybrid mistral | 4 / 4 | 1.000 | 9 / 8 | 1.000 |

## 2026-10-02 23:39 · answers · open-mistral-nemo · embedder mistral · top 5

| questions | right page given to the model | answered | cited the right page | cited it, when it was given |
|---|---|---|---|---|
| with an answer (24) | 19/24 | 22/24 | 19/24 | 19/19 |

Without an answer in the reports: refused 5/6 (answered anyway: u06).
Missed: q01, q03, q07, q17, q21. Answers to read: evals/answers/2026-10-02_2339.jsonl
LLM: 30 requests, 42854 tokens in, 2847 out.

## 2026-10-02 23:48 · answers · open-mistral-nemo · embedder e5 · top 5

| questions | right page given to the model | answered | cited the right page | cited it, when it was given |
|---|---|---|---|---|
| with an answer (24) | 23/24 | 24/24 | 22/24 | 22/23 |

Without an answer in the reports: refused 5/6 (answered anyway: u06).
Missed: q01, q17. Answers to read: evals/answers/2026-10-02_2348.jsonl
LLM: 30 requests, 44339 tokens in, 3058 out.

## 2026-10-02 23:56 · answers ask-v2 · open-mistral-nemo · embedder e5 · top 5

| questions | right page given to the model | answered | cited the right page | cited it, when it was given |
|---|---|---|---|---|
| with an answer (24) | 23/24 | 21/24 | 19/24 | 19/23 |

Without an answer in the reports: refused 5/6 (answered anyway: u06).
Missed: q01, q04, q08, q17, q22. Answers to read: evals/answers/2026-10-02_2356.jsonl
LLM: 30 requests, 46709 tokens in, 5903 out.

## 2026-10-03 00:01 · answers ask-v2 · open-mistral-nemo · embedder e5 · top 5

| questions | right page given to the model | answered | cited the right page | cited it, when it was given |
|---|---|---|---|---|
| with an answer (24) | 23/24 | 21/24 | 19/24 | 19/23 |

Without an answer in the reports: refused 5/6 (answered anyway: u06).
Missed: q01, q04, q08, q17, q22. Answers to read: evals/answers/2026-10-03_0001.jsonl
LLM: 30 requests, 46709 tokens in, 5519 out.

## 2026-10-03 00:04 · answers ask-v3 · open-mistral-nemo · embedder e5 · top 5

| questions | right page given to the model | answered | cited the right page | cited it, when it was given |
|---|---|---|---|---|
| with an answer (24) | 23/24 | 17/24 | 16/24 | 16/23 |

Without an answer in the reports: refused 6/6 (answered anyway: none).
Missed: q01, q04, q08, q13, q17, q18, q20, q22. Answers to read: evals/answers/2026-10-03_0004.jsonl
Refused although the right page was given: q04: figures not in a quote (4, 2, 1866, 332); q08: figures not in a quote (2050, 2100); q13: figures not in a quote (2030, 2040); q18: figures not in a quote (7.2, 2022); q20: figures not in a quote (42.2, 240); q22: figures not in a quote (27).
LLM: 30 requests, 47339 tokens in, 5717 out.

## 2026-10-03 00:12 · answers ask-v3.1 · open-mistral-nemo · embedder e5 · top 5

| questions | right page given to the model | answered | cited the right page | cited it, when it was given |
|---|---|---|---|---|
| with an answer (24) | 23/24 | 23/24 | 22/24 | 22/23 |

Without an answer in the reports: refused 6/6 (answered anyway: none).
Missed: q01, q17. Answers to read: evals/answers/2026-10-03_0012.jsonl
Refused although the right page was given: none.
LLM: 30 requests, 47339 tokens in, 5482 out.

## 2026-10-03 00:17 · answers ask-v3.2 · open-mistral-nemo · embedder e5 · top 5

| questions | right page given to the model | answered | cited the right page | cited it, when it was given |
|---|---|---|---|---|
| with an answer (24) | 23/24 | 21/24 | 21/24 | 21/23 |

Without an answer in the reports: refused 6/6 (answered anyway: none).
Missed: q01, q13, q17. Answers to read: evals/answers/2026-10-03_0017.jsonl
Refused although the right page was given: q01: figures not in a quote (1.0); q13: figures not in a quote (2030, 2040).
LLM: 30 requests, 47339 tokens in, 5469 out.
