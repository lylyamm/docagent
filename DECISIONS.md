# Decisions

A weekly log of what I chose, why, and what I ruled out.

## Week 1 · PDF core

### Tooling
- **Environment managed with uv** (instead of pip + venv or conda): fast installs, a single `pyproject.toml`, and a cross-platform `uv.lock` for reproducible environments.
- **Ruff** for both linting and formatting (replaces flake8 + black + isort with one tool).
- **src/ layout** (`src/docagent/`) so tests run against the installed package, not the working directory.
- **PyMuPDF on Windows needs the Microsoft Visual C++ Redistributable**; without it, importing `pymupdf` fails with `DLL load failed while importing _extra`.

### Test corpus
- **Short excerpts (8–12 pages) instead of full reports** (up to 500 pages / 20 MB): faster iteration and a light repo. Full originals stay in `data/raw/` (git-ignored).
- **Each sample targets one layout challenge**: charts (IPCC), dense tables (IEA), two-column text (HCC, IEEE paper), infographics (HCC public report).
- **Official French versions of IPCC and IEA summaries** kept as reference translations for the evaluation phase.

### Extraction: what `get_text("dict")` actually returns
Explored with `scripts/explore.py` on the 7 samples (debug PDFs in `data/output/`).

- **Blocks follow columns, not sentences.** On the HCC two-column pages, block order is correct (left column, then right), but a paragraph that flows from the bottom of one column to the top of the next is split into two blocks mid-sentence ("…Le rapport annuel du Haut Conseil" / "pour le Climat évalue…"). 25 such blocks in HCC, 54 in the arXiv paper. → Translating block by block would cut sentences; blocks must be merged into paragraphs before translation.
- **Spans are not a usable translation unit.** LaTeX PDFs (arXiv) emit one span per word *and* one per space: 14,321 spans for 1,678 lines, ~50% of them single characters. Inline style changes (footnote markers, bold) also create spans. → Work at line/block level and keep spans only for style (font, size, bold/italic).
- **Tables are exploded into one block per cell** (IEA Annex A: 1,516 numeric-only lines out of 2,087). → Numbers must be detected and left untouched; only row labels and headers get translated.
- **Chart text is real text.** IPCC figures expose axis ticks, bar labels and legends as text blocks, including rotated labels (`line["dir"] != (1, 0)`, 19 lines). → Skip numeric labels; decide separately whether to translate rotated labels.
- **Typography artefacts:** end-of-line hyphenation (153 lines in HCC, 97 in arXiv) and ligature characters such as `ﬃ` (126 lines in HCC). → De-hyphenate and expand ligatures before sending text to the LLM, or translation and search quality will drop.
- **Noise to filter:** the vertical arXiv identifier in the margin, running headers/footers ("Summary for Policymakers", page numbers).

### Extraction: `extract_blocks(page) -> list[TextBlock]`
- **Unit = the PyMuPDF block.** Spans are only used to compute the block's dominant style (font, size, color, bold, italic), weighted by character count, so a few bold words don't make a paragraph "bold".
- **Pydantic `TextBlock`** rather than a dataclass: validation for free (`page >= 0`), and `model_dump()` gives the JSON used for debugging now and for the API later.
- **Text cleaned at extraction time:** ligatures expanded via PyMuPDF flags (`TEXT_PRESERVE_LIGATURES` removed), lines joined with de-hyphenation only when a letter + `-` is followed by a lowercase word ("éco-/systèmes" → "écosystèmes", but "1850-/1900" and "Nord-/Est" keep their hyphen). Raw lines are kept in `TextBlock.lines` for reconstruction.
- **`numeric` and `horizontal` flags** mark blocks that should not be translated (axis ticks, table-only numbers, `2.0 ºC`, rotated labels). Note that IPCC PDFs use `º` (U+00BA), not `°`.
- **Known limitations (to handle later):**
  - Table rows come out as one block with the label *and* the numbers ("Solar 1 8 9 35…"), so only 5 IEA blocks are flagged numeric. Reconstruction will need to work line by line (one line = one cell) for these.
  - Paragraphs split across columns are not merged yet.
  - Superscript footnote markers are glued to the previous word ("climatiqueI").
  - Running headers/footers are still extracted as translatable blocks.

### Reconstruction with a fake translation
- **`fake_translate` = UPPERCASE + 20% longer** (the text repeated and cut at the target length). Uppercase makes rewritten text obvious in the side-by-side PDFs; +20% mimics EN→FR expansion. `--keep-case` gives a test closer to a real translation.
- **Erase with redactions, text only:** `add_redact_annot(fill=False)` + `apply_redactions(images=NONE, graphics=LINE_ART_NONE)`, one rectangle per line. Images, charts and table backgrounds survive.
- **Rewrite with `insert_htmlbox`** with CSS rebuilt from the block's style: size, color, line height, alignment and first-line indent guessed from line edges.

#### Round 2, after a visual review (text too small, wrong font, overlaps)
- **Original fonts are reused when they can draw the text.** Embedded fonts are subsets: `has_glyph()` is not enough because subsetting often keeps the glyph slot but empties its outline (letters came out blank). Each glyph is rendered once and checked for ink; if one character is missing, the whole text falls back to a generic family (mixing fonts inside a word looks worse). With the uppercase test, 54% of zones keep their font (74% with `--keep-case`); many subsets simply have no uppercase glyphs.
- **Text grows into free space before shrinking.** A 1 pt occupancy grid of the page (text lines, bullets, images, drawings) gives the free height below each zone; growth is capped at one extra line-height plus the original height, and never leaves the coloured panel or frame the text sits in.
- **Same style, same scale.** Blocks sharing font, size and weight start from the median scale their group needs, so neighbouring paragraphs don't end up in visibly different sizes. Costs ~3 points of average size, accepted for consistency.
- **Why the text is still smaller:** even rewriting the *identical* text needs ~95% scale on justified paragraphs (the source used hyphenation and tighter word spacing). Measured average size on the 7 samples: 90% identical text, 83% case kept + 20%, 72% uppercase + 20% (65–84% depending on the document).
- **Overlaps fixed at the source:**
  - *Bullets* are encoded as control characters (U+0007 in IPCC and HCC) or private-use glyphs. They are cut out of the text and of the line bbox and kept in place; they used to be erased and redrawn as "�".
  - *Scattered labels* (infographic labels next to bars of different lengths) were treated as a paragraph and reflowed over the bars. A block is now a paragraph only if its lines share an edge or a center (`TextBlock.alignment`); otherwise each line is rewritten in place.
  - *Leading labels* ("A.1.7" before an IPCC paragraph) sat on the first line's band and made the paragraph look like a table row. They are split into their own block and kept.
  - *Formulas:* blocks without a 2-letter word ("b", "t") or set in TeX math fonts (CMMI, CMSY…) are not translated; lines stacked on each other (a symbol and its sub/superscripts) are left untouched; math fragments inside a paragraph are erased with it.
- **Automatic check extended:** `verify.py` now counts pairs of words drawn over each other, before and after. All samples are at 0 new overlaps except the arXiv paper (48 → 50: drop cap and inline math).

#### Round 3, second visual review (lost colours, early line break, confusing padding)
- **Inline styles are kept as markup.** A red lead-in inside a blue paragraph ("Au cours de la dernière décennie," in the HCC report) was drawn in blue, because a block had one style. Runs whose colour, weight, slant or vertical position differ from the block's dominant style are now tagged: `<s1>Au cours de la dernière décennie,</s1> le réchauffement…`, with `TextBlock.styles["s1"]` holding the style. Rebuild turns tags back into `<span>`s (own colour, own embedded font when it can draw the run, `vertical-align` for CO₂ subscripts and footnote markers). `TextBlock.text` stays tag-free for search and checks.
  - Why tags rather than splitting a paragraph into styled pieces: a translation reorders words, so styles must travel *with* the words. In phase 2 the LLM will be asked to keep the tags, as professional CAT tools do. Unknown or unbalanced tags returned by a translator are dropped, never printed.
- **Half a font size of slack on the right**, when free. Boxes are the exact extent of the source glyphs, so a word could miss a line by less than a point. Checked on the reported case ("…UNE DYNAMIQUE S'ÉTAIT / ENCLENCHÉE,"): there the break is legitimate, "ENCLENCHÉE," needs ~60 pt and only ~44 pt were left, because uppercase words are much wider than the source's lowercase.
- **Fake translation padding is lowercase** ("LA FRANCE EST-ELLE… la fra") so it can't be mistaken for duplicated text. Square brackets were tried first but most embedded subsets have no "[" "]" glyph, which forced a fallback font on half the zones.
- **Known limitations:**
  - Inline LaTeX math inside paragraphs is lost; the drop cap of the arXiv paper overlaps the text slightly.
  - Uppercase test text often falls back to a generic font; a real translation (mostly lowercase) will keep the original font far more often.

## Week 2 · Translator v1

### LLM provider layer
- **`Translator` interface with one method, `translate_blocks(texts, source, target)`**: the PDF pipeline doesn't know which backend it talks to. `FakeTranslator` (layout tests) and `ChatTranslator` implement it; a plain `str -> str` function is adapted automatically (handy in tests).
- **One class for every OpenAI-compatible API** (`ChatTranslator`): Mistral, Groq, OpenRouter, Ollama and vLLM all serve `POST /chat/completions` with the same shape. The provider is a setting (`LLM_PROVIDER` preset + optional `LLM_BASE_URL` / `LLM_MODEL` / key). Triggered by a real problem: on day one the Mistral free tier answered `429 Rate limit exceeded` to every request; switching provider must not require code changes.
- **One request per page**, in reading order: the model sees a heading with its paragraph, and a page costs one round trip instead of one per block. Pages longer than `MISTRAL_BATCH_CHARS` (6,000 characters) are split.
- **JSON in, JSON out, validated with Pydantic**: blocks are sent as `{"blocks": [{"id", "text"}]}` and the answer must be `{"translations": [{"id", "text"}]}` (`response_format=json_object`). A missing id or invalid JSON is treated like a network error: retried.
- **Retries with exponential backoff + jitter** on 429, 5xx, timeouts and invalid answers; `Retry-After` is honoured. A 401/400 is not retried (a bad key or a bug won't fix itself). The API's own error message is shown ("Rate limit exceeded"), which made the day-one diagnosis possible.
- **Patient with free tiers**: requests are spaced at least `LLM_MIN_INTERVAL_S` (1.5 s) apart, 429s back off from 5 s up to 60 s, 8 retries (~5 min) before giving up. The first version retried after 1–2 s, which only burned the next attempts inside the same rate-limit window.
- **Style tags are part of the contract**: the prompt asks to keep `<s1>…</s1>` around the right words. If the tags come back broken, the translation is kept without tags (styles are lost, the text isn't) and counted in `tag_fallbacks` (see *First real translation* below for the repair round added since).
- **Config via environment variables** (`pydantic-settings`, `.env` git-ignored, `.env.example` committed). Model is a setting: `mistral-small-latest` by default for fast, cheap iterations; `mistral-medium-latest` to be compared in week 6 on the IPCC official French translation.
- **Optional glossary** (JSON source → target); only the entries present in the page are injected in the prompt, so it doesn't inflate every request.
- **Translation cache** on disk (hash of prompt version, model, language pair and text): re-running a rebuild after a layout change costs no API call. Written atomically.
- **Tests mock the HTTP layer** (`httpx.MockTransport`): retries, invalid JSON, broken tags, batching and a full PDF → API → PDF run are tested without network or cost.
- **`verify_rebuild` now reads what was written from the rebuild report** instead of re-running the translator, which would have paid for every translation twice.

### API (FastAPI)
- **Asynchronous jobs**: `POST /v1/translate` validates the upload and answers `202 {job_id}` immediately; translation runs as a background task (a page can take several seconds with an LLM). `GET /v1/jobs/{id}` gives status and page progress, `GET /v1/jobs/{id}/download` the PDF (409 until done). Chosen over a synchronous endpoint that would hit HTTP timeouts on long reports.
- **Validation before any work**: extension, `%PDF` magic bytes, size (`MAX_UPLOAD_MB`), opens with PyMuPDF, not encrypted, page count (`MAX_PAGES`), known and different languages. Each maps to a precise status code (413, 415, 422).
- **Jobs mirrored to disk** (`.jobs/<id>/job.json`, written atomically): finished jobs survive a restart; jobs interrupted by a restart are marked failed instead of hanging forever as "running". In-memory + files is enough for one instance; a queue (Redis/Celery) would be the next step for several workers.
- **A failing job never crashes the server**: the exception is recorded on the job (`status: failed`, `error`) and logged with its traceback.
- **Structured JSON logs** (one object per line) with `job_id`, pages, zones, duration, check result: ready for a log aggregator in week 5.
- **`TRANSLATOR=fake`** runs the whole API without any key (layout demo, CI, and the week-1 fake translation for visual checks).

### Packaging and front end
- **Multi-stage Dockerfile**: uv resolves the locked dependencies in a builder image; the runtime image only receives the virtual environment (project installed non-editable, so no source tree is needed), runs as a non-root user (uid 1000) and has a `HEALTHCHECK` on `/health`. The dependency layer is cached until `uv.lock` changes.
- **One image, two services** in `docker-compose.yml`: the API and the Streamlit front share the image (different commands), the front waits for the API to be healthy, jobs and the translation cache live in a named volume. `.env` is optional, so `TRANSLATOR=fake` gives a working demo without any key.
- **Streamlit front** (in a separate `front` dependency group, so the API doesn't need it locally): upload, languages, progress bar driven by the job's page counter, summary metrics, download, and page-by-page side-by-side preview rendered with PyMuPDF.
- **Not verified yet**: the image itself was not built in the development sandbox (container registries blocked by its network policy). The install step (`uv sync --frozen --no-dev --group front --no-editable`) and both services were run outside Docker, and the front was driven end to end in a headless browser.

### First real translation (IPCC SPM, pages 1–2, `open-mistral-nemo`)
Review of the result against the English original and the official IPCC French version, by eye and with a replay of the cached answers (no extra API call):
- **HTML printed as text**: the model wrote `CO<sub>2</sub>` and `XXI<sup>e</sup> siècle` on its own. Both are correct typography, so `<sub>`, `<sup>`, `<b>`, `<i>` are now rendered, never shown; the prompt forbids other HTML tags. `verify_rebuild` counts any tag left in the page text (`markup_leaks`) and fails on it.
- **Broken style tags: restore locally, then ask again, then drop.** On the full 8-page SPM, `open-mistral-nemo` lost the tags of 16 of 51 tagged blocks (long paragraphs with several italic terms and footnote calls), and asking it again fixed none. Most runs can be found again without the model (`translate/tags.py`): a footnote call stays glued to its word ("globe8", "2010–201911" located by the year), italic calibrated language goes through the glossary and tolerates inflection ("very likely" → "très probablement"), a bold caption lead-in ends at the same punctuation. Every run must be placed, in order, or nothing is: a style on the wrong words is worse than no style. Replayed on the 16 real failures: 11 restored. The rest (the model dropped the footnote call itself) is sent again with the exact tags each block must contain; what is still broken is kept as plain text and logged to `.cache/tag_failures.jsonl` for analysis.
- **The model also *adds* tags**: `.cache/tag_failures.jsonl` from the second full run showed that most "broken" blocks had extra tags, not missing ones: the model styled glossary terms and acronyms ("le <s1>principal facteur</s1>", "<s1>AR5</s1>"), even in blocks with no tag at all. `prune_tags` keeps the translation runs matching the source's (same footnote number, glossary translation, lead-in position) and turns the others back into plain text; the prompt now forbids tagging untagged words (`PROMPT_VERSION` v3). Replayed on the 19 logged failures: 13 fixed; the rest are footnote calls the model deleted, or generic words ("colours" → "couleurs") nothing can match safely.
- **Codes are never sent to the model**: on the hexagon map (199 labels), `open-mistral-nemo` expanded region codes ("MED" → "Méditerranée", and "SAH" → "Afrique de l'Est", which is wrong). A block that is a code ("MED", "CO2", "SSP1-2.6") is kept as is, and a block that is exactly a glossary entry ("SPM" → "RID") is replaced directly: correct by construction, and fewer tokens.
- **Text is never silently dropped**: when a translation doesn't fit even at `min_scale`, `insert_htmlbox` writes nothing. Those hexagons came out empty and every check still said OK. The text is now written anyway at whatever size fits (still counted as `overflow`), and `verify_rebuild` fails on any rewritten zone left without text (`empty_zones`).
- **Skipped blocks are asked again, alone**: on a chart page with ~75 labels the model stopped before the last ids, and the whole page was retried 3 times. Now a partial answer is kept and only the missing ids are requested again; pages are also cut at `LLM_BATCH_ITEMS` (40) blocks per request.
- **Official terminology**: the model invented "RSM" and "RE6" where the IPCC says "RID" and "AR6". The glossary now holds the terms of the official French SPM (RID, RT, GTI, AR6, *encadré thématique*, calibrated language: *degré de confiance élevé*, *quasi-certain*…). It is picked up automatically per language pair (`data/glossary_en_fr.json`; before, it was only used with `--glossary`), matched on whole words (acronyms case-sensitively, so "TS" doesn't match "results"), and shipped in the Docker image. `PROMPT_VERSION` → v2 so the cache doesn't serve the old answers.
- **Bold and italic read from the font name**: the IPCC fonts carry no bold/italic flag (`FrutigerLTPro-BlackCn`, `FrutigerLTPro-CondensedI`). Italic words (*likely*, *medium confidence*) were not tagged, and the fallback font lost the weight of headers. `font_style()` combines the flags with the name.
- **One-line texts stay on one line**: a running header wrapped ("Résumé à l'intention des / décideurs"). A one-line block is widened sideways, within its text column (so a left-column line never runs into the right column) and the free space, on the side it is anchored to (right-aligned headers grow to the left). It still wraps if one line would need a font below 85%, or if the line is glued to the text below it (a caption split into several blocks).
- **Paragraphs keep their spacing**: growing a paragraph downwards used to fill the blank space before the next one. `KEEP_GAP = 0.6`: 60% of the original gap stays empty; the font is reduced instead. Measured on the samples (fake translation, case kept): mean size unchanged within ±3 points, overlaps equal or lower.
- **Footnote numbers set apart from their text** ("10   Since AR5") are cut out of the line like bullets: kept in place, not translated, and the text keeps its hanging indent. Only digits followed by a gap wider than 0.6 × font size count, so "10 countries" stays text.
- **Known limitation**: English subset fonts lack accented glyphs, so most French text falls back to a generic sans-serif, wider than the condensed IPCC font; it's the main reason text is shrunk. Next step: ship an open condensed font (OFL) as the fallback for condensed originals.

### Source language detection
- **Metadata first, text otherwise**: the PDF catalog's `/Lang` ("en-GB") is the author's own statement and costs nothing, but none of the 7 sample PDFs has one. So the first pages' text (up to 5,000 characters) is classified too, and the metadata is kept unless the text contradicts it with at least 95% confidence (a translation that kept the original's `/Lang`). The API says which source decided (`method`: metadata, text, none) and how sure it is.
- **py3langid** for the text: small n-gram model, deterministic (langdetect is random unless seeded), restricted to the six supported languages; on a few thousand characters it is right on all 7 samples at 100%. lingua is more accurate on short texts, but heavier, and our samples are long. Asking the LLM would cost a request for something a local model does in milliseconds.
- **API**: `POST /v1/detect` for the front end; `source_lang` defaults to `auto` in `POST /v1/translate`. An undetectable document (scanned, only numbers) or one already in the target language is refused with a 422 and a clear message, before any LLM call.
- **Front end**: the source language is pre-selected from the detection ("Detected: Français (text analysis, 100%)") and the target defaults to French, or English for a French document; both stay editable.
