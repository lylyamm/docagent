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
