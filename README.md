# DocAgent

Layout-preserving PDF translation, document search (RAG) and a LangGraph agent, built on a public climate & energy corpus (IPCC, IEA, French High Council on Climate, arXiv).

**Status:** work in progress, phase 1 (PDF core: text extraction and in-place reconstruction).

## Roadmap

1. **PDF core**: extract text blocks and rebuild the PDF with a fake translation, keeping the layout
2. **Translator v1**: LLM translation, FastAPI, Docker, web front end
3. **RAG**: document search with page-level citations
4. **Agent**: LangGraph agent orchestrating translate / search / summarize / compare
5. **Industrialization**: tests, CI/CD, cloud deployment, tracing and cost monitoring
6. **Evaluation**: translation benchmark against official IPCC and IEA French versions

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest

uv run python scripts/explore.py data/samples/hcc_2025_2col.pdf   # inspect PDF structure
uv run python scripts/extract.py                                  # extract text blocks
uv run python scripts/rebuild.py                                  # rebuild with a fake translation

cp .env.example .env   # then choose LLM_PROVIDER and put its API key in .env
uv run python scripts/translate.py data/samples/giec_spm_en.pdf --pages 1 2   # EN -> FR

uv run uvicorn docagent.api.app:serve --factory --reload   # API on http://localhost:8000/docs
```

## Run with Docker

```bash
cp .env.example .env          # choose LLM_PROVIDER, add its key
docker compose up --build     # front: http://localhost:8501 · API: http://localhost:8000/docs
```

Without an LLM key, `TRANSLATOR=fake docker compose up --build` runs the full app with the
fake translation (layout demo).

## API

| Method | Path | |
|---|---|---|
| `POST` | `/v1/detect` | multipart `file` (PDF) → its language, from the PDF metadata or its text |
| `POST` | `/v1/translate` | multipart `file` (PDF), `source_lang` (`auto` by default), `target_lang` → `202 {job_id}` |
| `GET` | `/v1/jobs/{job_id}` | status (`queued`, `running`, `done`, `failed`), page progress, summary |
| `GET` | `/v1/jobs/{job_id}/download` | translated PDF |
| `GET` | `/health` | liveness and configured LLM |

Any OpenAI-compatible LLM works (Mistral, Groq, OpenRouter, local Ollama): see `.env.example`.

Outputs (annotated PDFs, JSON dumps, side-by-side comparisons) go to `data/output/`.

## Project layout

```
src/docagent/pdf/        # extraction (TextBlock) and in-place reconstruction
src/docagent/translate/  # translators: fake (layout tests) and any OpenAI-compatible LLM
src/docagent/api/        # FastAPI app: upload, background jobs, download
front/app.py             # Streamlit front end (upload, progress, side-by-side preview)
scripts/          # exploration and debug scripts
tests/            # pytest suite
data/samples/     # small public PDF excerpts (see data/SOURCES.md)
DECISIONS.md      # design decisions log
```

## License

MIT
