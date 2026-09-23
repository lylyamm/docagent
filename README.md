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
```

## Project layout

```
src/docagent/     # package source
scripts/          # exploration and debug scripts
tests/            # pytest suite
data/samples/     # small public PDF excerpts (see data/SOURCES.md)
DECISIONS.md      # design decisions log
```

## License

MIT
