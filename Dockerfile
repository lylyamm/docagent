# syntax=docker/dockerfile:1
# Multi-stage build: dependencies are resolved with uv in a builder image, and
# only the virtual environment (and the embedding model) are copied into a slim
# runtime image that runs as a non-root user.

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first: this layer is cached as long as uv.lock doesn't change.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group front --no-install-project

# Local embedding model (questions on the reports). The lock pins torch from
# PyPI, whose Linux wheel brings ~3 GB of CUDA libraries a CPU container never
# uses: install the CPU build of the same version instead, then the rest of the
# "local" extra at the locked versions, without torch and its CUDA packages.
RUN --mount=type=cache,target=/root/.cache/uv \
    TORCH=$(sed -n '/^name = "torch"$/{n;s/version = "\(.*\)"/\1/p}' uv.lock) \
    && uv pip install --python .venv/bin/python "torch==${TORCH}" \
        --index-url https://download.pytorch.org/whl/cpu \
    && uv export --frozen --no-dev --extra local --no-hashes --no-emit-project \
        | grep -vE '^(torch|triton|nvidia-|cuda-)' > /tmp/local.txt \
    && uv pip install --python .venv/bin/python -r /tmp/local.txt

# The model itself (~1.1 GB), downloaded once at build time: the container needs
# no network to answer questions.
ENV HF_HOME=/opt/hf
RUN .venv/bin/python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('intfloat/multilingual-e5-base')"

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group front --no-editable --inexact

FROM python:3.11-slim-bookworm AS runtime
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /data /index && chown app:app /data /index
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /opt/hf /opt/hf
COPY --chown=app:app front ./front
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app data/glossary_*.json data/corpus.json ./data/
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    JOBS_DIR=/data/jobs \
    TRANSLATION_CACHE=/data/cache/translations.json \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    RAG_INDEX_DIR=/index
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "docagent.api.app:serve", "--factory", "--host", "0.0.0.0", "--port", "8000"]
