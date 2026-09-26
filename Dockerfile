# syntax=docker/dockerfile:1
# Multi-stage build: dependencies are resolved with uv in a builder image, and
# only the virtual environment is copied into a slim runtime image that runs
# as a non-root user.

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first: this layer is cached as long as uv.lock doesn't change.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group front --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group front --no-editable

FROM python:3.11-slim-bookworm AS runtime
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /data && chown app:app /data
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app front ./front
COPY --chown=app:app data/glossary_*.json ./data/
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    JOBS_DIR=/data/jobs \
    TRANSLATION_CACHE=/data/cache/translations.json
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "docagent.api.app:serve", "--factory", "--host", "0.0.0.0", "--port", "8000"]
