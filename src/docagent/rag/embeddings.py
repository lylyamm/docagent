"""Embeddings: text -> vector, the "meaning" half of the hybrid search.

Real embedders, compared on the same questions (section 7 of the course):
- ``MistralEmbedder``: ``mistral-embed`` through the API (1024 dimensions,
  multilingual). Nothing to install, but every passage goes to the API: public
  documents only on the free tier.
- ``LocalEmbedder``: open models run locally with sentence-transformers,
  ``e5`` (multilingual-e5-base, 768 dimensions) and ``qwen3``
  (Qwen3-Embedding-0.6B, 1024 dimensions). No API and no data sent out, but
  PyTorch (~2 GB) and slow indexing on a CPU. Installed with ``uv sync --extra local``.
And ``HashEmbedder`` for tests: deterministic, no model, no network.

Vectors are L2-normalised, so the dot product IS the cosine similarity.
"""

import hashlib
import logging
import math
import random
import time
from dataclasses import dataclass
from typing import Protocol

import httpx

from ..config import Settings, get_settings
from .bm25 import tokenize

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str  # also names the Qdrant collection: vectors of different models never mix
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class HashEmbedder:
    """Feature hashing of the words: similar wording -> similar vectors. Tests only."""

    name = "hash"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in tokenize(text):
            h = int.from_bytes(hashlib.md5(token.encode()).digest()[:4], "little")
            vector[h % self.dim] += 1.0
        return _normalise(vector)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@dataclass(frozen=True)
class LocalModel:
    """A sentence-transformers model and the prompts it was trained with."""

    repo: str
    query_prefix: str = ""
    document_prefix: str = ""
    batch_size: int = 32


# Retrieval models are asymmetric: a short question vs a long passage. Each was
# trained with its own markers, and leaving them out costs accuracy.
QWEN3_TASK = "Given a question, retrieve passages from climate and energy reports that answer it"
LOCAL_MODELS = {
    # multilingual-e5-base (Wang et al., 2022): XLM-RoBERTa encoder, 278 M parameters,
    # 768 dims, mean pooling; "query: " / "passage: " prefixes.
    "e5": LocalModel("intfloat/multilingual-e5-base", "query: ", "passage: "),
    # Qwen3-Embedding-0.6B (2025): decoder LLM turned encoder, 600 M parameters,
    # 1024 dims, last-token pooling; an instruction before the query only (in
    # English, as recommended, whatever the language of the question).
    "qwen3": LocalModel(
        "Qwen/Qwen3-Embedding-0.6B",
        query_prefix=f"Instruct: {QWEN3_TASK}\nQuery:",
        batch_size=8,  # 28 layers: smaller batches keep memory flat on a laptop CPU
    ),
}


class LocalEmbedder:
    """A model from ``LOCAL_MODELS`` run on this machine with sentence-transformers."""

    def __init__(self, name: str, repo: str | None = None, max_tokens: int = 512) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise RuntimeError("local embeddings need: uv sync --extra local") from exc
        spec = LOCAL_MODELS[name]
        self.name = name
        self.spec = spec
        self.model_id = repo or spec.repo
        self.model = SentenceTransformer(self.model_id)
        # Our passages are under 1,000 characters (~300 tokens); the cap only bounds
        # the cost of an outlier (Qwen3 accepts 32k tokens, attention is quadratic).
        self.model.max_seq_length = min(self.model.max_seq_length or max_tokens, max_tokens)
        dimension = getattr(self.model, "get_embedding_dimension", None)
        self.dim = (dimension or self.model.get_sentence_embedding_dimension)()

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=self.spec.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 200,
        )
        return vectors.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode([self.spec.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.spec.query_prefix + text])[0]


class MistralEmbedder:
    """``POST /v1/embeddings`` in batches, with the translator's retry policy."""

    name = "mistral"
    dim = 1024

    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.Client | None = None,
        sleep=time.sleep,
        max_batch: int = 32,
        max_batch_chars: int = 24_000,
    ) -> None:
        self.settings = settings or get_settings()
        self._key = self.settings.mistral_key
        self.model = self.settings.mistral_embed_model
        self.client = client or httpx.Client(timeout=self.settings.llm_timeout_s)
        self.sleep = sleep
        self.max_batch = max_batch
        self.max_batch_chars = max_batch_chars
        self.requests = 0
        self.tokens = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in self._batches(texts):
            vectors.extend(self._embed(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def _batches(self, texts: list[str]) -> list[list[str]]:
        batches: list[list[str]] = [[]]
        size = 0
        for text in texts:
            if batches[-1] and (
                len(batches[-1]) >= self.max_batch or size + len(text) > self.max_batch_chars
            ):
                batches.append([])
                size = 0
            batches[-1].append(text)
            size += len(text)
        return [b for b in batches if b]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        attempts = self.settings.llm_max_retries + 1
        for attempt in range(attempts):
            if self.requests:
                self.sleep(self.settings.llm_min_interval_s)
            self.requests += 1
            try:
                response = self.client.post(
                    "https://api.mistral.ai/v1/embeddings",
                    headers={"Authorization": f"Bearer {self._key.get_secret_value()}"},
                    json={"model": self.model, "input": texts},
                )
            except httpx.TransportError as exc:
                error: str = str(exc)
            else:
                if response.status_code == 200:
                    payload = response.json()
                    self.tokens += (payload.get("usage") or {}).get("prompt_tokens", 0)
                    rows = sorted(payload["data"], key=lambda row: row["index"])
                    return [_normalise(row["embedding"]) for row in rows]
                if response.status_code != 429 and response.status_code < 500:
                    response.raise_for_status()  # a bad key or a bad request: no retry
                error = f"HTTP {response.status_code}"
            delay = min(5 * 2**attempt, 60) + random.uniform(0, 1)
            logger.warning("embedding request failed (%s), retrying in %.1fs", error, delay)
            self.sleep(delay)
        raise RuntimeError(f"embeddings failed after {attempts} attempts")


def build_embedder(settings: Settings | None = None, name: str | None = None) -> Embedder:
    settings = settings or get_settings()
    name = name or settings.embedder
    if name == "mistral":
        return MistralEmbedder(settings)
    if name in LOCAL_MODELS:
        repo = {"e5": settings.e5_model, "qwen3": settings.qwen3_model}.get(name)
        return LocalEmbedder(name, repo, settings.local_max_tokens)
    if name == "hash":
        return HashEmbedder()
    raise ValueError(f"unknown embedder {name!r}")
