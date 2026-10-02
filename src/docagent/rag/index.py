"""Build and load the search index: passages on disk + their vectors in Qdrant.

data/corpus.json            which PDFs, with their title, organisation, language
data/raw/*.pdf              the PDFs (git-ignored)
data/index/passages.jsonl   the passages: the source of truth, also used by BM25
data/index/qdrant/          the vectors (embedded Qdrant), one collection per embedder
"""

import json
import logging
from collections.abc import Callable
from pathlib import Path

import pymupdf
from pydantic import BaseModel

from ..config import Settings
from ..pdf.language import detect_language
from .chunking import Chunker, Passage
from .embeddings import Embedder
from .store import VectorStore, open_client

logger = logging.getLogger(__name__)


class CorpusDoc(BaseModel):
    id: str  # short and stable: it prefixes passage ids and appears in citations
    file: str
    title: str
    org: str
    year: int | None = None
    lang: str | None = None  # detected when missing
    url: str | None = None


def load_corpus(path: str | Path) -> list[CorpusDoc]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [CorpusDoc.model_validate(d) for d in data]


def build_passages(
    corpus: list[CorpusDoc], raw_dir: Path, chunker: Chunker, on_doc: Callable | None = None
) -> list[Passage]:
    passages: list[Passage] = []
    for doc in corpus:
        path = raw_dir / doc.file
        if not path.exists():
            logger.warning("%s missing, skipped (download it into %s)", doc.file, raw_dir)
            continue
        with pymupdf.open(path) as pdf:
            lang = doc.lang or detect_language(pdf).language
            found = chunker.split_document(pdf, doc.id, doc.title, lang)
        passages.extend(found)
        if on_doc:
            on_doc(doc, len(found))
    return passages


def save_passages(passages: list[Passage], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for p in passages:
            f.write(p.model_dump_json() + "\n")
    tmp.replace(path)


def load_passages(path: Path) -> list[Passage]:
    with path.open(encoding="utf-8") as f:
        return [Passage.model_validate_json(line) for line in f if line.strip()]


def open_store(settings: Settings, embedder: Embedder) -> VectorStore:
    location = settings.qdrant_url or settings.rag_index_dir / "qdrant"
    model = getattr(embedder, "model", embedder.name)
    model = model if isinstance(model, str) else embedder.name
    return VectorStore(open_client(location), f"passages_{embedder.name}", embedder.dim, model)


def index_vectors(
    store: VectorStore,
    embedder: Embedder,
    passages: list[Passage],
    batch: int = 128,
    on_batch: Callable[[int, int], None] | None = None,
) -> int:
    """Embed and store the passages not indexed yet; returns how many were embedded."""
    todo = store.missing(passages)
    for i in range(0, len(todo), batch):
        chunk = todo[i : i + batch]
        store.upsert(chunk, embedder.embed_documents([p.embed_text for p in chunk]))
        if on_batch:
            on_batch(min(i + batch, len(todo)), len(todo))
    return len(todo)
