"""Document search (RAG): passages, BM25, embeddings, Qdrant, hybrid search."""

from .answer import Answer, Answerer, Source, build_answerer
from .bm25 import BM25Index, tokenize
from .chunking import Chunker, Passage
from .embeddings import (
    LOCAL_MODELS,
    Embedder,
    HashEmbedder,
    LocalEmbedder,
    MistralEmbedder,
    build_embedder,
)
from .fusion import reciprocal_rank_fusion
from .index import (
    CorpusDoc,
    build_passages,
    index_vectors,
    load_corpus,
    load_passages,
    open_store,
    save_passages,
)
from .search import HybridSearcher, SearchHit
from .store import VectorStore, open_client

__all__ = [
    "Answer",
    "Answerer",
    "Source",
    "build_answerer",
    "BM25Index",
    "Chunker",
    "CorpusDoc",
    "LOCAL_MODELS",
    "Embedder",
    "HashEmbedder",
    "HybridSearcher",
    "LocalEmbedder",
    "MistralEmbedder",
    "Passage",
    "SearchHit",
    "VectorStore",
    "build_embedder",
    "build_passages",
    "index_vectors",
    "load_corpus",
    "load_passages",
    "open_client",
    "open_store",
    "reciprocal_rank_fusion",
    "save_passages",
    "tokenize",
]
