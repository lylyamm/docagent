"""Hybrid search: dense (meaning) + BM25 (exact words), fused with RRF.

    question ──► embed ──► Qdrant: 20 nearest passages ──┐
             └─► tokens ─► BM25:   20 best passages ─────┴─► RRF ─► top k

Each hit keeps its rank in both lists, so a result can be explained ("found by
BM25 only, 3rd") and the two halves can be evaluated separately.
"""

from typing import Literal

from pydantic import BaseModel

from .bm25 import BM25Index
from .chunking import Passage
from .embeddings import Embedder
from .fusion import reciprocal_rank_fusion
from .store import VectorStore

Mode = Literal["hybrid", "dense", "bm25"]


class SearchHit(BaseModel):
    passage: Passage
    score: float  # RRF score in hybrid mode, cosine or BM25 score otherwise
    dense_rank: int | None = None  # 1-based rank in the dense list, None if absent
    bm25_rank: int | None = None


class HybridSearcher:
    def __init__(
        self,
        passages: list[Passage],
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        candidates: int = 20,
        rrf_k: int = 60,
    ) -> None:
        self.passages = passages
        self.by_id = {p.id: p for p in passages}
        self.bm25 = BM25Index([p.embed_text for p in passages])
        self.embedder = embedder
        self.store = store
        self.candidates = candidates
        self.rrf_k = rrf_k
        self._query_vectors: dict[str, list[float]] = {}  # one API call per distinct query

    def dense(
        self, query: str, top_k: int, doc_ids: list[str] | None = None
    ) -> list[tuple[str, float]]:
        if self.embedder is None or self.store is None:
            raise RuntimeError("dense search needs an embedder and a vector store")
        if query not in self._query_vectors:
            self._query_vectors[query] = self.embedder.embed_query(query)
        hits = self.store.search(self._query_vectors[query], top_k, doc_ids)
        return [(pid, score) for pid, score in hits if pid in self.by_id]

    def lexical(
        self, query: str, top_k: int, doc_ids: list[str] | None = None
    ) -> list[tuple[str, float]]:
        if not doc_ids:
            return [(self.passages[i].id, score) for i, score in self.bm25.search(query, top_k)]
        # BM25 scores every passage that contains a query word anyway: keep those
        # of the wanted documents, then cut.
        wanted = set(doc_ids)
        ranked = self.bm25.search(query, len(self.passages))
        hits = [(self.passages[i].id, s) for i, s in ranked if self.passages[i].doc_id in wanted]
        return hits[:top_k]

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: Mode = "hybrid",
        doc_ids: list[str] | None = None,
    ) -> list[SearchHit]:
        """The ``top_k`` best passages; only from the documents ``doc_ids`` if given."""
        n = self.candidates
        dense = self.dense(query, n, doc_ids) if mode in ("hybrid", "dense") else []
        lexical = self.lexical(query, n, doc_ids) if mode in ("hybrid", "bm25") else []
        dense_rank = {pid: r for r, (pid, _) in enumerate(dense, start=1)}
        bm25_rank = {pid: r for r, (pid, _) in enumerate(lexical, start=1)}

        if mode == "hybrid":
            ranked = reciprocal_rank_fusion(
                [[pid for pid, _ in dense], [pid for pid, _ in lexical]], k=self.rrf_k
            )
        else:
            ranked = dense or lexical
        return [
            SearchHit(
                passage=self.by_id[pid],
                score=round(score, 5),
                dense_rank=dense_rank.get(pid),
                bm25_rank=bm25_rank.get(pid),
            )
            for pid, score in ranked[:top_k]
        ]
