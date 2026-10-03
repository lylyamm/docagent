"""Vector store: Qdrant, embedded on disk, in memory (tests) or as a server (Docker).

A point = one passage's vector; its id is derived from the embedded text and
the model, so re-indexing only embeds (and pays for) passages that changed.
"""

import atexit
import hashlib
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .chunking import Passage


def point_id(passage: Passage, model: str) -> str:
    """Qdrant ids are UUIDs: uuid5 of (model, text) is stable across runs."""
    digest = hashlib.sha256(f"{model}\n{passage.embed_text}".encode()).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, digest))


_embedded: dict[str, QdrantClient] = {}


@atexit.register
def _close_embedded() -> None:  # before interpreter shutdown, or Qdrant warns noisily
    for client in _embedded.values():
        client.close()


def open_client(location: str | Path) -> QdrantClient:
    location = str(location)
    if location == ":memory:":
        return QdrantClient(location=":memory:")  # a fresh, empty store each time
    if location.startswith(("http://", "https://")):
        return QdrantClient(url=location)
    # Embedded mode: a folder, no server. It locks the folder, so one client per
    # folder and per process, shared by every collection (one per embedder).
    folder = str(Path(location).resolve())
    if folder not in _embedded:
        Path(folder).mkdir(parents=True, exist_ok=True)
        try:
            _embedded[folder] = QdrantClient(path=folder)
        except RuntimeError as exc:
            if "already accessed" not in str(exc):
                raise
            raise RuntimeError(
                f"the vector store {folder} is open in another program (an indexing still "
                "running?). Wait for it to finish, or stop it, then run this again."
            ) from None
    return _embedded[folder]


class VectorStore:
    def __init__(self, client: QdrantClient, collection: str, dim: int, model: str) -> None:
        self.client = client
        self.collection = collection
        self.dim = dim
        self.model = model
        if not client.collection_exists(collection):
            client.create_collection(
                collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

    def missing(self, passages: list[Passage]) -> list[Passage]:
        """Passages whose vector is not stored yet."""
        ids = [point_id(p, self.model) for p in passages]
        found: set[str] = set()
        for i in range(0, len(ids), 512):
            found.update(
                str(r.id)
                for r in self.client.retrieve(self.collection, ids[i : i + 512], with_payload=False)
            )
        return [p for p, pid in zip(passages, ids, strict=True) if pid not in found]

    def upsert(self, passages: list[Passage], vectors: list[list[float]]) -> None:
        if len(passages) != len(vectors):
            raise ValueError(f"{len(passages)} passages but {len(vectors)} vectors")
        points = [
            PointStruct(
                id=point_id(p, self.model),
                vector=v,
                payload={"passage_id": p.id, "doc_id": p.doc_id, "page": p.page},
            )
            for p, v in zip(passages, vectors, strict=True)
        ]
        for i in range(0, len(points), 256):
            self.client.upsert(self.collection, points=points[i : i + 256])

    def search(self, vector: list[float], top_k: int) -> list[tuple[str, float]]:
        """(passage id, cosine similarity), best first."""
        hits = self.client.query_points(
            self.collection, query=vector, limit=top_k, with_payload=True
        ).points
        return [(h.payload["passage_id"], h.score) for h in hits]

    def count(self) -> int:
        return self.client.count(self.collection).count
