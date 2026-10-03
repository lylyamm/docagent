"""Copy the vectors of a local index into another Qdrant (the Docker service).

    docker compose up -d qdrant
    uv run python scripts/rag_copy_vectors.py                        # to http://localhost:6333
    uv run python scripts/rag_copy_vectors.py --collection passages_e5

Indexing 4,970 passages with the local model takes about 45 min on a laptop
CPU; copying the vectors already computed takes a few seconds. Point ids,
vectors and payloads are copied as they are, so the API finds exactly the index
that was evaluated. The passages themselves (data/index/passages.jsonl) are
mounted into the API container by docker-compose.yml.
"""

import argparse
from pathlib import Path

from qdrant_client.models import Distance, PointStruct, VectorParams

from docagent.config import get_settings
from docagent.rag import open_client


def copy_collection(source, target, name: str, batch: int = 256) -> int:
    """Copy one collection; returns the number of points in the target."""
    params = source.get_collection(name).config.params.vectors
    if not target.collection_exists(name):
        target.create_collection(
            name, vectors_config=VectorParams(size=params.size, distance=Distance.COSINE)
        )
    offset = None
    while True:
        points, offset = source.scroll(
            name, limit=batch, offset=offset, with_vectors=True, with_payload=True
        )
        if points:
            target.upsert(
                name,
                points=[PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points],
            )
        if offset is None:
            return target.count(name).count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    settings = get_settings()
    parser.add_argument("--source", type=Path, default=settings.rag_index_dir / "qdrant")
    parser.add_argument("--to", default="http://localhost:6333", help="Qdrant URL (or a folder)")
    parser.add_argument("--collection", action="append", help="default: every collection")
    args = parser.parse_args()

    source, target = open_client(args.source), open_client(args.to)
    names = args.collection or [c.name for c in source.get_collections().collections]
    for name in names:
        total = copy_collection(source, target, name)
        print(f"{name}: {source.count(name).count} points copied, {total} in {args.to}")


if __name__ == "__main__":
    main()
