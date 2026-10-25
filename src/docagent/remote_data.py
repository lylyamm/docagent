"""Data kept in a Cloud Storage bucket, copied to the local disk at start-up.

On Cloud Run the container's disk is empty at each start: the image holds the code
and the embedding model, the bucket holds the data (the search index, the corpus
PDFs). ``RAG_INDEX_URL`` and ``RAG_RAW_URL`` (``gs://bucket/folder``) name them, and
they are copied into ``RAG_INDEX_DIR`` and ``RAG_RAW_DIR`` before the first search.
Unset (local runs): nothing is downloaded.

The copy uses Cloud Storage's JSON API with the same Google sign-in as Vertex AI:
on Cloud Run the service's account, locally the application default credentials.
"""

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import httpx

from .config import Settings

logger = logging.getLogger("docagent.remote_data")

API = "https://storage.googleapis.com/storage/v1"
DONE = ".downloaded"  # written last: a folder without it is incomplete
_lock = threading.Lock()  # the answerer and the agent may ask at the same time


def parse_gs_url(url: str) -> tuple[str, str]:
    """``gs://bucket/a/b`` -> ``("bucket", "a/b/")``; the whole bucket: prefix ``""``."""
    if not url.startswith("gs://"):
        raise ValueError(f"expected gs://bucket/folder, got {url!r}")
    bucket, _, prefix = url.removeprefix("gs://").partition("/")
    if not bucket:
        raise ValueError(f"no bucket in {url!r}")
    prefix = prefix.strip("/")
    return bucket, f"{prefix}/" if prefix else ""


def _relative(name: str, prefix: str) -> PurePosixPath | None:
    """The object's path under the folder; None for a folder placeholder. An object
    whose name would write outside the folder (``..``) is refused."""
    path = PurePosixPath(name[len(prefix) :])
    if not path.parts or name.endswith("/"):
        return None
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe object name: {name!r}")
    return path


def list_objects(client: httpx.Client, bucket: str, prefix: str, headers: dict) -> list[str]:
    names: list[str] = []
    params = {"prefix": prefix, "fields": "items(name),nextPageToken"}
    while True:
        response = client.get(f"{API}/b/{bucket}/o", params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        names += [item["name"] for item in data.get("items", [])]
        if "nextPageToken" not in data:
            return names
        params["pageToken"] = data["nextPageToken"]


def download_folder(
    url: str, target: Path, token: Callable[[], str], client: httpx.Client | None = None
) -> int:
    """Copy every object under ``url`` into ``target``, keeping the relative paths.
    Returns the number of files copied: 0 when ``target`` is already complete."""
    target = Path(target)
    if (target / DONE).exists():
        return 0
    bucket, prefix = parse_gs_url(url)
    own_client = client is None
    client = client or httpx.Client(timeout=120)
    try:
        names = list_objects(client, bucket, prefix, {"Authorization": f"Bearer {token()}"})
        files = [(name, rel) for name in names if (rel := _relative(name, prefix))]
        if not files:
            raise RuntimeError(f"no file under {url}")
        for name, rel in files:
            path = target / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            headers = {"Authorization": f"Bearer {token()}"}  # refreshed when expired
            object_url = f"{API}/b/{bucket}/o/{quote(name, safe='')}"
            with client.stream("GET", object_url, params={"alt": "media"}, headers=headers) as r:
                r.raise_for_status()
                with path.open("wb") as out:
                    for chunk in r.iter_bytes(1 << 20):
                        out.write(chunk)
    finally:
        if own_client:
            client.close()
    (target / DONE).write_text(url, encoding="utf-8")
    return len(files)


def fetch_remote_data(
    settings: Settings,
    token: Callable[[], str] | None = None,
    client: httpx.Client | None = None,
) -> None:
    """Copy the folders named by RAG_INDEX_URL / RAG_RAW_URL, once. Raises
    ``RuntimeError`` (no sign-in, no file) or ``httpx.HTTPError`` (403: no access)."""
    folders = [
        (settings.rag_index_url, settings.rag_index_dir),
        (settings.rag_raw_url, settings.rag_raw_dir),
    ]
    with _lock:
        todo = [(url, Path(d)) for url, d in folders if url and not (Path(d) / DONE).exists()]
        if not todo:
            return
        if token is None:
            from .gcp import google_token

            token = google_token(settings.vertex_project)
        for url, folder in todo:
            start = time.perf_counter()
            count = download_folder(url, folder, token, client)
            duration = round(time.perf_counter() - start, 2)
            logger.info(
                "data downloaded",
                extra={"url": url, "files": count, "folder": str(folder), "duration_s": duration},
            )
