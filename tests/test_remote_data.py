"""Data copied from a Cloud Storage bucket at start-up (Cloud Run), and the front's
identity token for the private API. Cloud Storage and the metadata server are
faked with httpx.MockTransport: no network."""

from pathlib import Path

import httpx
import pytest

from docagent.api import app as api_app
from docagent.config import Settings
from docagent.gcp import METADATA_IDENTITY, identity_token
from docagent.remote_data import DONE, download_folder, fetch_remote_data, parse_gs_url

OBJECTS = {
    "index/passages.jsonl": b'{"id": "p1"}\n',
    "index/qdrant/meta.json": b"{}",
    "index/qdrant/": b"",  # a folder placeholder, as the console creates
    "raw/ipcc.pdf": b"%PDF-1.7",
}


def fake_storage(objects=OBJECTS, page_size=2):
    """Cloud Storage's JSON API: list (paged) and download, with a bearer token."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.headers.get("Authorization") != "Bearer tok":
            return httpx.Response(401)
        path = request.url.raw_path.decode().split("?")[0]
        if path == "/storage/v1/b/bucket/o":
            prefix = request.url.params.get("prefix", "")
            names = sorted(n for n in objects if n.startswith(prefix))
            start = int(request.url.params.get("pageToken", 0))
            body: dict = {"items": [{"name": n} for n in names[start : start + page_size]]}
            if start + page_size < len(names):
                body["nextPageToken"] = str(start + page_size)
            return httpx.Response(200, json=body)
        name = path.removeprefix("/storage/v1/b/bucket/o/").replace("%2F", "/")
        if request.url.params.get("alt") == "media" and name in objects:
            return httpx.Response(200, content=objects[name])
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler)), calls


def test_a_gs_url_gives_the_bucket_and_the_folder():
    assert parse_gs_url("gs://bucket/index") == ("bucket", "index/")
    assert parse_gs_url("gs://bucket/a/b/") == ("bucket", "a/b/")
    assert parse_gs_url("gs://bucket") == ("bucket", "")
    with pytest.raises(ValueError):
        parse_gs_url("/local/index")


def test_a_folder_is_copied_with_its_subfolders_then_not_again(tmp_path):
    client, calls = fake_storage()
    target = tmp_path / "index"
    assert download_folder("gs://bucket/index", target, lambda: "tok", client) == 2
    assert (target / "passages.jsonl").read_bytes() == b'{"id": "p1"}\n'
    assert (target / "qdrant" / "meta.json").read_bytes() == b"{}"
    assert (target / DONE).exists()
    assert not (target / "ipcc.pdf").exists()  # another folder of the bucket

    before = len(calls)
    assert download_folder("gs://bucket/index", target, lambda: "tok", client) == 0
    assert len(calls) == before  # complete: no call at all


def test_an_interrupted_copy_is_done_again(tmp_path):
    client, _ = fake_storage()
    target = tmp_path / "index"
    target.mkdir()
    (target / "passages.jsonl").write_text("half")  # no DONE marker: incomplete
    download_folder("gs://bucket/index", target, lambda: "tok", client)
    assert (target / "passages.jsonl").read_bytes() == b'{"id": "p1"}\n'


def test_an_empty_folder_or_a_refused_access_fails_loudly(tmp_path):
    client, _ = fake_storage()
    with pytest.raises(RuntimeError, match="no file"):
        download_folder("gs://bucket/nothing", tmp_path / "x", lambda: "tok", client)
    with pytest.raises(httpx.HTTPStatusError):
        download_folder("gs://bucket/index", tmp_path / "y", lambda: "bad", client)
    assert not (tmp_path / "y" / DONE).exists()


def test_an_object_name_cannot_write_outside_the_folder(tmp_path):
    client, _ = fake_storage({"index/../evil.txt": b"x"})
    with pytest.raises(ValueError, match="unsafe"):
        download_folder("gs://bucket/index", tmp_path / "index", lambda: "tok", client)
    assert not (tmp_path / "evil.txt").exists()


def test_the_settings_name_the_folders_to_copy(tmp_path):
    client, _ = fake_storage()
    settings = Settings(
        _env_file=None,
        rag_index_url="gs://bucket/index",
        rag_raw_url="gs://bucket/raw",
        rag_index_dir=tmp_path / "index",
        rag_raw_dir=tmp_path / "raw",
    )
    fetch_remote_data(settings, token=lambda: "tok", client=client)
    assert (tmp_path / "index" / "passages.jsonl").exists()
    assert (tmp_path / "raw" / "ipcc.pdf").read_bytes() == b"%PDF-1.7"


def test_without_urls_nothing_is_downloaded_and_no_sign_in_is_needed(tmp_path):
    def no_token() -> str:
        raise AssertionError("no sign-in expected")

    settings = Settings(_env_file=None, rag_index_dir=tmp_path / "index")
    fetch_remote_data(settings, token=no_token)
    assert not (tmp_path / "index").exists()


def test_the_default_answerer_copies_the_index_first(monkeypatch, tmp_path):
    order: list[str] = []
    monkeypatch.setattr(api_app, "fetch_remote_data", lambda s: order.append("fetch"))
    monkeypatch.setattr(api_app, "build_answerer", lambda s: order.append("build") or "answerer")
    assert api_app.default_answerer(Settings(_env_file=None)) == "answerer"
    assert order == ["fetch", "build"]


def test_the_identity_token_comes_from_the_metadata_server():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text="id-token")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert identity_token("https://api.run.app", client) == "id-token"
    assert str(seen[0].url).startswith(METADATA_IDENTITY)
    assert seen[0].url.params["audience"] == "https://api.run.app"
    assert seen[0].headers["Metadata-Flavor"] == "Google"


def test_no_metadata_server_raises_an_http_error():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    with pytest.raises(httpx.HTTPError):
        identity_token("https://api.run.app", client)


def test_paths_are_relative_to_the_folder(tmp_path):
    client, _ = fake_storage(page_size=1)  # several pages of results
    download_folder("gs://bucket/raw", tmp_path, lambda: "tok", client)
    assert sorted(p.name for p in Path(tmp_path).iterdir()) == [DONE, "ipcc.pdf"]
