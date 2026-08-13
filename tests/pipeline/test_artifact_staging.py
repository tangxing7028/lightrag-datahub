"""Pipeline hook tests for artifact staging (before chunking) and export
(after PROCESSED), driven by fake storages and a recording HTTP opener.

``_stage_doc_artifacts_before_chunking`` must upload sidecar assets and
rewrite ``<drawing>`` paths in the persisted full_docs body BEFORE the chunk
source is derived; ``_export_processed_doc_artifacts`` must publish
``document.md`` / ``chunks.json`` best-effort. No live object store.
"""

import json
from urllib.error import URLError

import pytest

import lightrag.pipeline as pipeline_mod
from lightrag.constants import FULL_DOCS_FORMAT_LIGHTRAG
from lightrag.sidecar.artifact_store import (
    ArtifactStore,
    ArtifactStoreError,
    config_from_env,
)
from lightrag.utils_pipeline import make_lightrag_doc_content, sidecar_uri_for

pytestmark = pytest.mark.offline

_ENV = {
    "ARTIFACT_S3_ENDPOINT": "http://minio.test:9000",
    "ARTIFACT_S3_BUCKET": "artifacts",
    "ARTIFACT_S3_ACCESS_KEY": "AKID",
    "ARTIFACT_S3_SECRET_KEY": "SECRET",
    "ARTIFACT_PUBLIC_URL_TEMPLATE": (
        "https://proxy.test/api/artifact?workspace={workspace}"
        "&doc_id={doc_id}&path={relpath}"
    ),
}


class _FakeResponse:
    def getcode(self):
        return 200

    def close(self):
        pass


class _RecordingOpener:
    def __init__(self, error=None):
        self.error = error
        self.requests = []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return _FakeResponse()


class _FakeFullDocs:
    def __init__(self, record):
        self.record = record
        self.upserts = []
        self.flush_calls = 0

    async def get_by_id(self, doc_id):
        return self.record if self.record.get("id") == doc_id else None

    async def upsert(self, data):
        self.upserts.append(data)
        ((doc_id, payload),) = data.items()
        if self.record.get("id") == doc_id:
            self.record.update(payload)

    async def index_done_callback(self):
        self.flush_calls += 1


class _FakeTextChunks:
    def __init__(self, rows):
        self.rows = rows

    async def get_by_ids(self, ids):
        return [self.rows.get(chunk_id) for chunk_id in ids]


class _FakePipeline(pipeline_mod._PipelineMixin):
    """_PipelineMixin carrier with only the attributes the hooks touch."""

    def __init__(self, workspace, full_docs, text_chunks=None):
        self.workspace = workspace
        self.full_docs = full_docs
        self.text_chunks = text_chunks


def _install_store(monkeypatch, opener, **env_overrides):
    config = config_from_env({**_ENV, **env_overrides})
    assert config is not None
    store = ArtifactStore(config, opener=opener)
    monkeypatch.setattr(pipeline_mod, "get_artifact_store", lambda: store)
    return store


def _disable_store(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "get_artifact_store", lambda: None)


def _make_sidecar(tmp_path, with_assets=True):
    parsed = tmp_path / "__parsed__" / "abc.parsed"
    parsed.mkdir(parents=True)
    (parsed / "abc.blocks.jsonl").write_text("{}\n", encoding="utf-8")
    if with_assets:
        assets = parsed / "abc.blocks.assets"
        assets.mkdir()
        (assets / "img.png").write_bytes(b"png-bytes")
    return parsed


def _content_data(parsed, body=None):
    drawing = (
        '<drawing id="im-abc-0001" format="png" caption="cap" '
        'path="abc.blocks.assets/img.png" src="orig" />'
    )
    return {
        "id": "doc-1",
        "content": make_lightrag_doc_content(
            body if body is not None else f"intro {drawing} outro"
        ),
        "parse_format": FULL_DOCS_FORMAT_LIGHTRAG,
        "sidecar_location": sidecar_uri_for(parsed),
        "process_options": "F",
    }


# ---------------------------------------------------------------------------
# _stage_doc_artifacts_before_chunking
# ---------------------------------------------------------------------------


async def test_staging_disabled_is_a_noop(tmp_path, monkeypatch):
    _disable_store(monkeypatch)
    parsed = _make_sidecar(tmp_path)
    content_data = _content_data(parsed)
    rag = _FakePipeline("ws_a", _FakeFullDocs(dict(content_data)))

    result = await rag._stage_doc_artifacts_before_chunking(
        doc_id="doc-1", content_data=content_data
    )
    assert result is content_data
    assert rag.full_docs.upserts == []


async def test_staging_uploads_and_rewrites_before_chunking(tmp_path, monkeypatch):
    opener = _RecordingOpener()
    _install_store(monkeypatch, opener)
    parsed = _make_sidecar(tmp_path)
    content_data = _content_data(parsed)
    rag = _FakePipeline("ws_a", _FakeFullDocs(dict(content_data)))

    result = await rag._stage_doc_artifacts_before_chunking(
        doc_id="doc-1", content_data=content_data
    )

    # One path-style PUT per asset file.
    assert [r.full_url for r in opener.requests] == [
        "http://minio.test:9000/artifacts/ws_a/doc-1/abc.blocks.assets/img.png"
    ]
    assert opener.requests[0].data == b"png-bytes"

    # The returned content_data carries the rewritten body; the {{LRdoc}}
    # marker and every non-path attribute survive.
    content = result["content"]
    assert content.startswith("{{LRdoc}}")
    assert (
        'path="https://proxy.test/api/artifact?workspace=ws_a&amp;'
        'doc_id=doc-1&amp;path=abc.blocks.assets/img.png"'
    ) in content
    assert 'src="orig"' in content
    assert 'caption="cap"' in content
    # Untouched fields are preserved.
    assert result["process_options"] == "F"
    assert result["sidecar_location"] == content_data["sidecar_location"]

    # The rewritten body is what full_docs persists (chunking reads it next).
    assert len(rag.full_docs.upserts) == 1
    assert rag.full_docs.record["content"] == content
    assert rag.full_docs.flush_calls == 1


async def test_staging_upload_failure_fails_closed(tmp_path, monkeypatch):
    opener = _RecordingOpener(error=URLError("store unreachable"))
    _install_store(monkeypatch, opener)
    parsed = _make_sidecar(tmp_path)
    content_data = _content_data(parsed)
    rag = _FakePipeline("ws_a", _FakeFullDocs(dict(content_data)))

    with pytest.raises(ArtifactStoreError, match="store unreachable"):
        await rag._stage_doc_artifacts_before_chunking(
            doc_id="doc-1", content_data=content_data
        )
    assert rag.full_docs.upserts == []


async def test_staging_fail_open_continues_with_local_paths(tmp_path, monkeypatch):
    opener = _RecordingOpener(error=URLError("store unreachable"))
    _install_store(monkeypatch, opener, ARTIFACT_UPLOAD_FAIL_OPEN="true")
    parsed = _make_sidecar(tmp_path)
    content_data = _content_data(parsed)
    rag = _FakePipeline("ws_a", _FakeFullDocs(dict(content_data)))

    result = await rag._stage_doc_artifacts_before_chunking(
        doc_id="doc-1", content_data=content_data
    )
    assert result is content_data  # unchanged: no rewrite, no persist
    assert rag.full_docs.upserts == []


async def test_staging_is_idempotent_for_rewritten_content(tmp_path, monkeypatch):
    opener = _RecordingOpener()
    _install_store(monkeypatch, opener)
    parsed = _make_sidecar(tmp_path)
    already = (
        '<drawing id="im-abc-0001" format="png" caption="cap" '
        'path="https://proxy.test/api/artifact?workspace=ws_a&amp;'
        'doc_id=doc-1&amp;path=abc.blocks.assets/img.png" src="orig" />'
    )
    content_data = _content_data(parsed, body=f"intro {already} outro")
    rag = _FakePipeline("ws_a", _FakeFullDocs(dict(content_data)))

    result = await rag._stage_doc_artifacts_before_chunking(
        doc_id="doc-1", content_data=content_data
    )
    # Assets are still (re-)uploaded, but no second rewrite / persist happens.
    assert len(opener.requests) == 1
    assert result is content_data
    assert rag.full_docs.upserts == []


async def test_staging_noop_without_assets_or_sidecar(tmp_path, monkeypatch):
    opener = _RecordingOpener()
    _install_store(monkeypatch, opener)
    rag = _FakePipeline("ws_a", _FakeFullDocs({"id": "doc-1"}))

    # No sidecar_location at all (raw documents).
    raw = {"id": "doc-1", "content": "plain", "parse_format": "raw"}
    assert (
        await rag._stage_doc_artifacts_before_chunking(doc_id="doc-1", content_data=raw)
        is raw
    )

    # Sidecar without an assets directory.
    parsed = _make_sidecar(tmp_path, with_assets=False)
    content_data = _content_data(parsed)
    assert (
        await rag._stage_doc_artifacts_before_chunking(
            doc_id="doc-1", content_data=content_data
        )
        is content_data
    )
    assert opener.requests == []


# ---------------------------------------------------------------------------
# _export_processed_doc_artifacts
# ---------------------------------------------------------------------------


def _export_rag(opener, monkeypatch):
    _install_store(monkeypatch, opener)
    full_docs = _FakeFullDocs(
        {
            "id": "doc-1",
            "content": make_lightrag_doc_content("final body with URLs"),
            "parse_format": FULL_DOCS_FORMAT_LIGHTRAG,
        }
    )
    text_chunks = _FakeTextChunks(
        {
            "chunk-b": {
                "tokens": 7,
                "content": "second",
                "chunk_order_index": 1,
            },
            "chunk-a": {
                "tokens": 5,
                "content": "first",
                "chunk_order_index": 0,
            },
            "chunk-missing": None,
        }
    )
    return _FakePipeline("ws_a", full_docs, text_chunks)


async def test_export_writes_document_md_and_sorted_chunks_json(monkeypatch):
    opener = _RecordingOpener()
    rag = _export_rag(opener, monkeypatch)

    await rag._export_processed_doc_artifacts(
        doc_id="doc-1",
        chunk_ids=["chunk-b", "chunk-a", "chunk-missing"],
        file_path="abc.docx",
    )

    by_url = {request.full_url: request for request in opener.requests}
    doc_req = by_url["http://minio.test:9000/artifacts/ws_a/doc-1/document.md"]
    assert doc_req.data == b"final body with URLs"  # marker stripped

    chunks_req = by_url["http://minio.test:9000/artifacts/ws_a/doc-1/chunks.json"]
    payload = json.loads(chunks_req.data.decode("utf-8"))
    assert payload == [
        {
            "chunk_id": "chunk-a",
            "chunk_order_index": 0,
            "tokens": 5,
            "content": "first",
        },
        {
            "chunk_id": "chunk-b",
            "chunk_order_index": 1,
            "tokens": 7,
            "content": "second",
        },
    ]


async def test_export_failure_is_warning_only(monkeypatch):
    opener = _RecordingOpener(error=URLError("store unreachable"))
    rag = _export_rag(opener, monkeypatch)

    # Must not raise: export is best-effort and never affects PROCESSED.
    await rag._export_processed_doc_artifacts(
        doc_id="doc-1", chunk_ids=["chunk-a"], file_path="abc.docx"
    )


async def test_export_disabled_is_a_noop(monkeypatch):
    _disable_store(monkeypatch)
    rag = _FakePipeline("ws_a", _FakeFullDocs({"id": "doc-1"}), _FakeTextChunks({}))
    await rag._export_processed_doc_artifacts(
        doc_id="doc-1", chunk_ids=["chunk-a"], file_path="abc.docx"
    )
