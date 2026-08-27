"""``POST /documents/upload`` with a caller-assigned ``doc_id`` form field.

The endpoint forwards a validated id to the pending_parse enqueue as the
``ids`` override, assigns that id a runtime-private physical filename, rejects
malformed ids with 400 before any byte is written, returns 409 when the id
already exists in a non-FAILED state, and retires a FAILED record via the
sanctioned deletion path so a re-upload under the same id is a clean retry.
"""

import importlib
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

from lightrag.base import DocStatus  # noqa: E402

DocumentManager = _document_routes.DocumentManager
create_document_routes = _document_routes.create_document_routes

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _ensure_shared_storage_initialized():
    """Same shared-storage bootstrap as the other route tests: the upload
    endpoint reserves a pending-enqueue slot via ``pipeline_status``."""
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    shared_storage.initialize_share_data()
    yield
    if shared_storage._shared_dicts is not None:
        for key in list(shared_storage._shared_dicts.keys()):
            if key.endswith("pipeline_status") or key == "pipeline_status":
                ns = shared_storage._shared_dicts[key]
                if isinstance(ns, dict):
                    ns["busy"] = False
                    ns["scanning"] = False


async def _await_managed(managed_tasks):
    import asyncio

    for task in list(managed_tasks):
        try:
            await task
        except asyncio.CancelledError:
            pass


class _DocStatus:
    """Minimal doc_status double: basename pre-check + point reads by id."""

    def __init__(self, docs_by_id):
        self.docs_by_id = docs_by_id

    async def get_doc_by_file_basename(self, basename):
        for doc_id, doc in self.docs_by_id.items():
            if doc.get("file_path") == basename:
                return doc_id, doc
        return None

    async def get_by_id(self, doc_id):
        return self.docs_by_id.get(doc_id)


class _UploadRag:
    def __init__(self, docs_by_id=None, delete_status="success"):
        self.workspace = f"upload-docid-{uuid4().hex}"
        self.doc_status = _DocStatus(docs_by_id or {})
        self.enqueued = []
        self.errors = []
        self.deleted = []
        self.delete_status = delete_status
        self.process_calls = 0

    async def apipeline_enqueue_documents(
        self,
        input,
        ids=None,
        file_paths=None,
        track_id=None,
        docs_format=None,
        parse_engine=None,
        process_options=None,
        chunk_options=None,
        admission_token=None,
        from_scan=False,
    ):
        self.enqueued.append(
            {
                "input": input,
                "ids": ids,
                "file_paths": file_paths,
                "track_id": track_id,
                "docs_format": docs_format,
                "parse_engine": parse_engine,
            }
        )
        return track_id

    async def apipeline_process_enqueue_documents(self):
        self.process_calls += 1

    async def apipeline_enqueue_error_documents(self, error_files, track_id=None):
        self.errors.append((error_files, track_id))

    async def adelete_by_doc_id(self, doc_id, delete_llm_cache=False):
        self.deleted.append(doc_id)
        doc = self.doc_status.docs_by_id.get(doc_id)
        return SimpleNamespace(
            status=self.delete_status,
            message=f"delete outcome: {self.delete_status}",
            file_path=(doc or {}).get("file_path"),
        )


def _upload_endpoint(rag, doc_manager):
    router = create_document_routes(rag, doc_manager)
    return [
        route.endpoint
        for route in router.routes
        if getattr(route, "name", "") == "upload_to_input_dir"
    ][-1]


async def test_upload_with_custom_doc_id_forwards_ids_override(tmp_path, monkeypatch):
    monkeypatch.setattr(
        _document_routes, "global_args", SimpleNamespace(max_upload_size=None)
    )
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    rag = _UploadRag()
    doc_manager = DocumentManager(str(tmp_path), workspace=rag.workspace)
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    endpoint = _upload_endpoint(rag, doc_manager)

    managed = set()
    upload_file = _document_routes.UploadFile(
        filename="report.md", file=BytesIO(b"# report body")
    )
    response = await endpoint(managed, upload_file, doc_id="8012345678901234567890")
    await _await_managed(managed)

    assert response.status == "success"
    storage_name = "__datahub_doc_8012345678901234567890__report.md"
    assert (doc_manager.input_dir / storage_name).exists()
    assert not (doc_manager.input_dir / "report.md").exists()
    assert len(rag.enqueued) == 1
    enqueued = rag.enqueued[0]
    assert enqueued["ids"] == ["8012345678901234567890"]
    assert enqueued["docs_format"] == "pending_parse"
    assert enqueued["file_paths"].endswith(storage_name)
    assert rag.process_calls == 1
    assert rag.deleted == []


async def test_upload_same_display_name_with_distinct_doc_ids_is_independent(
    tmp_path, monkeypatch
):
    """DataHub may archive same-name source files; the runtime must not 409.

    The physical name remains unique per document while the terminal native
    parser hint stays usable after the prefix is added.
    """
    monkeypatch.setattr(
        _document_routes, "global_args", SimpleNamespace(max_upload_size=None)
    )
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    rag = _UploadRag()
    doc_manager = DocumentManager(str(tmp_path), workspace=rag.workspace)
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    endpoint = _upload_endpoint(rag, doc_manager)

    first_id = "8012345678901234567891"
    second_id = "8012345678901234567892"
    first_tasks = set()
    second_tasks = set()
    first = await endpoint(
        first_tasks,
        _document_routes.UploadFile(
            filename="same.[native].md", file=BytesIO(b"first body")
        ),
        doc_id=first_id,
    )
    second = await endpoint(
        second_tasks,
        _document_routes.UploadFile(
            filename="same.[native].md", file=BytesIO(b"second body")
        ),
        doc_id=second_id,
    )
    await _await_managed(first_tasks)
    await _await_managed(second_tasks)

    assert first.status == "success"
    assert second.status == "success"
    first_name = f"__datahub_doc_{first_id}__same.[native].md"
    second_name = f"__datahub_doc_{second_id}__same.[native].md"
    assert (doc_manager.input_dir / first_name).exists()
    assert (doc_manager.input_dir / second_name).exists()
    assert len(rag.enqueued) == 2
    assert [entry["ids"] for entry in rag.enqueued] == [[first_id], [second_id]]
    assert [entry["parse_engine"] for entry in rag.enqueued] == ["native", "native"]
    assert [Path(entry["file_paths"]).name for entry in rag.enqueued] == [
        first_name,
        second_name,
    ]


async def test_upload_without_doc_id_passes_no_ids(tmp_path, monkeypatch):
    """Regression: the legacy upload path still lets the pipeline derive
    md5(canonical file_path) — no ``ids`` kwarg is forwarded."""
    monkeypatch.setattr(
        _document_routes, "global_args", SimpleNamespace(max_upload_size=None)
    )
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    rag = _UploadRag()
    doc_manager = DocumentManager(str(tmp_path), workspace=rag.workspace)
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    endpoint = _upload_endpoint(rag, doc_manager)

    managed = set()
    upload_file = _document_routes.UploadFile(
        filename="plain.md", file=BytesIO(b"plain body")
    )
    response = await endpoint(managed, upload_file)
    await _await_managed(managed)

    assert response.status == "success"
    assert len(rag.enqueued) == 1
    assert rag.enqueued[0]["ids"] is None


@pytest.mark.parametrize(
    "bad_doc_id",
    [
        "",  # blank
        "   ",  # whitespace only
        "has space",  # illegal character
        "slash/id",  # path separator
        "dot.id",  # dots not in the allowed charset
        "-leading-hyphen",  # must start with letter/digit
        "x" * 129,  # over the length cap
    ],
)
async def test_upload_rejects_invalid_doc_id_before_writing(
    tmp_path, monkeypatch, bad_doc_id
):
    monkeypatch.setattr(
        _document_routes, "global_args", SimpleNamespace(max_upload_size=None)
    )
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    rag = _UploadRag()
    doc_manager = DocumentManager(str(tmp_path), workspace=rag.workspace)
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    endpoint = _upload_endpoint(rag, doc_manager)

    upload_file = _document_routes.UploadFile(
        filename="invalid.md", file=BytesIO(b"body")
    )
    with pytest.raises(_document_routes.HTTPException) as excinfo:
        await endpoint(set(), upload_file, doc_id=bad_doc_id)

    assert excinfo.value.status_code == 400
    assert not (tmp_path / "invalid.md").exists()
    assert rag.enqueued == []


async def test_upload_conflicting_live_doc_id_returns_409(tmp_path, monkeypatch):
    monkeypatch.setattr(
        _document_routes, "global_args", SimpleNamespace(max_upload_size=None)
    )
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    rag = _UploadRag(
        {
            "8012345678901234567890": {
                "status": DocStatus.PROCESSED.value,
                "file_path": "old.md",
            }
        }
    )
    doc_manager = DocumentManager(str(tmp_path), workspace=rag.workspace)
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    endpoint = _upload_endpoint(rag, doc_manager)

    upload_file = _document_routes.UploadFile(
        filename="new.md", file=BytesIO(b"new body")
    )
    with pytest.raises(_document_routes.HTTPException) as excinfo:
        await endpoint(set(), upload_file, doc_id="8012345678901234567890")

    assert excinfo.value.status_code == 409
    assert "8012345678901234567890" in excinfo.value.detail
    assert "Status: processed" in excinfo.value.detail
    assert not (tmp_path / "new.md").exists()
    assert rag.deleted == []
    assert rag.enqueued == []


async def test_upload_failed_doc_id_is_retired_then_reuploaded(
    tmp_path, monkeypatch
):
    """A FAILED record under the requested id is the retry case: it is
    deleted through ``adelete_by_doc_id`` (which purges staged chunks) and
    the upload proceeds with the same id."""
    monkeypatch.setattr(
        _document_routes, "global_args", SimpleNamespace(max_upload_size=None)
    )
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    rag = _UploadRag(
        {
            "8012345678901234567890": {
                "status": DocStatus.FAILED.value,
                "file_path": "broken.md",
            }
        }
    )
    doc_manager = DocumentManager(str(tmp_path), workspace=rag.workspace)
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    endpoint = _upload_endpoint(rag, doc_manager)

    managed = set()
    upload_file = _document_routes.UploadFile(
        filename="fixed.md", file=BytesIO(b"fixed body")
    )
    response = await endpoint(managed, upload_file, doc_id="8012345678901234567890")
    await _await_managed(managed)

    assert response.status == "success"
    assert rag.deleted == ["8012345678901234567890"]
    assert len(rag.enqueued) == 1
    assert rag.enqueued[0]["ids"] == ["8012345678901234567890"]


async def test_upload_failed_doc_id_retire_refused_when_delete_not_allowed(
    tmp_path, monkeypatch
):
    """If the failed record cannot be retired (pipeline busy), the upload is
    refused with 409 instead of enqueueing a duplicate the pipeline's
    ``filter_keys`` would silently drop."""
    monkeypatch.setattr(
        _document_routes, "global_args", SimpleNamespace(max_upload_size=None)
    )
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    rag = _UploadRag(
        {
            "8012345678901234567890": {
                "status": DocStatus.FAILED.value,
                "file_path": "broken.md",
            }
        },
        delete_status="not_allowed",
    )
    doc_manager = DocumentManager(str(tmp_path), workspace=rag.workspace)
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    endpoint = _upload_endpoint(rag, doc_manager)

    upload_file = _document_routes.UploadFile(
        filename="fixed.md", file=BytesIO(b"fixed body")
    )
    with pytest.raises(_document_routes.HTTPException) as excinfo:
        await endpoint(set(), upload_file, doc_id="8012345678901234567890")

    assert excinfo.value.status_code == 409
    assert not (tmp_path / "fixed.md").exists()
    assert rag.enqueued == []
