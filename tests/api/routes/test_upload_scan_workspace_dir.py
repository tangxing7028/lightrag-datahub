"""Upload/scan input-directory routing follows the request workspace.

``POST /documents/upload`` and the ``/documents/scan`` discovery pass must
operate on the input directory of the workspace the request resolved to
(``LIGHTRAG-WORKSPACE`` header, Patch ①), reusing the ``workspace_input_dir``
semantics Patch ④e established for ``/documents/clear``: ``<base>/<workspace>``
for a header-selected instance, the base directory when no header is present
(historical behavior). The same-name 409 pre-check and the failed-retry stale
cleanup move with the same root.
"""

import importlib
import sys
from io import BytesIO
from types import SimpleNamespace
from uuid import uuid4

import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

from lightrag.base import SourceAbsent  # noqa: E402

DocumentManager = _document_routes.DocumentManager
create_document_routes = _document_routes.create_document_routes
run_scanning_process = _document_routes.run_scanning_process
workspace_input_dir = _document_routes.workspace_input_dir

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _ensure_shared_storage_initialized():
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
    def __init__(self, docs_by_id=None):
        self.docs_by_id = docs_by_id or {}

    async def get_doc_by_file_basename(self, basename):
        for doc_id, doc in self.docs_by_id.items():
            if doc.get("file_path") == basename:
                return doc_id, doc
        return None

    async def get_by_id(self, doc_id):
        return self.docs_by_id.get(doc_id)

    async def resolve_doc_source_strict(self, canonical_source_key):
        return SourceAbsent()


class _Rag:
    """Minimal rag double: a workspace name plus the enqueue capture."""

    def __init__(self, workspace):
        self.workspace = workspace
        self.doc_status = _DocStatus()
        self.enqueued_files = []

    async def apipeline_enqueue_documents(self, input, **kwargs):
        self.enqueued_files.append(kwargs.get("file_paths"))
        return kwargs.get("track_id")

    async def apipeline_process_enqueue_documents(self):
        pass

    async def apipeline_enqueue_error_documents(self, error_files, track_id=None):
        pass

    async def arollback_failed_custom_chunk_patches(self, **_kwargs):
        return {"rolled_back": [], "failed": []}


def _upload_endpoint(rag, doc_manager):
    router = create_document_routes(rag, doc_manager)
    return [
        route.endpoint
        for route in router.routes
        if getattr(route, "name", "") == "upload_to_input_dir"
    ][-1]


async def _do_upload(tmp_path, monkeypatch, workspace, filename="report.md"):
    monkeypatch.setattr(
        _document_routes, "global_args", SimpleNamespace(max_upload_size=None)
    )
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    # The route-module DocumentManager mirrors the SERVER's startup workspace
    # ("" here); the rag's workspace is what the header resolution produced.
    doc_manager = DocumentManager(str(tmp_path))
    rag = _Rag(workspace)
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    endpoint = _upload_endpoint(rag, doc_manager)

    managed = set()
    upload_file = _document_routes.UploadFile(
        filename=filename, file=BytesIO(b"# body")
    )
    response = await endpoint(managed, upload_file)
    await _await_managed(managed)
    return rag, doc_manager, response


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


async def test_upload_without_workspace_lands_in_base_dir(tmp_path, monkeypatch):
    """No header -> the default (startup) input directory, unchanged behavior."""
    rag, doc_manager, response = await _do_upload(tmp_path, monkeypatch, "")

    assert response.status == "success"
    assert (tmp_path / "report.md").exists()
    assert doc_manager.input_dir == tmp_path


async def test_upload_with_workspace_lands_in_workspace_dir(tmp_path, monkeypatch):
    workspace = f"ws-{uuid4().hex[:8]}"
    rag, doc_manager, response = await _do_upload(tmp_path, monkeypatch, workspace)

    assert response.status == "success"
    target = tmp_path / workspace / "report.md"
    assert target.exists()
    assert not (tmp_path / "report.md").exists()
    # The enqueue saw the workspace-dir path.
    assert rag.enqueued_files and str(rag.enqueued_files[0]).replace(
        "\\", "/"
    ).endswith(f"{workspace}/report.md")


async def test_upload_same_name_409_checks_the_workspace_dir(tmp_path, monkeypatch):
    """A same-name file in ANOTHER workspace's dir must not block the upload;
    a same-name file in THIS workspace's dir must 409."""
    workspace = f"ws-{uuid4().hex[:8]}"
    other_workspace = f"ws-other-{uuid4().hex[:8]}"

    # Same name in a sibling workspace: upload proceeds.
    sibling = tmp_path / other_workspace
    sibling.mkdir()
    (sibling / "report.md").write_text("other", encoding="utf-8")
    _, _, response = await _do_upload(tmp_path, monkeypatch, workspace)
    assert response.status == "success"
    assert (tmp_path / workspace / "report.md").exists()

    # Same name in the target workspace: 409.
    with pytest.raises(_document_routes.HTTPException) as excinfo:
        await _do_upload(tmp_path, monkeypatch, workspace)
    assert excinfo.value.status_code == 409


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------


def test_iter_new_files_scans_the_override_directory(tmp_path):
    doc_manager = DocumentManager(str(tmp_path))
    (tmp_path / "root.md").write_text("root", encoding="utf-8")
    workspace_dir = tmp_path / "ws_x"
    workspace_dir.mkdir()
    (workspace_dir / "scoped.md").write_text("scoped", encoding="utf-8")

    default_names = {path.name for path in doc_manager.iter_new_files()}
    override_names = {path.name for path in doc_manager.iter_new_files(workspace_dir)}
    assert default_names == {"root.md"}
    assert override_names == {"scoped.md"}


async def test_scan_discovers_files_in_the_request_workspace_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(
        _document_routes, "global_args", SimpleNamespace(scan_enqueue_batch_size=8)
    )
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    workspace = f"ws-scan-{uuid4().hex[:8]}"
    doc_manager = DocumentManager(str(tmp_path))
    rag = _Rag(workspace)
    await shared_storage.initialize_pipeline_status(workspace=workspace)

    # One file in the workspace dir, one decoy in the base dir.
    workspace_dir = tmp_path / workspace
    workspace_dir.mkdir()
    scoped = workspace_dir / "scoped.md"
    scoped.write_text("scoped", encoding="utf-8")
    (tmp_path / "decoy.md").write_text("decoy", encoding="utf-8")

    batched = []

    async def _capture_batch(_rag, candidates, _track_id):
        batched.extend(c.path for c in candidates)
        return len(candidates)

    monkeypatch.setattr(_document_routes, "pipeline_enqueue_scan_batch", _capture_batch)

    await run_scanning_process(rag, doc_manager, f"track-{uuid4().hex[:8]}")

    assert batched == [scoped]


def test_workspace_input_dir_matches_document_manager_layout(tmp_path):
    """Patch ④e semantics: header workspace -> <base>/<ws>; none -> base."""
    doc_manager = DocumentManager(str(tmp_path))
    assert workspace_input_dir(doc_manager, None) == tmp_path
    assert workspace_input_dir(doc_manager, "") == tmp_path
    assert workspace_input_dir(doc_manager, "ws_a") == tmp_path / "ws_a"
    # A manager built WITH a workspace (the startup instance on a
    # --workspace server) resolves that same workspace to its input_dir.
    scoped = DocumentManager(str(tmp_path), workspace="ws_a")
    assert workspace_input_dir(scoped, "ws_a") == scoped.input_dir
