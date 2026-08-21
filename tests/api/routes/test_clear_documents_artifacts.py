"""``/documents/clear`` removes parser artifact directories recursively.

The storage drop and the top-level input-file sweep leave the
``<input>/__parsed__/`` tree (sidecar ``*.parsed`` dirs and preserved raw
bundles — see ``PARSED_ARTIFACT_DIR_SUFFIXES``) behind. The clear now removes
that whole tree for the workspace the request resolved to, stays idempotent
when it is absent, and does not touch sibling workspace directories.
"""

import importlib
import sys
from uuid import uuid4

import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

DocumentManager = _document_routes.DocumentManager
create_document_routes = _document_routes.create_document_routes

pytestmark = pytest.mark.offline


class _NoopStorage:
    namespace = "noop"

    async def drop(self):
        return {"status": "success", "message": "data dropped"}

    async def initialize(self):
        return None


class _ClearRag:
    def __init__(self, workspace: str):
        self.workspace = workspace
        storage = _NoopStorage()
        storage.workspace = workspace
        # The eleven storage attributes the clear endpoint iterates over.
        self.text_chunks = storage
        self.full_docs = storage
        self.full_entities = storage
        self.full_relations = storage
        self.entity_chunks = storage
        self.relation_chunks = storage
        self.entities_vdb = storage
        self.relationships_vdb = storage
        self.chunks_vdb = storage
        self.chunk_entity_relation_graph = storage
        self.doc_status = storage

    async def aclear_cache(self, modes=None):
        return None


def _clear_endpoint(rag, doc_manager):
    router = create_document_routes(rag, doc_manager)
    return [
        route.endpoint
        for route in router.routes
        if getattr(route, "name", "") == "clear_documents"
    ][-1]


def _seed_workspace_tree(input_dir, parsed_names=("doc.a.parsed", "doc.b.mineru_raw")):
    """Create a top-level file, artifact dirs with nested content, and an
    unrelated subdirectory that must survive the clear."""
    (input_dir / "source.pdf").write_bytes(b"pdf bytes")
    for name in parsed_names:
        artifact = input_dir / "__parsed__" / name
        artifact.mkdir(parents=True)
        (artifact / "payload.json").write_text("{}")
    keep = input_dir / "other_subdir"
    keep.mkdir()
    (keep / "keep.txt").write_text("keep")


async def test_clear_documents_removes_parsed_artifact_tree(tmp_path):
    workspace = f"clear-artifacts-{uuid4().hex[:8]}"
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    shared_storage.initialize_share_data()
    await shared_storage.initialize_pipeline_status(workspace=workspace)

    doc_manager = DocumentManager(str(tmp_path), workspace=workspace)
    _seed_workspace_tree(doc_manager.input_dir)

    rag = _ClearRag(workspace)
    endpoint = _clear_endpoint(rag, doc_manager)

    response = await endpoint()
    assert response.status == "success"

    # Top-level input files are gone, the whole __parsed__ tree (sidecar and
    # raw-bundle dirs, recursively) is gone, unrelated subdirs survive.
    assert not (doc_manager.input_dir / "source.pdf").exists()
    assert not (doc_manager.input_dir / "__parsed__").exists()
    assert (doc_manager.input_dir / "other_subdir" / "keep.txt").exists()


async def test_clear_documents_artifact_cleanup_is_idempotent(tmp_path):
    """No ``__parsed__`` directory at all -> still a plain success."""
    workspace = f"clear-artifacts-empty-{uuid4().hex[:8]}"
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    shared_storage.initialize_share_data()
    await shared_storage.initialize_pipeline_status(workspace=workspace)

    doc_manager = DocumentManager(str(tmp_path), workspace=workspace)
    (doc_manager.input_dir / "only.pdf").write_bytes(b"bytes")

    rag = _ClearRag(workspace)
    endpoint = _clear_endpoint(rag, doc_manager)

    response = await endpoint()
    assert response.status == "success"
    assert not (doc_manager.input_dir / "only.pdf").exists()


async def test_clear_documents_scoped_to_resolved_workspace(tmp_path):
    """The artifact cleanup must only touch the workspace the request
    resolved to: a sibling workspace's input dir (files and ``__parsed__``)
    is left intact."""
    workspace = f"clear-ws-a-{uuid4().hex[:8]}"
    other_workspace = f"clear-ws-b-{uuid4().hex[:8]}"
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    shared_storage.initialize_share_data()
    await shared_storage.initialize_pipeline_status(workspace=workspace)

    doc_manager = DocumentManager(str(tmp_path), workspace=workspace)
    _seed_workspace_tree(doc_manager.input_dir)
    other_dir = tmp_path / other_workspace
    other_dir.mkdir()
    _seed_workspace_tree(other_dir)

    rag = _ClearRag(workspace)
    endpoint = _clear_endpoint(rag, doc_manager)

    response = await endpoint()
    assert response.status == "success"

    assert not (doc_manager.input_dir / "__parsed__").exists()
    assert not (doc_manager.input_dir / "source.pdf").exists()
    # Sibling workspace untouched.
    assert (other_dir / "source.pdf").exists()
    assert (other_dir / "__parsed__" / "doc.a.parsed" / "payload.json").exists()
    assert (
        other_dir / "__parsed__" / "doc.b.mineru_raw" / "payload.json"
    ).exists()
