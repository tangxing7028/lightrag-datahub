from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
_base = importlib.import_module("lightrag.base")
sys.argv = _original_argv

create_document_routes = _document_routes.create_document_routes
DocStatus = _base.DocStatus

pytestmark = pytest.mark.offline


def _row(doc_id: str, status: DocStatus = DocStatus.PROCESSING) -> dict:
    return {
        "track_id": f"track-{doc_id}",
        "status": status,
        "updated_at": "2026-09-09T10:00:00+00:00",
        "chunks_count": 3 if status == DocStatus.PROCESSED else 0,
        "error_msg": None,
        "metadata": {
            "parse_start_time": 1_789_010_520,
            "parse_end_time": 1_789_010_585,
            "parse_stage_skipped": False,
            "internal_field": "must-not-leave-runtime",
        },
    }


class _Storage:
    def __init__(self, rows: dict[str, dict] | None = None):
        self.rows = rows or {}
        self.requests: list[list[str]] = []
        self.failure: Exception | None = None
        self.override = None

    async def get_by_ids(self, ids: list[str]):
        self.requests.append(list(ids))
        if self.failure is not None:
            raise self.failure
        if self.override is not None:
            return self.override
        return [self.rows.get(doc_id) for doc_id in ids]


class _Manager:
    def __init__(self, instances: dict[str, object]):
        self.instances = instances
        self.requested: list[str] = []

    async def get_instance(self, workspace: str):
        self.requested.append(workspace)
        return self.instances[workspace]


def _client(storage: _Storage, *, manager: _Manager | None = None) -> TestClient:
    default_rag = SimpleNamespace(doc_status=storage, workspace="default")
    app = FastAPI()
    app.include_router(
        create_document_routes(default_rag, SimpleNamespace(), api_key="test-key")
    )
    if manager is not None:
        app.state.rag_manager = manager
    return TestClient(app)


def _headers(workspace: str = "kb_42") -> dict[str, str]:
    return {"X-API-Key": "test-key", "LIGHTRAG-WORKSPACE": workspace}


def test_batch_status_deduplicates_in_request_order_and_reports_missing():
    storage = _Storage({"2": _row("2", DocStatus.PROCESSED)})
    response = _client(storage).post(
        "/documents/status/batch",
        headers=_headers(),
        json={"doc_ids": ["2", "1", "2"]},
    )

    assert response.status_code == 200
    assert storage.requests == [["2", "1"]]
    assert [item["doc_id"] for item in response.json()["documents"]] == ["2"]
    assert response.json()["missing_doc_ids"] == ["1"]
    assert set(response.json()["documents"][0]) == {
        "doc_id",
        "track_id",
        "status",
        "updated_at",
        "chunks_count",
        "error_msg",
        "parse_start_time",
        "parse_end_time",
        "parse_stage_skipped",
    }
    assert response.json()["documents"][0]["parse_start_time"] == 1_789_010_520
    assert response.json()["documents"][0]["parse_end_time"] == 1_789_010_585
    assert response.json()["documents"][0]["parse_stage_skipped"] is None


def test_batch_status_reports_a_parse_cache_hit_without_an_invented_end_time():
    row = _row("2", DocStatus.PROCESSING)
    row["metadata"] = {
        "parse_start_time": 1_789_010_520,
        "parse_stage_skipped": True,
    }
    response = _client(_Storage({"2": row})).post(
        "/documents/status/batch",
        headers=_headers(),
        json={"doc_ids": ["2"]},
    )

    assert response.status_code == 200
    document = response.json()["documents"][0]
    assert document["parse_start_time"] == 1_789_010_520
    assert document["parse_end_time"] is None
    assert document["parse_stage_skipped"] is True


def test_batch_status_accepts_200_ids_and_rejects_201():
    storage = _Storage()
    client = _client(storage)
    two_hundred = [str(index) for index in range(1, 201)]

    accepted = client.post(
        "/documents/status/batch",
        headers=_headers(),
        json={"doc_ids": two_hundred},
    )
    rejected = client.post(
        "/documents/status/batch",
        headers=_headers(),
        json={"doc_ids": two_hundred + ["201"]},
    )

    assert accepted.status_code == 200
    assert accepted.json()["missing_doc_ids"] == two_hundred
    assert rejected.status_code == 422


@pytest.mark.parametrize("broken", [RuntimeError("db unavailable"), [None]])
def test_batch_status_returns_503_for_storage_error_or_incomplete_result(broken):
    storage = _Storage()
    if isinstance(broken, Exception):
        storage.failure = broken
    else:
        storage.override = broken

    response = _client(storage).post(
        "/documents/status/batch",
        headers=_headers(),
        json={"doc_ids": ["1", "2"]},
    )

    assert response.status_code == 503


@pytest.mark.parametrize("row", ["invalid-row", {"status": "unknown"}])
def test_batch_status_returns_503_for_invalid_storage_row(row):
    storage = _Storage()
    storage.override = [row]

    response = _client(storage).post(
        "/documents/status/batch",
        headers=_headers(),
        json={"doc_ids": ["1"]},
    )

    assert response.status_code == 503


def test_batch_status_uses_header_selected_workspace_and_requires_authentication():
    default = _Storage({"1": _row("default")})
    selected = _Storage({"1": _row("1")})
    manager = _Manager({"kb_99": SimpleNamespace(doc_status=selected, workspace="kb_99")})
    client = _client(default, manager=manager)

    unauthorized = client.post(
        "/documents/status/batch",
        headers={"LIGHTRAG-WORKSPACE": "kb_99"},
        json={"doc_ids": ["1"]},
    )
    response = client.post(
        "/documents/status/batch",
        headers=_headers("kb_99"),
        json={"doc_ids": ["1"]},
    )

    assert unauthorized.status_code in {401, 403}
    assert response.status_code == 200
    assert manager.requested == ["kb_99"]
    assert selected.requests == [["1"]]
    assert default.requests == []
