"""Header-routing tests for multi-workspace request dispatch (issue #2904).

Verifies that routers resolve the per-request LightRAG instance through
``app.state.rag_manager`` using the ``LIGHTRAG-WORKSPACE`` header:

- no header (the official Web UI case) -> the default instance serves;
- a workspace header -> a dedicated instance is built once and reused;
- header names are sanitized with the official rule before lookup;
- without a manager on ``app.state`` (single-workspace embedding) the
  factory-provided default instance always serves.

Everything runs against ``fastapi.testclient.TestClient`` with mocked
instances; no real storage backend is involved.
"""

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Importing routers loads ``lightrag.api.config`` which parses ``sys.argv``
# via argparse. Stash argv so pytest's CLI flags don't trip the parser.
_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_graph_routes = importlib.import_module("lightrag.api.routers.graph_routes")
_query_routes = importlib.import_module("lightrag.api.routers.query_routes")
_workspace_manager = importlib.import_module("lightrag.api.workspace_manager")
sys.argv = _original_argv

create_graph_routes = _graph_routes.create_graph_routes
create_query_routes = _query_routes.create_query_routes
RAGInstanceManager = _workspace_manager.RAGInstanceManager

pytestmark = pytest.mark.offline

_API_KEY = "test-key"
_AUTH_HEADERS = {"X-API-Key": _API_KEY}


class FakeRAG(SimpleNamespace):
    """LightRAG stand-in with a tracked storage lifecycle."""

    def __init__(self, workspace: str, label: str):
        super().__init__(
            workspace=workspace,
            get_graph_labels=AsyncMock(return_value=[f"{label}-label"]),
            aquery_llm=AsyncMock(
                return_value={
                    "llm_response": {"content": f"{label}-answer"},
                    "data": {"references": [], "chunks": []},
                }
            ),
        )
        self.init_calls = 0
        self.finalize_calls = 0

    async def initialize_storages(self):
        self.init_calls += 1

    async def finalize_storages(self):
        self.finalize_calls += 1


def _build_manager(default: FakeRAG) -> tuple[RAGInstanceManager, list[FakeRAG]]:
    created: list[FakeRAG] = []

    def factory(workspace: str) -> FakeRAG:
        instance = FakeRAG(workspace, label=f"ws-{workspace}")
        created.append(instance)
        return instance

    manager = RAGInstanceManager(
        factory,
        default_instance=default,
        default_workspace="",
        max_instances=4,
        ttl_seconds=0,
    )
    return manager, created


def _build_client(with_manager: bool = True):
    default = FakeRAG("", label="default")
    app = FastAPI()
    app.include_router(create_graph_routes(default, api_key=_API_KEY))
    app.include_router(create_query_routes(default, api_key=_API_KEY))
    manager, created = _build_manager(default)
    if with_manager:
        app.state.rag_manager = manager
    return TestClient(app), default, manager, created


# ---------------------------------------------------------------------------
# Default-workspace fallback (no header)
# ---------------------------------------------------------------------------


def test_no_header_serves_default_instance():
    client, default, manager, created = _build_client()
    response = client.get("/graph/label/list", headers=_AUTH_HEADERS)
    assert response.status_code == 200
    assert response.json() == ["default-label"]
    default.get_graph_labels.assert_awaited_once()
    assert created == []  # no per-workspace instance was built


def test_query_without_header_serves_default_instance():
    client, default, manager, created = _build_client()
    response = client.post(
        "/query",
        json={"query": "what is lightrag?"},
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["response"] == "default-answer"
    default.aquery_llm.assert_awaited_once()
    assert created == []


# ---------------------------------------------------------------------------
# Header-driven routing
# ---------------------------------------------------------------------------


def test_workspace_header_routes_to_dedicated_instance():
    client, default, manager, created = _build_client()
    headers = {**_AUTH_HEADERS, "LIGHTRAG-WORKSPACE": "beta"}

    response = client.get("/graph/label/list", headers=headers)
    assert response.status_code == 200
    assert response.json() == ["ws-beta-label"]

    assert len(created) == 1
    assert created[0].workspace == "beta"
    assert created[0].init_calls == 1  # initialized before serving
    default.get_graph_labels.assert_not_awaited()


def test_workspace_header_routes_query_endpoint():
    client, default, manager, created = _build_client()
    headers = {**_AUTH_HEADERS, "LIGHTRAG-WORKSPACE": "gamma"}

    response = client.post(
        "/query", json={"query": "what is lightrag?"}, headers=headers
    )
    assert response.status_code == 200
    assert response.json()["response"] == "ws-gamma-answer"
    assert [instance.workspace for instance in created] == ["gamma"]
    default.aquery_llm.assert_not_awaited()


def test_same_workspace_header_reuses_cached_instance():
    client, default, manager, created = _build_client()
    headers = {**_AUTH_HEADERS, "LIGHTRAG-WORKSPACE": "beta"}

    client.get("/graph/label/list", headers=headers)
    client.get("/graph/label/list", headers=headers)

    assert len(created) == 1
    assert created[0].init_calls == 1
    assert created[0].get_graph_labels.await_count == 2


def test_distinct_workspaces_get_distinct_instances():
    client, default, manager, created = _build_client()
    for workspace in ("ws1", "ws2"):
        response = client.get(
            "/graph/label/list",
            headers={**_AUTH_HEADERS, "LIGHTRAG-WORKSPACE": workspace},
        )
        assert response.status_code == 200
        assert response.json() == [f"ws-{workspace}-label"]
    assert [instance.workspace for instance in created] == ["ws1", "ws2"]


def test_header_value_is_sanitized_before_lookup():
    client, default, manager, created = _build_client()
    headers = {**_AUTH_HEADERS, "LIGHTRAG-WORKSPACE": "team one!"}
    response = client.get("/graph/label/list", headers=headers)
    assert response.status_code == 200
    assert [instance.workspace for instance in created] == ["team_one_"]


def test_explicit_default_workspace_header_uses_default_instance():
    client, default, manager, created = _build_client()
    headers = {**_AUTH_HEADERS, "LIGHTRAG-WORKSPACE": "   "}
    response = client.get("/graph/label/list", headers=headers)
    assert response.status_code == 200
    assert response.json() == ["default-label"]
    assert created == []


# ---------------------------------------------------------------------------
# No manager on app.state (single-workspace embedding)
# ---------------------------------------------------------------------------


def test_without_manager_default_instance_always_serves():
    client, default, manager, created = _build_client(with_manager=False)
    headers = {**_AUTH_HEADERS, "LIGHTRAG-WORKSPACE": "beta"}
    response = client.get("/graph/label/list", headers=headers)
    assert response.status_code == 200
    assert response.json() == ["default-label"]
    default.get_graph_labels.assert_awaited_once()
