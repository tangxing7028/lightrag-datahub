"""Hybrid/mix retrieval branches in _perform_kg_search run concurrently.

The local (_get_node_data), global (_get_edge_data) and vector-chunk
(_get_vector_context) branches have no data dependencies, so they are
gathered instead of awaited serially. These tests pin two behaviors:

- the enabled branches actually overlap in time;
- a failing branch cancels the survivors and its exception propagates
  (gather without return_exceptions + explicit cancellation drain).
"""

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from lightrag.base import QueryParam

pytestmark = pytest.mark.offline


def _make_text_chunks_db():
    mock = MagicMock()
    mock.embedding_func = None  # skip embedding pre-computation
    mock.global_config = {"kg_chunk_pick_method": "VECTOR"}
    return mock


def _make_chunks_vdb():
    mock = MagicMock()
    mock.cosine_better_than_threshold = 0.2
    return mock


async def _run_search(
    monkeypatch,
    branches,
    mode="mix",
    *,
    enable_summary_search=False,
    doc_summaries_vdb=None,
):
    """Run _perform_kg_search with the given fake branches patched in."""
    import lightrag.operate as operate

    monkeypatch.setattr(operate, "_get_node_data", branches["local"])
    monkeypatch.setattr(operate, "_get_edge_data", branches["global"])
    monkeypatch.setattr(operate, "_get_vector_context", branches["vector"])

    return await operate._perform_kg_search(
        query="test query",
        ll_keywords="entity keywords",
        hl_keywords="theme keywords",
        knowledge_graph_inst=MagicMock(),
        entities_vdb=MagicMock(),
        relationships_vdb=MagicMock(),
        text_chunks_db=_make_text_chunks_db(),
        query_param=QueryParam(
            mode=mode, top_k=5, enable_summary_search=enable_summary_search
        ),
        chunks_vdb=_make_chunks_vdb(),
        doc_summaries_vdb=doc_summaries_vdb,
    )


@pytest.mark.asyncio
async def test_mix_mode_branches_run_concurrently(monkeypatch):
    """All three branches overlap: total time stays near one branch duration."""
    branch_delay = 0.15

    async def fake_local(*args, **kwargs):
        await asyncio.sleep(branch_delay)
        return [{"entity_name": "e1", "rank": 1}], []

    async def fake_global(*args, **kwargs):
        await asyncio.sleep(branch_delay)
        return [], []

    async def fake_vector(*args, **kwargs):
        await asyncio.sleep(branch_delay)
        return [{"content": "c", "chunk_id": "chunk-1"}]

    start = time.perf_counter()
    result = await _run_search(
        monkeypatch,
        {"local": fake_local, "global": fake_global, "vector": fake_vector},
    )
    elapsed = time.perf_counter() - start

    assert elapsed < branch_delay * 2, (
        f"branches appear serial: {elapsed:.3f}s >= {branch_delay * 2:.3f}s"
    )
    assert [e["entity_name"] for e in result["final_entities"]] == ["e1"]
    assert [c["chunk_id"] for c in result["vector_chunks"]] == ["chunk-1"]
    assert result["chunk_tracking"]["chunk-1"]["source"] == "C"


@pytest.mark.asyncio
async def test_failing_branch_cancels_survivors_and_raises(monkeypatch):
    """A branch failure must cancel the other branches and propagate."""
    survivor_cancelled = asyncio.Event()

    async def failing_local(*args, **kwargs):
        await asyncio.sleep(0.01)
        raise RuntimeError("entities backend down")

    async def slow_global(*args, **kwargs):
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            survivor_cancelled.set()
            raise
        return [], []

    async def slow_vector(*args, **kwargs):
        await asyncio.sleep(30)
        return []

    start = time.perf_counter()
    with pytest.raises(RuntimeError, match="entities backend down"):
        await _run_search(
            monkeypatch,
            {"local": failing_local, "global": slow_global, "vector": slow_vector},
        )
    elapsed = time.perf_counter() - start

    assert survivor_cancelled.is_set(), "surviving branch was not cancelled"
    assert elapsed < 5, f"search did not abort promptly ({elapsed:.3f}s)"


@pytest.mark.asyncio
async def test_disabled_branches_stay_disabled(monkeypatch):
    """Branch enable conditions are unchanged: no keywords -> no KG branches."""
    called = []

    async def fake_local(*args, **kwargs):
        called.append("local")
        return [], []

    async def fake_global(*args, **kwargs):
        called.append("global")
        return [], []

    async def fake_vector(*args, **kwargs):
        called.append("vector")
        return [{"content": "c", "chunk_id": "chunk-1"}]

    import lightrag.operate as operate

    monkeypatch.setattr(operate, "_get_node_data", fake_local)
    monkeypatch.setattr(operate, "_get_edge_data", fake_global)
    monkeypatch.setattr(operate, "_get_vector_context", fake_vector)

    # mix mode with empty keywords: only the vector branch may run
    await operate._perform_kg_search(
        query="test query",
        ll_keywords="",
        hl_keywords="",
        knowledge_graph_inst=MagicMock(),
        entities_vdb=MagicMock(),
        relationships_vdb=MagicMock(),
        text_chunks_db=_make_text_chunks_db(),
        query_param=QueryParam(mode="mix", top_k=5),
        chunks_vdb=_make_chunks_vdb(),
    )
    assert called == ["vector"]

    # hybrid mode: the vector branch must not run even with keywords
    called.clear()
    await operate._perform_kg_search(
        query="test query",
        ll_keywords="entity keywords",
        hl_keywords="theme keywords",
        knowledge_graph_inst=MagicMock(),
        entities_vdb=MagicMock(),
        relationships_vdb=MagicMock(),
        text_chunks_db=_make_text_chunks_db(),
        query_param=QueryParam(mode="hybrid", top_k=5),
        chunks_vdb=_make_chunks_vdb(),
    )
    assert sorted(called) == ["global", "local"]


@pytest.mark.asyncio
async def test_hybrid_summary_search_enables_vector_branch(monkeypatch):
    """Summary-first retrieval must run inside the hybrid query path."""
    called = []

    async def fake_local(*args, **kwargs):
        called.append("local")
        return [], []

    async def fake_global(*args, **kwargs):
        called.append("global")
        return [], []

    async def fake_vector(*args, **kwargs):
        called.append("vector")
        assert kwargs["doc_summaries_vdb"] is summaries
        return [{"content": "c", "chunk_id": "chunk-1"}]

    summaries = MagicMock()
    result = await _run_search(
        monkeypatch,
        {"local": fake_local, "global": fake_global, "vector": fake_vector},
        mode="hybrid",
        enable_summary_search=True,
        doc_summaries_vdb=summaries,
    )

    assert sorted(called) == ["global", "local", "vector"]
    assert [chunk["chunk_id"] for chunk in result["vector_chunks"]] == ["chunk-1"]
