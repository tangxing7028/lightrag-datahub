"""aquery_data must forward doc_ids / cosine_threshold to the retrieval entry.

Regression for a real-PG finding: ``/query`` (aquery_llm) honoured doc_ids,
but ``/query/data`` (aquery_data) rebuilt an internal QueryParam with an
explicit field list that predated the fork additions, silently dropping
``doc_ids`` and ``cosine_threshold``. The retrieval then ran unfiltered and
an authorization-out / empty-list query still returned chunks (fail-open).

This test pins the forwarding contract at the data_param boundary, where
the drop happened, for both the naive (chunks-only) and KG (mix) paths.
"""

import asyncio
import sys
import unittest.mock as mock

import pytest

sys.path.insert(0, ".")

from lightrag.base import QueryParam
from lightrag.lightrag import LightRAG


class _FakeStorage:
    pass


def _bare_lightrag() -> LightRAG:
    """A LightRAG instance without running __init__ (heavy I/O)."""
    rag = LightRAG.__new__(LightRAG)
    rag.chunk_entity_relation_graph = _FakeStorage()
    rag.entities_vdb = _FakeStorage()
    rag.relationships_vdb = _FakeStorage()
    rag.text_chunks = _FakeStorage()
    rag.chunks_vdb = _FakeStorage()
    rag.llm_response_cache = _FakeStorage()
    rag._build_global_config = mock.MagicMock(return_value={})
    rag._query_done = mock.AsyncMock()
    return rag


@pytest.mark.offline
def test_aquery_data_naive_forwards_doc_ids_and_cosine_threshold():
    rag = _bare_lightrag()
    captured = {}

    async def fake_naive_query(query, chunks_vdb, param, global_config, **kwargs):
        captured["doc_ids"] = param.doc_ids
        captured["cosine_threshold"] = param.cosine_threshold
        captured["enable_summary_search"] = param.enable_summary_search
        return None

    param = QueryParam(
        mode="naive",
        doc_ids=["doc-1", "doc-2"],
        cosine_threshold=0.55,
        enable_summary_search=True,
    )
    with mock.patch("lightrag.lightrag.naive_query", side_effect=fake_naive_query):
        asyncio.run(rag.aquery_data("what is covered", param))

    assert captured["doc_ids"] == ["doc-1", "doc-2"]
    assert captured["cosine_threshold"] == 0.55
    assert captured["enable_summary_search"] is True


@pytest.mark.offline
def test_aquery_data_empty_doc_ids_survives_on_kg_path():
    """Empty list = fail-closed: it must reach kg_query as [] (not None)."""
    rag = _bare_lightrag()
    captured = {}

    async def fake_kg_query(query, graph, entities_vdb, relationships_vdb,
                            text_chunks, param, global_config, **kwargs):
        captured["doc_ids"] = param.doc_ids
        return None

    param = QueryParam(mode="mix", doc_ids=[])
    with mock.patch("lightrag.lightrag.kg_query", side_effect=fake_kg_query):
        asyncio.run(rag.aquery_data("what is covered", param))

    assert captured["doc_ids"] == []
    assert captured["doc_ids"] is not None
