"""PGVectorStorage document allow-list filtering and per-query cosine threshold.

These tests assert the assembled SQL and bound parameters only; no real
database is involved (``db`` is an AsyncMock). The filter contract:

- ``doc_ids=None`` leaves the base query untouched (4 positional params).
- chunks namespace: ``AND full_doc_id = ANY($5)``.
- entities/relationships namespaces: ``chunk_ids`` overlap against the chunks
  table derived from the same embedding model suffix.
- ``doc_ids=[]`` fails closed: zero results without touching the database.
- ``cosine_threshold`` overrides the storage-level threshold for one query.
"""

import numpy as np
import pytest
from unittest.mock import AsyncMock

from lightrag.kg.postgres_impl import PGVectorStorage
from lightrag.namespace import NameSpace
from lightrag.utils import EmbeddingFunc

pytestmark = pytest.mark.offline


@pytest.fixture
def mock_embedding_func():
    async def embed_func(texts, **kwargs):
        return np.array([[0.1] * 768 for _ in texts])

    return EmbeddingFunc(embedding_dim=768, func=embed_func, model_name="test_model")


@pytest.fixture
def mock_pg_db():
    db = AsyncMock()
    db.workspace = "test_ws"
    db.vector_index_type = None
    db.query = AsyncMock(return_value=[])
    return db


def _make_storage(namespace, mock_embedding_func, mock_pg_db):
    config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
    }
    storage = PGVectorStorage(
        namespace=namespace,
        global_config=config,
        embedding_func=mock_embedding_func,
        workspace="test_ws",
    )
    storage.db = mock_pg_db
    return storage


async def _run_query(storage, **kwargs):
    kwargs.setdefault("query_embedding", [0.1] * 768)
    return await storage.query("test query", top_k=5, **kwargs)


@pytest.mark.asyncio
async def test_query_without_doc_ids_keeps_base_sql(mock_embedding_func, mock_pg_db):
    storage = _make_storage(
        NameSpace.VECTOR_STORE_CHUNKS, mock_embedding_func, mock_pg_db
    )
    await _run_query(storage)

    sql, params = _captured_sql_params(mock_pg_db)
    assert "$5" not in sql
    assert "full_doc_id" not in sql
    assert len(params) == 4
    assert params[0] == "test_ws"
    assert params[1] == pytest.approx(0.8)  # 1 - 0.2 storage-level threshold
    assert params[2] == 5


@pytest.mark.asyncio
async def test_chunks_query_with_doc_ids_filters_full_doc_id(
    mock_embedding_func, mock_pg_db
):
    storage = _make_storage(
        NameSpace.VECTOR_STORE_CHUNKS, mock_embedding_func, mock_pg_db
    )
    await _run_query(storage, doc_ids=["doc-1", "doc-2"])

    sql, params = _captured_sql_params(mock_pg_db)
    assert "AND full_doc_id = ANY($5)" in sql
    assert params[4] == ["doc-1", "doc-2"]


@pytest.mark.asyncio
async def test_entities_query_with_doc_ids_filters_via_chunk_overlap(
    mock_embedding_func, mock_pg_db
):
    storage = _make_storage(
        NameSpace.VECTOR_STORE_ENTITIES, mock_embedding_func, mock_pg_db
    )
    await _run_query(storage, doc_ids=["doc-1"])

    sql, params = _captured_sql_params(mock_pg_db)
    assert "chunk_ids &&" in sql
    # The chunks table is derived from the same embedding model suffix.
    assert "FROM LIGHTRAG_VDB_CHUNKS_test_model_768d" in sql
    assert "full_doc_id = ANY($5)" in sql
    assert params[4] == ["doc-1"]


@pytest.mark.asyncio
async def test_relationships_query_with_doc_ids_filters_via_chunk_overlap(
    mock_embedding_func, mock_pg_db
):
    storage = _make_storage(
        NameSpace.VECTOR_STORE_RELATIONSHIPS, mock_embedding_func, mock_pg_db
    )
    await _run_query(storage, doc_ids=["doc-9"])

    sql, params = _captured_sql_params(mock_pg_db)
    assert "chunk_ids &&" in sql
    assert "FROM LIGHTRAG_VDB_CHUNKS_test_model_768d" in sql
    assert params[4] == ["doc-9"]


@pytest.mark.asyncio
async def test_empty_doc_ids_fails_closed_without_db(mock_embedding_func, mock_pg_db):
    storage = _make_storage(
        NameSpace.VECTOR_STORE_CHUNKS, mock_embedding_func, mock_pg_db
    )
    results = await _run_query(storage, doc_ids=[])

    assert results == []
    mock_pg_db.query.assert_not_called()


@pytest.mark.asyncio
async def test_cosine_threshold_overrides_storage_default(
    mock_embedding_func, mock_pg_db
):
    storage = _make_storage(
        NameSpace.VECTOR_STORE_CHUNKS, mock_embedding_func, mock_pg_db
    )
    await _run_query(storage, cosine_threshold=0.65)

    _, params = _captured_sql_params(mock_pg_db)
    assert params[1] == pytest.approx(0.35)  # 1 - 0.65 per-query override


@pytest.mark.asyncio
async def test_get_doc_ids_by_chunk_ids(mock_embedding_func, mock_pg_db):
    mock_pg_db.query = AsyncMock(
        return_value=[
            {"id": "chunk-1", "full_doc_id": "doc-1"},
            {"id": "chunk-2", "full_doc_id": "doc-2"},
        ]
    )
    storage = _make_storage(
        NameSpace.VECTOR_STORE_CHUNKS, mock_embedding_func, mock_pg_db
    )
    mapping = await storage.get_doc_ids_by_chunk_ids(["chunk-1", "chunk-2"])

    assert mapping == {"chunk-1": "doc-1", "chunk-2": "doc-2"}
    sql, params = _captured_sql_params(mock_pg_db)
    assert "SELECT id, full_doc_id" in sql
    assert params == ["test_ws", ["chunk-1", "chunk-2"]]


@pytest.mark.asyncio
async def test_get_doc_ids_by_chunk_ids_empty_input(mock_embedding_func, mock_pg_db):
    storage = _make_storage(
        NameSpace.VECTOR_STORE_CHUNKS, mock_embedding_func, mock_pg_db
    )
    assert await storage.get_doc_ids_by_chunk_ids([]) == {}
    mock_pg_db.query.assert_not_called()


@pytest.mark.asyncio
async def test_get_doc_ids_by_chunk_ids_wrong_namespace(
    mock_embedding_func, mock_pg_db
):
    storage = _make_storage(
        NameSpace.VECTOR_STORE_ENTITIES, mock_embedding_func, mock_pg_db
    )
    assert await storage.get_doc_ids_by_chunk_ids(["chunk-1"]) == {}
    mock_pg_db.query.assert_not_called()


def _captured_sql_params(mock_pg_db):
    """Extract (sql, params) from the most recent db.query call."""
    assert mock_pg_db.query.called
    _, kwargs = mock_pg_db.query.call_args
    return mock_pg_db.query.call_args[0][0], kwargs["params"]
