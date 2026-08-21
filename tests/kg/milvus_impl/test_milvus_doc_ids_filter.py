"""Milvus ``doc_ids`` allow-list filtering in ``MilvusVectorDBStorage.query``.

The chunks and doc_summaries collections carry a ``full_doc_id`` VARCHAR
field with an INVERTED scalar index, so ``query(doc_ids=...)`` must translate
the allow-list into a ``full_doc_id in [...]`` filter expression. Namespaces
without that field (entities/relationships) keep the historical
warn-and-ignore behavior, and an explicit empty allow-list fails closed
without touching the server (mirrors PGVectorStorage).
"""

from unittest.mock import MagicMock, patch

import pytest

from lightrag.kg.milvus_impl import MilvusVectorDBStorage
from lightrag.kg.shared_storage import initialize_share_data, finalize_share_data

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def setup_shared_data():
    initialize_share_data()
    yield
    finalize_share_data()


def _make_storage(
    *,
    namespace: str = "test_chunks",
    meta_fields: set[str] | None = None,
):
    mock_embedding_func = MagicMock()
    mock_embedding_func.embedding_dim = 128
    storage = MilvusVectorDBStorage(
        namespace=namespace,
        workspace="test_workspace",
        global_config={
            "embedding_batch_num": 100,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": 0.3,
            },
        },
        embedding_func=mock_embedding_func,
        meta_fields=meta_fields or {"full_doc_id", "content", "file_path"},
    )
    storage._client = MagicMock()
    storage._client.search.return_value = [
        [
            {
                "entity": {"content": "chunk text", "full_doc_id": "doc-1"},
                "id": "chunk-1",
                "distance": 0.9,
            }
        ]
    ]
    return storage


@pytest.mark.asyncio
async def test_query_applies_full_doc_id_filter():
    storage = _make_storage()

    results = await storage.query(
        "query",
        top_k=5,
        query_embedding=[0.1] * 128,
        doc_ids=["doc-1", "doc-2"],
    )

    assert len(results) == 1
    storage._client.search.assert_called_once()
    _, kwargs = storage._client.search.call_args
    assert kwargs["filter"] == 'full_doc_id in ["doc-1", "doc-2"]'


@pytest.mark.asyncio
async def test_query_doc_ids_filter_escapes_special_characters():
    storage = _make_storage()

    await storage.query(
        "query",
        top_k=5,
        query_embedding=[0.1] * 128,
        doc_ids=['doc"1', "doc\\2"],
    )

    _, kwargs = storage._client.search.call_args
    assert kwargs["filter"] == 'full_doc_id in ["doc\\"1", "doc\\\\2"]'


@pytest.mark.asyncio
async def test_query_empty_doc_ids_fails_closed_without_search():
    storage = _make_storage()

    results = await storage.query(
        "query",
        top_k=5,
        query_embedding=[0.1] * 128,
        doc_ids=[],
    )

    assert results == []
    storage._client.search.assert_not_called()


@pytest.mark.asyncio
async def test_query_without_doc_ids_passes_no_filter():
    storage = _make_storage()

    await storage.query("query", top_k=5, query_embedding=[0.1] * 128)

    _, kwargs = storage._client.search.call_args
    assert kwargs["filter"] == ""


@pytest.mark.asyncio
async def test_query_doc_ids_ignored_when_namespace_lacks_full_doc_id():
    storage = _make_storage(
        namespace="test_entities",
        meta_fields={"entity_name", "content"},
    )

    with patch("lightrag.kg.milvus_impl.logger") as mock_logger:
        results = await storage.query(
            "query",
            top_k=5,
            query_embedding=[0.1] * 128,
            doc_ids=["doc-1"],
        )

    assert len(results) == 1
    _, kwargs = storage._client.search.call_args
    assert kwargs["filter"] == ""
    assert any(
        "full_doc_id" in str(call)
        for call in mock_logger.warning.call_args_list
    )
