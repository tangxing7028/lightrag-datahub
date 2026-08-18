"""Two-stage summary-first retrieval in ``_get_vector_context``.

With ``QueryParam.enable_summary_search`` set, the vector stage first queries
the workspace's doc_summaries store (top_k=3), then restricts the chunk
search to the matched ``full_doc_id`` values through the ``doc_ids``
allow-list. Every non-hit case — no store configured, namespace not created
yet, summary-stage backend error, zero summary hits — must fall back to the
regular full-corpus chunk retrieval (``doc_ids=None``), never to an error
or an empty result.
"""

import pytest

from lightrag.base import QueryParam
from lightrag.operate import _get_vector_context

pytestmark = pytest.mark.offline


class _RecordingChunksVDB:
    cosine_better_than_threshold = 0.2

    def __init__(self):
        self.calls: list[dict] = []

    async def query(self, query, top_k, query_embedding=None, doc_ids=None, cosine_threshold=None):
        self.calls.append({"doc_ids": doc_ids, "top_k": top_k})
        return [{"content": "chunk", "id": "chunk-1", "file_path": "a.txt"}]


class _SummariesVDB:
    def __init__(self, results, *, exists=True, raises=False):
        self._results = results
        self._exists = exists
        self._raises = raises
        self.queries = 0

    async def probe_collection_exists(self):
        return self._exists

    async def query(
        self,
        query,
        top_k,
        query_embedding=None,
        doc_ids=None,
        cosine_threshold=None,
    ):
        self.queries += 1
        if self._raises:
            raise RuntimeError("summaries backend down")
        return self._results


def _param(**overrides) -> QueryParam:
    return QueryParam(enable_summary_search=True, **overrides)


async def test_summary_hits_restrict_chunk_search():
    chunks_vdb = _RecordingChunksVDB()
    summaries = _SummariesVDB(
        [
            {"full_doc_id": "doc-1", "content": "s1"},
            {"full_doc_id": "doc-2", "content": "s2"},
            {"full_doc_id": "doc-1", "content": "dup"},  # deduped
            {"content": "no doc id"},  # ignored
        ]
    )

    result = await _get_vector_context(
        "query", chunks_vdb, _param(), query_embedding=[0.1], doc_summaries_vdb=summaries
    )

    assert len(result) == 1
    assert chunks_vdb.calls[0]["doc_ids"] == ["doc-1", "doc-2"]


async def test_summary_hits_intersect_with_caller_allow_list():
    chunks_vdb = _RecordingChunksVDB()
    summaries = _SummariesVDB(
        [{"full_doc_id": "doc-1"}, {"full_doc_id": "doc-2"}]
    )

    await _get_vector_context(
        "query",
        chunks_vdb,
        _param(doc_ids=["doc-2", "doc-3"]),
        query_embedding=[0.1],
        doc_summaries_vdb=summaries,
    )

    assert chunks_vdb.calls[0]["doc_ids"] == ["doc-2"]


async def test_no_summary_hits_falls_back_to_full_retrieval():
    chunks_vdb = _RecordingChunksVDB()
    summaries = _SummariesVDB([])

    result = await _get_vector_context(
        "query", chunks_vdb, _param(), query_embedding=[0.1], doc_summaries_vdb=summaries
    )

    assert len(result) == 1
    assert chunks_vdb.calls[0]["doc_ids"] is None


async def test_missing_summaries_namespace_falls_back():
    chunks_vdb = _RecordingChunksVDB()
    summaries = _SummariesVDB([{"full_doc_id": "doc-1"}], exists=False)

    result = await _get_vector_context(
        "query", chunks_vdb, _param(), query_embedding=[0.1], doc_summaries_vdb=summaries
    )

    assert len(result) == 1
    assert summaries.queries == 0  # probed first, never queried
    assert chunks_vdb.calls[0]["doc_ids"] is None


async def test_summary_stage_error_falls_back():
    chunks_vdb = _RecordingChunksVDB()
    summaries = _SummariesVDB([], raises=True)

    result = await _get_vector_context(
        "query", chunks_vdb, _param(), query_embedding=[0.1], doc_summaries_vdb=summaries
    )

    assert len(result) == 1
    assert chunks_vdb.calls[0]["doc_ids"] is None


async def test_no_summaries_store_falls_back():
    chunks_vdb = _RecordingChunksVDB()

    result = await _get_vector_context(
        "query", chunks_vdb, _param(), query_embedding=[0.1], doc_summaries_vdb=None
    )

    assert len(result) == 1
    assert chunks_vdb.calls[0]["doc_ids"] is None


async def test_flag_off_never_touches_summaries_store():
    chunks_vdb = _RecordingChunksVDB()
    summaries = _SummariesVDB([{"full_doc_id": "doc-1"}])

    result = await _get_vector_context(
        "query", chunks_vdb, QueryParam(), query_embedding=[0.1], doc_summaries_vdb=summaries
    )

    assert len(result) == 1
    assert summaries.queries == 0
    assert chunks_vdb.calls[0]["doc_ids"] is None
