"""Caller-assigned doc ids on the pending_parse (server upload) path.

``apipeline_enqueue_documents(docs_format="pending_parse", ids=[...])`` must
use the supplied id as the document's full_doc_id end to end while still
deferring extraction to the parse worker — previously ``ids`` forced the
RAW direct-insert path, so an uploaded file could never carry a
caller-assigned id. These tests pin both the new combination and the two
pre-existing behaviors (id-less pending_parse → md5(file_path); RAW + ids →
verbatim direct insert).

The enqueue path only touches ``full_docs`` / ``doc_status`` (plus the
shared pipeline status), so the tests initialize exactly those two JSON
storages instead of the whole storage stack — the vector/graph backends
are irrelevant here and several require optional native dependencies the
offline test environment does not install.
"""

import asyncio
import os
from pathlib import Path

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.constants import FULL_DOCS_FORMAT_PENDING_PARSE, FULL_DOCS_FORMAT_RAW
from lightrag.kg import shared_storage
from lightrag.kg.json_doc_status_impl import JsonDocStatusStorage
from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.utils import EmbeddingFunc, Tokenizer, compute_mdhash_id

pytestmark = pytest.mark.offline


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _mock_embedding(texts: list[str]) -> np.ndarray:
    return np.random.rand(len(texts), 32)


async def _mock_llm(prompt, **kwargs):
    return "ok"


async def _new_rag(tmp_path: Path) -> LightRAG:
    """LightRAG with only the enqueue-relevant storages initialized."""
    # MilvusVectorDBStorage is only class-resolved (never instantiated) in
    # this rig, but __post_init__ still verifies its declared env vars.
    os.environ.setdefault("MILVUS_URI", "http://localhost:19530")
    os.environ.setdefault("MILVUS_DB_NAME", "lightrag_test")
    rag = LightRAG(
        working_dir=str(tmp_path),
        workspace=f"enqueue-docid-{tmp_path.name}",
        llm_model_func=_mock_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=32,
            max_token_size=4096,
            func=_mock_embedding,
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        max_parallel_insert=1,
        # __post_init__ eagerly resolves the configured storage CLASSES
        # (never instantiated here — the two enqueue-relevant storages are
        # replaced below). The default NanoVectorDBStorage module needs the
        # optional nano_vectordb package, absent from this offline
        # environment; MilvusVectorDBStorage imports cleanly and is never
        # constructed.
        vector_storage="MilvusVectorDBStorage",
    )
    shared_storage.initialize_share_data()
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    global_config = {"working_dir": str(tmp_path)}
    rag.full_docs = JsonKVStorage(
        namespace="full_docs",
        workspace=rag.workspace,
        global_config=global_config,
        embedding_func=rag.embedding_func,
    )
    rag.doc_status = JsonDocStatusStorage(
        namespace="doc_status",
        workspace=rag.workspace,
        global_config=global_config,
        embedding_func=rag.embedding_func,
    )
    await rag.full_docs.initialize()
    await rag.doc_status.initialize()
    return rag


def test_pending_parse_with_explicit_ids_uses_caller_doc_id(tmp_path):
    custom_id = "8012345678901234567890"

    async def _run():
        rag = await _new_rag(tmp_path)
        await rag.apipeline_enqueue_documents(
            "",
            ids=[custom_id],
            file_paths="report.pdf",
            docs_format=FULL_DOCS_FORMAT_PENDING_PARSE,
            parse_engine="native",
        )
        full_doc = await rag.full_docs.get_by_id(custom_id)
        status_doc = await rag.doc_status.get_by_id(custom_id)
        hash_id = compute_mdhash_id("report.pdf", prefix="doc-")
        hash_doc = await rag.full_docs.get_by_id(hash_id)
        return full_doc, status_doc, hash_doc

    full_doc, status_doc, hash_doc = asyncio.run(_run())

    assert full_doc is not None, "full_docs row missing under the caller doc_id"
    # Still a parse-worker document: empty body, pending_parse marker,
    # engine directive persisted — NOT a RAW direct insert.
    assert full_doc.get("parse_format") == FULL_DOCS_FORMAT_PENDING_PARSE
    assert full_doc.get("content", "") == ""
    assert full_doc.get("file_path") == "report.pdf"
    assert full_doc.get("parse_engine") == "native"

    assert status_doc is not None, "doc_status row missing under the caller doc_id"
    assert status_doc.get("status") == "pending"
    assert status_doc.get("file_path") == "report.pdf"

    assert hash_doc is None, "md5(file_path) row must not exist when ids override"


def test_pending_parse_without_ids_keeps_md5_doc_id(tmp_path):
    """Regression: the id-less upload path still derives md5(file_path)."""

    async def _run():
        rag = await _new_rag(tmp_path)
        await rag.apipeline_enqueue_documents(
            "",
            file_paths="plain.pdf",
            docs_format=FULL_DOCS_FORMAT_PENDING_PARSE,
            parse_engine="native",
        )
        hash_id = compute_mdhash_id("plain.pdf", prefix="doc-")
        return await rag.full_docs.get_by_id(hash_id)

    full_doc = asyncio.run(_run())
    assert full_doc is not None
    assert full_doc.get("parse_format") == FULL_DOCS_FORMAT_PENDING_PARSE


def test_raw_with_explicit_ids_stays_direct_insert(tmp_path):
    """Regression: ids + RAW (the ainsert path) still enqueues sanitized
    verbatim content with no parse-worker deferral."""
    custom_id = "8012345678901234567891"

    async def _run():
        rag = await _new_rag(tmp_path)
        await rag.apipeline_enqueue_documents(
            "Already extracted body text.",
            ids=[custom_id],
            file_paths="notes.md",
            docs_format=FULL_DOCS_FORMAT_RAW,
        )
        return await rag.full_docs.get_by_id(custom_id)

    full_doc = asyncio.run(_run())
    assert full_doc is not None
    assert full_doc.get("parse_format") == FULL_DOCS_FORMAT_RAW
    assert full_doc.get("content") == "Already extracted body text."
