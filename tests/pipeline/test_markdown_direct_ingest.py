"""The real upload/pipeline contract indexes Markdown without a parser service."""

import importlib
import json
import sys
from io import BytesIO
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG, QueryParam
from lightrag.base import DocStatus
from lightrag.utils import EmbeddingFunc, Tokenizer

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
routes = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

pytestmark = pytest.mark.offline


class _Tokenizer:
    def encode(self, content):
        return [ord(char) for char in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


async def _embedding(texts):
    return np.ones((len(texts), 8), dtype=float)


async def _unexpected_call(*args, **kwargs):
    raise AssertionError("Direct Markdown ingestion must not call parsers or an LLM")


@pytest.mark.parametrize("kb_parser", ["mineru", "docling"])
@pytest.mark.parametrize("skip_kg", [True, False])
async def test_upload_markdown_reaches_processed_and_is_retrievable(
    tmp_path, monkeypatch, kb_parser, skip_kg
):
    monkeypatch.setattr(routes, "global_args", SimpleNamespace(max_upload_size=None))
    monkeypatch.setenv("INPUT_DIR", str(tmp_path / "input"))
    for impl in (
        "lightrag.parser.native_dispatch.NativeParser.parse",
        "lightrag.parser.external.mineru.parser.MinerUParser.parse",
        "lightrag.parser.external.docling.parser.DoclingParser.parse",
    ):
        monkeypatch.setattr(impl, _unexpected_call)

    workspace = f"markdown-direct-{uuid4().hex[:8]}"
    extraction_calls = []

    async def extract_entities(prompt, **kwargs):
        extraction_calls.append(prompt)
        return "<|COMPLETE|>"

    rag = LightRAG(
        working_dir=str(tmp_path / "storage"),
        workspace=workspace,
        llm_model_func=_unexpected_call if skip_kg else extract_entities,
        embedding_func=EmbeddingFunc(embedding_dim=8, func=_embedding),
        tokenizer=Tokenizer("test", _Tokenizer()),
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    try:
        manager = routes.DocumentManager(str(tmp_path / "input"), workspace=workspace)
        router = routes.create_document_routes(rag, manager)
        upload = next(
            route.endpoint
            for route in router.routes
            if route.name == "upload_to_input_dir"
        )
        body = (
            "# MarkdownDirect20261014\n\n"
            "\u4e2d\u6587\u539f\u6587\n\n"
            "| Item | Value |\n| --- | --- |\n| Test | 42 |\n\n"
            "```python\nprint('kept')\n```\n\n"
            "![image](https://example.invalid/never-fetch.png)\n"
        )
        doc_id = "8012345678901234567800"
        managed = set()
        response = await upload(
            managed,
            routes.UploadFile(
                filename="original.MD", file=BytesIO(b"\xef\xbb\xbf" + body.encode())
            ),
            doc_id=doc_id,
            parser=kb_parser,
            parse_method="auto",
            enable_image="true",
            enable_table="true",
            enable_equation="true",
            skip_entity_extract=str(skip_kg).lower(),
            embedding_dim=8,
            datahub_job_id=None,
            chunking=json.dumps(
                {
                    "strategy": "fixed_token",
                    "params": {"chunk_token_size": 64, "chunk_overlap_token_size": 8},
                }
            ),
        )
        for task in list(managed):
            await task

        assert response.status == "success"
        status = await rag.doc_status.get_by_id(doc_id)
        assert status["status"] == DocStatus.PROCESSED
        assert status["chunks_count"] > 1
        assert status["metadata"]["parse_stage_skipped"] is True
        assert status["metadata"]["parse_engine"] == "legacy"
        full_doc = await rag.full_docs.get_by_id(doc_id)
        assert full_doc["content"] == body
        assert ("!" in full_doc["process_options"]) == skip_kg
        assert bool(extraction_calls) == (not skip_kg)
        result = await rag.aquery_data(
            "MarkdownDirect20261014", QueryParam(mode="naive", enable_rerank=False)
        )
        assert "MarkdownDirect20261014" in str(result)
        assert doc_id in str(result)
    finally:
        await rag.finalize_storages()
