"""Regression coverage for recoverable empty entity-extraction responses."""

from __future__ import annotations

import asyncio

import pytest
from tenacity import RetryError

from lightrag.llm.openai import InvalidResponseError
from lightrag.utils import EntityExtractionDegradationTally, Tokenizer, TokenizerInterface


class _TokenizerImpl(TokenizerInterface):
    def encode(self, content: str):
        return [ord(char) for char in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


def _config(extract_func) -> dict:
    return {
        "llm_model_func": extract_func,
        "role_llm_funcs": {
            "extract": extract_func,
            "keyword": extract_func,
            "query": extract_func,
            "vlm": extract_func,
        },
        "entity_extract_max_gleaning": 0,
        "entity_extract_max_records": 100,
        "entity_extract_max_entities": 40,
        "addon_params": {},
        "tokenizer": Tokenizer("test", _TokenizerImpl()),
        "llm_model_max_async": 3,
    }


def _chunks() -> dict[str, dict]:
    return {
        "chunk-empty": {
            "tokens": 14,
            "content": "Empty response.",
            "full_doc_id": "doc-1",
            "chunk_order_index": 0,
            "file_path": "fixture.pdf",
        },
        "chunk-good": {
            "tokens": 14,
            "content": "Good response.",
            "full_doc_id": "doc-1",
            "chunk_order_index": 1,
            "file_path": "fixture.pdf",
        },
    }


def _retry_error(message: str) -> RetryError:
    attempt = asyncio.Future()
    attempt.set_exception(InvalidResponseError(message))
    return RetryError(attempt)


async def _extract_with_one_empty_response(prompt: str, **_kwargs) -> str:
    if "Empty response." in prompt:
        raise _retry_error(
            "Received empty content from OpenAI API "
            "(finish_reason=stop, completion_tokens=0, reasoning_tokens=n/a, "
            "reasoning_content_len=0): model produced no output"
        )
    return "(entity<|#|>GOOD<|#|>CONCEPT<|#|>Recovered chunk)<|COMPLETE|>"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_empty_model_output_after_retries_degrades_one_chunk_only():
    from lightrag.operate import extract_entities

    tally = EntityExtractionDegradationTally()
    results = await extract_entities(
        chunks=_chunks(),
        global_config=_config(_extract_with_one_empty_response),
        degradation_tally=tally,
    )

    assert results[0] == ({}, {})
    assert "GOOD" in results[1][0]
    assert tally.as_metadata() == {
        "reason": "empty_model_output_after_retries",
        "events": 1,
        "affected": 1,
        "stages": {"initial": 1},
        "samples": ["chunk-empty"],
    }


@pytest.mark.offline
@pytest.mark.asyncio
async def test_nonempty_response_retry_error_remains_fatal():
    from lightrag.operate import extract_entities

    async def _invalid_response(prompt: str, **_kwargs) -> str:
        if "Empty response." in prompt:
            raise _retry_error("Invalid response from OpenAI API")
        return "(entity<|#|>GOOD<|#|>CONCEPT<|#|>Recovered chunk)<|COMPLETE|>"

    with pytest.raises(RetryError):
        await extract_entities(chunks=_chunks(), global_config=_config(_invalid_response))
