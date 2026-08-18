"""Tests for the `/documents/upload` ``summary_model_config`` contract.

The ai-service side sends the summary-scenario model override as a JSON
string of ``{"model": ..., "base_url": ..., "api_key": ...}`` (snake_case;
any field may be an empty string meaning "use the workspace default").
``_parse_summary_model_config`` must accept that payload, normalize
``base_url`` onto the role-metadata ``host`` key the role builder
understands, and drop blank values so the server-side fallback chain still
applies per field.
"""

import importlib
import json
import sys

import pytest
from fastapi import HTTPException

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_dr = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

_parse_summary_model_config = _dr._parse_summary_model_config

pytestmark = pytest.mark.offline


def test_ai_service_payload_normalizes_base_url_to_host():
    raw = json.dumps(
        {"model": "gpt-summary-x", "base_url": "https://llm.example/v1", "api_key": "sk-1"}
    )
    assert _parse_summary_model_config(raw) == {
        "model": "gpt-summary-x",
        "host": "https://llm.example/v1",
        "api_key": "sk-1",
    }


def test_blank_fields_are_dropped_for_fallback():
    raw = json.dumps({"model": "gpt-summary-x", "base_url": "", "api_key": "  "})
    assert _parse_summary_model_config(raw) == {"model": "gpt-summary-x"}


def test_all_blank_fields_mean_no_override():
    raw = json.dumps({"model": "", "base_url": "", "api_key": ""})
    assert _parse_summary_model_config(raw) is None


def test_explicit_host_wins_over_base_url():
    raw = json.dumps(
        {"model": "m", "base_url": "https://a.example", "host": "https://b.example"}
    )
    assert _parse_summary_model_config(raw) == {
        "model": "m",
        "host": "https://b.example",
    }


def test_omitted_field_returns_none():
    assert _parse_summary_model_config(None) is None
    assert _parse_summary_model_config("") is None
    assert _parse_summary_model_config("   ") is None


def test_invalid_json_is_422():
    with pytest.raises(HTTPException) as exc_info:
        _parse_summary_model_config("{not json")
    assert exc_info.value.status_code == 422


def test_non_object_is_422():
    with pytest.raises(HTTPException) as exc_info:
        _parse_summary_model_config('["model"]')
    assert exc_info.value.status_code == 422


def test_unknown_key_is_422():
    with pytest.raises(HTTPException) as exc_info:
        _parse_summary_model_config('{"model": "m", "secret_field": 1}')
    assert exc_info.value.status_code == 422
    assert "secret_field" in exc_info.value.detail


def test_non_string_scalar_is_422():
    with pytest.raises(HTTPException) as exc_info:
        _parse_summary_model_config('{"model": 123}')
    assert exc_info.value.status_code == 422
