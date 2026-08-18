"""QueryRequest doc_ids / cosine_threshold fields must reach QueryParam unchanged.

The retrieval allow-list is fail-closed: an empty ``doc_ids`` list means "zero
authorized documents" and must survive ``to_query_params()`` as an empty list
(not be normalized to None, which would mean "no filtering").
"""

import importlib
import sys

import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_qr = importlib.import_module("lightrag.api.routers.query_routes")
sys.argv = _original_argv

QueryRequest = _qr.QueryRequest

from lightrag.base import QueryParam

pytestmark = pytest.mark.offline


def test_doc_ids_and_cosine_threshold_pass_through_to_query_params():
    req = QueryRequest(
        query="what is covered",
        doc_ids=["doc-1", "doc-2"],
        cosine_threshold=0.55,
    )
    param = req.to_query_params(is_stream=False)
    assert param.doc_ids == ["doc-1", "doc-2"]
    assert param.cosine_threshold == 0.55


def test_omitted_fields_stay_none():
    req = QueryRequest(query="what is covered")
    param = req.to_query_params(is_stream=False)
    assert param.doc_ids is None
    assert param.cosine_threshold is None


def test_empty_doc_ids_survives_as_empty_list():
    """exclude_none must not swallow []; empty list = zero authorized docs."""
    req = QueryRequest(query="what is covered", doc_ids=[])
    param = req.to_query_params(is_stream=False)
    assert param.doc_ids == []
    assert param.doc_ids is not None


def test_query_param_defaults_are_unfiltered():
    param = QueryParam()
    assert param.doc_ids is None
    assert param.cosine_threshold is None
