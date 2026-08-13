"""Unit tests for ``lightrag.sidecar.artifact_store``.

Covers the env-gated config, the template renderer, the stdlib SigV4 PUT
signing (structural assertions plus an independently recomputed signature),
the path-style request shape via a recording opener, the ``<drawing>`` path
rewrite (string level), and the sidecar asset walk/upload. No live object
store is involved.
"""

import hashlib
import hmac as hmac_mod
import io
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError

import pytest

from lightrag.sidecar.artifact_store import (
    ArtifactStore,
    ArtifactStoreError,
    build_sigv4_put_headers,
    canonical_object_uri,
    collect_asset_files,
    config_from_env,
    render_artifact_template,
    rewrite_drawing_tag_paths,
    upload_document_assets,
)
from lightrag.sidecar.placeholders import render_drawing_tag

pytestmark = pytest.mark.offline

_FULL_ENV = {
    "ARTIFACT_S3_ENDPOINT": "http://minio.test:9000",
    "ARTIFACT_S3_BUCKET": "artifacts",
    "ARTIFACT_S3_ACCESS_KEY": "AKID",
    "ARTIFACT_S3_SECRET_KEY": "SECRET",
    "ARTIFACT_PUBLIC_URL_TEMPLATE": (
        "https://proxy.test/api/artifact?workspace={workspace}"
        "&doc_id={doc_id}&path={relpath}"
    ),
}


def _config(**overrides):
    env = {**_FULL_ENV, **overrides}
    config = config_from_env(env)
    assert config is not None
    return config


# ---------------------------------------------------------------------------
# config_from_env
# ---------------------------------------------------------------------------


def test_config_disabled_without_endpoint():
    assert config_from_env({}) is None


def test_config_disabled_with_incomplete_credentials():
    assert config_from_env({"ARTIFACT_S3_ENDPOINT": "http://minio.test:9000"}) is None
    env = {**_FULL_ENV, "ARTIFACT_S3_SECRET_KEY": ""}
    assert config_from_env(env) is None


def test_config_rejects_endpoint_with_path_or_bad_scheme():
    for bad in ("notaurl", "ftp://minio.test", "http://minio.test:9000/api"):
        env = {**_FULL_ENV, "ARTIFACT_S3_ENDPOINT": bad}
        assert config_from_env(env) is None


def test_config_parses_defaults_and_overrides():
    config = _config()
    assert config.endpoint == "http://minio.test:9000"
    assert config.bucket == "artifacts"
    assert config.region == "us-east-1"
    assert config.prefix_template == "{workspace}/{doc_id}"
    assert config.fail_open is False
    assert config.concurrency == 4

    custom = _config(
        ARTIFACT_S3_REGION="cn-north-1",
        ARTIFACT_S3_PREFIX_TEMPLATE="p/{workspace}",
        ARTIFACT_UPLOAD_FAIL_OPEN="true",
        ARTIFACT_UPLOAD_CONCURRENCY="9",
        ARTIFACT_S3_ENDPOINT="http://minio.test:9000/",  # trailing slash stripped
    )
    assert custom.endpoint == "http://minio.test:9000"
    assert custom.region == "cn-north-1"
    assert custom.prefix_template == "p/{workspace}"
    assert custom.fail_open is True
    assert custom.concurrency == 9

    invalid = _config(ARTIFACT_UPLOAD_CONCURRENCY="zero")
    assert invalid.concurrency == 4


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


def test_render_artifact_template_substitutes_known_tokens_only():
    rendered = render_artifact_template(
        "https://h/x?workspace={workspace}&doc_id={doc_id}&path={relpath}&keep={other}",
        workspace="ws_1",
        doc_id="doc-9",
        relpath="a.blocks.assets/i.png",
    )
    assert rendered == (
        "https://h/x?workspace=ws_1&doc_id=doc-9"
        "&path=a.blocks.assets/i.png&keep={other}"
    )


def test_canonical_object_uri_encodes_each_segment():
    assert canonical_object_uri("b", "ws/doc/a b.png") == "/b/ws/doc/a%20b.png"
    assert canonical_object_uri("b", "ws/doc/x#1.png") == "/b/ws/doc/x%231.png"
    assert canonical_object_uri("b", "k") == "/b/k"


# ---------------------------------------------------------------------------
# SigV4 signing
# ---------------------------------------------------------------------------


def test_sigv4_headers_match_independent_recomputation():
    now = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    payload_hash = hashlib.sha256(b"payload").hexdigest()
    headers = build_sigv4_put_headers(
        host="minio.test:9000",
        canonical_uri="/artifacts/ws/doc/x.png",
        payload_hash=payload_hash,
        access_key="AKID",
        secret_key="SECRET",
        region="us-east-1",
        now=now,
    )
    assert headers["x-amz-date"] == "20240102T030405Z"
    assert headers["x-amz-content-sha256"] == payload_hash

    authorization = headers["Authorization"]
    scope = "20240102/us-east-1/s3/aws4_request"
    assert authorization.startswith(
        f"AWS4-HMAC-SHA256 Credential=AKID/{scope}, "
        "SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature="
    )

    # Independent recomputation of the expected signature.
    signed_headers = "host;x-amz-content-sha256;x-amz-date"
    canonical_headers = (
        "host:minio.test:9000\n"
        f"x-amz-content-sha256:{payload_hash}\n"
        "x-amz-date:20240102T030405Z\n"
    )
    canonical_request = "\n".join(
        [
            "PUT",
            "/artifacts/ws/doc/x.png",
            "",
            canonical_headers,
            signed_headers,
            payload_hash,
        ]
    )
    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            "20240102T030405Z",
            scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        ]
    )

    def _sign(key, msg):
        return hmac_mod.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

    k_date = _sign(b"AWS4SECRET", "20240102")
    k_region = _sign(k_date, "us-east-1")
    k_service = _sign(k_region, "s3")
    k_signing = _sign(k_service, "aws4_request")
    expected = hmac_mod.new(
        k_signing, string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    assert authorization.endswith(f"Signature={expected}")


# ---------------------------------------------------------------------------
# ArtifactStore PUT (recording opener)
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status=200):
        self._status = status
        self.closed = False

    def getcode(self):
        return self._status

    def close(self):
        self.closed = True


class _RecordingOpener:
    def __init__(self, status=200, error=None):
        self.status = status
        self.error = error
        self.requests = []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return _FakeResponse(self.status)


def _store(opener, **env_overrides):
    return ArtifactStore(_config(**env_overrides), opener=opener)


async def test_put_builds_signed_path_style_request():
    opener = _RecordingOpener()
    store = _store(opener)
    body = b"png-bytes"
    await store.put_bytes("ws_a/doc-1/abc.blocks.assets/i.png", body, "image/png")

    assert len(opener.requests) == 1
    request = opener.requests[0]
    assert request.get_method() == "PUT"
    assert request.full_url == (
        "http://minio.test:9000/artifacts/ws_a/doc-1/abc.blocks.assets/i.png"
    )
    headers = {k.lower(): v for k, v in request.headers.items()}
    assert headers["authorization"].startswith("AWS4-HMAC-SHA256 Credential=AKID/")
    assert headers["x-amz-content-sha256"] == hashlib.sha256(body).hexdigest()
    assert headers["content-type"] == "image/png"


async def test_put_http_error_raises_store_error():
    error = HTTPError(
        "http://minio.test:9000/x", 403, "Forbidden", {}, io.BytesIO(b"denied")
    )
    opener = _RecordingOpener(error=error)
    with pytest.raises(ArtifactStoreError, match="HTTP 403"):
        await _store(opener).put_bytes("k", b"data")


async def test_put_url_error_raises_store_error():
    opener = _RecordingOpener(error=URLError("connection refused"))
    with pytest.raises(ArtifactStoreError, match="connection refused"):
        await _store(opener).put_bytes("k", b"data")


def test_object_key_and_public_url():
    store = _store(_RecordingOpener())
    assert (
        store.object_key("ws_a", "doc-1", "abc.blocks.assets/i.png")
        == "ws_a/doc-1/abc.blocks.assets/i.png"
    )
    assert store.public_url("ws_a", "doc-1", "abc.blocks.assets/i.png") == (
        "https://proxy.test/api/artifact?workspace=ws_a&doc_id=doc-1"
        "&path=abc.blocks.assets/i.png"
    )
    no_template = _store(_RecordingOpener(), ARTIFACT_PUBLIC_URL_TEMPLATE="")
    assert no_template.public_url("ws_a", "doc-1", "x") == ""
    custom_prefix = _store(
        _RecordingOpener(), ARTIFACT_S3_PREFIX_TEMPLATE="root/{workspace}"
    )
    assert custom_prefix.object_key("ws_a", "doc-1", "x") == "root/ws_a/x"


# ---------------------------------------------------------------------------
# <drawing> path rewrite
# ---------------------------------------------------------------------------


def _drawing_content(path):
    tag = render_drawing_tag("im-abc-0001", "png", "a caption", path, "orig src")
    return f"intro {tag} outro"


def test_rewrite_replaces_path_only_and_escapes_url():
    content = _drawing_content("abc.blocks.assets/i.png")
    url = (
        "https://proxy.test/api?workspace=ws&doc_id=doc-1&path=abc.blocks.assets/i.png"
    )
    new_content, count = rewrite_drawing_tag_paths(content, lambda raw: url)

    assert count == 1
    assert 'src="orig src"' in new_content
    assert 'caption="a caption"' in new_content
    assert 'id="im-abc-0001"' in new_content
    # The & separators of the query string are XML-escaped in the attribute.
    assert f'path="{url.replace("&", "&amp;")}"' in new_content
    assert 'path="abc.blocks.assets/i.png"' not in new_content


def test_rewrite_is_idempotent_for_urls():
    url = "https://proxy.test/api?path=abc.blocks.assets/i.png"
    content = _drawing_content(url)
    new_content, count = rewrite_drawing_tag_paths(
        content, lambda raw: "https://other.test/x"
    )
    assert count == 0
    assert new_content == content


def test_rewrite_skips_unmapped_and_empty_paths():
    content = _drawing_content("abc.blocks.assets/missing.png")
    new_content, count = rewrite_drawing_tag_paths(content, lambda raw: None)
    assert count == 0
    assert new_content == content

    empty = _drawing_content("")
    assert rewrite_drawing_tag_paths(empty, lambda raw: "https://h/x") == (empty, 0)


def test_rewrite_leaves_other_tags_untouched():
    content = '<table id="tb-1" format="json">[["a"]]</table> ' + _drawing_content(
        "abc.blocks.assets/i.png"
    )
    new_content, count = rewrite_drawing_tag_paths(content, lambda raw: "https://h/x")
    assert count == 1
    assert '<table id="tb-1" format="json">[["a"]]</table>' in new_content


# ---------------------------------------------------------------------------
# Asset walk + upload
# ---------------------------------------------------------------------------


def _make_parsed_dir(tmp_path):
    parsed = tmp_path / "__parsed__" / "abc.parsed"
    assets = parsed / "abc.blocks.assets"
    assets.mkdir(parents=True)
    (parsed / "abc.blocks.jsonl").write_text("{}\n", encoding="utf-8")
    (assets / "img one.png").write_bytes(b"one")
    (assets / "img2.jpg").write_bytes(b"two")
    return parsed, assets


async def test_upload_document_assets_preserves_relative_layout(tmp_path):
    parsed, assets = _make_parsed_dir(tmp_path)
    opener = _RecordingOpener()
    store = _store(opener)

    uploaded = await upload_document_assets(
        store, assets_dir=assets, workspace="ws_a", doc_id="doc-1"
    )

    assert set(uploaded) == {
        "abc.blocks.assets/img one.png",
        "abc.blocks.assets/img2.jpg",
    }
    urls = {request.full_url for request in opener.requests}
    assert urls == {
        "http://minio.test:9000/artifacts/ws_a/doc-1/abc.blocks.assets/img%20one.png",
        "http://minio.test:9000/artifacts/ws_a/doc-1/abc.blocks.assets/img2.jpg",
    }
    bodies = sorted(request.data for request in opener.requests)
    assert bodies == [b"one", b"two"]


def test_collect_asset_files_relpaths_are_parsed_dir_relative(tmp_path):
    parsed, assets = _make_parsed_dir(tmp_path)
    relpaths = [relpath for _, relpath in collect_asset_files(assets)]
    assert relpaths == [
        "abc.blocks.assets/img one.png",
        "abc.blocks.assets/img2.jpg",
    ]
    assert parsed == assets.parent
