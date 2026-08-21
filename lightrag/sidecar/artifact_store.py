"""Parse-artifact object-store staging (env-gated, stdlib only).

The ingestion pipeline can stage a document's sidecar assets (images and
other files under ``<base>.blocks.assets/``) to an S3-compatible object
store (e.g. MinIO) AFTER the analyze stage (which reads the local files)
and BEFORE chunking, then rewrite the ``path`` attribute of every
``<drawing ... />`` tag in the persisted ``full_docs`` body to a permanent,
caller-defined URL — so the chunk text itself carries the final, resolvable
address. After a document reaches PROCESSED, the final body and its chunk
list are exported next to the assets as ``document.md`` / ``chunks.json``.

Everything is driven by environment variables; with ``ARTIFACT_S3_ENDPOINT``
unset the whole feature is inert (historical behavior, zero overhead):

- ``ARTIFACT_S3_ENDPOINT``       scheme://host[:port] of the S3 API.
- ``ARTIFACT_S3_BUCKET``         target bucket (required when endpoint set).
- ``ARTIFACT_S3_ACCESS_KEY`` / ``ARTIFACT_S3_SECRET_KEY``  credentials.
- ``ARTIFACT_S3_REGION``         signing region (default ``us-east-1``).
- ``ARTIFACT_S3_PREFIX_TEMPLATE`` object-key prefix template; the tokens
  ``{workspace}``, ``{doc_id}`` and ``{relpath}`` are substituted
  (default ``{workspace}/{doc_id}``).
- ``ARTIFACT_PUBLIC_URL_TEMPLATE`` permanent-URL template with the same
  tokens, e.g.
  ``https://host/api/artifact?workspace={workspace}&doc_id={doc_id}&path={relpath}``.
  When unset, assets are still uploaded but ``<drawing>`` paths are left
  untouched (chunking then sees the historical local paths).
- ``ARTIFACT_UPLOAD_FAIL_OPEN``  ``true`` downgrades an upload failure from
  "fail the document" (default; a rerun is preferred over persisting dead
  links) to "log a warning and continue with local paths".
- ``ARTIFACT_UPLOAD_CONCURRENCY`` process-wide cap on concurrent object
  PUTs (default 4).
- ``ARTIFACT_S3_TIMEOUT_SECONDS`` per-request timeout (default 60).

The S3 client is a minimal SigV4 path-style PUT implemented on the standard
library (``urllib`` + ``hmac``/``hashlib``) so the fork gains no third-party
dependency; only the subset of SigV4 needed for ``PUT /<bucket>/<key>`` is
implemented.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import mimetypes
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

from lightrag.sidecar.placeholders import xml_attr_escape
from lightrag.utils import logger

ENV_S3_ENDPOINT = "ARTIFACT_S3_ENDPOINT"
ENV_S3_BUCKET = "ARTIFACT_S3_BUCKET"
ENV_S3_ACCESS_KEY = "ARTIFACT_S3_ACCESS_KEY"
ENV_S3_SECRET_KEY = "ARTIFACT_S3_SECRET_KEY"
ENV_S3_REGION = "ARTIFACT_S3_REGION"
ENV_S3_PREFIX_TEMPLATE = "ARTIFACT_S3_PREFIX_TEMPLATE"
ENV_PUBLIC_URL_TEMPLATE = "ARTIFACT_PUBLIC_URL_TEMPLATE"
ENV_UPLOAD_FAIL_OPEN = "ARTIFACT_UPLOAD_FAIL_OPEN"
ENV_UPLOAD_CONCURRENCY = "ARTIFACT_UPLOAD_CONCURRENCY"
ENV_S3_TIMEOUT_SECONDS = "ARTIFACT_S3_TIMEOUT_SECONDS"

DEFAULT_REGION = "us-east-1"
DEFAULT_PREFIX_TEMPLATE = "{workspace}/{doc_id}"
DEFAULT_UPLOAD_CONCURRENCY = 4
DEFAULT_TIMEOUT_SECONDS = 60


class ArtifactStoreError(Exception):
    """An artifact upload/export failed (network, auth or remote error)."""


@dataclass(frozen=True)
class ArtifactStoreConfig:
    """Resolved artifact-store configuration (see module docstring)."""

    endpoint: str  # scheme://host[:port], no trailing slash, no path
    bucket: str
    access_key: str
    secret_key: str
    region: str
    prefix_template: str
    public_url_template: str  # empty string disables the path rewrite
    fail_open: bool
    concurrency: int
    timeout_seconds: float


def _env_int(environ: Mapping[str, str], name: str, default: int, minimum: int) -> int:
    raw = (environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning(f"[artifacts] invalid {name}={raw!r}; using {default}")
        return default
    if value < minimum:
        logger.warning(f"[artifacts] {name}={value} below {minimum}; using {default}")
        return default
    return value


def config_from_env(
    environ: Mapping[str, str] | None = None,
) -> ArtifactStoreConfig | None:
    """Build the config from the environment; None when the feature is off.

    Disabled unless ``ARTIFACT_S3_ENDPOINT`` is set; a missing bucket or
    credentials with an endpoint set is a misconfiguration and also disables
    the feature (with a warning) rather than failing every document.
    """
    env = os.environ if environ is None else environ
    endpoint = (env.get(ENV_S3_ENDPOINT) or "").strip().rstrip("/")
    if not endpoint:
        return None
    parts = urlsplit(endpoint)
    if (
        parts.scheme not in ("http", "https")
        or not parts.netloc
        or parts.path not in ("", "/")
    ):
        logger.warning(
            f"[artifacts] {ENV_S3_ENDPOINT}={endpoint!r} must be a bare "
            "scheme://host[:port]; artifact staging disabled"
        )
        return None
    bucket = (env.get(ENV_S3_BUCKET) or "").strip().strip("/")
    access_key = (env.get(ENV_S3_ACCESS_KEY) or "").strip()
    secret_key = (env.get(ENV_S3_SECRET_KEY) or "").strip()
    if not bucket or not access_key or not secret_key:
        logger.warning(
            f"[artifacts] {ENV_S3_ENDPOINT} is set but {ENV_S3_BUCKET} / "
            f"{ENV_S3_ACCESS_KEY} / {ENV_S3_SECRET_KEY} are incomplete; "
            "artifact staging disabled"
        )
        return None
    fail_open = (env.get(ENV_UPLOAD_FAIL_OPEN) or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    return ArtifactStoreConfig(
        endpoint=endpoint,
        bucket=bucket,
        access_key=access_key,
        secret_key=secret_key,
        region=(env.get(ENV_S3_REGION) or "").strip() or DEFAULT_REGION,
        prefix_template=(env.get(ENV_S3_PREFIX_TEMPLATE) or "").strip()
        or DEFAULT_PREFIX_TEMPLATE,
        public_url_template=(env.get(ENV_PUBLIC_URL_TEMPLATE) or "").strip(),
        fail_open=fail_open,
        concurrency=_env_int(
            env, ENV_UPLOAD_CONCURRENCY, DEFAULT_UPLOAD_CONCURRENCY, 1
        ),
        timeout_seconds=float(
            _env_int(env, ENV_S3_TIMEOUT_SECONDS, DEFAULT_TIMEOUT_SECONDS, 1)
        ),
    )


def render_artifact_template(
    template: str, *, workspace: str, doc_id: str, relpath: str = ""
) -> str:
    """Substitute workspace, KB, document and relative-path tokens.

    Plain ``str.replace`` (not ``str.format``) so templates containing other
    braces — e.g. a URL query string — never raise; unknown tokens pass
    through unchanged.
    """
    kb_id = workspace[3:] if workspace.startswith("kb_") else workspace
    return (
        template.replace("{workspace}", workspace)
        .replace("{kb_id}", kb_id)
        .replace("{doc_id}", doc_id)
        .replace("{relpath}", relpath)
        # ``artifact_path`` is the DataHub browser-proxy spelling. Keep
        # ``relpath`` as the native fork spelling for existing deployments.
        .replace("{artifact_path}", relpath)
    )


# ---------------------------------------------------------------------------
# SigV4 signing (stdlib)
# ---------------------------------------------------------------------------


def _hmac_sha256(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()


def _signing_key(secret_key: str, date_stamp: str, region: str) -> bytes:
    k_date = _hmac_sha256(("AWS4" + secret_key).encode("utf-8"), date_stamp)
    k_region = _hmac_sha256(k_date, region)
    k_service = _hmac_sha256(k_region, "s3")
    return _hmac_sha256(k_service, "aws4_request")


def canonical_object_uri(bucket: str, key: str) -> str:
    """Path-style ``/<bucket>/<key>`` with per-segment SigV4 encoding."""
    segments = [bucket] + [seg for seg in key.split("/") if seg]
    return "/" + "/".join(quote(seg, safe="-_.~") for seg in segments)


def build_sigv4_put_headers(
    *,
    host: str,
    canonical_uri: str,
    payload_hash: str,
    access_key: str,
    secret_key: str,
    region: str,
    now: datetime | None = None,
) -> dict[str, str]:
    """Sign a ``PUT <canonical_uri>`` request (SigV4, service ``s3``).

    Signs exactly ``host;x-amz-content-sha256;x-amz-date`` — the payload is
    hashed (never UNSIGNED-PAYLOAD) so a corrupted in-flight body is
    rejected by the store. ``host`` is NOT returned in the dict (urllib
    derives an identical ``Host`` header from the request URL); it must be
    the authority component of that URL, port included when non-default.
    """
    moment = now or datetime.now(timezone.utc)
    amz_date = moment.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = moment.strftime("%Y%m%d")
    signed_headers = "host;x-amz-content-sha256;x-amz-date"
    canonical_headers = (
        f"host:{host}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{amz_date}\n"
    )
    canonical_request = "\n".join(
        ["PUT", canonical_uri, "", canonical_headers, signed_headers, payload_hash]
    )
    credential_scope = f"{date_stamp}/{region}/s3/aws4_request"
    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            amz_date,
            credential_scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        ]
    )
    signature = hmac.new(
        _signing_key(secret_key, date_stamp, region),
        string_to_sign.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    authorization = (
        "AWS4-HMAC-SHA256 "
        f"Credential={access_key}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, "
        f"Signature={signature}"
    )
    return {
        "x-amz-date": amz_date,
        "x-amz-content-sha256": payload_hash,
        "Authorization": authorization,
    }


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

# Process-wide upload semaphore: LightRAG instances are per-workspace, so a
# per-instance limiter would not bound total PUT pressure on the store.
_upload_semaphore: asyncio.Semaphore | None = None


def _get_upload_semaphore(limit: int) -> asyncio.Semaphore:
    global _upload_semaphore
    if _upload_semaphore is None:
        _upload_semaphore = asyncio.Semaphore(limit)
    return _upload_semaphore


class ArtifactStore:
    """Minimal S3-compatible object store client (path-style PUT only).

    ``opener`` is injectable for tests (same call signature as
    :func:`urllib.request.urlopen`).
    """

    def __init__(self, config: ArtifactStoreConfig, *, opener=None):
        self._config = config
        self._opener = opener if opener is not None else urlopen

    @property
    def config(self) -> ArtifactStoreConfig:
        return self._config

    def object_key(self, workspace: str, doc_id: str, relpath: str) -> str:
        """Object key for one artifact: rendered prefix + relative path."""
        prefix = render_artifact_template(
            self._config.prefix_template,
            workspace=workspace,
            doc_id=doc_id,
            relpath=relpath,
        ).strip("/")
        rel = relpath.lstrip("/")
        return f"{prefix}/{rel}" if prefix else rel

    def public_url(self, workspace: str, doc_id: str, relpath: str) -> str:
        """Permanent URL for one artifact; empty when no template is set."""
        if not self._config.public_url_template:
            return ""
        return render_artifact_template(
            self._config.public_url_template,
            workspace=workspace,
            doc_id=doc_id,
            relpath=relpath,
        )

    async def put_bytes(
        self, key: str, data: bytes, content_type: str | None = None
    ) -> None:
        """PUT one object, bounded by the process-wide upload semaphore."""
        async with _get_upload_semaphore(self._config.concurrency):
            await asyncio.to_thread(self._put_bytes_sync, key, data, content_type)

    def _put_bytes_sync(self, key: str, data: bytes, content_type: str | None) -> None:
        cfg = self._config
        payload_hash = hashlib.sha256(data).hexdigest()
        canonical_uri = canonical_object_uri(cfg.bucket, key)
        headers = build_sigv4_put_headers(
            host=urlsplit(cfg.endpoint).netloc,
            canonical_uri=canonical_uri,
            payload_hash=payload_hash,
            access_key=cfg.access_key,
            secret_key=cfg.secret_key,
            region=cfg.region,
        )
        headers["Content-Type"] = content_type or "application/octet-stream"
        headers["Content-Length"] = str(len(data))
        request = Request(
            f"{cfg.endpoint}{canonical_uri}", data=data, headers=headers, method="PUT"
        )
        try:
            response = self._opener(request, timeout=cfg.timeout_seconds)
        except HTTPError as exc:
            body = ""
            try:
                body = exc.read(512).decode("utf-8", "replace")
            except Exception:
                pass
            raise ArtifactStoreError(
                f"PUT {key} failed with HTTP {exc.code}: {body or exc.reason}"
            ) from exc
        except URLError as exc:
            raise ArtifactStoreError(f"PUT {key} failed: {exc.reason}") from exc
        except OSError as exc:
            raise ArtifactStoreError(f"PUT {key} failed: {exc}") from exc
        try:
            status = response.getcode()
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()
        if status is None or not (200 <= status < 300):
            raise ArtifactStoreError(f"PUT {key} returned HTTP {status}")


# Process-wide lazy singleton: env is process-wide, and sharing one instance
# keeps the configuration resolution (and its warnings) once-per-process.
_store_checked = False
_store_instance: ArtifactStore | None = None


def get_artifact_store() -> ArtifactStore | None:
    """Return the env-configured singleton store, or None when disabled."""
    global _store_checked, _store_instance
    if not _store_checked:
        config = config_from_env()
        _store_instance = ArtifactStore(config) if config is not None else None
        _store_checked = True
    return _store_instance


def _reset_artifact_store_cache() -> None:
    """Drop the cached singleton and semaphore (test isolation helper)."""
    global _store_checked, _store_instance, _upload_semaphore
    _store_checked = False
    _store_instance = None
    _upload_semaphore = None


# ---------------------------------------------------------------------------
# <drawing> path rewrite
# ---------------------------------------------------------------------------

_DRAWING_TAG_RE = re.compile(r"<drawing\b[^>]*?/>", re.DOTALL)
_PATH_ATTR_RE = re.compile(r'(\bpath=")([^"]*)(")')
_ALREADY_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


def _xml_attr_unescape(value: str) -> str:
    """Inverse of :func:`xml_attr_escape` (``&amp;`` must be last)."""
    return (
        value.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&amp;", "&")
    )


def rewrite_drawing_tag_paths(content: str, url_for) -> tuple[str, int]:
    """Rewrite the ``path`` attribute of every ``<drawing ... />`` tag.

    ``url_for(raw_path)`` receives the XML-unescaped attribute value and
    returns the replacement URL, or None to leave the tag untouched. A path
    that is already an ``http(s)://`` URL is skipped (idempotent re-runs).
    Every other attribute (``src`` / ``caption`` / ...) is preserved
    byte-for-byte. Returns ``(new_content, rewrite_count)``.
    """
    rewrites = 0

    def _fix_tag(tag_match: "re.Match[str]") -> str:
        def _fix_attr(attr_match: "re.Match[str]") -> str:
            nonlocal rewrites
            raw = _xml_attr_unescape(attr_match.group(2))
            if not raw or _ALREADY_URL_RE.match(raw):
                return attr_match.group(0)
            new_url = url_for(raw)
            if not new_url:
                return attr_match.group(0)
            rewrites += 1
            return attr_match.group(1) + xml_attr_escape(new_url) + attr_match.group(3)

        return _PATH_ATTR_RE.sub(_fix_attr, tag_match.group(0), count=1)

    return _DRAWING_TAG_RE.sub(_fix_tag, content), rewrites


# ---------------------------------------------------------------------------
# Sidecar asset upload
# ---------------------------------------------------------------------------


def collect_asset_files(assets_dir: Path) -> list[tuple[Path, str]]:
    """List ``(file, relpath)`` under ``assets_dir``, sorted for determinism.

    ``relpath`` is the POSIX path relative to the parsed directory (the
    assets dir's parent) — the same shape the sidecar writer stores in
    ``<drawing path="...">`` (``<base>.blocks.assets/<file>``), so the
    attribute value can be matched against upload results 1:1.
    """
    parsed_dir = assets_dir.parent
    out: list[tuple[Path, str]] = []
    for path in sorted(assets_dir.rglob("*")):
        if path.is_file():
            out.append((path, path.relative_to(parsed_dir).as_posix()))
    return out


async def upload_document_assets(
    store: ArtifactStore,
    *,
    assets_dir: Path,
    workspace: str,
    doc_id: str,
) -> dict[str, str]:
    """Upload every file under ``assets_dir``; return ``{relpath: key}``.

    Re-uploading the same document overwrites the same keys (the key layout
    is deterministic), which makes reruns idempotent.
    """
    uploaded: dict[str, str] = {}
    for path, relpath in collect_asset_files(assets_dir):
        key = store.object_key(workspace, doc_id, relpath)
        data = await asyncio.to_thread(path.read_bytes)
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        await store.put_bytes(key, data, content_type)
        uploaded[relpath] = key
    return uploaded
