"""MinerU raw bundle downloader.

Supports MinerU's official cloud and self-hosted API protocols and lands the
final parser bundle on disk under ``raw_dir/``:

- ``official`` — MinerU precision API v4: apply for signed upload URL, PUT the
  local file, poll batch results, download ``full_zip_url``.
- ``local`` — self-hosted ``mineru-api`` / ``mineru-router``: submit
  ``POST /tasks``, poll ``GET /tasks/{task_id}``, download
  ``GET /tasks/{task_id}/result``.
- ``wrapper`` — custom wrapper service: single synchronous
  ``POST {endpoint}/predict`` with a base64 JSON body. The response delivers
  the parse result as (in priority order) a ``package_url`` zip bundle, an
  inline ``content_list`` array, or markdown (inline ``md`` / ``markdown`` /
  ``md_content`` or a downloadable ``md_url``).

The official and local protocols request a zip result bundle. Archives are
extracted under ``raw_dir/`` and normalized so the adapter can read a
root-level ``content_list.json``; the wrapper mode normalizes every delivery
shape onto the same on-disk contract.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import math
import os
import re
import shutil
from collections.abc import AsyncIterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urljoin, urlparse

from lightrag.parser.external._common import raise_for_status_with_detail
from lightrag.parser.external._zip import result_bundle_limits, safe_extract_zip
from lightrag.parser.external.mineru.cache import (
    MinerUParserOptions,
    compute_size_and_hash,
)
from lightrag.parser.external.mineru.manifest import (
    Manifest,
    ManifestFile,
    write_manifest,
)
from lightrag.utils import logger

if TYPE_CHECKING:
    import httpx
else:
    try:
        import httpx
    except ImportError:  # pragma: no cover
        httpx = None

CONTENT_LIST_FILENAME = "content_list.json"
DEFAULT_MINERU_API_MODE = "local"
DEFAULT_MINERU_OFFICIAL_ENDPOINT = "https://mineru.net"
VALID_MINERU_API_MODES = {"official", "local", "wrapper"}
OFFICIAL_DONE_STATES = {"done"}
OFFICIAL_FAILED_STATES = {"failed"}
LOCAL_DONE_STATES = {"completed"}
LOCAL_FAILED_STATES = {"failed"}
UPLOAD_CHUNK_SIZE = 1024 * 1024
DEFAULT_MINERU_HTTP_TIMEOUT_SECONDS = 600.0
DEFAULT_MINERU_CONNECT_TIMEOUT_SECONDS = 30.0

# Markdown image references: ``![alt](path)`` and ``<img src="path">``.
_MD_IMAGE_REF_RES = (
    re.compile(r"!\[[^\]]*\]\(([^)]+)\)"),
    re.compile(r"<img\b[^>]*\bsrc=[\"']([^\"']+)[\"']", re.IGNORECASE),
)
_ABSOLUTE_REF_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def _get_by_path(payload: Any, path: str) -> Any:
    """Walk a dotted path through a nested dict; returns None if any segment
    is missing or non-dict."""
    if not path:
        return None
    cur = payload
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _strip_trailing_slash(url: str) -> str:
    return url.rstrip("/")


def _positive_float_env(names: tuple[str, ...], default: float) -> float:
    """Read a positive timeout, accepting the shared parser timeout alias."""
    for name in names:
        raw = os.getenv(name, "").strip()
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            logger.warning("%s must be a positive number; using the next fallback", name)
            continue
        if math.isfinite(value) and value > 0:
            return value
        logger.warning("%s must be a positive number; using the next fallback", name)
    return default


def _positive_float_option(
    configured: Any,
    env_names: tuple[str, ...],
    default: float,
) -> float:
    """Prefer a validated dynamic runtime value, then preserve env fallback."""
    if configured is not None:
        try:
            value = float(configured)
        except (TypeError, ValueError):
            value = 0.0
        if math.isfinite(value) and value > 0:
            return value
        logger.warning("MinerU dynamic timeout must be positive; using env fallback")
    return _positive_float_env(env_names, default)


def _positive_int_option(
    configured: Any,
    env_names: tuple[str, ...],
    default: int,
) -> int:
    """Prefer a validated dynamic poll count, then preserve env fallback."""
    if configured is not None and not isinstance(configured, bool):
        try:
            value = int(configured)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
        logger.warning("MinerU dynamic poll count must be positive; using env fallback")
    for name in env_names:
        raw = os.getenv(name, "").strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError:
            logger.warning("%s must be a positive integer; using the next fallback", name)
            continue
        if value > 0:
            return value
        logger.warning("%s must be a positive integer; using the next fallback", name)
    return default


def _timeout_exception_types() -> tuple[type[BaseException], ...]:
    """Return the timeout classes available in real or test httpx modules."""
    timeout_exception = getattr(httpx, "TimeoutException", None)
    if isinstance(timeout_exception, type) and issubclass(timeout_exception, BaseException):
        return (timeout_exception, TimeoutError)
    return (TimeoutError,)


def _resolve_upload_name(upload_name: str | None, source_file_path: Path) -> str:
    candidate = Path(str(upload_name or "")).name
    return candidate or source_file_path.name


async def _iter_file_bytes(path: Path) -> AsyncIterator[bytes]:
    with path.open("rb") as fh:
        while True:
            chunk = await asyncio.to_thread(fh.read, UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            yield chunk


def _validate_base_url(
    name: str, endpoint: str, forbidden_segments: tuple[str, ...]
) -> None:
    parsed = urlparse(endpoint)
    path = (parsed.path or "").rstrip("/")
    for segment in forbidden_segments:
        if path.endswith(segment) or f"{segment}/" in path:
            raise ValueError(
                f"{name} must be a base URL, not an API path: {endpoint!r}"
            )


class MinerURawClient:
    """Downloads MinerU bundles into ``raw_dir``.

    Construct once per call (cheap). Reads ``MINERU_*`` env vars at
    construction time. Methods are async and use a single shared httpx
    client across all calls in :meth:`download_into`.

    ``MINERU_HTTP_TIMEOUT_SECONDS`` controls the complete request/read window.
    ``REMOTE_PARSER_TIMEOUT`` is accepted as a fallback so the runtime can
    share the parser timeout configured by ai-service.

    Implements the MinerU-specific upload + poll + zip download flow
    inline; bundle handling needs the ``result_url`` *and* the
    ``Content-Type`` of the response, which a generic protocol helper
    cannot expose without leaking abstractions.
    """

    def __init__(
        self,
        *,
        overrides: "Mapping[str, Any] | None" = None,
        runtime_options: "Mapping[str, Any] | None" = None,
    ) -> None:
        self._overrides = overrides or {}
        self._runtime_options = runtime_options or {}
        self._remote_task_id = ""
        self._on_remote_task = self._runtime_options.get("on_remote_task")
        self.api_mode = (
            os.getenv("MINERU_API_MODE", DEFAULT_MINERU_API_MODE).strip().lower()
        )
        if self.api_mode not in VALID_MINERU_API_MODES:
            allowed = ", ".join(sorted(VALID_MINERU_API_MODES))
            raise ValueError(
                f"MINERU_API_MODE must be one of {allowed}, got {self.api_mode!r}"
            )

        self.official_endpoint = _strip_trailing_slash(
            os.getenv(
                "MINERU_OFFICIAL_ENDPOINT", DEFAULT_MINERU_OFFICIAL_ENDPOINT
            ).strip()
            or DEFAULT_MINERU_OFFICIAL_ENDPOINT
        )
        self.local_endpoint = _strip_trailing_slash(
            os.getenv("MINERU_LOCAL_ENDPOINT", "").strip()
        )
        self.wrapper_endpoint = _strip_trailing_slash(
            os.getenv("MINERU_WRAPPER_ENDPOINT", "").strip()
        )
        configured_base_url = str(
            self._runtime_options.get("mineru_base_url") or ""
        ).strip()
        if configured_base_url:
            if self.api_mode == "local":
                self.local_endpoint = _strip_trailing_slash(configured_base_url)
            elif self.api_mode == "wrapper":
                self.wrapper_endpoint = _strip_trailing_slash(configured_base_url)
        self.api_token = os.getenv("MINERU_API_TOKEN", "").strip()
        if self.api_mode == "official":
            if not self.api_token:
                raise ValueError(
                    "MINERU_API_TOKEN is required when MINERU_API_MODE=official"
                )
            _validate_base_url(
                "MINERU_OFFICIAL_ENDPOINT",
                self.official_endpoint,
                ("/api/v4", "/api/v4/file-urls/batch", "/api/v4/extract/task"),
            )
            self.endpoint = self.official_endpoint
        elif self.api_mode == "local":
            if not self.local_endpoint:
                raise ValueError(
                    "MINERU_LOCAL_ENDPOINT is required when MINERU_API_MODE=local"
                )
            _validate_base_url(
                "MINERU_LOCAL_ENDPOINT",
                self.local_endpoint,
                ("/tasks", "/file_parse", "/health"),
            )
            self.endpoint = self.local_endpoint
        elif self.api_mode == "wrapper":
            if not self.wrapper_endpoint:
                raise ValueError(
                    "MINERU_WRAPPER_ENDPOINT is required when "
                    "MINERU_API_MODE=wrapper"
                )
            _validate_base_url(
                "MINERU_WRAPPER_ENDPOINT",
                self.wrapper_endpoint,
                ("/predict", "/health"),
            )
            self.endpoint = self.wrapper_endpoint
        self.poll_interval = _positive_float_option(
            self._runtime_options.get("poll_interval_seconds"),
            ("MINERU_POLL_INTERVAL_SECONDS",),
            2.0,
        )
        # 600 * 2s client-side sleep ≈ 20 min worst case; raise for very large PDFs.
        self.max_polls = _positive_int_option(
            self._runtime_options.get("poll_max_attempts"),
            ("MINERU_MAX_POLLS",),
            600,
        )
        self.http_timeout_seconds = _positive_float_option(
            self._runtime_options.get("read_timeout_seconds"),
            ("MINERU_HTTP_TIMEOUT_SECONDS", "REMOTE_PARSER_TIMEOUT"),
            DEFAULT_MINERU_HTTP_TIMEOUT_SECONDS,
        )
        self.connect_timeout_seconds = _positive_float_option(
            self._runtime_options.get("connect_timeout_seconds"),
            ("MINERU_CONNECT_TIMEOUT_SECONDS",),
            DEFAULT_MINERU_CONNECT_TIMEOUT_SECONDS,
        )
        self.engine_version = os.getenv("MINERU_ENGINE_VERSION", "").strip()

        options = MinerUParserOptions.from_env(
            api_mode=self.api_mode, overrides=self._overrides
        )
        self._parser_options = options
        self.model_version = options.model_version
        self.language = options.language
        self.enable_table = options.enable_table
        self.enable_formula = options.enable_formula
        self.is_ocr = options.is_ocr
        self.page_ranges = options.page_ranges
        self.local_backend = options.local_backend
        self.local_parse_method = options.local_parse_method
        self.local_image_analysis = options.local_image_analysis
        self.local_start_page_id = options.local_start_page_id
        self.local_end_page_id = options.local_end_page_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def download_into(
        self,
        raw_dir: Path,
        source_file_path: Path,
        *,
        upload_name: str | None = None,
    ) -> Manifest:
        """Download a fresh bundle and write the manifest.

        Pre-condition: caller cleared ``raw_dir`` contents (recommended via
        :func:`clear_dir_contents`). This method does NOT clean the
        directory itself — leaving that to the caller keeps cache miss
        semantics explicit at the parse_mineru entry point.

        Returns the :class:`Manifest` describing the bundle.
        """
        if httpx is None:
            raise RuntimeError("httpx is required for MinerU parsing but not installed")
        raw_dir.mkdir(parents=True, exist_ok=True)
        resolved_upload_name = _resolve_upload_name(upload_name, source_file_path)

        timeout = httpx.Timeout(
            self.http_timeout_seconds,
            connect=self.connect_timeout_seconds,
        )
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                if self.api_mode == "official":
                    task_id = await self._download_official(
                        client, source_file_path, raw_dir, resolved_upload_name
                    )
                elif self.api_mode == "wrapper":
                    task_id = await self._download_wrapper(
                        client, source_file_path, raw_dir, resolved_upload_name
                    )
                else:
                    task_id = await self._download_local(
                        client, source_file_path, raw_dir, resolved_upload_name
                    )
        except _timeout_exception_types() as exc:
            from lightrag.parser.external.mineru.scheduling import MinerURequestTimeout

            timeout_kind = "poll" if self._remote_task_id else "read"
            remote_task_terminal, remote_task_state = (
                await self._confirm_remote_task_terminal(resolved_upload_name)
            )
            raise MinerURequestTimeout(
                f"MinerU {self.api_mode} backend {timeout_kind} timeout "
                f"(endpoint={self.endpoint}, task_id={self._remote_task_id or 'unknown'}): {exc}",
                remote_task_id=self._remote_task_id,
                timeout_kind=timeout_kind,
                remote_task_terminal=remote_task_terminal,
                remote_task_state=remote_task_state,
            ) from exc
        except httpx.RequestError as exc:
            # Transport-level failures (connection refused/reset, server
            # disconnect, read/connect timeout) bubble up from httpx with an
            # opaque, sometimes empty message like "All connection attempts
            # failed" that gives no hint the parse engine was MinerU. HTTP
            # status errors and protocol errors already raise context-rich
            # RuntimeErrors via raise_for_status_with_detail, so they stay
            # untouched. Re-raise with the engine + endpoint and the exception
            # class name so the doc_status error_msg is always non-empty and
            # clearly attributable to the MinerU backend.
            raise RuntimeError(
                f"MinerU {self.api_mode} backend request failed "
                f"(endpoint={self.endpoint}): {type(exc).__name__}: {exc}"
            ) from exc

        self._normalize_raw_bundle(raw_dir, source_file_path, resolved_upload_name)
        return self._build_and_write_manifest(
            raw_dir, source_file_path, task_id, resolved_upload_name
        )

    async def _record_remote_task(self, task_id: str) -> None:
        """Record a server-issued task id before polling can time out."""
        self._remote_task_id = str(task_id or "")
        callback = self._on_remote_task
        if callback is None or not self._remote_task_id:
            return
        try:
            result = callback(self._remote_task_id)
            if inspect.isawaitable(result):
                await result
        except Exception as error:  # observability must not fail a parse
            logger.warning("MinerU task-id callback failed: %s", error)

    async def _confirm_remote_task_terminal(
        self, upload_name: str
    ) -> tuple[bool, str]:
        """Best-effort terminal-state probe after a request or poll timeout.

        A local or official MinerU job can keep running after the client times
        out. The caller uses the result to decide between immediate release and
        a conservative recovery lease. Wrapper mode has no portable task-status
        contract, so it always reports an unknown state.
        """
        task_id = self._remote_task_id
        if not task_id or self.api_mode == "wrapper" or httpx is None:
            return False, "unknown"
        probe_timeout = httpx.Timeout(
            min(10.0, self.http_timeout_seconds),
            connect=min(5.0, self.connect_timeout_seconds),
        )
        try:
            async with httpx.AsyncClient(timeout=probe_timeout) as client:
                if self.api_mode == "official":
                    encoded_task_id = quote(task_id, safe="")
                    response = await client.get(
                        f"{self.official_endpoint}/api/v4/extract-results/batch/{encoded_task_id}",
                        headers=self._official_headers(),
                    )
                    raise_for_status_with_detail(
                        response, "MinerU official timeout status probe"
                    )
                    payload = response.json() if response.text else {}
                    self._raise_if_official_error(
                        payload, "MinerU official timeout status probe"
                    )
                    results = _get_by_path(payload, "data.extract_result")
                    if isinstance(results, dict):
                        results = [results]
                    if not isinstance(results, list):
                        return False, "unknown"
                    selected = _select_official_extract_result(results, upload_name)
                    state = str((selected or {}).get("state") or "").lower()
                    return state in OFFICIAL_DONE_STATES | OFFICIAL_FAILED_STATES, state or "unknown"

                encoded_task_id = quote(task_id, safe="")
                response = await client.get(
                    f"{self.local_endpoint}/tasks/{encoded_task_id}"
                )
                raise_for_status_with_detail(response, "MinerU local timeout status probe")
                payload = response.json() if response.text else {}
                state = str(payload.get("status") or "").lower()
                return state in LOCAL_DONE_STATES | LOCAL_FAILED_STATES, state or "unknown"
        except Exception as error:
            logger.warning(
                "MinerU timeout status probe failed for task %s: %s", task_id, error
            )
            return False, "unknown"

    # ------------------------------------------------------------------
    # Upload + poll
    # ------------------------------------------------------------------

    def _official_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_token}",
        }

    def _official_payload(self, upload_name: str) -> dict[str, Any]:
        file_entry: dict[str, Any] = {"name": upload_name}
        if self.is_ocr:
            file_entry["is_ocr"] = True
        if self.page_ranges:
            file_entry["page_ranges"] = self.page_ranges
        return {
            "files": [file_entry],
            "model_version": self.model_version,
            "language": self.language,
            "enable_table": self.enable_table,
            "enable_formula": self.enable_formula,
        }

    async def _download_official(
        self,
        client: "httpx.AsyncClient",
        source_file_path: Path,
        raw_dir: Path,
        upload_name: str,
    ) -> str:
        apply_url = f"{self.official_endpoint}/api/v4/file-urls/batch"
        resp = await client.post(
            apply_url,
            headers=self._official_headers(),
            json=self._official_payload(upload_name),
        )
        raise_for_status_with_detail(resp, "MinerU official upload URL request")
        payload = resp.json() if resp.text else {}
        self._raise_if_official_error(payload, "MinerU official upload URL request")
        data = payload.get("data") if isinstance(payload, dict) else {}
        batch_id = str((data or {}).get("batch_id") or "")
        file_urls = (data or {}).get("file_urls") or []
        if not batch_id or not isinstance(file_urls, list) or not file_urls:
            raise RuntimeError(
                f"MinerU official upload URL response missing batch_id/file_urls: "
                f"{payload}"
            )
        await self._record_remote_task(batch_id)

        first_file_url = file_urls[0]
        if isinstance(first_file_url, dict):
            upload_url = str(
                first_file_url.get("url") or first_file_url.get("file_url") or ""
            )
        else:
            upload_url = str(first_file_url)
        if not upload_url:
            raise RuntimeError(
                f"MinerU official upload URL response had an empty upload URL: "
                f"{payload}"
            )
        upload_resp = await client.put(
            upload_url,
            content=_iter_file_bytes(source_file_path),
            headers={"Content-Length": str(source_file_path.stat().st_size)},
        )
        raise_for_status_with_detail(upload_resp, "MinerU official file upload")

        result_url = await self._poll_official_batch(client, batch_id, upload_name)
        await self._download_zip(client, result_url, raw_dir)
        return batch_id

    async def _poll_official_batch(
        self,
        client: "httpx.AsyncClient",
        batch_id: str,
        upload_name: str,
    ) -> str:
        encoded_batch_id = quote(batch_id, safe="")
        poll_url = (
            f"{self.official_endpoint}/api/v4/extract-results/batch/{encoded_batch_id}"
        )
        for _ in range(self.max_polls):
            await asyncio.sleep(self.poll_interval)
            resp = await client.get(poll_url, headers=self._official_headers())
            raise_for_status_with_detail(resp, "MinerU official batch poll")
            payload = resp.json() if resp.text else {}
            self._raise_if_official_error(payload, "MinerU official batch poll")
            results = _get_by_path(payload, "data.extract_result")
            if isinstance(results, dict):
                results = [results]
            if not isinstance(results, list):
                continue

            selected = _select_official_extract_result(results, upload_name)
            if selected is None:
                continue
            state = str(selected.get("state") or "").lower()
            if state in OFFICIAL_DONE_STATES:
                full_zip_url = str(selected.get("full_zip_url") or "")
                if not full_zip_url:
                    raise RuntimeError(
                        f"MinerU official batch {batch_id} is done but has no "
                        f"full_zip_url: {selected}"
                    )
                return full_zip_url
            if state in OFFICIAL_FAILED_STATES:
                err = selected.get("err_msg") or selected.get("error") or selected
                raise RuntimeError(
                    f"MinerU official parse failed for batch {batch_id}: {err}"
                )

        raise TimeoutError(f"MinerU official batch polling timeout: {batch_id}")

    def _raise_if_official_error(self, payload: Any, operation: str) -> None:
        if not isinstance(payload, dict):
            raise RuntimeError(f"{operation} returned non-object payload: {payload!r}")
        code = payload.get("code", 0)
        if code not in (0, "0", None):
            raise RuntimeError(
                f"{operation} failed: code={code} msg={payload.get('msg')!r}"
            )

    def _local_form_data(self) -> dict[str, str]:
        return {
            "lang_list": self.language,
            "backend": self.local_backend,
            "parse_method": self.local_parse_method,
            "formula_enable": _bool_form(self.enable_formula),
            "table_enable": _bool_form(self.enable_table),
            "image_analysis": _bool_form(self.local_image_analysis),
            "return_md": "true",
            "return_middle_json": "true",
            "return_model_output": "true",
            "return_content_list": "true",
            "return_images": "true",
            "response_format_zip": "true",
            "return_original_file": "true",
            "start_page_id": str(self.local_start_page_id),
            "end_page_id": str(self.local_end_page_id),
        }

    async def _download_local(
        self,
        client: "httpx.AsyncClient",
        source_file_path: Path,
        raw_dir: Path,
        upload_name: str,
    ) -> str:
        submit_url = f"{self.local_endpoint}/tasks"
        # Keep data as a Mapping so httpx 0.28 builds an async MultipartStream
        # and reads the file handle in chunks instead of buffering the payload.
        with source_file_path.open("rb") as fh:
            files = {"files": (upload_name, fh, "application/octet-stream")}
            resp = await client.post(
                submit_url,
                data=self._local_form_data(),
                files=files,
            )
        raise_for_status_with_detail(
            resp,
            f"MinerU local task submission for {upload_name!r}",
        )
        payload = resp.json() if resp.text else {}
        task_id = str(payload.get("task_id") or "")
        if not task_id:
            raise RuntimeError(
                f"MinerU local /tasks response missing task_id: {payload}"
            )
        await self._record_remote_task(task_id)

        await self._poll_local_task(client, task_id)
        await self._download_zip(
            client,
            f"{self.local_endpoint}/tasks/{quote(task_id, safe='')}/result",
            raw_dir,
        )
        return task_id

    async def _poll_local_task(
        self,
        client: "httpx.AsyncClient",
        task_id: str,
    ) -> None:
        # ``task_id`` is service-returned; encode it as a single path segment so
        # a crafted value can't break out of ``/tasks/{id}``. The raw value is
        # kept for the error/timeout messages below where the real ID matters.
        poll_url = f"{self.local_endpoint}/tasks/{quote(task_id, safe='')}"
        for _ in range(self.max_polls):
            await asyncio.sleep(self.poll_interval)
            resp = await client.get(poll_url)
            raise_for_status_with_detail(resp, "MinerU local task poll")
            payload = resp.json() if resp.text else {}
            status = str(payload.get("status") or "").lower()
            if status in LOCAL_DONE_STATES:
                return
            if status in LOCAL_FAILED_STATES:
                err = payload.get("error") or payload.get("message") or payload
                raise RuntimeError(
                    f"MinerU local parse failed for task {task_id}: {err}"
                )

        raise TimeoutError(f"MinerU local task polling timeout: {task_id}")

    # ------------------------------------------------------------------
    # Wrapper mode: single synchronous POST {endpoint}/predict
    # ------------------------------------------------------------------

    def _wrapper_payload(self, file_b64: str, upload_name: str) -> dict[str, Any]:
        return {
            "file": file_b64,
            "options": {
                "orig_suffix": Path(upload_name).suffix or ".pdf",
                "method": self.local_parse_method,
                "backend": self.local_backend,
                "lang": self.language,
                "formula_enable": self.enable_formula,
                "table_enable": self.enable_table,
            },
        }

    async def _download_wrapper(
        self,
        client: "httpx.AsyncClient",
        source_file_path: Path,
        raw_dir: Path,
        upload_name: str,
    ) -> str:
        submit_url = f"{self.wrapper_endpoint}/predict"
        file_bytes = await asyncio.to_thread(source_file_path.read_bytes)
        payload = self._wrapper_payload(
            base64.b64encode(file_bytes).decode("ascii"), upload_name
        )
        resp = await client.post(submit_url, json=payload)
        raise_for_status_with_detail(
            resp,
            f"MinerU wrapper predict for {upload_name!r}",
        )
        data = resp.json() if resp.text else {}
        if not isinstance(data, dict) or not data:
            raise RuntimeError(
                f"MinerU wrapper predict returned an empty or non-object "
                f"response for {upload_name!r}"
            )

        # Delivery priority: package_url zip > inline content_list > markdown.
        package_url = str(data.get("package_url") or "").strip()
        if package_url:
            await self._download_zip(client, package_url, raw_dir)
            self._ensure_wrapper_package_text_artifact(
                raw_dir, source_file_path, upload_name
            )
            return ""

        content_list = data.get("content_list")
        if isinstance(content_list, list) and content_list:
            logger.warning(
                "[mineru_raw] wrapper delivered content_list without "
                "package_url for %r: this delivery carries no image bytes, "
                "so image artifacts will be missing from the raw bundle",
                upload_name,
            )
            target = raw_dir / CONTENT_LIST_FILENAME
            await asyncio.to_thread(
                target.write_text,
                json.dumps(content_list, ensure_ascii=False, indent=2),
                "utf-8",
            )
            return ""

        markdown, md_url = await self._fetch_wrapper_markdown(
            client, data, upload_name
        )
        if markdown.strip():
            await self._land_wrapper_markdown(
                client, raw_dir, upload_name, markdown, md_url
            )
            return ""

        raise RuntimeError(
            f"MinerU wrapper predict response for {upload_name!r} has no "
            f"usable delivery (expected package_url, content_list, or "
            f"md/md_url); got keys: {sorted(data.keys())}"
        )

    async def _fetch_wrapper_markdown(
        self,
        client: "httpx.AsyncClient",
        data: Mapping[str, Any],
        upload_name: str,
    ) -> tuple[str, str]:
        """Return ``(markdown_text, md_url)`` from an inline field or md_url."""
        inline = data.get("md") or data.get("markdown") or data.get("md_content")
        if isinstance(inline, str) and inline.strip():
            return inline, ""
        md_url = str(data.get("md_url") or "").strip()
        if not md_url:
            return "", ""
        resp = await client.get(md_url)
        raise_for_status_with_detail(
            resp,
            f"MinerU wrapper markdown download for {upload_name!r}",
        )
        return resp.text, md_url

    async def _land_wrapper_markdown(
        self,
        client: "httpx.AsyncClient",
        raw_dir: Path,
        upload_name: str,
        markdown: str,
        md_url: str,
    ) -> None:
        """Land a markdown delivery onto the canonical raw_dir layout.

        Writes ``<stem>.md`` at raw_dir root, downloads relative image
        references next to ``md_url`` (path-traversal guarded), and
        synthesizes a root ``content_list.json`` so the manifest / IR
        builder contract holds for a text-only delivery.
        """
        stem = Path(upload_name).stem or "document"
        md_path = raw_dir / f"{stem}.md"
        await asyncio.to_thread(md_path.write_text, markdown, "utf-8")

        downloaded: list[str] = []
        if md_url:
            downloaded = await self._download_wrapper_md_assets(
                client, markdown, md_url, raw_dir
            )
        elif _markdown_image_refs(markdown):
            logger.warning(
                "[mineru_raw] wrapper markdown for %r has relative image "
                "references but no md_url to resolve them against; image "
                "artifacts will be missing from the raw bundle",
                upload_name,
            )

        content_list: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": markdown,
                "content": markdown,
                "page_idx": 0,
            }
        ]
        for rel in downloaded:
            content_list.append({"type": "image", "img_path": rel, "page_idx": 0})
        target = raw_dir / CONTENT_LIST_FILENAME
        await asyncio.to_thread(
            target.write_text,
            json.dumps(content_list, ensure_ascii=False, indent=2),
            "utf-8",
        )

    async def _download_wrapper_md_assets(
        self,
        client: "httpx.AsyncClient",
        markdown: str,
        md_url: str,
        raw_dir: Path,
    ) -> list[str]:
        """Download relative markdown image refs next to ``md_url``.

        Mirrors the legacy wrapper client's remote-asset handling: refs with
        a scheme or a server-absolute path are left untouched, a ref that
        would land outside ``raw_dir`` is skipped (path traversal guard),
        and per-asset failures degrade to a warning so the text artifact
        still lands. Returns the raw_dir-relative posix paths that were
        written.
        """
        downloaded: list[str] = []
        raw_root = raw_dir.resolve()
        for ref in _markdown_image_refs(markdown):
            dest = (raw_dir / ref).resolve()
            try:
                rel = dest.relative_to(raw_root)
            except ValueError:
                logger.warning(
                    "[mineru_raw] wrapper markdown asset ref %r escapes "
                    "raw_dir; skipped",
                    ref,
                )
                continue
            asset_url = urljoin(md_url, ref)
            try:
                resp = await client.get(asset_url)
                raise_for_status_with_detail(
                    resp, f"MinerU wrapper asset download {ref!r}"
                )
            except Exception as exc:
                logger.warning(
                    "[mineru_raw] wrapper markdown asset download failed "
                    "ref=%s url=%s err=%s",
                    ref,
                    asset_url,
                    exc,
                )
                continue

            def _write(content: bytes = resp.content, path: Path = dest) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)

            await asyncio.to_thread(_write)
            downloaded.append(rel.as_posix())
        return downloaded

    def _ensure_wrapper_package_text_artifact(
        self,
        raw_dir: Path,
        source_file_path: Path,
        upload_name: str,
    ) -> None:
        """A package_url-only response must still yield a text artifact.

        Bundles normally carry both ``*_content_list.json`` and ``*.md`` —
        the post-download normalization pass hoists the content list to
        raw_dir root, so a nested candidate is fine. When the bundle has no
        content list at all, fall back to its markdown and synthesize a root
        ``content_list.json``; raise only when the package has neither.
        """
        if (raw_dir / CONTENT_LIST_FILENAME).is_file():
            return
        if (
            _select_content_list_candidate(raw_dir, source_file_path, upload_name)
            is not None
        ):
            return
        md_path = _find_bundle_markdown(raw_dir)
        if md_path is None:
            raise RuntimeError(
                f"MinerU wrapper package for {upload_name!r} contains neither "
                f"a content_list JSON nor a markdown artifact "
                f"(raw_dir={raw_dir})"
            )
        logger.warning(
            "[mineru_raw] wrapper package for %r has no content_list JSON; "
            "synthesizing one from bundle markdown %s",
            upload_name,
            md_path.name,
        )
        markdown = md_path.read_text(encoding="utf-8")
        raw_root = raw_dir.resolve()
        content_list: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": markdown,
                "content": markdown,
                "page_idx": 0,
            }
        ]
        # Refs in the bundle markdown are relative to the md file's own
        # directory; record existing image files with raw_dir-relative paths.
        for ref in _markdown_image_refs(markdown):
            candidate = (md_path.parent / ref).resolve()
            try:
                rel = candidate.relative_to(raw_root)
            except ValueError:
                continue
            if candidate.is_file():
                content_list.append(
                    {"type": "image", "img_path": rel.as_posix(), "page_idx": 0}
                )
        (raw_dir / CONTENT_LIST_FILENAME).write_text(
            json.dumps(content_list, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    async def _download_zip(
        self,
        client: "httpx.AsyncClient",
        result_url: str,
        raw_dir: Path,
        resp: Any = None,
    ) -> None:
        """Download (or re-use already-fetched response) and extract."""
        if resp is None or not hasattr(resp, "content"):
            resp = await client.get(result_url)
            raise_for_status_with_detail(resp, "MinerU result bundle download")
        # Safe-extract with the shared result-bundle budget: refuse path
        # traversal / absolute entries AND cap declared entry count / total
        # uncompressed size. The default MINERU_ENDPOINT is the remote
        # mineru.net API, so a compromised or misbehaving endpoint returning a
        # zip declaring gigabytes must not expand unbounded onto disk — the same
        # defense-in-depth the docling path applies.
        max_entries, max_total_bytes = result_bundle_limits()
        safe_extract_zip(
            resp.content,
            raw_dir,
            max_entries=max_entries,
            max_total_bytes=max_total_bytes,
        )

        # Normalize: if the zip nested everything under a single top-level
        # dir, hoist its contents up so content_list.json sits at raw_dir
        # root. This matches the common MinerU bundle layout.
        self._maybe_hoist_single_subdir(raw_dir)

    def _maybe_hoist_single_subdir(self, raw_dir: Path) -> None:
        entries = [p for p in raw_dir.iterdir() if p.name != "_manifest.json"]
        if len(entries) != 1 or not entries[0].is_dir():
            return
        sub = entries[0]
        for child in list(sub.iterdir()):
            child.rename(raw_dir / child.name)
        try:
            sub.rmdir()
        except OSError:
            pass

    def _normalize_raw_bundle(
        self,
        raw_dir: Path,
        source_file_path: Path,
        upload_name: str | None = None,
    ) -> None:
        """Ensure a downloaded bundle has root-level ``content_list.json``.

        Official and local MinerU zip archives commonly place parser outputs at
        ``<doc>/<parse_method>/<doc>_content_list.json``. The adapter consumes a
        canonical root ``content_list.json`` plus optional root ``images/``.

        After hoisting we delete the nested originals so the manifest does not
        bookkeep two copies (and disk usage doesn't double for big bundles).
        Sibling artifacts of the parse subdir (``*.md``, ``middle.json`` etc.)
        are also hoisted to ``raw_dir`` root for easier diagnostics.
        """
        if (raw_dir / CONTENT_LIST_FILENAME).is_file():
            return

        candidate = _select_content_list_candidate(
            raw_dir, source_file_path, upload_name
        )
        if candidate is None:
            return

        source_dir = candidate.parent
        target_root = raw_dir.resolve()
        # Guard: never hoist from above raw_dir (defensive — candidate already
        # comes from rglob inside raw_dir, but cheap to verify).
        try:
            source_dir.resolve().relative_to(target_root)
        except ValueError:
            shutil.copy2(candidate, raw_dir / CONTENT_LIST_FILENAME)
            return

        # Move the critical file first; then hoist sibling files/dirs that
        # don't already exist at raw_dir root.
        shutil.move(str(candidate), str(raw_dir / CONTENT_LIST_FILENAME))
        for entry in list(source_dir.iterdir()):
            target = raw_dir / entry.name
            if target.exists():
                continue
            shutil.move(str(entry), str(target))

        # Best-effort cleanup of the now-empty parse subtree.
        cursor = source_dir
        while cursor != raw_dir and cursor.is_dir():
            try:
                cursor.rmdir()
            except OSError:
                break
            cursor = cursor.parent

    # ------------------------------------------------------------------
    # Manifest construction
    # ------------------------------------------------------------------

    def _build_and_write_manifest(
        self,
        raw_dir: Path,
        source_file_path: Path,
        task_id: str,
        upload_name: str,
    ) -> Manifest:
        source_size, source_hash = compute_size_and_hash(source_file_path)

        # Critical file — required.
        crit_path = raw_dir / CONTENT_LIST_FILENAME
        if not crit_path.is_file():
            raise RuntimeError(
                f"MinerU bundle missing required {CONTENT_LIST_FILENAME} "
                f"after download (raw_dir={raw_dir})"
            )
        crit_size, crit_hash = compute_size_and_hash(crit_path)

        # Other files.
        others: list[ManifestFile] = []
        total = crit_size
        for p in sorted(raw_dir.rglob("*")):
            if not p.is_file():
                continue
            if p.name == "_manifest.json":
                continue
            rel = p.relative_to(raw_dir).as_posix()
            if rel == CONTENT_LIST_FILENAME:
                continue
            size = p.stat().st_size
            others.append(ManifestFile(path=rel, size=size))
            total += size

        manifest = Manifest(
            source_content_hash=source_hash,
            source_size_bytes=source_size,
            source_filename_at_parse=upload_name,
            critical_file=ManifestFile(
                path=CONTENT_LIST_FILENAME,
                size=crit_size,
                sha256=crit_hash,
            ),
            files=others,
            total_size_bytes=total,
            task_id=task_id,
            api_mode=self.api_mode,
            engine_version=self.engine_version,
            endpoint_signature=self.endpoint,
            options_signature=self._options_signature(),
            downloaded_at=datetime.now(timezone.utc).isoformat(),
        )
        write_manifest(raw_dir, manifest)
        return manifest

    def _options_signature(self) -> str:
        return self._parser_options.signature()


def _find_content_list(payload: Any, content_field: str) -> list[dict] | None:
    """Heuristic content_list extractor.

    Tries (in order):

    1. The provided dotted path if it lands on a list of dicts.
    2. Direct ``content_list`` / ``content`` / ``items`` / ``result`` keys.
    3. Recursive descent.
    """
    if isinstance(payload, list):
        if payload and all(isinstance(x, dict) for x in payload):
            return payload
        return None
    if not isinstance(payload, dict):
        return None

    via_field = _get_by_path(payload, content_field)
    candidate = _find_content_list(via_field, content_field)
    if candidate is not None:
        return candidate

    for key in ("content_list", "content", "items", "result"):
        value = payload.get(key)
        candidate = _find_content_list(value, content_field)
        if candidate is not None:
            return candidate

    for value in payload.values():
        candidate = _find_content_list(value, content_field)
        if candidate is not None:
            return candidate
    return None


def _bool_form(value: bool) -> str:
    return "true" if value else "false"


def _markdown_image_refs(markdown: str) -> list[str]:
    """Extract deduplicated relative image refs from markdown, in order.

    Absolute URLs (any scheme) and server-absolute paths are skipped — they
    are neither downloaded nor rewritten, matching the remote-asset handling
    of the legacy wrapper client.
    """
    refs: list[str] = []
    seen: set[str] = set()
    for pattern in _MD_IMAGE_REF_RES:
        for match in pattern.findall(markdown or ""):
            ref = str(match).strip()
            if not ref or ref.startswith("/") or _ABSOLUTE_REF_RE.match(ref):
                continue
            clean = ref.split("?", 1)[0].split("#", 1)[0]
            if not clean or clean in seen:
                continue
            seen.add(clean)
            refs.append(clean)
    return refs


def _find_bundle_markdown(raw_dir: Path) -> Path | None:
    """Pick the most likely primary markdown artifact inside a bundle."""
    candidates = [p for p in raw_dir.rglob("*.md") if p.is_file()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: (len(p.relative_to(raw_dir).parts), p.as_posix()))
    return candidates[0]


def _select_official_extract_result(
    results: list[Any],
    source_filename: str,
) -> dict[str, Any] | None:
    """Pick the extract_result entry that matches the file we uploaded.

    Invariant: :meth:`MinerURawClient._download_official` always submits a
    single-file batch, so a non-matching ``file_name`` from the API would
    indicate either a server response we don't understand or a future
    multi-file extension. We fall back to ``dict_results[0]`` to remain
    forward-compatible but log a warning so the mismatch is visible.
    """
    dict_results = [item for item in results if isinstance(item, dict)]
    if not dict_results:
        return None
    source_name = Path(source_filename).name
    source_stem = Path(source_filename).stem
    for item in dict_results:
        file_name = str(item.get("file_name") or item.get("name") or "")
        if Path(file_name).name == source_name or Path(file_name).stem == source_stem:
            return item
    logger.warning(
        "[mineru_raw] official extract_result did not contain a match for "
        "%r; falling back to the first entry (%r). This is unexpected for "
        "a single-file batch.",
        source_name,
        str(dict_results[0].get("file_name") or dict_results[0].get("name") or ""),
    )
    return dict_results[0]


def _select_content_list_candidate(
    raw_dir: Path,
    source_file_path: Path,
    upload_name: str | None = None,
) -> Path | None:
    source_stem = Path(upload_name or source_file_path.name).stem
    candidates: list[tuple[int, int, str, Path]] = []
    for path in raw_dir.rglob("*.json"):
        if not path.is_file():
            continue
        if path.name != CONTENT_LIST_FILENAME and not path.name.endswith(
            "_content_list.json"
        ):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        content_list = _find_content_list(payload, "content")
        if content_list is None:
            continue

        score = 10
        if path.name == CONTENT_LIST_FILENAME:
            score = 0
        elif path.name == f"{source_stem}_content_list.json":
            score = 1
        elif path.stem.endswith("_content_list"):
            score = 2
        depth = len(path.relative_to(raw_dir).parts)
        candidates.append((score, depth, path.as_posix(), path))

    if not candidates:
        return None
    candidates.sort()
    return candidates[0][3]


__all__ = ["MinerURawClient", "CONTENT_LIST_FILENAME"]
