"""Best-effort DataHub ingest terminal callback delivery.

LightRAG's persisted ``doc_status`` remains authoritative.  This bounded,
process-local queue only shortens ai-service's terminal observation latency;
dropping a notification never changes the runtime document result.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from lightrag.api.datahub_internal_auth import (
    resolve_datahub_internal_service_headers,
)
from lightrag.base import DocStatus
from lightrag.utils import logger

try:
    import httpx
except ImportError:  # pragma: no cover - API extra provides httpx
    httpx = None


_RETRY_DELAYS_SECONDS = (0.5, 2.0, 5.0, 10.0)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return min(maximum, max(minimum, value))


@dataclass(frozen=True)
class TerminalNotification:
    job_id: str
    doc_id: str
    kb_id: str
    workspace: str
    track_id: str
    status: str
    updated_at: str
    chunks_count: int | None = None
    error_msg: str | None = None

    @property
    def dedupe_key(self) -> tuple[str, str, str, str]:
        return (self.job_id, self.track_id, self.status, self.updated_at)

    def payload(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "kb_id": self.kb_id,
            "workspace": self.workspace,
            "track_id": self.track_id,
            "status": self.status,
            "runtime_updated_at": self.updated_at,
            "chunks_count": self.chunks_count,
            "error_msg": self.error_msg,
        }


class DataHubIngestCallbackDispatcher:
    """A bounded, non-blocking terminal notification dispatcher."""

    def __init__(
        self,
        *,
        enabled: bool,
        ai_service_url: str,
        queue_size: int = 4096,
        workers: int = 1,
        max_attempts: int = 5,
    ) -> None:
        self.enabled = bool(enabled)
        self.ai_service_url = str(ai_service_url or "").strip().rstrip("/")
        self.queue_size = max(1, int(queue_size))
        self.worker_count = max(1, int(workers))
        self.max_attempts = max(1, int(max_attempts))
        self._queue: asyncio.Queue[TerminalNotification] = asyncio.Queue(
            maxsize=self.queue_size
        )
        self._tasks: list[asyncio.Task] = []
        self._accepting = False
        self._seen: OrderedDict[tuple[str, str, str, str], None] = OrderedDict()
        self._seen_limit = max(self.queue_size * 2, 128)

    @classmethod
    def from_environment(cls) -> "DataHubIngestCallbackDispatcher":
        return cls(
            enabled=_env_bool("DATAHUB_INGEST_CALLBACK_ENABLED", False),
            ai_service_url=os.getenv("AI_SERVICE_URL", ""),
            queue_size=_env_int(
                "DATAHUB_INGEST_CALLBACK_QUEUE_SIZE", 4096, 1, 100_000
            ),
            workers=_env_int("DATAHUB_INGEST_CALLBACK_WORKERS", 1, 1, 16),
            max_attempts=_env_int(
                "DATAHUB_INGEST_CALLBACK_MAX_ATTEMPTS", 5, 1, 10
            ),
        )

    async def start(self) -> None:
        if not self.enabled:
            logger.info("DataHub ingest terminal callback is disabled")
            return
        if not self.ai_service_url:
            raise RuntimeError(
                "DATAHUB_INGEST_CALLBACK_ENABLED requires AI_SERVICE_URL"
            )
        if httpx is None:
            raise RuntimeError(
                "DATAHUB_INGEST_CALLBACK_ENABLED requires the httpx package"
            )
        prefix = os.getenv("AI_SERVICE_INTERNAL_PREFIX", "/ai/internal").strip("/")
        probe_url = f"{self.ai_service_url}/{prefix}/rag/ingest-jobs/1/terminal"
        if not resolve_datahub_internal_service_headers(probe_url):
            raise RuntimeError(
                "DATAHUB_INGEST_CALLBACK_ENABLED requires a valid internal "
                "ai-service URL and service token"
            )
        self._accepting = True
        self._tasks = [
            asyncio.create_task(self._worker(index), name=f"datahub-callback-{index}")
            for index in range(self.worker_count)
        ]
        logger.info(
            "DataHub ingest terminal callback started workers=%s queue_size=%s",
            self.worker_count,
            self.queue_size,
        )

    async def stop(self, timeout_seconds: float = 5.0) -> None:
        self._accepting = False
        if not self._tasks:
            return
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            logger.warning(
                "DataHub ingest callback shutdown timed out pending=%s",
                self._queue.qsize(),
            )
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def enqueue_terminal(self, notification: TerminalNotification) -> bool:
        if not self.enabled or not self._accepting:
            return False
        if notification.dedupe_key in self._seen:
            return True
        try:
            self._queue.put_nowait(notification)
        except asyncio.QueueFull:
            logger.warning(
                "DataHub ingest callback queue full job_id=%s doc_id=%s "
                "workspace=%s track_id=%s",
                notification.job_id,
                notification.doc_id,
                notification.workspace,
                notification.track_id,
            )
            return False
        self._seen[notification.dedupe_key] = None
        self._seen.move_to_end(notification.dedupe_key)
        while len(self._seen) > self._seen_limit:
            self._seen.popitem(last=False)
        return True

    async def _worker(self, worker_index: int) -> None:
        timeout = httpx.Timeout(5.0, connect=2.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            while True:
                notification = await self._queue.get()
                try:
                    await self._deliver(client, notification)
                except asyncio.CancelledError:
                    raise
                except Exception as error:  # delivery must never kill the worker
                    logger.warning(
                        "DataHub ingest callback worker error worker=%s job_id=%s "
                        "doc_id=%s error=%s",
                        worker_index,
                        notification.job_id,
                        notification.doc_id,
                        type(error).__name__,
                    )
                finally:
                    self._queue.task_done()

    async def _deliver(self, client: Any, notification: TerminalNotification) -> None:
        prefix = os.getenv("AI_SERVICE_INTERNAL_PREFIX", "/ai/internal").strip("/")
        url = (
            f"{self.ai_service_url}/{prefix}/rag/ingest-jobs/"
            f"{notification.job_id}/terminal"
        )
        headers = resolve_datahub_internal_service_headers(url)
        for attempt in range(1, self.max_attempts + 1):
            started = time.perf_counter()
            status_code: int | None = None
            retry = False
            try:
                response = await client.post(
                    url, json=notification.payload(), headers=headers
                )
                status_code = int(response.status_code)
                retry = status_code in {429} or status_code >= 500
                if status_code == 409:
                    retry = self._is_job_not_ready(response)
                if 200 <= status_code < 300:
                    logger.info(
                        "DataHub ingest callback delivered job_id=%s doc_id=%s "
                        "workspace=%s track_id=%s attempt=%s elapsed_ms=%s",
                        notification.job_id,
                        notification.doc_id,
                        notification.workspace,
                        notification.track_id,
                        attempt,
                        round((time.perf_counter() - started) * 1000),
                    )
                    return
            except Exception as error:
                retry = isinstance(
                    error,
                    (
                        httpx.TimeoutException,
                        httpx.NetworkError,
                        httpx.RemoteProtocolError,
                    ),
                )
                if not retry:
                    raise

            if not retry:
                logger.warning(
                    "DataHub ingest callback rejected job_id=%s doc_id=%s "
                    "workspace=%s track_id=%s http_status=%s attempt=%s",
                    notification.job_id,
                    notification.doc_id,
                    notification.workspace,
                    notification.track_id,
                    status_code,
                    attempt,
                )
                return
            if attempt >= self.max_attempts:
                break
            delay = _RETRY_DELAYS_SECONDS[
                min(attempt - 1, len(_RETRY_DELAYS_SECONDS) - 1)
            ]
            await asyncio.sleep(delay)
        logger.warning(
            "DataHub ingest callback exhausted retries job_id=%s doc_id=%s "
            "workspace=%s track_id=%s attempts=%s last_http_status=%s",
            notification.job_id,
            notification.doc_id,
            notification.workspace,
            notification.track_id,
            self.max_attempts,
            status_code,
        )

    @staticmethod
    def _is_job_not_ready(response: Any) -> bool:
        try:
            body = response.json()
        except Exception:
            return False
        return "RAG_JOB_NOT_READY" in str(body)


_dispatcher: DataHubIngestCallbackDispatcher | None = None


async def start_datahub_ingest_callback_dispatcher() -> None:
    global _dispatcher
    dispatcher = DataHubIngestCallbackDispatcher.from_environment()
    await dispatcher.start()
    _dispatcher = dispatcher


async def stop_datahub_ingest_callback_dispatcher() -> None:
    global _dispatcher
    dispatcher, _dispatcher = _dispatcher, None
    if dispatcher is not None:
        await dispatcher.stop()


async def enqueue_datahub_terminal_callback(
    *,
    doc_id: str,
    workspace: str,
    track_id: str,
    status: DocStatus | str,
    updated_at: str,
    metadata: dict[str, Any] | None,
    chunks_count: int | None,
    error_msg: str | None,
) -> bool:
    """Validate persisted identity and enqueue one terminal callback."""
    dispatcher = _dispatcher
    if dispatcher is None or not dispatcher.enabled:
        return False
    status_value = status.value if isinstance(status, DocStatus) else str(status)
    if status_value not in {DocStatus.PROCESSED.value, DocStatus.FAILED.value}:
        return False
    job_id = str((metadata or {}).get("datahub_job_id") or "").strip()
    workspace_value = str(workspace or "").strip()
    track_value = str(track_id or "").strip()
    doc_value = str(doc_id or "").strip()
    kb_id = workspace_value.removeprefix("kb_")
    if not (
        job_id.isdecimal()
        and 1 <= len(job_id) <= 20
        and kb_id.isdecimal()
        and doc_value
        and track_value
        and updated_at
    ):
        return False
    return await dispatcher.enqueue_terminal(
        TerminalNotification(
            job_id=job_id,
            doc_id=doc_value,
            kb_id=kb_id,
            workspace=workspace_value,
            track_id=track_value,
            status=status_value,
            updated_at=str(updated_at),
            chunks_count=chunks_count,
            error_msg=error_msg,
        )
    )
