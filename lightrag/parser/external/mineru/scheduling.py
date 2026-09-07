"""Dynamic, weighted admission control for remote MinerU parsing.

The Java configuration service owns the persisted contract. This module keeps
one short-lived, validated runtime snapshot per Python process and turns it
into an atomic shared-storage lease immediately before a MinerU request. The
lease therefore covers the upload/poll/result window, not later LightRAG
indexing work.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import threading
import time
import uuid
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from lightrag.kg import shared_storage
from lightrag.utils import logger

try:
    import httpx
except ImportError:  # pragma: no cover - API extra always provides httpx
    httpx = None


RUNTIME_CONFIG_CACHE_TTL_SECONDS = 10.0
LEASE_GROUP = "mineru_parse_weighted"
LEASE_POLL_SECONDS = 0.25
PRIORITY_LEASE_POLL_SECONDS = 0.05
HEARTBEAT_SECONDS = 5.0
DEFAULT_TIMEOUT_RECOVERY_TTL_SECONDS = 1200.0
MAX_TIMEOUT_RECOVERY_TTL_SECONDS = 3600.0
MEBIBYTE = 1024 * 1024


class MinerURequestTimeout(TimeoutError):
    """A MinerU timeout where the remote task may still be running."""

    def __init__(
        self,
        message: str,
        *,
        remote_task_id: str = "",
        timeout_kind: str = "read",
        remote_task_terminal: bool = False,
        remote_task_state: str = "",
    ) -> None:
        super().__init__(message)
        self.remote_task_id = str(remote_task_id or "")
        self.timeout_kind = str(timeout_kind or "read")
        self.remote_task_terminal = bool(remote_task_terminal)
        self.remote_task_state = str(remote_task_state or "")


@dataclass(frozen=True)
class SizeWeight:
    max_bytes: int | None
    weight: int


@dataclass(frozen=True)
class MinerUSchedulingConfig:
    enabled: bool
    global_capacity: int
    per_kb_capacity: int
    worker_count: int
    size_weights: tuple[SizeWeight, ...]
    mineru_connect_timeout_seconds: int
    mineru_read_timeout_seconds: int
    poll_interval_seconds: int
    poll_max_attempts: int
    config_version: str
    mineru_base_url: str = ""
    source: str = "fallback"

    @classmethod
    def defaults(cls) -> "MinerUSchedulingConfig":
        return cls(
            enabled=True,
            global_capacity=4,
            per_kb_capacity=4,
            worker_count=4,
            size_weights=(
                SizeWeight(5 * MEBIBYTE, 1),
                SizeWeight(15 * MEBIBYTE, 2),
                SizeWeight(30 * MEBIBYTE, 2),
                SizeWeight(None, 4),
            ),
            mineru_connect_timeout_seconds=30,
            mineru_read_timeout_seconds=600,
            poll_interval_seconds=2,
            poll_max_attempts=600,
            config_version="fallback",
        )

    @classmethod
    def from_runtime_payload(
        cls, payload: Mapping[str, Any]
    ) -> "MinerUSchedulingConfig":
        parsers = payload.get("parsers")
        if not isinstance(parsers, Mapping):
            raise ValueError("runtime config is missing parsers")
        scheduling = parsers.get("scheduling")
        if not isinstance(scheduling, Mapping):
            raise ValueError("runtime config is missing parsers.scheduling")
        defaults = cls.defaults()

        enabled = _strict_bool(scheduling.get("enabled"), "scheduling.enabled")
        global_capacity = _positive_int(
            scheduling.get("globalCapacity"), "scheduling.globalCapacity", 1, 64
        )
        per_kb_capacity = _positive_int(
            scheduling.get("perKbCapacity"),
            "scheduling.perKbCapacity",
            1,
            global_capacity,
        )
        worker_count = _positive_int(
            scheduling.get("workerCount"), "scheduling.workerCount", 1, global_capacity
        )
        connect_timeout = _positive_int(
            scheduling.get("mineruConnectTimeoutSeconds"),
            "scheduling.mineruConnectTimeoutSeconds",
            5,
            120,
        )
        read_timeout = _positive_int(
            scheduling.get("mineruReadTimeoutSeconds"),
            "scheduling.mineruReadTimeoutSeconds",
            60,
            3600,
        )
        poll_interval = _positive_int(
            scheduling.get("pollIntervalSeconds"),
            "scheduling.pollIntervalSeconds",
            1,
            30,
        )
        poll_max_attempts = _positive_int(
            scheduling.get("pollMaxAttempts"),
            "scheduling.pollMaxAttempts",
            1,
            3600,
        )
        raw_weights = scheduling.get("sizeWeights")
        if not isinstance(raw_weights, list) or len(raw_weights) != 4:
            raise ValueError("scheduling.sizeWeights must contain exactly four bands")
        weights: list[SizeWeight] = []
        previous_max: int | None = None
        for index, item in enumerate(raw_weights):
            if not isinstance(item, Mapping):
                raise ValueError("scheduling.sizeWeights entries must be objects")
            weight = _positive_int(
                item.get("weight"), f"sizeWeights[{index}].weight", 1, global_capacity
            )
            max_bytes_raw = item.get("maxBytes")
            if index == len(raw_weights) - 1:
                if max_bytes_raw is not None:
                    raise ValueError("the final size weight band must be unbounded")
                weights.append(SizeWeight(None, weight))
                continue
            max_bytes = _positive_int(
                max_bytes_raw, f"sizeWeights[{index}].maxBytes", 1, 2**63 - 1
            )
            if previous_max is not None and max_bytes <= previous_max:
                raise ValueError("size weight boundaries must be strictly increasing")
            previous_max = max_bytes
            weights.append(SizeWeight(max_bytes, weight))

        version = str(
            scheduling.get("configVersion") or defaults.config_version
        ).strip()
        services = parsers.get("services")
        mineru_service = services.get("mineru") if isinstance(services, Mapping) else {}
        mineru_base_url = ""
        if isinstance(mineru_service, Mapping):
            mineru_base_url = (
                str(mineru_service.get("baseUrl") or "").strip().rstrip("/")
            )
        return cls(
            enabled=enabled,
            global_capacity=global_capacity,
            per_kb_capacity=per_kb_capacity,
            worker_count=worker_count,
            size_weights=tuple(weights),
            mineru_connect_timeout_seconds=connect_timeout,
            mineru_read_timeout_seconds=read_timeout,
            poll_interval_seconds=poll_interval,
            poll_max_attempts=poll_max_attempts,
            config_version=version or defaults.config_version,
            mineru_base_url=mineru_base_url,
            source="ai-service",
        )

    def weight_for_size(self, file_size_bytes: int | None) -> int:
        """Return a conservative capacity charge for one server-side file."""
        if file_size_bytes is None or file_size_bytes <= 0:
            return self.size_weights[-1].weight
        for band in self.size_weights:
            if band.max_bytes is None or file_size_bytes <= band.max_bytes:
                return band.weight
        return self.size_weights[-1].weight

    def effective_weight_for_size(self, file_size_bytes: int | None) -> int:
        # Disabling weighted scheduling retains a fixed, bounded compatibility
        # policy. It never falls back to unbounded remote requests.
        return self.weight_for_size(file_size_bytes) if self.enabled else 1

    def timeout_recovery_seconds(self) -> float:
        poll_window = self.poll_interval_seconds * self.poll_max_attempts
        return min(
            MAX_TIMEOUT_RECOVERY_TTL_SECONDS,
            max(DEFAULT_TIMEOUT_RECOVERY_TTL_SECONDS, float(poll_window)),
        )

    def as_client_options(self, on_remote_task: Callable[[str], Any]) -> dict[str, Any]:
        return {
            "mineru_base_url": self.mineru_base_url,
            "connect_timeout_seconds": self.mineru_connect_timeout_seconds,
            "read_timeout_seconds": self.mineru_read_timeout_seconds,
            "poll_interval_seconds": self.poll_interval_seconds,
            "poll_max_attempts": self.poll_max_attempts,
            "on_remote_task": on_remote_task,
        }


def _positive_int(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}")
    return value


def _strict_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def server_file_size(path: Path) -> int | None:
    """Read the final server-side byte count; unknown and empty are conservative."""
    try:
        size = path.stat().st_size
    except OSError as error:
        logger.warning("MinerU scheduling could not stat source %s: %s", path, error)
        return None
    return size if isinstance(size, int) and size > 0 else None


class _RuntimeConfigProvider:
    """Per-process cache with a short TTL and last-valid-value fallback."""

    def __init__(self) -> None:
        self._config = MinerUSchedulingConfig.defaults()
        self._expires_at = 0.0
        self._last_error = ""
        self._guard = threading.RLock()
        self._loop_locks: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = weakref.WeakKeyDictionary()
        self._loader: Optional[Callable[[], Any]] = None

    async def get(self) -> MinerUSchedulingConfig:
        now = time.monotonic()
        with self._guard:
            if now < self._expires_at:
                return self._config
        loop = asyncio.get_running_loop()
        with self._guard:
            lock = self._loop_locks.get(loop)
            if lock is None:
                lock = asyncio.Lock()
                self._loop_locks[loop] = lock
        async with lock:
            now = time.monotonic()
            with self._guard:
                if now < self._expires_at:
                    return self._config
            try:
                loaded = await self._load()
            except Exception as error:
                with self._guard:
                    self._last_error = str(error)
                    self._expires_at = now + RUNTIME_CONFIG_CACHE_TTL_SECONDS
                    return self._config
            with self._guard:
                self._config = loaded
                self._last_error = ""
                self._expires_at = now + RUNTIME_CONFIG_CACHE_TTL_SECONDS
                return self._config

    async def _load(self) -> MinerUSchedulingConfig:
        if self._loader is not None:
            payload = self._loader()
            if hasattr(payload, "__await__"):
                payload = await payload
            if not isinstance(payload, Mapping):
                raise ValueError(
                    "configured MinerU runtime loader returned a non-object payload"
                )
            return MinerUSchedulingConfig.from_runtime_payload(payload)

        base_url = os.getenv("AI_SERVICE_URL", "").strip().rstrip("/")
        if not base_url:
            return MinerUSchedulingConfig.defaults()
        if httpx is None:
            raise RuntimeError(
                "httpx is required to load the MinerU runtime configuration"
            )
        prefix = os.getenv("AI_SERVICE_INTERNAL_PREFIX", "/ai/internal").strip("/")
        url = f"{base_url}/{prefix}/rag/config"
        timeout = httpx.Timeout(5.0, connect=2.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json={})
        response.raise_for_status()
        envelope = response.json()
        if not isinstance(envelope, Mapping) or envelope.get("code") != 200:
            raise RuntimeError(
                "ai-service rejected the MinerU runtime configuration request"
            )
        payload = envelope.get("data")
        if not isinstance(payload, Mapping):
            raise ValueError(
                "ai-service returned an invalid MinerU runtime configuration payload"
            )
        return MinerUSchedulingConfig.from_runtime_payload(payload)

    def status(self) -> dict[str, Any]:
        with self._guard:
            return {
                "source": self._config.source,
                "config_version": self._config.config_version,
                "cache_ttl_seconds": max(0.0, self._expires_at - time.monotonic()),
                "last_error": self._last_error,
            }

    def reset(self) -> None:
        with self._guard:
            self._config = MinerUSchedulingConfig.defaults()
            self._expires_at = 0.0
            self._last_error = ""
            self._loader = None


_runtime_config_provider = _RuntimeConfigProvider()


async def get_mineru_runtime_config() -> MinerUSchedulingConfig:
    """Return the current validated scheduling snapshot."""
    return await _runtime_config_provider.get()


def reset_mineru_runtime_config_cache() -> None:
    """Test/support hook: clear the process-local dynamic configuration cache."""
    _runtime_config_provider.reset()


class MinerUParseLease:
    """One acquired MinerU capacity lease and its immutable task snapshot."""

    def __init__(
        self,
        *,
        workspace: str,
        doc_id: str,
        source_path: Path,
        static_request_cap: int | None = None,
    ) -> None:
        self.workspace = workspace or "default"
        self.doc_id = str(doc_id or "")
        self.source_path = source_path
        self.file_size_bytes = server_file_size(source_path)
        self.owner_id = f"{socket.gethostname()}:{os.getpid()}"
        self.static_request_cap = self._positive_request_cap(static_request_cap)
        self.waiter_id = f"{self.owner_id}:{uuid.uuid4().hex}"
        self.config: MinerUSchedulingConfig | None = None
        self.weight = 0
        self.lease_id = ""
        self.admission: dict[str, Any] = {}
        self.remote_task_id = ""
        self._heartbeat_task: asyncio.Task | None = None
        self._recovery_held = False

    @staticmethod
    def _positive_request_cap(value: Any) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return max(1, value)

    def _owner_request_limit(self, config: MinerUSchedulingConfig) -> int:
        if self.static_request_cap is None:
            return config.worker_count
        return min(config.worker_count, self.static_request_cap)

    async def __aenter__(self) -> "MinerUParseLease":
        try:
            while True:
                # Queued documents are deliberately re-evaluated against the
                # newest config. Once admitted, this snapshot is never changed.
                config = await get_mineru_runtime_config()
                weight = config.effective_weight_for_size(self.file_size_bytes)
                owner_request_limit = self._owner_request_limit(config)
                metadata = {
                    "doc_id": self.doc_id,
                    "file_size_bytes": self.file_size_bytes,
                    "config_version": config.config_version,
                    "global_capacity_at_admission": config.global_capacity,
                    "per_kb_capacity_at_admission": config.per_kb_capacity,
                    "worker_count_at_admission": owner_request_limit,
                    "read_timeout_seconds": config.mineru_read_timeout_seconds,
                }
                (
                    admission,
                    priority_waiter,
                ) = await shared_storage.try_acquire_weighted_lease(
                    LEASE_GROUP,
                    kb_key=self.workspace,
                    weight=weight,
                    global_capacity=config.global_capacity,
                    per_kb_capacity=config.per_kb_capacity,
                    waiter_id=self.waiter_id,
                    owner_id=self.owner_id,
                    owner_request_limit=owner_request_limit,
                    metadata=metadata,
                )
                if admission is not None:
                    self.config = config
                    self.weight = weight
                    self.lease_id = str(admission["lease_id"])
                    self.admission = admission
                    self._heartbeat_task = asyncio.create_task(self._heartbeat())
                    logger.info(
                        "[mineru-scheduling] admitted doc_id=%s workspace=%s size=%s weight=%s "
                        "wait=%.3fs global=%s/%s kb=%s/%s requests=%s/%s config=%s",
                        self.doc_id,
                        self.workspace,
                        self.file_size_bytes,
                        self.weight,
                        admission["queue_wait_seconds"],
                        admission["global_used_capacity"],
                        admission["global_capacity"],
                        admission["kb_used_capacity"],
                        admission["per_kb_capacity"],
                        admission.get("owner_active_requests", 0),
                        admission.get("owner_request_limit", 0),
                        config.config_version,
                    )
                    return self
                await asyncio.sleep(
                    PRIORITY_LEASE_POLL_SECONDS
                    if priority_waiter
                    else LEASE_POLL_SECONDS
                )
        except BaseException:
            await shared_storage.clear_weighted_lease_waiter(
                LEASE_GROUP, self.waiter_id
            )
            raise

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            if isinstance(exc, MinerURequestTimeout):
                await self.hold_for_timeout(exc)
            await self._stop_heartbeat()
            if self.lease_id and not self._recovery_held:
                released = await shared_storage.release_weighted_lease(
                    LEASE_GROUP, self.lease_id
                )
                if released:
                    logger.info(
                        "[mineru-scheduling] released doc_id=%s workspace=%s weight=%s lease=%s",
                        self.doc_id,
                        self.workspace,
                        self.weight,
                        self.lease_id,
                    )
        finally:
            await shared_storage.clear_weighted_lease_waiter(
                LEASE_GROUP, self.waiter_id
            )

    def client_options(self) -> dict[str, Any]:
        if self.config is None:
            raise RuntimeError("MinerU lease has not been acquired")
        return self.config.as_client_options(self.record_remote_task)

    async def record_remote_task(self, remote_task_id: str) -> None:
        self.remote_task_id = str(remote_task_id or "")
        if self.lease_id and self.remote_task_id:
            recorded = await shared_storage.record_weighted_lease_remote_task(
                LEASE_GROUP, self.lease_id, self.remote_task_id
            )
            if not recorded:
                logger.warning(
                    "[mineru-scheduling] could not persist remote task id doc_id=%s lease=%s",
                    self.doc_id,
                    self.lease_id,
                )

    async def hold_for_timeout(self, error: MinerURequestTimeout) -> None:
        if not self.lease_id or self.config is None or self._recovery_held:
            return
        if error.remote_task_terminal:
            logger.info(
                "[mineru-scheduling] timeout confirmed terminal doc_id=%s workspace=%s "
                "weight=%s remote_task_id=%s state=%s",
                self.doc_id,
                self.workspace,
                self.weight,
                error.remote_task_id or self.remote_task_id or "unknown",
                error.remote_task_state or "unknown",
            )
            return
        remote_task_id = error.remote_task_id or self.remote_task_id
        held = await shared_storage.mark_weighted_lease_recovery(
            LEASE_GROUP,
            self.lease_id,
            recovery_seconds=self.config.timeout_recovery_seconds(),
            remote_task_id=remote_task_id,
            timeout_kind=error.timeout_kind,
        )
        if held:
            self._recovery_held = True
            logger.warning(
                "[mineru-scheduling] timeout recovery lease retained doc_id=%s workspace=%s "
                "weight=%s remote_task_id=%s recovery_seconds=%s config=%s",
                self.doc_id,
                self.workspace,
                self.weight,
                remote_task_id or "unknown",
                self.config.timeout_recovery_seconds(),
                self.config.config_version,
            )

    async def _heartbeat(self) -> None:
        try:
            while True:
                await asyncio.sleep(HEARTBEAT_SECONDS)
                if not self.lease_id:
                    return
                renewed = await shared_storage.renew_weighted_lease(
                    LEASE_GROUP, self.lease_id
                )
                if not renewed:
                    logger.warning(
                        "[mineru-scheduling] lease disappeared before heartbeat doc_id=%s lease=%s",
                        self.doc_id,
                        self.lease_id,
                    )
                    return
        except asyncio.CancelledError:
            raise
        except Exception as error:  # pragma: no cover - defensive heartbeat path
            logger.warning(
                "[mineru-scheduling] heartbeat failed doc_id=%s lease=%s: %s",
                self.doc_id,
                self.lease_id,
                error,
            )

    async def _stop_heartbeat(self) -> None:
        if self._heartbeat_task is None:
            return
        self._heartbeat_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._heartbeat_task
        self._heartbeat_task = None


def mineru_parse_lease(
    *,
    workspace: str,
    doc_id: str,
    source_path: Path,
    static_request_cap: int | None = None,
) -> MinerUParseLease:
    """Construct the context manager used at the actual MinerU request edge."""
    return MinerUParseLease(
        workspace=workspace,
        doc_id=doc_id,
        source_path=source_path,
        static_request_cap=static_request_cap,
    )


async def mineru_document_has_active_lease(workspace: str, doc_id: str) -> bool:
    """Whether a document still owns a live MinerU parse lease.

    Upload recovery uses this to distinguish an interrupted local pipeline row
    from a request whose remote MinerU task may still be running.
    """
    return await shared_storage.weighted_lease_exists_for_document(
        LEASE_GROUP,
        kb_key=workspace,
        doc_id=doc_id,
    )


async def mineru_scheduling_status() -> dict[str, Any]:
    """Authenticated-health snapshot without endpoint credentials or tokens."""
    config = await get_mineru_runtime_config()
    try:
        leases = await shared_storage.weighted_lease_snapshot(LEASE_GROUP)
        snapshot_available = True
    except Exception:  # health must not fail because a diagnostic store is down
        leases = {
            "global_used_capacity": 0,
            "by_kb": {},
            "active_leases": 0,
            "recovery_leases": 0,
            "waiters": 0,
            "recovered_count": 0,
        }
        snapshot_available = False
    provider_status = _runtime_config_provider.status()
    return {
        "enabled": config.enabled,
        "global_capacity": config.global_capacity,
        "per_kb_capacity": config.per_kb_capacity,
        "worker_count": config.worker_count,
        "size_weights": [
            {"max_bytes": item.max_bytes, "weight": item.weight}
            for item in config.size_weights
        ],
        "connect_timeout_seconds": config.mineru_connect_timeout_seconds,
        "read_timeout_seconds": config.mineru_read_timeout_seconds,
        "poll_interval_seconds": config.poll_interval_seconds,
        "poll_max_attempts": config.poll_max_attempts,
        "config_version": config.config_version,
        "runtime_config": {
            "source": provider_status["source"],
            "config_version": provider_status["config_version"],
            "cache_ttl_seconds": provider_status["cache_ttl_seconds"],
        },
        "snapshot_available": snapshot_available,
        "leases": leases,
    }
