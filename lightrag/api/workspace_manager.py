"""Multi-workspace LightRAG instance management for the API server.

The official server historically bound one global LightRAG instance to a
single workspace (``--workspace`` / ``WORKSPACE``). Storage backends already
isolate data per workspace (PostgreSQL tables are keyed by ``(workspace,
id)``, file-based storages place data in per-workspace subdirectories), but
every API route used the same instance regardless of the caller's
``LIGHTRAG-WORKSPACE`` header — the cross-workspace leakage reported in
upstream issue #2904.

This module provides the routing half of the fix:

- :func:`sanitize_workspace_name` / :func:`workspace_from_headers` apply the
  official workspace sanitizing rule (alphanumeric + underscore only, see
  ``lightrag/api/config.py``) to the ``LIGHTRAG-WORKSPACE`` request header.
- :class:`RAGInstanceManager` creates, caches and expires one LightRAG
  instance per workspace on demand, with an LRU capacity cap and an idle
  TTL so a long-lived process does not accumulate instances forever.

Shared-resource note: every instance built here shares the process-wide
PostgreSQL connection pool (``ClientManager.get_client`` in
``lightrag/kg/postgres_impl.py`` is ref-counted and keyed by the process
configuration, not by instance), so the number of physical database
connections does NOT scale with the number of active workspaces. The
``max_instances`` cap exists to bound per-instance in-memory state (locks,
pipeline executors, tokenizer caches), not connection count.
"""

import asyncio
import os
import re
import time
from collections import OrderedDict
from typing import Any, Awaitable, Callable, Optional

from fastapi import Request
from fastapi.params import Depends

from lightrag.utils import logger

#: Request header selecting the target workspace.
WORKSPACE_HEADER = "LIGHTRAG-WORKSPACE"

#: Default cap on simultaneously cached non-default instances.
DEFAULT_MAX_WORKSPACES = 32

#: Default idle time after which a cached instance is finalized and dropped.
DEFAULT_WORKSPACE_TTL_SECONDS = 86400

#: Official sanitize rule (mirrors lightrag/api/config.py): only alphanumeric
#: characters and underscores are allowed; anything else becomes "_".
_WORKSPACE_SANITIZE_PATTERN = re.compile(r"[^a-zA-Z0-9_]")


def sanitize_workspace_name(workspace: str) -> str:
    """Sanitize a workspace name using the official rule.

    Only alphanumeric characters and underscores are allowed; every other
    character is replaced with an underscore (same behavior as the
    ``--workspace`` startup sanitization in ``lightrag/api/config.py``).
    """
    sanitized = _WORKSPACE_SANITIZE_PATTERN.sub("_", workspace)
    if sanitized != workspace:
        logger.warning(
            f"Workspace name '{workspace}' contains invalid characters. "
            f"Sanitized to '{sanitized}'. "
            "Only alphanumeric characters and underscores are allowed."
        )
    return sanitized


def workspace_from_headers(headers: Any) -> Optional[str]:
    """Extract the sanitized workspace from request headers.

    Returns ``None`` when the ``LIGHTRAG-WORKSPACE`` header is absent or
    blank, meaning "use the server default workspace".
    """
    workspace = headers.get(WORKSPACE_HEADER, "").strip()
    if not workspace:
        return None
    return sanitize_workspace_name(workspace)


class RAGDependency(Depends):
    """FastAPI dependency resolving the per-request LightRAG instance.

    Used directly as a parameter default (it *is* a ``Depends`` marker):

        resolve_request_rag = make_rag_dependency(rag)
        ...
        async def endpoint(rag: LightRAG = resolve_request_rag): ...

    Resolution order: when the serving app exposes a
    :class:`RAGInstanceManager` on ``app.state.rag_manager``, the
    ``LIGHTRAG-WORKSPACE`` header selects the instance (absent header → the
    default workspace). Without a manager — single-workspace embedding and
    the route unit tests that build routers directly — the dependency
    always returns ``default_rag``, preserving the historical behavior.

    The marker also proxies attribute access to the default instance: tests
    that invoke endpoint functions in-process (bypassing FastAPI's
    dependency injection) receive this object as ``rag`` and transparently
    operate on the default instance, exactly as they did when the routes
    closed over a single global instance.
    """

    def __init__(self, default_rag: Any):
        super().__init__(dependency=self._resolve)
        self._default_rag = default_rag

    async def _resolve(self, request: Request) -> Any:
        manager = getattr(request.app.state, "rag_manager", None)
        if manager is None:
            return self._default_rag
        return await manager.get_instance(workspace_from_headers(request.headers))

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes not found on the marker itself.
        return getattr(self._default_rag, name)


def make_rag_dependency(default_rag: Any) -> RAGDependency:
    """Build the per-request instance dependency for one router factory."""
    return RAGDependency(default_rag)


def workspace_limits_from_env() -> tuple[int, int]:
    """Read manager capacity knobs from the environment.

    - ``LIGHTRAG_MAX_WORKSPACES``: max cached instances including the
      default one (default 32). Must be >= 1.
    - ``LIGHTRAG_WORKSPACE_TTL_SECONDS``: idle TTL for cached instances
      (default 86400). 0 disables TTL eviction.

    Invalid values fall back to the defaults with a warning.
    """
    max_workspaces = DEFAULT_MAX_WORKSPACES
    raw_max = os.getenv("LIGHTRAG_MAX_WORKSPACES")
    if raw_max:
        try:
            max_workspaces = int(raw_max)
            if max_workspaces < 1:
                raise ValueError
        except ValueError:
            logger.warning(
                f"Invalid LIGHTRAG_MAX_WORKSPACES={raw_max!r}; "
                f"falling back to {DEFAULT_MAX_WORKSPACES}"
            )
            max_workspaces = DEFAULT_MAX_WORKSPACES

    ttl_seconds = DEFAULT_WORKSPACE_TTL_SECONDS
    raw_ttl = os.getenv("LIGHTRAG_WORKSPACE_TTL_SECONDS")
    if raw_ttl:
        try:
            ttl_seconds = int(raw_ttl)
            if ttl_seconds < 0:
                raise ValueError
        except ValueError:
            logger.warning(
                f"Invalid LIGHTRAG_WORKSPACE_TTL_SECONDS={raw_ttl!r}; "
                f"falling back to {DEFAULT_WORKSPACE_TTL_SECONDS}"
            )
            ttl_seconds = DEFAULT_WORKSPACE_TTL_SECONDS

    return max_workspaces, ttl_seconds


class RAGInstanceManager:
    """Create, cache and expire one LightRAG instance per workspace.

    The ``factory`` builds an *uninitialized* instance for a workspace name;
    the manager awaits ``initialize_storages()`` before an instance becomes
    visible to concurrent callers, so a partially-initialized instance is
    never served. Initialization is guarded by a double-checked asyncio
    lock: concurrent requests for the same workspace trigger exactly one
    build.

    The default instance (the one the server builds for ``--workspace``) is
    pinned: it is never evicted by the LRU cap or the TTL, because the
    server lifespan owns its shutdown. All other instances are finalized
    (``finalize_storages()``) when evicted or when :meth:`aclose` runs.

    Eviction is best-effort safe against in-flight ingestion: when
    ``is_workspace_busy`` is provided, an instance whose workspace pipeline
    reports activity is skipped (the cache may temporarily exceed
    ``max_instances``). A request already holding an instance reference is
    not tracked; the TTL/cap knobs should be sized so instances outlive the
    longest expected request.

    Args:
        factory: sync callable ``(workspace: str) -> LightRAG`` building an
            uninitialized instance.
        default_instance: the already-initialized default LightRAG instance.
        default_workspace: workspace name of the default instance.
        max_instances: LRU capacity including the default instance.
        ttl_seconds: idle seconds after which an instance is evicted;
            0 disables TTL eviction.
        is_workspace_busy: optional async callable ``(workspace) -> bool``
            consulted before evicting; eviction is skipped for busy
            workspaces.
        clock: monotonic clock, injectable for tests.
    """

    def __init__(
        self,
        factory: Callable[[str], Any],
        *,
        default_instance: Any = None,
        default_workspace: str = "",
        max_instances: int = DEFAULT_MAX_WORKSPACES,
        ttl_seconds: int = DEFAULT_WORKSPACE_TTL_SECONDS,
        is_workspace_busy: Optional[Callable[[str], Awaitable[bool]]] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        if max_instances < 1:
            raise ValueError("max_instances must be >= 1")
        self._factory = factory
        self._default_workspace = default_workspace or ""
        self._max_instances = max_instances
        self._ttl_seconds = ttl_seconds
        self._is_workspace_busy = is_workspace_busy
        self._clock = clock
        # LRU order: oldest entry first.
        self._instances: OrderedDict[str, Any] = OrderedDict()
        self._last_used: dict[str, float] = {}
        self._lock = asyncio.Lock()
        if default_instance is not None:
            self._instances[self._default_workspace] = default_instance
            self._last_used[self._default_workspace] = self._clock()

    @property
    def default_workspace(self) -> str:
        return self._default_workspace

    @property
    def active_workspaces(self) -> list[str]:
        """Workspace names with a cached (initialized) instance, LRU last."""
        return list(self._instances.keys())

    def __len__(self) -> int:
        return len(self._instances)

    def __contains__(self, workspace: str) -> bool:
        return workspace in self._instances

    async def get_instance(self, workspace: Optional[str]) -> Any:
        """Return the initialized instance for ``workspace``.

        ``None``/empty resolves to the default workspace. A cache miss
        builds and initializes a new instance under the manager lock, so
        concurrent first-touch requests share exactly one build.
        """
        ws = workspace or self._default_workspace
        instance = self._instances.get(ws)
        if instance is not None:
            self._touch(ws)
            return instance
        async with self._lock:
            await self._evict_expired()
            instance = self._instances.get(ws)
            if instance is None:
                instance = self._factory(ws)
                await instance.initialize_storages()
                self._instances[ws] = instance
                self._last_used[ws] = self._clock()
                await self._enforce_capacity(protect=ws)
            else:
                self._touch(ws)
            return instance

    async def aclose(self) -> None:
        """Finalize and drop all non-default cached instances.

        The default instance is deliberately left alive: the server
        lifespan finalizes it as part of its normal shutdown sequence.
        """
        async with self._lock:
            for ws in list(self._instances.keys()):
                if ws == self._default_workspace:
                    continue
                await self._evict(ws)

    def _touch(self, workspace: str) -> None:
        self._last_used[workspace] = self._clock()
        self._instances.move_to_end(workspace)

    async def _workspace_busy(self, workspace: str) -> bool:
        if self._is_workspace_busy is None:
            return False
        try:
            return bool(await self._is_workspace_busy(workspace))
        except Exception as e:
            # Fail closed for eviction: when liveness cannot be determined,
            # keep the instance rather than finalize storage under a
            # possibly-running pipeline.
            logger.warning(
                f"Busy check failed for workspace '{workspace}'; "
                f"skipping eviction: {e}"
            )
            return True

    async def _evict(self, workspace: str) -> None:
        instance = self._instances.pop(workspace, None)
        self._last_used.pop(workspace, None)
        if instance is None:
            return
        try:
            await instance.finalize_storages()
        except Exception as e:
            logger.error(
                f"Error finalizing LightRAG instance for workspace "
                f"'{workspace}': {e}"
            )
        logger.info(f"Evicted LightRAG instance for workspace '{workspace}'")

    async def _evict_expired(self) -> None:
        if self._ttl_seconds <= 0:
            return
        now = self._clock()
        for ws in list(self._instances.keys()):
            if ws == self._default_workspace:
                continue
            if now - self._last_used[ws] <= self._ttl_seconds:
                continue
            if await self._workspace_busy(ws):
                continue
            await self._evict(ws)

    async def _enforce_capacity(self, protect: Optional[str] = None) -> None:
        while len(self._instances) > self._max_instances:
            victim = None
            for ws in self._instances.keys():  # LRU first
                if ws == self._default_workspace:
                    continue
                # Never finalize the instance just handed to the current
                # caller; it would serve a request with closed storages.
                if ws == protect:
                    continue
                if await self._workspace_busy(ws):
                    continue
                victim = ws
                break
            if victim is None:
                logger.warning(
                    f"All non-default LightRAG instances are busy; cache "
                    f"size {len(self._instances)} exceeds "
                    f"LIGHTRAG_MAX_WORKSPACES={self._max_instances}"
                )
                return
            await self._evict(victim)
