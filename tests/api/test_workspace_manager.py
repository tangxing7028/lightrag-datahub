"""Unit tests for multi-workspace instance management (upstream issue #2904).

Covers ``lightrag/api/workspace_manager.py``:

- header parsing + official sanitize rule (alphanumeric + underscore only,
  mirroring ``lightrag/api/config.py`` startup sanitization);
- env knobs (``LIGHTRAG_MAX_WORKSPACES`` / ``LIGHTRAG_WORKSPACE_TTL_SECONDS``);
- ``RAGInstanceManager``: default-instance pinning, cache hits, exactly-one
  concurrent build, LRU capacity eviction, idle TTL eviction, busy-workspace
  eviction skip, initialization failure not being cached, and ``aclose``.

All tests use a fake instance factory; no real storage backend is involved.
"""

import asyncio

import pytest

from lightrag.api.workspace_manager import (
    DEFAULT_MAX_WORKSPACES,
    DEFAULT_WORKSPACE_TTL_SECONDS,
    RAGInstanceManager,
    sanitize_workspace_name,
    workspace_from_headers,
    workspace_limits_from_env,
)

pytestmark = pytest.mark.offline


class FakeRAG:
    """Minimal LightRAG stand-in tracking the storage lifecycle."""

    def __init__(self, workspace: str, init_delay: float = 0.0):
        self.workspace = workspace
        self.init_delay = init_delay
        self.init_calls = 0
        self.finalize_calls = 0
        self.fail_init = False

    async def initialize_storages(self):
        self.init_calls += 1
        if self.fail_init:
            raise RuntimeError("init boom")
        if self.init_delay:
            await asyncio.sleep(self.init_delay)

    async def finalize_storages(self):
        self.finalize_calls += 1


class FakeClock:
    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_factory(init_delay: float = 0.0):
    """Factory returning (callable, created-instances list) for assertions."""
    created: list[FakeRAG] = []

    def factory(workspace: str) -> FakeRAG:
        instance = FakeRAG(workspace, init_delay=init_delay)
        created.append(instance)
        return instance

    return factory, created


def make_manager(**overrides) -> tuple[RAGInstanceManager, list[FakeRAG], FakeRAG]:
    """Manager with a pinned default instance and a recording factory."""
    factory, created = make_factory(init_delay=overrides.pop("init_delay", 0.0))
    default = FakeRAG("default_ws")
    default.initialized = True
    kwargs = dict(
        default_instance=default,
        default_workspace="default_ws",
        max_instances=4,
        ttl_seconds=0,
    )
    kwargs.update(overrides)
    return RAGInstanceManager(factory, **kwargs), created, default


# ---------------------------------------------------------------------------
# Header parsing and sanitization
# ---------------------------------------------------------------------------


def test_sanitize_workspace_name_replaces_invalid_characters():
    assert sanitize_workspace_name("team alpha!") == "team_alpha_"
    assert sanitize_workspace_name("already_ok_123") == "already_ok_123"
    assert sanitize_workspace_name("a-b.c/d") == "a_b_c_d"


def test_workspace_from_headers_absent_or_blank_returns_none():
    assert workspace_from_headers({}) is None
    assert workspace_from_headers({"LIGHTRAG-WORKSPACE": ""}) is None
    assert workspace_from_headers({"LIGHTRAG-WORKSPACE": "   "}) is None


def test_workspace_from_headers_sanitizes_value():
    assert workspace_from_headers({"LIGHTRAG-WORKSPACE": "beta"}) == "beta"
    assert workspace_from_headers({"LIGHTRAG-WORKSPACE": "team one"}) == "team_one"


# ---------------------------------------------------------------------------
# Environment knobs
# ---------------------------------------------------------------------------


def test_workspace_limits_from_env_defaults(monkeypatch):
    monkeypatch.delenv("LIGHTRAG_MAX_WORKSPACES", raising=False)
    monkeypatch.delenv("LIGHTRAG_WORKSPACE_TTL_SECONDS", raising=False)
    assert workspace_limits_from_env() == (
        DEFAULT_MAX_WORKSPACES,
        DEFAULT_WORKSPACE_TTL_SECONDS,
    )


def test_workspace_limits_from_env_valid_values(monkeypatch):
    monkeypatch.setenv("LIGHTRAG_MAX_WORKSPACES", "7")
    monkeypatch.setenv("LIGHTRAG_WORKSPACE_TTL_SECONDS", "60")
    assert workspace_limits_from_env() == (7, 60)


def test_workspace_limits_from_env_invalid_values_fall_back(monkeypatch):
    monkeypatch.setenv("LIGHTRAG_MAX_WORKSPACES", "not-a-number")
    monkeypatch.setenv("LIGHTRAG_WORKSPACE_TTL_SECONDS", "-5")
    assert workspace_limits_from_env() == (
        DEFAULT_MAX_WORKSPACES,
        DEFAULT_WORKSPACE_TTL_SECONDS,
    )
    monkeypatch.setenv("LIGHTRAG_MAX_WORKSPACES", "0")
    max_ws, _ = workspace_limits_from_env()
    assert max_ws == DEFAULT_MAX_WORKSPACES


# ---------------------------------------------------------------------------
# Instance resolution and caching
# ---------------------------------------------------------------------------


async def test_default_workspace_serves_pinned_instance():
    manager, created, default = make_manager()
    assert await manager.get_instance(None) is default
    assert await manager.get_instance("") is default
    assert await manager.get_instance("default_ws") is default
    assert created == []


async def test_cache_hit_builds_instance_only_once():
    manager, created, _ = make_manager()
    first = await manager.get_instance("alpha")
    second = await manager.get_instance("alpha")
    assert first is second
    assert len(created) == 1
    assert created[0].init_calls == 1
    assert created[0].workspace == "alpha"


async def test_concurrent_first_touch_builds_exactly_once():
    manager, created, _ = make_manager(init_delay=0.05)
    instances = await asyncio.gather(
        *(manager.get_instance("alpha") for _ in range(10))
    )
    assert len(created) == 1
    assert all(instance is created[0] for instance in instances)
    assert created[0].init_calls == 1


async def test_failed_initialization_is_not_cached():
    manager, created, _ = make_manager()
    created_future = created  # alias for clarity

    # Poison the next built instance.
    factory = manager._factory

    def failing_factory(workspace: str) -> FakeRAG:
        instance = factory(workspace)
        instance.fail_init = True
        return instance

    manager._factory = failing_factory
    with pytest.raises(RuntimeError, match="init boom"):
        await manager.get_instance("alpha")
    assert "alpha" not in manager
    assert len(created_future) == 1

    # A later request retries the build and succeeds.
    manager._factory = factory
    instance = await manager.get_instance("alpha")
    assert instance.init_calls == 1
    assert "alpha" in manager


# ---------------------------------------------------------------------------
# LRU capacity and TTL eviction
# ---------------------------------------------------------------------------


async def test_lru_capacity_evicts_oldest_non_default():
    manager, created, default = make_manager(max_instances=2)
    ws_a = await manager.get_instance("ws_a")
    ws_b = await manager.get_instance("ws_b")

    # Default + ws_b remain; ws_a was the LRU victim and got finalized.
    assert manager.active_workspaces == ["default_ws", "ws_b"]
    assert ws_a.finalize_calls == 1
    assert ws_b.finalize_calls == 0
    assert default.finalize_calls == 0

    # Re-requesting the evicted workspace builds a fresh instance.
    ws_a2 = await manager.get_instance("ws_a")
    assert ws_a2 is not ws_a
    assert ws_b.finalize_calls == 1  # evicted in turn


async def test_default_instance_is_never_evicted_by_capacity():
    manager, created, default = make_manager(max_instances=1)
    await manager.get_instance("ws_a")
    assert "default_ws" in manager
    assert default.finalize_calls == 0


async def test_ttl_evicts_idle_instances():
    clock = FakeClock()
    manager, created, _ = make_manager(ttl_seconds=100, clock=clock)
    ws_a = await manager.get_instance("ws_a")
    clock.advance(200)

    # Any cache-miss path sweeps expired entries first.
    await manager.get_instance("ws_b")
    assert ws_a.finalize_calls == 1
    assert "ws_a" not in manager
    assert "ws_b" in manager


async def test_ttl_disabled_keeps_instances_forever():
    clock = FakeClock()
    manager, created, _ = make_manager(ttl_seconds=0, clock=clock)
    ws_a = await manager.get_instance("ws_a")
    clock.advance(10_000)
    await manager.get_instance("ws_b")
    assert ws_a.finalize_calls == 0
    assert "ws_a" in manager


async def test_recently_used_instance_survives_ttl_sweep():
    clock = FakeClock()
    manager, created, _ = make_manager(ttl_seconds=100, clock=clock)
    ws_a = await manager.get_instance("ws_a")
    await manager.get_instance("ws_b")
    clock.advance(50)
    await manager.get_instance("ws_a")  # touch ws_a, ws_b now idler
    clock.advance(80)  # ws_a idle 80s (<100), ws_b idle 130s (>100)
    await manager.get_instance("ws_c")
    assert "ws_a" in manager
    assert "ws_b" not in manager


async def test_busy_workspace_skips_eviction():
    busy: set[str] = {"ws_a"}

    async def is_busy(workspace: str) -> bool:
        return workspace in busy

    manager, created, _ = make_manager(max_instances=2, is_workspace_busy=is_busy)
    ws_a = await manager.get_instance("ws_a")
    ws_b = await manager.get_instance("ws_b")

    # Capacity exceeded but the only eviction candidate is busy: the cache is
    # allowed to grow past the cap rather than finalize a running pipeline.
    assert "ws_a" in manager
    assert ws_a.finalize_calls == 0
    assert len(manager) == 3  # default + ws_a + ws_b

    busy.clear()
    await manager.get_instance("ws_c")
    assert ws_a.finalize_calls == 1


async def test_busy_check_error_keeps_instance():
    async def broken_busy_check(workspace: str) -> bool:
        raise RuntimeError("probe down")

    manager, created, _ = make_manager(
        max_instances=2, is_workspace_busy=broken_busy_check
    )
    ws_a = await manager.get_instance("ws_a")
    await manager.get_instance("ws_b")
    # Fail-closed: eviction skipped while liveness cannot be determined.
    assert ws_a.finalize_calls == 0
    assert "ws_a" in manager


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


async def test_aclose_finalizes_non_default_instances_only():
    manager, created, default = make_manager()
    ws_a = await manager.get_instance("ws_a")
    ws_b = await manager.get_instance("ws_b")

    await manager.aclose()

    assert ws_a.finalize_calls == 1
    assert ws_b.finalize_calls == 1
    assert default.finalize_calls == 0
    assert manager.active_workspaces == ["default_ws"]

    # The default instance remains fully usable after aclose.
    assert await manager.get_instance(None) is default


async def test_eviction_finalize_error_does_not_break_manager():
    manager, created, _ = make_manager(max_instances=2)
    ws_a = await manager.get_instance("ws_a")

    async def broken_finalize():
        raise RuntimeError("finalize boom")

    ws_a.finalize_storages = broken_finalize
    ws_b = await manager.get_instance("ws_b")
    assert "ws_a" not in manager
    assert manager.active_workspaces == ["default_ws", "ws_b"]
