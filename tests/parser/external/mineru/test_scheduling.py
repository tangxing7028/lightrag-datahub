"""Offline tests for the dynamic MinerU scheduling contract."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from lightrag.kg import shared_storage as ss
from lightrag.parser.external.mineru import scheduling as sched
from lightrag.pipeline import _mineru_admission_worker_count


pytestmark = pytest.mark.offline


def _payload(
    *,
    version: str = "config-v1",
    enabled: bool = True,
    worker_count: int = 4,
) -> dict:
    return {
        "parsers": {
            "services": {"mineru": {"baseUrl": "http://mineru.internal:9014"}},
            "scheduling": {
                "enabled": enabled,
                "globalCapacity": 4,
                "perKbCapacity": 4,
                "workerCount": worker_count,
                "sizeWeights": [
                    {"maxBytes": 5 * sched.MEBIBYTE, "weight": 1},
                    {"maxBytes": 15 * sched.MEBIBYTE, "weight": 2},
                    {"maxBytes": 30 * sched.MEBIBYTE, "weight": 2},
                    {"maxBytes": None, "weight": 4},
                ],
                "mineruConnectTimeoutSeconds": 30,
                "mineruReadTimeoutSeconds": 600,
                "pollIntervalSeconds": 2,
                "pollMaxAttempts": 600,
                "configVersion": version,
            },
        }
    }


@pytest.fixture(autouse=True)
def reset_scheduling_state():
    sched.reset_mineru_runtime_config_cache()
    ss.finalize_share_data()
    yield
    sched.reset_mineru_runtime_config_cache()
    ss.finalize_share_data()


def test_size_weight_boundaries_and_unknown_size_are_conservative():
    config = sched.MinerUSchedulingConfig.from_runtime_payload(_payload())

    assert config.weight_for_size(5 * sched.MEBIBYTE) == 1
    assert config.weight_for_size(5 * sched.MEBIBYTE + 1) == 2
    assert config.weight_for_size(15 * sched.MEBIBYTE) == 2
    assert config.weight_for_size(30 * sched.MEBIBYTE) == 2
    assert config.weight_for_size(30 * sched.MEBIBYTE + 1) == 4
    assert config.weight_for_size(None) == 4
    assert config.weight_for_size(0) == 4


def test_disabled_weighting_keeps_a_bounded_compatibility_weight():
    config = sched.MinerUSchedulingConfig.from_runtime_payload(_payload(enabled=False))

    assert config.effective_weight_for_size(100 * sched.MEBIBYTE) == 1


async def test_runtime_provider_keeps_last_valid_snapshot_after_failure():
    provider = sched._RuntimeConfigProvider()

    async def valid_loader():
        return _payload(version="loaded-v1")

    provider._loader = valid_loader
    first = await provider.get()
    assert first.config_version == "loaded-v1"

    async def failing_loader():
        raise RuntimeError("temporary config outage")

    provider._loader = failing_loader
    provider._expires_at = 0.0
    fallback = await provider.get()
    assert fallback is first
    assert provider.status()["last_error"] == "temporary config outage"


async def test_lease_persists_task_id_and_releases_a_confirmed_terminal_timeout(
    tmp_path: Path,
):
    ss.initialize_share_data(1)

    async def loader():
        return _payload(version="lease-v1")

    sched._runtime_config_provider._loader = loader
    source = tmp_path / "small.pdf"
    source.write_bytes(b"x")

    with pytest.raises(sched.MinerURequestTimeout):
        async with sched.mineru_parse_lease(
            workspace="kb_42", doc_id="doc-42", source_path=source
        ) as lease:
            await lease.record_remote_task("remote-42")
            namespace = await ss._get_lease_namespace()
            state = ss._load_weighted_gate_state(namespace, sched.LEASE_GROUP)
            assert state["leases"][lease.lease_id]["remote_task_id"] == "remote-42"
            raise sched.MinerURequestTimeout(
                "poll timed out",
                remote_task_id="remote-42",
                timeout_kind="poll",
                remote_task_terminal=True,
                remote_task_state="completed",
            )

    snapshot = await ss.weighted_lease_snapshot(sched.LEASE_GROUP)
    assert snapshot["global_used_capacity"] == 0
    assert snapshot["recovery_leases"] == 0


async def test_unconfirmed_timeout_holds_a_recovery_lease(tmp_path: Path):
    ss.initialize_share_data(1)

    async def loader():
        return _payload(version="recovery-v1")

    sched._runtime_config_provider._loader = loader
    source = tmp_path / "small.pdf"
    source.write_bytes(b"x")

    with pytest.raises(sched.MinerURequestTimeout):
        async with sched.mineru_parse_lease(
            workspace="kb_99", doc_id="doc-99", source_path=source
        ):
            raise sched.MinerURequestTimeout("read timed out")

    snapshot = await ss.weighted_lease_snapshot(sched.LEASE_GROUP)
    assert snapshot["global_used_capacity"] == 1
    assert snapshot["recovery_leases"] == 1


async def test_waiting_lease_rechecks_a_lowered_worker_count(tmp_path: Path):
    ss.initialize_share_data(1)
    configured_worker_count = 2

    async def loader():
        return _payload(
            version=f"workers-{configured_worker_count}",
            worker_count=configured_worker_count,
        )

    sched._runtime_config_provider._loader = loader
    source_paths = []
    for index in range(3):
        source = tmp_path / f"worker-{index}.pdf"
        source.write_bytes(b"x")
        source_paths.append(source)

    first_ready = asyncio.Event()
    second_ready = asyncio.Event()
    release_first = asyncio.Event()
    release_second = asyncio.Event()
    third_entered = asyncio.Event()

    async def hold_lease(doc_id: str, source: Path, ready, release):
        async with sched.mineru_parse_lease(
            workspace="kb_100", doc_id=doc_id, source_path=source
        ):
            ready.set()
            await release.wait()

    first_task = asyncio.create_task(
        hold_lease("doc-1", source_paths[0], first_ready, release_first)
    )
    second_task = asyncio.create_task(
        hold_lease("doc-2", source_paths[1], second_ready, release_second)
    )
    await asyncio.wait_for(first_ready.wait(), timeout=2)
    await asyncio.wait_for(second_ready.wait(), timeout=2)

    async def enter_third_lease():
        async with sched.mineru_parse_lease(
            workspace="kb_100", doc_id="doc-3", source_path=source_paths[2]
        ):
            third_entered.set()

    third_task = asyncio.create_task(enter_third_lease())
    await asyncio.sleep(sched.LEASE_POLL_SECONDS / 2)

    configured_worker_count = 1
    sched._runtime_config_provider._expires_at = 0.0
    release_first.set()
    await asyncio.sleep(sched.LEASE_POLL_SECONDS * 2)

    assert not third_entered.is_set()

    release_second.set()
    await asyncio.wait_for(third_entered.wait(), timeout=2)
    await asyncio.gather(first_task, second_task, third_task)


async def test_later_small_lease_bypasses_blocked_large_waiters(tmp_path: Path):
    """Waiting large documents cannot hide a one-unit gap from later small work."""
    ss.initialize_share_data(1)

    async def loader():
        return _payload(version="fair-admission-v1", worker_count=4)

    sched._runtime_config_provider._loader = loader
    source = tmp_path / "source.pdf"
    source.write_bytes(b"x")

    def lease(doc_id: str, file_size_bytes: int) -> sched.MinerUParseLease:
        result = sched.mineru_parse_lease(
            workspace="kb_101",
            doc_id=doc_id,
            source_path=source,
            static_request_cap=4,
        )
        result.file_size_bytes = file_size_bytes
        return result

    async def hold(
        parse_lease: sched.MinerUParseLease,
        entered: asyncio.Event,
        release: asyncio.Event,
    ) -> None:
        async with parse_lease:
            entered.set()
            await release.wait()

    medium_size = 5 * sched.MEBIBYTE + 1
    entered_events = [asyncio.Event() for _ in range(5)]
    release_events = [asyncio.Event() for _ in range(5)]
    leases = [
        lease("occupied-medium", medium_size),
        lease("occupied-small", 1),
        lease("blocked-large-a", medium_size),
        lease("blocked-large-b", medium_size),
        lease("later-small", 1),
    ]
    tasks: list[asyncio.Task] = []
    try:
        tasks.extend(
            asyncio.create_task(
                hold(leases[index], entered_events[index], release_events[index])
            )
            for index in (0, 1)
        )
        await asyncio.wait_for(entered_events[0].wait(), timeout=2)
        await asyncio.wait_for(entered_events[1].wait(), timeout=2)

        # The two medium waiters are older but cannot fit while the initial
        # medium + small requests use three of four global capacity units.
        tasks.extend(
            asyncio.create_task(
                hold(leases[index], entered_events[index], release_events[index])
            )
            for index in (2, 3)
        )
        for _ in range(100):
            if (await ss.weighted_lease_snapshot(sched.LEASE_GROUP))["waiters"] >= 2:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("blocked MinerU waiters were not registered")

        # The later small document must join the shared waiter set and use the
        # remaining unit. The old local gate held all four parser workers here.
        tasks.append(
            asyncio.create_task(hold(leases[4], entered_events[4], release_events[4]))
        )
        await asyncio.wait_for(entered_events[4].wait(), timeout=2)
        snapshot = await ss.weighted_lease_snapshot(sched.LEASE_GROUP)
        assert snapshot["global_used_capacity"] == 4
        assert snapshot["waiters"] == 2
    finally:
        for release in release_events:
            release.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_status_does_not_disclose_the_mineru_endpoint():
    ss.initialize_share_data(1)

    async def loader():
        return _payload(version="status-v1")

    sched._runtime_config_provider._loader = loader
    status = await sched.mineru_scheduling_status()

    assert status["config_version"] == "status-v1"
    assert "mineru.internal" not in str(status)
    assert "last_error" not in status["runtime_config"]


def test_pipeline_keeps_a_bounded_pool_for_pending_mineru_admission():
    assert _mineru_admission_worker_count(4, 16) == 16
    assert _mineru_admission_worker_count(4, 2) == 4
    assert _mineru_admission_worker_count(2, 4) == 4
    assert _mineru_admission_worker_count(0, 0) == 1
