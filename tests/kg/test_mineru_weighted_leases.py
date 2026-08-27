"""Offline coverage for MinerU's weighted, cross-worker capacity ledger."""

from __future__ import annotations

import asyncio
import time

import pytest

from lightrag.kg import shared_storage as ss


GROUP = "mineru:weighted-test"

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def clean_shared_storage():
    ss.finalize_share_data()
    ss.initialize_share_data(1)
    yield
    ss.finalize_share_data()


async def _acquire(
    waiter_id: str,
    *,
    kb_key: str = "kb_1",
    weight: int = 1,
    global_capacity: int = 4,
    per_kb_capacity: int = 4,
    owner_id: str = "",
    owner_request_limit: int | None = None,
):
    return await ss.try_acquire_weighted_lease(
        GROUP,
        kb_key=kb_key,
        weight=weight,
        global_capacity=global_capacity,
        per_kb_capacity=per_kb_capacity,
        waiter_id=waiter_id,
        owner_id=owner_id,
        owner_request_limit=owner_request_limit,
    )


async def _state() -> dict:
    namespace = await ss._get_lease_namespace()
    return ss._load_weighted_gate_state(namespace, GROUP)


async def test_mixed_weights_share_one_capacity_ledger():
    first, _ = await _acquire("small-a", weight=1)
    second, _ = await _acquire("small-b", weight=1)
    third, _ = await _acquire("medium", weight=2)

    assert first and second and third
    exhausted, _ = await _acquire("one-too-many", weight=1)
    assert exhausted is None

    snapshot = await ss.weighted_lease_snapshot(GROUP)
    assert snapshot["global_used_capacity"] == 4
    assert snapshot["by_kb"] == {"kb_1": 4}
    assert snapshot["active_leases"] == 3

    for admission in (first, second, third):
        assert await ss.release_weighted_lease(GROUP, admission["lease_id"])
    assert (await ss.weighted_lease_snapshot(GROUP))["global_used_capacity"] == 0


async def test_per_kb_and_global_caps_are_both_enforced():
    kb_a, _ = await _acquire("kb-a", kb_key="kb_a", weight=2, per_kb_capacity=2)
    assert kb_a
    same_kb, _ = await _acquire("kb-a-over", kb_key="kb_a", weight=1, per_kb_capacity=2)
    assert same_kb is None

    kb_b, _ = await _acquire("kb-b", kb_key="kb_b", weight=2, per_kb_capacity=2)
    assert kb_b
    global_over, _ = await _acquire(
        "kb-c-over", kb_key="kb_c", weight=1, per_kb_capacity=2
    )
    assert global_over is None

    snapshot = await ss.weighted_lease_snapshot(GROUP)
    assert snapshot["global_used_capacity"] == 4
    assert snapshot["by_kb"] == {"kb_a": 2, "kb_b": 2}


async def test_small_waiter_can_use_capacity_while_large_waiter_is_blocked():
    occupied, _ = await _acquire("occupied", weight=3)
    assert occupied

    large_waiter, _ = await _acquire("large", weight=2)
    assert large_waiter is None
    small_waiter, _ = await _acquire("small", weight=1)
    assert small_waiter

    assert await ss.release_weighted_lease(GROUP, occupied["lease_id"])
    large_after_release, priority = await _acquire("large", weight=2)
    assert large_after_release
    assert priority is True


async def test_concurrent_attempts_never_oversell_global_capacity():
    results = await asyncio.gather(
        *(_acquire(f"concurrent-{index}") for index in range(12))
    )
    admitted = [admission for admission, _ in results if admission is not None]

    assert len(admitted) == 4
    assert (await ss.weighted_lease_snapshot(GROUP))["global_used_capacity"] == 4


async def test_owner_request_limit_counts_active_and_recovery_leases():
    owner_id = "runtime-a:1234"
    first, _ = await _acquire("owner-first", owner_id=owner_id, owner_request_limit=2)
    second, _ = await _acquire("owner-second", owner_id=owner_id, owner_request_limit=2)
    assert first and second

    blocked, _ = await _acquire("owner-third", owner_id=owner_id, owner_request_limit=2)
    assert blocked is None

    assert await ss.mark_weighted_lease_recovery(
        GROUP, first["lease_id"], recovery_seconds=60
    )
    still_blocked, _ = await _acquire(
        "owner-third", owner_id=owner_id, owner_request_limit=2
    )
    assert still_blocked is None

    assert await ss.release_weighted_lease(GROUP, second["lease_id"])
    admitted, _ = await _acquire(
        "owner-third", owner_id=owner_id, owner_request_limit=2
    )
    assert admitted
    assert admitted["owner_active_requests"] == 2
    assert admitted["owner_request_limit"] == 2

    state = await _state()
    owner_leases = [
        lease for lease in state["leases"].values() if lease.get("owner_id") == owner_id
    ]
    assert len(owner_leases) == 2
    assert sum(lease.get("state") == "recovery" for lease in owner_leases) == 1

    assert await ss.release_weighted_lease(GROUP, first["lease_id"])
    assert await ss.release_weighted_lease(GROUP, admitted["lease_id"])


async def test_weighted_lease_rejects_non_integer_capacity_inputs():
    with pytest.raises(ValueError, match="positive integers"):
        await _acquire("fractional-weight", weight=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive integers"):
        await _acquire("fractional-capacity", global_capacity=4.5)  # type: ignore[arg-type]


async def test_heartbeat_remote_task_and_recovery_expiry_are_idempotent(monkeypatch):
    admission, _ = await _acquire("recoverable")
    assert admission
    lease_id = admission["lease_id"]

    assert await ss.record_weighted_lease_remote_task(GROUP, lease_id, "mineru-task-7")
    state = await _state()
    assert state["leases"][lease_id]["remote_task_id"] == "mineru-task-7"

    monkeypatch.setattr(ss, "_heartbeat_ttl", 1.0)
    monkeypatch.setattr(ss, "_suspect_grace", 1.0)
    state["leases"][lease_id]["updated_at"] = time.time() - 5.0
    namespace = await ss._get_lease_namespace()
    namespace[GROUP] = state
    await ss.weighted_lease_snapshot(GROUP)
    assert "suspect_since" in (await _state())["leases"][lease_id]
    assert await ss.renew_weighted_lease(GROUP, lease_id)
    assert "suspect_since" not in (await _state())["leases"][lease_id]

    assert await ss.mark_weighted_lease_recovery(
        GROUP, lease_id, recovery_seconds=60, remote_task_id="mineru-task-7"
    )
    assert not await ss.renew_weighted_lease(GROUP, lease_id)
    state = await _state()
    state["leases"][lease_id]["recovery_until"] = time.time() - 1.0
    namespace[GROUP] = state

    snapshot = await ss.weighted_lease_snapshot(GROUP)
    assert snapshot["global_used_capacity"] == 0
    assert snapshot["active_leases"] == 0
    assert snapshot["recovered_count"] == 1
    assert not await ss.release_weighted_lease(GROUP, lease_id)


async def test_document_lease_lookup_matches_workspace_and_recovery_state():
    admission, _ = await ss.try_acquire_weighted_lease(
        GROUP,
        kb_key="kb_a",
        weight=1,
        global_capacity=4,
        per_kb_capacity=4,
        waiter_id="document-owner",
        metadata={"doc_id": "document-1"},
    )
    assert admission
    lease_id = admission["lease_id"]

    assert await ss.weighted_lease_exists_for_document(
        GROUP, kb_key="kb_a", doc_id="document-1"
    )
    assert not await ss.weighted_lease_exists_for_document(
        GROUP, kb_key="kb_b", doc_id="document-1"
    )
    assert not await ss.weighted_lease_exists_for_document(
        GROUP, kb_key="kb_a", doc_id="document-2"
    )

    assert await ss.mark_weighted_lease_recovery(GROUP, lease_id, recovery_seconds=60)
    assert await ss.weighted_lease_exists_for_document(
        GROUP, kb_key="kb_a", doc_id="document-1"
    )
    assert await ss.release_weighted_lease(GROUP, lease_id)
    assert not await ss.weighted_lease_exists_for_document(
        GROUP, kb_key="kb_a", doc_id="document-1"
    )
