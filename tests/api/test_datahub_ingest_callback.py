from __future__ import annotations

from dataclasses import replace

import httpx
import pytest

from lightrag.api import datahub_ingest_callback as callback


pytestmark = pytest.mark.offline


def _notification(**changes) -> callback.TerminalNotification:
    notification = callback.TerminalNotification(
        job_id="101",
        doc_id="201",
        kb_id="301",
        workspace="kb_301",
        track_id="track-1",
        status="processed",
        updated_at="2026-09-09T10:00:00+00:00",
        chunks_count=7,
        error_msg=None,
    )
    return replace(notification, **changes)


class _Response:
    def __init__(self, status_code: int, body=None):
        self.status_code = status_code
        self._body = body or {}

    def json(self):
        return self._body


class _Client:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    async def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture(autouse=True)
def callback_environment(monkeypatch):
    monkeypatch.setenv("AI_SERVICE_URL", "http://ai-service:8085")
    monkeypatch.setenv("AI_SERVICE_INTERNAL_PREFIX", "/ai/internal")
    monkeypatch.setenv("AI_SERVICE_INTERNAL_TOKEN", "service-token")


async def test_callback_retries_not_ready_throttle_and_server_error(monkeypatch):
    dispatcher = callback.DataHubIngestCallbackDispatcher(
        enabled=True,
        ai_service_url="http://ai-service:8085",
        max_attempts=5,
    )
    client = _Client(
        [
            _Response(409, {"message": "RAG_JOB_NOT_READY"}),
            _Response(429),
            _Response(503),
            _Response(200),
        ]
    )
    sleeps = []

    async def no_wait(delay):
        sleeps.append(delay)

    monkeypatch.setattr(callback.asyncio, "sleep", no_wait)

    await dispatcher._deliver(client, _notification())

    assert len(client.calls) == 4
    assert sleeps == [0.5, 2.0, 5.0]
    assert client.calls[0][1]["headers"] == {
        "X-Internal-Service-Token": "service-token"
    }
    assert client.calls[0][1]["json"]["doc_id"] == "201"


async def test_callback_retries_network_error_but_not_validation_rejection(monkeypatch):
    request = httpx.Request("POST", "http://ai-service:8085/callback")
    dispatcher = callback.DataHubIngestCallbackDispatcher(
        enabled=True,
        ai_service_url="http://ai-service:8085",
        max_attempts=3,
    )
    network_client = _Client(
        [httpx.ConnectError("offline", request=request), _Response(200)]
    )

    async def no_wait(_delay):
        return None

    monkeypatch.setattr(callback.asyncio, "sleep", no_wait)
    await dispatcher._deliver(network_client, _notification())
    assert len(network_client.calls) == 2

    rejected_client = _Client([_Response(422), _Response(200)])
    await dispatcher._deliver(rejected_client, _notification())
    assert len(rejected_client.calls) == 1


async def test_callback_enqueue_is_non_blocking_deduplicated_and_bounded():
    dispatcher = callback.DataHubIngestCallbackDispatcher(
        enabled=True,
        ai_service_url="http://ai-service:8085",
        queue_size=1,
    )
    dispatcher._accepting = True

    first = await dispatcher.enqueue_terminal(_notification())
    duplicate = await dispatcher.enqueue_terminal(_notification())
    full = await dispatcher.enqueue_terminal(
        _notification(job_id="102", doc_id="202", track_id="track-2")
    )

    assert first is True
    assert duplicate is True
    assert full is False
    assert dispatcher._queue.qsize() == 1


async def test_disabled_callback_does_not_enqueue():
    dispatcher = callback.DataHubIngestCallbackDispatcher(
        enabled=False,
        ai_service_url="",
    )

    assert await dispatcher.enqueue_terminal(_notification()) is False
