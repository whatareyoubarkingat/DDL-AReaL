# SPDX-License-Identifier: Apache-2.0
"""CPU RPC diagnostics tests using real queue dispatch and mocked transports."""

import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("torch")
pytest.importorskip("colorlog")

import requests
from flask import Flask

from areal.infra.rpc.guard import engine_blueprint as engine_bp
from areal.infra.rpc.guard.app import GuardState
from areal.infra.scheduler import local
from areal.infra.scheduler.exceptions import EngineCallError
from areal.infra.utils.launcher import get_env_vars
from areal.utils import execution_diagnostics as diagnostics


def read_records(directory):
    return [
        json.loads(line)
        for path in directory.glob("execution-*.jsonl")
        for line in path.read_text().splitlines()
    ]


@pytest.fixture
def diagnostic_output(monkeypatch, tmp_path):
    monkeypatch.setenv("AREAL_EXECUTION_DIAGNOSTICS_DIR", str(tmp_path))
    monkeypatch.setenv("AREAL_EXECUTION_DIAGNOSTICS_STALL_SECONDS", "0.03")
    return tmp_path


@pytest.fixture
def scheduler(monkeypatch):
    # Avoid GPU detection, worker launch and destructor process management.
    monkeypatch.setattr(
        local.LocalScheduler, "delete_workers", lambda *args, **kwargs: None
    )
    instance = object.__new__(local.LocalScheduler)
    instance._workers = {
        "actor": [
            SimpleNamespace(
                worker=SimpleNamespace(
                    id="actor/0", ip="127.0.0.1", worker_ports=["12345"]
                ),
                process=None,
            )
        ]
    }
    return instance


def test_sync_retry_preserves_request_identity_and_rpc_controls(
    scheduler, monkeypatch, diagnostic_output
):
    """Retries retain identity while caller metadata and HTTP options stay intact."""
    retry = Mock(status_code=503)
    success = Mock(status_code=200)
    success.json.return_value = {"result": 42}
    post = Mock(side_effect=[retry, success])
    monkeypatch.setattr(local.requests, "post", post)
    delay = Mock()
    monkeypatch.setattr(local.time, "sleep", delay)
    meta = {"broadcast": False}

    assert (
        scheduler.call_engine("actor/0", "train", rpc_meta=meta, http_timeout=9) == 42
    )

    payloads = [call.kwargs["json"] for call in post.call_args_list]
    tags = [payload["rpc_meta"]["execution_diagnostics"] for payload in payloads]
    assert tags[0]["request_id"] == tags[1]["request_id"]
    assert [tag["attempt"] for tag in tags] == [1, 2]
    assert all(payload["rpc_meta"]["broadcast"] is False for payload in payloads)
    assert all(call.kwargs["timeout"] == 9 for call in post.call_args_list)
    assert meta == {"broadcast": False}
    delay.assert_called_once_with(1.0)


def test_async_retry_preserves_request_identity_and_timeout(
    scheduler, monkeypatch, diagnostic_output
):
    """The asynchronous path uses the same trace identity across retries."""
    responses = []
    for status in (503, 200):
        response = AsyncMock()
        response.status = status
        response.json.return_value = {"result": 42}
        response.__aenter__.return_value = response
        responses.append(response)
    session = AsyncMock()
    session.__aenter__.return_value = session
    session.post = Mock(side_effect=responses)
    monkeypatch.setattr(local.aiohttp, "ClientSession", Mock(return_value=session))
    delay = AsyncMock()
    monkeypatch.setattr(local.asyncio, "sleep", delay)
    meta = {"broadcast": False}

    result = asyncio.run(
        scheduler.async_call_engine("actor/0", "train", rpc_meta=meta, http_timeout=9)
    )

    assert result == 42
    payloads = [call.kwargs["json"] for call in session.post.call_args_list]
    tags = [payload["rpc_meta"]["execution_diagnostics"] for payload in payloads]
    assert tags[0]["request_id"] == tags[1]["request_id"]
    assert [tag["attempt"] for tag in tags] == [1, 2]
    assert all(
        call.kwargs["timeout"].total == 9 for call in session.post.call_args_list
    )
    assert meta == {"broadcast": False}
    delay.assert_awaited_once_with(1.0)


def test_timeout_logs_identity_without_changing_retries(
    scheduler, monkeypatch, diagnostic_output
):
    """A diagnostic timeout event retains request identity but no exception text."""
    post = Mock(side_effect=requests.exceptions.Timeout("private exception contents"))
    monkeypatch.setattr(local.requests, "post", post)
    monkeypatch.setattr(local.time, "sleep", Mock())
    with pytest.raises(EngineCallError):
        scheduler.call_engine("actor/0", "train", max_retries=2, http_timeout=0.01)
    records = read_records(diagnostic_output)
    timeouts = [record for record in records if record["event"] == "rpc_client_timeout"]
    assert [record["attempt"] for record in timeouts] == [1, 2]
    assert timeouts[0]["request_id"] == timeouts[1]["request_id"]
    assert post.call_count == 2
    assert "private exception contents" not in json.dumps(records)


def test_disabled_scheduler_does_not_add_rpc_metadata(scheduler, monkeypatch):
    """Disabling diagnostics preserves the original transport payload."""
    monkeypatch.delenv("AREAL_EXECUTION_DIAGNOSTICS_DIR", raising=False)
    response = Mock(status_code=200)
    response.json.return_value = {"result": 42}
    post = Mock(return_value=response)
    monkeypatch.setattr(local.requests, "post", post)
    assert scheduler.call_engine("actor/0", "train") == 42
    assert post.call_args.kwargs["json"]["rpc_meta"] is None


def test_env_propagation_keeps_explicit_override(monkeypatch, diagnostic_output):
    """Only named diagnostic environment variables propagate to workers."""
    monkeypatch.setenv("AREAL_EXECUTION_DIAGNOSTICS_SECRET", "private")
    env = get_env_vars()
    assert env["AREAL_EXECUTION_DIAGNOSTICS_DIR"] == str(diagnostic_output)
    assert env["AREAL_EXECUTION_DIAGNOSTICS_STALL_SECONDS"] == "0.03"
    assert "AREAL_EXECUTION_DIAGNOSTICS_SECRET" not in env
    assert (
        get_env_vars("AREAL_EXECUTION_DIAGNOSTICS_STALL_SECONDS=7")[
            "AREAL_EXECUTION_DIAGNOSTICS_STALL_SECONDS"
        ]
        == "7"
    )


def test_blocked_engine_watchdog_observes_queued_retry_and_propagates_context(
    monkeypatch, diagnostic_output
):
    """Stacks remain observable while attempts execute serially on EngineWorker."""
    started = threading.Event()
    release = threading.Event()
    queued_retry = threading.Event()
    deadline_seen = threading.Event()
    calls = []
    responses = []
    real_event = diagnostics.event
    real_snapshot = diagnostics.snapshot_threads

    def event(name, **metadata):
        real_event(name, **metadata)
        if name == "rpc_queued" and len(calls) == 1:
            queued_retry.set()

    def snapshot(reason, **metadata):
        real_snapshot(reason, **metadata)
        if reason == "http_deadline":
            deadline_seen.set()

    def train():
        calls.append(threading.get_ident())
        with diagnostics.scope("fake.backward", rank=7, microbatch=173):
            started.set()
            assert release.wait(3)
        return 42

    monkeypatch.setattr(diagnostics, "event", event)
    monkeypatch.setattr(diagnostics, "snapshot_threads", snapshot)
    monkeypatch.setattr(
        engine_bp, "_engines", {"actor/0": SimpleNamespace(train=train)}
    )
    monkeypatch.setattr(engine_bp, "_engine_thread", None)
    monkeypatch.setattr(engine_bp, "_engine_work_queue", None)
    monkeypatch.setattr(
        engine_bp.RTensor, "localize", staticmethod(lambda value: value)
    )
    monkeypatch.setattr(
        engine_bp.RTensor, "remotize", staticmethod(lambda value, **kwargs: value)
    )
    app = Flask(__name__)
    app.config["guard_state"] = GuardState()
    app.register_blueprint(engine_bp.engine_bp)

    def request(attempt):
        with app.test_client() as client:
            responses.append(
                client.post(
                    "/call",
                    json={
                        "engine_name": "actor/0",
                        "method": "train",
                        "rpc_meta": {
                            "broadcast": False,
                            "execution_diagnostics": {
                                "request_id": "same-logical-request",
                                "attempt": attempt,
                                "http_timeout_s": 0.08,
                            },
                        },
                    },
                )
            )

    first = threading.Thread(target=request, args=(1,))
    second = threading.Thread(target=request, args=(2,))
    first.start()
    try:
        assert started.wait(2)
        second.start()
        assert queued_retry.wait(2)
        assert deadline_seen.wait(2)
        assert len(calls) == 1, "retry executed concurrently with the blocked method"
        records = read_records(diagnostic_output)
        snapshots = [
            record for record in records if record["event"] == "thread_snapshot"
        ]
        assert any(
            phase.get("phase") == "fake.backward"
            and phase.get("request_id") == "same-logical-request"
            for record in snapshots
            for phase in record["active_phases"]
        )
        assert not any(
            record["event"] == "rpc_running" and record["attempt"] == 2
            for record in records
        )
    finally:
        release.set()
        first.join(3)
        if second.ident is not None:
            second.join(3)
        queue = engine_bp._engine_work_queue
        worker = engine_bp._engine_thread
        if queue is not None:
            queue.put(None)
        if worker is not None:
            worker.join(3)

    assert [response.status_code for response in responses] == [200, 200]
    assert len(calls) == 2 and len(set(calls)) == 1
    records = read_records(diagnostic_output)
    assert [
        record["attempt"] for record in records if record["event"] == "rpc_running"
    ] == [1, 2]
    assert [
        record["attempt"] for record in records if record["event"] == "rpc_done"
    ] == [1, 2]
    assert not diagnostics._active
