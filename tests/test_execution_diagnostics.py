# SPDX-License-Identifier: Apache-2.0
"""CPU-only diagnostics tests; load the stdlib helper without optional ML deps."""

import importlib.util
import json
import threading
import time
from pathlib import Path

import pytest

_PATH = Path(__file__).parents[1] / "areal/utils/execution_diagnostics.py"
_SPEC = importlib.util.spec_from_file_location(
    "execution_diagnostics_test_module", _PATH
)
diagnostics = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(diagnostics)


def read_records(directory):
    return [
        json.loads(line)
        for path in directory.glob("execution-*.jsonl")
        for line in path.read_text().splitlines()
    ]


def test_disabled_diagnostics_does_not_write_or_start_watchdog(monkeypatch, tmp_path):
    """Disabled instrumentation leaves no files, active scopes or helper thread."""
    monkeypatch.delenv("AREAL_EXECUTION_DIAGNOSTICS_DIR", raising=False)
    monkeypatch.setattr(
        threading.Thread, "start", lambda self: pytest.fail("thread started")
    )
    with (
        diagnostics.context(rank=2),
        diagnostics.scope("forward"),
        diagnostics.watchdog(),
    ):
        diagnostics.event("progress")
    assert not read_records(tmp_path)
    assert not diagnostics._active


def test_nested_scopes_preserve_context_and_original_exception(monkeypatch, tmp_path):
    """Errors retain their original identity and only safe metadata is emitted."""
    monkeypatch.setenv("AREAL_EXECUTION_DIAGNOSTICS_DIR", str(tmp_path))

    class TensorLike:
        def __repr__(self):
            pytest.fail("tensor repr evaluated")

        def item(self):
            pytest.fail("tensor item evaluated")

    failure = RuntimeError("secret exception message")
    with pytest.raises(RuntimeError) as caught:
        with diagnostics.context(rank=6, request_id="request", prompt="secret prompt"):
            with diagnostics.scope("backward", microbatch=173, value=TensorLike()):
                raise failure
    diagnostics.event("outside")
    records = read_records(tmp_path)
    assert caught.value is failure
    assert records[0]["rank"] == 6
    assert records[0]["microbatch"] == 173
    assert records[0]["value"] is None
    assert records[1]["error_type"] == "RuntimeError"
    assert records[1]["elapsed_s"] >= 0
    assert "rank" not in records[-1]
    assert "secret" not in json.dumps(records)
    assert not diagnostics._active


def test_watchdog_dumps_blocked_phase_at_interval_and_deadline(monkeypatch, tmp_path):
    """The independent watchdog sees a blocked phase and never reads locals."""
    monkeypatch.setenv("AREAL_EXECUTION_DIAGNOSTICS_DIR", str(tmp_path))
    monkeypatch.setenv("AREAL_EXECUTION_DIAGNOSTICS_STALL_SECONDS", "0.02")
    dumped = threading.Event()
    real_snapshot = diagnostics.snapshot_threads

    def snapshot(reason, **metadata):
        real_snapshot(reason, **metadata)
        if reason == "http_deadline":
            dumped.set()

    monkeypatch.setattr(diagnostics, "snapshot_threads", snapshot)
    with diagnostics.context(request_id="blocked", rank=7):
        with diagnostics.scope("backward"), diagnostics.watchdog(timeout_s=0.07):
            assert dumped.wait(2), "watchdog was blocked behind the engine work"
    time.sleep(0.05)
    count = len(read_records(tmp_path))
    time.sleep(0.04)
    records = read_records(tmp_path)
    assert len(records) == count, "watchdog continued after request completion"
    snapshots = [record for record in records if record["event"] == "thread_snapshot"]
    assert {record["reason"] for record in snapshots} >= {
        "stall_interval",
        "http_deadline",
    }
    assert all(record["request_id"] == "blocked" for record in snapshots)
    assert any(phase["phase"] == "backward" for phase in snapshots[0]["active_phases"])
    assert all(
        set(frame) == {"filename", "function", "lineno"}
        for record in snapshots
        for thread in record["threads"]
        for frame in thread["frames"]
    )


def test_output_failure_does_not_change_execution(monkeypatch, tmp_path):
    """An unwritable diagnostics path cannot replace a training error."""
    bad_directory = tmp_path / "file"
    bad_directory.write_text("not a directory")
    monkeypatch.setenv("AREAL_EXECUTION_DIAGNOSTICS_DIR", str(bad_directory))
    with pytest.raises(ValueError, match="original"):
        with diagnostics.scope("optimizer"):
            raise ValueError("original")
    assert not diagnostics._active


def test_thread_snapshot_does_not_wait_for_active_phase(monkeypatch, tmp_path):
    """Snapshotting is independent from a worker holding an active scope."""
    monkeypatch.setenv("AREAL_EXECUTION_DIAGNOSTICS_DIR", str(tmp_path))
    started = threading.Event()
    release = threading.Event()

    def worker():
        with diagnostics.context(rank=3), diagnostics.scope("optimizer"):
            started.set()
            release.wait(2)

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        assert started.wait(2)
        diagnostics.snapshot_threads("manual")
        assert thread.is_alive()
        snapshot = read_records(tmp_path)[-1]
        assert any(phase["rank"] == 3 for phase in snapshot["active_phases"])
    finally:
        release.set()
        thread.join(2)
