# SPDX-License-Identifier: Apache-2.0

"""Opt-in, CPU-only execution diagnostics for stalled training and RPCs.

Set ``AREAL_EXECUTION_DIAGNOSTICS_DIR`` to write one JSONL file per host/PID.
Durations measure host execution; they do not synchronize accelerator streams.
``AREAL_EXECUTION_DIAGNOSTICS_STALL_SECONDS`` controls periodic thread snapshots
(default: 60 seconds). No tensor values, exception messages, locals or source
lines are recorded. Diagnostic failures never replace a training exception.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import socket
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_metadata: ContextVar[dict[str, Any]] = ContextVar("execution_diagnostics", default={})
_active: dict[int, dict[str, Any]] = {}
_active_lock = threading.Lock()
_scope_ids = itertools.count()
_FORBIDDEN_KEYS = {
    "args",
    "kwargs",
    "prompt",
    "prompts",
    "messages",
    "input_ids",
    "password",
    "secret",
    "token",
    "api_key",
    "authorization",
    "tensor",
}


def enabled() -> bool:
    """Return whether this process has an explicitly configured output directory."""
    return bool(os.environ.get("AREAL_EXECUTION_DIAGNOSTICS_DIR"))


def _safe_value(value: Any) -> Any:
    # Exact types intentionally avoid invoking tensor conversion/repr hooks.
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) is str:
        return value[:512]
    if type(value) in (list, tuple) and len(value) <= 64:
        if all(type(item) in (bool, int, float, str, type(None)) for item in value):
            return [_safe_value(item) for item in value]
    return None


def _safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _safe_value(value)
        for key, value in metadata.items()
        if type(key) is str and key.lower() not in _FORBIDDEN_KEYS
    }


def _write(record: dict[str, Any]) -> None:
    directory = os.environ.get("AREAL_EXECUTION_DIAGNOSTICS_DIR")
    if not directory:
        return
    try:
        os.makedirs(directory, mode=0o700, exist_ok=True)
        host = socket.gethostname().replace("/", "_")
        path = os.path.join(directory, f"execution-{host}-{os.getpid()}.jsonl")
        encoded = (
            json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()
        # One append syscall per record; no registry lock is held during I/O.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, encoded)
        finally:
            os.close(fd)
    except Exception:
        # Logging must not introduce another error path into training/RPCs.
        return


def event(event: str, **metadata: Any) -> None:
    """Append an event containing only primitive metadata and current context."""
    if not enabled():
        return
    try:
        record = {**_metadata.get(), **_safe_metadata(metadata)}
        record.update(
            event=event,
            ts_unix_ns=time.time_ns(),
            monotonic_s=time.monotonic(),
            pid=os.getpid(),
            thread_id=threading.get_ident(),
        )
        _write(record)
    except Exception:
        return


@contextmanager
def context(**metadata: Any) -> Iterator[None]:
    """Attach primitive rank, microbatch and request tags to nested events."""
    if not enabled():
        yield
        return
    token = _metadata.set({**_metadata.get(), **_safe_metadata(metadata)})
    try:
        yield
    finally:
        _metadata.reset(token)


@contextmanager
def scope(phase: str, **metadata: Any) -> Iterator[None]:
    """Record paired phase events and host duration, preserving exceptions."""
    if not enabled():
        yield
        return
    with context(**metadata):
        scope_id = next(_scope_ids)
        started = time.monotonic()
        snapshot = {
            **_metadata.get(),
            "scope_id": scope_id,
            "phase": phase,
            "started_monotonic_s": started,
            "thread_id": threading.get_ident(),
        }
        with _active_lock:
            _active[scope_id] = snapshot
        event("phase_start", phase=phase, scope_id=scope_id)
        error_type = None
        try:
            yield
        except BaseException as exc:
            error_type = type(exc).__name__
            raise
        finally:
            with _active_lock:
                _active.pop(scope_id, None)
            event(
                "phase_end",
                phase=phase,
                scope_id=scope_id,
                elapsed_s=time.monotonic() - started,
                error_type=error_type,
            )


def snapshot_threads(reason: str, **metadata: Any) -> None:
    """Record bounded Python stacks without reading locals or source text."""
    if not enabled():
        return
    try:
        with _active_lock:
            active = list(_active.values())
        stacks = []
        for thread_id, frame in list(sys._current_frames().items())[:128]:
            frames = []
            while frame is not None and len(frames) < 80:
                frames.append(
                    {
                        "filename": frame.f_code.co_filename,
                        "function": frame.f_code.co_name,
                        "lineno": frame.f_lineno,
                    }
                )
                frame = frame.f_back
            stacks.append({"thread_id": thread_id, "frames": frames})
        record = {**_metadata.get(), **_safe_metadata(metadata)}
        record.update(
            event="thread_snapshot",
            reason=reason,
            ts_unix_ns=time.time_ns(),
            monotonic_s=time.monotonic(),
            pid=os.getpid(),
            thread_id=threading.get_ident(),
            active_phases=active,
            threads=stacks,
        )
        _write(record)
    except Exception:
        return


@contextmanager
def watchdog(timeout_s: float | None = None, **metadata: Any) -> Iterator[None]:
    """Dump stacks independently of the engine queue; never cancel work.

    ``timeout_s`` is the original client deadline duration, measured from server
    receipt, not an execution timeout. Snapshots continue after it until the
    request actually exits. A native extension holding the GIL can prevent this
    Python watchdog from running.
    """
    if not enabled():
        yield
        return
    try:
        interval = float(
            os.environ.get("AREAL_EXECUTION_DIAGNOSTICS_STALL_SECONDS", "60")
        )
        if not math.isfinite(interval) or interval <= 0:
            interval = 60.0
    except (ValueError, TypeError):
        interval = 60.0
    if (
        type(timeout_s) not in (int, float)
        or not math.isfinite(timeout_s)
        or timeout_s <= 0
    ):
        timeout_s = None
    tags = {**_metadata.get(), **_safe_metadata(metadata)}
    finished = threading.Event()

    def watch() -> None:
        started = time.monotonic()
        next_stall = started + interval
        deadline = started + timeout_s if timeout_s is not None else math.inf
        while not finished.wait(max(0.0, min(next_stall, deadline) - time.monotonic())):
            now = time.monotonic()
            reason = "http_deadline" if now >= deadline else "stall_interval"
            snapshot_threads(reason, **{**tags, "elapsed_s": now - started})
            if now >= deadline:
                deadline = math.inf
            if now >= next_stall:
                next_stall = now + interval

    thread = threading.Thread(
        target=watch, name="ExecutionDiagnosticsWatchdog", daemon=True
    )
    try:
        thread.start()
    except Exception:
        # Continue the original operation if diagnostic thread creation fails.
        event("watchdog_unavailable")
    try:
        yield
    finally:
        finished.set()
