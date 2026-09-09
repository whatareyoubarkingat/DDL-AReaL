# SPDX-License-Identifier: Apache-2.0
"""TMS transitions are idempotent and paused allocations are restored for free."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from areal.engine import fsdp_engine as module
from areal.engine.fsdp_engine import FSDPEngine


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setattr(module, "is_tms_enabled", lambda: True)
    monkeypatch.setattr(
        module, "torch_memory_saver", SimpleNamespace(pause=Mock(), resume=Mock())
    )
    monkeypatch.setattr(module.current_platform, "clear_memory", Mock())
    monkeypatch.setattr(module.current_platform, "synchronize", Mock())
    monkeypatch.setattr(module.current_platform, "empty_cache", Mock())
    monkeypatch.setattr(module.dist, "barrier", Mock())
    monkeypatch.setattr(module.dist, "is_initialized", lambda: False)
    engine = object.__new__(FSDPEngine)
    engine._initialized = True
    engine._cpu_group = object()
    engine._per_layer_optim_wrapper = None
    engine.own_global_group = False
    engine.is_offload = False
    engine.get_device_stats = Mock(return_value=SimpleNamespace(log=Mock()))
    return engine


def test_duplicate_offload_and_onload_only_transition_once(engine):
    """Initialization and RPC retries must not pause an already paused mapping."""
    engine.onload()
    module.torch_memory_saver.resume.assert_not_called()
    engine.offload()
    engine.offload()
    assert engine.is_offload
    module.torch_memory_saver.pause.assert_called_once_with()
    engine.onload()
    engine.onload()
    assert not engine.is_offload
    module.torch_memory_saver.resume.assert_called_once_with()


@pytest.mark.parametrize(
    "method,initial,target", [("offload", False, True), ("onload", True, False)]
)
def test_state_tracks_successful_transition_even_if_barrier_fails(
    engine, method, initial, target
):
    """Cleanup must know whether CUDA storage was actually unmapped or restored."""
    engine.is_offload = initial
    module.dist.barrier.side_effect = RuntimeError("barrier failed")
    with pytest.raises(RuntimeError, match="barrier failed"):
        getattr(engine, method)()
    assert engine.is_offload is target


@pytest.mark.parametrize(
    "method,initial,operation",
    [("offload", False, "pause"), ("onload", True, "resume")],
)
def test_transition_error_preserves_prior_state(engine, method, initial, operation):
    """A transition that did not complete must not report the target state."""
    engine.is_offload = initial
    getattr(module.torch_memory_saver, operation).side_effect = RuntimeError(
        "transition failed"
    )
    with pytest.raises(RuntimeError, match="transition failed"):
        getattr(engine, method)()
    assert engine.is_offload is initial


def test_destroy_restores_paused_storage_before_freeing_tensors(engine):
    """A paused worker can close without TMS aborting in its CUDA free hook."""
    engine.model = object()
    engine.optimizer = object()
    engine.is_offload = True

    def resume():
        assert engine.initialized
        assert hasattr(engine, "model") and hasattr(engine, "optimizer")

    module.torch_memory_saver.resume.side_effect = resume
    engine.destroy()
    engine.destroy()
    module.torch_memory_saver.resume.assert_called_once_with()
    assert not engine.is_offload and not engine.initialized
    assert not hasattr(engine, "model") and not hasattr(engine, "optimizer")


def test_destroy_does_not_free_paused_storage_when_restore_fails(engine):
    """Propagate the error to scheduler cleanup instead of freeing unmapped data."""
    engine.model = object()
    engine.optimizer = object()
    engine.is_offload = True
    module.torch_memory_saver.resume.side_effect = RuntimeError("restore failed")
    with pytest.raises(RuntimeError, match="restore failed"):
        engine.destroy()
    assert engine.is_offload and engine.initialized
    assert hasattr(engine, "model") and hasattr(engine, "optimizer")
