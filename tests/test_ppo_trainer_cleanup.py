# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression tests for partial PPO trainer cleanup."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from areal.trainer import rl_trainer
from areal.trainer.rl_trainer import PPOTrainer


@pytest.fixture
def tracer(monkeypatch):
    """Avoid writing profiler output in lifecycle tests."""
    save = Mock()
    monkeypatch.setattr(rl_trainer.perf_tracer, "save", save)
    return save


def test_close_empty_trainer_is_safe_and_idempotent(tracer):
    """A failure before resource assignment still permits cleanup."""
    trainer = object.__new__(PPOTrainer)
    trainer.close()
    trainer.close()
    tracer.assert_called_once_with(force=True)


def test_close_partial_trainer_cleans_owned_scheduler(tracer):
    """Scheduler-owned workers are cleaned even before actor is assigned."""
    trainer = object.__new__(PPOTrainer)
    trainer.scheduler = SimpleNamespace(delete_workers=Mock())
    trainer.close()
    trainer.close()
    trainer.scheduler.delete_workers.assert_called_once_with(reverse_order=True)


def test_close_continues_after_error_and_retries_only_failed_resource(tracer):
    """A saver failure must not strand workers or become a successful close."""
    trainer = object.__new__(PPOTrainer)
    failure = RuntimeError("checkpoint finalize failed")
    trainer.saver = SimpleNamespace(finalize=Mock(side_effect=[failure, None]))
    trainer.actor = SimpleNamespace(destroy=Mock())
    trainer.scheduler = SimpleNamespace(delete_workers=Mock())
    with pytest.raises(RuntimeError) as caught:
        trainer.close()
    assert caught.value is failure
    trainer.actor.destroy.assert_called_once_with()
    trainer.scheduler.delete_workers.assert_called_once_with(reverse_order=True)
    trainer.close()
    assert trainer.saver.finalize.call_count == 2
    trainer.actor.destroy.assert_called_once_with()
    tracer.assert_called_once_with(force=True)


def test_close_releases_full_trainer_in_order(tracer):
    """Complete trainers retain deterministic shutdown ordering."""
    trainer = object.__new__(PPOTrainer)
    calls = []
    names = (
        ("saver", "finalize"),
        ("_train_rdataset", "close"),
        ("_valid_rdataset", "close"),
        ("data_controller", "destroy"),
        ("stats_logger", "close"),
        ("eval_rollout", "destroy"),
        ("rollout", "destroy"),
        ("teacher", "destroy"),
        ("ref", "destroy"),
        ("critic", "destroy"),
        ("actor", "destroy"),
    )
    for name, method in names:
        resource = SimpleNamespace(**{method: lambda name=name: calls.append(name)})
        setattr(trainer, name, resource)
    trainer.scheduler = SimpleNamespace(
        delete_workers=lambda **kwargs: calls.append("scheduler")
    )
    trainer.close()
    assert calls == [name for name, _ in names] + ["scheduler"]


def test_close_missing_cleanup_method_still_releases_workers(tracer):
    """Even method lookup failures cannot bypass the scheduler fallback."""
    trainer = object.__new__(PPOTrainer)
    trainer.saver = object()
    trainer.scheduler = SimpleNamespace(delete_workers=Mock())
    with pytest.raises(AttributeError):
        trainer.close()
    trainer.scheduler.delete_workers.assert_called_once_with(reverse_order=True)


def test_constructor_preserves_original_error_when_cleanup_fails(monkeypatch, tracer):
    """The initial configuration error must remain the reported exception."""
    failure = ValueError("worker configuration failed")
    scheduler = SimpleNamespace(delete_workers=Mock())

    def fail_init(self, *args):
        self.scheduler = scheduler
        self.saver = SimpleNamespace(finalize=Mock(side_effect=RuntimeError("cleanup")))
        raise failure

    monkeypatch.setattr(PPOTrainer, "_init_impl", fail_init)
    with pytest.raises(ValueError) as caught:
        PPOTrainer(None)
    assert caught.value is failure
    scheduler.delete_workers.assert_called_once_with(reverse_order=True)


def test_context_manager_preserves_training_error(tracer):
    """Cleanup failures do not mask a failed training step."""
    trainer = object.__new__(PPOTrainer)
    trainer.saver = SimpleNamespace(finalize=Mock(side_effect=RuntimeError("cleanup")))
    trainer.scheduler = SimpleNamespace(delete_workers=Mock())
    failure = ValueError("training failed")
    with pytest.raises(ValueError) as caught:
        with trainer:
            raise failure
    assert caught.value is failure
    trainer.scheduler.delete_workers.assert_called_once_with(reverse_order=True)


def test_successful_context_manager_reports_cleanup_failure(tracer):
    """A failed final checkpoint must not turn into a successful training exit."""
    trainer = object.__new__(PPOTrainer)
    trainer.saver = SimpleNamespace(
        finalize=Mock(side_effect=RuntimeError("save failed"))
    )
    with pytest.raises(RuntimeError, match="save failed"):
        with trainer:
            pass
