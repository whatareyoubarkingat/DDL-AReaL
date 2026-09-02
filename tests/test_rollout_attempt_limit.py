# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from unittest.mock import Mock

import pytest

from areal.api.cli_args import PPOConfig
from areal.infra.workflow_executor import BatchTaskDispatcher


@dataclass
class _TaskInput:
    task_id: int


class _StalenessManager:
    def get_pending_limit(self) -> int:
        return 100


def _input_generator():
    task_id = 0
    while True:
        yield _TaskInput(task_id=task_id)
        task_id += 1


def _dispatcher(monkeypatch, result_batches):
    dispatcher = BatchTaskDispatcher(
        max_queue_size=16,
        task_factory=lambda _: None,
        staleness_manager=_StalenessManager(),
    )
    monkeypatch.setattr(dispatcher.runner, "get_input_queue_size", lambda: 0)
    submitted = []
    monkeypatch.setattr(dispatcher, "submit_task_input", submitted.append)
    waits = []
    batches = iter(result_batches)

    def wait_results(count, timeout):
        waits.append(count)
        batch = next(batches)
        assert not batch or len(batch) == count
        return batch

    monkeypatch.setattr(dispatcher, "wait_results", wait_results)
    return dispatcher, submitted, waits


def test_active_submit_and_wait_attempt_limit_raises_with_counts(monkeypatch):
    """Fixed-size collection stops after the configured number of attempts."""
    dispatcher, submitted, waits = _dispatcher(
        monkeypatch,
        result_batches=[[None] * 4, [None] * 4, [None] * 4],
    )

    with pytest.raises(
        RuntimeError,
        match=r"max_attempts_per_batch=12: accepted=0, rejected=12, batch_size=4",
    ):
        dispatcher.active_submit_and_wait(
            _input_generator(),
            batch_size=4,
            max_attempts_per_batch=12,
        )

    assert len(submitted) == 12
    assert waits == [4, 4, 4]


def test_active_submit_and_wait_default_keeps_retrying(monkeypatch):
    """The default None limit preserves the existing unlimited retry behavior."""
    accepted = [Mock(name=f"result-{index}") for index in range(4)]
    dispatcher, submitted, waits = _dispatcher(
        monkeypatch,
        result_batches=[[None] * 4, accepted],
    )

    results = dispatcher.active_submit_and_wait(
        _input_generator(),
        batch_size=4,
    )

    assert results == accepted
    assert len(submitted) == 8
    assert waits == [4, 4]


def test_active_submit_and_wait_does_not_oversubmit_during_timeouts(monkeypatch):
    """Slow results do not enqueue more than the configured attempt budget."""
    dispatcher, submitted, waits = _dispatcher(
        monkeypatch,
        result_batches=[[], [], [], [None] * 4, [None] * 4, [None] * 4],
    )

    with pytest.raises(RuntimeError, match="rejected=12"):
        dispatcher.active_submit_and_wait(
            _input_generator(),
            batch_size=4,
            max_attempts_per_batch=12,
        )

    assert len(submitted) == 12
    assert waits == [4, 4, 4, 4, 4, 4]


def test_ppo_config_rejects_nonpositive_attempt_limit():
    """PPO config rejects limits that cannot admit any rollout."""
    with pytest.raises(ValueError, match="max_attempts_per_batch must be positive"):
        PPOConfig(
            experiment_name="test",
            trial_name="attempt_limit",
            max_attempts_per_batch=0,
        )


def test_fixed_batch_rejects_attempt_limit_smaller_than_batch(monkeypatch):
    """A fixed-size batch cannot use an attempt budget smaller than its size."""
    dispatcher, _, _ = _dispatcher(monkeypatch, result_batches=[])

    with pytest.raises(ValueError, match="must be at least batch_size"):
        dispatcher.active_submit_and_wait(
            _input_generator(),
            batch_size=4,
            max_attempts_per_batch=3,
        )
