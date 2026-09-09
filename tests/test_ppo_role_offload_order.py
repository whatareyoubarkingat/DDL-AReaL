# SPDX-License-Identifier: Apache-2.0
"""CPU execution of the real trainer loop, with observable role boundaries."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from areal.trainer import rl_trainer
from areal.trainer.rl_trainer import PPOTrainer


@pytest.mark.parametrize("actor_offload", [False, True])
@pytest.mark.parametrize("critic_offload", [False, True])
@pytest.mark.parametrize("ref_offload", [False, True])
def test_ppo_role_handoff_precedes_next_compute_and_weight_sync(
    monkeypatch, actor_offload, critic_offload, ref_offload
):
    """No role is computed while offloaded, including actor weight version one."""
    events = []
    flags = dict(actor=actor_offload, critic=critic_offload, ref=ref_offload)
    resident = {role: not flag for role, flag in flags.items()}

    def operation(role, name, result=None):
        def call(*args, **kwargs):
            assert resident[role], f"{role}.{name} while offloaded"
            if all(flags.values()):
                assert sum(resident.values()) == 1
            events.append(f"{role}.{name}")
            return result

        return call

    def onload(role):
        assert not resident[role]
        resident[role] = True
        events.append(f"{role}.onload")

    def offload(role):
        assert resident[role]
        resident[role] = False
        events.append(f"{role}.offload")

    batch = [{}]
    engines = {}
    for role in flags:
        engines[role] = SimpleNamespace(
            onload=lambda role=role: onload(role),
            offload=lambda role=role: offload(role),
            get_device_stats=Mock(return_value=SimpleNamespace(log=Mock())),
            compute_logp=operation(role, "logp", [object()]),
            ppo_update=operation(role, "update"),
            step_lr_scheduler=operation(role, "lr"),
            set_version=Mock(),
            clear_batches=Mock(),
        )
    engines["critic"].compute_values = operation("critic", "values", [object()])
    engines["actor"].prepare_batch = Mock(return_value=batch)
    engines["actor"].compute_advantages = operation("actor", "advantages", batch)
    engines["actor"].update_weights = operation("actor", "weights")

    trainer = object.__new__(PPOTrainer)
    for role, engine in engines.items():
        setattr(trainer, role, engine)
    trainer.config = SimpleNamespace(
        total_train_epochs=1,
        total_train_steps=None,
        actor=SimpleNamespace(should_compute_prox_logp=lambda: True),
        gconfig=SimpleNamespace(
            n_samples=1, reward_normalization=None, drop_incomplete_group=False
        ),
        dynamic_bs=False,
        max_attempts_per_batch=12,
        memory_profiler=None,
    )
    trainer.recover_info = None
    trainer.train_dataloader = [None]
    trainer._should_offload_rollout = False
    trainer._should_offload_actor = actor_offload
    trainer._should_offload_critic = critic_offload
    trainer._should_offload_ref = ref_offload
    trainer._should_offload_teacher = False
    trainer.teacher = trainer.eval_rollout = trainer.data_controller = None
    trainer.saver = SimpleNamespace(maybe_wait_for_staging=Mock())
    trainer.rollout = SimpleNamespace(pause=Mock(), resume=Mock(), set_version=Mock())
    trainer.weight_update_meta = SimpleNamespace(with_version=lambda version: version)
    trainer._requires_proxy_workflow = lambda workflow: False
    trainer._is_v1_awex_colocate = lambda config: False
    trainer._save_training_state = Mock()
    trainer._evaluate = Mock()
    trainer._export_and_commit_stats = Mock()
    trainer._save_perf_tracer = Mock()
    monkeypatch.setattr(
        rl_trainer.stats_tracker, "record_timing", lambda *a, **k: nullcontext()
    )
    monkeypatch.setattr(
        rl_trainer.perf_tracer, "trace_scope", lambda *a, **k: nullcontext()
    )
    monkeypatch.setattr(rl_trainer, "is_single_controller", lambda: True)

    trainer.train(workflow="diagnostic")

    expected = []

    def add(role, action):
        if flags[role]:
            expected.append(f"{role}.{action}")

    add("critic", "onload")
    expected.append("critic.values")
    add("critic", "offload")
    add("ref", "onload")
    expected.append("ref.logp")
    add("ref", "offload")
    add("actor", "onload")
    expected += ["actor.logp", "actor.advantages", "actor.update", "actor.lr"]
    add("actor", "offload")
    add("critic", "onload")
    expected += ["critic.update", "critic.lr"]
    add("critic", "offload")
    add("actor", "onload")
    expected.append("actor.weights")
    add("actor", "offload")
    assert events == expected
    for role in ("actor", "critic"):
        engines[role].set_version.assert_called_once_with(1)
    trainer.rollout.set_version.assert_called_once_with(1)
    trainer.rollout.resume.assert_called_once_with()
