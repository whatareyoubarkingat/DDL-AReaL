# SPDX-License-Identifier: Apache-2.0
"""Microbatch-local tensors must die before the next large model forward."""

import weakref
from types import SimpleNamespace

import pytest
import torch

from areal.engine.fsdp_engine import FSDPEngine, FSDPPPOActor, FSDPPPOCritic


@pytest.mark.parametrize("forward_only", [True, False])
@pytest.mark.parametrize("return_loss", [True, False])
def test_microbatch_outputs_released_before_next_forward(forward_only, return_loss):
    """Exercise the real loop without mocking FSDP or DTensor internals."""
    references = []
    gradients = []
    calls = []

    def model(**inputs):
        assert all(ref() is None for ref in references), (
            "Previous microbatch logits or loss survive into the next forward"
        )
        value = torch.ones(1, 4, 8, requires_grad=not forward_only)
        if not forward_only:
            value.register_hook(lambda grad: gradients.append(grad.clone()))
        references.append(weakref.ref(value))
        calls.append(inputs)
        return SimpleNamespace(logits=value)

    def process_output(logits, context):
        references.append(weakref.ref(logits))
        if return_loss:
            loss = logits.sum()
            references.append(weakref.ref(loss))
            return loss
        return None

    engine = object.__new__(FSDPEngine)
    engine.enable_tree_training = False
    engine.model = model
    engine._prepare_mb_inputs = lambda item: (
        {"item": item},
        SimpleNamespace(to_dict=lambda: {}),
    )

    engine.forward_backward_batch([0, 1, 2], process_output, forward_only)

    assert len(calls) == 3
    assert all(ref() is None for ref in references)
    assert len(gradients) == (3 if return_loss and not forward_only else 0)
    for gradient in gradients:
        torch.testing.assert_close(gradient, torch.ones(1, 4, 8), rtol=0, atol=0)


@pytest.mark.parametrize(
    "engine_type,role,method",
    [
        (FSDPPPOActor, "actor", "compute_logp"),
        (FSDPPPOActor, "actor", "ppo_update"),
        (FSDPPPOCritic, "critic", "compute_values"),
        (FSDPPPOCritic, "critic", "ppo_update"),
    ],
)
@pytest.mark.parametrize("fails", [False, True])
def test_ppo_role_releases_cache_only_after_success(
    monkeypatch, engine_type, role, method, fails
):
    """Keep results live and never mask a primary CUDA error with cache cleanup."""
    import areal.engine.fsdp_engine as module

    events = []
    result = torch.ones(3) if method != "ppo_update" else None

    def delegated(*args, **kwargs):
        events.append("compute")
        if fails:
            raise RuntimeError("primary compute failure")
        return result

    monkeypatch.setattr(
        module.current_platform, "clear_memory", lambda: events.append("clear")
    )
    engine = object.__new__(engine_type)
    setattr(engine, role, SimpleNamespace(**{method: delegated}))

    if fails:
        with pytest.raises(RuntimeError, match="primary compute failure"):
            getattr(engine, method)([])
        assert events == ["compute"]
    else:
        assert getattr(engine, method)([]) is result
        assert events == ["compute", "clear"]
