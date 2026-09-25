# SPDX-License-Identifier: Apache-2.0
"""Numerical unit fixtures only; these tensors are never PPO experiment data."""

import copy

import pytest
import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import TorchDispatchMode

from areal.models.transformers.qwen3_5_training import chunked_policy_statistics


@pytest.mark.parametrize("chunk_size", [1, 3, 32])
@pytest.mark.parametrize("loss_mode", ["logprob", "entropy", "both"])
@pytest.mark.parametrize("train_head", [True, False])
def test_chunked_projection_matches_full_values_and_all_gradients(
    chunk_size, loss_mode, train_head
):
    torch.manual_seed(12)
    full_head = torch.nn.Linear(12, 31, bias=False)
    full_head.requires_grad_(train_head)
    chunk_head = copy.deepcopy(full_head)
    full_hidden = torch.randn(1, 13, 12, requires_grad=True)
    chunk_hidden = full_hidden.detach().clone().requires_grad_(True)
    labels = torch.randint(0, 31, (1, 13))
    logits = full_head(full_hidden).float() / 0.8
    logp = logits.log_softmax(-1)
    selected = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    entropy = -(logp.exp() * logp).sum(-1)
    result = chunked_policy_statistics(
        chunk_hidden, chunk_head, labels, temperature=0.8, chunk_size=chunk_size
    )
    torch.testing.assert_close(result.logprobs, selected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(result.entropy, entropy, rtol=1e-5, atol=1e-6)

    def loss(logprobs, entropy):
        if loss_mode == "logprob":
            return logprobs.square().mean()
        if loss_mode == "entropy":
            return entropy.mean()
        return logprobs.square().mean() + 0.1 * entropy.mean()

    loss(selected, entropy).backward()
    loss(result.logprobs, result.entropy).backward()
    torch.testing.assert_close(
        chunk_hidden.grad, full_hidden.grad, rtol=1e-5, atol=1e-6
    )
    if train_head:
        torch.testing.assert_close(
            chunk_head.weight.grad, full_head.weight.grad, rtol=1e-5, atol=1e-6
        )
    else:
        assert chunk_head.weight.grad is None
        assert full_head.weight.grad is None


@pytest.mark.parametrize("train_head", [True, False])
@pytest.mark.parametrize("loss_mode", ["logprob", "entropy", "both", "zero", "sparse"])
def test_partial_output_backward_releases_recomputed_vocab_storage(
    train_head, loss_mode
):
    """Unused output fields must not retain each chunk's reconstructed softmax."""

    class TrackLogSoftmaxStorage(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.storage_refs = []
            self.peak_live = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if func is torch.ops.aten._log_softmax.default:
                # StorageImpl weak references survive tensor/detach aliases,
                # unlike weakrefs to transient Python tensor/storage wrappers.
                self.storage_refs.append(StorageWeakRef(result.untyped_storage()))
                self.peak_live = max(
                    self.peak_live,
                    sum(not ref.expired() for ref in self.storage_refs),
                )
            return result

    torch.manual_seed(24)
    head = torch.nn.Linear(12, 61, bias=False).requires_grad_(train_head)
    hidden = torch.randn(2, 19, 12, requires_grad=True)
    labels = torch.randint(0, 61, (2, 19))
    tracker = TrackLogSoftmaxStorage()
    with tracker:
        output = chunked_policy_statistics(
            hidden, head, labels, temperature=0.8, chunk_size=3
        )
        assert not any(not ref.expired() for ref in tracker.storage_refs)
        if loss_mode == "entropy":
            loss = output.entropy.mean()
        elif loss_mode == "both":
            loss = output.logprobs.mean() + 0.1 * output.entropy.mean()
        elif loss_mode == "zero":
            loss = output.logprobs.mean() * 0.0
        elif loss_mode == "sparse":
            loss = output.logprobs[0, 0]
        else:
            loss = output.logprobs.mean()
        loss.backward()

    # Keep all outputs alive, exactly as FSDP does until backward has returned.
    assert output.logprobs.requires_grad and output.entropy.requires_grad
    assert len(tracker.storage_refs) >= 14  # Seven forward and recomputed chunks.
    assert tracker.peak_live <= 1
    assert not any(not ref.expired() for ref in tracker.storage_refs)
    assert torch.isfinite(hidden.grad).all()


def test_chunked_projection_no_grad_returns_detached_statistics():
    """Evaluation preserves values without building checkpoint/autograd graphs."""
    torch.manual_seed(25)
    head = torch.nn.Linear(12, 31, bias=False)
    hidden = torch.randn(2, 13, 12, requires_grad=True)
    labels = torch.randint(0, 31, (2, 13))
    with torch.no_grad():
        output = chunked_policy_statistics(
            hidden, head, labels, temperature=0.8, chunk_size=3
        )
        logits = head(hidden).float()
        logp = (logits / 0.8).log_softmax(-1)
        expected = (
            logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1),
            -(logp.exp() * logp).sum(-1),
            logits.amin(-1),
            logits.amax(-1),
        )
    for actual, reference in zip(output, expected, strict=True):
        assert not actual.requires_grad
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)


def test_chunked_projection_rejects_misaligned_labels():
    with pytest.raises(ValueError, match="align"):
        chunked_policy_statistics(
            torch.zeros(1, 4, 3),
            torch.nn.Linear(3, 7),
            torch.zeros(1, 3, dtype=torch.long),
            temperature=1.0,
            chunk_size=2,
        )


def test_chunked_projection_does_not_retain_full_vocab_activations():
    hidden = torch.randn(1, 19, 12, requires_grad=True)
    head = torch.nn.Linear(12, 61, bias=False)
    saved = []

    def pack(tensor):
        saved.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        output = chunked_policy_statistics(
            hidden,
            head,
            torch.zeros(1, 19, dtype=torch.long),
            temperature=1.0,
            chunk_size=3,
        )
    assert not any(shape and shape[-1] == 61 for shape in saved)
    output.logprobs.mean().backward()
    assert head.weight.grad is not None


def test_policy_output_exposes_tensors_to_fsdp_pytree_hooks():
    from torch.utils._pytree import tree_leaves

    hidden = torch.randn(1, 4, 3, requires_grad=True)
    output = chunked_policy_statistics(
        hidden,
        torch.nn.Linear(3, 7),
        torch.zeros(1, 4, dtype=torch.long),
        temperature=1.0,
        chunk_size=2,
    )
    leaves = tree_leaves(output)
    assert len(leaves) == 4
    assert leaves[0] is output.logprobs
    assert leaves[0].requires_grad
