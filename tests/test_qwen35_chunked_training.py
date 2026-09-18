# SPDX-License-Identifier: Apache-2.0
"""Numerical unit fixtures only; these tensors are never PPO experiment data."""

import copy

import pytest
import torch

from areal.models.transformers.qwen3_5_training import chunked_policy_statistics


@pytest.mark.parametrize("chunk_size", [1, 3, 32])
def test_chunked_projection_matches_full_values_and_all_gradients(chunk_size):
    torch.manual_seed(12)
    full_head = torch.nn.Linear(12, 31, bias=False)
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
    (selected.square().mean() + 0.1 * entropy.mean()).backward()
    (result.logprobs.square().mean() + 0.1 * result.entropy.mean()).backward()
    torch.testing.assert_close(
        chunk_hidden.grad, full_hidden.grad, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        chunk_head.weight.grad, full_head.weight.grad, rtol=1e-5, atol=1e-6
    )


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
