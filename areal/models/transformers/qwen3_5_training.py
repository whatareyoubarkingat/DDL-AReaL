# SPDX-License-Identifier: Apache-2.0
"""Opt-in, full-sequence Qwen3.5 training primitives (no inference changes)."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from typing import NamedTuple

import torch
from torch.utils.checkpoint import checkpoint


class PolicyStatistics(NamedTuple):
    """O(sequence length) policy outputs, deliberately not named logits."""

    logprobs: torch.Tensor
    entropy: torch.Tensor
    vocab_min: torch.Tensor
    vocab_max: torch.Tensor


def chunked_policy_statistics(
    hidden: torch.Tensor,
    head: Callable,
    labels: torch.Tensor,
    *,
    temperature: float,
    chunk_size: int,
) -> PolicyStatistics:
    """Checkpoint the projection itself, not just log-softmax of full logits.

    The head is called inside the FSDP root forward/recomputation. Do not capture
    a parameter view: FSDP can replace its unsharded storage before backward.
    """
    if chunk_size <= 0 or temperature <= 0:
        raise ValueError("positive chunk_size and temperature are required")
    if hidden.shape[:-1] != labels.shape:
        raise ValueError("hidden states and next-token labels must align")

    def project(x, target):
        logits = head(x).float()
        minimum, maximum = logits.detach().amin(-1), logits.detach().amax(-1)
        logits = logits / temperature
        logp = logits.log_softmax(-1)
        selected = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        entropy = -(logp.exp() * logp).sum(-1)
        # Keep both differentiable statistics on one checkpoint output edge.
        # PPO only differentiates selected logprobs. Separate outputs leave the
        # unused entropy graph alive and retain its recomputed vocabulary-sized
        # tensors until the entire microbatch output is released. A shared edge
        # sends a zero gradient to an unused statistic and drains that cache.
        return (
            torch.stack((selected, entropy), dim=0),
            minimum,
            maximum,
        )

    chunks = []
    # One SplitBackward rejoins chunk gradients once. Independent slices would
    # allocate a full-sequence hidden gradient for every small vocabulary chunk.
    for x, y in zip(
        hidden.split(chunk_size, dim=-2),
        labels.split(chunk_size, dim=-1),
        strict=True,
    ):
        if torch.is_grad_enabled():
            chunks.append(checkpoint(project, x, y, use_reentrant=False))
        else:
            chunks.append(project(x, y))
    statistics, minimum, maximum = (
        torch.cat(xs, dim=-1) for xs in zip(*chunks, strict=True)
    )
    selected, entropy = statistics.unbind(dim=0)
    return PolicyStatistics(selected, entropy, minimum, maximum)


@contextmanager
def offload_checkpoint_inputs(enabled: bool):
    """Offload only outer checkpoint inputs, with ordered asynchronous copies.

    This context must wrap the *checkpoint call*, not its recomputation or an
    entire model forward. In particular it must not intercept saved parameters.
    """
    if not enabled or not torch.cuda.is_available():
        yield
        return
    stream = torch.cuda.Stream()

    def pack(tensor):
        if not tensor.is_cuda or tensor.numel() == 0:
            return tensor
        if tensor.ndim != 3 or not tensor.is_floating_point():
            raise ValueError("activation offload expects a [batch, seq, hidden] input")
        cpu = torch.empty_like(tensor, device="cpu", pin_memory=True)
        source_stream = torch.cuda.current_stream(tensor.device)
        stream.wait_stream(source_stream)
        with torch.cuda.stream(stream):
            cpu.copy_(tensor, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(stream)
        tensor.record_stream(stream)
        return cpu, tensor.device, ready

    def unpack(saved):
        if isinstance(saved, torch.Tensor):
            return saved
        cpu, device, ready = saved
        torch.cuda.current_stream(device).wait_event(ready)
        return cpu.to(device, non_blocking=True)

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield


def grouped_text_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=False,
    cache_position=None,
    **kwargs,
):
    """HF 5.3 Qwen3.5 text forward with explicit group checkpoint boundaries.

    Text-only training is intentional: images belong to the frozen VL service.
    The original HF decoder layers, mask builders, RoPE, and GDN are retained.
    No TP/CP or sequence concatenation is introduced here.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ModelOutputWithPast,
        create_causal_mask,
    )

    if past_key_values is not None or use_cache:
        raise ValueError("full-sequence training does not accept a generation cache")
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("provide exactly one of input_ids and inputs_embeds")
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    if cache_position is None:
        cache_position = torch.arange(
            inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(
            4, inputs_embeds.shape[0], -1
        )
    elif position_ids.ndim == 2:
        position_ids = position_ids[None].expand(4, *position_ids.shape)
    text_positions = None
    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_positions, position_ids = position_ids[0], position_ids[1:]
    causal_mask = create_causal_mask(
        config=self.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=text_positions,
    )
    linear_mask = self._update_linear_attn_mask(attention_mask, cache_position)
    position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
    group_size = self.areal_checkpoint_group_size
    do_checkpoint = self.training and torch.is_grad_enabled()

    def make_group(start, end):
        def group(hidden):
            for index in range(start, end):
                layer = self.layers[index]

                def run_layer(x, layer=layer):
                    return layer(
                        x,
                        position_embeddings=position_embeddings,
                        attention_mask=(
                            linear_mask
                            if layer.layer_type == "linear_attention"
                            else causal_mask
                        ),
                        # RoPE consumes three spatial axes, but Flash Attention
                        # infers sequence boundaries from [batch, sequence] IDs.
                        # Passing the axes here triples the apparent token count.
                        position_ids=text_positions,
                        past_key_values=None,
                        use_cache=False,
                        cache_position=cache_position,
                        **kwargs,
                    )

                # Inner boundaries are recomputed on demand by the outer group;
                # they are not all kept resident/offloaded for the full sequence.
                hidden = (
                    checkpoint(run_layer, hidden, use_reentrant=False)
                    if do_checkpoint
                    else run_layer(hidden)
                )
            return hidden

        return group

    hidden = inputs_embeds
    for start in range(0, len(self.layers), group_size):
        run_group = make_group(start, min(start + group_size, len(self.layers)))
        if do_checkpoint:
            context = offload_checkpoint_inputs(self.areal_activation_offload)
            with context:
                hidden = checkpoint(run_group, hidden, use_reentrant=False)
        else:
            hidden = run_group(hidden)
    return Qwen3_5ModelOutputWithPast(
        last_hidden_state=self.norm(hidden), past_key_values=None
    )


def configure_grouped_checkpointing(model, *, group_size=4, activation_offload=True):
    """Patch one instance only; preserve parameter names and HF checkpoint format."""
    from types import MethodType

    if group_size <= 0:
        raise ValueError("checkpoint group size must be positive")
    model.gradient_checkpointing_disable()
    text_model = model.model.language_model
    text_model.areal_checkpoint_group_size = group_size
    text_model.areal_activation_offload = activation_offload
    text_model.forward = MethodType(grouped_text_forward, text_model)
