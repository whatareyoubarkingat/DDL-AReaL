# SPDX-License-Identifier: Apache-2.0
"""Synthetic Qwen3.5/FSDP2/LoRA test, launched only by its pytest wrapper.

No pretrained weights, tokenizer, rollout data, reward, or PPO update is used.
This isolates unequal execution slots with the real hybrid backbone, PEFT,
chunked policy head, grouped checkpointing and CPU parameter/activation offload.
It does not validate long-context capacity or DDLive's RPC/host-memory guard.
"""

import argparse
import json
import math
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

from areal.api.cli_args import MicroBatchSpec
from areal.engine.fsdp_engine import FSDPEngine
from areal.engine.fsdp_utils import apply_fsdp2
from areal.models.transformers.qwen3_5_training import (
    PolicyStatistics,
    configure_grouped_checkpointing,
)
from areal.utils import execution_diagnostics as diagnostics
from areal.utils import logging
from areal.utils.data import MicroBatchList


class _CompactPolicyEngine(FSDPEngine):
    """Only adapt the output type; execution/synchronization stay production code."""

    def _forward_microbatch(self, inputs):
        self.forward_calls += 1
        return self.model(
            **inputs, areal_policy_temperature=0.8, areal_head_chunk_size=16
        )

    def _zero_microbatch_loss(self, output: PolicyStatistics):
        self.padding_calls += 1
        return output.logprobs.sum() * 0.0


def _tiny_model(checkpointing: bool):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

    from areal.models.transformers.qwen3_5_value import (
        AReaLQwen3_5ForPolicyTraining,
    )

    config = Qwen3_5Config(
        text_config={
            "vocab_size": 128,
            "hidden_size": 256,
            "intermediate_size": 384,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "max_position_embeddings": 256,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 64,
            "linear_value_head_dim": 64,
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 4,
            "layer_types": ["linear_attention", "full_attention"] * 2,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
                "mrope_section": [16, 8, 8],
            },
            "attention_dropout": 0.0,
            "pad_token_id": 0,
            "use_cache": False,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 2,
            "out_hidden_size": 256,
            "num_position_embeddings": 16,
        },
        image_token_id=124,
        video_token_id=125,
        vision_start_token_id=126,
        vision_end_token_id=127,
        tie_word_embeddings=False,
    )
    # Eager attention avoids a Flash Attention installation requirement. Native
    # GDN uses installed CUDA kernels when available, otherwise HF's fallback.
    model = AReaLQwen3_5ForPolicyTraining._from_config(
        config, dtype=torch.float32, attn_implementation="eager"
    )
    if checkpointing:
        configure_grouped_checkpointing(model, group_size=4, activation_offload=True)
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            target_modules=[
                "in_proj_qkv",
                "in_proj_z",
                "in_proj_b",
                "in_proj_a",
                "out_proj",
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
        ),
        autocast_adapter_dtype=False,
    )
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert all("lora_" in n for n, p in model.named_parameters() if p.requires_grad)
    return model


def _microbatches(rank: int, world_size: int, device: torch.device):
    lengths = [64, 96] if rank < world_size * 3 // 4 else [48]
    inputs = []
    for index, length in enumerate(lengths):
        # Different token IDs and lengths, with no image/special-token inputs.
        ids = ((torch.arange(length, device=device) + rank + index) % 120 + 1)[None]
        inputs.append(
            dict(
                input_ids=ids,
                attention_mask=torch.ones_like(ids, dtype=torch.bool),
                position_ids=torch.arange(length, device=device)[None],
                use_cache=False,
                return_dict=True,
            )
        )
    return MicroBatchList(
        data={},
        mb_spec=MicroBatchSpec(max_tokens_per_mb=128),
        mbs=[{"input_ids": row["input_ids"]} for row in inputs],
        padded_mbs=inputs,
        group_lens=lengths,
        forward_indices=list(range(len(lengths))),
        backward_indices=list(range(len(lengths))),
        padding_lengths=[0] * len(lengths),
        padded_to_lengths=lengths,
        _max_seqlen=max(lengths),
    )


def _local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _run(args, rank: int, world_size: int, device: torch.device):
    cpu_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
    mesh = init_device_mesh("cuda", (world_size, 1), mesh_dim_names=("dp_sp", "tp"))
    torch.manual_seed(41)
    model = _tiny_model(not args.without_checkpoint).to(device)
    with diagnostics.scope("test_fsdp_wrap"):
        apply_fsdp2(
            model,
            {
                "mesh": mesh["dp_sp"],
                "mp_policy": MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16, reduce_dtype=torch.float32
                ),
                "offload_policy": CPUOffloadPolicy(),
                "reshard_after_forward": True,
            },
            None,
        )
    model.train()
    engine = object.__new__(_CompactPolicyEngine)
    engine._initialized = True
    engine._cpu_group = cpu_group
    engine.model = model
    engine.device = device
    engine.rank = rank
    engine.enable_tree_training = False
    engine.parallel_helper = SimpleNamespace(sp_size=1, dp_size=world_size)
    engine.world_mesh = mesh
    engine.config = SimpleNamespace(
        is_critic=False,
        fsdp=SimpleNamespace(offload_params=True, per_layer_optim_step=False),
    )
    engine.optimizer_config = SimpleNamespace(gradient_clipping=1.0)
    engine.logger = logging.getLogger(f"FSDPEngine Rank {rank}")
    batches = _microbatches(rank, world_size, device)
    parameters = dict(model.named_parameters())
    frozen_before = {
        name: _local(parameter).detach().cpu().clone()
        for name, parameter in parameters.items()
        if not parameter.requires_grad
    }
    engine.optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=0
    )
    engine.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        engine.optimizer, lr_lambda=lambda _: 1.0
    )
    for step in range(2):
        callbacks = []
        engine.forward_calls = engine.padding_calls = 0
        engine.optimizer_zero_grad()
        trainable_before = {
            name: _local(parameter).detach().cpu().clone()
            for name, parameter in parameters.items()
            if parameter.requires_grad
        }

        def loss_callback(output, context):
            callbacks.append(context["mb_input"]["input_ids"].shape[-1])
            # Sum on the actual graph, including every chunk; padding bypasses
            # this callback and contributes zero through the same output edge.
            return -output.logprobs.float().mean()

        with diagnostics.context(test_step=step):
            engine.forward_backward_batch(batches, loss_callback)
            assert engine.forward_calls == 2
            assert engine.padding_calls == 2 - len(batches)
            assert callbacks == batches.group_lens
            assert len(batches) == len(callbacks)
            with diagnostics.scope("test_gradient_validation"):
                torch.cuda.synchronize(device)
                gradients = [
                    _local(p.grad)
                    for p in model.parameters()
                    if p.requires_grad and p.grad is not None
                ]
                assert gradients
                assert all(torch.isfinite(g).all() for g in gradients)
                assert any(torch.count_nonzero(g) for g in gradients)
            with diagnostics.scope("test_optimizer_step"):
                step_stats = engine.optimizer_step()
                torch.cuda.synchronize(device)
                assert step_stats["update_successful"] == 1.0
                assert math.isfinite(step_stats["grad_norm"])
                assert step_stats["grad_norm"] > 0.0
            assert any(
                not torch.equal(_local(parameters[name]).cpu(), before)
                for name, before in trainable_before.items()
            )
            for name, before in frozen_before.items():
                torch.testing.assert_close(
                    _local(parameters[name]).detach().cpu(), before, rtol=0, atol=0
                )
            diagnostics.event("test_step_complete", completed_steps=step + 1)
    receipt = {
        "rank": rank,
        "completed_steps": 2,
        "real_microbatches": len(batches),
        "slots_per_step": 2,
        "synthetic_fixture": True,
        "frozen_parameters_unchanged": True,
        "grouped_checkpointing": not args.without_checkpoint,
        "cpu_parameter_offload": True,
        "production_gradient_clip": True,
        "attention_implementation": "eager",
    }
    (args.output_dir / f"rank-{rank}.json").write_text(json.dumps(receipt, indent=2))
    with diagnostics.scope("test_all_ranks_complete"):
        dist.monitored_barrier(group=cpu_group, timeout=timedelta(seconds=30))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--without-checkpoint", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    rank, world_size = dist.get_rank(), dist.get_world_size()
    try:
        with diagnostics.context(rank=rank, role="synthetic_policy"):
            diagnostics.event("test_begin", world_size=world_size)
            with diagnostics.watchdog(timeout_s=120, operation="fsdp_uneven_fixture"):
                _run(args, rank, world_size, device)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
