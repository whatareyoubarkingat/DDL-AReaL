# SPDX-License-Identifier: Apache-2.0
"""Tiny real NCCL/FSDP2 export regression; never PPO experiment data."""

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, Shard

from areal.engine.fsdp_engine import FSDPEngine


class TinyProjection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 8, bias=False)

    def forward(self, inputs):
        return self.proj(inputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    cpu_group = dist.new_group(backend="gloo")
    try:
        torch.manual_seed(42)
        model = get_peft_model(
            TinyProjection(), LoraConfig(r=2, lora_alpha=4, target_modules=["proj"])
        )
        mesh = init_device_mesh("cuda", (dist.get_world_size(),))
        fully_shard(
            model,
            mesh=mesh,
            offload_policy=CPUOffloadPolicy(),
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32
            ),
        )
        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad], lr=0.01
        )
        # Populate real gradients before export: the target is post-update CPU
        # shards, not an unwrapped or never-used inference-only model.
        inputs = torch.ones(2, 8, device="cuda", dtype=torch.bfloat16)
        model(inputs).float().square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        engine = SimpleNamespace(model=model, cpu_group=cpu_group)
        engine._get_full_tensor = lambda param: FSDPEngine._get_full_tensor(
            engine, param
        )
        engine._cast_to_compute_dtype = lambda tensor: tensor.to(torch.bfloat16)
        before = {}
        expected = {}
        for name, param in model.named_parameters():
            assert isinstance(param, DTensor)
            assert param.device.type == "cpu"
            assert param.dtype == torch.float32
            before[name] = param.to_local().detach().clone()
            if param.requires_grad and "lora_" in name:
                # Independent CPU/Gloo oracle, not the export helper under test.
                assert param.placements == (Shard(0),)
                shards = [
                    torch.empty_like(before[name]) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(shards, before[name], group=cpu_group)
                expected[name.replace(".default.weight", ".weight")] = torch.cat(
                    shards, dim=0
                ).to(torch.bfloat16)

        FSDPEngine._save_lora_to_hf(engine, str(args.output))
        if dist.get_rank() == 0:
            actual = load_file(args.output / "adapter_model.safetensors")
            assert actual.keys() == expected.keys()
            for name, value in actual.items():
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
            assert (args.output / "adapter_config.json").is_file()
        for name, param in model.named_parameters():
            assert param.device.type == "cpu"
            torch.testing.assert_close(param.to_local(), before[name], rtol=0, atol=0)
        dist.barrier(group=cpu_group)
        if dist.get_rank() == 0:
            print("LORA_CPU_OFFLOAD_EXPORT_OK", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
