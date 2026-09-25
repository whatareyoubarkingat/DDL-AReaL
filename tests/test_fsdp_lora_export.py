# SPDX-License-Identifier: Apache-2.0
"""LoRA export must use the CPU-offload-aware parameter gather path."""

import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file

from areal.engine.fsdp_engine import FSDPEngine


class TinyProjection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 8, bias=False)

    def forward(self, inputs):
        return self.proj(inputs)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_lora_export_gathers_on_every_rank_without_mutating_storage(
    tmp_path, monkeypatch, rank, dtype
):
    """All ranks gather; only rank zero writes compute-dtype adapter tensors."""
    model = get_peft_model(
        TinyProjection(), LoraConfig(r=2, lora_alpha=4, target_modules=["proj"])
    )
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    gathered = []
    barriers = []

    def gather(param):
        gathered.append(id(param))
        # A distinctive gathered value proves export uses this helper's result,
        # not param.data. Distributed tensor internals are not mocked here.
        return param.detach().clone() + 0.125

    engine = SimpleNamespace(
        model=model,
        cpu_group=object(),
        _get_full_tensor=gather,
        _cast_to_compute_dtype=lambda tensor: tensor.to(dtype),
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(
        torch.distributed, "barrier", lambda group: barriers.append(group)
    )
    FSDPEngine._save_lora_to_hf(engine, str(tmp_path))

    expected = {
        name.replace(".default.weight", ".weight"): (param.detach() + 0.125).to(dtype)
        for name, param in model.named_parameters()
        if param.requires_grad and "lora_" in name
    }
    assert gathered == [
        id(param)
        for name, param in model.named_parameters()
        if param.requires_grad and "lora_" in name
    ]
    assert barriers == [engine.cpu_group]
    if rank == 0:
        saved = load_file(tmp_path / "adapter_model.safetensors")
        assert saved.keys() == expected.keys()
        assert (tmp_path / "adapter_config.json").is_file()
        for name, value in saved.items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    else:
        assert not list(tmp_path.iterdir())
    for name, param in model.named_parameters():
        torch.testing.assert_close(param.detach(), before[name], rtol=0, atol=0)


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_cpu_offloaded_lora_exports_through_nccl(tmp_path):
    """Exercise real FSDP2 CPU offload and NCCL, without downloading a model."""
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two available CUDA GPUs")
    worker = Path(__file__).parent / "torchrun/run_fsdp_lora_export.py"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(worker),
            "--output",
            str(tmp_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        # This session was created above solely for this test's torchrun workers.
        assert os.getpgid(process.pid) == process.pid
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate(timeout=10)
        pytest.fail("LoRA export worker timed out:\n" + stdout + stderr)
    assert process.returncode == 0, stdout + stderr
    assert "LORA_CPU_OFFLOAD_EXPORT_OK" in stdout
