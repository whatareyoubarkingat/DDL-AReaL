# SPDX-License-Identifier: Apache-2.0
"""Bounded real-CUDA reproduction; synthetic fixtures never enter PPO data."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch


def _stop_owned_process_group(process: subprocess.Popen) -> None:
    """torchrun and its workers belong to this newly created process group."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    # The leader may have exited before a stuck worker; reap the whole group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


@pytest.mark.slow
@pytest.mark.multi_gpu
def test_fsdp_lora_uneven_slots_complete_two_optimizer_steps(tmp_path):
    """Two versus one real MBs exercise nested checkpointing and CPU offload."""
    world_size = int(os.environ.get("AREAL_FSDP_UNEVEN_WORLD_SIZE", "2"))
    if world_size not in (2, 8):
        pytest.fail("AREAL_FSDP_UNEVEN_WORLD_SIZE must be 2 or 8")
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} CUDA GPUs")
    pytest.importorskip("transformers", minversion="5.3.0")
    pytest.importorskip("peft")
    root = Path(__file__).resolve().parents[1]
    diagnostics_dir = tmp_path / "diagnostics"
    environment = dict(os.environ)
    environment["AREAL_EXECUTION_DIAGNOSTICS_DIR"] = str(diagnostics_dir)
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world_size}",
        str(root / "tests/torchrun/run_fsdp_uneven_diagnostics.py"),
        "--output-dir",
        str(tmp_path),
    ]
    if os.environ.get("AREAL_FSDP_UNEVEN_NO_CHECKPOINT") == "1":
        command.append("--without-checkpoint")
    log_path = tmp_path / "torchrun.log"
    timed_out = False
    with log_path.open("w") as output:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            process.wait(timeout=180)
        except subprocess.TimeoutExpired:
            timed_out = True
            _stop_owned_process_group(process)
        finally:
            if process.poll() is None:
                _stop_owned_process_group(process)
    if timed_out or process.returncode:
        pytest.fail(
            f"FSDP reproduction {'timed out' if timed_out else 'failed'}; "
            f"logs: {log_path}; diagnostics: {diagnostics_dir}\n"
            f"{log_path.read_text()[-24000:]}"
        )
    for rank in range(world_size):
        receipt = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        assert receipt["completed_steps"] == 2
        assert receipt["real_microbatches"] == (2 if rank < world_size * 3 // 4 else 1)
        assert receipt["slots_per_step"] == 2
        assert receipt["synthetic_fixture"] is True
        assert receipt["frozen_parameters_unchanged"] is True
    assert len(list(diagnostics_dir.glob("*.jsonl"))) >= world_size
