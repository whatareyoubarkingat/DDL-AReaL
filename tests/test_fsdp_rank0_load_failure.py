# SPDX-License-Identifier: Apache-2.0
"""Two CPU ranks exercise real checkpoint-status collectives and clean exit."""

import json
import multiprocessing
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist  # noqa: E402 -- optional torch dependency

from areal.engine.fsdp_engine import FSDPEngine  # noqa: E402


def _rank_worker(rank, rendezvous, output):
    """Keep Gloo usable after both checkpoint loading and serialization fail."""
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    results = []
    try:
        for scenario in ("success", "load_failure", "state_failure", "no_reload"):
            calls = []

            def load():
                calls.append("load")
                if scenario == "load_failure":
                    raise ValueError("injected checkpoint mismatch")

            def state_dict():
                calls.append("state")
                if scenario == "state_failure":
                    raise RuntimeError("injected state_dict failure")
                return {"weight": torch.tensor([2.0])}

            engine = SimpleNamespace(
                _initialized=False,
                _cpu_group=dist.group.WORLD,
                _load_rank0_pretrained_state=load,
                model=SimpleNamespace(state_dict=state_dict),
            )
            error = None
            try:
                state = FSDPEngine._prepare_rank0_state_dict(
                    engine, load_pretrained=scenario != "no_reload"
                )
                assert scenario in ("success", "no_reload")
                if rank == 0:
                    torch.testing.assert_close(state["weight"], torch.tensor([2.0]))
                else:
                    assert state == {}
            except RuntimeError as exc:
                assert scenario in ("load_failure", "state_failure")
                error = str(exc)
                assert "Rank-zero checkpoint preparation failed:" in error
            expected_calls = []
            if rank == 0:
                if scenario != "no_reload":
                    expected_calls.append("load")
                if scenario != "load_failure":
                    expected_calls.append("state")
            assert calls == expected_calls
            results.append({"scenario": scenario, "error": error})
            # This is the same sort of all-rank synchronization used by destroy.
            dist.barrier()
        Path(output, f"rank-{rank}.json").write_text(json.dumps(results))
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.ci
@pytest.mark.skipif(not dist.is_gloo_available(), reason="CPU Gloo is unavailable")
def test_rank0_failures_reach_all_ranks_and_allow_clean_exit(tmp_path):
    """One failing loader must not leave its peers waiting for model weights."""
    ctx = multiprocessing.get_context("spawn")
    rendezvous = (tmp_path / "rendezvous").as_uri()
    processes = [
        ctx.Process(target=_rank_worker, args=(rank, rendezvous, str(tmp_path)))
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + 120
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
        assert all(not process.is_alive() for process in processes), "rank hung"
        assert [process.exitcode for process in processes] == [0, 0]
        results = [
            json.loads((tmp_path / f"rank-{rank}.json").read_text())
            for rank in range(2)
        ]
        assert results[0] == results[1]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(5)
