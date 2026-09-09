# SPDX-License-Identifier: Apache-2.0
"""Real CPU collectives validate the group used by offloaded FSDP RPC."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from areal.engine.fsdp_engine import FSDPEngine
from areal.infra.rpc.guard.engine_blueprint import resolve_broadcast_target
from areal.utils.data import broadcast_tensor_container


def _check_mirrors(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=90),
    )
    try:
        for memberships in (
            ((0,), (1,), (2,), (3,)),
            ((0, 1), (2, 3)),
            ((0, 1, 2, 3),),
            ((0, 2), (1, 3)),
        ):
            engine = object.__new__(FSDPEngine)
            engine.config = SimpleNamespace(offload=True)
            engine.rank = rank
            engine._cpu_group = dist.group.WORLD
            for members in memberships:
                group = dist.new_group(list(members), backend="gloo")
                if rank in members:
                    engine.mp_group = group
            engine._init_cpu_model_parallel_group()
            original = dist.get_process_group_ranks(engine.mp_group)
            mirror = engine.cpu_model_parallel_group
            assert mirror is not engine.mp_group
            assert dist.get_process_group_ranks(mirror) == original
            assert dist.get_backend(mirror) == "gloo"
            engine.is_offload = True
            target, device = resolve_broadcast_target(engine, "cuda:0")
            assert target is mirror and device == "cpu"
            head = original[0]
            payload = (
                {"head": head, "tensor": torch.tensor([head, head + 7])}
                if rank == head
                else None
            )
            result = broadcast_tensor_container(payload, src_rank=head, group=target)
            assert result["head"] == head
            torch.testing.assert_close(
                result["tensor"], torch.tensor([head, head + 7]), rtol=0, atol=0
            )
            engine.is_offload = False
            target, device = resolve_broadcast_target(engine, "cuda:0")
            assert target is engine.mp_group and device == "cuda:0"
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.ci
def test_fsdp_offloaded_rpc_preserves_actual_rank_partitions(tmp_path):
    """Cover singleton, d2c2, whole-world and noncontiguous model groups."""
    mp.spawn(
        _check_mirrors, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=4, join=True
    )


def test_non_offloaded_fsdp_skips_unused_cpu_mirror(monkeypatch):
    """Do not add startup collectives when neither TMS nor offload is enabled."""
    import areal.engine.fsdp_engine as module

    monkeypatch.setattr(module, "is_tms_enabled", lambda: False)
    engine = object.__new__(FSDPEngine)
    engine.config = SimpleNamespace(offload=False)
    engine._init_cpu_model_parallel_group()
    assert engine.cpu_model_parallel_group is None
