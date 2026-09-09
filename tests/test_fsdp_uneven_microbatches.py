# SPDX-License-Identifier: Apache-2.0
"""Unequal DP trajectory lengths require equal compute slots, not fake samples."""

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from areal.api.cli_args import MicroBatchSpec
from areal.engine.fsdp_engine import FSDPEngine
from areal.utils.data import allocate_balanced_mbs_synced


class _ScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.calls = 0

    def forward(self, input_ids):
        self.calls += 1
        return SimpleNamespace(logits=input_ids.float().unsqueeze(-1) * self.weight)


def _check_uneven_batches(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=90),
    )
    try:
        for members in ([0, 1], [2, 3]):
            group = dist.new_group(members, backend="gloo")
            if rank in members:
                model_group = group
        engine = object.__new__(FSDPEngine)
        engine._initialized = True
        engine.enable_tree_training = False
        engine._cpu_group = dist.group.WORLD
        engine.mp_group = model_group
        engine.model_config = SimpleNamespace(model_type="qwen2")
        engine.config = SimpleNamespace(
            mb_spec=MicroBatchSpec(max_tokens_per_mb=64),
            pad_to_maximum=False,
        )
        engine.logger = Mock()

        # Exact production planner shape, without allocating 64K token tensors.
        row_count = 97 if rank < 2 else 73
        plan = allocate_balanced_mbs_synced(
            MicroBatchSpec(max_tokens_per_mb=65536),
            [60000] * row_count,
            group=model_group,
        )
        assert len(plan) == row_count

        # Exercise real FSDP batch preparation and real Gloo/DDP collectives.
        rows = 3 if rank < 2 else 1
        ids = torch.stack([torch.full((64,), i + 1) for i in range(rows)])
        batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        mb_list = engine._prepare_mb_list(batch)
        assert len(mb_list) == rows
        assert mb_list.forward_indices == list(range(rows))
        assert mb_list.backward_indices == list(range(rows))
        assert sum(mb_list.group_lens) == rows * 64

        engine.model = DistributedDataParallel(_ScalarModel())
        engine._prepare_mb_inputs = lambda item: (
            {"input_ids": item.padded_mb["input_ids"]},
            SimpleNamespace(to_dict=lambda: {}),
        )
        for forward_only in (True, False):
            outputs = []
            engine.model.module.calls = 0
            engine.model.zero_grad()

            def callback(logits, context):
                outputs.append(logits.detach().clone())
                return logits.sum()

            engine.forward_backward_batch(mb_list, callback, forward_only)
            assert engine.model.module.calls == 3
            assert len(outputs) == rows
            assert len(mb_list) == rows  # Padding slots never become samples.
            for i, output in enumerate(outputs):
                torch.testing.assert_close(
                    output[:64],
                    torch.full((64, 1), 2.0 * (i + 1)),
                    rtol=0,
                    atol=0,
                )
            if not forward_only:
                # DDP averages over all 4 ranks. CP replicas have the same data:
                # (2 * (1+2+3)*64 + 2 * 1*64) / 4 == 224.
                torch.testing.assert_close(
                    engine.model.module.weight.grad,
                    torch.tensor(224.0),
                    rtol=0,
                    atol=0,
                )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.ci
def test_uneven_dp_microbatches_preserve_outputs_and_gradient(tmp_path):
    """DP shards with 3 versus 1 real MBs run equal slots with unbiased gradients."""
    mp.spawn(
        _check_uneven_batches,
        args=(f"file://{tmp_path / 'rendezvous'}",),
        nprocs=4,
        join=True,
    )
