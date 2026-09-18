# SPDX-License-Identifier: Apache-2.0
"""A stuck engine RPC must not prevent run-owned worker cleanup."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("torch")
from areal.infra.controller import train_controller


@pytest.mark.parametrize("mode", ["ok", "failure", "hang"])
def test_destroy_always_deletes_only_its_workers(monkeypatch, mode):
    """Successful, failing, and blocked RPCs all reach bounded scheduler cleanup."""
    ended = []

    async def destroy_rpc(**kwargs):
        try:
            if mode == "hang":
                await asyncio.Event().wait()
            elif mode == "failure":
                raise RuntimeError("engine unavailable")
        finally:
            ended.append(kwargs["worker_id"])

    scheduler = SimpleNamespace(
        async_call_engine=AsyncMock(side_effect=destroy_rpc), delete_workers=Mock()
    )
    controller = object.__new__(train_controller.TrainController)
    controller.scheduler = scheduler
    controller.workers = [SimpleNamespace(id="actor/0"), SimpleNamespace(id="actor/1")]
    controller.workers_is_dp_head = [True, True]
    controller._worker_role = "actor"
    controller._engine_name = lambda rank: f"actor_engine/{rank}"
    controller._own_process_group = False
    monkeypatch.setattr(train_controller, "_ENGINE_DESTROY_TIMEOUT_SECONDS", 0.05)

    controller.destroy()

    assert sorted(ended) == ["actor/0", "actor/1"]
    scheduler.delete_workers.assert_called_once_with(role="actor", reverse_order=True)
    assert controller.workers == []
    assert controller.workers_is_dp_head == []
