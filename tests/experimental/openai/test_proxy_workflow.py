# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

import pytest

from areal.experimental.openai.proxy import workflow as workflow_module
from areal.experimental.openai.proxy.workflow import OpenAIProxyWorkflow
from areal.infra import workflow_context


class _RejectingAgent:
    async def run(self, data, **extra_kwargs):
        return None


class _FakeProxyClient:
    def __init__(self, **kwargs):
        self.session_api_key = "session-key"
        self.entered = False
        self.exited = False
        self.exit_args = None
        self.set_reward = AsyncMock()
        self.set_last_reward = AsyncMock()
        self.export_interactions = AsyncMock(return_value={"discarded": object()})

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.exited = True
        self.exit_args = (exc_type, exc_val, exc_tb)


@pytest.mark.asyncio
async def test_none_reward_rejects_without_set_and_exports_for_cleanup(monkeypatch):
    """A None result exports only to clean up and never creates training data."""
    workflow = OpenAIProxyWorkflow(
        mode="inline",
        agent=_RejectingAgent(),
        proxy_addr="http://proxy.invalid",
    )
    proxy_client = _FakeProxyClient()
    monkeypatch.setattr(
        workflow_module, "OpenAIProxyClient", lambda **kwargs: proxy_client
    )
    monkeypatch.setattr(
        workflow_context,
        "get_aiohttp_session",
        AsyncMock(return_value=object()),
    )
    monkeypatch.setattr(workflow, "_grant_capacity", AsyncMock())
    monkeypatch.setattr(workflow, "_run_agent", AsyncMock(return_value=None))
    workflow_context.set(workflow_context.WorkflowContext(task_id=7))

    result = await workflow.arun_episode(engine=None, data={"query": "test"})

    assert result is None
    assert proxy_client.entered is True
    assert proxy_client.exited is True
    assert proxy_client.exit_args == (None, None, None)
    proxy_client.set_reward.assert_not_awaited()
    proxy_client.set_last_reward.assert_not_awaited()
    proxy_client.export_interactions.assert_awaited_once_with(
        discount=1.0,
        style="individual",
        drop_retry_orphans=False,
    )
