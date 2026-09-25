# SPDX-License-Identifier: Apache-2.0
"""CPU-only coverage of native template options and preserved reasoning."""

from __future__ import annotations

import json
import threading
from copy import deepcopy
from types import SimpleNamespace

import httpx
import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from areal.api import ModelResponse
from areal.experimental.openai import client as client_module
from areal.experimental.openai.client import (
    ArealOpenAI,
    _split_native_reasoning_content,
)
from areal.experimental.openai.proxy import proxy_rollout_server as srv
from areal.experimental.openai.proxy.server import SessionData


@pytest.mark.parametrize(
    "text,prompt,options,expected",
    [
        (
            "<think>reason</think>answer",
            "<think>\n",
            {},
            (None, "<think>reason</think>answer"),
        ),
        (
            "<think>reason</think>answer",
            "<think>\n",
            {"preserve_thinking": True},
            ("reason", "answer"),
        ),
        (
            "reason</think>answer",
            "<think>\n",
            {"preserve_thinking": True},
            ("reason", "answer"),
        ),
        ("reason", "<think>\n", {"preserve_thinking": True}, ("reason", "")),
        ("<think>reason", "<think>\n", {"preserve_thinking": True}, ("reason", "")),
        (
            "answer",
            "<think>\n</think>\n",
            {"preserve_thinking": True, "enable_thinking": False},
            (None, "answer"),
        ),
        (
            "ordinary answer",
            "assistant\n",
            {"preserve_thinking": True},
            (None, "ordinary answer"),
        ),
        (
            "reason</think>answer",
            "<think>\n",
            {"preserve_thinking": False},
            ("reason", "answer"),
        ),
        ("</think>answer", "<think>\n", {"preserve_thinking": True}, (None, "answer")),
    ],
)
def test_native_reasoning_split_respects_prompt_and_opt_in(
    text, prompt, options, expected
):
    """Only an explicit native profile changes response reasoning representation."""
    assert (
        _split_native_reasoning_content(
            text, prompt_suffix=prompt, template_kwargs=options
        )
        == expected
    )


@pytest.fixture
def proxy_capture(monkeypatch):
    """Capture the actual kwargs surviving FastAPI and proxy validation."""
    captured = []

    async def create(
        *, messages, extra_body=None, temperature=None, top_p=None, areal_cache=None
    ):
        captured.append({"messages": messages, "extra_body": extra_body})
        return ChatCompletion(
            id="test-completion",
            choices=[
                Choice(
                    index=0,
                    finish_reason="stop",
                    message=ChatCompletionMessage(role="assistant", content="ok"),
                )
            ],
            created=0,
            model="test",
            object="chat.completion",
        )

    monkeypatch.setattr(
        srv,
        "_openai_client",
        SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
    )
    monkeypatch.setattr(
        srv, "_session_cache", {"session": SessionData(session_id="session")}
    )
    monkeypatch.setattr(srv, "_api_key_to_session", {"test-key": "session"})
    monkeypatch.setattr(srv, "_lock", threading.Lock())
    return captured


async def _post_chat(payload):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=srv.app), base_url="http://test"
    ) as client:
        return await client.post(
            "/chat/completions",
            headers={"Authorization": "Bearer test-key"},
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                **payload,
            },
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("nested", [False, True])
async def test_http_template_kwargs_survive_validation(proxy_capture, nested):
    """Both SDK-flattened and historical nested options reach the tokenizer client."""
    options = {"reasoning_effort": "low", "preserve_thinking": True}
    extension = {"chat_template_kwargs": options}
    payload = {"extra_body": extension} if nested else extension
    response = await _post_chat(payload)
    assert response.status_code == 200
    assert proxy_capture[0]["extra_body"] == extension


@pytest.mark.asyncio
async def test_http_template_kwargs_merge_without_mutation(proxy_capture):
    """Equal shared keys are allowed and unrelated extra_body entries survive."""
    response = await _post_chat(
        {
            "chat_template_kwargs": {
                "reasoning_effort": "low",
                "preserve_thinking": True,
            },
            "extra_body": {
                "chat_template_kwargs": {"reasoning_effort": "low"},
                "metadata_tag": "audit",
            },
        }
    )
    assert response.status_code == 200
    assert proxy_capture[0]["extra_body"] == {
        "metadata_tag": "audit",
        "chat_template_kwargs": {"reasoning_effort": "low", "preserve_thinking": True},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"chat_template_kwargs": []},
        {"chat_template_kwargs": None},
        {"extra_body": []},
        {"extra_body": {"chat_template_kwargs": "low"}},
        {
            "chat_template_kwargs": {"reasoning_effort": "low"},
            "extra_body": {"chat_template_kwargs": {"reasoning_effort": "high"}},
        },
    ],
)
async def test_http_invalid_template_kwargs_rejected_before_inference(
    proxy_capture, payload
):
    """Malformed mappings and ambiguous precedence fail explicitly with HTTP 400."""
    response = await _post_chat(payload)
    assert response.status_code == 400
    assert not proxy_capture


@pytest.mark.asyncio
async def test_http_native_history_preserves_reasoning_content(proxy_capture):
    """The native assistant extension must not disappear during TypedDict parsing."""
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "answer", "reasoning_content": "reason"},
        {"role": "user", "content": "continue"},
    ]
    response = await _post_chat(
        {"messages": messages, "chat_template_kwargs": {"preserve_thinking": True}}
    )
    assert response.status_code == 200
    assert proxy_capture[0]["messages"][1]["reasoning_content"] == "reason"


@pytest.mark.asyncio
async def test_http_native_history_rejects_non_string_reasoning(proxy_capture):
    """Native reasoning is text, not arbitrary nested provider data."""
    response = await _post_chat(
        {
            "messages": [
                {"role": "assistant", "content": "answer", "reasoning_content": []}
            ],
            "chat_template_kwargs": {"preserve_thinking": True},
        }
    )
    assert response.status_code == 400
    assert not proxy_capture


@pytest.mark.asyncio
async def test_http_legacy_top_level_options_do_not_enable_native_mode(proxy_capture):
    """No arbitrary kwargs bypass and no new interpretation of legacy effort."""
    response = await _post_chat(
        {
            "reasoning_effort": "low",
            "arbitrary_sampler_option": 123,
            "areal_cache": {"untrusted": "replacement"},
        }
    )
    assert response.status_code == 200
    assert proxy_capture[0]["extra_body"] is None


@pytest.mark.asyncio
async def test_http_standard_message_validation_is_retained(proxy_capture):
    """Extension recovery does not replace the existing request validation."""
    response = await _post_chat({"messages": "not a message list"})
    assert response.status_code == 422
    assert not proxy_capture


def test_template_kwargs_normalization_does_not_mutate_request():
    """Merging a flattened extension never edits caller-owned nested objects."""
    request = {
        "extra_body": {"chat_template_kwargs": {"preserve_thinking": True}},
        "chat_template_kwargs": {"reasoning_effort": "low"},
    }
    original = deepcopy(request)
    normalized = srv._normalize_chat_template_kwargs(request)
    assert request == original
    assert normalized["extra_body"]["chat_template_kwargs"] == {
        "preserve_thinking": True,
        "reasoning_effort": "low",
    }


class _Tokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text, **kwargs):
        return [ord(character) for character in text]

    def decode(self, token_ids, **kwargs):
        return "".join(chr(token) for token in token_ids)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("native", [False, True])
async def test_native_client_response_stream_and_cache_agree(
    monkeypatch, stream, native
):
    """A second turn links to the first and token/logprob capture is unchanged."""
    tokenizer = _Tokenizer()
    raw_output = "<think>consider</think>answer<tool_call><function=read_file><parameter=path>/tmp/example</parameter></function></tool_call>"
    output_tokens = tokenizer.encode(raw_output) + [0]
    output_logprobs = [-0.125] * len(output_tokens)
    responses = []
    rendered_kwargs = []

    def render(tokenizer, messages, **kwargs):
        rendered_kwargs.append(deepcopy(kwargs))
        return tokenizer.encode("assistant\n<think>\n")

    async def generate(request):
        response = ModelResponse(
            input_tokens=request.input_ids,
            output_tokens=list(output_tokens),
            output_logprobs=list(output_logprobs),
            output_versions=[0] * len(output_tokens),
            tokenizer=tokenizer,
        )
        responses.append(response)
        return response

    monkeypatch.setattr(client_module, "apply_chat_template", render)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
        }
    ]
    extra_body = (
        {"chat_template_kwargs": {"preserve_thinking": True, "reasoning_effort": "low"}}
        if native
        else {}
    )
    async with ArealOpenAI(
        engine=SimpleNamespace(agenerate=generate),
        tokenizer=tokenizer,
        tool_call_parser="qwen3_coder",
        api_key="test",
        base_url="http://test",
    ) as client:
        messages = [{"role": "user", "content": "read"}]
        completion = await client.chat.completions.create(
            messages=messages, tools=tools, stream=stream, extra_body=extra_body
        )
        if stream:
            chunks = [chunk async for chunk in completion]
            completion_id = chunks[0].id
            interaction = client.get_interaction(completion_id)
            message = interaction.completion.choices[0].message.model_dump(
                exclude_none=True
            )
            streamed_reasoning = "".join(
                getattr(chunk.choices[0].delta, "reasoning_content", "") or ""
                for chunk in chunks
            )
            streamed_content = "".join(
                chunk.choices[0].delta.content or "" for chunk in chunks
            )
            assert streamed_reasoning == ("consider" if native else "")
            assert streamed_content == message["content"]
        else:
            completion_id = completion.id
            interaction = client.get_interaction(completion_id)
            message = completion.choices[0].message.model_dump(exclude_none=True)
        assert message["content"] == (
            "answer" if native else "<think>consider</think>answer"
        )
        assert message.get("reasoning_content") == ("consider" if native else None)
        assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
            "path": "/tmp/example"
        }
        assert interaction.output_message_list == [message]
        assert responses[0].output_tokens == output_tokens
        assert responses[0].output_logprobs == output_logprobs
        messages.extend(
            [
                message,
                {
                    "role": "tool",
                    "tool_call_id": message["tool_calls"][0]["id"],
                    "content": "file text",
                },
            ]
        )
        child = await client.chat.completions.create(
            messages=messages, tools=tools, extra_body=extra_body
        )
        assert client.get_interaction(child.id).parent is interaction
        assert rendered_kwargs[0].get("preserve_thinking") is (True if native else None)
