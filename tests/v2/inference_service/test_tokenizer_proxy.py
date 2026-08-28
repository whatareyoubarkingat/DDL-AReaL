# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tokenizer chat-template overrides in the data proxy."""

from unittest.mock import MagicMock

import pytest

from areal.utils.hf_utils import configure_hf_chat_template
from areal.v2.inference_service.data_proxy import tokenizer_proxy as tokenizer_proxy_mod


def test_tokenizer_proxy_overrides_chat_template(monkeypatch, tmp_path):
    """TokenizerProxy loads and installs the configured UTF-8 Jinja template."""
    tokenizer = MagicMock()
    monkeypatch.setattr(
        tokenizer_proxy_mod,
        "load_hf_tokenizer",
        lambda tokenizer_path: tokenizer,
    )
    template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    template_path = tmp_path / "chat_template.jinja"
    template_path.write_text(template, encoding="utf-8")

    proxy = tokenizer_proxy_mod.TokenizerProxy(
        "test-tokenizer",
        chat_template_path=str(template_path),
    )

    assert proxy._tok is tokenizer
    assert tokenizer.chat_template == template


def test_configure_hf_chat_template_rejects_missing_file(tmp_path):
    """A missing override path fails before the tokenizer is used."""
    missing_path = tmp_path / "missing.jinja"

    with pytest.raises(FileNotFoundError, match="Chat template path"):
        configure_hf_chat_template(MagicMock(), str(missing_path))
