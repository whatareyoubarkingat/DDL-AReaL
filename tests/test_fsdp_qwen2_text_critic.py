# SPDX-License-Identifier: Apache-2.0
"""Real CPU model construction, checkpoint mapping and value-head regression."""

from types import SimpleNamespace

import pytest
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    Qwen2Config,
)

from areal.engine.fsdp_engine import FSDPEngine


@pytest.mark.parametrize(
    "memory_efficient_load,init_from_scratch",
    [(True, False), (False, True), (False, False)],
)
def test_text_critic_has_one_value_and_retains_backbone(
    tmp_path, memory_efficient_load, init_from_scratch
):
    """All text critic load modes preserve the backbone and reload a scalar head."""
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    actor = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    actor.save_pretrained(tmp_path / "actor")
    engine = object.__new__(FSDPEngine)
    engine.config = SimpleNamespace(
        is_critic=True,
        optimizer_dtype="float32",
        attn_impl="eager",
        init_from_scratch=init_from_scratch,
        fsdp=SimpleNamespace(memory_efficient_load=memory_efficient_load),
        path=str(tmp_path / "actor"),
    )
    engine.model_config = config
    model = engine._create_llm_actor_or_critic()
    if memory_efficient_load or init_from_scratch:
        result = model.load_state_dict(actor.state_dict(), strict=False)
        assert set(result.missing_keys) <= {"score.weight", "score.bias"}
        assert result.unexpected_keys == ["lm_head.weight"]
    for name, value in actor.state_dict().items():
        if name.startswith("model."):
            torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
    assert config.num_labels != 1
    assert engine.model_config.num_labels == 1
    assert engine.model_config.architectures == [type(model).__name__]
    logits = model(input_ids=torch.tensor([[1, 2, 3]])).logits
    assert logits.shape == (1, 3, 1)
    logits.square().mean().backward()
    assert model.score.weight.grad is not None
    assert torch.isfinite(model.score.weight.grad).all()
    model.save_pretrained(tmp_path / "critic")
    # Match AReaL's saver, which overwrites the model's exported config.
    engine.model_config.save_pretrained(tmp_path / "critic")
    reloaded = AutoModelForTokenClassification.from_pretrained(
        tmp_path / "critic", attn_implementation="eager"
    )
    assert reloaded.config.num_labels == 1
    assert reloaded.config.architectures == [type(model).__name__]
    torch.testing.assert_close(
        reloaded.score.weight, model.score.weight, rtol=0, atol=0
    )
