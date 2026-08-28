# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)

from areal.engine.fsdp_engine import FSDPEngine, _validate_fsdp_critic_lora
from areal.models.transformers.qwen2_5_vl_value import (
    AReaLQwen2_5_VLForTokenClassification,
)
from areal.trainer.ppo.critic import ppo_loss_fn
from areal.utils import stats_tracker


def _tiny_config() -> Qwen2_5_VLConfig:
    return Qwen2_5_VLConfig(
        text_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 64,
            "use_cache": False,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 1_000_000.0,
                "mrope_section": [1, 1, 2],
            },
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "window_size": 8,
            "out_hidden_size": 32,
            "fullatt_block_indexes": [0],
        },
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
        tie_word_embeddings=False,
    )


def _text_inputs() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[1, 3, 4, 2]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }


def _bare_engine(path, model_config, *, is_critic=True, use_lora=False):
    engine = object.__new__(FSDPEngine)
    engine.config = SimpleNamespace(
        path=str(path),
        is_critic=is_critic,
        use_lora=use_lora,
        attn_impl="eager",
    )
    engine.model_config = model_config
    return engine


def test_qwen25_vl_critic_returns_one_value_per_token_and_backprops():
    model = AReaLQwen2_5_VLForTokenClassification(_tiny_config())
    output = model(**_text_inputs(), use_cache=False)

    assert output.logits.shape == (1, 4, 1)
    assert torch.isfinite(output.logits).all()

    values = output.logits.squeeze(0)
    old_values = values.detach().squeeze(-1)
    try:
        loss = ppo_loss_fn(
            values,
            {
                "values": old_values,
                "returns": old_values + 1.0,
                "loss_mask": torch.ones(4, dtype=torch.bool),
            },
            eps_clip=0.2,
        )
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        loss.backward()
        assert model.score.weight.grad is not None
        assert torch.isfinite(model.score.weight.grad).all()
        assert model.score.weight.grad.count_nonzero() > 0
    finally:
        stats_tracker.export_all(reset=True)


def test_qwen25_vl_critic_forwards_inputs_but_structured_output_omits_cache():
    model = AReaLQwen2_5_VLForTokenClassification(_tiny_config())
    calls = []

    class RecordingBackbone(nn.Module):
        def forward(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                last_hidden_state=torch.ones(1, 4, 32),
                hidden_states=None,
                attentions=None,
            )

    model.model = RecordingBackbone()
    pixel_values = torch.randn(2, 12)
    image_grid_thw = torch.tensor([[1, 1, 2]])
    mm_token_type_ids = torch.ones(1, 4, dtype=torch.int)
    past_key_values = object()
    output = model(
        **_text_inputs(),
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        mm_token_type_ids=mm_token_type_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )

    assert output.logits.shape == (1, 4, 1)
    assert calls[0]["pixel_values"] is pixel_values
    assert calls[0]["image_grid_thw"] is image_grid_thw
    assert calls[0]["mm_token_type_ids"] is mm_token_type_ids
    assert calls[0]["past_key_values"] is past_key_values
    assert calls[0]["use_cache"] is True
    assert not hasattr(output, "past_key_values")


def test_qwen25_vl_critic_warmstarts_and_reloads_through_areal_engine(tmp_path):
    actor_path = tmp_path / "actor"
    critic_path = tmp_path / "critic"
    actor = Qwen2_5_VLForConditionalGeneration(_tiny_config())
    actor.save_pretrained(actor_path)

    model_config = Qwen2_5_VLConfig.from_pretrained(actor_path)
    engine = _bare_engine(actor_path, model_config)
    critic = engine._create_vision_actor_or_critic(torch.float32)

    assert isinstance(critic, AReaLQwen2_5_VLForTokenClassification)
    assert critic.config.architectures == engine.model_config.architectures
    assert critic.score.weight.shape == (1, 32)
    assert "lm_head.weight" not in critic.state_dict()
    assert torch.equal(
        critic.model.language_model.embed_tokens.weight,
        actor.model.language_model.embed_tokens.weight,
    )
    assert critic(**_text_inputs(), use_cache=False).logits.shape == (1, 4, 1)

    with torch.no_grad():
        critic.score.weight.fill_(0.125)
    critic.save_pretrained(critic_path)
    # Match AReaL's saver, which writes the engine config after save_pretrained.
    engine.model_config.save_pretrained(critic_path)

    reload_config = Qwen2_5_VLConfig.from_pretrained(critic_path)
    reload_engine = _bare_engine(critic_path, reload_config)
    reloaded = reload_engine._create_vision_actor_or_critic(torch.float32)
    assert reloaded.score.weight.shape == (1, 32)
    assert torch.equal(reloaded.score.weight, critic.score.weight)
    assert reload_config.architectures == ["AReaLQwen2_5_VLForTokenClassification"]


def test_qwen25_vl_actor_still_uses_image_text_auto_factory(monkeypatch):
    sentinel = object()
    calls = []

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(**kwargs):
            calls.append(kwargs)
            return sentinel

    import areal.engine.fsdp_engine as fsdp_module

    monkeypatch.setattr(fsdp_module, "AutoModelForImageTextToText", FakeAutoModel)
    engine = _bare_engine("actor", _tiny_config(), is_critic=False)
    assert engine._create_vision_actor_or_critic(torch.bfloat16) is sentinel
    assert calls[0]["pretrained_model_name_or_path"] == "actor"
    assert calls[0]["dtype"] == torch.bfloat16
    assert calls[0]["attn_implementation"] == "eager"


def test_unsupported_vision_critic_model_fails_early():
    config = _tiny_config()
    config.model_type = "other_vlm"
    engine = _bare_engine("critic", config)
    with pytest.raises(NotImplementedError, match="support only qwen2_5_vl"):
        engine._create_vision_actor_or_critic(torch.float32)


@pytest.mark.parametrize("model_type", ["qwen2", "qwen2_5_vl"])
def test_all_fsdp_critics_with_lora_fail_before_model_loading(model_type):
    config = SimpleNamespace(is_critic=True, use_lora=True, model_type=model_type)

    with pytest.raises(
        NotImplementedError, match="LoRA is not supported for FSDP critics"
    ):
        _validate_fsdp_critic_lora(config)


def test_fsdp_actor_with_lora_remains_supported():
    config = SimpleNamespace(is_critic=False, use_lora=True)

    _validate_fsdp_critic_lora(config)
