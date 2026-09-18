# SPDX-License-Identifier: Apache-2.0
"""Real tiny HF models on CPU; these fixtures are not PPO rollout data.

The native and patched models both use HF's own PyTorch Gated DeltaNet fallback.
This deliberately does not validate CUDA/FLA, FSDP, or asynchronous offloading.
"""

from __future__ import annotations

import copy
import importlib

import pytest
import torch

from areal.models.transformers.qwen3_5_training import (
    configure_grouped_checkpointing,
)


@pytest.fixture
def hf_cpu(monkeypatch):
    """Select the real HF CPU kernels before any Qwen3.5 model is constructed."""
    pytest.importorskip("transformers", minversion="5.3.0")
    from transformers.utils import import_utils

    # Optional CUDA libraries may be installed on the test host. Disabling their
    # availability selects the shipped mathematical fallback, not a mock model.
    monkeypatch.setattr(
        import_utils, "is_flash_linear_attention_available", lambda: False
    )
    monkeypatch.setattr(import_utils, "is_causal_conv1d_available", lambda: False)
    hf = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    if not callable(getattr(hf, "torch_chunk_gated_delta_rule", None)):
        pytest.skip("this HF version does not provide a real CPU GDN fallback")
    # Also handle a module already imported by an earlier test in this process.
    for name in (
        "FusedRMSNormGated",
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
    ):
        monkeypatch.setattr(hf, name, None)
    monkeypatch.setattr(hf, "is_fast_path_available", False)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.device("cpu"):
            yield hf
    finally:
        torch.set_num_threads(previous_threads)


def _tiny_config(hf):
    return hf.Qwen3_5Config(
        text_config={
            "vocab_size": 41,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "max_position_embeddings": 64,
            "linear_conv_kernel_dim": 3,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "layer_types": ["linear_attention", "full_attention"] * 2,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
                "mrope_section": [2, 1, 1],
            },
            "attention_dropout": 0.0,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "use_cache": False,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 2,
            "out_hidden_size": 16,
            "num_position_embeddings": 16,
        },
        image_token_id=37,
        video_token_id=38,
        vision_start_token_id=39,
        vision_end_token_id=40,
        tie_word_embeddings=False,
    )


def _model(factory, config):
    return factory._from_config(
        copy.deepcopy(config), dtype=torch.float32, attn_implementation="eager"
    )


def _inputs(*, explicit_positions=True):
    tokens = torch.tensor([[1, 7, 11, 5, 19, 13, 2]], dtype=torch.long)
    inputs = {
        "input_ids": tokens,
        "attention_mask": torch.ones_like(tokens, dtype=torch.bool),
        "use_cache": False,
        "return_dict": True,
    }
    if explicit_positions:
        inputs["position_ids"] = torch.arange(tokens.shape[1]).unsqueeze(0) + 3
    return inputs


def _assert_real_cpu_hybrid(model, hf):
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    layers = model.model.language_model.layers
    assert {layer.layer_type for layer in layers} == {
        "linear_attention",
        "full_attention",
    }
    for layer in layers:
        if layer.layer_type == "linear_attention":
            assert (
                layer.linear_attn.chunk_gated_delta_rule
                is hf.torch_chunk_gated_delta_rule
            )
            assert isinstance(layer.linear_attn.norm, hf.Qwen3_5RMSNormGated)
            assert layer.linear_attn.causal_conv1d_fn is None


def _assert_all_parameter_gradients_match(actual, expected):
    actual_parameters = dict(actual.named_parameters())
    expected_parameters = dict(expected.named_parameters())
    assert actual_parameters.keys() == expected_parameters.keys()
    language_gradients = 0
    for name, expected_parameter in expected_parameters.items():
        actual_parameter = actual_parameters[name]
        if name.startswith("model.visual."):
            # Text-only forward must leave the retained vision tower unused.
            assert actual_parameter.grad is None, name
            assert expected_parameter.grad is None, name
            continue
        assert expected_parameter.grad is not None, name
        assert actual_parameter.grad is not None, name
        assert torch.isfinite(actual_parameter.grad).all(), name
        torch.testing.assert_close(
            actual_parameter.grad,
            expected_parameter.grad,
            rtol=2e-4,
            atol=2e-6,
            msg=lambda message, name=name: f"{name}: {message}",
        )
        language_gradients += 1
    assert language_gradients > 20


@pytest.mark.parametrize(
    "group_size,explicit_positions", [(1, False), (2, True), (3, True)]
)
def test_grouped_checkpoint_matches_native_hybrid_outputs_and_gradients(
    hf_cpu, group_size, explicit_positions
):
    """Nested and uneven layer groups preserve native outputs and all gradients."""
    torch.manual_seed(31)
    native = _model(hf_cpu.Qwen3_5ForConditionalGeneration, _tiny_config(hf_cpu))
    grouped = copy.deepcopy(native)
    configure_grouped_checkpointing(
        grouped, group_size=group_size, activation_offload=False
    )
    _assert_real_cpu_hybrid(native, hf_cpu)
    _assert_real_cpu_hybrid(grouped, hf_cpu)
    native.train()
    grouped.train()
    inputs = _inputs(explicit_positions=explicit_positions)

    # A second backward must not reuse a stale checkpoint graph or GDN cache.
    for _ in range(2):
        native.zero_grad(set_to_none=True)
        grouped.zero_grad(set_to_none=True)
        expected = native(**inputs).logits
        actual = grouped(**inputs).logits
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        expected.float().square().mean().backward()
        actual.float().square().mean().backward()
        _assert_all_parameter_gradients_match(grouped, native)


@pytest.mark.parametrize("position_mode", ["implicit", "offset", "separate_axes"])
def test_grouped_decoder_uses_text_positions_and_rope_keeps_three_axes(
    hf_cpu, position_mode
):
    """Decoder checkpoint recomputation must never interpret RoPE axes as batches."""
    model = _model(hf_cpu.Qwen3_5ForConditionalGeneration, _tiny_config(hf_cpu))
    configure_grouped_checkpointing(model, group_size=3, activation_offload=False)
    model.train()
    text_model = model.model.language_model
    inputs = _inputs(explicit_positions=False)
    text_positions = torch.arange(inputs["input_ids"].shape[1]).unsqueeze(0)
    if position_mode == "offset":
        text_positions = text_positions + 3
        inputs["position_ids"] = text_positions
    rope_positions = text_positions.unsqueeze(0).expand(3, -1, -1)
    if position_mode == "separate_axes":
        text_positions = text_positions + 5
        rope_positions = rope_positions + torch.tensor([3, 7, 11])[:, None, None]
        inputs["position_ids"] = torch.cat(
            (text_positions.unsqueeze(0), rope_positions), dim=0
        )
    decoder_positions = {layer: [] for layer in text_model.layers}
    rotary_positions = []

    def record_decoder_positions(layer, args, kwargs):
        decoder_positions[layer].append(kwargs["position_ids"].detach().clone())

    def record_rotary_positions(module, args):
        rotary_positions.append(args[1].detach().clone())

    handles = [
        layer.register_forward_pre_hook(record_decoder_positions, with_kwargs=True)
        for layer in text_model.layers
    ]
    handles.append(
        text_model.rotary_emb.register_forward_pre_hook(record_rotary_positions)
    )
    try:
        hidden = text_model(**inputs).last_hidden_state
        assert torch.isfinite(hidden).all()
        hidden.float().square().mean().backward()
    finally:
        for handle in handles:
            handle.remove()

    for layer, calls in decoder_positions.items():
        assert calls, layer.layer_type
        for actual in calls:
            assert actual.ndim == 2
            torch.testing.assert_close(actual, text_positions, rtol=0, atol=0)
    assert rotary_positions
    for actual in rotary_positions:
        assert actual.shape == (3, *text_positions.shape)
        torch.testing.assert_close(actual, rope_positions, rtol=0, atol=0)
    for name, parameter in text_model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_hf_flash_attention_metadata_uses_one_text_sequence(hf_cpu):
    """Check HF 5.3 metadata on CPU; never launch a kernel with invalid lengths."""
    from transformers.modeling_flash_attention_utils import (
        _is_packed_sequence,
        _prepare_from_posids,
    )

    text_positions = torch.arange(64).unsqueeze(0)
    rope_positions = text_positions.unsqueeze(0).expand(3, -1, -1)
    query = torch.empty((1, 64, 4, 32), dtype=torch.float32, device="cpu")

    assert not _is_packed_sequence(text_positions, batch_size=1)
    assert not _is_packed_sequence(text_positions + 7, batch_size=1)
    packed_query, _, _, cu_lengths, max_lengths = _prepare_from_posids(
        query, query, query, text_positions
    )
    assert packed_query.shape == (64, 4, 32)
    for actual in cu_lengths:
        torch.testing.assert_close(
            actual, torch.tensor([0, 64], dtype=torch.int32), rtol=0, atol=0
        )
        assert actual[-1].item() == packed_query.shape[0]
    assert tuple(int(length) for length in max_lengths) == (64, 64)

    # This is the previous decoder input: HF mistakes RoPE's axes for packing.
    # Inspect only CPU metadata to reproduce the fault without invoking CUDA.
    assert _is_packed_sequence(rope_positions, batch_size=1)
    wrong_query, _, _, wrong_lengths, _ = _prepare_from_posids(
        query, query, query, rope_positions
    )
    for actual in wrong_lengths:
        torch.testing.assert_close(
            actual,
            torch.tensor([0, 64, 128, 192], dtype=torch.int32),
            rtol=0,
            atol=0,
        )
        assert actual[-1].item() > wrong_query.shape[0]


def test_compact_actor_matches_native_statistics_and_all_gradients(hf_cpu):
    """The real hybrid backbone and chunked head preserve values and gradients."""
    from areal.models.transformers.qwen3_5_value import (
        AReaLQwen3_5ForPolicyTraining,
    )

    torch.manual_seed(32)
    config = _tiny_config(hf_cpu)
    native = _model(hf_cpu.Qwen3_5ForConditionalGeneration, config)
    actor = _model(AReaLQwen3_5ForPolicyTraining, config)
    actor.load_state_dict(native.state_dict(), strict=True)
    configure_grouped_checkpointing(actor, group_size=3, activation_offload=False)
    _assert_real_cpu_hybrid(actor, hf_cpu)
    native.train()
    actor.train()
    inputs = _inputs()
    logits = native(**inputs).logits.float()
    logp = (logits / 0.8).log_softmax(-1)
    labels = inputs["input_ids"].roll(-1, dims=-1)
    expected_logp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    expected_entropy = -(logp.exp() * logp).sum(-1)
    actual = actor(**inputs, areal_policy_temperature=0.8, areal_head_chunk_size=3)
    assert actual.logprobs.shape == inputs["input_ids"].shape
    torch.testing.assert_close(actual.logprobs, expected_logp, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual.entropy, expected_entropy, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual.vocab_min, logits.amin(-1), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual.vocab_max, logits.amax(-1), rtol=1e-5, atol=1e-6)
    (expected_logp.square().mean() + 0.03 * expected_entropy.mean()).backward()
    (actual.logprobs.square().mean() + 0.03 * actual.entropy.mean()).backward()
    _assert_all_parameter_gradients_match(actor, native)


def test_critic_loads_native_checkpoint_and_reloads_scalar_head(hf_cpu, tmp_path):
    """Original composite keys load intact; only lm_head is replaced by score."""
    from safetensors.torch import load_file

    from areal.models.transformers.qwen3_5_value import (
        AReaLQwen3_5ForTokenClassification,
    )

    torch.manual_seed(33)
    config = _tiny_config(hf_cpu)
    native = _model(hf_cpu.Qwen3_5ForConditionalGeneration, config)
    native.save_pretrained(tmp_path / "actor")
    checkpoint_state = load_file(str(tmp_path / "actor" / "model.safetensors"))
    assert "model.language_model.embed_tokens.weight" in checkpoint_state
    assert any(name.startswith("model.visual.") for name in checkpoint_state)
    critic = _model(AReaLQwen3_5ForTokenClassification, config)
    incompatible = critic.load_state_dict(checkpoint_state, strict=False)
    assert incompatible.missing_keys == ["score.weight"]
    assert incompatible.unexpected_keys == ["lm_head.weight"]
    for name, tensor in checkpoint_state.items():
        if name.startswith("model."):
            torch.testing.assert_close(
                critic.state_dict()[name], tensor, rtol=0, atol=0
            )
    assert critic.config.num_labels == 1
    assert config.num_labels != 1
    _assert_real_cpu_hybrid(critic, hf_cpu)
    grouped = copy.deepcopy(critic)
    configure_grouped_checkpointing(grouped, group_size=3, activation_offload=False)
    critic.train()
    grouped.train()
    inputs = _inputs()
    expected = critic(**inputs).logits
    actual = grouped(**inputs).logits
    assert actual.shape == (1, inputs["input_ids"].shape[1], 1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    expected.float().square().mean().backward()
    actual.float().square().mean().backward()
    _assert_all_parameter_gradients_match(grouped, critic)
    assert torch.count_nonzero(grouped.score.weight.grad) > 0

    grouped.save_pretrained(tmp_path / "critic")
    reloaded = AReaLQwen3_5ForTokenClassification.from_pretrained(
        tmp_path / "critic", dtype=torch.float32, attn_implementation="eager"
    )
    assert reloaded.config.num_labels == 1
    assert reloaded.config.architectures == ["AReaLQwen3_5ForTokenClassification"]
    assert reloaded.state_dict().keys() == grouped.state_dict().keys()
    for name, tensor in grouped.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[name], tensor, rtol=0, atol=0)
    reloaded.eval()
    grouped.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            reloaded(**inputs).logits,
            grouped(**inputs).logits,
            rtol=1e-5,
            atol=1e-6,
        )
