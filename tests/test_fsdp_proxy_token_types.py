# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from areal.api import ModelResponse
from areal.api.cli_args import MicroBatchSpec
from areal.engine.fsdp_engine import FSDPEngine
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.utils.data import unpad_logits


def _engine(monkeypatch):
    import areal.engine.fsdp_engine as module

    monkeypatch.setattr(module.dist, "get_rank", lambda: 0)
    engine = object.__new__(FSDPEngine)
    engine.enable_tree_training = False
    engine.model_config = SimpleNamespace(
        model_type="qwen2_5_vl", image_token_id=60, video_token_id=61
    )
    engine.config = SimpleNamespace(
        mb_spec=MicroBatchSpec(max_tokens_per_mb=64), pad_to_maximum=False
    )
    engine.logger = Mock()
    engine.model = SimpleNamespace(
        model=SimpleNamespace(
            rope_deltas=None, compute_3d_position_ids=Mock(return_value=None)
        )
    )
    return engine


def _proxy_batch():
    interaction = InteractionWithTokenLogpReward(
        model_response=ModelResponse(
            input_tokens=[1, 3, 4],
            output_tokens=[5, 2],
            output_logprobs=[-0.2, -0.3],
            output_versions=[0, 0],
        ),
        reward=-1.0,
    )
    return interaction.to_tensor_dict()


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_real_proxy_text_export_reaches_qwen_microbatch(monkeypatch, dtype):
    engine = _engine(monkeypatch)
    batch = _proxy_batch()
    assert "mm_token_type_ids" not in batch  # Exact production boundary.
    batch["input_ids"] = batch["input_ids"].to(dtype)

    microbatches = engine._prepare_mb_list(batch)

    kwargs = engine.model.model.compute_3d_position_ids.call_args.kwargs
    assert kwargs["mm_token_type_ids"].dtype == torch.long
    torch.testing.assert_close(
        kwargs["mm_token_type_ids"], torch.zeros(1, 5, dtype=torch.long)
    )
    assert "mm_token_type_ids" not in batch  # Do not mutate the shared rollout batch.
    assert microbatches.padded_mbs
    padded_length = microbatches.padded_to_lengths[0]
    assert microbatches.padded_mbs[0]["position_ids"].shape == (3, 1, padded_length)
    assert microbatches.padded_mbs[0]["mm_token_type_ids"].shape == (1, padded_length)
    torch.testing.assert_close(
        microbatches.padded_mbs[0]["position_ids"][:, 0, :5],
        torch.arange(5).expand(3, -1),
    )


@pytest.mark.parametrize("length", [255, 256, 257, 512])
def test_fixed_padding_preserves_real_qwen_tokens_positions_and_loss_mask(
    monkeypatch, length
):
    """Fixed allocation shapes must not introduce samples or effective loss tokens."""
    engine = _engine(monkeypatch)
    engine.config.mb_spec = MicroBatchSpec(max_tokens_per_mb=1024)
    batch = {
        "input_ids": torch.ones((1, length), dtype=torch.long),
        "attention_mask": torch.ones((1, length), dtype=torch.bool),
        "loss_mask": torch.zeros((1, length), dtype=torch.bool),
    }
    batch["loss_mask"][:, -4:-1] = True
    normal = engine._prepare_mb_list(batch)
    engine.config.pad_to_maximum = True
    fixed = engine._prepare_mb_list(batch)

    assert len(normal) == len(fixed) == 1
    assert fixed.group_lens == normal.group_lens == [length]
    assert fixed.forward_indices == normal.forward_indices
    assert fixed.backward_indices == normal.backward_indices
    assert fixed.padded_to_lengths == [1024]
    assert fixed.padding_lengths == [1024 - length]
    original, padded = normal.padded_mbs[0], fixed.padded_mbs[0]
    for key in ("input_ids", "mm_token_type_ids", "loss_mask"):
        torch.testing.assert_close(
            padded[key][..., :length], original[key][..., :length], rtol=0, atol=0
        )
    torch.testing.assert_close(
        padded["position_ids"][..., :length],
        original["position_ids"][..., :length],
        rtol=0,
        atol=0,
    )
    assert not padded["mm_token_type_ids"][..., length:].any()
    assert not padded["loss_mask"][..., length:].any()
    assert int(padded["loss_mask"].sum()) == int(batch["loss_mask"].sum()) == 3
    assert "mm_token_type_ids" not in batch
    # Values/log-probabilities are cropped before real-token loss aggregation.
    values = torch.arange(1024, dtype=torch.float32, requires_grad=True)
    real = unpad_logits(values, fixed.padding_lengths[0])
    assert real.shape == (length,)
    (real * batch["loss_mask"].reshape(-1)).sum().backward()
    torch.testing.assert_close(
        values.grad[:length], batch["loss_mask"].reshape(-1).float(), rtol=0, atol=0
    )
    assert not values.grad[length:].any()


@pytest.mark.parametrize("token_id", [60, 61])
def test_missing_visual_metadata_is_not_relabelled_as_text(monkeypatch, token_id):
    engine = _engine(monkeypatch)
    batch = _proxy_batch()
    batch["input_ids"][0, 1] = token_id
    with pytest.raises(ValueError, match="mm_token_type_ids.*multimodal"):
        engine._prepare_mb_list(batch)
    engine.model.model.compute_3d_position_ids.assert_not_called()


def test_visual_payload_requires_types_even_without_marker_tokens(monkeypatch):
    engine = _engine(monkeypatch)
    batch = _proxy_batch()
    batch["multi_modal_input"] = [{"pixel_values": torch.ones(2, 4)}]
    with pytest.raises(ValueError, match="mm_token_type_ids.*multimodal"):
        engine._prepare_mb_list(batch)


def test_padding_visual_ids_do_not_make_text_batch_multimodal(monkeypatch):
    engine = _engine(monkeypatch)
    batch = _proxy_batch()
    batch["input_ids"][0, -1] = 60
    batch["attention_mask"][0, -1] = False
    engine._prepare_mb_list(batch)
    kwargs = engine.model.model.compute_3d_position_ids.call_args.kwargs
    assert not kwargs["mm_token_type_ids"].any()


def test_existing_types_are_preserved(monkeypatch):
    engine = _engine(monkeypatch)
    batch = _proxy_batch()
    types = torch.zeros_like(batch["input_ids"])
    batch["mm_token_type_ids"] = types
    engine._prepare_mb_list(batch)
    kwargs = engine.model.model.compute_3d_position_ids.call_args.kwargs
    assert kwargs["mm_token_type_ids"] is types
