from types import SimpleNamespace

import pytest
import torch

from areal.engine.fsdp_engine import (
    _compute_qwen_vl_position_ids,
    _fallback_qwen_vl_position_ids,
)
from areal.models.transformers.qwen2_vl import _get_mrope_section


def test_text_only_qwen_vl_position_ids_use_three_identical_axes():
    attention_mask = torch.tensor(
        [[True, True, False, False], [True, True, True, False]]
    )

    position_ids = _fallback_qwen_vl_position_ids(attention_mask)

    expected = torch.tensor([[0, 1, 0, 0], [0, 1, 2, 0]])
    assert position_ids.shape == (3, 2, 4)
    assert position_ids.dtype == torch.long
    assert position_ids.is_contiguous()
    assert 0 not in position_ids.stride()
    torch.testing.assert_close(position_ids[0], expected)
    torch.testing.assert_close(position_ids[1], expected)
    torch.testing.assert_close(position_ids[2], expected)


def test_qwen_vl_position_ids_clear_stale_multimodal_rope_delta():
    class FakeQwenVLModel:
        def __init__(self):
            self.rope_deltas = torch.tensor([[17]])
            self.call_kwargs = None

        def compute_3d_position_ids(self, **kwargs):
            assert self.rope_deltas is None
            self.call_kwargs = kwargs
            return None

    model = FakeQwenVLModel()
    input_ids = torch.tensor([[10, 11, 0]])
    attention_mask = torch.tensor([[True, True, False]])
    mm_token_type_ids = torch.zeros_like(input_ids)

    position_ids = _compute_qwen_vl_position_ids(
        model,
        input_ids=input_ids,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=attention_mask,
        mm_token_type_ids=mm_token_type_ids,
    )

    assert model.call_kwargs["inputs_embeds"] is None
    assert model.call_kwargs["past_key_values"] is None
    assert position_ids.shape == (3, 1, 3)
    torch.testing.assert_close(position_ids[:, 0], torch.tensor([[0, 1, 0]] * 3))


@pytest.mark.parametrize(
    "attention",
    [
        SimpleNamespace(rope_parameters={"mrope_section": [16, 24, 24]}),
        SimpleNamespace(rope_scaling={"mrope_section": [16, 24, 24]}),
    ],
)
def test_mrope_section_supports_transformers_attention_apis(attention):
    assert _get_mrope_section(attention) == [16, 24, 24]


@pytest.mark.parametrize(
    "attention",
    [
        SimpleNamespace(),
        SimpleNamespace(rope_parameters={}),
        SimpleNamespace(rope_scaling={}),
    ],
)
def test_mrope_section_rejects_missing_configuration(attention):
    with pytest.raises(ValueError, match="mrope_section"):
        _get_mrope_section(attention)
