from types import SimpleNamespace

import pytest
import torch

from areal.engine.fsdp_engine import _fallback_qwen_vl_position_ids
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
