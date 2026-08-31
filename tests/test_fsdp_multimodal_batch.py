# SPDX-License-Identifier: Apache-2.0

import torch

from areal.engine.fsdp_engine import _prepare_multimodal_forward_inputs


def test_multimodal_forward_inputs_are_not_kept_in_loss_mb():
    pixel_values = [torch.randn(2, 4), torch.randn(3, 4)]
    image_grid_thw = [torch.tensor([[1, 1, 2]]), torch.tensor([[1, 1, 3]])]
    mb = {
        "input_ids": torch.ones(5, dtype=torch.long),
        "multi_modal_input": [
            {
                "pixel_values": pixel_values[0],
                "image_grid_thw": image_grid_thw[0],
            },
            {
                "pixel_values": pixel_values[1],
                "image_grid_thw": image_grid_thw[1],
            },
        ],
    }
    padded_mb = dict(mb)

    _prepare_multimodal_forward_inputs(mb, padded_mb)

    assert "multi_modal_input" not in mb
    assert "multi_modal_input" not in padded_mb
    assert "pixel_values" not in mb
    assert "image_grid_thw" not in mb
    assert torch.equal(padded_mb["pixel_values"], torch.cat(pixel_values, dim=0))
    assert torch.equal(padded_mb["image_grid_thw"], torch.cat(image_grid_thw, dim=0))


def test_multimodal_forward_inputs_fall_back_to_padded_mb():
    pixel_values = [torch.randn(2, 4)]
    mb = {"input_ids": torch.ones(2, dtype=torch.long)}
    padded_mb = {
        "input_ids": torch.ones(2, dtype=torch.long),
        "multi_modal_input": [{"pixel_values": pixel_values[0]}],
    }

    _prepare_multimodal_forward_inputs(mb, padded_mb)

    assert "multi_modal_input" not in mb
    assert "multi_modal_input" not in padded_mb
    assert "pixel_values" not in mb
    assert torch.equal(padded_mb["pixel_values"], torch.cat(pixel_values, dim=0))


def test_multimodal_forward_inputs_skip_text_only_samples():
    pixel_values = torch.randn(2, 4)
    image_grid_thw = torch.tensor([[1, 1, 2]])
    video_grid_thw = torch.tensor([[2, 1, 2]])
    mb = {
        "input_ids": torch.ones(4, dtype=torch.long),
        "multi_modal_input": [
            None,
            {
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
            },
            None,
        ],
    }
    padded_mb = dict(mb)

    _prepare_multimodal_forward_inputs(mb, padded_mb)

    assert "multi_modal_input" not in mb
    assert "multi_modal_input" not in padded_mb
    assert torch.equal(padded_mb["pixel_values"], pixel_values)
    assert torch.equal(padded_mb["image_grid_thw"], image_grid_thw)
    assert torch.equal(padded_mb["video_grid_thw"], video_grid_thw)


def test_multimodal_forward_inputs_accept_all_text_samples():
    mb = {
        "input_ids": torch.ones(2, dtype=torch.long),
        "multi_modal_input": [None, None],
    }
    padded_mb = dict(mb)

    _prepare_multimodal_forward_inputs(mb, padded_mb)

    assert "multi_modal_input" not in mb
    assert "multi_modal_input" not in padded_mb
    assert "pixel_values" not in padded_mb
    assert "image_grid_thw" not in padded_mb
    assert "video_grid_thw" not in padded_mb
