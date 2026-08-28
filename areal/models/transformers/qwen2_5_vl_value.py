# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModel,
    Qwen2_5_VLPreTrainedModel,
)


class AReaLQwen2_5_VLForTokenClassification(Qwen2_5_VLPreTrainedModel):
    """Qwen2.5-VL backbone with one scalar value for every input token."""

    _checkpoint_conversion_mapping = {
        "^visual": "model.visual",
        r"^model(?!\.(language_model|visual))": "model.language_model",
    }

    def __init__(self, config):
        config.num_labels = 1
        super().__init__(config)
        self.num_labels = 1
        self.model = Qwen2_5_VLModel(config)
        self.score = nn.Linear(config.text_config.hidden_size, 1, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        rope_deltas: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        second_per_grid_ts: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple | TokenClassifierOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            mm_token_type_ids=mm_token_type_ids,
            cache_position=cache_position,
            second_per_grid_ts=second_per_grid_ts,
            **kwargs,
        )
        logits = self.score(outputs.last_hidden_state)
        if return_dict is False:
            return (logits, *outputs.to_tuple()[1:])
        return TokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["AReaLQwen2_5_VLForTokenClassification"]
