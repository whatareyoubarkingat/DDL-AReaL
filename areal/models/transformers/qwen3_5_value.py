# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 scalar critic and opt-in compact policy forward for full-sequence PPO."""

from __future__ import annotations

import copy

from torch import nn
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
    Qwen3_5PreTrainedModel,
)

from areal.models.transformers.qwen3_5_training import chunked_policy_statistics


class AReaLQwen3_5ForTokenClassification(Qwen3_5PreTrainedModel):
    def __init__(self, config):
        config = copy.deepcopy(config)
        config.num_labels = 1
        super().__init__(config)
        self.model = Qwen3_5Model(config)
        self.score = nn.Linear(config.text_config.hidden_size, 1, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def forward(self, input_ids=None, **kwargs):
        outputs = self.model(input_ids=input_ids, **kwargs)
        return TokenClassifierOutput(logits=self.score(outputs.last_hidden_state))


class AReaLQwen3_5ForPolicyTraining(Qwen3_5ForConditionalGeneration):
    """Retain native generation by default; only the training engine opts in."""

    def forward(
        self,
        input_ids=None,
        *,
        areal_policy_temperature=None,
        areal_head_chunk_size=256,
        **kwargs,
    ):
        if areal_policy_temperature is None:
            return super().forward(input_ids=input_ids, **kwargs)
        if input_ids is None:
            raise ValueError("policy statistics require exact input token IDs")
        if any(
            kwargs.get(k) is not None for k in ("pixel_values", "pixel_values_videos")
        ):
            raise ValueError("split-policy training does not accept image tensors")
        outputs = self.model(input_ids=input_ids, **kwargs)
        return chunked_policy_statistics(
            outputs.last_hidden_state,
            self.lm_head,
            input_ids.roll(-1, dims=-1),
            temperature=areal_policy_temperature,
            chunk_size=areal_head_chunk_size,
        )
