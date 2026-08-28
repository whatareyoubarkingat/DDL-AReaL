# Qwen2.5-VL PPO Compatibility

This page describes the AReaL compatibility layer for running PPO with a Qwen2.5-VL
actor and critic through FSDP and SGLang. The changes are framework compatibility code:
they do not contain machine addresses, storage paths, reward policy, or
application-specific workflow logic.

## Support Scope

The implementation covers three boundaries:

1. a scalar Qwen2.5-VL critic for FSDP;
1. Qwen-VL position and mRoPE compatibility with Transformers 5.3; and
1. image-placeholder normalization at the AReaL-to-SGLang request boundary.

The actor path remains unchanged. A vision actor is still loaded with
`AutoModelForImageTextToText`; only `is_critic=true` selects the scalar value model.

The following stack has been validated together. It is a tested matrix, not a claim that
every listed version is a minimum requirement.

| Component        | Validated version      |
| ---------------- | ---------------------- |
| Python           | 3.11                   |
| PyTorch          | 2.9.1+cu129            |
| Transformers     | 5.3.0                  |
| SGLang           | 0.5.10.post1           |
| Model            | Qwen2.5-VL-7B-Instruct |
| Training backend | FSDP                   |

The vLLM rollout path was not validated by this compatibility work.

## Actor and Critic Roles

Both roles use a Qwen2.5-VL backbone so they can interpret the same image, prompt, and
generated prefix. They are separate model instances with different output heads and
optimization objectives:

```text
Actor:  hidden state -> vocabulary logits -> next-token distribution
Critic: hidden state -> scalar score       -> value for each token
```

`AReaLQwen2_5_VLForTokenClassification` adds a bias-free scalar `score` head to
`Qwen2_5_VLModel`. The serialized config records `num_labels=1`, the custom architecture
name, and the critic role so a saved critic can be reloaded without mistaking it for a
language-model checkpoint.

The FSDP loader chooses the model from `TrainEngineConfig.is_critic`:

- `false`: load the normal image-to-text actor;
- `true`: require `model_type=qwen2_5_vl` and load the scalar critic.

A critic can warm-start its backbone from an actor checkpoint. The scalar head is newly
initialized unless the source checkpoint already contains `score.weight`.

## Transformers Position Compatibility

Qwen-VL uses three-axis multimodal rotary positions. Transformers versions expose the
`mrope_section` setting through different attention attributes. AReaL now reads the
current `rope_parameters` API first and falls back to the older `rope_scaling` API.

When a text-only Qwen-VL batch does not supply `position_ids`, the FSDP path derives
ordinary positions from the attention mask and expands them into the required three-axis
layout. Padding positions remain zero. Processor-provided multimodal positions are
preserved.

## SGLang Image Placeholders

A Hugging Face Qwen2.5-VL processor expands one image placeholder into a consecutive run
of image patch tokens. SGLang processes `image_data` again, so sending both the expanded
run and the image would expand the placeholder twice.

For `Qwen2_5_VLProcessor` requests, AReaL therefore prepares a generation-only copy of
`input_ids` and collapses each consecutive image-token run to one placeholder. It then
verifies that the number of placeholders equals the number of images. The original
expanded `ModelRequest.input_ids` remains unchanged for log-probability and training
alignment. Other processor classes are not modified.

## Critic Configuration

A minimal critic configuration has the following shape. Parallel dimensions and token
limits must be adjusted to the available hardware and dataset.

```yaml
critic:
  backend: "fsdp:d4c2"
  is_critic: true
  path: <MODEL_OR_CRITIC_CHECKPOINT>
  use_lora: false
  dtype: bfloat16
  gradient_checkpointing: true
  mb_spec:
    max_tokens_per_mb: 65536

scheduler:
  type: local

train_dataset:
  path: <VALUE_HF_DATASET>
  type: value
  max_length: 65536
```

The critic receives the same `input_ids`, attention information, image tensors, and
