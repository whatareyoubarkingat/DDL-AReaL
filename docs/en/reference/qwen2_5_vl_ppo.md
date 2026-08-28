# Qwen2.5-VL PPO Compatibility

This page describes the AReaL compatibility layer for running PPO with a Qwen2.5-VL
actor and critic through FSDP and SGLang. The changes are framework compatibility code:
they do not contain machine addresses, storage paths, reward policy, or
application-specific workflow logic.

## Support Scope

The implementation covers three boundaries:

1. a scalar Qwen2.5-VL critic for FSDP;
1. Qwen-VL position and mRoPE compatibility with Transformers 5.3; and
1. capability-based image-placeholder normalization at the AReaL-to-SGLang request
   boundary; and
1. a shared tool-aware chat-template override for SGLang and the rollout tokenizer.

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
`Qwen2_5_VLModel`. The serialized config records `num_labels=1` and the custom
architecture name. Model selection still comes from `TrainEngineConfig.is_critic`; the
architecture string is metadata, not an automatic loader registration.

The FSDP loader chooses the model as follows:

- `is_critic=false`: load the normal image-to-text actor;
- `is_critic=true`: require `model_type=qwen2_5_vl` and load the scalar critic.

A critic can warm-start its backbone from an actor checkpoint. The scalar head is newly
initialized unless the source checkpoint already contains `score.weight`.

LoRA is not supported for any FSDP critic, including text-only critics. AReaL fails
before model loading when both `is_critic=true` and `use_lora=true` because the PEFT
wrapper uses a causal-LM task and the adapter-only saver does not persist the scalar
score head. Full-parameter critic training must use `use_lora=false`.

## Transformers Position Compatibility

Qwen-VL uses three-axis multimodal rotary positions. Transformers versions expose the
`mrope_section` setting through different attention attributes. AReaL reads the current
`rope_parameters` API first and falls back to the older `rope_scaling` API.

When a text-only Qwen-VL batch does not supply `position_ids`, the FSDP path derives
ordinary positions from the attention mask and materializes three identical axes in
contiguous storage. Padding positions remain zero. Processor-provided multimodal
positions are preserved.

## SGLang Image Placeholders

A Hugging Face Qwen-VL processor can expand one image placeholder into a consecutive run
of image patch tokens. SGLang processes `image_data` again, so sending both the expanded
run and the image would expand the placeholder twice.

For processors that expose an `image_token_id`, AReaL prepares a generation-only copy of
`input_ids` and collapses each consecutive image-token run to one placeholder. This
capability contract covers Qwen2-VL, Qwen2.5-VL, and Qwen3-VL image requests without
relying on a processor class name. AReaL then verifies that the number of placeholders
equals the number of images. The original expanded `ModelRequest.input_ids` remains
unchanged for log-probability and training alignment.

If no processor is available, AReaL cannot safely infer the image token ID. It emits a
`RuntimeWarning` and leaves the request tokens unchanged. Callers should provide the
processor for multimodal requests.

## Tool Chat-Template Alignment

Some Qwen2.5-VL snapshots ship a multimodal template that ignores `tools`, assistant
`tool_calls`, and tool results. A tool parser only parses generated output; it cannot
restore schemas that were omitted from the prompt. AReaL therefore supports two explicit
overrides that should point to the same shared Jinja file:

```yaml
sglang:
  chat_template: <TOOL_AWARE_CHAT_TEMPLATE.jinja>

rollout:
  agent:
    chat_template_path: <TOOL_AWARE_CHAT_TEMPLATE.jinja>
```

`sglang.chat_template` configures server-side request rendering.
`rollout.agent.chat_template_path` configures the AReaL data-proxy tokenizer used for
prompt token accounting and trajectory alignment. Setting only one side can produce
different prompt token IDs between serving and training. The path must be a readable,
non-empty UTF-8 file visible to every rollout node.

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

The critic receives the same token, attention, image, grid, and multimodal token-type
inputs as the actor. It returns one scalar logit per input token.

## Checkpoint and Cache Contract

Full-parameter critic checkpoints saved by AReaL can be reloaded through the AReaL FSDP
critic path when this compatibility code is installed and `is_critic=true` is set. The
checkpoint contains the Qwen2.5-VL backbone and `score.weight`.

The custom critic is not registered with a Transformers `AutoModel*` mapping and the
checkpoint does not publish remote `auto_map` code. The `architectures` string alone
does not make `AutoModelForTokenClassification.from_pretrained()` externally loadable.
Use the AReaL FSDP loader (or import the custom class explicitly); do not treat a critic
checkpoint as a standalone generic Hugging Face model.

vLLM and SGLang rollout servers expect a causal generation model. They must load the
actor checkpoint, not this scalar critic checkpoint. Critic serving through either
rollout backend is unsupported.

The critic accepts and forwards `past_key_values` to its backbone, but its structured
`TokenClassifierOutput` does not return cache state. AReaL PPO scores complete sequences
and explicitly uses `use_cache=false`; incremental cached decoding is outside the critic
contract.

## Targeted Validation

The CPU compatibility tests are:

```bash
pytest -q \
  tests/test_fsdp_qwen2_5_vl_critic.py \
  tests/test_qwen2_vl_transformers_compat.py \
  tests/test_sglang_vlm_request.py
```

These tests do not replace a one-step distributed FSDP PPO smoke with a real Qwen2.5-VL
checkpoint and image sample.
