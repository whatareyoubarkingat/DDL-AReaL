# Qwen2.5-VL PPO 兼容说明

本文说明 AReaL 为 Qwen2.5-VL Actor/Critic、FSDP 和 SGLang PPO 链路增加的兼容层。
这些修改属于通用框架兼容代码，不包含机器地址、存储路径、Reward 策略或具体应用工作流。

## 支持范围

本次实现覆盖三个边界：

1. 为 FSDP 增加 Qwen2.5-VL 标量 Critic；
1. 兼容 Transformers 5.3 下 Qwen-VL 的 position-id 和 mRoPE；
1. 在 AReaL 到 SGLang 的请求边界规范化图片占位 token。

Actor 加载路径没有改变。视觉 Actor 仍使用 `AutoModelForImageTextToText`；只有 `is_critic=true` 才会选择标量
Value 模型。

以下是已经联合验证的环境矩阵，不代表所有组件的最低版本要求。

| 组件         | 已验证版本             |
| ------------ | ---------------------- |
| Python       | 3.11                   |
| PyTorch      | 2.9.1+cu129            |
| Transformers | 5.3.0                  |
| SGLang       | 0.5.10.post1           |
| 模型         | Qwen2.5-VL-7B-Instruct |
| 训练后端     | FSDP                   |

本次兼容工作没有验证 vLLM rollout 路径。

## Actor 与 Critic

两者都使用 Qwen2.5-VL 主干，因此能够理解相同的图片、Prompt 和已生成前缀；但它们是 两套独立模型，输出头和优化目标不同：

```text
Actor： 隐藏状态 -> 词表 logits -> 下一个 token 的分布
Critic：隐藏状态 -> 标量 score  -> 每个 token 的 Value
```

`AReaLQwen2_5_VLForTokenClassification` 在 `Qwen2_5_VLModel` 后增加一个不带 bias 的标量 `score`
头。保存配置会记录 `num_labels=1`、自定义 architecture 名称和 Critic 角色，从而避免重新加载时被误认成语言模型 checkpoint。

FSDP 加载器根据 `TrainEngineConfig.is_critic` 二选一：

- `false`：加载标准图文 Actor；
- `true`：要求 `model_type=qwen2_5_vl`，加载标量 Critic。

Critic 可以使用 Actor checkpoint 初始化主干。如果源 checkpoint 不含 `score.weight`， 标量头会单独初始化。

## Transformers Position 兼容

Qwen-VL 使用三轴多模态旋转位置编码。不同 Transformers 版本分别通过 `rope_parameters` 和旧版 `rope_scaling` 暴露
`mrope_section`。AReaL 现在优先读取 新版接口，并兼容旧版接口。

当纯文本 Qwen-VL batch 没有提供 `position_ids` 时，FSDP 路径会根据 attention mask
构造普通位置，再扩展成三轴布局；Padding 位置保持为零。Processor 已提供的多模态位置不会 被覆盖。

## SGLang 图片占位符

Hugging Face 的 Qwen2.5-VL Processor 会把一个图片占位符展开成一段连续的图片 patch token。SGLang 收到
`image_data` 后还会再次处理图片；如果同时传入已展开 token，图片占位符 就会被重复展开。

因此，对于 `Qwen2_5_VLProcessor` 请求，AReaL 会生成一份仅用于推理请求的 `input_ids` 副本，把每段连续图片 token
折叠为一个占位符，并检查占位符数量与图片数量 一致。训练和 logprob 对齐仍使用原始展开后的 `ModelRequest.input_ids`。其他 Processor
不会受到影响。

## Critic 配置

最小配置如下。并行维度和 token 上限必须根据实际硬件与数据调整。

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
```
