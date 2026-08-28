# Qwen2.5-VL PPO 兼容说明

本文说明 AReaL 为 Qwen2.5-VL Actor/Critic、FSDP 和 SGLang PPO 链路增加的兼容层。
这些修改属于通用框架兼容代码，不包含机器地址、存储路径、Reward 策略或具体应用工作流。

## 支持范围

本次实现覆盖三个边界：

1. 为 FSDP 增加 Qwen2.5-VL 标量 Critic；
1. 兼容 Transformers 5.3 下 Qwen-VL 的 position-id 和 mRoPE；
1. 在 AReaL 到 SGLang 的请求边界，按能力规范化图片占位 token。

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

两者都使用 Qwen2.5-VL 主干，因此能够理解相同的图片、Prompt 和已生成前缀；但它们是两套独立模型， 输出头和优化目标不同：

```text
Actor： 隐藏状态 -> 词表 logits -> 下一个 token 的分布
Critic：隐藏状态 -> 标量 score  -> 每个 token 的 Value
```

`AReaLQwen2_5_VLForTokenClassification` 在 `Qwen2_5_VLModel` 后增加一个不带 bias 的 标量 `score`
头。保存配置会记录 `num_labels=1` 和自定义 architecture 名称。模型选择仍由 `TrainEngineConfig.is_critic`
决定；architecture 字符串只是元数据，并不等同于自动加载器注册。

FSDP 加载器按以下规则选择模型：

- `is_critic=false`：加载标准图文 Actor；
- `is_critic=true`：要求 `model_type=qwen2_5_vl`，加载标量 Critic。

Critic 可以使用 Actor checkpoint 初始化主干。如果源 checkpoint 不含 `score.weight`，标量头会单独初始化。

任何 FSDP Critic（包括纯文本 Critic）都不支持 LoRA。当 `is_critic=true` 且 `use_lora=true` 时，AReaL
会在加载模型前失败，因为 PEFT 包装器使用 causal-LM task， 而 adapter-only saver 不会保存标量 score 头。全参数 Critic
训练必须设置 `use_lora=false`。

## Transformers Position 兼容

Qwen-VL 使用三轴多模态旋转位置编码。不同 Transformers 版本分别通过 `rope_parameters` 和旧版 `rope_scaling` 暴露
`mrope_section`。AReaL 现在优先读取新版接口，并兼容旧版接口。

当纯文本 Qwen-VL batch 没有提供 `position_ids` 时，FSDP 路径会根据 attention mask 构造普通位置，
并将三个相同的位置轴物化为连续存储；Padding 位置保持为零。Processor 已提供的多模态位置不会被覆盖。

## SGLang 图片占位符

Hugging Face 的 Qwen-VL Processor 可能把一个图片占位符展开成一段连续的图片 patch token。 SGLang 收到
`image_data` 后还会再次处理图片；如果同时传入已展开 token，图片占位符就会被重复展开。

对于提供 `image_token_id` 的 Processor，AReaL 会生成一份仅用于推理请求的 `input_ids` 副本， 把每段连续图片 token
折叠为一个占位符。该能力契约不依赖 Processor 类名，统一覆盖 Qwen2-VL、 Qwen2.5-VL 和 Qwen3-VL 图片请求。AReaL
随后检查占位符数量与图片数量是否一致。训练和 logprob 对齐仍使用原始展开后的 `ModelRequest.input_ids`。

如果没有 Processor，AReaL 无法安全推断图片 token ID，会发出 `RuntimeWarning` 并保持请求 token 不变。多模态请求应始终提供
Processor。

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
  type: value
  max_length: 65536
```

Critic 接收与 Actor 相同的 token、attention、图片、网格和多模态 token-type 输入，并为每个输入 token 返回一个标量 logit。

## Checkpoint 与缓存契约

AReaL 保存的全参数 Critic checkpoint，可以在安装本兼容代码且设置 `is_critic=true` 时， 通过 AReaL FSDP Critic
路径重新加载。Checkpoint 包含 Qwen2.5-VL 主干和 `score.weight`。

自定义 Critic 没有注册到 Transformers 的 `AutoModel*` 映射，checkpoint 也没有发布远程 `auto_map` 代码。仅有
`architectures` 字符串，不能保证 `AutoModelForTokenClassification.from_pretrained()` 从外部自动加载。应使用
AReaL FSDP 加载器， 或显式导入自定义类；不要把 Critic checkpoint 当作独立、通用的 Hugging Face 模型。

vLLM 和 SGLang rollout server 需要 causal generation 模型，因此必须加载 Actor checkpoint， 不能加载该标量
Critic checkpoint。两种 rollout 后端都不支持 Critic serving。

Critic 会接收 `past_key_values` 并转发给主干，但结构化 `TokenClassifierOutput` 不返回缓存状态。 AReaL PPO
对完整序列打分，并显式设置 `use_cache=false`；增量缓存解码不属于 Critic 支持契约。

## 定向验证

CPU 兼容测试命令如下：

```bash
pytest -q \
  tests/test_fsdp_qwen2_5_vl_critic.py \
  tests/test_qwen2_vl_transformers_compat.py \
  tests/test_sglang_vlm_request.py
```

这些测试不能替代使用真实 Qwen2.5-VL checkpoint 和图片样本执行的一步分布式 FSDP PPO smoke。
