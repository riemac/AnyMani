r"""EmbodimentPolicy 装配契约：tokenizer → backbone → heads。

本模块对应 `Research/总体/网络架构.md` 的整体架构骨架，是未来 teacher / student
共享的纯 PyTorch policy model 入口。

== 逻辑流程 ==

```text
raw grouped token fields
    └── tokenizer.py      : palm/joint/tip 分组 projection + 聚合 + mask/type_ids
          └── attention_bias.py : 可选生成 $b_{ij}$（第一版可为 None / 0）
                └── backbone.py : Encoder-only self-attention
                      └── heads.py : joint action / value / tip aux
```

== teacher 与 student 的统一 ==

- teacher / Specialist Policy：同一个 pre-made 拓扑下的 post-mutate 变体并行训练，
  token 数在该 teacher 内通常固定；mask 路径大多全有效，但代码保持 mask-ready。
- student / Unified Single Policy：跨 pre-made 拓扑、真实 Leap/Allegro URDF、
  get-zero 变体等，joint/tip 数可变；同一模型通过 padding + mask 处理。

因此 `EmbodimentPolicy` 不能写成“固定 N 个 joint 的 MLP”，也不能把 teacher 的固定
拓扑假设泄漏到模型结构里。固定拓扑只应体现在 batch 数据中，而不是类定义中。

== 与 rl_games / IL 的边界 ==

本模块不直接依赖 rl_games，也不写训练 loop。rl_games 的 network builder / adapter
应放在 `distill/rl/`，蒸馏训练 loop 应放在 `distill/il/`。二者都 import 这里的
纯模型定义，避免 teacher 和 student 各自复制网络实现。

TOAGENT:
    当前不写 `nn.Module` 真实装配代码。实现时应让 `EmbodimentPolicy.forward` 返回
    一个结构化输出（action mean/log_std/value/aux），而不是裸 tensor，避免后续
    RL 与 IL 入口层对字段语义产生分歧。
"""

# TODO: 定义 `EmbodimentPolicyOutput` 数据结构，字段至少包括：
#       `action_mean`、`action_log_std`、`value`、`aux_outputs`、`valid_masks`。

# TODO: 定义 `EmbodimentPolicy`，用 `EmbodimentPolicyCfg` 装配 tokenizer/backbone/heads。
#       forward 输入应是 tokenizer 可消费的分组 token batch，而不是固定扁平 obs。

# TODO: rl_games 需要扁平 obs/action 接口时，应在 `distill/rl/` 写 adapter，
#       不要让 `policy.py` 直接变成 rl_games 专用网络。
