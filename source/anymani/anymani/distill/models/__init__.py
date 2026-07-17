r"""AnyMani 策略网络模型层 —— 跨手型手内操作的具身 Token Transformer。

本包是 `distill` 子项目的**网络架构共享落点**（见 `distill/AGENTS.md`）：
expert / student / unified policy、morphology encoder、joint-centric token
encoder、tokenizer、adapter 统一在此定义，**不在 `rl/` 与 `il/` 中各自复制**。
`rl/` 与 `il/` 只是算法入口层，import 本包的纯 PyTorch 模块。

== 设计依据 ==

完整架构论证见 `Research/总体/网络架构.md`。核心收敛结论（当前为暂定，非最终）：

1. 动作空间为 joint space，输出为每个 revolute joint 的增量动作；
2. 输入端按语义类型分组编码（palm / joint / tip），各类用自己的 projection
   投影到统一隐空间 $\mathbb{R}^{D}$，**不强行共用一个原始 Linear**；
3. 投影后聚合为统一 token 序列，主干做 Encoder-only 双向 self-attention；
4. 输出端按 token 类型路由：仅 `JOINT` token 接 action head，
   `TIP` / `PALM` 不输出关节动作；
5. teacher（固定拓扑）与 student（可变拓扑）共用同一套 schema，
   teacher 只是 mask 全有效的退化情形。

== 模块职责（与 `网络架构.md` 小节对应）==

- `config.py`        : 所有 dataclass 配置集中地（逐行核对超参/结构开关的入口）
- `tokens.py`        : §1   TokenType / role 枚举 + palm/joint/tip 分组语义契约
- `tokenizer.py`     : §3-4 分组 projection + 聚合为统一 token 序列 + mask 组装
- `backbone.py`      : §5   Encoder-only self-attention 主干（mask-ready）
- `attention_bias.py`: §8   注意力 bias $b_{ij}$ 接口（默认 0，可插拔 edge/graph bias）
- `relations.py`     : token pair 的 edge feature 构造契约（teacher 默认 all-pairs dynamic SE(3)）
- `heads.py`         : §6   action / value / (tip) auxiliary 输出头
- `policy.py`        : 装配 tokenizer → backbone → heads 的 EmbodimentPolicy
- `geometry/`        : §3.4 mesh / link 几何特征**编码器**（可替换 adapter，最易膨胀）

== 与 `rl/geo_obs.py` 的边界 ==

mesh / 几何特征要分两件事，落点不同：

- **几何特征提取**（asset → 原始张量）：从 URDF / HandCfg / articulation
  metadata 抠出 l/w/h、mesh offset、tip 形状参数等。属 obs/资产侧，
  归 `rl/geo_obs.py`。
- **几何特征编码**（原始张量 → token embedding）：把上述静态量编码进
  tip / joint token，是**可学习的网络部件**，归本包 `geometry/`。

当前实现状态：`temporal_encoder.py` 已提供 GM tactile rotation 的可运行 causal TCN；
tokenizer / backbone / policy 等 joint-centric Transformer 模块仍处于设计契约阶段。新增实现时
应逐文件把已兑现 TODO 迁移为稳定科研注释，不得用某个模块已落地来推断整个模型栈均已完成。
"""

# NOTE: 暂不在此 re-export 具体符号。TCN 由 rl_games adapter 直接 import 模块路径；
#       joint-centric 模型公共 API 尚未冻结，避免 package root 过早暴露会频繁变动的接口。
__all__: list[str] = []
