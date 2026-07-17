r"""TODO: GM tactile rotation 的 GRU / causal TCN 时间编码器设计契约。

本文件属于 `distill.models`，未来承载可被 RL、IL 与 teacher-student 路线共同复用的纯
PyTorch temporal encoder。rl_games builder 只做 adapter，不在 `distill.rl` 复制网络本体。

两个 PPO 分支共享 20 Hz 单帧部署信号：

$$
x_t=
\left[
q_t/\pi,
u_t/\pi,
a_{t-1}^{policy},
c_t^{tip}
\right]
\in\mathbb{R}^{52}.
$$

GRU 分支每步接收一个 52D frame：

$$
h_t=\operatorname{GRU}(x_t,h_{t-1}).
$$

hidden size 第一版保持 256。rl_games `seq_length=30` 只定义训练时反向传播覆盖的 1.5 s；
运行时 hidden state 在 episode 内持续传递，并在 done 后清零。环境不再预先拼三帧 observation。

TCN 分支显式接收 30 个 causal frames：

$$
X_t
=
[x_{t-29},\ldots,x_t]
\in\mathbb{R}^{30\times52}.
$$

第一版沿用 AnyRotate 报告的三层 temporal-convolution 骨架：kernel `[9,5,5]`、stride
`[2,1,1]`、hidden channels `[64,64,64]`，最终投影为：

$$
z_t\in\mathbb{R}^{64}.
$$

单个第三层 temporal position 的理论 receptive field 只有 25 帧：

$$
1+(9-1)+2(5-1)+2(5-1)=25.
$$

因此不能只取 final temporal position。实现应使用 causal、latest-frame-aligned 的 stride-2
padding，保留第三层完整 temporal feature map；随后展平或汇聚所有 temporal positions，再
投影到 64D。这样各位置感受野的并集覆盖 30 个输入 frame。tensor test 必须逐一扰动
`x_{t-29},\ldots,x_t`，证明每帧都能影响 latent，且最新帧不会因 stride alignment 被遗漏。

policy MLP 同时读取当前 frame 与历史摘要：

$$
a_t
=
\pi_{MLP}
\left([x_t,z_t]\right).
$$

TCN 作为 end-to-end PPO actor 训练，不使用 AnyRotate 的 8D privileged-latent distillation。
64D 是当前历史摘要容量，不是每帧 observation 维度。后续只有在首版出现欠拟合或梯度不稳证据后，
才比较 residual dilated TCN；不要在第一轮同时改变 MDP 与 temporal backbone。

GRU hidden 为 256D，TCN latent 为 64D，因此首轮比较是两套 temporal actor package 的实用
对照，不是严格参数量匹配的单因素网络实验。两者的 policy MLP 都显式拼接当前 52D frame
与 temporal state。GRU 使用 rl_games：

```yaml
before_mlp: true
concat_input: false
concat_output: true
```

对应：

$$
h_t=\operatorname{GRU}(x_t,h_{t-1}),
\qquad
a_t=\operatorname{MLP}([h_t,x_t]).
$$

rl_games 的 `concat_input` 只在 MLP 先于 RNN 时拼入原始输入；它不是当前帧到 policy MLP
的旁路。参数量差异通过参数总数、FLOPs、rollout throughput 和推理 latency 显式报告，不在
首轮为追求等参数而压缩成熟 GRU baseline。

两种 encoder 的参数量、FLOPs、4096-env rollout throughput 与单策略前向 latency 都必须记录。
部署硬约束是单步控制周期不超过 50 ms。性能比较不能只看 reward，还需看真实净旋转、速度带
占用率、speed jitter、tip/non-tip contact、掉落与 real-time replay gait。

本模块当前实现 TCN branch；GRU branch 仍由 rl_games 原生 recurrent builder 承载，避免在
AnyMani 内复制其 sequence batching、done mask 与 hidden-state checkpoint contract。
"""

from __future__ import annotations

from typing import Literal, overload

import torch
import torch.nn as nn
import torch.nn.functional as F


class TactileTemporalConvEncoder(nn.Module):
    r"""把 30 个 52D tactile/proprioceptive frames 编码为 64D 历史摘要。

    输入按 oldest-to-latest 排列：

    $$
    X_t=[x_{t-29},\ldots,x_t]\in\mathbb R^{B\times30\times52}.
    $$

    网络采用 AnyRotate Appendix G 的三层骨架：kernel `[9,5,5]`、stride `[2,1,1]`、
    channels `[64,64,64]`。第一层前只在左侧复制一帧 oldest observation，使时间长度为
    $31$，随后三层 valid convolution 的长度依次为：

    $$
    30\xrightarrow{\text{left replicate }1}31
    \xrightarrow{k=9,s=2}12
    \xrightarrow{k=5,s=1}8
    \xrightarrow{k=5,s=1}4.
    $$

    第三层单个 position 的 receptive field 为 25 帧；四个 positions 的并集覆盖全部
    30 帧。实现保留完整 $[B,64,4]$ temporal map，展平为 256D 后投影到 64D：

    $$
    z_t=W_z\operatorname{vec}(H_t^{(3)})+b_z\in\mathbb R^{64}.
    $$

    不使用 final-position 截取或 mean pooling：前者会漏掉最早历史，后者会抹去相对时间
    位置。当前 frame $x_t$ 不在本模块重复拼接；rl_games actor adapter 负责构造
    $[x_t,z_t]$ 并交给共同 policy MLP。

    Args:
        frame_dim (int): 单帧 observation 维度，tactile rotation v1 固定为 52。
        latent_dim (int): 历史摘要维度，v1 固定为 64。
        hidden_channels (tuple[int, int, int]): 三层 Conv1d channel 数，v1 为 `(64,64,64)`。

    Raises:
        ValueError: frame/latent/channel 配置不是正整数，或 channel tuple 长度不是 3 时抛出。
    """

    history_length: int = 30
    """环境提供的 causal history 长度 $T=30$，在 20 Hz 下覆盖 1.5 s。"""

    stage_lengths: tuple[int, int, int] = (12, 8, 4)
    """三层 valid temporal convolution 的固定输出长度，供 tensor contract test 核验。"""

    def __init__(
        self,
        frame_dim: int = 52,
        latent_dim: int = 64,
        hidden_channels: tuple[int, int, int] = (64, 64, 64),
    ) -> None:
        r"""构造 causal temporal convolutions 与完整-map projection。"""

        super().__init__()
        if frame_dim <= 0 or latent_dim <= 0:
            raise ValueError(f"frame_dim and latent_dim must be positive, got {frame_dim=} and {latent_dim=}")
        if len(hidden_channels) != 3 or any(channel <= 0 for channel in hidden_channels):
            raise ValueError(f"hidden_channels must contain three positive widths, got {hidden_channels!r}")

        self.frame_dim = int(frame_dim)  # 单帧 feature 维度 $D$，生产配置为 52
        self.latent_dim = int(latent_dim)  # temporal latent 维度 $d_z$，生产配置为 64
        c1, c2, c3 = (int(channel) for channel in hidden_channels)  # 三层 hidden channels $[64,64,64]$

        # Conv1d 使用 channels-first 张量 `[B,D,T]`；padding 在 forward 中显式做 oldest replication。
        self.conv1 = nn.Conv1d(self.frame_dim, c1, kernel_size=9, stride=2, padding=0)  # $31\to12$
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=5, stride=1, padding=0)  # $12\to8$
        self.conv3 = nn.Conv1d(c2, c3, kernel_size=5, stride=1, padding=0)  # $8\to4$
        self.activation = nn.ReLU()  # AnyRotate Appendix G 的 temporal activation

        # 完整 map 为 `[B,c3,4]`；flatten 后保留 position identity，再投影到 $z_t\in\mathbb R^{d_z}$。
        self.projection = nn.Linear(c3 * self.stage_lengths[-1], self.latent_dim)  # `[B,4c_3] -> [B,d_z]`

    @overload
    def temporal_features(
        self,
        history: torch.Tensor,
        *,
        return_stage_lengths: Literal[False] = False,
    ) -> torch.Tensor: ...

    @overload
    def temporal_features(
        self,
        history: torch.Tensor,
        *,
        return_stage_lengths: Literal[True],
    ) -> tuple[torch.Tensor, tuple[int, int, int]]: ...

    def temporal_features(
        self,
        history: torch.Tensor,
        *,
        return_stage_lengths: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[int, int, int]]:
        r"""计算第三层完整 temporal feature map。

        Args:
            history (torch.Tensor): oldest-to-latest observation，形状 `[B,30,frame_dim]`。
            return_stage_lengths (bool): 是否同时返回运行时三层长度，供 contract/debug 使用。

        Returns:
            torch.Tensor | tuple: 默认返回 `[B,c_3,4]` temporal map；debug 模式返回
            `(temporal_map, (12,8,4))`。

        Raises:
            ValueError: 输入 rank、history length 或 frame dimension 不符合固定环境契约时抛出。
        """

        if history.ndim != 3:
            raise ValueError(f"TCN history must have shape [B,30,{self.frame_dim}], got {tuple(history.shape)}")
        if history.shape[1:] != (self.history_length, self.frame_dim):
            raise ValueError(
                f"TCN history must have per-sample shape {(self.history_length, self.frame_dim)}, "
                f"got {tuple(history.shape[1:])}"
            )

        features = history.transpose(1, 2)  # `[B,30,D] -> [B,D,30]`，Conv1d 的 channels-first 约定
        features = F.pad(features, (1, 0), mode="replicate")  # `[B,D,31]`，复制 oldest frame 到左侧一格
        stage1 = self.activation(self.conv1(features))  # `[B,c_1,12]`，receptive-field 起点间隔 2 帧
        stage2 = self.activation(self.conv2(stage1))  # `[B,c_2,8]`，不在右侧读取越界 frame
        stage3 = self.activation(self.conv3(stage2))  # `[B,c_3,4]`，四位置并集覆盖原始 30 帧
        lengths = (stage1.shape[-1], stage2.shape[-1], stage3.shape[-1])  # 实际长度链，防止 config 漂移
        if lengths != self.stage_lengths:
            raise RuntimeError(f"TCN stage lengths changed from {self.stage_lengths} to {lengths}")
        if return_stage_lengths:
            return stage3, lengths  # debug/test 路径保留精确 temporal geometry
        return stage3  # production forward 使用完整 `[B,c_3,4]` map

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        r"""把 causal history 编码为固定 64D latent。

        Args:
            history (torch.Tensor): `[B,30,frame_dim]`，最后一帧严格为当前 observation。

        Returns:
            torch.Tensor: 历史摘要 $z_t$，形状 `[B,latent_dim]`。
        """

        temporal_map = self.temporal_features(history)  # `[B,c_3,4]`，保留全部 temporal positions
        flattened = temporal_map.flatten(start_dim=1)  # `[B,4c_3]`，position-major 信息仍由列位置编码
        return self.projection(flattened)  # `[B,d_z]`，默认 `[B,64]`


__all__ = ["TactileTemporalConvEncoder"]
