r"""rl_games network builders for AnyMani GM teacher training.

本模块是 `distill/rl` 的 adapter 层：它把 rl_games 期望的 flat observation /
continuous action 接口，转换为 AnyMani teacher 的轻量 token Transformer。

第一版网络故意保持“薄而真实”：

- 输入仍来自 `tasks/gm` 的 flat policy obs；
- adapter 按 joint-centric 语义拆成 `JOINT` tokens 和一个 `COMMAND` token；
- encoder 使用 PyTorch `TransformerEncoder`；
- actor head 只从 joint tokens 输出逐关节 action mean；
- value head 从所有 tokens 的 masked mean pooling 输出标量 value。

这不是 MLP baseline。它是完整 joint-token Transformer 的 first runnable slice，后续
可把 `distill/models/tokenizer.py`、`attention_bias.py`、`policy.py` 的正式结构逐步替换
进来，而不改变 rl_games builder contract。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from rl_games.algos_torch import model_builder, network_builder


@dataclass(frozen=True)
class GmFlatObsLayout:
    r"""GM flat observation 的拆分布局。

    当前 `tasks/gm/inhand_env_cfg.py` 中 policy obs 拼接顺序为：

    $$
    [q,\dot q,\Delta a_{t-1},q^{min},q^{max},c],
    $$

    其中每个 joint token 消费 5 个标量：
    $[q_i,\dot q_i,\Delta a_{i,t-1},q_i^{min},q_i^{max}]$，command token 消费
    `[axis_h,error_so3_h]` 六维。

    Args:
        joint_count (int): action joint 数，也等于 actor 输出维度。
        token_feature_dim (int): 每个 joint token 的 raw feature 维度，当前为 5。
        command_feature_dim (int): command token raw feature 维度，当前为 6。
    """

    joint_count: int
    token_feature_dim: int = 5
    command_feature_dim: int = 6

    @property
    def expected_obs_dim(self) -> int:
        r"""返回 flat policy obs 的期望维度。"""

        return self.joint_count * self.token_feature_dim + self.command_feature_dim


class AnyManiGmTransformerBuilder(network_builder.NetworkBuilder):
    r"""rl_games custom network builder for GM Transformer teacher.

    rl_games 的 `model_builder.register_network()` 会先构造 builder，再调用：

    ```python
    builder.load(params["network"])
    net = builder.build(name, actions_num=..., input_shape=...)
    ```

    因此本类只保存 YAML 中的 network params，并在 `build()` 时根据 env 的
    action/observation 维度创建真实 `nn.Module`。
    """

    def __init__(self, **kwargs):
        r"""初始化空 builder。rl_games 注册接口不会传入有效参数。"""

        super().__init__(**kwargs)
        self.params: dict[str, Any] = {}

    def load(self, params: dict[str, Any]):
        r"""保存 YAML 中的 network 配置。"""

        self.params = params

    def build(self, name: str, **kwargs) -> AnyManiGmTransformerNetwork:
        r"""按 rl_games 传入的 env 维度构造 Transformer 网络。

        Args:
            name (str): rl_games 传入的 network name；此处仅用于接口兼容。
            **kwargs: 包含 `actions_num`、`input_shape`、`value_size`、`num_seqs`。

        Returns:
            AnyManiGmTransformerNetwork: 满足 `continuous_a2c_logstd` contract 的网络。
        """

        _ = name
        return AnyManiGmTransformerNetwork(self.params, **kwargs)


class AnyManiGmTransformerNetwork(network_builder.NetworkBuilder.BaseNetwork):
    r"""First runnable GM teacher Transformer.

    输入 flat obs 被解释为 `N` 个 joint tokens 和 1 个 command token。Transformer
    encoder 在这些 tokens 上做双向 self-attention：joint tokens 可读 command，
    command 也可聚合当前 hand state。actor head 对每个 joint token 输出一个 raw
    rad delta action mean；value head 对 token 池化后输出 critic value。
    """

    def __init__(self, params: dict[str, Any], **kwargs):
        r"""构造网络模块。

        Args:
            params (dict[str, Any]): YAML `network` 字段。
            **kwargs: rl_games 传入的 env 维度信息。
        """

        super().__init__()
        actions_num = int(kwargs.pop("actions_num"))  # joint action 数，等于 DOF
        input_shape = kwargs.pop("input_shape")  # flat obs shape，例如 `(86,)`
        self.value_size = int(kwargs.pop("value_size", 1))  # critic 输出维度，通常为 1
        self.num_seqs = int(kwargs.pop("num_seqs", 1))  # 非 RNN 网络只保留接口字段

        obs_dim = int(input_shape[0])  # flat policy obs 维度
        tfm_cfg = params.get("transformer", {})  # AnyMani 自定义网络字段
        token_feature_dim = int(tfm_cfg.get("token_feature_dim", 5))  # 每 joint token raw 维度
        command_feature_dim = int(tfm_cfg.get("command_feature_dim", 6))  # command raw 维度
        self.layout = GmFlatObsLayout(actions_num, token_feature_dim, command_feature_dim)
        if obs_dim != self.layout.expected_obs_dim:
            raise ValueError(
                "GM Transformer obs/action layout mismatch: "
                f"obs_dim={obs_dim}, actions_num={actions_num}, expected={self.layout.expected_obs_dim}. "
                "Check tasks/gm observation order or transformer token_feature_dim."
            )

        embed_dim = int(tfm_cfg.get("embed_dim", 128))  # token hidden width $D$
        num_layers = int(tfm_cfg.get("num_layers", 3))  # encoder depth
        num_heads = int(tfm_cfg.get("num_heads", 4))  # attention heads
        dropout = float(tfm_cfg.get("dropout", 0.0))  # PPO 第一阶段默认 0，避免额外随机性

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim must be divisible by num_heads, got {embed_dim} and {num_heads}.")

        self.actions_num = actions_num  # 输出 action mean 的 joint 数
        self.embed_dim = embed_dim  # hidden dim $D$
        self.joint_projection = nn.Linear(token_feature_dim, embed_dim)  # raw joint feature -> token embedding
        self.command_projection = nn.Linear(command_feature_dim, embed_dim)  # command feature -> command token
        self.type_embedding = nn.Embedding(2, embed_dim)  # 0=JOINT, 1=COMMAND，最小 type id

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)  # `[B,T,D] -> [B,T,D]`
        self.final_norm = nn.LayerNorm(embed_dim)  # Pre-LN encoder 后的输出归一化
        self.action_head = nn.Linear(embed_dim, 1)  # 每个 joint token 输出一个 action mean
        self.value = nn.Linear(embed_dim, self.value_size)  # rl_games 会通过 get_value_layer 读取该层

        # fixed_sigma=True 时 rl_games 期望网络持有可广播的 logstd 参数。
        sigma_cfg = params.get("space", {}).get("continuous", {}).get("sigma_init", {})
        init_logstd = float(sigma_cfg.get("val", -1.5))  # 初始标准差 $\exp(-1.5)\approx0.22$
        self.logstd = nn.Parameter(torch.full((actions_num,), init_logstd, dtype=torch.float32))

        self._reset_parameters()

    def _reset_parameters(self):
        r"""初始化投影和 head，使第一版 PPO 数值更温和。"""

        for module in (self.joint_projection, self.command_projection, self.action_head, self.value):
            nn.init.xavier_uniform_(module.weight)  # Transformer 常用 Xavier 初始化
            nn.init.zeros_(module.bias)  # 零偏置，避免初始 action/value 偏移过大
        nn.init.normal_(self.type_embedding.weight, mean=0.0, std=1.0 / math.sqrt(self.embed_dim))

    def is_rnn(self) -> bool:
        r"""rl_games contract：当前网络不是 RNN。"""

        return False

    def get_default_rnn_state(self):
        r"""rl_games contract：非 RNN 无 recurrent state。"""

        return

    def get_aux_loss(self):
        r"""rl_games contract：第一版无辅助损失。"""

        return

    def get_value_layer(self):
        r"""rl_games contract：返回 value head，供部分工具 introspection。"""

        return self.value

    def _tokenize_flat_obs(self, obs: torch.Tensor) -> torch.Tensor:
        r"""把 flat obs 切成 joint tokens 和 command token。

        Args:
            obs (torch.Tensor): flat obs，形状 `[B, 5N+6]`。

        Returns:
            torch.Tensor: token embedding，形状 `[B, N+1, D]`。
        """

        bsz = obs.shape[0]  # batch size $B$
        n = self.layout.joint_count  # joint token 数 $N$
        q = obs[:, 0:n]  # `[B,N]`，关节角 rad
        qd = obs[:, n : 2 * n]  # `[B,N]`，关节速度 rad/s
        last_action = obs[:, 2 * n : 3 * n]  # `[B,N]`，上一动作 processed rad delta
        limits = obs[:, 3 * n : 5 * n].reshape(bsz, n, 2)  # `[B,N,2]`，soft limits rad
        command = obs[:, 5 * n : 5 * n + self.layout.command_feature_dim]  # `[B,6]`，axis + error_so3

        joint_features = torch.stack((q, qd, last_action, limits[..., 0], limits[..., 1]), dim=-1)  # `[B,N,5]`
        joint_tokens = self.joint_projection(joint_features)  # `[B,N,D]`
        command_token = self.command_projection(command).unsqueeze(1)  # `[B,1,D]`

        joint_type = torch.zeros(n, dtype=torch.long, device=obs.device)  # type id 0 = JOINT
        command_type = torch.ones(1, dtype=torch.long, device=obs.device)  # type id 1 = COMMAND
        type_ids = torch.cat((joint_type, command_type), dim=0).unsqueeze(0)  # `[1,N+1]`
        tokens = torch.cat((joint_tokens, command_token), dim=1)  # `[B,N+1,D]`
        return tokens + self.type_embedding(type_ids)  # 加 type embedding，保留 token 角色

    def forward(self, obs_dict: dict[str, torch.Tensor]):
        r"""rl_games forward contract。

        Args:
            obs_dict (dict[str, torch.Tensor]): rl_games 输入字典，必须包含 `obs`。

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
            `(mu, logstd, value, states)`，分别为 action mean、log sigma、critic value、RNN state。
        """

        obs = obs_dict["obs"]  # `[B, obs_dim]`，已由 rl_games norm_obs 处理
        tokens = self._tokenize_flat_obs(obs)  # `[B,N+1,D]`
        encoded = self.final_norm(self.encoder(tokens))  # `[B,N+1,D]`
        joint_tokens = encoded[:, : self.actions_num, :]  # `[B,N,D]`
        pooled = encoded.mean(dim=1)  # `[B,D]`，第一版无 padding，直接 mean pooling

        mu = self.action_head(joint_tokens).squeeze(-1)  # `[B,N]`，每个 joint 一个 action mean
        value = self.value(pooled)  # `[B,1]`，critic value
        logstd = self.logstd.expand_as(mu)  # `[B,N]`，fixed log sigma broadcast
        return mu, logstd, value, None


def register_anymani_rl_games_networks() -> None:
    r"""向 rl_games 注册 AnyMani 自定义网络。"""

    model_builder.register_network("anymani_gm_transformer", AnyManiGmTransformerBuilder)


__all__ = [
    "AnyManiGmTransformerBuilder",
    "AnyManiGmTransformerNetwork",
    "GmFlatObsLayout",
    "register_anymani_rl_games_networks",
]
