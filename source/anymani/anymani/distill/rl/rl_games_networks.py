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

TODO(tactile rotation temporal adapters):
    新增两个训练 adapter，但网络本体归 `distill.models.temporal_encoder`：

    - GRU alias 消费 52D CurrentObs，hidden size 256，`seq_length=30`；
    - TCN alias 消费 `[30,52]` History30Obs，以 causal TCN 得到 64D latent，再拼当前 52D frame。

    GRU 需要 `before_mlp=true`、`concat_input=false`、`concat_output=true`，使当前 52D frame
    与 256D hidden 一起进入 policy MLP。TCN 同样把当前 52D frame 与 64D latent 拼接后进入
    同规格 MLP。首轮是完整实用 actor 对照，不强求 temporal state 或总参数量相等；必须报告
    参数量、FLOPs、rollout throughput 与推理 latency。

    两个 actor 都必须满足 rl_games continuous-logstd contract，并使用完全独立的 privileged
    central critic。`network.separate` 不是 asymmetric critic 开关；agent YAML 必须配置
    `central_value_config`，且环境 `critic` observation 必须实际映射到 rl_games `states`。

    critic 是 `[512,256,128]` feed-forward MLP，不使用 temporal history。actor 与 critic 不共享
    参数；部署导出只包含 actor。contract test 必须证明 privileged object/goal/contact/ADR state
    不会拼入 actor obs。

    PPO 共同采样预算：

    $$
    horizon\_length=30,
    \qquad
    minibatch\_size=30720,
    $$

    因为 4096 env 的 iteration batch 为 122880，并维持 4 个 minibatch；GRU 每个 minibatch
    恰含 1024 条完整 30-step sequence。TCN 沿用相同 transition 与 optimizer budget。
    `central_value_config` 必须另行显式写 `minibatch_size=30720`，rl_games 不自动继承 actor
    PPO config 的 minibatch。TCN 是非 RNN；其 30 帧来自 env observation，不依赖 rl_games
    `seq_length` 产生历史。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from anymani.distill.models.temporal_encoder import TactileTemporalConvEncoder
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


class AnyManiTactileTcnBuilder(network_builder.NetworkBuilder):
    r"""为 rl_games 构造 end-to-end tactile TCN actor。

    环境通过 Isaac Lab wrapper 保留 `[30,52]` sample shape；builder 不 flatten history，
    只在 `build()` 时把 env action/observation shape 注入网络。GRU 对照使用 rl_games
    原生 `actor_critic` builder，因此本类只服务 History30Obs TCN alias。
    """

    def __init__(self, **kwargs):
        r"""初始化空 builder；network 参数由后续 `load()` 注入。"""

        super().__init__(**kwargs)
        self.params: dict[str, Any] = {}  # YAML `network` 字段，build 前保持空 mapping

    def load(self, params: dict[str, Any]) -> None:
        r"""保存 TCN actor 的 YAML network 配置。

        Args:
            params (dict[str, Any]): 含 `space.continuous` 与 `tactile_tcn` 的 network mapping。
        """

        self.params = params  # rl_games builder 生命周期要求 load 与 build 分离

    def build(self, name: str, **kwargs) -> AnyManiTactileTcnNetwork:
        r"""按环境 shape 构造 TCN actor network。

        Args:
            name (str): rl_games network name，注册分派已使用，构造本体不再依赖。
            **kwargs: `actions_num`、`input_shape`、`value_size` 与 `num_seqs`。

        Returns:
            AnyManiTactileTcnNetwork: continuous-logstd actor network。
        """

        _ = name  # 名称只服务 model_builder registry，不进入网络数值图
        return AnyManiTactileTcnNetwork(self.params, **kwargs)


class AnyManiTactileTcnNetwork(network_builder.NetworkBuilder.BaseNetwork):
    r"""History30Obs 的 causal TCN + current-frame policy MLP。

    网络计算：

    $$
    z_t=\operatorname{TCN}(X_t)\in\mathbb R^{64},\qquad
    y_t=\operatorname{MLP}([x_t,z_t]),\qquad
    \mu_t=W_\mu y_t.
    $$

    `value` 是 rl_games `continuous_a2c_logstd` 接口要求的占位输出。训练 YAML 必须同时
    配置 `central_value_config` 与 `use_experimental_cv=false`，使真正的 value/advantage
    只来自 privileged `states`，并阻断该占位 head 的 value-loss gradient。
    """

    def __init__(self, params: dict[str, Any], **kwargs) -> None:
        r"""构造 temporal encoder、共享规格 policy MLP 与 continuous heads。

        Args:
            params (dict[str, Any]): YAML network 配置。
            **kwargs: rl_games 注入的 action/observation/value shape。

        Raises:
            ValueError: env observation 不是 `[30,frame_dim]`、action 数非正，或 MLP 配置非法时抛出。
        """

        super().__init__()
        self.actions_num = int(kwargs.pop("actions_num"))  # canonical revolute-joint action 数，single asset 为 16
        input_shape = tuple(int(value) for value in kwargs.pop("input_shape"))  # sample shape，必须为 `[30,52]`
        self.value_size = int(kwargs.pop("value_size", 1))  # rl_games value tensor 最后一维，通常为 1
        self.num_seqs = int(kwargs.pop("num_seqs", 1))  # 非 RNN，仅保留 BaseNetwork contract 字段
        if self.actions_num <= 0:
            raise ValueError(f"actions_num must be positive, got {self.actions_num}")

        tcn_cfg = params.get("tactile_tcn", {})  # AnyMani 自定义 TCN 与 actor MLP 字段
        frame_dim = int(tcn_cfg.get("frame_dim", 52))  # 部署单帧 $x_t\in\mathbb R^{52}$
        latent_dim = int(tcn_cfg.get("latent_dim", 64))  # 历史摘要 $z_t\in\mathbb R^{64}$
        hidden_channel_values = tuple(int(value) for value in tcn_cfg.get("hidden_channels", (64, 64, 64)))
        if len(hidden_channel_values) != 3:
            raise ValueError(f"tactile_tcn.hidden_channels must contain three widths, got {hidden_channel_values!r}")
        hidden_channels = (
            hidden_channel_values[0],
            hidden_channel_values[1],
            hidden_channel_values[2],
        )  # 精确三元 tuple 类型，对应三层 temporal convolution
        kernels = tuple(int(value) for value in tcn_cfg.get("kernels", (9, 5, 5)))  # YAML 可复现声明
        strides = tuple(int(value) for value in tcn_cfg.get("strides", (2, 1, 1)))  # YAML 可复现声明
        if kernels != (9, 5, 5) or strides != (2, 1, 1):
            raise ValueError(
                "TactileTemporalConvEncoder v1 fixes kernels/strides to [9,5,5]/[2,1,1]; "
                f"got kernels={kernels!r}, strides={strides!r}."
            )
        expected_shape = (TactileTemporalConvEncoder.history_length, frame_dim)  # `[30,52]` env contract
        if input_shape != expected_shape:
            raise ValueError(f"tactile TCN input shape must be {expected_shape}, got {input_shape}")

        mlp_units = tuple(int(value) for value in tcn_cfg.get("mlp_units", (512, 256, 128)))
        if not mlp_units or any(unit <= 0 for unit in mlp_units):
            raise ValueError(f"mlp_units must contain positive widths, got {mlp_units!r}")
        activation_name = str(tcn_cfg.get("activation", "elu"))  # GRU/TCN 共同 policy activation
        activation_type = _policy_activation_type(activation_name)  # 每层独立实例，避免共享 module state

        self.frame_dim = frame_dim  # 当前 frame 维度，用于 forward shape 校验和 latest-frame slice
        self.temporal_encoder = TactileTemporalConvEncoder(
            frame_dim=frame_dim,
            latent_dim=latent_dim,
            hidden_channels=hidden_channels,
        )  # `[B,30,52] -> [B,64]`

        # 两个 temporal actor 都采用 `[512,256,128]` ELU policy MLP；输入只因 temporal state 容量不同。
        layers: list[nn.Module] = []  # `[Linear, activation, ...]`，按配置显式展开
        input_dim = frame_dim + latent_dim  # `[x_t,z_t]`，默认 $52+64=116$
        for hidden_dim in mlp_units:
            layers.append(nn.Linear(input_dim, hidden_dim))  # 当前 hidden affine map
            layers.append(activation_type())  # ELU 无 trainable state，但仍为每层创建独立 module
            input_dim = hidden_dim  # 下一层输入宽度
        self.policy_mlp = nn.Sequential(*layers)  # `[B,116] -> [B,128]`
        self.mu = nn.Linear(input_dim, self.actions_num)  # actor mean `[B,128] -> [B,16]`
        self.value = nn.Linear(input_dim, self.value_size)  # 接口占位；central critic 才参与 value learning

        sigma_cfg = params.get("space", {}).get("continuous", {}).get("sigma_init", {})
        init_logstd = float(sigma_cfg.get("val", -0.5))  # $\exp(-0.5)\approx0.61$ 的初始 policy std
        self.logstd = nn.Parameter(torch.full((self.actions_num,), init_logstd, dtype=torch.float32))
        self._reset_parameters()  # 对 Linear 采用与现有 GM adapter 一致的温和 Xavier 初始化

    def _reset_parameters(self) -> None:
        r"""初始化 policy/head 的 affine 参数，TCN convolution 使用 PyTorch 默认 Kaiming 初始化。"""

        for module in (*self.policy_mlp.modules(), self.mu, self.value):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)  # 保持初始 actor/value 激活方差可控
                nn.init.zeros_(module.bias)  # 零偏置避免初始关节方向产生系统偏移

    def is_rnn(self) -> bool:
        r"""rl_games contract：TCN history 属于 observation，不是 recurrent state。"""

        return False

    def get_default_rnn_state(self):
        r"""rl_games contract：非 RNN actor 无 hidden state。"""

        return

    def get_aux_loss(self):
        r"""rl_games contract：v1 TCN 不使用 distillation 或辅助损失。"""

        return

    def get_value_layer(self) -> nn.Linear:
        r"""返回接口占位 value head，供 rl_games introspection。"""

        return self.value

    def forward(
        self,
        obs_dict: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        r"""执行 causal TCN actor forward。

        Args:
            obs_dict (dict[str, torch.Tensor]): 必含 `[B,30,52]` 的 `obs`。

        Returns:
            tuple: `(mu, logstd, value, None)`，shape 分别为 `[B,16]`、`[B,16]`、
            `[B,1]` 与无 recurrent state。
        """

        history = obs_dict["obs"]  # `[B,30,52]`，oldest-to-latest causal window
        if history.ndim != 3 or history.shape[-1] != self.frame_dim:
            raise ValueError(f"TCN actor expected [B,30,{self.frame_dim}], got {tuple(history.shape)}")
        current_frame = history[:, -1, :]  # `[B,52]`，环境 history 最后一帧严格等于当前部署 observation
        temporal_latent = self.temporal_encoder(history)  # `[B,64]`，完整 temporal map 的投影摘要
        policy_input = torch.cat((current_frame, temporal_latent), dim=-1)  # `[B,116]=[x_t,z_t]`
        hidden = self.policy_mlp(policy_input)  # `[B,128]`，共同 policy MLP 的最终 hidden
        mu = self.mu(hidden)  # `[B,16]`，canonical joint action means
        logstd = self.logstd.expand_as(mu)  # `[B,16]`，fixed trainable log sigma
        value = self.value(hidden)  # `[B,1]`，接口占位，不参与 central-value 配置下的 value loss
        return mu, logstd, value, None


def _policy_activation_type(name: str) -> type[nn.Module]:
    r"""把 YAML activation 名映射到显式 PyTorch module 类型。

    Args:
        name (str): 当前支持 `elu`、`relu`、`tanh`。

    Returns:
        type[nn.Module]: 可无参构造的 activation module 类型。

    Raises:
        ValueError: activation 不在稳定白名单时抛出，避免 YAML typo 静默退化。
    """

    activations: dict[str, type[nn.Module]] = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}
    try:
        return activations[name.lower()]  # 大小写不影响配置语义
    except KeyError as error:
        raise ValueError(f"unsupported tactile policy activation: {name!r}") from error


def register_anymani_rl_games_networks() -> None:
    r"""向 rl_games 注册 AnyMani 自定义网络。"""

    model_builder.register_network("anymani_gm_transformer", AnyManiGmTransformerBuilder)
    model_builder.register_network("anymani_tactile_tcn", AnyManiTactileTcnBuilder)


__all__ = [
    "AnyManiGmTransformerBuilder",
    "AnyManiGmTransformerNetwork",
    "AnyManiTactileTcnBuilder",
    "AnyManiTactileTcnNetwork",
    "GmFlatObsLayout",
    "register_anymani_rl_games_networks",
]
