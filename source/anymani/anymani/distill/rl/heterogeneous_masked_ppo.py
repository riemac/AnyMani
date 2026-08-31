r"""N000 current-frame + frozen $Z$ 的 heterogeneous masked policy adapter。

环境 policy group 交付一个固定 69D flat tensor：

$$
o_t=[q_t/\pi,\ u_t/\pi,\ a_{t-1}^{policy},\ c_t^{tip},\ asset\_row,\ m],
$$

其中前 52D 是 N000 已验证的 deployable current frame，后 17D 只服务固定资产路由与 mask
一致性检查，不作为连续 actor feature。``FrozenZProvider`` 按 asset row 交付 `[21,128]` $Z$、
真实 owner/joint mask 和 graph；adapter 把三项逐关节状态送入 JOINT owners，把四个 contact bits
送入 TIP owners，再调用共享 ``EmbodimentPolicy``。

本文件只实现 rl_games network builder，不改变 masked Normal、PPO agent 或 central-value。后两者继续
复用 ``masked_ppo.py`` 与 rl_games 原生 asymmetric critic。
"""

from __future__ import annotations

from typing import Any

import torch
from rl_games.algos_torch import model_builder, network_builder

from anymani.distill.models.policy import (
    CANONICAL_JOINT_COUNT,
    CANONICAL_OWNER_COUNT,
    CanonicalPolicyCfg,
    EmbodimentPolicy,
    EmbodimentPolicyInput,
)
from anymani.distill.models.temporal_encoder import PerJointHistoryStackEncoder, PerJointTactileTemporalEncoder
from anymani.distill.rl.frozen_z import FrozenZProvider
from anymani.distill.rl.runtime.retained_geometry import RetainedGeometryProvider

HETEROGENEOUS_N000_FRAME_DIM = 52
"""N000 current frame：`q16 + target16 + last_action16 + tip_contact4`。"""

HETEROGENEOUS_ROUTE_DIM = 1 + CANONICAL_JOINT_COUNT
"""不进入 actor projection 的 `asset_row1 + active_mask16` metadata。"""

HETEROGENEOUS_MASKED_OBS_DIM = HETEROGENEOUS_N000_FRAME_DIM + HETEROGENEOUS_ROUTE_DIM
"""RlGamesVecEnvWrapper 交付的 policy flat observation 总维度 69。"""

HETEROGENEOUS_CRITIC_OBS_DIM = 103
"""N000 privileged state 删除 ADR 48D 与 reward-release 1D 后的 central critic 维度。"""

HETEROGENEOUS_N040_CRITIC_OBS_DIM = 127
"""N040 task-aware critic：103D state + active mask16 + morphology cell one-hot8。"""

HETEROGENEOUS_HISTORY_LENGTH = 30
"""20 Hz下覆盖1.5 s的逐JOINT raw history长度。"""

HETEROGENEOUS_JOINT_FRAME_DIM = 4
"""每JOINT每帧`[q/pi,target/pi,last_action,owner-tip-contact]`。"""

HETEROGENEOUS_N040_HISTORY_OBS_DIM = (
    HETEROGENEOUS_HISTORY_LENGTH * CANONICAL_JOINT_COUNT * HETEROGENEOUS_JOINT_FRAME_DIM
    + 2 * CANONICAL_JOINT_COUNT
    + HETEROGENEOUS_ROUTE_DIM
)
"""History1920 + limits32 + asset_row1 + active_mask16 = 1969。"""


class HeterogeneousN000MaskedPpoBuilder(network_builder.NetworkBuilder):
    r"""把 69D heterogeneous flat observation lower 成 owner/joint policy input。"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.params: dict[str, Any] = {}  # YAML network mapping 与 runtime provider

    def load(self, params: dict[str, Any]) -> None:
        r"""保存 builder 参数；实际 module 在 rl_games 注入 env shapes 后构造。"""

        self.params = params

    def build(self, name: str, **kwargs: Any) -> HeterogeneousN000MaskedPpoNetwork:
        r"""构造 frozen-$Z$ heterogeneous policy network。"""

        _ = name  # rl_games factory 名称已由 registry 分派
        return HeterogeneousN000MaskedPpoNetwork(self.params, **kwargs)


class HeterogeneousN040HistoryPpoBuilder(network_builder.NetworkBuilder):
    r"""把1969D History30 flat observation lower成冻结N040 policy-adapter input。"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.params: dict[str, Any] = {}

    def load(self, params: dict[str, Any]) -> None:
        r"""保存YAML与runtime retained provider；module在rl_games注入shape后构造。"""

        self.params = params

    def build(self, name: str, **kwargs: Any) -> HeterogeneousN000MaskedPpoNetwork:
        r"""构造History30 + frozen N040 graph policy adapter。"""

        _ = name
        return HeterogeneousN000MaskedPpoNetwork(self.params, history30=True, **kwargs)


class HeterogeneousN000MaskedPpoNetwork(network_builder.NetworkBuilder.BaseNetwork):
    r"""rl_games facade：N000 frame + manifest routing + frozen $Z$ → shared policy。"""

    def __init__(self, params: dict[str, Any], *, history30: bool = False, **kwargs: Any) -> None:
        r"""验证 action/observation schema并持有 checkpoint-persistent frozen provider。"""

        super().__init__()
        actions_num = int(kwargs.pop("actions_num"))
        input_shape = tuple(int(value) for value in kwargs.pop("input_shape"))
        self.value_size = int(kwargs.pop("value_size", 1))
        self.num_seqs = int(kwargs.pop("num_seqs", 1))
        if actions_num != CANONICAL_JOINT_COUNT:
            raise ValueError(f"heterogeneous masked PPO requires actions_num=16, got {actions_num}")
        expected_obs_dim = HETEROGENEOUS_N040_HISTORY_OBS_DIM if history30 else HETEROGENEOUS_MASKED_OBS_DIM
        if input_shape != (expected_obs_dim,):
            raise ValueError(
                f"heterogeneous masked PPO expects flat obs {(expected_obs_dim,)}, got {input_shape}"
            )
        frozen_provider = params.get("frozen_z_provider")
        retained_provider = params.get("retained_geometry_provider")
        if (frozen_provider is None) == (retained_provider is None):
            raise TypeError("heterogeneous masked PPO requires exactly one geometry provider")
        if frozen_provider is not None and not isinstance(frozen_provider, FrozenZProvider):
            raise TypeError("frozen_z_provider must be a FrozenZProvider")
        if retained_provider is not None and not isinstance(retained_provider, RetainedGeometryProvider):
            raise TypeError("retained_geometry_provider must be a RetainedGeometryProvider")
        self.frozen_z_provider = frozen_provider  # hash-Z infrastructure baseline；正式N040 route为None
        self.retained_geometry_provider = retained_provider  # frozen q-dependent N040；旧baseline为None
        provider = retained_provider if retained_provider is not None else frozen_provider
        assert provider is not None  # 上述exclusive provider gate已证明存在一个合法provider
        if history30 and retained_provider is None:
            raise TypeError("History30 N040 PPO requires a RetainedGeometryProvider")
        self.history30 = bool(history30)  # network identity与flat observation parser的固定分叉
        self.parallel_geometry_temporal = bool(params.get("parallel_geometry_temporal", False))
        self._geometry_stream: Any | None = None  # CUDA runtime私有stream类型按首次device惰性创建
        policy_values = dict(params.get("heterogeneous_policy", {}))
        policy_values.setdefault("owner_feature_dim", 1)  # TIP current contact进入owner task feature
        policy_values.setdefault("joint_feature_dim", 6 if history30 else 3)  # current frame + limit margins
        if history30:
            policy_values.setdefault("temporal_feature_dim", 64)  # 逐JOINT History30摘要宽度
        policy_values.setdefault("geometry_entity_width", provider.width)  # frozen Z width 128
        self.policy = EmbodimentPolicy(CanonicalPolicyCfg(**policy_values))
        if bool(params.get("compile_policy_adapter", False)):
            compiled_policy = torch.compile(self.policy, mode="reduce-overhead", fullgraph=True)
            self._policy_forward = compiled_policy.forward  # bound method不注册第二份nn.Module/state_dict namespace
        else:
            self._policy_forward = self.policy.forward
        temporal_widths = tuple(int(value) for value in params.get("temporal_hidden_channels", (64, 64, 64)))
        if len(temporal_widths) != 3:
            raise ValueError("temporal_hidden_channels must contain exactly three widths")
        temporal_hidden_channels = (temporal_widths[0], temporal_widths[1], temporal_widths[2])
        temporal_encoder_name = str(params.get("temporal_encoder", "tcn"))
        if history30 and temporal_encoder_name == "tcn":
            self.temporal_encoder = PerJointTactileTemporalEncoder(
                joint_count=CANONICAL_JOINT_COUNT,
                frame_dim=HETEROGENEOUS_JOINT_FRAME_DIM,
                latent_dim=self.policy.config.temporal_feature_dim,
                hidden_channels=temporal_hidden_channels,
            )  # 单一共享TCN；不实例化16个独立时序网络
        elif history30 and temporal_encoder_name == "stack_mlp":
            self.temporal_encoder = PerJointHistoryStackEncoder(
                joint_count=CANONICAL_JOINT_COUNT,
                frame_dim=HETEROGENEOUS_JOINT_FRAME_DIM,
                latent_dim=self.policy.config.temporal_feature_dim,
            )  # 固定时间列位置的shared ordered-stack MLP
        elif history30:
            raise ValueError(f"unsupported heterogeneous temporal_encoder={temporal_encoder_name!r}")
        else:
            self.temporal_encoder = None
        self.actions_num = actions_num
        self.last_active_joint_mask: torch.Tensor | None = None  # masked Normal/PPO agent 的共享边界

    @property
    def anymani_identity(self) -> dict[str, Any]:
        r"""返回 checkpoint/run metadata 使用的 frozen provider identity。"""

        return (
            self.retained_geometry_provider.identity
            if self.retained_geometry_provider is not None
            else self._require_frozen_provider().identity
        )

    def _require_frozen_provider(self) -> FrozenZProvider:
        r"""返回旧N000 hash provider；只有exclusive-provider gate后的baseline route可调用。"""

        if self.frozen_z_provider is None:
            raise RuntimeError("heterogeneous network has no frozen-Z provider")
        return self.frozen_z_provider

    def is_rnn(self) -> bool:
        r"""当前 heterogeneous spatial policy 无 recurrent state。"""

        return False

    def get_default_rnn_state(self):
        r"""非 RNN network 没有 hidden state。"""

        return

    def get_aux_loss(self):
        r"""基础设施 stage 不增加辅助学习目标。"""

        return

    def get_value_layer(self):
        r"""返回兼容 placeholder value head；正式 value 由 rl_games central critic 提供。"""

        return self.policy.value_head

    def _build_policy_input(self, obs: torch.Tensor) -> EmbodimentPolicyInput:
        r"""解析 N000 current frame，并按 asset row 注入 frozen $Z$/mask/graph。"""

        if self.history30:
            return self._build_history_policy_input(obs)

        if obs.ndim != 2 or obs.shape[-1] != HETEROGENEOUS_MASKED_OBS_DIM:
            raise ValueError(f"heterogeneous actor obs must have shape [B,{HETEROGENEOUS_MASKED_OBS_DIM}]")
        batch_size = obs.shape[0]
        q_end = CANONICAL_JOINT_COUNT  # 16D $q_t/\pi$
        target_end = q_end + CANONICAL_JOINT_COUNT  # 16D $u_t/\pi$
        action_end = target_end + CANONICAL_JOINT_COUNT  # 16D previous raw policy action
        contact_end = action_end + 4  # thumb/index/middle/ring contact bits
        asset_row = obs[:, contact_end].round().long()  # `[B]` discrete lookup row
        observed_mask = obs[:, contact_end + 1 :] > 0.5  # `[B,16]` environment routing certificate
        # Hash baseline只按asset row查表；正式N040接收physical q，单位从环境的$q/\pi$恢复为rad。
        static = (
            self.retained_geometry_provider.resolve(asset_row, obs[:, :q_end] * torch.pi)
            if self.retained_geometry_provider is not None
            else self._require_frozen_provider().resolve(asset_row)
        )
        if not torch.equal(observed_mask, static.joint_valid_mask):
            raise RuntimeError("environment active mask disagrees with frozen Z manifest provider")

        joint_features = torch.stack(
            (obs[:, :q_end], obs[:, q_end:target_end], obs[:, target_end:action_end]),
            dim=-1,
        )  # `[B,16,3]`，N000 current per-joint frame
        owner_features = torch.zeros(
            batch_size,
            CANONICAL_OWNER_COUNT,
            self.policy.config.owner_feature_dim,
            dtype=obs.dtype,
            device=obs.device,
        )  # `[B,21,1]`，PALM/JOINT task feature 默认为零
        # Task contact layout 与 canonical TIP owner 均固定为 `index,middle,ring,thumb`；保持逐列同序。
        owner_features[:, 17:21, 0] = obs[:, action_end:contact_end]
        return EmbodimentPolicyInput(
            owner_features=owner_features,
            joint_features=joint_features,
            owner_valid_mask=static.owner_valid_mask,
            joint_valid_mask=static.joint_valid_mask,
            shortest_path=static.shortest_path,
            parent_direction=static.parent_direction,
            child_direction=static.child_direction,
            asset_row=asset_row,
            geometry_entities=static.geometry_entities,
        )

    def _build_history_policy_input(self, obs: torch.Tensor) -> EmbodimentPolicyInput:
        r"""解析History30/limits/routing，并注入冻结N040与共享逐JOINT TCN。"""

        if obs.ndim != 2 or obs.shape[-1] != HETEROGENEOUS_N040_HISTORY_OBS_DIM:
            raise ValueError(
                f"heterogeneous N040 actor obs must have shape [B,{HETEROGENEOUS_N040_HISTORY_OBS_DIM}]"
            )
        batch_size = obs.shape[0]
        history_end = HETEROGENEOUS_HISTORY_LENGTH * CANONICAL_JOINT_COUNT * HETEROGENEOUS_JOINT_FRAME_DIM
        limits_end = history_end + 2 * CANONICAL_JOINT_COUNT
        history = obs[:, :history_end].reshape(
            batch_size,
            HETEROGENEOUS_HISTORY_LENGTH,
            CANONICAL_JOINT_COUNT,
            HETEROGENEOUS_JOINT_FRAME_DIM,
        )  # `[B,30,16,4]` oldest-to-latest
        limits = obs[:, history_end:limits_end].reshape(batch_size, CANONICAL_JOINT_COUNT, 2)  # `/pi`
        asset_row = obs[:, limits_end].round().long()
        observed_mask = obs[:, limits_end + 1 :] > 0.5
        latest = history[:, -1]  # `[B,16,4]`，当前q/target/action/contact旁路
        if self.retained_geometry_provider is None or self.temporal_encoder is None:
            raise RuntimeError("History30 network lacks retained geometry or temporal provider")
        q_rad = latest[:, :, 0] * torch.pi  # N040 physical q，单位rad；该tensor在current stream形成
        if self.parallel_geometry_temporal and obs.is_cuda:
            current_stream = torch.cuda.current_stream(obs.device)
            if self._geometry_stream is None:
                self._geometry_stream = torch.cuda.Stream(device=obs.device)
            geometry_stream = self._geometry_stream
            geometry_stream.wait_stream(current_stream)  # 等待obs/q在current stream就绪
            with torch.cuda.stream(geometry_stream):
                static = self.retained_geometry_provider.resolve(asset_row, q_rad)  # frozen N040 side stream
            temporal = self.temporal_encoder(history, observed_mask)  # trainable TCN留在current stream
            current_stream.wait_stream(geometry_stream)  # token fusion前等待N040完成
            for tensor in (
                static.geometry_entities,
                static.owner_valid_mask,
                static.joint_valid_mask,
                static.shortest_path,
                static.parent_direction,
                static.child_direction,
            ):
                tensor.record_stream(current_stream)  # allocator生命周期覆盖后续adapter消费
        else:
            static = self.retained_geometry_provider.resolve(asset_row, q_rad)
            temporal = self.temporal_encoder(history, static.joint_valid_mask)
        if not torch.equal(observed_mask, static.joint_valid_mask):
            raise RuntimeError("environment active mask disagrees with retained geometry provider")

        margin_lo = latest[:, :, 0] - limits[:, :, 0]  # $(q-q_{min})/\pi$，无量纲
        margin_hi = limits[:, :, 1] - latest[:, :, 0]  # $(q_{max}-q)/\pi$，无量纲
        joint_features = torch.cat((latest, margin_lo.unsqueeze(-1), margin_hi.unsqueeze(-1)), dim=-1)
        owner_features = torch.zeros(
            batch_size,
            CANONICAL_OWNER_COUNT,
            self.policy.config.owner_feature_dim,
            dtype=obs.dtype,
            device=obs.device,
        )
        owner_features[:, 17:21, 0] = latest[:, :4, 3]  # depth0四指复制值恢复TIP canonical order
        return EmbodimentPolicyInput(
            owner_features=owner_features,
            joint_features=joint_features,
            owner_valid_mask=static.owner_valid_mask,
            joint_valid_mask=static.joint_valid_mask,
            shortest_path=static.shortest_path,
            parent_direction=static.parent_direction,
            child_direction=static.child_direction,
            asset_row=asset_row,
            geometry_entities=static.geometry_entities,
            temporal_features=temporal,
        )

    def forward(self, obs_dict: dict[str, torch.Tensor]):
        r"""执行 rl_games `(mu, logstd,value,states)` network contract。"""

        policy_input = self._build_policy_input(obs_dict["obs"])
        output = self._policy_forward(policy_input)  # spatial Transformer + shared joint/value heads
        self.last_active_joint_mask = output.joint_valid_mask.detach()
        logstd = torch.where(
            output.joint_valid_mask,
            output.action_log_std,
            torch.zeros_like(output.action_log_std),
        )  # inactive sigma=1；custom masked model仍完全排除 ghost probability
        return output.action_mean, logstd, output.value, None


def register_heterogeneous_masked_network() -> None:
    r"""向 rl_games model registry 注册 N000-frame heterogeneous builder。"""

    model_builder.register_network("anymani_heterogeneous_n000_masked", HeterogeneousN000MaskedPpoBuilder)
    model_builder.register_network("anymani_heterogeneous_n040_history30", HeterogeneousN040HistoryPpoBuilder)


__all__ = [
    "HETEROGENEOUS_CRITIC_OBS_DIM",
    "HETEROGENEOUS_MASKED_OBS_DIM",
    "HETEROGENEOUS_N000_FRAME_DIM",
    "HETEROGENEOUS_N040_HISTORY_OBS_DIM",
    "HETEROGENEOUS_N040_CRITIC_OBS_DIM",
    "HETEROGENEOUS_ROUTE_DIM",
    "HeterogeneousN000MaskedPpoBuilder",
    "HeterogeneousN000MaskedPpoNetwork",
    "HeterogeneousN040HistoryPpoBuilder",
    "register_heterogeneous_masked_network",
]
