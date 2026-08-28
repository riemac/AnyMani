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
from anymani.distill.rl.frozen_z import FrozenZProvider

HETEROGENEOUS_N000_FRAME_DIM = 52
"""N000 current frame：`q16 + target16 + last_action16 + tip_contact4`。"""

HETEROGENEOUS_ROUTE_DIM = 1 + CANONICAL_JOINT_COUNT
"""不进入 actor projection 的 `asset_row1 + active_mask16` metadata。"""

HETEROGENEOUS_MASKED_OBS_DIM = HETEROGENEOUS_N000_FRAME_DIM + HETEROGENEOUS_ROUTE_DIM
"""RlGamesVecEnvWrapper 交付的 policy flat observation 总维度 69。"""

HETEROGENEOUS_CRITIC_OBS_DIM = 103
"""N000 privileged state 删除 ADR 48D 与 reward-release 1D 后的 central critic 维度。"""


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


class HeterogeneousN000MaskedPpoNetwork(network_builder.NetworkBuilder.BaseNetwork):
    r"""rl_games facade：N000 frame + manifest routing + frozen $Z$ → shared policy。"""

    def __init__(self, params: dict[str, Any], **kwargs: Any) -> None:
        r"""验证 action/observation schema并持有 checkpoint-persistent frozen provider。"""

        super().__init__()
        actions_num = int(kwargs.pop("actions_num"))
        input_shape = tuple(int(value) for value in kwargs.pop("input_shape"))
        self.value_size = int(kwargs.pop("value_size", 1))
        self.num_seqs = int(kwargs.pop("num_seqs", 1))
        if actions_num != CANONICAL_JOINT_COUNT:
            raise ValueError(f"heterogeneous masked PPO requires actions_num=16, got {actions_num}")
        if input_shape != (HETEROGENEOUS_MASKED_OBS_DIM,):
            raise ValueError(
                f"heterogeneous masked PPO expects flat obs {(HETEROGENEOUS_MASKED_OBS_DIM,)}, got {input_shape}"
            )
        provider = params.get("frozen_z_provider")
        if not isinstance(provider, FrozenZProvider):
            raise TypeError("heterogeneous masked PPO requires a FrozenZProvider")
        self.frozen_z_provider = provider  # nn.Module 子模块；buffers 自动进入 actor checkpoint
        policy_values = dict(params.get("heterogeneous_policy", {}))
        policy_values.setdefault("owner_feature_dim", 1)  # 只有 TIP contact bit 进入 owner task feature
        policy_values.setdefault("joint_feature_dim", 3)  # `[q/pi,target/pi,last_policy_action]`
        policy_values.setdefault("geometry_entity_width", provider.width)  # frozen Z width 128
        self.policy = EmbodimentPolicy(CanonicalPolicyCfg(**policy_values))
        self.actions_num = actions_num
        self.last_active_joint_mask: torch.Tensor | None = None  # masked Normal/PPO agent 的共享边界

    @property
    def anymani_identity(self) -> dict[str, Any]:
        r"""返回 checkpoint/run metadata 使用的 frozen provider identity。"""

        return self.frozen_z_provider.identity

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

        if obs.ndim != 2 or obs.shape[-1] != HETEROGENEOUS_MASKED_OBS_DIM:
            raise ValueError(f"heterogeneous actor obs must have shape [B,{HETEROGENEOUS_MASKED_OBS_DIM}]")
        batch_size = obs.shape[0]
        q_end = CANONICAL_JOINT_COUNT  # 16D $q_t/\pi$
        target_end = q_end + CANONICAL_JOINT_COUNT  # 16D $u_t/\pi$
        action_end = target_end + CANONICAL_JOINT_COUNT  # 16D previous raw policy action
        contact_end = action_end + 4  # thumb/index/middle/ring contact bits
        asset_row = obs[:, contact_end].round().long()  # `[B]` discrete lookup row
        observed_mask = obs[:, contact_end + 1 :] > 0.5  # `[B,16]` environment routing certificate
        static = self.frozen_z_provider.resolve(asset_row)  # 同步 gather Z/mask/graph
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
        owner_features[:, 17:21, 0] = obs[:, action_end:contact_end][:, [1, 2, 3, 0]]
        # env contact 是 thumb/index/middle/ring；owner axis 是 index/middle/ring/thumb。
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

    def forward(self, obs_dict: dict[str, torch.Tensor]):
        r"""执行 rl_games `(mu, logstd,value,states)` network contract。"""

        policy_input = self._build_policy_input(obs_dict["obs"])
        output = self.policy(policy_input)  # spatial Transformer + shared joint/value heads
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


__all__ = [
    "HETEROGENEOUS_CRITIC_OBS_DIM",
    "HETEROGENEOUS_MASKED_OBS_DIM",
    "HETEROGENEOUS_N000_FRAME_DIM",
    "HETEROGENEOUS_ROUTE_DIM",
    "HeterogeneousN000MaskedPpoBuilder",
    "HeterogeneousN000MaskedPpoNetwork",
    "register_heterogeneous_masked_network",
]
