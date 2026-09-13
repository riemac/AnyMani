r"""MVP80 small-base action policy、zero-init bounded residual与独立structured critic。

Actor raw input保留$O\to Z\to X\to H$阶段：task交付逐JOINT current/History30、limits、owner contact与masks；
冻结N040交付$q$-dependent owner geometry $Z^e\in\mathbb R^{B\times21\times128}$。Base路径不要求复杂
contextual token先学会动作，而由共享逐JOINT TCN、local MLP和finger-first set pooling直接输出：

$$
\mu^{base}_{j,t}=f_{base}(x_{j,t},s_t).
$$

一层graph-biased actor backbone只输出有界修正：

$$
\mu_{j,t}=\mu^{base}_{j,t}+0.2\tanh r_{j,t}.
$$

Residual最后一层权重与bias初始化为0，因此初始策略逐元素严格等于base；第一轮PPO先更新residual head，随后
梯度自然进入global trunk。Critic与actor完全分参，使用两层Pre-LN graph context、privileged object/contact
及critic-only LayerNorm-c，最终每environment输出一个hand-level scalar value。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from .backbones.geometry_transformer import GraphBiasedTransformer, GraphBiasedTransformerCfg
from .temporal_encoder import PerJointRawHistoryStack, PerJointTactileTemporalEncoder

JOINT_COUNT = 16
TIP_COUNT = 4
OWNER_COUNT = 21
HISTORY_LENGTH = 30
GEOMETRY_WIDTH = 128
JOINT_FRAME_WIDTH = 5
LOCAL_WIDTH = 64
HISTORY_WIDTH = 32
FINGER_WIDTH = 48
HAND_WIDTH = 64
BASE_ACTION_LIMIT = 0.8
RESIDUAL_LIMIT = 0.2
FILM_LIMIT = 0.25
RECOVERY_LIMIT_MARGIN_RAD = 0.02
RECOVERY_OUTWARD_ACTION_MIN = 0.05


def recovery_exploration_contract(
    sigma_floor: float | None, *, max_log_std: float, sigma_mode: str = "global"
) -> dict[str, object] | None:
    r"""定义可选的限位恢复探索合同；尺度属于tanh之前的无量纲高斯。

    下限不能超过既有探索上限。该规则仅改变给定合法观测下的采样方差，不改变动作中心、
    $a/24$控制权限或任何MDP状态；与带新参数的conditional sigma分开研究。
    """
    if sigma_floor is None:
        return None
    if (
        sigma_mode != "global"
        or not math.isfinite(sigma_floor)
        or sigma_floor <= 0.0
        or not math.isfinite(max_log_std)
        or math.log(sigma_floor) > max_log_std
    ):
        raise ValueError("recovery sigma floor requires global sigma and a finite positive value within its ceiling")
    return {
        "rule": "tip-silent-target-limit-outward-previous-action-v1",
        "sigma_floor": float(sigma_floor),
        "limit_margin_rad": RECOVERY_LIMIT_MARGIN_RAD,
        "previous_outward_action_min": RECOVERY_OUTWARD_ACTION_MIN,
        "contact_scope": "all-valid-tips-zero",
        "target_and_limits_units": "rad-divided-by-pi",
        "mean": "unchanged",
        "effective_log_sigma": "gate?max(global_log_std,log(sigma_floor)):global_log_std",
        "maximum_log_sigma": float(max_log_std),
    }


class _PalmRotationGraphBiasedTransformer(GraphBiasedTransformer):
    r"""以直接Embedding查表形成掌旋PPO的静态图偏置。

    对关系桶$d_{ij}$和每头偏置表$W\in\mathbb R^{K\times H}$，生产公式为：

    $$
    b_{ijh}=W_{d_{ij},h}.
    $$

    直接查表与$\operatorname{onehot}(d_{ij})^TW$前向逐值相同，但不物化
    ``[B,21,21,K]`` FP32 one-hot activation。MVP80每个activation slice同时运行一层actor graph与
    两层共享同一bias的critic graph；该局部实现减少PPO update显存与SGEMM，而不改变参数、checkpoint key、
    attention公式或N040 retained encoder的独立compile合同。
    """

    def _graph_bias(
        self,
        shortest_path: torch.Tensor,
        parent_direction: torch.Tensor,
        child_direction: torch.Tensor,
    ) -> torch.Tensor:
        r"""返回三类关系Embedding之和，形状为`[H,N,N]`或`[B,H,N,N]`。"""

        matrices = (shortest_path, parent_direction, child_direction)  # 三种同轴离散图关系
        if any(matrix.ndim not in {2, 3} or matrix.shape[-2] != matrix.shape[-1] for matrix in matrices):
            raise ValueError("graph relation matrices must have square shape [N_E,N_E] or [B,N_E,N_E]")
        if parent_direction.shape != shortest_path.shape or child_direction.shape != shortest_path.shape:
            raise ValueError("all graph relation matrices must have identical shape")

        shortest = shortest_path.clamp(min=0, max=self.max_graph_distance)  # 无向图距离桶$d_{ij}^{sp}$
        parent = parent_direction.clamp(min=0, max=self.max_graph_distance)  # parent方向桶$d_{ij}^{pa}$
        child = child_direction.clamp(min=0, max=self.max_graph_distance)  # child方向桶$d_{ij}^{ch}$
        bias = (
            self.shortest_path_bias(shortest) + self.parent_direction_bias(parent) + self.child_direction_bias(child)
        )  # `[...,N,N,H]`，三类可学习head-wise scalar bias相加
        if bias.ndim == 3:
            return bias.permute(2, 0, 1).contiguous()  # 单结构共享图`[H,N,N]`
        return bias.permute(0, 3, 1, 2).contiguous()  # 逐sample异构图`[B,H,N,N]`


def _bool_mask(value: torch.Tensor, *, name: str, shape: tuple[int, ...]) -> torch.Tensor:
    r"""把task transport中的bool或0/1 mask规约为bool并验证shape。"""

    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if value.dtype == torch.bool:
        return value
    torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]  # 避免GPU mask检查触发host sync
        torch.all(torch.isfinite(value) & ((value == 0) | (value == 1))),
        f"{name} numeric transport must contain finite 0/1 values",
    )
    return value.to(dtype=torch.bool)


def _phase_clock_input(
    phase_clock: torch.Tensor | None,
    *,
    enabled: bool,
    batch_size: int,
    reference: torch.Tensor,
    role: str,
    validated: bool,
) -> torch.Tensor | None:
    r"""验证可选的 episode phase 编码，并保留 ``torch.func`` 的纯张量路径。

    phase clock 由 runtime 以

    $$
    \mathbf{p}_t=[\sin(\phi_t),\cos(\phi_t)]\in\mathbb{R}^{2}
    $$

    形成；模型只消费已经编码好的 `[B,2]` 有限 tensor，不在 forward 内推进或保存时钟。普通
    forward 对有限性做 tensor assertion；``_validated=True`` 是 functional/vmap 路径的边界，
    其输入已在 vmap 外完成有限性验证，因此这里不做 data-dependent Python 判断。

    Args:
        phase_clock (torch.Tensor | None): runtime 交付的 sin/cos 编码，形状 `[B,2]`。
        enabled (bool): 当前模型是否声明 phase 分支。
        batch_size (int): observation 的批大小。
        reference (torch.Tensor): 用来约束 dtype/device 的主 observation tensor。
        role (str): 错误信息中的模型角色名。
        validated (bool): 是否处于已完成外部检查的 functional/vmap 路径。

    Returns:
        torch.Tensor | None: 启用时返回原 phase tensor；禁用时返回 ``None`` 并保持旧路径。

    Raises:
        ValueError: 启用时 phase 缺失、shape、dtype 或 device 不符合合同。
        RuntimeError: 普通 forward 检测到非有限 phase 时由 tensor assertion 抛出。
    """

    if not enabled:
        return None  # 禁用分支不读取 phase，确保旧模型不注册也不消费新输入。
    if phase_clock is None:
        raise ValueError(f"{role} phase_clock_enabled=True requires phase_clock with shape [{batch_size},2]")
    expected_shape = (batch_size, 2)  # sin/cos 两列与 observation batch 轴严格对齐。
    if tuple(phase_clock.shape) != expected_shape:
        raise ValueError(
            f"{role} phase_clock must have shape {expected_shape}, got {tuple(phase_clock.shape)}"
        )
    if phase_clock.dtype != reference.dtype:
        raise ValueError(
            f"{role} phase_clock must have dtype {reference.dtype}, got {phase_clock.dtype}"
        )
    if phase_clock.device != reference.device:
        raise ValueError(
            f"{role} phase_clock must be on device {reference.device}, got {phase_clock.device}"
        )
    if not validated:
        torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]  # 保持 assertion 为纯 tensor 操作
            torch.isfinite(phase_clock).all(),
            f"{role} phase_clock must contain finite sin/cos values",
        )
    return phase_clock


def masked_mean(tokens: torch.Tensor, mask: torch.Tensor, *, dim: int) -> torch.Tensor:
    r"""沿entity轴计算有效项均值，ghost既不进分子也不进分母。"""

    if tokens.shape[:-1] != mask.shape:
        raise ValueError("masked_mean tokens/mask shapes disagree")
    weights = mask.unsqueeze(-1).to(dtype=tokens.dtype)
    return (tokens * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


def masked_max(tokens: torch.Tensor, mask: torch.Tensor, *, dim: int) -> torch.Tensor:
    r"""沿entity轴计算有效项max；全空slice返回0而不是dtype最小值。"""

    if tokens.shape[:-1] != mask.shape:
        raise ValueError("masked_max tokens/mask shapes disagree")
    has_value = mask.any(dim=dim, keepdim=True)  # `[... ,1,...]`，保留归约轴供安全broadcast
    empty_slice = ~has_value  # 没有有效JOINT/TIP的padding group
    safe_tokens = torch.where(empty_slice.unsqueeze(-1), torch.zeros_like(tokens), tokens)
    effective_mask = mask | empty_slice.expand_as(mask)  # 全空slice把全部零sentinels视为有效
    minimum = torch.finfo(tokens.dtype).min
    result = safe_tokens.masked_fill(~effective_mask.unsqueeze(-1), minimum).amax(dim=dim)
    return result  # 非空slice是原masked max；全空slice由zero sentinels得到精确0且零梯度


@dataclass(frozen=True)
class PalmRotationGeometry:
    r"""冻结N040 owner tokens、valid mask与同序离散运动学图。"""

    tokens: torch.Tensor  # `[B,21,128]` FP32 policy边界
    owner_valid: torch.Tensor  # bool`[B,21]`
    shortest_path: torch.Tensor  # long`[B,21,21]`
    parent_direction: torch.Tensor  # long`[B,21,21]`
    child_direction: torch.Tensor  # long`[B,21,21]`

    def __post_init__(self) -> None:
        r"""验证geometry、mask、graph shape与共同device。"""

        batch = self.tokens.shape[0]
        if self.tokens.shape != (batch, OWNER_COUNT, GEOMETRY_WIDTH):
            raise ValueError("palm-rotation geometry tokens must have shape [B,21,128]")
        object.__setattr__(
            self,
            "owner_valid",
            _bool_mask(self.owner_valid, name="geometry owner_valid", shape=(batch, OWNER_COUNT)),
        )
        graph_shape = (batch, OWNER_COUNT, OWNER_COUNT)
        if any(
            matrix.shape != graph_shape for matrix in (self.shortest_path, self.parent_direction, self.child_direction)
        ):
            raise ValueError("palm-rotation graph matrices must have shape [B,21,21]")
        tensors = (self.tokens, self.owner_valid, self.shortest_path, self.parent_direction, self.child_direction)
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("geometry tensors must share one device")


@dataclass(frozen=True)
class PalmRotationActorObservation:
    r"""Simulation-contact actor raw structured observation。"""

    jnt_current: torch.Tensor  # `[B,16,5]` q/u/a/own-contact/TIP-contact
    jnt_history: torch.Tensor  # `[B,30,16,5]`
    jnt_limits: torch.Tensor  # `[B,16,2]` qmin/qmax divided by pi
    owner_contact: torch.Tensor  # `[B,21,1]` binary
    jnt_valid: torch.Tensor  # bool`[B,16]`
    tip_valid: torch.Tensor  # bool`[B,4]`
    owner_valid: torch.Tensor  # bool`[B,21]`

    def __post_init__(self) -> None:
        r"""验证role/history axes、mask关系与共同device。"""

        batch = self.jnt_current.shape[0]
        expected = {
            "jnt_current": (batch, JOINT_COUNT, JOINT_FRAME_WIDTH),
            "jnt_history": (batch, HISTORY_LENGTH, JOINT_COUNT, JOINT_FRAME_WIDTH),
            "jnt_limits": (batch, JOINT_COUNT, 2),
            "owner_contact": (batch, OWNER_COUNT, 1),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"actor {name} must have shape {shape}")
        object.__setattr__(self, "jnt_valid", _bool_mask(self.jnt_valid, name="jnt_valid", shape=(batch, 16)))
        object.__setattr__(self, "tip_valid", _bool_mask(self.tip_valid, name="tip_valid", shape=(batch, 4)))
        object.__setattr__(
            self,
            "owner_valid",
            _bool_mask(self.owner_valid, name="owner_valid", shape=(batch, 21)),
        )
        expected_owner = torch.cat(
            (torch.ones(batch, 1, dtype=torch.bool, device=self.jnt_valid.device), self.jnt_valid, self.tip_valid),
            dim=-1,
        )
        torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
            torch.all(self.owner_valid == expected_owner), "actor owner masks disagree"
        )
        tensors = (
            self.jnt_current,
            self.jnt_history,
            self.jnt_limits,
            self.owner_contact,
            self.jnt_valid,
            self.tip_valid,
            self.owner_valid,
        )
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("actor observation tensors must share one device")

    @classmethod
    def from_task_dict(cls, observation: Mapping[str, torch.Tensor]) -> PalmRotationActorObservation:
        r"""从ManagerBased named policy group恢复actor输入。"""

        names = {"jnt_current", "jnt_history", "jnt_limits", "owner_contact", "jnt_valid", "tip_valid", "owner_valid"}
        missing = names - set(observation)
        if missing:
            raise KeyError(f"palm-rotation actor observation misses {sorted(missing)}")
        return cls(**{name: observation[name] for name in names})


@dataclass(frozen=True)
class PalmRotationCriticObservation:
    r"""Privileged critic raw structured observation。"""

    jnt_state: torch.Tensor  # `[B,16,4]` q/qd/u/a
    owner_contact: torch.Tensor  # `[B,21,2]` force N + bit
    obj: torch.Tensor  # `[B,1,15]`
    task: torch.Tensor  # `[B,1,8]`
    reward_release: torch.Tensor  # `[B,1]` cell-level lambda
    jnt_valid: torch.Tensor
    tip_valid: torch.Tensor
    owner_valid: torch.Tensor

    def __post_init__(self) -> None:
        r"""验证privileged roles、masks与device。"""

        batch = self.jnt_state.shape[0]
        expected = {
            "jnt_state": (batch, 16, 4),
            "owner_contact": (batch, 21, 2),
            "obj": (batch, 1, 15),
            "task": (batch, 1, 8),
            "reward_release": (batch, 1),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"critic {name} must have shape {shape}")
        object.__setattr__(self, "jnt_valid", _bool_mask(self.jnt_valid, name="critic jnt_valid", shape=(batch, 16)))
        object.__setattr__(self, "tip_valid", _bool_mask(self.tip_valid, name="critic tip_valid", shape=(batch, 4)))
        object.__setattr__(
            self,
            "owner_valid",
            _bool_mask(self.owner_valid, name="critic owner_valid", shape=(batch, 21)),
        )
        tensors = (
            self.jnt_state,
            self.owner_contact,
            self.obj,
            self.task,
            self.reward_release,
            self.jnt_valid,
            self.tip_valid,
            self.owner_valid,
        )
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("critic observation tensors must share one device")

    @classmethod
    def from_task_dict(cls, observation: Mapping[str, torch.Tensor]) -> PalmRotationCriticObservation:
        r"""从ManagerBased named critic group恢复privileged输入。"""

        names = {
            "jnt_state",
            "owner_contact",
            "obj",
            "task",
            "reward_release",
            "jnt_valid",
            "tip_valid",
            "owner_valid",
        }
        missing = names - set(observation)
        if missing:
            raise KeyError(f"palm-rotation critic observation misses {sorted(missing)}")
        return cls(**{name: observation[name] for name in names})


@dataclass(frozen=True)
class PalmRotationActorOutput:
    r"""Actor mean、共享log-standard-deviation与arm-specific机制分解。"""

    mean: torch.Tensor  # `[B,16]`，最终有界物理动作均值
    film_modulation_rms: torch.Tensor  # `[B,16]`，geometry对dynamic local hidden的RMS改变量
    log_std: torch.Tensor  # global标量，或可选条件/恢复探索的`[B,16]`有效log sigma
    base_mean: torch.Tensor | None = None  # residual/base arm；两种direct为None
    residual_mean: torch.Tensor | None = None  # residual/base arm；两种direct为None
    direct_mean: torch.Tensor | None = None  # 两种direct arm；residual/base为None


class PalmRotationResidualActor(nn.Module):
    r"""Small local base为主、one-block full-hand action residual为辅的共享actor。"""

    def __init__(
        self,
        *,
        residual_enabled: bool = True,
        initial_log_std: float = -0.5,
        max_log_std: float = -0.43,
        base_action_limit: float = BASE_ACTION_LIMIT,
        history_encoder: Literal["tcn", "raw_stack"] = "tcn",
        sigma_mode: Literal["global", "conditional"] = "global",
        recovery_sigma_floor: float | None = None,
    ) -> None:
        r"""构造History30路径、hierarchical pooling、base head与zero-init residual head。

        ``tcn``先把每个JOINT的$30\times5$序列压成32D；``raw_stack``保留全部150个固定lag标量，
        直接交给共享local FiLM-MLP。两者读取完全相同的actor observation，后者不添加object privilege。
        """

        super().__init__()
        if initial_log_std > max_log_std:
            raise ValueError("initial_log_std must not exceed the exploration ceiling")
        if not 0.0 < base_action_limit <= 1.0 - RESIDUAL_LIMIT:
            raise ValueError("base_action_limit plus residual limit must remain within physical action bounds")
        self.residual_enabled = bool(residual_enabled)
        self.phase_clock_enabled = False  # base/residual arm 永远不启用 phase；统一暴露只读语义标记供 runtime 查询。
        if sigma_mode not in {"global", "conditional"} or (sigma_mode == "conditional" and not residual_enabled):
            raise ValueError("conditional sigma requires a contextual Actor")
        self.sigma_mode = sigma_mode  # 共享标量或由同一关节上下文产生的条件探索尺度。
        self.max_log_std = float(max_log_std)  # $\sigma_{max}=e^{-0.43}\approx0.65$，匹配N000 early budget
        recovery_exploration_contract(recovery_sigma_floor, max_log_std=max_log_std, sigma_mode=sigma_mode)
        self.recovery_sigma_floor = recovery_sigma_floor  # 固定分布规则，无新增parameter/buffer或跨步隐藏状态。
        self.base_action_limit = float(base_action_limit)  # base保留80% authority，residual保留20%
        self.history_encoder_name = history_encoder  # checkpoint外run identity明确区分两种时间归纳偏置
        if history_encoder == "tcn":
            self.history_encoder: nn.Module = PerJointTactileTemporalEncoder(
                joint_count=JOINT_COUNT,
                frame_dim=JOINT_FRAME_WIDTH,
                latent_dim=HISTORY_WIDTH,
                hidden_channels=(32, 32, 32),
            )  # `[B,30,16,5] -> [B,16,32]` learned temporal compression
            history_width = HISTORY_WIDTH  # TCN latent宽度32
        elif history_encoder == "raw_stack":
            self.history_encoder = PerJointRawHistoryStack(
                joint_count=JOINT_COUNT,
                frame_dim=JOINT_FRAME_WIDTH,
            )  # `[B,30,16,5] -> [B,16,150]`，无learned bottleneck
            history_width = HISTORY_LENGTH * JOINT_FRAME_WIDTH  # $30\times5=150$
        else:
            raise ValueError(f"unknown palm-rotation history encoder: {history_encoder!r}")
        local_input_width = JOINT_FRAME_WIDTH + 2 + 2 + history_width  # current/limits/lag/q/history
        self.local_encoder = nn.Sequential(
            nn.LayerNorm(local_input_width),
            nn.Linear(local_input_width, LOCAL_WIDTH),
            nn.GELU(),
            nn.Linear(LOCAL_WIDTH, LOCAL_WIDTH),
        )
        self.joint_geometry_film = nn.Sequential(
            nn.LayerNorm(GEOMETRY_WIDTH),
            nn.Linear(GEOMETRY_WIDTH, 2 * LOCAL_WIDTH),
        )  # $Z_j^e\mapsto(\gamma_j,\beta_j)$，不与低维dynamic vector直接拼接
        nn.init.zeros_(self.joint_geometry_film[-1].weight)  # type: ignore[arg-type]
        nn.init.zeros_(self.joint_geometry_film[-1].bias)  # type: ignore[arg-type]
        self.tip_geometry_projection = nn.Linear(GEOMETRY_WIDTH, 32)
        self.finger_encoder = nn.Sequential(
            nn.LayerNorm(2 * LOCAL_WIDTH + 32 + 2),
            nn.Linear(2 * LOCAL_WIDTH + 32 + 2, FINGER_WIDTH),
            nn.GELU(),
            nn.Linear(FINGER_WIDTH, FINGER_WIDTH),
        )
        self.palm_geometry_projection = nn.Linear(GEOMETRY_WIDTH, 32)
        self.hand_encoder = nn.Sequential(
            nn.LayerNorm(2 * FINGER_WIDTH + 32 + 2),
            nn.Linear(2 * FINGER_WIDTH + 32 + 2, HAND_WIDTH),
            nn.GELU(),
            nn.Linear(HAND_WIDTH, HAND_WIDTH),
        )
        self.base_head = nn.Sequential(
            nn.LayerNorm(LOCAL_WIDTH + HAND_WIDTH),
            nn.Linear(LOCAL_WIDTH + HAND_WIDTH, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        nn.init.orthogonal_(self.base_head[-1].weight, gain=0.01)  # type: ignore[arg-type]
        nn.init.zeros_(self.base_head[-1].bias)  # type: ignore[arg-type]

        # Dynamic adapters只构造actor-specific$X^a$；冻结$Z^e$本身不被修改或纳入optimizer。
        self.geometry_adapter = nn.Linear(GEOMETRY_WIDTH, GEOMETRY_WIDTH)
        self.owner_contact_projection = nn.Linear(1, GEOMETRY_WIDTH, bias=False)
        self.palm_dynamic_projection = nn.Linear(HAND_WIDTH, GEOMETRY_WIDTH)
        self.joint_dynamic_projection = nn.Linear(LOCAL_WIDTH, GEOMETRY_WIDTH)
        self.tip_dynamic_projection = nn.Linear(FINGER_WIDTH, GEOMETRY_WIDTH)
        self.global_backbone = _PalmRotationGraphBiasedTransformer(
            GraphBiasedTransformerCfg(
                hidden_width=GEOMETRY_WIDTH,
                layers=1,
                attention_heads=4,
                feedforward_width=256,
                dropout=0.0,
            )
        )
        self.residual_head = nn.Sequential(
            nn.LayerNorm(GEOMETRY_WIDTH + LOCAL_WIDTH),
            nn.Linear(GEOMETRY_WIDTH + LOCAL_WIDTH, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)  # type: ignore[arg-type]
        nn.init.zeros_(self.residual_head[-1].bias)  # type: ignore[arg-type]
        self.global_log_std = nn.Parameter(torch.tensor(float(initial_log_std)))
        self.conditional_sigma_head = None
        if sigma_mode == "conditional":
            self.conditional_sigma_head = nn.Linear(GEOMETRY_WIDTH, 1)  # 所有关节共用头，不读任何资产ID。
            nn.init.zeros_(self.conditional_sigma_head.weight)
            nn.init.zeros_(self.conditional_sigma_head.bias)  # 初始每个关节恢复统一的sigma基线。

    @torch.no_grad()
    def project_exploration_parameters(self) -> None:
        r"""把trainable global $\log\sigma$投影到N000 early-budget探索上界。

        PPO仍可根据数据降低标准差；entropy或policy gradient只能把它恢复到$-0.43$，不能再次进入物理动作
        30%以上被clamp、deterministic mean随噪声训练而抖动的区域。
        """

        self.global_log_std.clamp_(max=self.max_log_std)
        if self.sigma_mode == "conditional":
            self.global_log_std.clamp_(min=math.log(0.05))  # 保持基线处于可学习区间。

    def recovery_exploration_mask(self, observation: PalmRotationActorObservation) -> torch.Tensor:
        r"""从动作前合法观测识别无TIP接触且仍向目标限位外推的真实关节。

        $u$是控制器目标而非物理$q$；两者可能存在跟踪误差。归一化目标与限位同为rad/pi，
        因此0.02 rad邻域先除以pi；上一动作保持无量纲。任一有效TIP接触就关闭整手的该规则。
        不把History30初始padding解释成持续失触，也不跨forward维护额外时钟。
        """
        if self.recovery_sigma_floor is None:
            return torch.zeros_like(observation.jnt_valid)
        tip_contact = observation.owner_contact[:, 17:21, 0] > 0.5
        no_tip = observation.tip_valid.any(dim=-1) & ~(tip_contact & observation.tip_valid).any(dim=-1)
        target = observation.jnt_current[..., 1]
        previous_action = observation.jnt_current[..., 2]
        margin = RECOVERY_LIMIT_MARGIN_RAD / math.pi
        lower_outward = (target <= observation.jnt_limits[..., 0] + margin) & (
            previous_action < -RECOVERY_OUTWARD_ACTION_MIN
        )
        upper_outward = (target >= observation.jnt_limits[..., 1] - margin) & (
            previous_action > RECOVERY_OUTWARD_ACTION_MIN
        )
        return observation.jnt_valid & no_tip[:, None] & (lower_outward | upper_outward)

    def _policy_log_std(
        self, contextual: torch.Tensor | None, observation: PalmRotationActorObservation
    ) -> torch.Tensor:
        r"""所有采样与PPO概率路径共同消费此处的有效log sigma。

        默认global仍返回原标量。恢复规则只在gate内设下限；下限绑定时对base log sigma的梯度为零，
        gate外保留原梯度。规则只依赖存储观测，使rollout与重算log-prob使用同一条件。
        """
        if self.sigma_mode == "global":
            if self.recovery_sigma_floor is None:
                return self.global_log_std  # 关闭新规则时保持原标量合同和数值路径。
            floor = self.global_log_std.new_tensor(math.log(self.recovery_sigma_floor))
            log_std = torch.where(
                self.recovery_exploration_mask(observation),
                torch.maximum(self.global_log_std, floor),
                self.global_log_std,
            )
            return torch.where(observation.jnt_valid, log_std, torch.zeros_like(log_std))
        if contextual is None or self.conditional_sigma_head is None:
            raise RuntimeError("conditional sigma requires contextual joint features")
        offset = self.conditional_sigma_head(contextual[:, 1:17]).squeeze(-1)
        log_std = (self.global_log_std + math.log(2.0) * torch.tanh(offset)).clamp(math.log(0.05), self.max_log_std)
        return torch.where(observation.jnt_valid, log_std, torch.zeros_like(log_std))  # ghost的辅助Normal取sigma=1。

    def _local_and_hand(
        self,
        observation: PalmRotationActorObservation,
        geometry: PalmRotationGeometry,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""形成dynamic-first FiLM local、per-finger与whole-hand summary。

        低维控制状态先形成$h_{t,j}^{dyn}$；冻结geometry只通过有界FiLM调制：

        $$
        h_{t,j}^{loc}
        =\left(1+0.25\tanh\gamma(Z_{t,j}^e)\right)\odot h_{t,j}^{dyn}
        +0.25\tanh\beta(Z_{t,j}^e).
        $$

        FiLM末层零初始化使初始$h^{loc}=h^{dyn}$，避免128维$Z^e$在训练起点覆盖$q,u,a$与History30。
        """

        joint_weight = observation.jnt_valid.unsqueeze(-1).to(dtype=observation.jnt_current.dtype)
        history = self.history_encoder(observation.jnt_history, observation.jnt_valid)  # `[B,16,32]`
        current = observation.jnt_current * joint_weight  # `[B,16,5]`
        limits = observation.jnt_limits * joint_weight  # `[B,16,2]`
        tracking_lag = current[..., 1:2] - current[..., 0:1]  # $(u-q)/\pi$
        span = (limits[..., 1:2] - limits[..., 0:1]).clamp_min(1.0e-6)
        normalized_q = 2.0 * (current[..., 0:1] - limits[..., 0:1]) / span - 1.0
        local_input = torch.cat((current, limits, tracking_lag, normalized_q, history), dim=-1)
        dynamic_local = self.local_encoder(local_input) * joint_weight  # $h^{dyn}$，`[B,16,64]`
        gamma_raw, beta_raw = self.joint_geometry_film(geometry.tokens[:, 1:17]).chunk(2, dim=-1)
        gamma = FILM_LIMIT * torch.tanh(gamma_raw)  # 有界multiplicative geometry modulation
        beta = FILM_LIMIT * torch.tanh(beta_raw)  # 有界additive geometry modulation
        local = ((1.0 + gamma) * dynamic_local + beta) * joint_weight  # $h^{loc}$，ghost严格为0
        film_modulation_rms = torch.linalg.vector_norm(local - dynamic_local, dim=-1) / math.sqrt(
            LOCAL_WIDTH
        )  # `[B,16]`；与sqrt(mean(square))逐值相同，零向量处autograd采用有限零次梯度

        # Canonical JOINT axis是depth-major；转成`[B,finger,depth,D]`后先在每根finger内pool。
        batch = local.shape[0]
        local_by_finger = local.reshape(batch, 4, 4, LOCAL_WIDTH).transpose(1, 2)
        mask_by_finger = observation.jnt_valid.reshape(batch, 4, 4).transpose(1, 2)
        finger_mean = masked_mean(local_by_finger, mask_by_finger, dim=2)
        finger_max = masked_max(local_by_finger, mask_by_finger, dim=2)
        joint_count = mask_by_finger.sum(dim=2, keepdim=True).to(dtype=local.dtype) / 4.0
        tip_geometry = self.tip_geometry_projection(geometry.tokens[:, 17:21])
        tip_contact = observation.owner_contact[:, 17:21]
        finger_input = torch.cat((finger_mean, finger_max, tip_geometry, tip_contact, joint_count), dim=-1)
        finger = self.finger_encoder(finger_input) * observation.tip_valid.unsqueeze(-1)  # `[B,4,48]`

        finger_mean_hand = masked_mean(finger, observation.tip_valid, dim=1)
        finger_max_hand = masked_max(finger, observation.tip_valid, dim=1)
        tip_count = observation.tip_valid.sum(dim=-1, keepdim=True).to(dtype=local.dtype) / 4.0
        palm_geometry = self.palm_geometry_projection(geometry.tokens[:, 0])
        palm_contact = observation.owner_contact[:, 0]
        hand_input = torch.cat((finger_mean_hand, finger_max_hand, palm_geometry, palm_contact, tip_count), dim=-1)
        hand = self.hand_encoder(hand_input)  # `[B,64]`
        return local, finger, hand, film_modulation_rms

    def forward(
        self,
        observation: PalmRotationActorObservation,
        geometry: PalmRotationGeometry,
    ) -> PalmRotationActorOutput:
        r"""输出base、bounded residual与最终masked Gaussian mean。"""

        if observation.jnt_current.shape[0] != geometry.tokens.shape[0]:
            raise ValueError("actor observation and geometry batch sizes disagree")
        torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
            torch.all(observation.owner_valid == geometry.owner_valid), "actor/geometry masks disagree"
        )
        local, finger, hand, film_modulation_rms = self._local_and_hand(observation, geometry)
        hand_per_joint = hand.unsqueeze(1).expand(-1, JOINT_COUNT, -1)
        raw_base = self.base_head(torch.cat((local, hand_per_joint), dim=-1)).squeeze(-1)
        base = self.base_action_limit * torch.tanh(raw_base)  # $\mu^{base}\in[-0.8,0.8]$
        base = torch.where(observation.jnt_valid, base, torch.zeros_like(base))

        contextual = None
        if self.residual_enabled:
            contextual = self._contextual_tokens(observation, geometry, local, finger, hand)
            raw_residual = self.residual_head(torch.cat((contextual[:, 1:17], local), dim=-1)).squeeze(-1)
            residual = RESIDUAL_LIMIT * torch.tanh(raw_residual)
            residual = torch.where(observation.jnt_valid, residual, torch.zeros_like(residual))
        else:
            residual = torch.zeros_like(base)
        mean = base + residual
        return PalmRotationActorOutput(
            mean=mean,
            film_modulation_rms=film_modulation_rms,
            log_std=self._policy_log_std(contextual, observation),
            base_mean=base,
            residual_mean=residual,
        )

    def _contextual_tokens(
        self,
        observation: PalmRotationActorObservation,
        geometry: PalmRotationGeometry,
        local: torch.Tensor,
        finger: torch.Tensor,
        hand: torch.Tensor,
    ) -> torch.Tensor:
        r"""把共享dynamic owner states与冻结geometry送入一层整手graph context。

        PALM接收whole-hand summary，JOINT接收local FiLM state，TIP接收finger summary；owner contact再以同序
        binary embedding叠加。Residual与direct arm逐值共享该$X^a\to H^a$变换。
        """

        dynamic = torch.zeros_like(geometry.tokens)  # `[B,21,128]` actor-specific动态owner tokens
        dynamic[:, 0] = self.palm_dynamic_projection(hand)
        dynamic[:, 1:17] = self.joint_dynamic_projection(local)
        dynamic[:, 17:21] = self.tip_dynamic_projection(finger)
        tokens = self.geometry_adapter(geometry.tokens) + dynamic
        tokens = tokens + self.owner_contact_projection(observation.owner_contact)
        return self.global_backbone(
            tokens,
            geometry.shortest_path,
            geometry.parent_direction,
            geometry.child_direction,
            geometry.owner_valid,
        )


class PalmRotationDirectActor(PalmRotationResidualActor):
    r"""用contextual JOINT tokens产生全authority动作的两种matched Direct actor。

    该arm共享residual actor的History30、dynamic-first FiLM、finger/hand pooling、owner adapters与一层graph
    backbone，但没有base/residual动作加法分解。Canonical ``direct_token``只读取contextual token：

    $$
    \mu_{t,j}=\tanh f_{direct}(H^a_{t,j})\in[-1,1].
    $$

    已有``direct``checkpoint保持``[H^a_{t,j},Z^{a,loc}_{t,j}]``的local-skip语义。Token-only用96维head
    hidden，使首层权重数$128\times96$与local-skip的$192\times64$相同；两者及Residual总参数量差保持在
    ±5%内。它们都不是``residual_enabled=False``，后者表示只有local base、没有global context。

    ``phase_clock_enabled`` 只在 direct/direct_token route 上打开。runtime 传入的
    ``phase_clock=[\sin\phi,\cos\phi]`` 经过零初始化的 ``Linear(2,128,bias=False)`` 后，
    仅广播到真实 JOINT contextual token，再进入原有 direct head；因此 phase 不会改变 local skip、
    TCN、geometry FiLM 或 global/conditional log sigma 的定义。
    """

    def __init__(
        self,
        *,
        initial_log_std: float = -0.5,
        max_log_std: float = -0.43,
        history_encoder: Literal["tcn", "raw_stack"] = "tcn",
        local_skip: bool = True,
        sigma_mode: Literal["global", "conditional"] = "global",
        recovery_sigma_floor: float | None = None,
        phase_clock_enabled: bool = False,
    ) -> None:
        r"""构造共享trunk，并按显式feature-route以direct head替换两条Residual动作heads。"""

        super().__init__(
            residual_enabled=True,
            initial_log_std=initial_log_std,
            max_log_std=max_log_std,
            base_action_limit=BASE_ACTION_LIMIT,
            history_encoder=history_encoder,
            sigma_mode=sigma_mode,
            recovery_sigma_floor=recovery_sigma_floor,
        )
        del self.base_head  # direct没有local base动作读出；local仍进入context与最终head
        del self.residual_head  # direct没有0.2 correction语义
        del self.residual_enabled  # 防止外部把direct误判成residual-off
        self.local_skip = bool(local_skip)  # true保留历史concat bypass；false对应notation-contract token-only
        head_input_width = GEOMETRY_WIDTH + LOCAL_WIDTH if self.local_skip else GEOMETRY_WIDTH
        head_hidden_width = 64 if self.local_skip else 96  # 两种Direct首层均为12,288个weights
        self.direct_head = nn.Sequential(
            nn.LayerNorm(head_input_width),
            nn.Linear(head_input_width, head_hidden_width),
            nn.GELU(),
            nn.Linear(head_hidden_width, 1),
        )
        nn.init.orthogonal_(self.direct_head[-1].weight, gain=0.01)  # type: ignore[arg-type]
        nn.init.zeros_(self.direct_head[-1].bias)  # type: ignore[arg-type]
        self.phase_clock_enabled = bool(phase_clock_enabled)  # phase 只属于 Direct 两个显式 feature route
        if self.phase_clock_enabled:
            # 零初始化 $\Delta H^a_j=W_p\mathbf{p}_t$，使启用模型起点逐值复现旧 contextual token。
            self.phase_contextual_adapter = nn.Linear(2, GEOMETRY_WIDTH, bias=False)
            nn.init.zeros_(self.phase_contextual_adapter.weight)  # `[128,2]`，新 route 初始严格无影响

    def forward(
        self,
        observation: PalmRotationActorObservation,
        geometry: PalmRotationGeometry,
        *,
        phase_clock: torch.Tensor | None = None,
        _validated: bool = False,
    ) -> PalmRotationActorOutput:
        r"""输出全authority direct mean、FiLM机制量与共享探索尺度。"""

        # 向量化任务梯度仅对外部已完整验证的批次切片设置_validated；普通forward仍执行生产检查。
        if not _validated:
            if observation.jnt_current.shape[0] != geometry.tokens.shape[0]:
                raise ValueError("actor observation and geometry batch sizes disagree")
            torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
                torch.all(observation.owner_valid == geometry.owner_valid), "actor/geometry masks disagree"
            )
        phase_clock = _phase_clock_input(
            phase_clock,
            enabled=self.phase_clock_enabled,
            batch_size=observation.jnt_current.shape[0],
            reference=observation.jnt_current,
            role="direct actor",
            validated=_validated,
        )
        local, finger, hand, film_modulation_rms = self._local_and_hand(observation, geometry)
        contextual = self._contextual_tokens(observation, geometry, local, finger, hand)
        contextual_joint = contextual[:, 1:17]  # 旧 global contextual JOINT token，形状 `[B,16,128]`
        if phase_clock is not None:
            # 同一 hand-level $\mathbf{p}_t$ 广播到真实 JOINT；ghost 行严格不接收 phase residual。
            phase_delta = self.phase_contextual_adapter(phase_clock).unsqueeze(1).expand(-1, JOINT_COUNT, -1)
            joint_mask = observation.jnt_valid.unsqueeze(-1).to(dtype=phase_delta.dtype)
            contextual_joint = contextual_joint + phase_delta * joint_mask  # 只改变均值分支的 direct 输入
        direct_input = (
            torch.cat((contextual_joint, local), dim=-1) if self.local_skip else contextual_joint
        )  # local-skip为`[B,16,192]`；token-only为$H^a_{t,j}\in\mathbb R^{128}$
        raw_direct = self.direct_head(direct_input).squeeze(-1)
        mean = torch.tanh(raw_direct)  # 完整物理动作authority$[-1,1]$
        mean = torch.where(observation.jnt_valid, mean, torch.zeros_like(mean))  # ghost严格零
        return PalmRotationActorOutput(
            mean=mean,
            film_modulation_rms=film_modulation_rms,
            log_std=self._policy_log_std(contextual, observation),
            direct_mean=mean,
        )


class PalmRotationStructuredCritic(nn.Module):
    r"""两层graph context与critic-only LN-c的privileged hand-level value。

    启用 ``phase_clock_enabled`` 时，``[\sin\phi,\cos\phi]`` 经过零初始化的
    ``Linear(2,896,bias=False)`` 加到七个 128D readout 拼接结果，再进入原有 value-head LayerNorm；
    该 route 只读 hand-level value，不改变 Actor 或其探索尺度。
    """

    def __init__(self, *, phase_clock_enabled: bool = False) -> None:
        r"""构造owner/object/task adapters、两层Pre-LN backbone与scalar readout。"""

        super().__init__()
        self.phase_clock_enabled = bool(phase_clock_enabled)  # phase adapter 只影响 hand-level value readout
        owner_input_width = GEOMETRY_WIDTH + 2 + 4
        self.owner_adapter = nn.Sequential(
            nn.Linear(owner_input_width, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
        )  # Linear→LN→activation是TOPPO critic-only LN-c对应边界
        self.backbone = _PalmRotationGraphBiasedTransformer(
            GraphBiasedTransformerCfg(
                hidden_width=128,
                layers=2,
                attention_heads=4,
                feedforward_width=256,
                dropout=0.0,
            )
        )
        self.object_adapter = nn.Sequential(nn.Linear(15, 128), nn.LayerNorm(128), nn.GELU())
        self.task_adapter = nn.Sequential(nn.Linear(9, 128), nn.LayerNorm(128), nn.GELU())
        self.value_head = nn.Sequential(
            nn.LayerNorm(7 * 128),
            nn.Linear(7 * 128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        if self.phase_clock_enabled:
            # 零初始化 $\Delta r=W_v\mathbf{p}_t$，保持旧 value_head LayerNorm 与宽度完全不变。
            self.phase_readout_adapter = nn.Linear(2, 7 * GEOMETRY_WIDTH, bias=False)
            nn.init.zeros_(self.phase_readout_adapter.weight)  # `[896,2]`，启用起点与旧价值逐值等价

    def forward(
        self,
        observation: PalmRotationCriticObservation,
        geometry: PalmRotationGeometry,
        *,
        phase_clock: torch.Tensor | None = None,
        _validated: bool = False,
    ) -> torch.Tensor:
        r"""融合privileged owner/object/task state并输出`[B]` scalar value。"""

        if not _validated:
            if observation.jnt_state.shape[0] != geometry.tokens.shape[0]:
                raise ValueError("critic observation and geometry batch sizes disagree")
            torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
                torch.all(observation.owner_valid == geometry.owner_valid), "critic/geometry masks disagree"
            )
        phase_clock = _phase_clock_input(
            phase_clock,
            enabled=self.phase_clock_enabled,
            batch_size=observation.jnt_state.shape[0],
            reference=observation.jnt_state,
            role="structured critic",
            validated=_validated,
        )
        # PALM/TIP无joint state；函数式拼接保留vmap的任务轴，不向未分批零张量原位copy。
        owner_joint = torch.cat(
            (
                torch.zeros_like(observation.jnt_state[:, :1]),
                observation.jnt_state,
                torch.zeros_like(observation.jnt_state[:, :4]),
            ),
            dim=1,
        )
        owner_raw = torch.cat((geometry.tokens, observation.owner_contact, owner_joint), dim=-1)
        owner = self.owner_adapter(owner_raw) * observation.owner_valid.unsqueeze(-1)
        contextual = self.backbone(
            owner,
            geometry.shortest_path,
            geometry.parent_direction,
            geometry.child_direction,
            geometry.owner_valid,
        )
        palm = contextual[:, 0]
        joint = contextual[:, 1:17]
        tip = contextual[:, 17:21]
        joint_mean = masked_mean(joint, observation.jnt_valid, dim=1)
        joint_max = masked_max(joint, observation.jnt_valid, dim=1)
        tip_mean = masked_mean(tip, observation.tip_valid, dim=1)
        tip_max = masked_max(tip, observation.tip_valid, dim=1)
        object_hidden = self.object_adapter(observation.obj[:, 0])
        task_hidden = self.task_adapter(torch.cat((observation.task[:, 0], observation.reward_release), dim=-1))
        readout = torch.cat((palm, joint_mean, joint_max, tip_mean, tip_max, object_hidden, task_hidden), dim=-1)
        if phase_clock is not None:
            readout = readout + self.phase_readout_adapter(phase_clock)  # `[B,896]`，只进入 value 分支
        return self.value_head(readout).squeeze(-1)


class PalmRotationActorCritic(nn.Module):
    r"""完全分参的actor/critic checkpoint namespace容器。

    ``phase_clock_enabled`` 是显式方法开关，仅对 ``direct`` 与 ``direct_token`` 合法；base/residual
    arm 保持 phase-disabled。模型不保存或推进 episode 时钟，时钟周期与 episode reset 语义由 runtime 提供。
    """

    def __init__(
        self,
        *,
        arm: Literal["base", "residual", "direct", "direct_token"] = "residual",
        initial_log_std: float = -0.5,
        max_log_std: float = -0.43,
        base_action_limit: float = BASE_ACTION_LIMIT,
        history_encoder: Literal["tcn", "raw_stack"] = "tcn",
        sigma_mode: Literal["global", "conditional"] = "global",
        recovery_sigma_floor: float | None = None,
        phase_clock_enabled: bool = False,
    ) -> None:
        r"""按显式arm实例化actor与共同critic；冻结N040不属于本module。"""

        super().__init__()
        self.phase_clock_enabled = bool(phase_clock_enabled)  # phase 只对 direct/direct_token 合法
        if self.phase_clock_enabled and arm not in {"direct", "direct_token"}:
            raise ValueError("phase_clock_enabled is supported only for direct or direct_token arm")
        if arm in {"direct", "direct_token"}:
            self.actor: PalmRotationResidualActor | PalmRotationDirectActor = PalmRotationDirectActor(
                initial_log_std=initial_log_std,
                max_log_std=max_log_std,
                history_encoder=history_encoder,
                local_skip=arm == "direct",
                sigma_mode=sigma_mode,
                recovery_sigma_floor=recovery_sigma_floor,
                phase_clock_enabled=self.phase_clock_enabled,
            )
        elif arm in {"base", "residual"}:
            self.actor = PalmRotationResidualActor(
                residual_enabled=arm == "residual",
                initial_log_std=initial_log_std,
                max_log_std=max_log_std,
                base_action_limit=base_action_limit,
                history_encoder=history_encoder,
                sigma_mode=sigma_mode,
                recovery_sigma_floor=recovery_sigma_floor,
            )
        else:
            raise ValueError(f"unsupported palm-rotation actor arm: {arm!r}")
        self.arm = arm  # checkpoint/model诊断显式区分Residual、local-skip Direct与token-only Direct
        self.critic = PalmRotationStructuredCritic(phase_clock_enabled=self.phase_clock_enabled)

    def trainable_parameter_sets(self) -> tuple[set[int], set[int]]:
        r"""返回actor/critic parameter object IDs供optimizer/checkpoint断言。"""

        return {id(parameter) for parameter in self.actor.parameters()}, {
            id(parameter) for parameter in self.critic.parameters()
        }


def expanded_policy_log_std(log_std: torch.Tensor, mean: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    r"""只允许全局标量或精确[B,16]条件尺度；在概率计算前中和ghost。"""
    if log_std.numel() == 1:
        expanded = log_std.expand_as(mean)
    elif log_std.shape == mean.shape:
        expanded = log_std
    else:
        raise ValueError("policy log_std must be scalar or match the [batch,joint] mean")
    return torch.where(active, expanded, torch.zeros_like(expanded))


__all__ = [
    "expanded_policy_log_std",
    "BASE_ACTION_LIMIT",
    "GEOMETRY_WIDTH",
    "FILM_LIMIT",
    "HISTORY_LENGTH",
    "JOINT_COUNT",
    "OWNER_COUNT",
    "RESIDUAL_LIMIT",
    "TIP_COUNT",
    "PalmRotationActorCritic",
    "PalmRotationActorObservation",
    "PalmRotationActorOutput",
    "PalmRotationCriticObservation",
    "PalmRotationDirectActor",
    "PalmRotationGeometry",
    "PalmRotationResidualActor",
    "PalmRotationStructuredCritic",
    "masked_max",
    "masked_mean",
]
