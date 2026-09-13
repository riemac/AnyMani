r"""三组离线 family student 的共享策略主体与 FK 辅助头。

``PalmRotationDirectActor`` 已经固定了 History30、TIP-only contact、16-action direct-token 和 graph-biased
owner backbone。本文件只增加共同的静态关节运动学条件：每个 JOINT 的 15 维向量按

$$
k_j=[p^{home}_{parent\rightarrow joint}/L, R^{home}_{parent\rightarrow joint}, a^{local}_j]
\in\mathbb{R}^{15},\qquad L=0.1\,\mathrm{m},
$$

排列为 parent-to-joint home 平移（米除以 $L$）、9 个旋转矩阵元素和 joint-local axis。它经过零初始化
``Linear(15,128)``，只加到有效 JOINT owner token；PALM/TIP 的 kinematic token 精确为零。

``n040`` 保留缓存的 N040 $Z^e$，``no_z`` 与 ``fk`` 先把同一输入中的 $Z^e$ 清零，但三组都保留图、mask、
limits、current/history/contact 和 kinematics。FK 组额外使用 ``128 -> 64 -> 3`` 训练头预测当前 joint
origin/L；动作均值不消费该预测，因此 FK loss 不会改变部署动作接口。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from .palm_rotation_policy import (
    GEOMETRY_WIDTH,
    JOINT_COUNT,
    PalmRotationActorObservation,
    PalmRotationDirectActor,
    PalmRotationGeometry,
)

FAMILY_ROTATION_VARIANTS = ("n040", "no_z", "fk")
FamilyRotationVariant = Literal["n040", "no_z", "fk"]
JOINT_KINEMATICS_WIDTH = 15
JOINT_ORIGIN_WIDTH = 3
LINK_LENGTH_M = 0.1


@dataclass(frozen=True)
class FamilyRotationActorOutput:
    r"""动作均值、共享探索尺度、FiLM 诊断量和可选 FK 预测。

    ``fk_prediction`` 的单位是无量纲的 joint-origin/L；只有 FK variant 返回非 ``None``。动作均值
    ``mean`` 始终是 canonical 16-slot 无量纲 action，推理代码不读取 FK 输出。
    """

    mean: torch.Tensor  # `[B,16]`，真实有效 joint 上为 `[-1,1]` 的 direct action
    film_modulation_rms: torch.Tensor  # `[B,16]`，几何 FiLM 对 dynamic local 的 RMS 改变
    log_std: torch.Tensor  # global 标量或 `[B,16]`；离线 student 中 global 参数冻结
    fk_prediction: torch.Tensor | None = None  # FK variant 为 `[B,16,3]`，单位 origin/L

    @property
    def direct_mean(self) -> torch.Tensor:
        r"""返回父类 direct actor 使用的同一动作均值别名；不新增动作分支。"""

        return self.mean


class FamilyRotationStudentActor(PalmRotationDirectActor):
    r"""继承 direct-token trunk、加入静态 kinematics adapter 的三组共享学生主体。

    共享部分严格来自父类：History30 TCN、dynamic-first geometry FiLM、finger/hand pooling、owner adapter、
    一层 graph backbone 和 ``local_skip=False`` direct head。新增 adapter 零初始化，所以在同 seed 的初始
    时刻三组共享参数逐值一致；FK head 在共享 adapter 创建后才初始化，属于 FK 训练专属参数。
    """

    def __init__(
        self,
        variant: FamilyRotationVariant,
        *,
        initial_log_std: float = -0.5,
        max_log_std: float = -0.43,
    ) -> None:
        r"""构造一组随机初始化 actor；不读取或继承任何 teacher 参数。"""

        if variant not in FAMILY_ROTATION_VARIANTS:
            raise ValueError(f"family rotation variant must be one of {FAMILY_ROTATION_VARIANTS}, got {variant!r}")
        super().__init__(
            initial_log_std=initial_log_std,
            max_log_std=max_log_std,
            history_encoder="tcn",
            local_skip=False,
            sigma_mode="global",
            phase_clock_enabled=False,
        )
        self.variant: FamilyRotationVariant = variant
        self.representation: FamilyRotationVariant = variant
        self.joint_kinematics_width = JOINT_KINEMATICS_WIDTH
        self.link_length_m = LINK_LENGTH_M
        # 零初始化使静态 kinematics 在训练起点不遮蔽原 direct-token 动态/几何路径。
        self.joint_kinematics_adapter = nn.Linear(JOINT_KINEMATICS_WIDTH, GEOMETRY_WIDTH)
        nn.init.zeros_(self.joint_kinematics_adapter.weight)
        nn.init.zeros_(self.joint_kinematics_adapter.bias)
        self.fk_head: nn.Module | None = None
        if variant == "fk":
            # $H_j^a\in R^{128}\rightarrow64\rightarrow3$，输出当前 origin/L 的辅助监督。
            self.fk_head = nn.Sequential(
                nn.Linear(GEOMETRY_WIDTH, 64),
                nn.GELU(),
                nn.Linear(64, JOINT_ORIGIN_WIDTH),
            )
        self.family_actor_config: dict[str, object] = {
            "variant": variant,
            "representation": variant,
            "history_encoder": "tcn",
            "history_length": 30,
            "local_skip": False,
            "sigma_mode": "global",
            "phase_clock_enabled": False,
            "joint_kinematics_width": JOINT_KINEMATICS_WIDTH,
            "joint_origin_width": JOINT_ORIGIN_WIDTH,
            "link_length_m": LINK_LENGTH_M,
            "initial_log_std": float(initial_log_std),
            "max_log_std": float(max_log_std),
        }
        # mean-only IL 的 sigma 是冻结探索常数；它仍进入 state_dict 以便严格 round-trip。
        self.global_log_std.requires_grad_(False)

    def _effective_geometry(self, geometry: PalmRotationGeometry) -> PalmRotationGeometry:
        r"""按 variant 形成 N040 或清零 token 的 geometry view，同时保持图和 mask。"""

        if self.variant in {"no_z", "fk"}:
            # 只执行 $\tilde Z^e=0$ 干预；三类 graph matrix 和 owner_valid 仍是原始对象中的结构证据。
            return PalmRotationGeometry(
                tokens=torch.zeros_like(geometry.tokens),
                owner_valid=geometry.owner_valid,
                shortest_path=geometry.shortest_path,
                parent_direction=geometry.parent_direction,
                child_direction=geometry.child_direction,
            )
        return geometry

    def _tip_only_observation(self, observation: PalmRotationActorObservation) -> PalmRotationActorObservation:
        r"""执行 TIP-only actor 信息边界，保留本体状态与所属手指 TIP 触觉。

        current/history 五通道的索引约定是 ``[q/pi, u/pi, previous_action, non_tip_contact, tip_contact]``；
        deployment actor 只允许前三项本体状态和最后一项 TIP contact。owner contact 的 PALM/JOINT 行
        同样置零，21 行 tensor 仍保留以匹配 graph/owner ABI。
        """

        current = observation.jnt_current.clone()  # `[B,16,5]`，只在边界复制可被策略读取的动态 packet
        history = observation.jnt_history.clone()  # `[B,30,16,5]`，历史 non-tip contact 同样不能泄漏
        current[..., 3] = 0.0  # non-tip contact channel：TIP-only 合同要求恒为零
        history[..., 3] = 0.0  # 对所有 History30 lag 执行同一信息边界
        owner_contact = observation.owner_contact.clone()  # `[B,21,1]`，保留 TIP 行的 binary contact
        owner_contact[:, :17] = 0.0  # PALM/JOINT contact 不进入 Actor，TIP rows `[17:21]` 原样保留
        return PalmRotationActorObservation(
            jnt_current=current,
            jnt_history=history,
            jnt_limits=observation.jnt_limits,
            owner_contact=owner_contact,
            jnt_valid=observation.jnt_valid,
            tip_valid=observation.tip_valid,
            owner_valid=observation.owner_valid,
        )

    def _contextual_tokens_with_kinematics(
        self,
        observation: PalmRotationActorObservation,
        geometry: PalmRotationGeometry,
        local: torch.Tensor,
        finger: torch.Tensor,
        hand: torch.Tensor,
        joint_kinematics: torch.Tensor,
    ) -> torch.Tensor:
        r"""在 inherited owner-token dynamic path中注入零初始化 kinematic JOINT embedding。

        继承父类的动态构造：PALM 接收 hand summary、JOINT 接收 local FiLM state、TIP 接收 finger summary，
        owner contact 按同序加和。新增项仅是

        $$
        X^{a}_{j}=Z^{e}_{j}+D^{a}_{j}+C_{j}+W_k k_j,
        \quad j\in\mathrm{JOINT},
        $$

        其中 $W_k=0$ 初始化；PALM/TIP 行没有 kinematic 项。
        """

        dynamic = torch.zeros_like(geometry.tokens)  # `[B,21,128]`，与父类完全同序的动态 owner token
        dynamic[:, 0] = self.palm_dynamic_projection(hand)  # PALM hand summary
        dynamic[:, 1:17] = self.joint_dynamic_projection(local)  # JOINT local FiLM state
        dynamic[:, 17:21] = self.tip_dynamic_projection(finger)  # TIP finger summary
        kinematic_delta = torch.zeros_like(geometry.tokens)  # `[B,21,128]`，仅 JOINT 接收静态 adapter
        joint_embedding = self.joint_kinematics_adapter(joint_kinematics)  # `[B,16,128]`
        joint_mask = observation.jnt_valid.unsqueeze(-1).to(dtype=joint_embedding.dtype)
        kinematic_delta[:, 1:17] = joint_embedding * joint_mask  # ghost joint 不产生静态条件梯度
        tokens = self.geometry_adapter(geometry.tokens) + dynamic  # 父类 geometry adapter 路径
        tokens = tokens + self.owner_contact_projection(observation.owner_contact)  # TIP-only contact ABI
        tokens = tokens + kinematic_delta  # 零初始化的 kinematic condition
        return self.global_backbone(
            tokens,
            geometry.shortest_path,
            geometry.parent_direction,
            geometry.child_direction,
            geometry.owner_valid,
        )

    def forward(
        self,
        observation: PalmRotationActorObservation,
        geometry: PalmRotationGeometry,
        *,
        joint_kinematics: torch.Tensor | None = None,
        phase_clock: torch.Tensor | None = None,
        _validated: bool = False,
    ) -> FamilyRotationActorOutput:
        r"""返回 canonical action mean，并为 FK variant 附加训练期 origin 预测。

        Args:
            observation: History30/current/limits/contact/masks，形状与父类 actor ABI 相同。
            geometry: FP32 N040 tokens 与 21-owner graph；No-Z/FK 内部只将 tokens 置零。
            joint_kinematics: 静态 `[B,16,15]`，平移单位米除以 ``0.1 m``、旋转矩阵无量纲、axis 无量纲。
            phase_clock: 当前合同关闭；传入任何值都会显式报错。
            _validated: 保留父类 functional ABI 标记；此模块仍要求输入 shape/device 合法。
        """

        del _validated  # 训练/评价两端均采用同一显式 shape 检查，不绕过静态条件边界。
        if phase_clock is not None:
            raise ValueError("family rotation student requires phase_clock_enabled=False")
        if joint_kinematics is None:
            raise ValueError("family rotation student requires joint_kinematics with shape [B,16,15]")
        batch = observation.jnt_current.shape[0]
        expected_shape = (batch, JOINT_COUNT, JOINT_KINEMATICS_WIDTH)
        if tuple(joint_kinematics.shape) != expected_shape:
            raise ValueError(
                f"joint_kinematics must have shape {expected_shape}, got {tuple(joint_kinematics.shape)}"
            )
        if joint_kinematics.dtype != observation.jnt_current.dtype:
            raise ValueError(
                f"joint_kinematics must have dtype {observation.jnt_current.dtype}, got {joint_kinematics.dtype}"
            )
        if joint_kinematics.device != observation.jnt_current.device:
            raise ValueError(
                f"joint_kinematics must be on device {observation.jnt_current.device}, got {joint_kinematics.device}"
            )
        if geometry.tokens.shape[0] != batch:
            raise ValueError("family student observation and geometry batch sizes disagree")
        torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]  # GPU 端有限性检查保持 tensor-only
            torch.isfinite(joint_kinematics).all(),
            "joint_kinematics must contain finite values",
        )
        torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
            torch.all(observation.owner_valid == geometry.owner_valid),
            "family student actor/geometry masks disagree",
        )
        effective_geometry = self._effective_geometry(geometry)
        effective_observation = self._tip_only_observation(observation)
        local, finger, hand, film_modulation_rms = self._local_and_hand(effective_observation, effective_geometry)
        contextual = self._contextual_tokens_with_kinematics(
            effective_observation,
            effective_geometry,
            local,
            finger,
            hand,
            joint_kinematics,
        )
        contextual_joint = contextual[:, 1:17]  # `[B,16,128]`，direct_token 只读取 JOINT contextual token
        raw_direct = self.direct_head(contextual_joint).squeeze(-1)  # `[B,16]`，全 authority action logit
        mean = torch.tanh(raw_direct)  # $μ_{t,j}\in[-1,1]$，canonical action 不改变
        mean = torch.where(effective_observation.jnt_valid, mean, torch.zeros_like(mean))  # ghost action 严格置零
        fk_prediction = None
        if self.fk_head is not None:
            # FK 头只读同一个 H_joint，不回写 contextual/action 分支；输出 target/L 的无量纲坐标。
            fk_prediction = self.fk_head(contextual_joint)
            fk_prediction = torch.where(
                observation.jnt_valid.unsqueeze(-1),
                fk_prediction,
                torch.zeros_like(fk_prediction),
            )
        return FamilyRotationActorOutput(
            mean=mean,
            film_modulation_rms=film_modulation_rms,
            log_std=self._policy_log_std(contextual, effective_observation),
            fk_prediction=fk_prediction,
        )


def build_family_rotation_policy(
    variant: FamilyRotationVariant,
    *,
    device: torch.device | str | None = None,
    initial_log_std: float = -0.5,
    max_log_std: float = -0.43,
) -> FamilyRotationStudentActor:
    r"""构造随机初始化的指定 family student variant，并固定为 FP32。"""

    actor = FamilyRotationStudentActor(
        variant,
        initial_log_std=initial_log_std,
        max_log_std=max_log_std,
    )
    return actor.to(device=device, dtype=torch.float32)


# 语义别名便于模型合同测试与 IL artifact loader 共用同一主体类。
FamilyRotationPolicy = FamilyRotationStudentActor


__all__ = [
    "FAMILY_ROTATION_VARIANTS",
    "FamilyRotationActorOutput",
    "FamilyRotationStudentActor",
    "FamilyRotationPolicy",
    "FamilyRotationVariant",
    "JOINT_KINEMATICS_WIDTH",
    "JOINT_ORIGIN_WIDTH",
    "LINK_LENGTH_M",
    "build_family_rotation_policy",
]
