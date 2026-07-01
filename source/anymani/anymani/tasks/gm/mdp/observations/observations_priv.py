r"""Privileged observation terms for GM teacher policies.

本模块放置 teacher / specialist policy 可以读取、但不一定要求 student 或 sim2real
策略可观测的 object 状态。当前单资产 MDP 核验阶段优先让 teacher 学会任务，
因此 policy obs 可以显式读取 object pose；后续 distillation / student 阶段再决定
哪些项保留、压缩或遮蔽。

本项目主线是手型泛化，但保留物体泛化接口。物体泛化必须在 teacher 训练阶段引入
多元物体资产并通过 privileged info 显式条件化策略。本段定义从仿真器提取哪些 raw
physical values，交给 `distill/models` 侧的 object token encoder 投影为 `[OBJ]` 全局 token。

物体表征路线（已与用户对齐）：当前阶段采用扩展 HORA 路线：

- teacher 显式喂 raw physical properties（mass, scale, friction, COM 等），不做 HORA 风格
  的压缩嵌入，避免嵌入与具体手型策略耦合；
- 策略侧由 `distill/models` 将 raw values 投影为 `[OBJ]` token，并在 self-attention 池中与
  joint tokens 交互；
- `[OBJ]` token 的 projection 模块未来可额外接收手形态特征，使物体表征对当前 hand
  embodiment 有感知；
- 若几何形状多样性成为瓶颈，`[OBJ]` token 可升级为多个静态 mesh token（BPS / 几何描述符，
  离线预计算）。

DONE(object pose obs):
    当前已经落地 object pose 的可配置 frame / representation 版本：

    - `object_pos`: object root 的位置，可选择 `{h}` / `{e}` 轴和 hand / env reference，
      单位 m；
    - `object_orientation`: object frame `{o}` 的姿态，可选择 `{h}` / `{e}` frame 与
      `rot6d` / `quat` / `axis_angle` / `matrix` 表示。默认 `rot6d` 仍沿用 Zhou 6D
      continuous rotation representation，避免裸 quaternion 的 $q/-q$ 双覆盖，也避免
      9D 矩阵冗余。

TODO(privileged physics obs): 物体物理属性属于 teacher-only 信息，sim2real 不可得。
未来可从仿真器提取约 21 维 raw physical values：

- object mass $m$；
- object scale $(s_x,s_y,s_z)$；
- object friction $\mu$；
- object COM offset $(dx,dy,dz)$；
- object pose $(x,y,z,q_w,q_x,q_y,q_z)$；
- object velocity $(v_x,v_y,v_z,\omega_x,\omega_y,\omega_z)$。

数据来源包括 `RigidObject.root_physx_view`、`object.data.root_pos_w`、
`object.data.root_quat_w`、`object.data.root_lin_vel_w` 与 `object.data.root_ang_vel_w`。
所有值均为 SI 物理量，不依赖手构型，满足跨手型泛化的表征解耦需求。

具体投影维度、是否拆分为多个 token、是否加 position embedding，由
`distill/models` 的 Specialist Policy Transformer 设计决定。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


FrameName = Literal["h", "e"]
PositionReference = Literal["hand", "env"]
RotationRepresentation = Literal["rot6d", "quat", "axis_angle", "matrix"]


def _semantic_R_ha_tensor(env: ManagerBasedRLEnv, semantic_R_ha: tuple[float, ...]) -> torch.Tensor:
    r"""把配置层 row-major $R_{ha}$ 转成 runtime tensor。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env，用于确定 device。
        semantic_R_ha (tuple[float, ...]): row-major 9 元组，语义为列向量约定下
            $v^h = R_{ha} v^a$。

    Returns:
        torch.Tensor: 旋转矩阵 $R_{ha}$，形状 `[3, 3]`，位于 env device。
    """

    return torch.tensor(semantic_R_ha, dtype=torch.float32, device=env.device).reshape(3, 3)  # $R_{ha}$


def _semantic_p_ha_tensor(env: ManagerBasedRLEnv, semantic_p_ha: tuple[float, float, float]) -> torch.Tensor:
    r"""把配置层 $p_{ha}$ 转成 runtime row-vector tensor。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env，用于确定 device。
        semantic_p_ha (tuple[float, float, float]): raw asset/root 原点在 hand semantic frame `{h}`
            下的坐标 $p_{ha}$，单位 m。

    Returns:
        torch.Tensor: 平移向量 $p_{ha}$，形状 `[3]`，位于 env device。
    """

    return torch.tensor(semantic_p_ha, dtype=torch.float32, device=env.device).reshape(3)  # $p_{ha}$，单位 m


def _hand_origin_offset_a(R_ha: torch.Tensor, p_ha: torch.Tensor) -> torch.Tensor:
    r"""由 $T_{ha}$ 反解 hand semantic origin 在 raw asset frame `{a}` 中的位置。

    $T_{ha}$ 的列向量约定为：
    $$
    p^{\{h\}} = R_{ha}p^{\{a\}} + p_{ha}.
    $$
    hand semantic 原点满足 $p^{\{h\}}=0$，因此：
    $$
    p_{ah} = -R_{ha}^\top p_{ha}.
    $$

    Args:
        R_ha (torch.Tensor): `{a}->{h}` 旋转矩阵，形状 `[3,3]`。
        p_ha (torch.Tensor): raw asset/root 原点在 `{h}` 中的位置，形状 `[3]`，单位 m。

    Returns:
        torch.Tensor: hand semantic 原点在 `{a}` 中的位置，形状 `[3]`，单位 m。
    """

    return -(p_ha @ R_ha)  # row-vector 写法：$p_{ah}^\top=-p_{ha}^\top R_{ha}$


def _rotation_representation(
    rot: torch.Tensor,
    representation: RotationRepresentation,
    make_quat_unique: bool,
) -> torch.Tensor:
    r"""把旋转矩阵批量编码成 policy-facing orientation representation。

    Args:
        rot (torch.Tensor): 旋转矩阵，形状 `[B,3,3]`，列向量约定下表示 $R$。
        representation (RotationRepresentation): 输出表示；`rot6d` 为 Zhou 6D，`quat` 为
            IsaacLab `(w,x,y,z)`，`axis_angle` 为 $so(3)$ 向量，`matrix` 为 row-major 9D。
        make_quat_unique (bool): 当输出或中间转换使用 quaternion 时，是否折叠到 $q_w\ge0$。

    Returns:
        torch.Tensor: 扁平 orientation 表示，第一维为 batch。
    """

    if representation == "rot6d":
        return torch.cat((rot[:, :, 0], rot[:, :, 1]), dim=-1)  # `[B,6]`，按列拼 $R[:,0],R[:,1]$
    if representation == "matrix":
        return rot.reshape(rot.shape[0], 9)  # `[B,9]`，row-major 展平，保留完整 $SO(3)$ 信息

    quat = math_utils.quat_from_matrix(rot)  # `[B,4]`，IsaacLab `(w,x,y,z)` quaternion
    if make_quat_unique:
        quat = math_utils.quat_unique(quat)  # 折叠 $q/-q$ 双覆盖，保证 policy-facing 符号一致性
    if representation == "quat":
        return quat  # `[B,4]`，absolute / relative quaternion 表示
    if representation == "axis_angle":
        return math_utils.axis_angle_from_quat(quat)  # `[B,3]`，$\log(R)\in\mathbb{R}^3$，单位 rad
    raise ValueError(f"Unsupported rotation representation: {representation}.")


def object_pos(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    semantic_R_ha: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    semantic_p_ha: tuple[float, float, float] = (0.0, 0.0, 0.0),
    frame: FrameName = "h",
    reference: PositionReference = "hand",
) -> torch.Tensor:
    r"""读取 object root 位置，并按配置选择坐标 frame 与 reference origin。

    该项把旧的 hand-frame object position 推广为可配置表示。最常用的 hand-relative hand-frame
    形式为：
    $$
    p_o^{\{h\}} = p_{ha} + R_{ha} R_{aw} (p_o^{\{w\}} - p_a^{\{w\}}),
    $$
    其中 `{a}` 是 raw asset/root frame，`{h}` 是 hand semantic frame。
    对 official LEAP 这类真实资产，$p_{ha}$ 约为厘米量级，不能继续假设为 0。

    `frame="e"` 时保留 env/world 轴向；`reference="env"` 时不再减 hand origin，
    得到 env-local absolute position。两者组合用于诊断“相对量是否让 RL 观测非平稳”。

    在当前 fixed hand anchor 的 single-asset 阶段，reset 后该项应接近标定台
    导出的 contact basin，例如 $(0.02, 0.08, 0.06)$。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        object_cfg (SceneEntityCfg): object rigid body 配置。
        robot_cfg (SceneEntityCfg): hand articulation 配置。
        semantic_R_ha (tuple[float, ...]): $R_{ha}$，row-major 9 元组。
        semantic_p_ha (tuple[float, float, float]): $p_{ha}$，单位 m。
        frame (FrameName): 输出坐标轴，`"h"` 为 hand semantic 轴，`"e"` 为 env/world 轴。
        reference (PositionReference): 输出原点，`"hand"` 减 hand semantic origin，`"env"` 减 env origin。

    Returns:
        torch.Tensor: object 位置表示，形状 `[num_envs, 3]`，单位 m。
    """

    object_asset: RigidObject = env.scene[object_cfg.name]  # object root pose 来源
    robot_asset: Articulation = env.scene[robot_cfg.name]  # hand root pose 来源
    R_ha = _semantic_R_ha_tensor(env, semantic_R_ha)  # `{a}->{h}` 语义对齐矩阵
    p_ha = _semantic_p_ha_tensor(env, semantic_p_ha)  # $p_{ha}$，raw root 在 `{h}` 中的位置，单位 m

    if reference == "hand":
        rel_pos_w = object_asset.data.root_pos_w - robot_asset.data.root_pos_w  # $p_o^w-p_a^w$，`[B,3]`
        rel_pos_a = math_utils.quat_apply_inverse(robot_asset.data.root_quat_w, rel_pos_w)  # $R_{aw}(p_o^w-p_a^w)$
        if frame == "h":
            return rel_pos_a @ R_ha.T + p_ha  # $p_o^h=p_{ha}+p_o^aR_{ha}^\top$，`[B,3]`
        if frame == "e":
            p_ah = _hand_origin_offset_a(R_ha, p_ha)  # $p_{ah}$，hand semantic origin 在 `{a}` 下的位置
            p_ah_w = math_utils.quat_apply(robot_asset.data.root_quat_w, p_ah.repeat(env.num_envs, 1))  # $R_{wa}p_{ah}$
            return object_asset.data.root_pos_w - (robot_asset.data.root_pos_w + p_ah_w)  # $p_o^e-p_h^e$，`[B,3]`
        raise ValueError(f"Unsupported frame for object_pos: {frame}.")

    if reference == "env":
        pos_e = object_asset.data.root_pos_w - env.scene.env_origins  # $p_o^e$，env-local absolute position，`[B,3]`
        if frame == "e":
            return pos_e  # env-local axes 与 world axes 同向，直接返回 $p_o^e$
        if frame == "h":
            pos_a = math_utils.quat_apply_inverse(robot_asset.data.root_quat_w, pos_e)  # 把 env-origin 向量表达进 `{a}` 轴
            return pos_a @ R_ha.T  # row-vector 写法：$p^h=R_{ha}p^a$
        raise ValueError(f"Unsupported frame for object_pos: {frame}.")

    raise ValueError(f"Unsupported position reference for object_pos: {reference}.")


def object_orientation(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    semantic_R_ha: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    frame: FrameName = "h",
    representation: RotationRepresentation = "rot6d",
    make_quat_unique: bool = False,
) -> torch.Tensor:
    r"""读取 object orientation，并按配置选择坐标 frame 与旋转表示。

    `frame` 只决定**姿态矩阵左侧的参考坐标系**，也就是“object body axes 用哪套
    外部坐标轴来表达”。它不涉及位置原点，因此不像 `object_pos` 那样还需要
    `reference` 参数。这里始终读取 object body frame `{o}` 的朝向，只是在 `{h}`
    或 `{e}` 中表达这组朝向。

    `frame="h"` 时返回 object body frame `{o}` 相对 hand semantic frame `{h}` 的姿态：
    $$
    R_{ho} = R_{ha} R_{aw} R_{wo}.
    $$
    其中 $R_{wo}$ 来自 IsaacLab object root quaternion，$R_{aw}=R_{wa}^{\top}$
    把 world/env 轴旋回 raw hand asset frame `{a}`，$R_{ha}$ 再把 `{a}` 对齐到
    AnyMani hand semantic frame `{h}`。该模式适合手型泛化主线：同一个 object
    朝向会随手的语义轴一起被表达。

    `frame="e"` 时返回 object body frame `{o}` 相对 env/world frame `{e}` 的姿态：
    $$
    R_{eo} = R_{wo}.
    $$
    该模式不使用 hand pose 与 $R_{ha}$，适合诊断“hand-frame 表示是否引入额外非平稳性”
    或与 IsaacLab / LEAP 官方 world-frame observation 做对照。

    `representation` 决定把上面的旋转矩阵 $R\in SO(3)$ 编码成哪种 policy-facing
    向量；目前完整选项为：

    - `"rot6d"`: 输出 `[B,6]`，拼接旋转矩阵前两列
      $[R_{:,0},R_{:,1}]$。这是默认值，来自 Zhou 6D continuous rotation
      representation；它去掉第三列冗余，又避免 quaternion 的 $q/-q$ 双覆盖。
    - `"quat"`: 输出 `[B,4]`，IsaacLab 约定的 quaternion `(w,x,y,z)`。该表示紧凑，
      但同一旋转存在 $q$ 与 $-q$ 两个等价符号；若 `make_quat_unique=True`，则折叠到
      IsaacLab `quat_unique` 选取的单侧符号。
    - `"axis_angle"`: 输出 `[B,3]`，李代数向量 $\omega\theta=\log(R)$，单位 rad。
      它适合表达局部姿态残差，但在 $\theta\approx\pi$ 附近存在分支不连续；若
      `make_quat_unique=True`，会先规范化中间 quaternion 再转 axis-angle。
    - `"matrix"`: 输出 `[B,9]`，按 row-major 展平完整 $3\times3$ 旋转矩阵。它最冗余，
      但最接近 $SO(3)$ 原始几何对象，适合 debugging 或做表征消融。

    Args:
        env (ManagerBasedRLEnv): Isaac Lab manager-based RL env。
        object_cfg (SceneEntityCfg): object rigid body 配置。
        robot_cfg (SceneEntityCfg): hand articulation 配置。
        semantic_R_ha (tuple[float, ...]): $R_{ha}$，row-major 9 元组；仅在 `frame="h"`
            时使用，语义为 $v^{\{h\}}=R_{ha}v^{\{a\}}$。
        frame (FrameName): 输出参考坐标系；`"h"` 表示 hand semantic frame，`"e"`
            表示 env/world frame。
        representation (RotationRepresentation): 输出旋转表示，可选 `"rot6d"`、`"quat"`、
            `"axis_angle"`、`"matrix"`，对应输出维度分别为 6、4、3、9。
        make_quat_unique (bool): 若 `representation="quat"` 或 `"axis_angle"`，是否先通过
            `quat_unique` 折叠 $q/-q$；对 `"rot6d"` 和 `"matrix"` 没有影响。

    Returns:
        torch.Tensor: object orientation 表示，batch 维为 `num_envs`；shape 为 `[B,6]`
        (`rot6d`)、`[B,4]` (`quat`)、`[B,3]` (`axis_angle`) 或 `[B,9]` (`matrix`)。
    """

    object_asset: RigidObject = env.scene[object_cfg.name]  # object orientation $R_{wo}$
    R_wo = math_utils.matrix_from_quat(object_asset.data.root_quat_w)  # `[B,3,3]`，object `{o}->{w}`
    if frame == "e":
        return _rotation_representation(R_wo, representation, make_quat_unique)  # env/world frame 下的 $R_{eo}$
    if frame == "h":
        robot_asset: Articulation = env.scene[robot_cfg.name]  # hand root orientation $R_{wa}$
        R_ha = _semantic_R_ha_tensor(env, semantic_R_ha).unsqueeze(0)  # `[1,3,3]`，广播到所有 env
        R_wa = math_utils.matrix_from_quat(robot_asset.data.root_quat_w)  # `[B,3,3]`，hand root `{a}->{w}`
        R_aw = R_wa.transpose(-1, -2)  # `[B,3,3]`，world/env 到 hand raw asset `{a}`
        R_ho = R_ha @ R_aw @ R_wo  # `[B,3,3]`，object `{o}` 在 hand semantic `{h}` 中的姿态
        return _rotation_representation(R_ho, representation, make_quat_unique)  # 编码为 policy-facing 表示
    raise ValueError(f"Unsupported frame for object_orientation: {frame}.")


__all__ = ["object_orientation", "object_pos"]
