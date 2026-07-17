r"""Generated hand spawn adapter.

本模块是 `robots` 侧的 **IsaacLab runtime spawn 适配层**。它的长期职责是把
`assets.bank` 选出的 generated hand assets 包装成 IsaacLab 可消费的
`ArticulationCfg`，供 `tasks/gm` 与后续 `distill` runtime smoke 共同复用。

当前文件实现第一版 **URDF runtime spawn adapter**：字段、公式、接口、职责边界、
bank resolve、schema check、URDF importer cfg、可选 USD material restore 与 root pose
anchor lower。orientation reset 仍不在本文件实现，它属于 `mdp/events.py` 的 episode
级 reset 语义。

设计目标：

```text
HandSpawnCfg
  ├─ bank: HandBankCfg                  # assets 层资产选择配置
  ├─ frame: HandFrameCfg                # {a}->{h} 语义对齐与默认 T_eh anchor
  ├─ urdf: HandUrdfSpawnCfg             # URDF importer 参数
  ├─ actuator: HandActuatorSpawnCfg     # implicit actuator 参数
  └─ ...

HandSpawnAdapter(cfg)
  ├─ selection                          # lazy: HandBank(cfg.bank).resolve()
  ├─ build_articulation_cfg(...)        # -> ArticulationCfg
  ├─ build_multi_hand_spawn_cfg(...)    # -> MultiAssetSpawnerCfg
  └─ semantic_R_ha                      # env cfg 显式同步给 command cfg
```

边界约定：

- `assets.bank` 负责路径解析、资产选择、虚拟 bundle、URDF mesh / color 解析；
- `robots.hand_spawn` 只负责把已选 embodiment lower 成 IsaacLab robot cfg；
- `tasks/gm` 只消费 robot cfg 来表达 MDP，不拥有资产生成或训练算法；
- `distill` 负责训练时选用哪个 task / agent YAML / checkpoint / manifest。

Frame 语义：

- `{a}`：raw asset/root frame，即 URDF/USD 被 IsaacLab 加载后的资产根坐标系；
- `{h}`：hand semantic frame，任务语义使用的手坐标系；
- `semantic_R_ha` 表示 $R_{ha}$，即 $v^h = R_{ha}v^a$；
- `semantic_p_ha` 表示 $p_{ha}$，即 `{a}` 原点在 `{h}` 中的位置；
- `anchor_R_eh` / `anchor_p_eh` 表示 reset / spawn 的默认 hand semantic pose
  $T_{eh}^{anchor}$；
- 第一版默认目标是让 hand semantic frame 初始满足 $R_{wh}=I$，即
  $R_{eh}^{anchor}=I$。

在 IsaacLab cloned env 默认只相对 world 平移、无旋转的假设下，默认 anchor 对应：

$$
T_{ea}^{anchor}=T_{eh}^{anchor}T_{ha},\qquad
R_{ea}^{anchor}=R_{eh}^{anchor}R_{ha},\qquad
p_{ea}^{anchor}=p_{eh}^{anchor}+R_{eh}^{anchor}p_{ha}.
$$

当前默认 $R_{eh}^{anchor}=I$，所以退化为 $R_{ea}=R_{ha}$，
$p_{ea}=p_{eh}+p_{ha}$。episode 级任意 hand orientation 由 `events.py` 中的
orientation reset scaffold 表达：采样 $\Delta R_h$ 并右乘到 anchor 上，而不是
在 spawn 层局部随机化 root pose。

TOAGENT: 本文件只实现 spawn/bank adapter。episode 级 orientation reset、object reset、
Grasp Cache 和 command update 不要塞进这里。
"""

from __future__ import annotations

import logging
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.converters import UrdfConverterCfg
from isaaclab.utils import configclass

from anymani.assets.bank import HandBank, HandBankCfg, HandContainer, HandSelection, UrdfRgba
from anymani.assets.bank.urdf_utils import parse_urdf_visual_rgba_by_name

logger = logging.getLogger(__name__)

DEFAULT_HAND_ANCHOR_POS_E = (0.0, 0.0, 0.5)
r"""默认 hand semantic origin anchor 在 env frame `{e}` 中的位置，单位 m。"""

IDENTITY_R = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
"""Row-major $3\times3$ identity rotation。"""


@dataclass(frozen=True)
class _VisualMaterialRestorePlan:
    r"""同拓扑 hand selection 共享的 URDF debug color 恢复计划。

    pre-made topology 的 post-mutate variants 共享 visual name、parent link name 与 debug
    color palette；几何可以变（例如不同 fingertip mesh），但 `visual_name -> link_name`
    和 `visual_name -> rgba` 仍是拓扑级语义。因而 `MultiAssetSpawnerCfg` 内只需从第一个
    selected URDF 解析一次，后续 variants 复用该计划即可，避免每个 prototype spawn
    重复读取 XML。
    """

    source_urdf_path: Path
    """提供 visual/link/color contract 的 reference `hand.urdf` 路径。"""

    visual_rgba_by_name: dict[str, UrdfRgba]
    """URDF visual name -> RGBA debug color；颜色只服务 GUI/debug，不参与动力学。"""

    visual_link_by_name: dict[str, str]
    """URDF visual name -> parent link name；用于定位 spawned USD 的 `/<link>/visuals`。"""


def _serialize_visual_material_restore_plan(plan: _VisualMaterialRestorePlan) -> dict[str, object]:
    r"""把内部颜色恢复计划转成 `UrdfFileCfg.to_dict()` 可 JSON hash 的 payload。

    IsaacLab `UrdfConverter` 会在转换前对整个 cfg 做 `json.dumps(cfg.to_dict())`。
    因此不能把 `Path` 或 dataclass 直接挂到 `UrdfFileCfg` 上；这里显式降为
    `str + dict + list[float]`，既保留同拓扑共享计划，也不污染 converter hash。
    """

    return {
        "source_urdf_path": str(plan.source_urdf_path),
        "visual_rgba_by_name": {
            visual_name: list(rgba) for visual_name, rgba in plan.visual_rgba_by_name.items()
        },
        "visual_link_by_name": dict(plan.visual_link_by_name),
    }


def _deserialize_visual_material_restore_plan(payload: object) -> _VisualMaterialRestorePlan | None:
    r"""把挂在 `UrdfFileCfg` 上的 JSON-safe payload 还原为内部颜色恢复计划。"""

    if not isinstance(payload, dict):
        return None

    source_urdf_path = payload.get("source_urdf_path")
    visual_rgba_by_name = payload.get("visual_rgba_by_name")
    visual_link_by_name = payload.get("visual_link_by_name")
    if not isinstance(source_urdf_path, str) or not isinstance(visual_rgba_by_name, dict):
        return None
    if not isinstance(visual_link_by_name, dict):
        return None

    return _VisualMaterialRestorePlan(
        source_urdf_path=Path(source_urdf_path),
        visual_rgba_by_name={
            str(visual_name): tuple(float(value) for value in rgba)  # type: ignore[misc]
            for visual_name, rgba in visual_rgba_by_name.items()
        },
        visual_link_by_name={str(visual_name): str(link_name) for visual_name, link_name in visual_link_by_name.items()},
    )


@configclass
class HandFrameCfg:
    r"""Hand raw asset frame `{a}` 与 hand semantic frame `{h}` 的对齐配置。

    配置层使用 $(R,p)$；未来实现层应组合为 $SE(3)$ 计算，最后仅在 IsaacLab 边界
    转成 quaternion。`semantic_*` 是资产校准 $T_{ha}$；`anchor_*` 是 hand semantic
    frame 在 env frame 中的默认参考 pose $T_{eh}^{anchor}$。reset-time orientation
    DR 应以该 anchor 为默认参考，而不是覆盖 `{h}` 的语义定义。
    """

    semantic_R_ha: tuple[float, ...] = IDENTITY_R
    r"""$R_{ha}$，row-major 9 个 float，语义为 $v^h=R_{ha}v^a$。"""

    semantic_p_ha: tuple[float, float, float] = (0.0, 0.0, 0.0)
    r"""$p_{ha}$，raw asset origin `{a}` 在 hand semantic frame `{h}` 中的位置，单位 m。"""

    anchor_R_eh: tuple[float, ...] = IDENTITY_R
    r"""$R_{eh}^{anchor}$，hand semantic frame `{h}` 在 env frame `{e}` 中的默认参考朝向。"""

    anchor_p_eh: tuple[float, float, float] = DEFAULT_HAND_ANCHOR_POS_E
    r"""$p_{eh}^{anchor}$，hand semantic origin `{h}` 在 env frame `{e}` 中的默认参考位置，单位 m。"""

    align_hand_frame_to_env: bool = True
    r"""是否按 `anchor_R_eh` / `anchor_p_eh` 自动推导 spawn root pose。

    第一版设计只支持 `True`：spawn 使用 $T_{ea}^{anchor}=T_{eh}^{anchor}T_{ha}$。
    任意 hand orientation 的 episode 级采样应由 reset event 在该 anchor 上右乘扰动。
    """


@configclass
class HandJointInitCfg:
    r"""Hand articulation 初始关节状态配置。"""

    joint_pos: dict[str, float] = field(default_factory=lambda: {".*": 0.0})
    """默认关节位置，key 为 IsaacLab joint regex。"""

    joint_vel: dict[str, float] = field(default_factory=lambda: {".*": 0.0})
    """默认关节速度，key 为 IsaacLab joint regex。"""


@configclass
class HandUrdfSpawnCfg:
    r"""Generated hand URDF importer 参数 scaffold。

    数值锚点来自 `heterogeneous_test_env_cfg.py` 的 generated-hand MVP：它已经通过
    3 个 same-schema post-mutate variants 的 IsaacLab GUI / random-agent smoke。这里
    把这些数值迁移到可复用 adapter，避免每个 GM env 重复维护 URDF importer 细节。
    """

    fix_base: bool = True
    merge_fixed_joints: bool = False
    force_usd_conversion: bool = False
    make_instanceable: bool = True
    r"""是否让 URDF converter 生成 instanceable USD。

    `restore_visual_materials=True` 的 debug 可视化路径会在 child `UrdfFileCfg` 中强制
    设为 `False`：颜色恢复需要在 spawned prim 子树上 author material binding，而 GUI
    模式遍历 instance proxy 曾在第三个 heterogeneous prototype 之后触发 Kit hang。
    动力学 smoke / 训练路径不依赖 URDF debug 色，仍可保持默认 instanceable 优化。
    """

    collision_from_visuals: bool = False
    self_collision: bool = True
    activate_contact_sensors: bool = False
    drive_stiffness: float = 3.0
    drive_damping: float = 0.1


@configclass
class HandActuatorSpawnCfg:
    r"""Generated hand implicit actuator 参数 scaffold。"""

    joint_names_expr: tuple[str, ...] = (".*",)
    effort_limit_sim: float = 0.95
    velocity_limit_sim: float = 8.48
    stiffness: float = 3.0
    damping: float = 0.1
    friction: float = 0.01
    armature: float = 0.001


@configclass
class HandSpawnCfg:
    r"""GM hand spawn 声明式配置 scaffold。

    `bank` 保持嵌套，避免在 GM 层重复 asset-bank schema。便利写法应通过
    `HandBankCfg.containers=("id0", "id1")` 这类资产层接口解决。
    """

    bank: HandBankCfg = field(default_factory=HandBankCfg)
    """资产选择配置；由 `assets.bank.HandBank` 负责解析。"""

    frame: HandFrameCfg = field(default_factory=HandFrameCfg)
    """`{a}->{h}` frame 对齐配置；env cfg 应显式同步到 command cfg。"""

    joint_init: HandJointInitCfg = field(default_factory=HandJointInitCfg)
    """Articulation 初始关节状态。"""

    urdf: HandUrdfSpawnCfg = field(default_factory=HandUrdfSpawnCfg)
    """URDF importer 参数。"""

    actuator: HandActuatorSpawnCfg = field(default_factory=HandActuatorSpawnCfg)
    """Implicit actuator 参数。"""

    spawn_backend: Literal["urdf", "usd"] = "urdf"
    """spawn 后端；`usd` 预留给未来离线 USD cache，第一版应显式报 `NotImplementedError`。"""

    asset_routing: Literal["round_robin", "random_choice"] = "round_robin"
    r"""多资产 env routing。

    `round_robin` 对应 IsaacLab `MultiAssetSpawnerCfg.random_choice=False`，确定且便于
    smoke；`random_choice` 透传 IsaacLab 全局 random，第一版不承诺 seed 可复现。
    """

    restore_visual_materials: bool = False
    """是否在 URDF spawn 后用 `HandContainer.visual_rgba_by_name` 恢复 generated debug color。"""

    validate_same_schema: bool = True
    """是否轻量检查 selection 内所有 assets 的 `topology_name` 与 `dof` 一致。"""


class HandSpawnAdapter:
    r"""`HandSpawnCfg` 的 runtime adapter。

    构造函数保持无 IO；首次访问 `selection` 或构造 articulation 时才调用
    `HandBank.resolve()`。这使 env cfg import 阶段仍保持轻量，而 IsaacLab 真正需要
    spawn cfg 时可以得到完整的 `MultiAssetSpawnerCfg`。
    """

    def __init__(self, cfg: HandSpawnCfg):
        r"""保存配置；不在构造阶段扫描 asset bank。"""

        self.cfg = cfg  # 声明式 hand spawn 配置；不在此处触发文件 IO
        self._selection: HandSelection | None = None  # lazy resolve cache，保持 env import 轻量

    @property
    def selection(self) -> HandSelection:
        r"""Resolved hand selection。

        Returns:
            HandSelection: asset bank resolve 后的有序 hand container 列表。
        """

        if self._selection is None:
            self._selection = HandBank(self.cfg.bank).resolve()  # 解析 hand.urdf / hand.yaml / mesh refs
        return self._selection

    @property
    def semantic_R_ha(self) -> tuple[float, ...]:
        r"""供 env cfg 显式同步到 `ReorientCommandCfg.semantic_R_ha` 的矩阵。"""

        return tuple(float(value) for value in self.cfg.frame.semantic_R_ha)

    def build_articulation_cfg(self, *, prim_path: str) -> ArticulationCfg:
        r"""构造 IsaacLab `ArticulationCfg`。

        Args:
            prim_path (str): scene 中 robot articulation 的 prim path。

        Returns:
            ArticulationCfg: 可直接赋给 `scene.robot` 的 hand articulation 配置。
        """

        if self.cfg.spawn_backend != "urdf":
            raise NotImplementedError(f"HandSpawnAdapter spawn_backend={self.cfg.spawn_backend!r} is not implemented")

        if self.cfg.validate_same_schema:
            _validate_same_hand_schema(self.selection.assets)  # 多资产 articulation 必须同关节 schema

        root_pos_e, root_quat_ea = _compose_anchor_root_pose(self.cfg.frame)  # $T_{ea}=T_{eh}^{anchor}T_{ha}$
        return ArticulationCfg(
            prim_path=prim_path,
            spawn=self.build_multi_hand_spawn_cfg(),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=root_pos_e,
                rot=root_quat_ea,
                joint_pos=dict(self.cfg.joint_init.joint_pos),
                joint_vel=dict(self.cfg.joint_init.joint_vel),
            ),
            actuators={"fingers": _build_implicit_actuator_cfg(self.cfg.actuator)},
            soft_joint_pos_limit_factor=1.0,
        )

    def build_multi_hand_spawn_cfg(self) -> sim_utils.MultiAssetSpawnerCfg:
        r"""构造同拓扑 generated hands 的 `MultiAssetSpawnerCfg`。

        Returns:
            sim_utils.MultiAssetSpawnerCfg: IsaacLab 多资产 spawner 配置。
        """

        if self.cfg.spawn_backend != "urdf":
            raise NotImplementedError(f"HandSpawnAdapter spawn_backend={self.cfg.spawn_backend!r} is not implemented")

        assets = self.selection.assets  # resolved post-mutate hand variants；同一个 spawner 内应为 same-schema
        visual_material_plan = _build_visual_material_restore_plan(assets[0].urdf_path) if (
            self.cfg.restore_visual_materials and len(assets) > 0
        ) else None  # 同拓扑颜色/visual-link contract 只从 reference URDF 解析一次
        assets_cfg = [
            _build_hand_urdf_file_cfg(container, self.cfg, visual_material_plan=visual_material_plan)
            for container in assets
        ]  # 每个 child cfg 对应一个 post-mutate hand variant；材质计划共享
        return sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=assets_cfg,
            random_choice=self.cfg.asset_routing == "random_choice",
            activate_contact_sensors=self.cfg.urdf.activate_contact_sensors,
        )


def _build_hand_urdf_file_cfg(
    container: HandContainer,
    cfg: HandSpawnCfg,
    *,
    visual_material_plan: _VisualMaterialRestorePlan | None = None,
) -> sim_utils.UrdfFileCfg:
    r"""为单个 generated hand container 构造 `UrdfFileCfg`。

    Args:
        container (HandContainer): asset bank 输出的单 hand container。
        cfg (HandSpawnCfg): GM hand spawn 配置。
        visual_material_plan (_VisualMaterialRestorePlan | None): 同拓扑 selection 共享的
            debug color 恢复计划；`None` 表示 wrapper 需要按当前 URDF fallback 解析。

    Returns:
        sim_utils.UrdfFileCfg: IsaacLab URDF importer cfg。
    """

    urdf_cfg = cfg.urdf  # URDF importer 超参锚点，来自 heterogeneous MVP
    # 材质恢复是 GUI/debug 语义：为了给每个 visual name 绑定 URDF RGB，需要 author USD material binding。
    # 若继续生成 instanceable USD，就必须遍历 instance proxy；该路径在 GUI smoke 中出现过 Kit hang。
    make_instanceable = False if cfg.restore_visual_materials else urdf_cfg.make_instanceable

    urdf_file_cfg = sim_utils.UrdfFileCfg(
        asset_path=str(container.urdf_path.resolve()),
        fix_base=urdf_cfg.fix_base,
        merge_fixed_joints=urdf_cfg.merge_fixed_joints,
        force_usd_conversion=urdf_cfg.force_usd_conversion,
        make_instanceable=make_instanceable,
        collision_from_visuals=urdf_cfg.collision_from_visuals,
        self_collision=urdf_cfg.self_collision,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            drive_type="force",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=urdf_cfg.drive_stiffness,
                damping=urdf_cfg.drive_damping,
            ),
        ),
        activate_contact_sensors=urdf_cfg.activate_contact_sensors,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            enable_gyroscopic_forces=False,
            angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=64.0 / math.pi * 180.0,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
            fix_root_link=True,
        ),
    )
    if cfg.restore_visual_materials:
        urdf_file_cfg.func = _spawn_urdf_with_restored_visual_materials  # 只恢复 GUI debug color，不改动力学
        # IsaacLab converter 会 JSON-hash `UrdfFileCfg.to_dict()`；因此挂载的计划必须只含 JSON-safe 数据。
        urdf_file_cfg._anymani_visual_material_plan = (
            _serialize_visual_material_restore_plan(visual_material_plan)
            if visual_material_plan is not None
            else None
        )
    return urdf_file_cfg


def _build_implicit_actuator_cfg(cfg: HandActuatorSpawnCfg) -> ImplicitActuatorCfg:
    r"""构造 generated hand 的 implicit actuator 配置。

    Args:
        cfg (HandActuatorSpawnCfg): hand actuator 数值锚点。

    Returns:
        ImplicitActuatorCfg: IsaacLab articulation actuator cfg。
    """

    return ImplicitActuatorCfg(
        joint_names_expr=list(cfg.joint_names_expr),
        effort_limit_sim=cfg.effort_limit_sim,
        velocity_limit_sim=cfg.velocity_limit_sim,
        stiffness=cfg.stiffness,
        damping=cfg.damping,
        friction=cfg.friction,
        armature=cfg.armature,
    )


def _compose_anchor_root_pose(frame: HandFrameCfg) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    r"""把 hand semantic anchor lower 成 IsaacLab raw root pose。

    核心公式：
    $$
    T_{ea}^{anchor}=T_{eh}^{anchor}T_{ha},\qquad
    R_{ea}=R_{eh}^{anchor}R_{ha},\qquad
    p_{ea}=p_{eh}^{anchor}+R_{eh}^{anchor}p_{ha}.
    $$

    Args:
        frame (HandFrameCfg): `{a}->{h}` 静态校准与 `{h}` 在 `{e}` 中的 anchor。

    Returns:
        tuple[tuple[float, float, float], tuple[float, float, float, float]]: IsaacLab
            `InitialStateCfg` 需要的 `(pos, quat_wxyz)`。
    """

    if not frame.align_hand_frame_to_env:
        raise NotImplementedError("HandFrameCfg.align_hand_frame_to_env=False is reserved for future manual root pose")

    R_ha = _as_matrix3(frame.semantic_R_ha, label="semantic_R_ha")  # $R_{ha}$，raw asset axis -> hand semantic axis
    R_eh = _as_matrix3(frame.anchor_R_eh, label="anchor_R_eh")  # $R_{eh}^{anchor}$，hand semantic axis -> env axis
    p_ha = tuple(float(value) for value in frame.semantic_p_ha)  # $p_{ha}$，raw origin in hand semantic frame, m
    p_eh = tuple(float(value) for value in frame.anchor_p_eh)  # $p_{eh}^{anchor}$，hand semantic origin in env frame, m

    R_ea = _matmul3(R_eh, R_ha)  # $R_{ea}=R_{eh}R_{ha}$，raw asset orientation in env frame
    p_ea = _vec_add3(p_eh, _matvec3(R_eh, p_ha))  # $p_{ea}=p_{eh}+R_{eh}p_{ha}$，raw root position in env frame
    quat_ea = _quat_wxyz_from_matrix3(R_ea)  # IsaacLab boundary 表示，内部语义仍是 $SO(3)$
    return p_ea, quat_ea


def _as_matrix3(values: tuple[float, ...], *, label: str) -> tuple[tuple[float, float, float], ...]:
    r"""把 row-major 9 元组解析为 $3\times3$ 旋转矩阵。"""

    if len(values) != 9:
        raise ValueError(f"{label} must contain 9 row-major values, got {len(values)}")
    scalar_values = tuple(float(value) for value in values)  # row-major $[r_{00},r_{01},...,r_{22}]$
    return (scalar_values[0:3], scalar_values[3:6], scalar_values[6:9])


def _matmul3(
    lhs: tuple[tuple[float, float, float], ...],
    rhs: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    r"""计算 $3\times3$ 矩阵乘法 $C=AB$。"""

    return tuple(
        tuple(sum(lhs[row][k] * rhs[k][col] for k in range(3)) for col in range(3))
        for row in range(3)
    )


def _matvec3(
    matrix: tuple[tuple[float, float, float], ...],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    r"""计算 $3\times3$ 矩阵与三维向量乘法 $y=Rv$。"""

    return tuple(sum(matrix[row][col] * vector[col] for col in range(3)) for row in range(3))


def _vec_add3(lhs: tuple[float, float, float], rhs: tuple[float, float, float]) -> tuple[float, float, float]:
    r"""计算三维平移向量相加 $p=p_1+p_2$。"""

    return tuple(lhs[index] + rhs[index] for index in range(3))


def _quat_wxyz_from_matrix3(matrix: tuple[tuple[float, float, float], ...]) -> tuple[float, float, float, float]:
    r"""把旋转矩阵转换为 IsaacLab `(w,x,y,z)` 四元数。

    该函数只用于 IsaacLab cfg 边界。内部 frame 语义仍以 $R\in SO(3)$ 表达，避免在
    研究代码里把四元数双覆盖问题扩散到上游配置。
    """

    m00, m01, m02 = matrix[0]  # 第一行，row-major $R_{0*}$
    m10, m11, m12 = matrix[1]  # 第二行，row-major $R_{1*}$
    m20, m21, m22 = matrix[2]  # 第三行，row-major $R_{2*}$
    trace = m00 + m11 + m22  # $	ext{tr}(R)$，选择稳定分支的数值锚点
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0  # $s=4q_w$，trace 正时 $q_w$ 分支稳定
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0  # $s=4q_x$，x 对角项主导
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0  # $s=4q_y$，y 对角项主导
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0  # $s=4q_z$，z 对角项主导
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s

    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)  # 数值归一化，抵消浮点舍入误差
    if norm == 0.0:
        raise ValueError("rotation matrix produced a zero quaternion")
    return (qw / norm, qx / norm, qy / norm, qz / norm)


def _validate_same_hand_schema(containers: tuple[HandContainer, ...]) -> None:
    r"""检查 MultiAssetSpawner 内所有 hands 是否共享 articulation schema。

    IsaacLab 的一个 batched `Articulation` 要求各 prototype 的 joint/body schema 兼容。
    这里用 sidecar 中最直接的科研语义字段做 fail-fast：topology、DOF、slot 顺序、
    每根手指的 revolute DOF，以及完整有序 revolute-joint name sequence。

    后一项是 action / observation schema 的必要条件。Isaac Lab 的
    `joint_names=[".*"], preserve_order=True` 只保留 articulation importer 已有顺序，
    不会把相同 joint-name 集合自动重排到 canonical 顺序。因此同拓扑资产一旦 joint
    sequence 不同，就不能进入同一个 batched articulation。
    """

    if len(containers) == 0:
        raise ValueError("HandSpawnAdapter requires at least one selected hand asset")

    reference = _hand_schema_signature(containers[0])  # 第一个 asset 作为 same-schema 参照
    for container in containers[1:]:
        signature = _hand_schema_signature(container)  # 当前 asset 的 sidecar schema 摘要
        if signature != reference:
            raise ValueError(
                "selected hand assets are not same-schema: "
                f"reference={containers[0].asset_id}:{reference!r}, "
                f"offender={container.asset_id}:{signature!r}"
            )


def _hand_schema_signature(container: HandContainer) -> tuple[object, ...]:
    r"""从 `hand.yaml` sidecar 抽取 same-schema 有序签名。

    Returns:
        tuple[object, ...]: topology、DOF、slot、finger summary 和 ordered revolute
        joint names。最后一项直接定义 batched action 第二维的关节语义。

    Raises:
        ValueError: `hand_cfg` 不完整、joint name 缺失，或解析到的 revolute joint
        数量与顶层 `dof` 不一致时抛出。
    """

    sidecar = container.sidecar  # generated hand sidecar，保持 dict 以兼容资产 schema 演化
    finger_signature = tuple(
        (finger.get("name"), finger.get("revolute_dof"))
        for finger in sidecar.get("fingers", [])
    )  # 有序 finger schema，避免同 DOF 但 finger routing 不同的资产混入
    joint_sequence = _ordered_revolute_joint_names(sidecar, asset_id=container.asset_id)  # `[J]`，canonical joint order
    return (
        sidecar.get("topology_name"),
        sidecar.get("dof"),
        tuple(sidecar.get("surviving_slots", [])),
        finger_signature,
        joint_sequence,
    )


def _ordered_revolute_joint_names(sidecar: dict[str, object], *, asset_id: str) -> tuple[str, ...]:
    r"""按 sidecar finger/joint 顺序提取 revolute articulation joint names。

    Generated exporter 依次遍历 `hand_cfg.fingers` 与 `finger.joints` 写 URDF；这里沿用
    同一顺序构造 canonical schema：

    $$
    \mathcal J=(j_0,j_1,\ldots,j_{J-1}),\qquad J=\texttt{sidecar.dof}.
    $$

    fixed joints 只改变 link hierarchy，不进入 policy action，因此从序列中排除。

    Args:
        sidecar (dict[str, object]): generated `hand.yaml` 内容。
        asset_id (str): 当前资产 id，写入 fail-fast 错误消息。

    Returns:
        tuple[str, ...]: 有序 revolute joint 名称，长度 $J$。

    Raises:
        ValueError: sidecar 结构、joint 类型/名称或 DOF closure 不合法时抛出。
    """

    hand_cfg = sidecar.get("hand_cfg")  # 完整 generated hand schema；顶层 summary 不含 joint names
    if not isinstance(hand_cfg, dict):
        raise ValueError(f"asset {asset_id!r} sidecar must provide mapping hand_cfg for joint-order validation")
    fingers = hand_cfg.get("fingers")  # 有序 finger 列表；顺序与 URDF exporter 一致
    if not isinstance(fingers, list):
        raise ValueError(f"asset {asset_id!r} sidecar hand_cfg.fingers must be a list")

    joint_names: list[str] = []  # 只收集 policy 可控 revolute joints，保持 sidecar 遍历顺序
    for finger_index, finger_cfg in enumerate(fingers):
        if not isinstance(finger_cfg, dict) or not isinstance(finger_cfg.get("joints"), list):
            raise ValueError(f"asset {asset_id!r} hand_cfg.fingers[{finger_index}].joints must be a list")
        for joint_index, joint_cfg in enumerate(finger_cfg["joints"]):
            if not isinstance(joint_cfg, dict):
                raise ValueError(
                    f"asset {asset_id!r} hand_cfg.fingers[{finger_index}].joints[{joint_index}] must be a mapping"
                )
            if joint_cfg.get("joint_type") != "revolute":
                continue  # fixed joints 建立 link chain，但不占 action / observation slot
            joint_name = joint_cfg.get("name")  # articulation joint name，必须与 URDF `<joint name=...>` 相同
            if not isinstance(joint_name, str) or not joint_name:
                raise ValueError(f"asset {asset_id!r} has a revolute joint without a non-empty name")
            joint_names.append(joint_name)

    expected_dof = sidecar.get("dof")  # 顶层 exporter summary 中的可控 DOF 数 $J$
    if not isinstance(expected_dof, int) or len(joint_names) != expected_dof:
        raise ValueError(
            f"asset {asset_id!r} ordered revolute-joint count {len(joint_names)} does not match dof={expected_dof!r}"
        )
    if len(set(joint_names)) != len(joint_names):
        raise ValueError(f"asset {asset_id!r} ordered revolute-joint sequence contains duplicate names: {joint_names!r}")
    return tuple(joint_names)  # tuple 使 schema signature 可哈希、可直接精确比较


def _spawn_urdf_with_restored_visual_materials(
    prim_path: str,
    cfg: sim_utils.UrdfFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    r"""官方 URDF spawn 后恢复 generated hand 的 per-visual debug color。

    该 wrapper 只修复 GUI / render 语义。动力学资产仍由 IsaacLab 官方
    `spawn_from_urdf` 创建；collision、mass、joint、drive、root pose 均不在这里改变。
    """

    from isaaclab.sim.spawners.from_files import spawn_from_urdf

    spawned_prim = spawn_from_urdf(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    visual_material_plan = _deserialize_visual_material_restore_plan(
        getattr(cfg, "_anymani_visual_material_plan", None)
    )  # adapter 预计算的同拓扑共享计划
    if visual_material_plan is None:
        visual_material_plan = _build_visual_material_restore_plan(Path(cfg.asset_path))  # direct wrapper 使用时的安全 fallback
    _restore_visual_materials_on_spawned_prim(
        spawned_prim,
        visual_material_plan.visual_rgba_by_name,
        visual_material_plan.visual_link_by_name,
    )
    return spawned_prim


def _build_visual_material_restore_plan(urdf_path: Path) -> _VisualMaterialRestorePlan:
    r"""从一个 reference URDF 构造同拓扑 variants 共享的 debug color 恢复计划。

    Args:
        urdf_path (Path): reference `hand.urdf`。对于 post-mutate same-topology selection，
            只使用第一个 selected asset 即可，因为 visual/link/color contract 是 topology-level。

    Returns:
        _VisualMaterialRestorePlan: 可挂到多个 child `UrdfFileCfg` 上复用的颜色恢复计划。
    """

    resolved_urdf_path = Path(urdf_path).expanduser().resolve(strict=False)
    return _VisualMaterialRestorePlan(
        source_urdf_path=resolved_urdf_path,
        visual_rgba_by_name=parse_urdf_visual_rgba_by_name(resolved_urdf_path),
        visual_link_by_name=_parse_urdf_visual_link_by_name(resolved_urdf_path),
    )


def _parse_urdf_visual_link_by_name(urdf_path: Path) -> dict[str, str]:
    r"""解析 URDF visual name 对应的 parent link name。

    Isaac Sim URDF importer 会把每个 link 的可视几何组织到 spawned USD 的
    `/<link_name>/visuals` 可编辑 prim 下；真正的 URDF visual name（例如
    `palm_visual`）位于该 prim 的 instance proxy 子树里。为了避免 GUI 模式遍历
    instance proxy 导致 Kit hang，本函数从原始 URDF 直接恢复
    `visual_name -> link_name` 映射，再把 material 绑定到 `/<link_name>/visuals`。

    Args:
        urdf_path (Path): 已解析到真实磁盘的 `hand.urdf` 路径。

    Returns:
        dict[str, str]: URDF visual name 到 parent link name 的映射。
    """

    resolved_urdf_path = Path(urdf_path).expanduser().resolve(strict=False)
    if not resolved_urdf_path.is_file():
        raise FileNotFoundError(f"URDF file does not exist: {resolved_urdf_path}")

    root = ET.parse(resolved_urdf_path).getroot()  # URDF XML root，纯解析，不触碰 USD/Isaac state
    link_by_visual_name: dict[str, str] = {}
    for link_elem in root.findall("./link"):
        link_name = link_elem.attrib.get("name")  # URDF link name，对应 spawned USD 的一级 body prim
        if not link_name:
            continue
        for visual_elem in link_elem.findall("./visual"):
            visual_name = visual_elem.attrib.get("name")  # URDF visual name，对应 debug color key
            if visual_name:
                link_by_visual_name[visual_name] = link_name
    return link_by_visual_name


def _restore_visual_materials_on_spawned_prim(
    spawned_prim,
    visual_rgba_by_name: dict[str, UrdfRgba],
    visual_link_by_name: dict[str, str],
) -> None:
    r"""在 spawned hand prim 子树上绑定 URDF visual colors。

    颜色恢复只服务 GUI/debug 语义，不改变 collision、mass、joint、drive 或 root pose。
    绑定目标优先选择 `/<link_name>/visuals` 这个可编辑 ancestor，而不是进入
    instance proxy 里的真实 mesh prim；这样能保留 URDF 颜色，又规避 GUI hang 风险。
    """

    if len(visual_rgba_by_name) == 0:
        return

    visual_prims = _find_spawned_visual_prims_by_name(spawned_prim, visual_link_by_name)
    bound_target_by_path: dict[str, str] = {}
    missing_visual_names: list[str] = []

    for visual_name, rgba in visual_rgba_by_name.items():
        visual_prim = visual_prims.get(visual_name)
        if visual_prim is None:
            missing_visual_names.append(visual_name)
            continue

        target_prim = _nearest_editable_material_binding_prim(visual_prim)
        target_path = str(target_prim.GetPath())
        previous_visual_name = bound_target_by_path.get(target_path)
        if previous_visual_name is not None and visual_rgba_by_name[previous_visual_name][:3] != rgba[:3]:
            logger.warning(
                "Skip URDF visual color for %s because editable USD target %s was already bound for %s.",
                visual_name,
                target_path,
                previous_visual_name,
            )
            continue

        try:
            _bind_urdf_preview_surface(spawned_prim, target_prim, visual_name, rgba)
        except Exception as exc:
            logger.warning("Failed to restore URDF visual color for %s on %s: %s", visual_name, target_path, exc)
            continue

        bound_target_by_path[target_path] = visual_name

    if missing_visual_names:
        logger.warning(
            "Could not find %d URDF visual prims under spawned hand %s; examples: %s",
            len(missing_visual_names),
            spawned_prim.GetPath(),
            missing_visual_names[:5],
        )


def _find_spawned_visual_prims_by_name(spawned_prim, visual_link_by_name: dict[str, str]) -> dict[str, object]:
    r"""在 spawned hand 子树内查找每个 URDF visual 对应的 editable USD target prim。

    本函数刻意不使用 `Usd.TraverseInstanceProxies()`。URDF debug color 是纯可视化注解，
    不值得为了进入 instance proxy 子树承担 GUI hang 风险；调用方在
    `restore_visual_materials=True` 时会把 material 绑定到 `/<link_name>/visuals`，该 prim
    是 instanceable 但不是 instance proxy，仍然允许 author material binding。
    """

    visual_prims: dict[str, object] = {}
    stage = spawned_prim.GetStage()
    root_path = str(spawned_prim.GetPath())
    for visual_name, link_name in visual_link_by_name.items():
        target_path = f"{root_path}/{link_name}/visuals"  # editable visual ancestor；内部 mesh 可能是 instance proxy
        target_prim = stage.GetPrimAtPath(target_path)
        if target_prim.IsValid():
            visual_prims[visual_name] = target_prim
    return visual_prims


def _nearest_editable_material_binding_prim(visual_prim):
    r"""为 material binding 选择最近的非 instance-proxy ancestor。"""

    target_prim = visual_prim
    while target_prim.IsInstanceProxy():
        parent_prim = target_prim.GetParent()
        if not parent_prim.IsValid():
            break
        target_prim = parent_prim
    return target_prim


def _bind_urdf_preview_surface(spawned_prim, target_prim, visual_name: str, rgba: UrdfRgba) -> None:
    r"""创建并绑定一个表示 URDF RGB 的 USD PreviewSurface material。"""

    from pxr import UsdShade

    stage = spawned_prim.GetStage()
    root_path = str(spawned_prim.GetPath())
    looks_path = f"{root_path}/Looks"
    material_path = f"{looks_path}/{_sanitize_usd_prim_name('urdf_' + visual_name)}"

    if not stage.GetPrimAtPath(looks_path).IsValid():
        stage.DefinePrim(looks_path, "Scope")

    if not stage.GetPrimAtPath(material_path).IsValid():
        material_cfg = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(rgba[0], rgba[1], rgba[2]),
            roughness=0.5,
            metallic=0.0,
        )
        material_cfg.func(material_path, material_cfg)

    material = UsdShade.Material(stage.GetPrimAtPath(material_path))
    if target_prim.HasAPI(UsdShade.MaterialBindingAPI):
        material_binding_api = UsdShade.MaterialBindingAPI(target_prim)
    else:
        material_binding_api = UsdShade.MaterialBindingAPI.Apply(target_prim)
    material_binding_api.Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)


def _sanitize_usd_prim_name(raw_name: str) -> str:
    r"""把 URDF visual name 转成保守合法的 USD prim name 片段。"""

    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", raw_name)
    if sanitized == "" or sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized


__all__ = [
    "DEFAULT_HAND_ANCHOR_POS_E",
    "HandActuatorSpawnCfg",
    "HandFrameCfg",
    "HandJointInitCfg",
    "HandSpawnAdapter",
    "HandSpawnCfg",
    "HandUrdfSpawnCfg",
    "_compose_anchor_root_pose",
    "_spawn_urdf_with_restored_visual_materials",
]
