r"""Bind one generated hand asset into the `gm` task scene.

这一层不是 asset bank 管理器。它只描述“给定一个已经由 `distill`
选中的 hand asset，`tasks/gm` 需要怎样把它变成 Isaac Lab 的
`ArticulationCfg`”。

最低输入 contract：

- `hand.urdf`：当前 hand 的可加载 URDF，mesh 路径应相对自身目录闭合；
- `hand.yaml`：sidecar 元数据，至少包含 `family`、`handedness`、`dof`、
  `finger_count`、`topology_name`、`surviving_slots`、`slot_family_map`、
  `per_finger_connectivity`；
- same-topology RL 主线要求一批资产共享 action joint schema。跨拓扑 padding
  / mask / token 化是后续 `distill/models` 问题，不在本文件解决。

TODO:
    后续实现 `build_hand_articulation_cfg(...)` 时，应只做薄绑定：
    路径校验、URDF importer 参数、初始位姿、actuator 默认值、joint order
    contract 提取。不要在这里扫描 generated root，也不要决定 train/heldout split。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.converters import UrdfConverterCfg

DEFAULT_HAND_INIT_POS = (0.0, 0.0, 0.5)
r"""生成手在 env frame `{e}` 下的默认 root 位置（单位: m）。

当前数值沿用旧 LeapHand in-hand task 的手部高度锚点，使 object 默认位置
`(0,-0.10,0.56)` 位于掌面附近，便于无 Grasp Cache 的第一版 smoke。
"""


DEFAULT_HAND_INIT_ROT = (0.5, 0.5, -0.5, 0.5)
r"""生成手 root 的默认四元数 `(w,x,y,z)`。

这是旧 LeapHand IsaacLab 配置中的视觉/操作坐标系锚点。第一版先复用它，
因为当前目标是验证 generated URDF 能进入 same-topology 训练闭环；严格的
`{a}->{h}` 语义校准后续由 asset metadata / distill manifest 显式记录。
"""


@dataclass(frozen=True)
class GmHandAssetRef:
    r"""A selected generated hand asset consumed by `gm`.

    Args:
        root_dir (Path): 单个 hand bundle 目录；通常包含 `hand.urdf` 与 `hand.yaml`。
        topology_name (str | None): 可选拓扑名，用于训练 manifest 与日志核对。
        asset_id (str | None): 可选样本 ID；post-mutate 样本通常有短 hash。

    NOTE:
        这个 dataclass 表达的是“已选资产引用”，不是“资产库”。如果未来需要
        64 / 128 个 assets 的采样、分段训练或 heldout eval，应在 `distill`
        侧生成一组 `GmHandAssetRef`，再逐段交给环境配置。
    """

    root_dir: Path
    topology_name: str | None = None
    asset_id: str | None = None

    @property
    def urdf_path(self) -> Path:
        r"""Return the expected URDF path for the selected hand."""

        return self.root_dir / "hand.urdf"

    @property
    def sidecar_path(self) -> Path:
        r"""Return the expected sidecar metadata path for the selected hand."""

        return self.root_dir / "hand.yaml"


def _require_file(path: Path, *, label: str) -> Path:
    r"""检查单个 hand bundle 文件是否存在。

    Args:
        path (Path): 待检查路径。
        label (str): 错误消息中的科研语义标签，例如 `hand.urdf` 或 `hand.yaml`。

    Returns:
        Path: 解析后的绝对路径。

    Raises:
        FileNotFoundError: 当 bundle 缺失最低 contract 文件时抛出。
    """

    resolved = path.expanduser().resolve()  # 绝对路径，避免 Isaac/URDF importer 受 cwd 影响
    if not resolved.is_file():
        raise FileNotFoundError(f"Selected gm hand asset is missing {label}: {resolved}")
    return resolved


def build_hand_articulation_cfg(asset: GmHandAssetRef, *, prim_path: str) -> ArticulationCfg:
    r"""Build an Isaac Lab articulation cfg for one selected generated hand.

    这是 `gm` 对 generated asset 的薄绑定层。函数固定后续数据流：

    $$
    \texttt{GmHandAssetRef} \rightarrow \texttt{ArticulationCfg}
    \rightarrow \texttt{GmInHandSceneCfg.robot}
    $$

    Args:
        asset (GmHandAssetRef): `distill` 或 debug cfg 已经选好的单个 hand bundle。
        prim_path (str): Isaac Lab scene 中 robot articulation 的 prim path。

    Returns:
        ArticulationCfg: 可被 `scene.robot` 消费的 articulation config。

    Raises:
        FileNotFoundError: 当 `hand.urdf` 或 `hand.yaml` 缺失时抛出。

    DONE:
        当前第一版使用 `sim_utils.UrdfFileCfg(asset_path=..., fix_base=True, ...)`
        在线转换 URDF。它适合 debug / smoke；若追求 4096/8192 env 并行，后续
        应在 `distill` 的训练 manifest 中缓存离线 USD 路径，再切换到 `UsdFileCfg`。
    """

    urdf_path = _require_file(asset.urdf_path, label="hand.urdf")  # generated 手的运动学/几何主文件
    _require_file(asset.sidecar_path, label="hand.yaml")  # sidecar 暂不解析，但必须存在以保证 provenance 闭包

    # URDF importer 负责把 hand.urdf 转成 USD 后 spawn。`fix_base=True` 保持手掌/root 静止，
    # 使第一版 teacher 专注 object-in-hand reorientation，而非同时学习手臂/手腕位姿控制。
    spawn_cfg = sim_utils.UrdfFileCfg(
        asset_path=str(urdf_path),  # 绝对路径，URDF 内 mesh relpath 仍相对 hand.urdf 自身解析
        fix_base=True,  # 固定 hand root；当前 task 是手内操作，不训练手腕自由体
        merge_fixed_joints=False,  # 保留 fixed tip/root 结构，便于后续 token / fingertip 语义核对
        force_usd_conversion=False,  # debug 阶段允许 IsaacLab 复用转换 cache，减少反复启动成本
        make_instanceable=True,  # 同一 asset 多 env clone 时共享 USD 实例，降低显存/内存压力
        collision_from_visuals=False,  # exporter 已写 collision mesh，不从 visual 反推碰撞几何
        self_collision=True,  # 灵巧手接触任务需要手指自碰撞真实暴露，后续可按吞吐消融
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="position",  # action term 输出 joint position target，而不是 velocity/torque
            drive_type="force",  # 保持与 URDF/IsaacLab implicit actuator 的力矩限幅语义一致
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=3.0, damping=0.1),
        ),
        activate_contact_sensors=False,  # 第一版不启用 fingertip contact reward，避免先背负 sensor schema
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,  # 手是固定基座 articulation，link 不应受重力整体下坠
            retain_accelerations=False,  # 与旧 LeapHand cfg 对齐，减少跨步残余加速度噪声
            enable_gyroscopic_forces=False,  # 手 link 小惯量下先关闭陀螺项，优先稳定 debug
            angular_damping=0.01,  # 轻微角阻尼，抑制 URDF 转换初期的高频抖动
            max_linear_velocity=1000.0,  # 宽松上限，避免 solver 人为截断指尖接触速度
            max_angular_velocity=64.0 / 3.141592653589793 * 180.0,  # 沿用旧 LeapHand 数值锚点
            max_depenetration_velocity=1000.0,  # 接触解穿透速度上限，避免深穿透后无法分离
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,  # finger-finger / finger-palm 接触对稳定抓持有实际影响
            solver_position_iteration_count=8,  # in-hand 接触密集，位置迭代数先取旧任务锚点
            solver_velocity_iteration_count=0,  # 与旧 LeapHand cfg 对齐，先不引入额外速度迭代成本
            sleep_threshold=0.005,  # 小速度下允许休眠，减少静止手 link 的 solver 负担
            stabilization_threshold=0.0005,  # 保持旧 LeapHand 的稳定化阈值
            fix_root_link=True,  # 双重声明固定 root，避免 importer / articulation 属性不一致
        ),
    )

    # ImplicitActuatorCfg 让 PhysX 处理 PD drive；其 joint_names_expr 使用 `.*`，
    # 因为 same-topology action order 校验在训练 manifest / distill 侧完成，而非这里扫描资产库。
    return ArticulationCfg(
        prim_path=prim_path,  # IsaacLab scene 中 `{ENV_REGEX_NS}/robot`
        spawn=spawn_cfg,  # URDF -> USD -> articulation 的 spawn 配置
        init_state=ArticulationCfg.InitialStateCfg(
            pos=DEFAULT_HAND_INIT_POS,  # `{e}` 下 hand root 位置，单位 m
            rot=DEFAULT_HAND_INIT_ROT,  # `{e}` 下 hand root 姿态 `(w,x,y,z)`
            joint_pos={".*": 0.0},  # home pose；no-cache 第一版不从 grasp cache 写稳定抓持姿态
            joint_vel={".*": 0.0},  # reset 后关节速度清零，避免初态带入随机动量
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*"],  # 所有 revolute joints；fixed joints 不会成为 actuator joint
                effort_limit_sim=0.95,  # URDF LEAP 风格 effort 数值锚点，单位 N*m
                velocity_limit_sim=8.48,  # URDF LEAP 风格 velocity 数值锚点，单位 rad/s
                stiffness=3.0,  # PD 位置刚度，沿用旧 LeapHand debug 值
                damping=0.1,  # PD 阻尼，沿用旧 LeapHand debug 值
                friction=0.01,  # 轻微 joint friction，抑制无意义抖动
                armature=0.001,  # 小 armature，改善小连杆数值稳定性
            ),
        },
        soft_joint_pos_limit_factor=1.0,  # soft limit 等于 URDF hard limit，保持 action clamp 可解释
    )


__all__ = [
    "DEFAULT_HAND_INIT_POS",
    "DEFAULT_HAND_INIT_ROT",
    "GmHandAssetRef",
    "build_hand_articulation_cfg",
]
