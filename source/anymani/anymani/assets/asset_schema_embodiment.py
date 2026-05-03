"""手部资产声明的 embodiment 层 schema。

本模块开始进入“手是什么”这一层语义。相比 `asset_schema_core.py`
中的基础描述对象，这里定义的是更接近手部结构本身的对象：

- `JointCfg`
- `FingerCfg`
- `PalmCfg`
- `HandCfg`

它们共同构成当前项目中最核心的 canonical hand description。
generator、validator、exporter 等流水线模块，都以这一层对象作为
主要读写接口。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from .asset_schema_core import (
    AssetCfgBase,
    CollisionGeometryCfg,
    Handedness,
    InertialCfg,
    JointLimitCfg,
    JointPropertiesCfg,
    JointType,
    MimicCfg,
    PoseCfg,
    Vector3,
    _FLOAT_TOLERANCE,
    _ensure_list,
    _ensure_tuple,
    _make_collision_cfg,
    _make_visual_cfg,
    _normalize_axis,
    _sanitize_identifier,
)
from .asset_schema_core import VisualGeometryCfg


@dataclass
class JointCfg(AssetCfgBase):
    r"""以 joint 为中心的局部结构描述。

    当前建模约定下，一个 `JointCfg` 同时负责描述：

    - 该 joint 自身的 kinematic 属性；
    - 该 joint 所连接的 child link；
    - 该 child link 的 collision / visual / inertial。

    这种 joint-centric 组织方式是刻意选择的。原因在于当前研究更关心
    “关节之后那一段 finger skin / collision primitive 如何参数化”，
    因此把 child link 几何放在 joint 下面会更符合 joint-level 资产生成
    的思考方式。
    """

    name: str
    """当前 joint 名称，也是后续索引与导出时常用的主键。"""

    parent: str
    """父 link 名称。"""

    joint_type: JointType = "revolute"
    """当前项目范围内支持的 joint 类型。"""

    child: str | None = None
    """子 link 名称；若省略则自动派生。"""

    axis: Vector3 = (0.0, 0.0, 1.0)
    """关节轴方向 $\vec{a}$。"""

    limit: JointLimitCfg | Mapping[str, Any] | Sequence[float] | None = (-3.141592653589793, 3.141592653589793)
    """关节限位，可写为对象 / 字典 / `(lower, upper)` 简写。"""

    origin: PoseCfg | Sequence[float] | Mapping[str, Any] | None = field(default_factory=PoseCfg)
    """关节坐标系相对父 link 坐标系的局部位姿。"""

    inertial: InertialCfg | Mapping[str, Any] | None = None
    """子 link 的惯性描述。"""

    joint_properties: JointPropertiesCfg | Mapping[str, Any] | None = None
    r"""joint 级附加物理属性。

    这里和 `inertial` 刻意分开：

    - `joint_properties` 跟随当前 revolute joint 的广义坐标 $q$；
    - `inertial` 跟随当前 joint 所携带的 child link 刚体；
    - pre-made 的 joint delete 删除的是这一整对 joint-child embodiment。
    """

    collisions: list[CollisionGeometryCfg] = field(default_factory=list)
    """子 link 的 collision 几何列表。"""

    visuals: list[VisualGeometryCfg] = field(default_factory=list)
    """子 link 的 visual 几何列表。"""

    mimic: MimicCfg | Mapping[str, Any] | None = None
    """可选 mimic 关系；当前仅保留在 schema 中，generator v1 不消费。"""

    is_tip: bool = False
    """当前 joint/link 对是否在 v1 中被视为 fingertip 相关。"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """预留扩展 metadata。"""

    def __post_init__(self):
        # 名称规范化在 schema 层完成，避免明显非法名字流入后续导出与仿真。
        self.name = _sanitize_identifier(self.name, field_name="joint.name")
        if self.joint_type not in {"revolute", "fixed"}:
            raise ValueError(f"invalid joint_type: {self.joint_type}, must be 'revolute' or 'fixed'")
        self.parent = _sanitize_identifier(self.parent, field_name="joint.parent")
        self.child = _sanitize_identifier(self.child or f"{self.name}_link", field_name="joint.child")
        self.origin = PoseCfg.from_value(self.origin)

        axis_tuple = _ensure_tuple(self.axis, length=3, field_name="joint.axis")
        if self.joint_type == "fixed" and all(abs(value) <= _FLOAT_TOLERANCE for value in axis_tuple):
            # 固定关节没有真实转轴语义，因此允许零轴输入，并回填成一个
            # 约定默认值，避免后续代码处理 `None` 或零向量特判。
            self.axis = (0.0, 0.0, 1.0)
        else:
            self.axis = _normalize_axis(axis_tuple)

        if self.limit is None:
            if self.joint_type != "fixed":
                raise ValueError("Non-fixed joint must provide limit")
        elif isinstance(self.limit, JointLimitCfg):
            self.limit = self.limit.copy()
        elif isinstance(self.limit, Mapping):
            self.limit = JointLimitCfg(**self.limit)
        elif isinstance(self.limit, Sequence) and not isinstance(self.limit, (str, bytes)):
            packed = _ensure_tuple(self.limit, length=2, field_name="joint.limit")
            self.limit = JointLimitCfg(lower=packed[0], upper=packed[1])
        else:
            raise TypeError(f"Unsupported joint limit: {self.limit!r}")

        if self.joint_properties is not None and not isinstance(self.joint_properties, JointPropertiesCfg):
            if not isinstance(self.joint_properties, Mapping):
                raise TypeError(
                    f"joint_properties must be JointPropertiesCfg or mapping, got {self.joint_properties!r}"
                )
            self.joint_properties = JointPropertiesCfg(**self.joint_properties)

        if self.inertial is not None and not isinstance(self.inertial, InertialCfg):
            if not isinstance(self.inertial, Mapping):
                raise TypeError(f"inertial must be InertialCfg or mapping, got {self.inertial!r}")
            self.inertial = InertialCfg(**self.inertial)

        # 在 schema 边界内统一收敛 collision / visual 这类宽松输入。
        self.collisions = [_make_collision_cfg(item) for item in _ensure_list(self.collisions, field_name="collisions")]
        self.visuals = [_make_visual_cfg(item) for item in _ensure_list(self.visuals, field_name="visuals")]

        if self.mimic is not None and not isinstance(self.mimic, MimicCfg):
            if not isinstance(self.mimic, Mapping):
                raise TypeError(f"mimic must be MimicCfg or mapping, got {self.mimic!r}")
            self.mimic = MimicCfg(**self.mimic)

    @property
    def dof_count(self) -> int:
        """该 joint 提供的自由度数量。"""

        return 0 if self.joint_type == "fixed" else 1

    @property
    def uses_only_primitive_collision(self) -> bool:
        """该 child link 是否仅使用 primitive collision 几何。"""

        return all(collision.geometry.is_primitive for collision in self.collisions)


@dataclass
class FingerCfg(AssetCfgBase):
    r"""由串联 joint 链构成的逻辑手指描述。"""

    name: str
    """逻辑 finger 名称。"""

    parent_link: str = "palm"
    """该 finger 挂载到的 link，默认是 palm。"""

    mount: PoseCfg | Sequence[float] | Mapping[str, Any] | None = field(default_factory=PoseCfg)
    """手指级挂载位姿入口。"""

    joints: list[JointCfg] = field(default_factory=list)
    """构成该 finger 的串联 joint 列表。"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """预留扩展 metadata。"""

    def __post_init__(self):
        self.name = _sanitize_identifier(self.name, field_name="finger.name")
        self.parent_link = _sanitize_identifier(self.parent_link, field_name="finger.parent_link")
        self.mount = PoseCfg.from_value(self.mount)
        self.joints = [joint if isinstance(joint, JointCfg) else JointCfg(**joint) for joint in self.joints]

        # 手指至少要形成一条非空链，否则在当前项目语义下不构成有效手指。
        if not self.joints:
            raise ValueError(f"finger '{self.name}' must contain at least one joint")

        first_parent = self.joints[0].parent
        if first_parent != self.parent_link:
            raise ValueError(
                f"finger '{self.name}' first joint parent must be '{self.parent_link}', got '{first_parent}'"
            )

        # 对串联手指，后一关节的 parent 必须严格等于前一关节的 child。
        # 这一步在 schema 层就尽量拦截“运动学链断裂”的错误。
        for previous, current in zip(self.joints[:-1], self.joints[1:]):
            if current.parent != previous.child:
                raise ValueError(
                    f"finger '{self.name}' chain broken: joint '{current.name}' parent is "
                    f"'{current.parent}', expected '{previous.child}'"
                )

        joint_names = [joint.name for joint in self.joints]
        if len(joint_names) != len(set(joint_names)):
            raise ValueError(f"finger '{self.name}' contains duplicated joint names: {joint_names}")

    @property
    def joint_names(self) -> list[str]:
        """返回 finger 内 joint 名称列表。"""

        return [joint.name for joint in self.joints]

    @property
    def tip_joint(self) -> JointCfg:
        """返回 finger 末端 joint。"""

        return self.joints[-1]

    @property
    def tip_link(self) -> str:
        """返回 finger 末端 link 名称。"""

        return cast(str, self.tip_joint.child)

    @property
    def dof_count(self) -> int:
        """返回 finger 的总自由度数。"""

        return sum(joint.dof_count for joint in self.joints)


@dataclass
class PalmCfg(AssetCfgBase):
    r"""手掌 / 根 link 描述。

    当前 v1 中，`PalmCfg` 与 `FingerCfg` 仍保持分离，而不强行抽象成
    “若干 joint 构成的一般 assembly”。这样做是为了保留手掌在建模上的
    特殊地位：它既是整手的根，也常常是多个 finger 的共同挂载基座。
    """

    name: str = "palm"
    """手掌 / root link 名称。"""

    origin: PoseCfg | Sequence[float] | Mapping[str, Any] | None = field(default_factory=PoseCfg)
    """掌部坐标系相对整手根参考系的位姿。"""

    inertial: InertialCfg | Mapping[str, Any] | None = None
    """掌部的惯性描述。"""

    collisions: list[CollisionGeometryCfg] = field(default_factory=list)
    """掌部的 collision 几何列表。"""

    visuals: list[VisualGeometryCfg] = field(default_factory=list)
    """掌部的 visual 几何列表。"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """预留扩展 metadata。"""

    def __post_init__(self):
        self.name = _sanitize_identifier(self.name, field_name="palm.name")
        self.origin = PoseCfg.from_value(self.origin)
        if self.inertial is not None and not isinstance(self.inertial, InertialCfg):
            if not isinstance(self.inertial, Mapping):
                raise TypeError(f"inertial must be InertialCfg or mapping, got {self.inertial!r}")
            self.inertial = InertialCfg(**self.inertial)
        self.collisions = [_make_collision_cfg(item) for item in _ensure_list(self.collisions, field_name="collisions")]
        self.visuals = [_make_visual_cfg(item) for item in _ensure_list(self.visuals, field_name="visuals")]


@dataclass
class HandCfg(AssetCfgBase):
    r"""整手的 canonical 顶层描述。

    在当前子项目里，`HandCfg` 是最重要的统一接口对象。它并不是某个
    runtime-only 容器，而是生成、校验、导出等后续流程共享的“规范描述”。
    换句话说，builder 的目标不是发明另一套 hand runtime object，而是
    构造一个稳定、可检查、可导出的 `HandCfg`。
    """

    name: str
    """手资产名称。"""

    palm: PalmCfg | Mapping[str, Any] = field(default_factory=PalmCfg)
    """手掌 / root-link 配置。"""

    fingers: list[FingerCfg] = field(default_factory=list)
    """整手包含的所有 finger。"""

    family: str = "generic"
    """手 family 标签，例如 `leap`、`allegro` 或 `generic`。"""

    handedness: Handedness = "unknown"
    """左右手标签；`unknown` 预留给非典型结构。"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """预留扩展 metadata。"""

    def __post_init__(self):
        self.name = _sanitize_identifier(self.name, field_name="hand.name")
        self.family = _sanitize_identifier(self.family, field_name="hand.family")
        if self.handedness not in {"left", "right", "unknown"}:
            raise ValueError(f"invalid handedness: {self.handedness}")

        if not isinstance(self.palm, PalmCfg):
            if not isinstance(self.palm, Mapping):
                raise TypeError(f"palm must be PalmCfg or mapping, got {self.palm!r}")
            self.palm = PalmCfg(**self.palm)

        self.fingers = [finger if isinstance(finger, FingerCfg) else FingerCfg(**finger) for finger in self.fingers]
        if not self.fingers:
            raise ValueError("hand must contain at least one finger")

        # 这里统一做全局唯一性检查，是为了尽量把“结构明显不一致”的 hand
        # 在 schema 层就拦住，而不是拖到 exporter 或仿真阶段才暴露。
        finger_names = [finger.name for finger in self.fingers]
        if len(finger_names) != len(set(finger_names)):
            raise ValueError(f"hand contains duplicated finger names: {finger_names}")

        all_joint_names = [joint.name for joint in self.iter_joints()]
        if len(all_joint_names) != len(set(all_joint_names)):
            raise ValueError(f"hand contains duplicated joint names: {all_joint_names}")

        all_link_names = [self.palm.name] + [joint.child for joint in self.iter_joints()]
        if len(all_link_names) != len(set(all_link_names)):
            raise ValueError(f"hand contains duplicated link names: {all_link_names}")

        for finger in self.fingers:
            if finger.parent_link != self.palm.name:
                raise ValueError(
                    f"finger '{finger.name}' is mounted on '{finger.parent_link}', expected palm link '{self.palm.name}'"
                )

    def iter_joints(self) -> list[JointCfg]:
        r"""按 finger 顺序展平整手的 joint 列表。

        Returns:
            list[JointCfg]: 展平后的 joint 列表。
        """

        return [joint for finger in self.fingers for joint in finger.joints]

    @property
    def joint_name_to_index(self) -> dict[str, int]:
        r"""为调试 / 导出提供 joint 名到索引的便利映射。

        Returns:
            dict[str, int]: 按 finger 顺序排列的 joint 索引映射。

        Notes:
            这个属性当前只是 convenience helper，不应被视为未来图网络
            或更高层语义系统的唯一主键来源。
        """

        return {joint.name: index for index, joint in enumerate(self.iter_joints())}

    @property
    def dof_count(self) -> int:
        """整手的总自由度数。"""

        return sum(joint.dof_count for joint in self.iter_joints())

    @property
    def fingertip_links(self) -> list[str]:
        """所有 finger 末端 link 的名称列表。"""

        return [finger.tip_link for finger in self.fingers]


joint = JointCfg
finger = FingerCfg
palm = PalmCfg
hand = HandCfg


__all__ = [
    "JointCfg",
    "FingerCfg",
    "PalmCfg",
    "HandCfg",
    "joint",
    "finger",
    "palm",
    "hand",
]
