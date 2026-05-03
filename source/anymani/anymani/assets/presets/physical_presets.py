r"""官方手型关节物理属性的 pre-made preset。

本文件保存的是 **离线提取后人工审查过** 的 joint-level physical profile。
它和 `finger_presets.py` / `hand_presets.py` 一样，是 pre-made 阶段的声明式锚点，
不是运行时解析器。

科研语义上，本文件只继承官方 LEAP / Allegro URDF 中和关节广义坐标 $q$
直接相关的属性：

- `<limit lower upper effort velocity>`
- LEAP 风格的 `<joint_properties friction="..."/>`

而 link 的 `mass / inertial` 不在这里继承。原因是 AnyMani 的 link skin
遵循 canonical primitive / custom tip 建模约定，几何已经不同于官方 mesh，
所以刚体质量和惯量必须由 AnyMani 自己的 geometry lowering 重新计算。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..asset_schema_core import AssetCfgBase, JointLimitCfg, JointPropertiesCfg, _ensure_tuple


_NON_THUMB_SUFFIXES = ("mcp1", "mcp2", "pip", "dip")
r"""非拇指完整运动链的 canonical child-link 语义顺序。"""


_THUMB_SUFFIXES = ("cmc1", "cmc2", "mcp", "dip")
r"""拇指完整运动链的 canonical child-link 语义顺序。"""


@dataclass
class JointPhysicalPreset(AssetCfgBase):
    r"""单个 canonical child-link slot 的 joint 物理属性。

    一个 `JointCfg` 在当前项目中同时代表：

    1. 当前 revolute joint 的广义坐标 $q_i$；
    2. 该 joint 所携带的 child link embodiment。

    因此这里不用会被 connectivity lowering 重编号的 `j0/j1` 作为锚点，
    而是使用 child link 后缀，例如 `mcp1` / `pip` / `cmc1`。这样删除某个
    joint-link 节点后，剩余 `JointCfg` 自然保留原本 child link 的物理语义。
    """

    child_suffix: str
    r"""canonical child link 后缀，如 `mcp1` / `pip` / `cmc1`。"""

    source_joints: tuple[str, ...] | str
    r"""官方 URDF 中对应的 source joint 名；非拇指可记录多根手指的同槽来源。"""

    limit: JointLimitCfg | Mapping[str, Any] | Sequence[float]
    r"""官方 URDF 的 `<limit>` 数值，包含 $q_{\min},q_{\max}$ 及 effort / velocity。"""

    friction: float | None = None
    r"""官方 URDF 的 joint friction；`None` 表示来源文件没有该标签。"""

    def __post_init__(self):
        # source joint 名只作为 provenance，不参与运行时 build；仍规约为 tuple 便于测试核对。
        if isinstance(self.source_joints, str):
            self.source_joints = (self.source_joints,)
        else:
            self.source_joints = tuple(str(item) for item in self.source_joints)

        # limit 允许直接写 `JointLimitCfg`、dict 或 `(lower, upper)`，最终统一成对象。
        if isinstance(self.limit, JointLimitCfg):
            self.limit = self.limit.copy()
        elif isinstance(self.limit, Mapping):
            self.limit = JointLimitCfg(**dict(self.limit))
        elif isinstance(self.limit, Sequence) and not isinstance(self.limit, (str, bytes)):
            lower, upper = _ensure_tuple(self.limit, length=2, field_name="physical.limit")
            self.limit = JointLimitCfg(lower=lower, upper=upper)
        else:
            raise TypeError(f"Unsupported physical limit: {self.limit!r}")

        # friction 为 None 时表示“不写标签”，和 LEAP 的 friction=0.0 不同。
        if self.friction is not None:
            self.friction = float(self.friction)

    def joint_properties(self) -> JointPropertiesCfg | None:
        r"""返回可直接塞进 `JointCfg` 的 joint properties。

        Returns:
            JointPropertiesCfg | None: LEAP 等有 friction 来源时返回对象；
            Allegro 等无来源时返回 `None`，exporter 因而不写 `<joint_properties>`。
        """

        if self.friction is None:
            return None
        return JointPropertiesCfg(friction=self.friction)


def _limit(lower: float, upper: float, effort: float, velocity: float) -> JointLimitCfg:
    r"""用官方 URDF 数值构造 `JointLimitCfg`。

    这个 helper 只是减少表格里的重复字段名；所有数字仍显式写在 preset 中，
    方便科研巡检时直接和官方 URDF 对照。
    """

    return JointLimitCfg(lower=lower, upper=upper, effort=effort, velocity=velocity)


LEAP_NON_THUMB_PHYSICAL_PROFILE: tuple[JointPhysicalPreset, ...] = (
    JointPhysicalPreset("mcp1", ("0", "4", "8"), _limit(-1.047, 1.047, 0.95, 8.48), friction=0.0),
    JointPhysicalPreset("mcp2", ("1", "5", "9"), _limit(-0.314, 2.23, 0.95, 8.48), friction=0.0),
    JointPhysicalPreset("pip", ("2", "6", "10"), _limit(-0.506, 1.885, 0.95, 8.48), friction=0.0),
    JointPhysicalPreset("dip", ("3", "7", "11"), _limit(-0.366, 2.042, 0.95, 8.48), friction=0.0),
)
r"""LEAP 非拇指 official profile，来源 `source/anymani/assets/hands/leap_hand/leap_hand_right.urdf`。"""


LEAP_THUMB_PHYSICAL_PROFILE: tuple[JointPhysicalPreset, ...] = (
    JointPhysicalPreset("cmc1", "12", _limit(-0.349, 2.094, 0.95, 8.48), friction=0.0),
    JointPhysicalPreset("cmc2", "13", _limit(-0.47, 2.443, 0.95, 8.48), friction=0.0),
    JointPhysicalPreset("mcp", "14", _limit(-1.20, 1.90, 0.95, 8.48), friction=0.0),
    JointPhysicalPreset("dip", "15", _limit(-1.34, 1.88, 0.95, 8.48), friction=0.0),
)
r"""LEAP 拇指 official profile。"""


ALLEGRO_NON_THUMB_PHYSICAL_PROFILE: tuple[JointPhysicalPreset, ...] = (
    JointPhysicalPreset("mcp1", ("joint_0.0", "joint_4.0", "joint_8.0"), _limit(-0.47, 0.47, 10.0, 3.14)),
    JointPhysicalPreset("mcp2", ("joint_1.0", "joint_5.0", "joint_9.0"), _limit(-0.196, 1.61, 10.0, 3.14)),
    JointPhysicalPreset("pip", ("joint_2.0", "joint_6.0", "joint_10.0"), _limit(-0.174, 1.709, 10.0, 3.14)),
    JointPhysicalPreset("dip", ("joint_3.0", "joint_7.0", "joint_11.0"), _limit(-0.227, 1.618, 10.0, 3.14)),
)
r"""Allegro 非拇指 official profile；官方 URDF 未提供 joint friction。"""


ALLEGRO_THUMB_PHYSICAL_PROFILE: tuple[JointPhysicalPreset, ...] = (
    JointPhysicalPreset("cmc1", "joint_12.0", _limit(0.263, 1.396, 10.0, 3.14)),
    JointPhysicalPreset("cmc2", "joint_13.0", _limit(-0.105, 1.163, 10.0, 3.14)),
    JointPhysicalPreset("mcp", "joint_14.0", _limit(-0.189, 1.644, 10.0, 3.14)),
    JointPhysicalPreset("dip", "joint_15.0", _limit(-0.162, 1.719, 10.0, 3.14)),
)
r"""Allegro 拇指 official profile。"""


FINGER_PHYSICAL_PROFILE_REGISTRY: dict[str, tuple[JointPhysicalPreset, ...]] = {
    "leap_non_thumb_v1": LEAP_NON_THUMB_PHYSICAL_PROFILE,
    "leap_thumb_v1": LEAP_THUMB_PHYSICAL_PROFILE,
    "allegro_non_thumb_v1": ALLEGRO_NON_THUMB_PHYSICAL_PROFILE,
    "allegro_thumb_v1": ALLEGRO_THUMB_PHYSICAL_PROFILE,
}
r"""finger preset 名到 physical profile 的轻量注册表。"""


def get_finger_physical_profile(preset_name: str) -> tuple[JointPhysicalPreset, ...]:
    r"""按 finger preset 名返回一份 physical profile 副本。

    Args:
        preset_name (str): 例如 `leap_non_thumb_v1` 或 `allegro_thumb_v1`。

    Returns:
        tuple[JointPhysicalPreset, ...]: 与完整 canonical revolute chain 对齐的物理表。
    """

    try:
        return tuple(item.copy() for item in FINGER_PHYSICAL_PROFILE_REGISTRY[preset_name])
    except KeyError as exc:
        raise KeyError(f"Unknown finger physical profile: {preset_name!r}") from exc


def apply_physical_profile_to_finger_cfg(preset_name: str, cfg):
    r"""把 official joint physical profile 注入 finger builder cfg。

    该函数是 pre-made 运行时真正消费的入口，但它不解析 URDF，只读取本文件中
    已固化的 Python preset。注入结果是两条和完整 revolute chain 对齐的列表：

    - `joint_limits[i]`
    - `joint_properties[i]`

    之后 connectivity lowering 删除某个 joint-link 节点时，对应物理属性会随
    `JointCfg` 一起被删除或保留，不再需要按重编号后的 `j0/j1` 重新查表。
    """

    profile = get_finger_physical_profile(preset_name)
    if len(profile) != cfg.num_joints:
        raise ValueError(f"physical profile length must be {cfg.num_joints}, got {len(profile)} for {preset_name!r}")

    # 这里按 profile 的 canonical 顺序写入 builder cfg；顺序本身由上方 child_suffix 表明确锁定。
    return cfg.replace(
        joint_limits=[item.limit.copy() for item in profile],
        joint_properties=[item.joint_properties() for item in profile],
    )


__all__ = [
    "JointPhysicalPreset",
    "LEAP_NON_THUMB_PHYSICAL_PROFILE",
    "LEAP_THUMB_PHYSICAL_PROFILE",
    "ALLEGRO_NON_THUMB_PHYSICAL_PROFILE",
    "ALLEGRO_THUMB_PHYSICAL_PROFILE",
    "FINGER_PHYSICAL_PROFILE_REGISTRY",
    "get_finger_physical_profile",
    "apply_physical_profile_to_finger_cfg",
]
