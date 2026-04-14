"""pre-made connectivity preset 注册表。

这个文件保存的不是 fingertip 几何，也不是 finger 的底层长度/宽度测量值，
而是 **pre-made 阶段合法的 joint / child-link 组合语义**。

之所以单独立这个文件，是因为你已经明确拍板了两个边界：

1. **合法注册的主体是 connectivity**，即哪些 joint / child-link 组合是科研上允许进入
   pre-made 离散空间的；
2. **fingertip 与 connectivity 解耦**，不再像 `get_zero` 那样一起绑死进同一张合法表。

换句话说，本文件回答的是：

- “一根 Allegro / LEAP finger 在 pre-made 阶段，允许保留多少个 revolute joint？”
- “thumb / non-thumb 的合法删减范围分别是什么？”
- “整只手把这些 finger-level 合法组合做笛卡尔积之后，生成哪些 hand-level
   connectivity preset 名？”

而它**刻意不回答**：

- “tip 到底是 `cs` 还是 custom mesh？”
- “tip 现在要不要做 geometry swap / mesh perturb？”

这些 tip 变化属于后续 `post-mutate` 的空间；pre-made v1 只要求：
合法 connectivity recipe 可以与任意当前可用的 tip 选择组合。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Literal


# 当前 human-like hand 的 non-thumb slot 顺序先收敛到 index / middle / ring。
# 这与现有 `single_palm_leap` / `single_palm_allegro` hand preset 完全对齐，
# 也与当前 hand builder 默认 `num_non_thumb=3` 的实际落地路径一致。
NON_THUMB_SLOTS: tuple[str, ...] = ("index", "middle", "ring")


@dataclass
class FingerConnectivityPreset:
    r"""单根 finger 的合法 connectivity recipe。

    这里故意只记录“保留多少个 revolute joint”而不是直接记录 tip 名称，原因是：

    1. 当前 `JointDeleteMutator` 的执行语义本来就是：
       保留若干运动关节，删掉剩余运动关节，并自动把剩余链路重连；
    2. 对当前 v1 的 prefix-style 合法链而言，
       “保留前 $k$ 个 revolute joint”已经足够精确表达科研意图；
    3. tip 始终作为末端 fixed joint 保留，因此不应写进 connectivity legality 本体。

    Attributes:
        name (str): 稳定 preset 名；供 provenance / CLI / sidecar 直接回溯。
        family (str): 适用 hand family，例如 `allegro` / `leap`。
        finger_kind (Literal["non_thumb", "thumb"]): 该 recipe 面向的 finger 类别。
        retained_revolute (int): 保留的 revolute joint 数量 $k$。
        regroup_strategy (Literal["merge", "drop"]): 删除剩余 joint 后如何处理被删段几何。
        note (str): 面向科研阅读的短说明，解释这个 recipe 的物理含义。
        metadata (dict[str, Any]): 预留扩展字段；例如未来的 canonical pose 标签。
    """

    name: str
    family: str
    finger_kind: Literal["non_thumb", "thumb"]
    retained_revolute: int
    regroup_strategy: Literal["merge", "drop"] = "merge"
    note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HandConnectivityPreset:
    r"""整手级 connectivity recipe。

    hand-level preset 只是把已经合法的 finger-level recipe 组合起来：

    - `index / middle / ring` 分别选哪条 non-thumb recipe；
    - `thumb` 选哪条 thumb recipe。

    这让 `HandGenerator` 可以直接按：

    $$
    \text{base hand preset} \times \text{hand connectivity preset}
    $$

    的离散空间进行 pre-made 枚举，而不需要再在主循环里硬编码四重 for-loop。
    """

    name: str
    family: str
    finger_slots: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)


def _build_finger_connectivity_registry() -> dict[str, FingerConnectivityPreset]:
    r"""构建 finger-level 合法 connectivity 注册表。

    这里收敛的是你当前已经拍板的 family-specific 合法范围：

    - Allegro non-thumb：保留 $k \in \{2, 3, 4\}$ 个 revolute joint；
    - LEAP non-thumb：保留 $k \in \{1, 2, 3, 4\}$ 个 revolute joint；
    - Allegro / LEAP thumb：保留 $k \in \{3, 4\}$ 个 revolute joint。

    这组范围直接来自你给出的 TODO 算法与当前科研语义，而不是 generic delete
    能力的上界。也就是说：

    - generic delete 允许“任意删”
    - 但这里的合法注册只允许“进入 pre-made 主线的删法”
    """

    registry: dict[str, FingerConnectivityPreset] = {}

    # Allegro 非拇指的合法离散链：
    # - 4 DOF：完整链
    # - 3 DOF：去掉最远端一个运动关节
    # - 2 DOF：保留更近端的双关节骨干
    for retained in (4, 3, 2):
        name = f"allegro_non_thumb_r{retained}"
        registry[name] = FingerConnectivityPreset(
            name=name,
            family="allegro",
            finger_kind="non_thumb",
            retained_revolute=retained,
            regroup_strategy="merge",
            note=f"Allegro 非拇指保留前 {retained} 个 revolute joint；tip 始终保留。",
        )

    # LEAP 非拇指当前允许更激进的离散骨架压缩：
    # - 4 / 3 / 2 / 1 DOF 都进入合法 pre-made 空间
    # - 但 tip 依然不进入 legality 本体，而是继续挂在链尾
    for retained in (4, 3, 2, 1):
        name = f"leap_non_thumb_r{retained}"
        registry[name] = FingerConnectivityPreset(
            name=name,
            family="leap",
            finger_kind="non_thumb",
            retained_revolute=retained,
            regroup_strategy="merge",
            note=f"LEAP 非拇指保留前 {retained} 个 revolute joint；tip 独立于 connectivity legality。",
        )

    # thumb 当前先收敛到 full / minus-distal 两档：
    # - 4 DOF：完整链
    # - 3 DOF：删去最远端一个运动关节
    for family in ("allegro", "leap"):
        for retained in (4, 3):
            name = f"{family}_thumb_r{retained}"
            registry[name] = FingerConnectivityPreset(
                name=name,
                family=family,
                finger_kind="thumb",
                retained_revolute=retained,
                regroup_strategy="merge",
                note=f"{family} thumb 保留前 {retained} 个 revolute joint；tip 不与合法 recipe 绑定。",
            )

    return registry


def _sorted_finger_recipe_names(*, family: str, finger_kind: Literal["non_thumb", "thumb"]) -> tuple[str, ...]:
    r"""按保留 DOF 从大到小返回指定 family / kind 的 recipe 名。

    之所以做显式排序，是为了让：

    - `family_full`
    - `family_t3_i2_m2_r2`
    - ...

    这样的 hand-level preset 生成顺序保持稳定，可用于测试与 provenance。
    """

    names = [
        name
        for name, preset in FINGER_CONNECTIVITY_PRESET_REGISTRY.items()
        if preset.family == family and preset.finger_kind == finger_kind
    ]
    return tuple(
        name
        for _, name in sorted(
            (
                FINGER_CONNECTIVITY_PRESET_REGISTRY[name].retained_revolute,
                name,
            )
            for name in names
        )[::-1]
    )


def _build_hand_connectivity_registry() -> dict[str, HandConnectivityPreset]:
    r"""由 finger-level 合法 recipe 自动展开 hand-level 注册表。

    这里做的不是“软工层面的炫技自动生成”，而是把一条非常朴素的科研事实代码化：

    - 合法性首先定义在单根 finger 的 joint / child-link 组合上；
    - 整手的 connectivity variation 就是这些合法单指组合的笛卡尔积。

    因而 hand-level registry 的名字虽然自动生成，但它们的来源完全透明：

    $$
    \mathcal{H}_{family}
    =
    \mathcal{T}_{thumb}
    \times
    \mathcal{T}_{index}
    \times
    \mathcal{T}_{middle}
    \times
    \mathcal{T}_{ring}.
    $$
    """

    registry: dict[str, HandConnectivityPreset] = {}

    for family in ("allegro", "leap"):
        non_thumb_names = _sorted_finger_recipe_names(family=family, finger_kind="non_thumb")
        thumb_names = _sorted_finger_recipe_names(family=family, finger_kind="thumb")

        full_non_thumb = max(FINGER_CONNECTIVITY_PRESET_REGISTRY[name].retained_revolute for name in non_thumb_names)
        full_thumb = max(FINGER_CONNECTIVITY_PRESET_REGISTRY[name].retained_revolute for name in thumb_names)

        for thumb_name, index_name, middle_name, ring_name in product(
            thumb_names,
            non_thumb_names,
            non_thumb_names,
            non_thumb_names,
        ):
            thumb_dof = FINGER_CONNECTIVITY_PRESET_REGISTRY[thumb_name].retained_revolute
            index_dof = FINGER_CONNECTIVITY_PRESET_REGISTRY[index_name].retained_revolute
            middle_dof = FINGER_CONNECTIVITY_PRESET_REGISTRY[middle_name].retained_revolute
            ring_dof = FINGER_CONNECTIVITY_PRESET_REGISTRY[ring_name].retained_revolute

            # 完整链使用更短、更直观的别名，便于 quick usage 直接喊：
            # - `allegro_full`
            # - `leap_full`
            if (thumb_dof, index_dof, middle_dof, ring_dof) == (full_thumb, full_non_thumb, full_non_thumb, full_non_thumb):
                name = f"{family}_full"
            else:
                name = f"{family}_t{thumb_dof}_i{index_dof}_m{middle_dof}_r{ring_dof}"

            registry[name] = HandConnectivityPreset(
                name=name,
                family=family,
                finger_slots={
                    "thumb": thumb_name,
                    "index": index_name,
                    "middle": middle_name,
                    "ring": ring_name,
                },
                metadata={
                    "thumb_revolute": thumb_dof,
                    "index_revolute": index_dof,
                    "middle_revolute": middle_dof,
                    "ring_revolute": ring_dof,
                },
            )

    return registry


FINGER_CONNECTIVITY_PRESET_REGISTRY: dict[str, FingerConnectivityPreset] = _build_finger_connectivity_registry()
"""finger-level 合法 connectivity 注册表。"""

HAND_CONNECTIVITY_PRESET_REGISTRY: dict[str, HandConnectivityPreset] = _build_hand_connectivity_registry()
"""hand-level 合法 connectivity 注册表。"""


def get_finger_connectivity_preset_data(name: str) -> FingerConnectivityPreset:
    r"""按名字返回一份 finger-level connectivity preset 副本。"""

    try:
        return deepcopy(FINGER_CONNECTIVITY_PRESET_REGISTRY[name])
    except KeyError as exc:
        raise KeyError(f"Unknown finger connectivity preset: {name!r}") from exc


def get_hand_connectivity_preset_data(name: str) -> HandConnectivityPreset:
    r"""按名字返回一份 hand-level connectivity preset 副本。"""

    try:
        return deepcopy(HAND_CONNECTIVITY_PRESET_REGISTRY[name])
    except KeyError as exc:
        raise KeyError(f"Unknown hand connectivity preset: {name!r}") from exc


def list_hand_connectivity_preset_names(family: str | None = None) -> tuple[str, ...]:
    r"""列出 hand-level connectivity preset 名称。

    Args:
        family (str | None): 若给定，则只返回该 family 的 preset 名。

    Returns:
        tuple[str, ...]: 稳定排序后的 preset 名。
    """

    names = [
        name
        for name, preset in HAND_CONNECTIVITY_PRESET_REGISTRY.items()
        if family is None or preset.family == family
    ]
    return tuple(sorted(names))


def get_default_hand_connectivity_preset_name(family: str) -> str:
    r"""返回某个 family 的默认 full connectivity preset 名。"""

    return f"{family}_full"


__all__ = [
    "NON_THUMB_SLOTS",
    "FingerConnectivityPreset",
    "HandConnectivityPreset",
    "FINGER_CONNECTIVITY_PRESET_REGISTRY",
    "HAND_CONNECTIVITY_PRESET_REGISTRY",
    "get_finger_connectivity_preset_data",
    "get_hand_connectivity_preset_data",
    "list_hand_connectivity_preset_names",
    "get_default_hand_connectivity_preset_name",
]
