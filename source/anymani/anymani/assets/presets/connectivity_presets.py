"""pre-made connectivity preset 注册表。

这个文件保存的不是 fingertip 几何，也不是 finger 的底层长度/宽度测量值，
而是 **pre-made 阶段合法的 joint / child-link 删减 recipe**。

这次重写的核心动机，直接来自你此前提出的两条批评：

1. 不能再把 connectivity recipe 抽象成 `retained_revolute=k` 这种“只剩计数”的语义；
2. 应当回到更直率的第一性原理表达：
   **从 canonical finger / hand preset 中，显式删掉哪些 joint / child-link。**

也就是说，这里记录的是：

- 哪些合法 connectivity 变体允许进入 pre-made 离散空间；
- 每条变体要删除哪些 joint；
- 删除后采用哪种 regroup 语义。

而它**刻意不回答**：

- fingertip 要不要换形状；
- tip 几何要不要变长、替换、扰动。

这些 tip 变化仍属于后续 `post-mutate` 的空间，不再混进 connectivity legality 本体。

# NOTE:
当前项目的 `JointCfg` 是 joint-centric 的：一个 joint 同时携带其 child link 的
collision / visual / inertial。因此这里所谓“删除某个 joint”，在科研语义上也就等价于：
**删除这个 joint 所代表的 child link 节点及其几何。**

# NOTE:
为保证科研人员能够直接从名字回忆对应骨架，这里采用两层表达：

1. **finger-level 显式 delete recipe**：明确写出要删的 joint 后缀，例如 `("j2", "j3")`；
2. **hand-level 组合 preset**：把 thumb / index / middle / ring 的 finger-level recipe
   组合成整手 connectivity preset。

hand-level 名字仍保留 `allegro_t3_i2_m2_r2` 这种简洁形式，
但这只是**命名与 provenance 的缩写**；真正的执行语义来自显式 delete 列表。
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


# 这里显式写出 canonical hand 中各类 finger 的 revolute 数量。
# 这些常量只承担“命名 / provenance 缩写”的职责，不再承担 connectivity 语义本体。
_CANONICAL_REVOLUTE_COUNT: dict[tuple[str, Literal["non_thumb", "thumb"]], int] = {
    ("allegro", "non_thumb"): 4,  # Allegro 非拇指 canonical 链：`j0, j1, j2, j3`
    ("allegro", "thumb"): 4,      # Allegro 拇指 canonical 链：`j0, j1, j2, j3`
    ("leap", "non_thumb"): 4,     # LEAP 非拇指 canonical 链：`root_fixed + j0, j1, j2, j3`
    ("leap", "thumb"): 4,         # LEAP 拇指 canonical 链：`j0, j1, j2, j3`
}


@dataclass
class FingerConnectivityPreset:
    r"""单根 finger 的显式 connectivity delete recipe。

    这里不再说“保留前 $k$ 个 revolute joint”，而是直接写：
    **要从 canonical finger preset 里删掉哪些 joint / child-link。**

    `deleted_joint_suffixes` 采用 slot-agnostic 的后缀写法，例如：

    - `("j3",)`：表示删除最远端一个 revolute 段；
    - `("j2", "j3")`：表示删除更远端两段；
    - `()`：表示 full chain，不删任何运动段。

    运行时在具体 finger slot 上 lower 时，会把这些后缀扩展成：

    - `index_j3`
    - `middle_j3`
    - `ring_j3`
    - `thumb_j3`

    这类真正存在于 `HandCfg` 里的 joint 名。

    Attributes:
        name (str): 稳定 preset 名；供 provenance / sidecar / debug 直接回溯。
        family (str): 适用 hand family，例如 `allegro` / `leap`。
        finger_kind (Literal["non_thumb", "thumb"]): 该 recipe 面向的 finger 类别。
        deleted_joint_suffixes (tuple[str, ...]): 要从 canonical chain 中删除的 joint 后缀。
        regroup_strategy (Literal["drop", "merge"]): 删除后如何处理被删段几何。
            pre-made joint-centric 主线默认采用 `drop`，即删除 joint 时同步删除 child-link 几何。
        note (str): 面向科研阅读的短说明，解释这条 recipe 的物理含义。
        metadata (dict[str, Any]): 预留扩展字段；例如未来的 canonical pose 标签。
    """

    name: str
    family: str
    finger_kind: Literal["non_thumb", "thumb"]
    deleted_joint_suffixes: tuple[str, ...] = ()
    regroup_strategy: Literal["drop", "merge"] = "drop"
    note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HandConnectivityPreset:
    r"""整手级 connectivity recipe。

    hand-level preset 只是把已经合法的 finger-level delete recipe 组合起来：

    - `thumb` 用哪条 thumb recipe；
    - `index / middle / ring` 各用哪条 non-thumb recipe。

    这使 `HandGenerator` 可以继续按：
    $$
    \text{base hand preset} \times \text{hand connectivity preset}
    $$
    的离散空间做 pre-made 枚举，
    但 connectivity 的语义主体已经从“保留 DOF 计数”切回“显式删除哪些 joint / child-link”。
    """

    name: str
    family: str
    finger_slots: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)


def _build_finger_connectivity_registry() -> dict[str, FingerConnectivityPreset]:
    r"""构建 finger-level 合法 connectivity 注册表。

    这里不再动态推导 legal recipe，而是把当前 v1 允许进入 pre-made 主线的删法
    显式写出来。这样做的好处是：

    1. 科研人员可以直接在代码里读到“删的是哪几个 joint”；
    2. `JointDeleteMutator` 退回到纯执行器，不再承担 legality 本体；
    3. future work 若要扩展到更复杂的非 prefix 删除，也只需要在这里继续加 recipe。
    """

    return {
        # ------------------------------------------------------------------
        # Allegro 非拇指：full / drop j3 / drop j2+j3
        # ------------------------------------------------------------------
        "allegro_non_thumb_full": FingerConnectivityPreset(
            name="allegro_non_thumb_full",  # 这个name名称才是合法注册，给外部用的str字段
            family="allegro",
            finger_kind="non_thumb",
            deleted_joint_suffixes=(),
            regroup_strategy="drop",
            note="Allegro 非拇指 full chain；保留 `j0, j1, j2, j3, tip`。",
        ),
        "allegro_non_thumb_drop_j3": FingerConnectivityPreset(
            name="allegro_non_thumb_drop_j3",
            family="allegro",
            finger_kind="non_thumb",
            deleted_joint_suffixes=("j3",),
            regroup_strategy="drop",
            note="Allegro 非拇指删除最远端 `j3`；child-link 几何同步删除，不 merge 回父段。",
        ),
        "allegro_non_thumb_drop_j2_j3": FingerConnectivityPreset(
            name="allegro_non_thumb_drop_j2_j3",
            family="allegro",
            finger_kind="non_thumb",
            deleted_joint_suffixes=("j2", "j3"),
            regroup_strategy="drop",
            note="Allegro 非拇指删除 `j2, j3`；仅保留更近端双关节骨干与 tip。",
        ),
        # ------------------------------------------------------------------
        # LEAP 非拇指：full / drop j3 / drop j2+j3 / drop j1+j2+j3
        # `root_fixed` 是显式固定根部段，不属于 connectivity 裁剪对象。
        # ------------------------------------------------------------------
        "leap_non_thumb_full": FingerConnectivityPreset(
            name="leap_non_thumb_full",
            family="leap",
            finger_kind="non_thumb",
            deleted_joint_suffixes=(),
            regroup_strategy="drop",
            note="LEAP 非拇指 full chain；保留 `root_fixed + j0, j1, j2, j3, tip`。",
        ),
        "leap_non_thumb_drop_j3": FingerConnectivityPreset(
            name="leap_non_thumb_drop_j3",
            family="leap",
            finger_kind="non_thumb",
            deleted_joint_suffixes=("j3",),
            regroup_strategy="drop",
            note="LEAP 非拇指删除最远端 `j3`；fixed root 与 tip 继续保留。",
        ),
        "leap_non_thumb_drop_j2_j3": FingerConnectivityPreset(
            name="leap_non_thumb_drop_j2_j3",
            family="leap",
            finger_kind="non_thumb",
            deleted_joint_suffixes=("j2", "j3"),
            regroup_strategy="drop",
            note="LEAP 非拇指删除 `j2, j3`；保留 `root_fixed + j0 + j1 + tip`。",
        ),
        "leap_non_thumb_drop_j1_j2_j3": FingerConnectivityPreset(
            name="leap_non_thumb_drop_j1_j2_j3",
            family="leap",
            finger_kind="non_thumb",
            deleted_joint_suffixes=("j1", "j2", "j3"),
            regroup_strategy="drop",
            note="LEAP 非拇指删除 `j1, j2, j3`；仅保留 `root_fixed + j0 + tip`。",
        ),
        # ------------------------------------------------------------------
        # 拇指：当前先收敛到 full / drop j3 两档。
        # ------------------------------------------------------------------
        "allegro_thumb_full": FingerConnectivityPreset(
            name="allegro_thumb_full",
            family="allegro",
            finger_kind="thumb",
            deleted_joint_suffixes=(),
            regroup_strategy="drop",
            note="Allegro 拇指 full chain；保留 `j0, j1, j2, j3, tip`。",
        ),
        "allegro_thumb_drop_j3": FingerConnectivityPreset(
            name="allegro_thumb_drop_j3",
            family="allegro",
            finger_kind="thumb",
            deleted_joint_suffixes=("j3",),
            regroup_strategy="drop",
            note="Allegro 拇指删除最远端 `j3`；tip 仍保留并重新接回近端剩余链。",
        ),
        "leap_thumb_full": FingerConnectivityPreset(
            name="leap_thumb_full",
            family="leap",
            finger_kind="thumb",
            deleted_joint_suffixes=(),
            regroup_strategy="drop",
            note="LEAP 拇指 full chain；保留 `j0, j1, j2, j3, tip`。",
        ),
        "leap_thumb_drop_j3": FingerConnectivityPreset(
            name="leap_thumb_drop_j3",
            family="leap",
            finger_kind="thumb",
            deleted_joint_suffixes=("j3",),
            regroup_strategy="drop",
            note="LEAP 拇指删除最远端 `j3`；tip 仍保留并重新接回近端剩余链。",
        ),
    }


# 这里显式给出“finger-level recipe 的枚举顺序”。
# 顺序本身就是科研语义的一部分：先 full，再从近似保守的轻删减走向更激进的压缩。
_FINGER_CONNECTIVITY_ENUMERATION_ORDER: dict[tuple[str, Literal["non_thumb", "thumb"]], tuple[str, ...]] = {
    ("allegro", "non_thumb"): (
        "allegro_non_thumb_full",
        "allegro_non_thumb_drop_j3",
        "allegro_non_thumb_drop_j2_j3",
    ),
    ("allegro", "thumb"): (
        "allegro_thumb_full",
        "allegro_thumb_drop_j3",
    ),
    ("leap", "non_thumb"): (
        "leap_non_thumb_full",
        "leap_non_thumb_drop_j3",
        "leap_non_thumb_drop_j2_j3",
        "leap_non_thumb_drop_j1_j2_j3",
    ),
    ("leap", "thumb"): (
        "leap_thumb_full",
        "leap_thumb_drop_j3",
    ),
}


def _remaining_revolute_count(preset: FingerConnectivityPreset) -> int:
    r"""返回某条 finger-level delete recipe 剩余的 revolute joint 数。

    # NOTE:
    这里的计数只服务于 hand-level 名字缩写与 metadata 标注，
    并不再承担 recipe 本体语义。recipe 的主体仍是 `deleted_joint_suffixes`。
    """

    canonical = _CANONICAL_REVOLUTE_COUNT[(preset.family, preset.finger_kind)]  # canonical revolute 总数
    return canonical - len(preset.deleted_joint_suffixes)  # 删除几段，就少几个 revolute


def _build_hand_connectivity_registry() -> dict[str, HandConnectivityPreset]:
    r"""由显式 finger-level delete recipe 展开 hand-level 注册表。

    这里保留 product 展开，不是为了“炫技自动生成”，而是为了把一个非常朴素的科研事实
    写成代码：

    - legality 先定义在单根 finger 的 joint / child-link 删减上；
    - 整手 connectivity variation 就是这些合法单指 recipe 的组合。

    与旧实现的区别在于：

    - **旧实现**：先定义 `retained_revolute` 计数，再由计数反推删法；
    - **当前实现**：先显式写出删法，再只把剩余 DOF 数当作 hand-level 命名缩写。
    """

    registry: dict[str, HandConnectivityPreset] = {}

    for family in ("allegro", "leap"):
        non_thumb_names = _FINGER_CONNECTIVITY_ENUMERATION_ORDER[(family, "non_thumb")]
        thumb_names = _FINGER_CONNECTIVITY_ENUMERATION_ORDER[(family, "thumb")]

        for thumb_name, index_name, middle_name, ring_name in product(
            thumb_names,
            non_thumb_names,
            non_thumb_names,
            non_thumb_names,
        ):
            thumb_recipe = FINGER_CONNECTIVITY_PRESET_REGISTRY[thumb_name]
            index_recipe = FINGER_CONNECTIVITY_PRESET_REGISTRY[index_name]
            middle_recipe = FINGER_CONNECTIVITY_PRESET_REGISTRY[middle_name]
            ring_recipe = FINGER_CONNECTIVITY_PRESET_REGISTRY[ring_name]

            thumb_dof = _remaining_revolute_count(thumb_recipe)   # hand-level 名字里的 `t`
            index_dof = _remaining_revolute_count(index_recipe)   # hand-level 名字里的 `i`
            middle_dof = _remaining_revolute_count(middle_recipe) # hand-level 名字里的 `m`
            ring_dof = _remaining_revolute_count(ring_recipe)     # hand-level 名字里的 `r`

            if (
                thumb_recipe.deleted_joint_suffixes == ()
                and index_recipe.deleted_joint_suffixes == ()
                and middle_recipe.deleted_joint_suffixes == ()
                and ring_recipe.deleted_joint_suffixes == ()
            ):
                name = f"{family}_full"  # full chain 给稳定短别名，方便 CLI / quick usage 直接喊
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
                    "thumb_deleted_joint_suffixes": list(thumb_recipe.deleted_joint_suffixes),
                    "index_deleted_joint_suffixes": list(index_recipe.deleted_joint_suffixes),
                    "middle_deleted_joint_suffixes": list(middle_recipe.deleted_joint_suffixes),
                    "ring_deleted_joint_suffixes": list(ring_recipe.deleted_joint_suffixes),
                },
            )

    return registry


FINGER_CONNECTIVITY_PRESET_REGISTRY: dict[str, FingerConnectivityPreset] = _build_finger_connectivity_registry()
"""finger-level 合法 connectivity delete recipe 注册表。"""

HAND_CONNECTIVITY_PRESET_REGISTRY: dict[str, HandConnectivityPreset] = _build_hand_connectivity_registry()
"""hand-level 合法 connectivity 组合注册表。"""


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


def list_finger_connectivity_preset_names(
    *,
    family: str | None = None,
    finger_kind: Literal["non_thumb", "thumb"] | None = None,
) -> tuple[str, ...]:
    r"""按稳定枚举顺序列出 finger-level connectivity recipe 名。

    这个 helper 的主要使用场景，是 pre-made 新 façade 里的 slot-level candidate pool：

    - topology 先决定“这个 slot 当前来自哪个 family、属于 thumb 还是 non-thumb”
    - 然后再由这里给出该 slot 合法的 finger-level connectivity 候选集

    与 `list_hand_connectivity_preset_names()` 的区别在于：

    - hand-level 列表服务 legacy alias / 兼容层
    - finger-level 列表服务新的 slot 级枚举主线
    """

    ordered_names: list[str] = []
    seen: set[str] = set()
    for (current_family, current_kind), names in _FINGER_CONNECTIVITY_ENUMERATION_ORDER.items():
        if family is not None and current_family != family:
            continue
        if finger_kind is not None and current_kind != finger_kind:
            continue
        for name in names:
            if name in seen:
                continue
            ordered_names.append(name)
            seen.add(name)
    return tuple(ordered_names)


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
    "list_finger_connectivity_preset_names",
    "list_hand_connectivity_preset_names",
    "get_default_hand_connectivity_preset_name",
]
