r"""供 exporter、bank、运动学与几何表征共用的版本化整手静态语义。

本模块与 ``asset_schema_core.py``、``asset_schema_embodiment.py`` 同属资产 schema 层，因为
PALM/JOINT/TIP 归属、碰撞片稳定身份、显式基准构型和锚点种子
都是资产真值，而不是 SSL 模型应从 URDF/link 名重新猜测的知识。它只描述不可变事实，不执行
批量 FK、最近点查询或可学习编码；动态解释属于 ``robots``，训练目标属于 ``distill``。

坐标系约定：

- ``{a}``：资产导出坐标系，即 ``HandCfg`` 与 URDF 几何当前表达的坐标系；
- ``{h}``：部署期统一手部语义坐标系；
- ``asset_to_hand_*`` 保存刚体变换 $T_{ha}$，满足

$$
p_h=R_{ha}p_a+t_{ha}.
$$

所有长度使用 m，所有关节角和 RPY 使用 rad。``q_home_rad`` 必须按活动关节顺序显式保存；
generated v1 旧 sidecar 的确定性迁移规则声明零基准，但公共 schema 不把零值当作普遍默认。

碰撞片归属规则仅适用于 generated ``HandCfg``：

1. palm 自身与首活动关节之前的 fixed 根段归 PALM；
2. 每个 revolute joint 的 child-link 碰撞片归该 JOINT；
3. 最后活动关节之后、且显式 ``is_tip=True`` 的 fixed descendant 归对应 TIP；
4. 其他携带碰撞片的 fixed descendant 含混，严格拒绝。

official LEAP/Allegro 不调用这套迁移规则；它们必须提供人工核验的显式 sidecar，否则 fail closed。
``exporter`` 负责写入该 schema，``bank`` 负责解析/迁移并把类型化结果交付给下游；二者共用本模块，
但本模块不依赖任一运行时目录。

字段生命周期：

```text
HandCfg / generator truth
    -> derive_generated_geometry_semantics
    -> HandGeometrySemanticsCfg
    -> exporter: hand.yaml.geometry_semantics
    -> bank: HandContainer.geometry_semantics
    -> robots: EmbodimentGeometrySpec
    -> distill: field/query/encoder targets
```

`HandGeometrySemanticsCfg` 是跨目录的静态事实合同，不是训练配置。它不保存当前 batch 的 q、
query、最近点、signed distance、场标签或任何 learned activation。它保存的是“如果给定任意
合法或参考 q，robots 能否无歧义地解释这只手”的最小充分信息。

完整 kinematic joint 轴包含 fixed joint。原因是 fixed root、fixed spacer 与 fixed tip 会改变
后续 child link frame；如果只保存 revolute 名称，robots 只能回头读取 URDF 才能恢复 owner home
transform，进而产生第二个隐式运动学真源。每个 revolute 的 axis 位于 joint frame，origin 是
parent link 到 joint frame 的固定 $SE(3)$ 变换；首 joint 的 mount folding 已在 exporter contract
中按逐分量平移/RPY 叠加落实，然后下游统一使用严格刚体复合。

`active_joint_index` 是唯一允许进入 q 轴的索引。fixed joint 的 axis 只是 schema 占位，不能被
误当成可驱动自由度。limits 保存完整合法域，但不参与 hash 外的模型输入约定；q_home 保存 POE
参考，允许处于 limits 外。这个区分同时覆盖 URDF 零参考和控制器合法采样域不重合的 Allegro 情形。

owner 顺序固定为 PALM、全部活动 JOINT、全部 TIP。component 顺序来自 palm/finger/joint/collision
遍历，只用于 provenance；网络不得把 collision index 当作语义 embedding。owner/component 的
双向覆盖验证保证一片 collision 不能被两个 owner 复用，也保证 sidecar 不能遗漏一片 geometry。

内容哈希覆盖 schema 版本、单位、frame calibration、kinematic chain、limits、home、owner、
component payload、anchor seed 和所有 provenance-relevant 顺序。反序列化先构造冻结 dataclass，再
重新计算 hash；任何手工改字段而不更新 hash 都在 bank 边界拒绝。哈希不是加密签名，不能替代文件
来源认证；它的职责是拒绝 cache/sidecar 串包与 stale materialization。

anchor seed 的 position 属于 `{a}`，不是 `{h}`，因为 seed 是资产 mount truth；robots lower 时
统一应用 $T_{ha}$。seed 的 rotation 只用于支持邻域的局部审计，目前不作为网络输入，也不代表
物理表面 normal。surface/interior 比例、径向支持半径和固定随机 seed 属于 anchor realization
记录，不允许通过 finger name 进入 encoder routing。

本 schema deliberately 不拥有 Boolean backend。不同几何后端的可信度、watertight 检查、owner
union、surface area sample 和 GPU BVH 属于 robots；这样改变 manifold3d 或 Warp 版本时，不会改变
资产 sidecar 的逻辑身份，只会更新 geometry cache provenance 与 reference evidence。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeAlias, cast

from .asset_schema_core import CollisionGeometryCfg, JointLimitCfg, PoseCfg
from .asset_schema_embodiment import FingerCfg, HandCfg, JointCfg, PalmCfg

SEMANTICS_SCHEMA_VERSION = "1.0.0"
"""当前几何语义 sidecar 的版本。"""

GENERATED_MIGRATION_VERSION = "generated-handcfg-v1"
"""从 legacy generated ``HandCfg`` 确定性恢复语义的规则版本。"""

OwnerRole: TypeAlias = Literal["palm", "joint", "tip"]
"""物理表面归属体类型；字符串值直接稳定写入 YAML/JSON。"""

Vector3: TypeAlias = tuple[float, float, float]
Matrix3Flat: TypeAlias = tuple[float, float, float, float, float, float, float, float, float]


@dataclass(frozen=True)
class CollisionComponentSemanticsCfg:
    r"""一个不可再拆分的碰撞几何实例及其唯一表面归属。

    ``component_id`` 是逻辑稳定 ID，由 hand/finger/joint 与局部 collision 序号构造，不依赖临时
    mesh 文件名。``carrier_link`` 只供 ``robots`` lower 变换链；``distill`` 不读取 link 名。
    """

    component_id: str  # 逻辑稳定组件 ID
    owner_id: str  # 唯一 PALM/JOINT/TIP 归属体 ID
    carrier_link: str  # 几何实例直接附着的 child link
    collision_index: int  # 在 carrier link 内的稳定局部序号
    collision_name: str | None  # 人类可读诊断名，不作为主键
    geometry_kind: str  # box/cylinder/elliptic_cylinder/sphere/mesh
    geometry_payload: dict[str, Any]  # 含类型和尺寸/路径/缩放的完整静态几何
    origin_pos_m: Vector3  # 相对 carrier link 的碰撞几何平移，m
    origin_rpy_rad: Vector3  # 相对 carrier link 的碰撞几何 RPY，rad
    source_joint_name: str | None  # palm 自身碰撞片为 None


@dataclass(frozen=True)
class KinematicJointSemanticsCfg:
    r"""一个按实际导出语义冻结的 fixed/revolute joint。

    首 joint 的 ``origin_*`` 已折入 ``finger.mount``，与实际 URDF 一致；后续 joint 保持
    ``HandCfg`` 局部 origin。``active_joint_index`` 只对 revolute joint 非空，并与全局
    ``active_joint_names/q_home_rad`` 轴严格一致。
    """

    joint_name: str  # 全资产唯一 joint 名
    joint_type: Literal["fixed", "revolute"]  # 运动学类型
    parent_link: str  # 父 link
    child_link: str  # 子 link
    origin_pos_m: Vector3  # joint frame 相对 parent link 的平移，m
    origin_rpy_rad: Vector3  # joint frame 相对 parent link 的固定轴 RPY，rad
    axis_local: Vector3  # revolute 轴在 joint frame 中的单位向量；fixed 为约定占位轴
    active_joint_index: int | None  # revolute 在规范 $N_J$ 轴上的索引


@dataclass(frozen=True)
class GeometryOwnerSemanticsCfg:
    r"""一个连续密度场表面归属体的静态身份。

    一个 owner 可以覆盖多个碰撞片。后续真实边界必须对这些 solid 做同 owner Boolean union；
    ``component_ids`` 的顺序只用于审计和稳定哈希，不进入网络语义。
    """

    owner_id: str  # 全资产唯一 owner ID
    owner_index: int  # 规范实体轴索引，PALM 后接 JOINT，再接 TIP
    role: OwnerRole  # PALM/JOINT/TIP 类型
    parent_owner_id: str | None  # owner 图中的直接父节点
    finger_name: str | None  # PALM 为 None；仅用于资产审计
    joint_name: str | None  # 仅 JOINT owner 非空
    reference_link: str  # owner 局部几何缓存所选的参考 link
    component_ids: tuple[str, ...]  # 被该 owner 唯一覆盖的碰撞片


@dataclass(frozen=True)
class AnchorSeedSemanticsCfg:
    r"""每根手指在 palm solid 内生成物理锚点的局部采样种子。

    seed 不是网络 token，也不赋予锚点手指编号。它只定位 palm 内的一处支持邻域；固定随机种子
    产生的全部锚点在进入编码器后组成无序、等地位的完整 $K$ 集合。
    """

    seed_id: str  # 稳定种子 ID
    finger_name: str  # provenance：该 seed 来源分支
    first_active_joint_name: str  # 审计实际采用的首活动关节
    support_owner_id: str  # 当前规则固定为 palm
    position_a_m: Vector3  # 首活动关节 frame 原点在 `{a}` 中的位置，m
    rotation_a: Matrix3Flat  # 首活动关节 frame 在 `{a}` 中的旋转，按行展开


@dataclass(frozen=True)
class HandGeometrySemanticsCfg:
    r"""下游唯一允许消费的版本化整手静态几何语义。

    该对象完整区分模型信息与采样信息：``joint_limits_rad`` 供构型采样，``q_home_rad`` 与运动学
    几何共同定义物理输入，但 limits 绝不进入编码器。``content_hash`` 覆盖本对象其余全部字段，
    因此 owner cache、anchor realization 与训练 manifest 可以拒绝陈旧资产。
    """

    schema_version: str  # 几何语义 schema 版本
    migration_version: str  # legacy generated 恢复规则版本
    source_kind: Literal["generated", "official"]  # 语义来源边界
    asset_id: str  # sidecar 中的稳定资产 ID
    asset_name: str  # HandCfg 名称
    topology_key: str | None  # 严格 morphology split 使用的 topology 身份
    family: str  # leap/allegro/mixed/generic 等来源族
    handedness: Literal["left", "right", "unknown"]  # 左右手标签
    units: dict[str, str]  # 固定为 length=m、angle=rad
    asset_to_hand_rotation: Matrix3Flat  # $R_{ha}$，按行展开
    asset_to_hand_translation_m: Vector3  # $t_{ha}$，m
    palm_link: str  # 运动学树根 link
    palm_origin_pos_m: Vector3  # palm link 相对 `{a}` 的平移，m
    palm_origin_rpy_rad: Vector3  # palm link 相对 `{a}` 的固定轴 RPY，rad
    kinematic_joints: tuple[KinematicJointSemanticsCfg, ...]  # 含 fixed descendants 的完整导出链
    active_joint_names: tuple[str, ...]  # 规范活动关节顺序
    q_home_rad: tuple[float, ...]  # 与活动关节同轴的显式基准构型
    joint_limits_rad: tuple[tuple[float, float], ...]  # 仅采样/验证消费
    owners: tuple[GeometryOwnerSemanticsCfg, ...]  # 规范实体轴
    components: tuple[CollisionComponentSemanticsCfg, ...]  # 全部碰撞片，唯一覆盖
    anchor_seeds: tuple[AnchorSeedSemanticsCfg, ...]  # 每指一个 palm 支持 seed
    content_hash: str  # 以上字段的 SHA-256

    def __post_init__(self) -> None:
        r"""在资产边界拒绝重复、遗漏、维度不一致和非法单位。

        Raises:
            ValueError: 任一静态合同不成立时抛出。这里的拒绝优先于下游静默修复。
        """

        if self.schema_version != SEMANTICS_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported geometry semantics schema_version={self.schema_version!r}; "
                f"expected {SEMANTICS_SCHEMA_VERSION!r}"
            )
        if self.source_kind not in {"generated", "official"}:
            raise ValueError(f"invalid geometry semantics source_kind={self.source_kind!r}")
        if self.handedness not in {"left", "right", "unknown"}:
            raise ValueError(f"invalid handedness={self.handedness!r}")

        joint_count = len(self.active_joint_names)  # 活动关节轴长度 $N_J$
        if len(set(self.active_joint_names)) != joint_count:
            raise ValueError("active_joint_names must be unique")
        if len(self.q_home_rad) != joint_count or len(self.joint_limits_rad) != joint_count:
            raise ValueError("q_home_rad and joint_limits_rad must align with active_joint_names")
        if self.units != {"length": "m", "angle": "rad"}:
            raise ValueError(f"geometry semantics require SI units, got {self.units}")
        if len(self.asset_to_hand_rotation) != 9 or len(self.asset_to_hand_translation_m) != 3:
            raise ValueError("asset-to-hand transform must contain a 3x3 rotation and 3D translation")

        kinematic_names = [joint.joint_name for joint in self.kinematic_joints]
        if len(kinematic_names) != len(set(kinematic_names)):
            raise ValueError("kinematic joint names must be unique")
        active_kinematic = tuple(
            joint for joint in self.kinematic_joints if joint.joint_type == "revolute"
        )
        if tuple(joint.joint_name for joint in active_kinematic) != self.active_joint_names:
            raise ValueError("revolute kinematic joint order must equal active_joint_names")
        if tuple(joint.active_joint_index for joint in active_kinematic) != tuple(range(joint_count)):
            raise ValueError("revolute active_joint_index must be contiguous and canonical")
        if any(joint.active_joint_index is not None for joint in self.kinematic_joints if joint.joint_type == "fixed"):
            raise ValueError("fixed kinematic joints must not have active_joint_index")

        known_links = {self.palm_link, *(joint.child_link for joint in self.kinematic_joints)}
        available_parents = {self.palm_link}
        for joint in self.kinematic_joints:
            if joint.parent_link not in available_parents:
                raise ValueError(
                    f"kinematic joint '{joint.joint_name}' parent '{joint.parent_link}' is unavailable"
                )
            available_parents.add(joint.child_link)

        for joint_name, limits in zip(self.active_joint_names, self.joint_limits_rad):
            lower, upper = limits  # 关节合法域，单位 rad
            if not lower < upper:
                raise ValueError(f"joint '{joint_name}' has invalid limits {limits}")
        # $q_{home}$ 是 POE/URDF 的运动学参考，不要求属于控制合法域；例如 Allegro thumb_j0
        # 的 URDF 零参考位于正角度 limits 外。limits 只约束采样和控制，不能反向改写几何参考。

        owner_ids = [owner.owner_id for owner in self.owners]  # 规范 owner 主键
        component_ids = [component.component_id for component in self.components]  # 逻辑 collision 主键
        if len(owner_ids) != len(set(owner_ids)):
            raise ValueError("owner IDs must be unique")
        if [owner.owner_index for owner in self.owners] != list(range(len(self.owners))):
            raise ValueError("owner_index must be contiguous and match owners order")
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("collision component IDs must be unique")

        known_owners = set(owner_ids)  # 所有 parent/component 引用必须闭合于此集合
        assigned_components: list[str] = []
        for owner in self.owners:
            if owner.parent_owner_id is not None and owner.parent_owner_id not in known_owners:
                raise ValueError(f"owner '{owner.owner_id}' references unknown parent '{owner.parent_owner_id}'")
            assigned_components.extend(owner.component_ids)
        if len(assigned_components) != len(set(assigned_components)):
            raise ValueError("a collision component is assigned to more than one owner")
        if set(assigned_components) != set(component_ids):
            raise ValueError("owner component coverage must equal the complete collision component set")
        for component in self.components:
            if component.owner_id not in known_owners:
                raise ValueError(f"component '{component.component_id}' references unknown owner")
            if component.component_id not in self.owners[owner_ids.index(component.owner_id)].component_ids:
                raise ValueError(f"component '{component.component_id}' owner back-reference is inconsistent")
            if component.carrier_link not in known_links:
                raise ValueError(f"component '{component.component_id}' references unknown carrier link")
        for owner in self.owners:
            if owner.reference_link not in known_links:
                raise ValueError(f"owner '{owner.owner_id}' references unknown reference link")

        for seed in self.anchor_seeds:
            if seed.support_owner_id not in known_owners:
                raise ValueError(f"anchor seed '{seed.seed_id}' references unknown support owner")
            support = self.owners[owner_ids.index(seed.support_owner_id)]
            if support.role != "palm":
                raise ValueError(f"anchor seed '{seed.seed_id}' support owner must be PALM")
        if len(self.content_hash) != 64:
            raise ValueError("content_hash must be a SHA-256 hexadecimal digest")
        hash_payload = asdict(self)  # 反序列化时必须证明确实对应声明内容，而不只检查字符串长度
        declared_hash = hash_payload.pop("content_hash")
        if _content_hash(hash_payload) != declared_hash:
            raise ValueError("geometry semantics content_hash does not match its payload")


@dataclass
class _OwnerBuilder:
    """derive 期间累积碰撞片，最终冻结为公开 owner schema。"""

    owner_id: str
    role: OwnerRole
    parent_owner_id: str | None
    finger_name: str | None
    joint_name: str | None
    reference_link: str
    component_ids: list[str]


def derive_generated_geometry_semantics(
    hand: HandCfg,
    *,
    asset_id: str,
    topology_key: str | None = None,
    q_home_rad: Mapping[str, float] | None = None,
    asset_to_hand_rotation: Sequence[float] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    asset_to_hand_translation_m: Sequence[float] = (0.0, 0.0, 0.0),
) -> HandGeometrySemanticsCfg:
    r"""从 generated ``HandCfg`` 确定性恢复几何语义。

    Args:
        hand (HandCfg): generator/validator/exporter 共用的规范手资产描述。
        asset_id (str): sidecar 顶层稳定样本 ID。
        topology_key (str | None): morphology split 使用的拓扑身份。
        q_home_rad (Mapping[str, float] | None): 可选的逐活动关节显式基准构型。旧 generated sidecar
            缺失时使用本迁移规则声明的零构型；只要显式提供，就必须完整覆盖且不能含未知关节。
        asset_to_hand_rotation (Sequence[float]): 按行展开的 $R_{ha}$。
        asset_to_hand_translation_m (Sequence[float]): $t_{ha}$，单位 m。

    Returns:
        HandGeometrySemanticsCfg: 已完成唯一覆盖与内容哈希验证的静态语义。

    Raises:
        ValueError: fixed descendant 归属含混、活动 owner 无几何、TIP 缺失或 home 不完整时抛出。
    """

    palm = cast(PalmCfg, hand.palm)  # HandCfg.__post_init__ 已把宽松 mapping 收敛为 PalmCfg
    active_joints = tuple(joint for joint in hand.iter_joints() if joint.joint_type == "revolute")
    active_joint_names = tuple(joint.name for joint in active_joints)  # 规范 $N_J$ 轴
    home_by_joint = _resolve_generated_q_home(active_joint_names, q_home_rad)
    home_values = tuple(home_by_joint[name] for name in active_joint_names)  # 显式按规范轴存储
    joint_limits = tuple(_joint_limits(joint) for joint in active_joints)  # 仅用于采样与合法性验证
    active_index_by_name = {name: index for index, name in enumerate(active_joint_names)}
    kinematic_joints = tuple(
        KinematicJointSemanticsCfg(
            joint_name=joint.name,
            joint_type=joint.joint_type,
            parent_link=joint.parent,
            child_link=str(joint.child),
            origin_pos_m=_vector3(
                _exported_joint_origin(finger, joint, is_first=joint_index == 0).pos,
                f"joint[{joint.name}].origin.pos",
            ),
            origin_rpy_rad=_vector3(
                _exported_joint_origin(finger, joint, is_first=joint_index == 0).rpy,
                f"joint[{joint.name}].origin.rpy",
            ),
            axis_local=_vector3(joint.axis, f"joint[{joint.name}].axis"),
            active_joint_index=active_index_by_name.get(joint.name),
        )
        for finger in hand.fingers
        for joint_index, joint in enumerate(finger.joints)
    )  # 完整保存 fixed roots、active segments 与 fixed tips

    palm_owner = _OwnerBuilder(
        owner_id="palm",
        role="palm",
        parent_owner_id=None,
        finger_name=None,
        joint_name=None,
        reference_link=palm.name,
        component_ids=[],
    )
    joint_owners: list[_OwnerBuilder] = []  # 保持 HandCfg 中活动 JOINT 的规范顺序
    tip_owners: list[_OwnerBuilder] = []  # 在全部 JOINT 后追加 TIP，形成统一实体轴
    owner_by_id: dict[str, _OwnerBuilder] = {palm_owner.owner_id: palm_owner}
    components: list[CollisionComponentSemanticsCfg] = []

    for collision_index, collision in enumerate(palm.collisions):
        component = _make_component(
            collision,
            component_id=f"palm/{palm.name}/collision/{collision_index}",
            owner_id=palm_owner.owner_id,
            carrier_link=palm.name,
            collision_index=collision_index,
            source_joint_name=None,
        )
        components.append(component)  # palm 自身 collision 属于 PALM owner
        palm_owner.component_ids.append(component.component_id)

    anchor_seeds: list[AnchorSeedSemanticsCfg] = []
    for finger in hand.fingers:
        _derive_finger_owners(
            hand,
            finger,
            palm_owner=palm_owner,
            joint_owners=joint_owners,
            tip_owners=tip_owners,
            owner_by_id=owner_by_id,
            components=components,
        )
        anchor_seeds.append(_derive_anchor_seed(hand, finger))  # 每指首活动关节 frame 提供一个 seed

    owner_builders = [palm_owner, *joint_owners, *tip_owners]  # PALM -> JOINT -> TIP 规范实体轴
    owners = tuple(
        GeometryOwnerSemanticsCfg(
            owner_id=owner.owner_id,
            owner_index=owner_index,
            role=owner.role,
            parent_owner_id=owner.parent_owner_id,
            finger_name=owner.finger_name,
            joint_name=owner.joint_name,
            reference_link=owner.reference_link,
            component_ids=tuple(owner.component_ids),
        )
        for owner_index, owner in enumerate(owner_builders)
    )

    for owner in owners:
        if not owner.component_ids:
            raise ValueError(f"generated owner '{owner.owner_id}' has no collision geometry")

    payload = {
        "schema_version": SEMANTICS_SCHEMA_VERSION,
        "migration_version": GENERATED_MIGRATION_VERSION,
        "source_kind": "generated",
        "asset_id": str(asset_id),
        "asset_name": hand.name,
        "topology_key": topology_key,
        "family": hand.family,
        "handedness": hand.handedness,
        "units": {"length": "m", "angle": "rad"},
        "asset_to_hand_rotation": _float_tuple(asset_to_hand_rotation, length=9, field_name="asset_to_hand_rotation"),
        "asset_to_hand_translation_m": _vector3(asset_to_hand_translation_m, "asset_to_hand_translation_m"),
        "palm_link": palm.name,
        "palm_origin_pos_m": _vector3(cast(PoseCfg, palm.origin).pos, "palm.origin.pos"),
        "palm_origin_rpy_rad": _vector3(cast(PoseCfg, palm.origin).rpy, "palm.origin.rpy"),
        "kinematic_joints": kinematic_joints,
        "active_joint_names": active_joint_names,
        "q_home_rad": home_values,
        "joint_limits_rad": joint_limits,
        "owners": owners,
        "components": tuple(components),
        "anchor_seeds": tuple(anchor_seeds),
    }
    content_hash = _content_hash(payload)  # 缓存键覆盖全部静态几何与归属事实
    return HandGeometrySemanticsCfg(**payload, content_hash=content_hash)


def geometry_semantics_to_dict(semantics: HandGeometrySemanticsCfg) -> dict[str, Any]:
    r"""把冻结 schema 转成可稳定写入 YAML/JSON 的原生容器。

    Args:
        semantics (HandGeometrySemanticsCfg): 已验证的版本化静态语义。

    Returns:
        dict[str, Any]: dataclass 递归展开后的 sidecar 文档片段。
    """

    return asdict(semantics)


def geometry_semantics_from_dict(document: Mapping[str, Any]) -> HandGeometrySemanticsCfg:
    r"""从 sidecar 映射严格恢复并验证类型化几何语义。

    Args:
        document (Mapping[str, Any]): ``hand.yaml.geometry_semantics`` 文档。

    Returns:
        HandGeometrySemanticsCfg: 嵌套 owner/component/seed 已冻结，且内容哈希已复核的 schema。

    Raises:
        TypeError: 文档或嵌套条目不是映射时抛出。
        KeyError: 必需字段缺失时抛出。
        ValueError: schema 版本、唯一覆盖、单位或内容哈希不合法时抛出。
    """

    if not isinstance(document, Mapping):
        raise TypeError(f"geometry semantics must be a mapping, got {type(document).__name__}")

    owners = tuple(_owner_from_dict(item) for item in _mapping_sequence(document["owners"], "owners"))
    kinematic_joints = tuple(
        _kinematic_joint_from_dict(item)
        for item in _mapping_sequence(document["kinematic_joints"], "kinematic_joints")
    )
    components = tuple(
        _component_from_dict(item) for item in _mapping_sequence(document["components"], "components")
    )
    anchor_seeds = tuple(
        _anchor_seed_from_dict(item) for item in _mapping_sequence(document["anchor_seeds"], "anchor_seeds")
    )
    source_kind = str(document["source_kind"])
    handedness = str(document["handedness"])
    return HandGeometrySemanticsCfg(
        schema_version=str(document["schema_version"]),
        migration_version=str(document["migration_version"]),
        source_kind=cast(Literal["generated", "official"], source_kind),
        asset_id=str(document["asset_id"]),
        asset_name=str(document["asset_name"]),
        topology_key=None if document.get("topology_key") is None else str(document["topology_key"]),
        family=str(document["family"]),
        handedness=cast(Literal["left", "right", "unknown"], handedness),
        units={str(key): str(value) for key, value in _mapping(document["units"], "units").items()},
        asset_to_hand_rotation=cast(
            Matrix3Flat,
            _float_tuple(document["asset_to_hand_rotation"], length=9, field_name="asset_to_hand_rotation"),
        ),
        asset_to_hand_translation_m=_vector3(
            document["asset_to_hand_translation_m"], "asset_to_hand_translation_m"
        ),
        palm_link=str(document["palm_link"]),
        palm_origin_pos_m=_vector3(document["palm_origin_pos_m"], "palm_origin_pos_m"),
        palm_origin_rpy_rad=_vector3(document["palm_origin_rpy_rad"], "palm_origin_rpy_rad"),
        kinematic_joints=kinematic_joints,
        active_joint_names=tuple(str(name) for name in document["active_joint_names"]),
        q_home_rad=tuple(float(value) for value in document["q_home_rad"]),
        joint_limits_rad=tuple(
            cast(tuple[float, float], _float_tuple(limits, length=2, field_name="joint_limits_rad[]"))
            for limits in document["joint_limits_rad"]
        ),
        owners=owners,
        components=components,
        anchor_seeds=anchor_seeds,
        content_hash=str(document["content_hash"]),
    )


def _owner_from_dict(document: Mapping[str, Any]) -> GeometryOwnerSemanticsCfg:
    """恢复一个 owner 条目，并把列表字段冻结为 tuple。"""

    role = str(document["role"])
    return GeometryOwnerSemanticsCfg(
        owner_id=str(document["owner_id"]),
        owner_index=int(document["owner_index"]),
        role=cast(OwnerRole, role),
        parent_owner_id=None if document.get("parent_owner_id") is None else str(document["parent_owner_id"]),
        finger_name=None if document.get("finger_name") is None else str(document["finger_name"]),
        joint_name=None if document.get("joint_name") is None else str(document["joint_name"]),
        reference_link=str(document["reference_link"]),
        component_ids=tuple(str(component_id) for component_id in document["component_ids"]),
    )


def _kinematic_joint_from_dict(document: Mapping[str, Any]) -> KinematicJointSemanticsCfg:
    """恢复一个包含实际导出 origin 的 fixed/revolute joint。"""

    joint_type = str(document["joint_type"])
    return KinematicJointSemanticsCfg(
        joint_name=str(document["joint_name"]),
        joint_type=cast(Literal["fixed", "revolute"], joint_type),
        parent_link=str(document["parent_link"]),
        child_link=str(document["child_link"]),
        origin_pos_m=_vector3(document["origin_pos_m"], "kinematic_joint.origin_pos_m"),
        origin_rpy_rad=_vector3(document["origin_rpy_rad"], "kinematic_joint.origin_rpy_rad"),
        axis_local=_vector3(document["axis_local"], "kinematic_joint.axis_local"),
        active_joint_index=(
            None if document.get("active_joint_index") is None else int(document["active_joint_index"])
        ),
    )


def _component_from_dict(document: Mapping[str, Any]) -> CollisionComponentSemanticsCfg:
    """恢复一个碰撞片条目，保留完整静态几何 payload。"""

    return CollisionComponentSemanticsCfg(
        component_id=str(document["component_id"]),
        owner_id=str(document["owner_id"]),
        carrier_link=str(document["carrier_link"]),
        collision_index=int(document["collision_index"]),
        collision_name=None if document.get("collision_name") is None else str(document["collision_name"]),
        geometry_kind=str(document["geometry_kind"]),
        geometry_payload=dict(_mapping(document["geometry_payload"], "geometry_payload")),
        origin_pos_m=_vector3(document["origin_pos_m"], "origin_pos_m"),
        origin_rpy_rad=_vector3(document["origin_rpy_rad"], "origin_rpy_rad"),
        source_joint_name=(
            None if document.get("source_joint_name") is None else str(document["source_joint_name"])
        ),
    )


def _anchor_seed_from_dict(document: Mapping[str, Any]) -> AnchorSeedSemanticsCfg:
    """恢复一个首活动关节锚点种子条目。"""

    return AnchorSeedSemanticsCfg(
        seed_id=str(document["seed_id"]),
        finger_name=str(document["finger_name"]),
        first_active_joint_name=str(document["first_active_joint_name"]),
        support_owner_id=str(document["support_owner_id"]),
        position_a_m=_vector3(document["position_a_m"], "position_a_m"),
        rotation_a=cast(Matrix3Flat, _float_tuple(document["rotation_a"], length=9, field_name="rotation_a")),
    )


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    """验证一个 sidecar 字段确为映射。"""

    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping, got {type(value).__name__}")
    return value


def _mapping_sequence(value: Any, field_name: str) -> tuple[Mapping[str, Any], ...]:
    """验证 sidecar 中的嵌套条目序列，不接受字符串等伪序列。"""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence of mappings")
    return tuple(_mapping(item, f"{field_name}[]") for item in value)


def _derive_finger_owners(
    hand: HandCfg,
    finger: FingerCfg,
    *,
    palm_owner: _OwnerBuilder,
    joint_owners: list[_OwnerBuilder],
    tip_owners: list[_OwnerBuilder],
    owner_by_id: dict[str, _OwnerBuilder],
    components: list[CollisionComponentSemanticsCfg],
) -> None:
    r"""按显式 joint type 与 ``is_tip`` 归属一根 generated 手指的碰撞片。"""

    active_indices = [index for index, joint in enumerate(finger.joints) if joint.joint_type == "revolute"]
    if not active_indices:
        raise ValueError(f"generated finger '{finger.name}' has no active joint")
    first_active = active_indices[0]
    last_active = active_indices[-1]
    previous_active_owner_id = palm_owner.owner_id  # 第一活动 JOINT 的 owner parent 是 PALM

    for joint_index, joint in enumerate(finger.joints):
        if joint.joint_type == "revolute":
            owner_id = f"joint/{joint.name}"
            owner = _OwnerBuilder(
                owner_id=owner_id,
                role="joint",
                parent_owner_id=previous_active_owner_id,
                finger_name=finger.name,
                joint_name=joint.name,
                reference_link=str(joint.child),
                component_ids=[],
            )
            joint_owners.append(owner)  # 全局 JOINT 顺序与 HandCfg 展平顺序一致
            owner_by_id[owner_id] = owner
            previous_active_owner_id = owner_id
        elif joint_index < first_active:
            if joint.is_tip:
                raise ValueError(f"fixed root '{joint.name}' before first active joint cannot be marked is_tip")
            owner = palm_owner  # fixed 根壳与 palm solid 同 owner
        elif joint_index > last_active:
            if joint.collisions and not joint.is_tip:
                raise ValueError(
                    f"fixed descendant '{joint.name}' carries collision geometry but is_tip is false; "
                    "generated owner assignment is ambiguous"
                )
            if not joint.collisions and not joint.is_tip:
                continue  # collision-free distal spacer 只延长运动学链，不创建空 TIP surface owner
            owner_id = f"tip/{finger.name}"
            owner = owner_by_id.get(owner_id)
            if owner is None:
                owner = _OwnerBuilder(
                    owner_id=owner_id,
                    role="tip",
                    parent_owner_id=previous_active_owner_id,
                    finger_name=finger.name,
                    joint_name=None,
                    reference_link=str(finger.joints[last_active].child),
                    component_ids=[],
                )
                tip_owners.append(owner)  # 一个手指全部显式 tip descendants 合并为一个 TIP owner
                owner_by_id[owner_id] = owner
        else:
            if joint.collisions:
                raise ValueError(
                    f"fixed joint '{joint.name}' lies between active joints and carries collision geometry; "
                    "generated owner assignment requires an explicit sidecar"
                )
            continue  # 无几何 fixed spacer 只影响 robots 运动学链，不产生 surface owner

        for collision_index, collision in enumerate(joint.collisions):
            component = _make_component(
                collision,
                component_id=f"finger/{finger.name}/joint/{joint.name}/collision/{collision_index}",
                owner_id=owner.owner_id,
                carrier_link=str(joint.child),
                collision_index=collision_index,
                source_joint_name=joint.name,
            )
            components.append(component)  # 每个 collision 实例只在此处写入一次
            owner.component_ids.append(component.component_id)


def _derive_anchor_seed(hand: HandCfg, finger: FingerCfg) -> AnchorSeedSemanticsCfg:
    r"""以导出后首活动关节 frame 原点构造一根手指的 palm 支持 seed。"""

    palm = cast(PalmCfg, hand.palm)  # HandCfg schema 已完成运行时归一化
    rotation, translation = _pose_transform(cast(PoseCfg, palm.origin))  # `{a}` -> palm 基准位姿
    for joint_index, joint in enumerate(finger.joints):
        origin = _exported_joint_origin(finger, joint, is_first=joint_index == 0)
        local_rotation, local_translation = _pose_transform(origin)
        rotation, translation = _compose_transform(rotation, translation, local_rotation, local_translation)
        if joint.joint_type == "revolute":
            return AnchorSeedSemanticsCfg(
                seed_id=f"finger/{finger.name}/first-active",
                finger_name=finger.name,
                first_active_joint_name=joint.name,
                support_owner_id="palm",
                position_a_m=translation,
                rotation_a=_flatten_rotation(rotation),
            )
        # 首活动关节之前只有 fixed 根段；其 home 变换已在上面严格复合。
    raise ValueError(f"generated finger '{finger.name}' has no active joint")


def _exported_joint_origin(finger: FingerCfg, joint: JointCfg, *, is_first: bool) -> PoseCfg:
    r"""复现当前 exporter 对首 joint 的 mount folding，再交给严格 $SE(3)$ 链复合。

    当前资产格式把 ``finger.mount`` 与首 joint origin 的平移、RPY 分量分别相加后写进 URDF。
    这不是一般刚体位姿组合，但它是已落盘 generated 资产的权威导出语义；迁移必须逐值复现。
    """

    joint_origin = cast(PoseCfg, joint.origin)  # JointCfg.__post_init__ 已归一化 PoseCfg
    if not is_first:
        return joint_origin.copy()
    mount = cast(PoseCfg, finger.mount)  # FingerCfg.__post_init__ 已归一化 PoseCfg
    return PoseCfg(
        pos=cast(Vector3, tuple(mount.pos[axis] + joint_origin.pos[axis] for axis in range(3))),
        rpy=cast(Vector3, tuple(mount.rpy[axis] + joint_origin.rpy[axis] for axis in range(3))),
    )


def _make_component(
    collision: CollisionGeometryCfg,
    *,
    component_id: str,
    owner_id: str,
    carrier_link: str,
    collision_index: int,
    source_joint_name: str | None,
) -> CollisionComponentSemanticsCfg:
    """把碰撞实例冻结为包含几何类型的 sidecar 事实。"""

    origin = cast(PoseCfg, collision.origin)  # CollisionGeometryCfg schema 已归一化 PoseCfg
    geometry_payload = collision.geometry.to_dict()  # ClassVar 类型不会自动进入 dataclass 字典
    geometry_payload = {"type": collision.geometry.kind, **geometry_payload}  # 显式补齐可逆分发标签
    return CollisionComponentSemanticsCfg(
        component_id=component_id,
        owner_id=owner_id,
        carrier_link=carrier_link,
        collision_index=collision_index,
        collision_name=collision.name,
        geometry_kind=collision.geometry.kind,
        geometry_payload=geometry_payload,
        origin_pos_m=_vector3(origin.pos, "collision.origin.pos"),
        origin_rpy_rad=_vector3(origin.rpy, "collision.origin.rpy"),
        source_joint_name=source_joint_name,
    )


def _resolve_generated_q_home(
    active_joint_names: tuple[str, ...],
    q_home_rad: Mapping[str, float] | None,
) -> dict[str, float]:
    """恢复 legacy generated 零 home，或验证调用方给出的完整显式 home。"""

    if q_home_rad is None:
        return {name: 0.0 for name in active_joint_names}  # generated-handcfg-v1 明确声明的迁移规则
    provided = set(q_home_rad)
    expected = set(active_joint_names)
    if provided != expected:
        missing = sorted(expected - provided)
        unknown = sorted(provided - expected)
        raise ValueError(f"q_home_rad must exactly cover active joints; missing={missing}, unknown={unknown}")
    return {name: float(q_home_rad[name]) for name in active_joint_names}


def _joint_limits(joint: JointCfg) -> tuple[float, float]:
    """读取活动关节有限合法域，禁止用默认无限区间掩盖资产缺失。"""

    if joint.limit is None:
        raise ValueError(f"active joint '{joint.name}' is missing limits")
    limit = cast(JointLimitCfg, joint.limit)  # JointCfg.__post_init__ 已把 mapping/sequence 收敛为 JointLimitCfg
    return (float(limit.lower), float(limit.upper))


def _pose_transform(pose: PoseCfg) -> tuple[tuple[Vector3, Vector3, Vector3], Vector3]:
    r"""把 URDF 固定轴 RPY 转成 $R=R_z(yaw)R_y(pitch)R_x(roll)$ 与平移。"""

    roll, pitch, yaw = pose.rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    return rotation, _vector3(pose.pos, "pose.pos")


def _compose_transform(
    parent_rotation: tuple[Vector3, Vector3, Vector3],
    parent_translation: Vector3,
    local_rotation: tuple[Vector3, Vector3, Vector3],
    local_translation: Vector3,
) -> tuple[tuple[Vector3, Vector3, Vector3], Vector3]:
    r"""严格复合 $T_{ac}=T_{ab}T_{bc}$。"""

    rotation = tuple(
        tuple(
            sum(parent_rotation[row][inner] * local_rotation[inner][column] for inner in range(3))
            for column in range(3)
        )
        for row in range(3)
    )
    rotated_translation = tuple(
        sum(parent_rotation[row][inner] * local_translation[inner] for inner in range(3)) for row in range(3)
    )
    translation = tuple(rotated_translation[axis] + parent_translation[axis] for axis in range(3))
    return rotation, translation  # type: ignore[return-value]


def _flatten_rotation(rotation: tuple[Vector3, Vector3, Vector3]) -> Matrix3Flat:
    """按行展开 $3\times3$ 旋转矩阵，便于 YAML 稳定序列化。"""

    values = tuple(value for row in rotation for value in row)
    return values  # type: ignore[return-value]


def _vector3(values: Sequence[float], field_name: str) -> Vector3:
    """把任意长度序列收敛为有限三维浮点 tuple。"""

    return _float_tuple(values, length=3, field_name=field_name)  # type: ignore[return-value]


def _float_tuple(values: Sequence[float], *, length: int, field_name: str) -> tuple[float, ...]:
    """验证固定长度浮点序列及其有限性。"""

    packed = tuple(float(value) for value in values)
    if len(packed) != length:
        raise ValueError(f"{field_name} must have length {length}, got {len(packed)}")
    if not all(math.isfinite(value) for value in packed):
        raise ValueError(f"{field_name} must contain finite values")
    return packed


def _content_hash(payload: Mapping[str, Any]) -> str:
    """对递归 dataclass/tuple/dict 计算顺序稳定的 SHA-256。"""

    serializable = _to_serializable(payload)
    encoded = json.dumps(serializable, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _to_serializable(value: Any) -> Any:
    """递归展开内容哈希所需的 dataclass 与不可变容器。"""

    if hasattr(value, "__dataclass_fields__"):
        return {key: _to_serializable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_serializable(item) for item in value]
    return value


__all__ = [
    "AnchorSeedSemanticsCfg",
    "CollisionComponentSemanticsCfg",
    "GENERATED_MIGRATION_VERSION",
    "GeometryOwnerSemanticsCfg",
    "HandGeometrySemanticsCfg",
    "KinematicJointSemanticsCfg",
    "SEMANTICS_SCHEMA_VERSION",
    "derive_generated_geometry_semantics",
    "geometry_semantics_from_dict",
    "geometry_semantics_to_dict",
]
