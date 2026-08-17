r"""统一 16-DOF / 25-body PhysX hand schema 与派生资产。

本模块是 generated hand 到 canonical runtime 的唯一资产真源。输入必须是 sidecar
恢复出的 typed ``HandCfg``；URDF 只作为 hash 与最终 importer 产物，不参与语义猜测。

v1 的物理模板固定为：

$$
\mathrm{palm} + 4\times(\mathrm{root\ fixed}+4\times\mathrm{revolute}+\mathrm{tip\ fixed}),
$$

因此每个 canonical hand 都有 $16$ 个 PhysX revolute slots、$25$ 个 bodies。真实
活动链的 proximal-to-distal 局部变换、axis、limit、collision、visual、mass 与 inertia
原样继承；缺失的 revolute slots 只在真实 tip 的 proximal 侧补入 inertial-only ghost。
ghost 不进入 geometry owner、contact learning 或 SSL identity。

IsaacLab importer 的 joint array 是 depth-major；本模块将该事实写入 manifest，避免把
序列化位置误当作 finger 语义：

$$
[\text{index}_{j0},\text{middle}_{j0},\text{ring}_{j0},\text{thumb}_{j0},
  \ldots,\text{index}_{j3},\text{middle}_{j3},\text{ring}_{j3},\text{thumb}_{j3}].
$$
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .asset_schema_core import InertialCfg, JointLimitCfg, MaterialCfg, PoseCfg, SphereGeometryCfg
from .asset_schema_embodiment import FingerCfg, HandCfg, JointCfg
from .exporter import UrdfWriter, UrdfWriterCfg
from .handedness import compose_poses


CANONICAL_HAND_SCHEMA_VERSION = "v1"  # 派生 URDF 与 manifest 的稳定 schema namespace
CANONICAL_HAND_SCHEMA_V1 = None  # 在模块末尾实例化，供运行时与测试共享同一对象
_IDENTIFIER_PATTERN = re.compile(r"^(?P<slot>[a-z]+)_j(?P<depth>[0-9]+)$")
_GHOST_LIMIT = 1.0e-3  # rad；先给 importer 一个有限小区间，startup event 再写精确 [0, 0]
_GHOST_MASS = 1.0e-6  # kg；只保持 PhysX articulated body 数值可用，不参与真实动力学语义
_GHOST_INERTIA = 1.0e-10  # kg m^2；ghost 的三轴对角惯量
_GHOST_MARKER_RADIUS = 1.0e-7  # m；透明 visual marker，不生成 collision


def _stable_digest(payload: Any) -> str:
    r"""对 JSON-compatible payload 做稳定 SHA-256 编码。"""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CanonicalHandSchemaCfg:
    r"""v1 canonical hand 的静态拓扑与数组顺序合同。

    ``semantic_finger_slots`` 用于 evidence / owner 语义；``physx_finger_order`` 用于
    importer 实际数组的 depth-major 序列化。两者必须是同一个四指集合，但不要求顺序相同。

    Args:
        semantic_finger_slots: 语义 finger slot，当前固定为 thumb/index/middle/ring。
        physx_finger_order: IsaacLab importer 使用的 traversal slot 顺序。
        max_revolute_per_finger: 每个 slot 的最大活动关节数，v1 固定为 4。
        version: schema namespace；改变它意味着新的 canonical artifact family。
    """

    semantic_finger_slots: tuple[str, ...] = ("thumb", "index", "middle", "ring")
    physx_finger_order: tuple[str, ...] = ("index", "middle", "ring", "thumb")
    max_revolute_per_finger: int = 4
    version: str = CANONICAL_HAND_SCHEMA_VERSION

    def __post_init__(self) -> None:
        # tuple 化保证 digest、manifest 与 dataclass equality 不受 list alias 影响。
        semantic = tuple(self.semantic_finger_slots)
        physx = tuple(self.physx_finger_order)
        object.__setattr__(self, "semantic_finger_slots", semantic)
        object.__setattr__(self, "physx_finger_order", physx)
        if set(semantic) != set(physx) or len(set(semantic)) != len(semantic) or len(set(physx)) != len(physx):
            raise ValueError("semantic_finger_slots and physx_finger_order must have the same unique finger set")
        if self.max_revolute_per_finger != 4:
            raise ValueError("canonical v1 requires exactly four revolute slots per finger")
        if self.version != CANONICAL_HAND_SCHEMA_VERSION:
            raise ValueError(f"unsupported canonical schema version: {self.version!r}")

    @property
    def dof_count(self) -> int:
        r"""返回 canonical revolute slot 总数 $4\times4=16$。"""

        return len(self.physx_finger_order) * self.max_revolute_per_finger

    @property
    def body_count(self) -> int:
        r"""返回 palm 加四指五段 body 的总数 $1+4\times6=25$。"""

        return 1 + len(self.semantic_finger_slots) * (self.max_revolute_per_finger + 2)

    @property
    def joint_names(self) -> tuple[str, ...]:
        r"""返回 importer 的 depth-major revolute joint 名称序列。"""

        return tuple(
            f"{slot}_j{depth}"
            for depth in range(self.max_revolute_per_finger)
            for slot in self.physx_finger_order
        )

    @property
    def body_names(self) -> tuple[str, ...]:
        r"""返回 canonical link 名称；finger body 按 semantic slot 解释。"""

        names = ["palm"]
        for slot in self.semantic_finger_slots:
            names.append(f"{slot}_root")
            names.extend(f"{slot}_link_j{depth}" for depth in range(self.max_revolute_per_finger))
            names.append(f"{slot}_tip")
        return tuple(names)

    @property
    def digest(self) -> str:
        r"""返回 schema 内容 digest，作为 runtime schema compatibility key。"""

        return _stable_digest(
            {
                "version": self.version,
                "semantic_finger_slots": self.semantic_finger_slots,
                "physx_finger_order": self.physx_finger_order,
                "max_revolute_per_finger": self.max_revolute_per_finger,
            }
        )


@dataclass(frozen=True)
class CanonicalHandRouting:
    r"""单个 source hand 在 canonical arrays 中的 active routing。

    ``active_joint_mask`` 与 ``schema.joint_names`` 同序；它是 action、observation、reward、
    PPO log-prob 与 geometry owner 共用的唯一 active mask。``source_to_canonical`` 只记录真实
    revolute joint 的对应关系，不把 ghost 当作 source asset 的物理语义。
    """

    asset_id: str
    source_dof_count: int
    source_joint_names: tuple[str, ...]
    active_joint_names: tuple[str, ...]
    active_joint_mask: tuple[bool, ...]
    active_tip_mask: tuple[bool, ...]
    source_to_canonical: tuple[tuple[str, str], ...]
    asset_row: int = 0
    handedness: str = "unknown"
    family: str = "generic"
    topology: str = "unknown"
    q_home: tuple[float, ...] = ()

    @property
    def active_dof_count(self) -> int:
        r"""返回真实 source revolute 数量。"""

        return sum(self.active_joint_mask)

    def to_dict(self) -> dict[str, Any]:
        r"""把 routing 序列化为 JSON manifest 原生容器。"""

        return {
            "asset_id": self.asset_id,
            "asset_row": self.asset_row,
            "source_dof_count": self.source_dof_count,
            "source_joint_names": list(self.source_joint_names),
            "active_joint_names": list(self.active_joint_names),
            "active_joint_mask": list(self.active_joint_mask),
            "active_tip_mask": list(self.active_tip_mask),
            "source_to_canonical": [list(item) for item in self.source_to_canonical],
            "handedness": self.handedness,
            "family": self.family,
            "topology": self.topology,
            "q_home": list(self.q_home),
        }


@dataclass(frozen=True)
class CanonicalHandArtifact:
    r"""一个 canonical URDF 与其可审计 routing manifest。"""

    schema_version: str
    schema_digest: str
    asset_id: str
    source_content_hash: str
    source_urdf_hash: str
    physical_geometry_hash: str
    canonical_urdf_hash: str
    canonical_urdf_path: str
    manifest_path: str
    routing: CanonicalHandRouting

    def to_manifest(self) -> dict[str, Any]:
        r"""返回完整 manifest；数组顺序与 hash 都显式落盘。"""

        return {
            "artifact_type": "anymani.canonical_hand",
            "schema_version": self.schema_version,
            "schema_digest": self.schema_digest,
            "asset_id": self.asset_id,
            "source_content_hash": self.source_content_hash,
            "source_urdf_hash": self.source_urdf_hash,
            "physical_geometry_hash": self.physical_geometry_hash,
            "canonical_urdf_hash": self.canonical_urdf_hash,
            "canonical_urdf_path": self.canonical_urdf_path,
            "manifest_path": self.manifest_path,
            "schema": {
                "semantic_finger_slots": list(CANONICAL_HAND_SCHEMA_V1.semantic_finger_slots),
                "physx_finger_order": list(CANONICAL_HAND_SCHEMA_V1.physx_finger_order),
                "joint_names": list(CANONICAL_HAND_SCHEMA_V1.joint_names),
                "body_names": list(CANONICAL_HAND_SCHEMA_V1.body_names),
            },
            "routing": self.routing.to_dict(),
        }

    @classmethod
    def from_manifest(cls, manifest_path: Path) -> "CanonicalHandArtifact":
        r"""从 manifest 恢复 artifact，并保留 manifest 中的相对路径语义。"""

        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        routing = CanonicalHandRouting(
            asset_id=document["routing"]["asset_id"],
            asset_row=document["routing"].get("asset_row", 0),
            source_dof_count=document["routing"]["source_dof_count"],
            source_joint_names=tuple(document["routing"]["source_joint_names"]),
            active_joint_names=tuple(document["routing"]["active_joint_names"]),
            active_joint_mask=tuple(document["routing"]["active_joint_mask"]),
            active_tip_mask=tuple(document["routing"]["active_tip_mask"]),
            source_to_canonical=tuple(tuple(item) for item in document["routing"]["source_to_canonical"]),
            handedness=document["routing"].get("handedness", "unknown"),
            family=document["routing"].get("family", "generic"),
            topology=document["routing"].get("topology", "unknown"),
            q_home=tuple(document["routing"].get("q_home", [])),
        )
        return cls(
            schema_version=document["schema_version"],
            schema_digest=document["schema_digest"],
            asset_id=document["asset_id"],
            source_content_hash=document["source_content_hash"],
            source_urdf_hash=document["source_urdf_hash"],
            physical_geometry_hash=document["physical_geometry_hash"],
            canonical_urdf_hash=document["canonical_urdf_hash"],
            canonical_urdf_path=document["canonical_urdf_path"],
            manifest_path=str(manifest_path),
            routing=routing,
        )


@dataclass(frozen=True)
class CanonicalHandGroupManifest:
    r"""同一 schema 下的一组 canonical artifacts。"""

    schema_version: str
    schema_digest: str
    artifacts: tuple[CanonicalHandArtifact, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        r"""返回 group manifest 的 JSON-compatible 表示。"""

        return {
            "artifact_type": "anymani.canonical_hand_group",
            "schema_version": self.schema_version,
            "schema_digest": self.schema_digest,
            "artifacts": [artifact.to_manifest() for artifact in self.artifacts],
        }

    def write(self, path: Path) -> None:
        r"""写入 group manifest，并保持稳定排序和 UTF-8 编码。"""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ghost_inertial() -> InertialCfg:
    r"""返回 ghost link 的固定最小质量与惯量。"""

    return InertialCfg(
        mass=_GHOST_MASS,
        inertia={"ixx": _GHOST_INERTIA, "iyy": _GHOST_INERTIA, "izz": _GHOST_INERTIA},
    )


def _ghost_visual() -> list[Any]:
    r"""返回透明、无 collision 的 marker visual。"""

    from .asset_schema_core import VisualGeometryCfg

    return [
        VisualGeometryCfg(
            name="canonical_ghost_marker",
            geometry=SphereGeometryCfg(radius=_GHOST_MARKER_RADIUS),
            material=MaterialCfg(name="canonical_ghost_invisible", rgba=(0.0, 0.0, 0.0, 0.0)),
        )
    ]


def _ghost_joint(
    name: str,
    parent: str,
    child: str,
    *,
    joint_type: str = "revolute",
    origin: PoseCfg | None = None,
    is_tip: bool = False,
) -> JointCfg:
    r"""构造 canonical ghost joint；真实活动 joint 不经过此函数。"""

    return JointCfg(
        name=name,
        parent=parent,
        child=child,
        joint_type=joint_type,
        axis=(1.0, 0.0, 0.0),
        limit=(
            JointLimitCfg(lower=-_GHOST_LIMIT, upper=_GHOST_LIMIT, effort=10.0, velocity=3.14)
            if joint_type != "fixed"
            else None
        ),
        origin=origin or PoseCfg(),
        inertial=_ghost_inertial(),
        visuals=_ghost_visual(),
        is_tip=is_tip,
        metadata={"canonical_ghost": True, "canonical_active": False},
    )


def _joint_depth(name: str, *, slot: str) -> int:
    r"""解析并校验 source revolute joint 的紧凑 proximal depth。"""

    match = _IDENTIFIER_PATTERN.fullmatch(name)
    if match is None or match.group("slot") != slot:
        raise ValueError(f"source joint {name!r} violates compact proximal order for slot {slot!r}")
    return int(match.group("depth"))


def _canonical_finger(
    finger: FingerCfg,
    schema: CanonicalHandSchemaCfg,
    *,
    force_ghost: bool = False,
) -> tuple[FingerCfg, list[str], bool, list[tuple[str, str]]]:
    r"""将一根 source finger lower 为 root + 4 revolute + tip 的固定链。"""

    source_revolute = [] if force_ghost else [joint for joint in finger.joints if joint.joint_type == "revolute"]
    if len(source_revolute) > schema.max_revolute_per_finger:
        raise ValueError(
            f"finger {finger.name!r} has {len(source_revolute)} revolute joints; canonical v1 supports at most 4"
        )
    depths = [_joint_depth(joint.name, slot=finger.name) for joint in source_revolute]
    if depths != list(range(len(depths))):
        raise ValueError(f"finger {finger.name!r} violates compact proximal order: {depths}")

    # 固定 root / tip 只从链的两端识别；活动链中间不能夹杂 fixed joint。
    first_fixed = None if force_ghost else (finger.joints[0] if finger.joints and finger.joints[0].joint_type == "fixed" else None)
    last_fixed = None if force_ghost else (finger.joints[-1] if finger.joints and finger.joints[-1].joint_type == "fixed" else None)
    interior_start = 1 if first_fixed is not None else 0
    interior_end = -1 if last_fixed is not None else None
    interior = [] if force_ghost else finger.joints[interior_start:interior_end]
    if any(joint.joint_type == "fixed" for joint in interior):
        raise ValueError(f"finger {finger.name!r} contains an interior fixed joint")
    if first_fixed is not None and last_fixed is first_fixed:
        last_fixed = None

    root_name = f"{finger.name}_root"
    tip_name = f"{finger.name}_tip"
    mount = finger.mount
    source_root = first_fixed
    source_tip = last_fixed
    if source_root is not None:
        root_origin = compose_poses(mount, source_root.origin)
        root = source_root.replace(
            name=f"{finger.name}_root_fixed",
            parent="palm",
            child=root_name,
            origin=root_origin,
            metadata={**source_root.metadata, "canonical_active": True, "source_joint": source_root.name},
        )
    else:
        first_active_origin = source_revolute[0].origin if source_revolute else PoseCfg()
        root = _ghost_joint(
            f"{finger.name}_root_fixed",
            "palm",
            root_name,
            joint_type="fixed",
            origin=compose_poses(mount, first_active_origin),
        )

    canonical_joints = [root]
    source_joint_names: list[str] = []
    source_to_canonical: list[tuple[str, str]] = []
    for depth in range(schema.max_revolute_per_finger):
        child = f"{finger.name}_link_j{depth}"
        if depth < len(source_revolute):
            source_joint = source_revolute[depth]
            origin = source_joint.origin if source_root is not None or depth > 0 else PoseCfg()
            # 无 root source 的首个活动 origin 已吸收到 adapter root；后续局部变换保持原样。
            canonical_joint = source_joint.replace(
                name=f"{finger.name}_j{depth}",
                parent=canonical_joints[-1].child,
                child=child,
                origin=origin,
                metadata={**source_joint.metadata, "canonical_active": True, "source_joint": source_joint.name},
            )
            source_joint_names.append(source_joint.name)
            source_to_canonical.append((source_joint.name, canonical_joint.name))
        else:
            canonical_joint = _ghost_joint(
                f"{finger.name}_j{depth}",
                canonical_joints[-1].child,
                child,
            )
        canonical_joints.append(canonical_joint)

    if source_tip is not None:
        tip_joint = source_tip.replace(
            name=f"{finger.name}_tip_fixed",
            parent=canonical_joints[-1].child,
            child=tip_name,
            metadata={**source_tip.metadata, "canonical_active": True, "source_joint": source_tip.name},
            is_tip=True,
        )
    else:
        tip_joint = _ghost_joint(
            f"{finger.name}_tip_fixed",
            canonical_joints[-1].child,
            tip_name,
            joint_type="fixed",
            is_tip=True,
        )
    canonical_joints.append(tip_joint)

    canonical_finger = FingerCfg(
        name=finger.name,
        parent_link="palm",
        mount=PoseCfg(),  # root adapter 已吸收 source mount，canonical exporter 不再重复折叠
        joints=canonical_joints,
        metadata={**finger.metadata, "canonical_schema": schema.version},
    )
    return canonical_finger, source_joint_names, bool(source_revolute), source_to_canonical


def lower_hand_to_canonical(
    hand: HandCfg,
    *,
    asset_id: str,
    schema: CanonicalHandSchemaCfg = None,
    asset_row: int = 0,
    topology: str = "unknown",
    q_home: tuple[float, ...] = (),
) -> tuple[HandCfg, CanonicalHandRouting]:
    r"""从 typed source ``HandCfg`` 派生 canonical hand 与 active routing。

    活动 source joint 必须已经是每指从 proximal 开始的 ``j0...j(n-1)``；本函数不对
    删除后的空洞重新猜测，也不删除真实 link。返回的 ``HandCfg`` 仅改变 canonical
    命名、补充 adapter/ghost 链和吸收 mount，真实活动 joint 的动力学字段保持不变。
    """

    schema = schema or CANONICAL_HAND_SCHEMA_V1
    by_slot = {finger.name: finger for finger in hand.fingers}
    unknown = sorted(set(by_slot) - set(schema.semantic_finger_slots))
    if unknown:
        raise ValueError(f"source hand contains unsupported finger slots: {unknown}")

    canonical_fingers: list[FingerCfg] = []
    source_joint_names: list[str] = []
    source_to_canonical: list[tuple[str, str]] = []
    tip_mask_by_slot: dict[str, bool] = {}
    for slot in schema.semantic_finger_slots:
        if slot in by_slot:
            canonical, names, has_active, routing = _canonical_finger(by_slot[slot], schema)
        else:
            canonical, names, has_active, routing = _canonical_finger(
                FingerCfg(
                    name=slot,
                    parent_link="palm",
                    mount=PoseCfg(),
                    joints=[_ghost_joint(f"{slot}_tip_fixed", "palm", f"{slot}_source_tip", joint_type="fixed")],
                ),
                schema,
                force_ghost=True,
            )
        canonical_fingers.append(canonical)
        source_joint_names.extend(names)
        source_to_canonical.extend(routing)
        tip_mask_by_slot[slot] = has_active

    canonical = HandCfg(
        name=f"{hand.name}_canonical_{schema.version}",
        palm=hand.palm.copy(),
        fingers=canonical_fingers,
        family=hand.family,
        handedness=hand.handedness,
        metadata={
            **hand.metadata,
            "canonical_schema_version": schema.version,
            "canonical_schema_digest": schema.digest,
            "canonical_source_asset_id": asset_id,
        },
    )
    source_names_to_canonical = dict(source_to_canonical)
    active_mask = tuple(name in source_names_to_canonical.values() for name in schema.joint_names)
    active_names = tuple(name for name, active in zip(schema.joint_names, active_mask) if active)
    routing = CanonicalHandRouting(
        asset_id=asset_id,
        asset_row=asset_row,
        source_dof_count=hand.dof_count,
        source_joint_names=tuple(source_joint_names),
        active_joint_names=active_names,
        active_joint_mask=active_mask,
        active_tip_mask=tuple(tip_mask_by_slot[slot] for slot in schema.physx_finger_order),
        source_to_canonical=tuple(source_to_canonical),
        handedness=hand.handedness,
        family=hand.family,
        topology=topology,
        q_home=tuple(q_home),
    )
    return canonical, routing


def materialize_canonical_artifact(
    hand: HandCfg,
    *,
    asset_id: str,
    output_root: Path,
    source_urdf_path: Path | None = None,
    schema: CanonicalHandSchemaCfg = None,
    asset_row: int = 0,
    topology: str = "unknown",
    q_home: tuple[float, ...] = (),
) -> CanonicalHandArtifact:
    r"""把 source hand materialize 到 ``outputs/canonical_runtime/v1/<cache-key>/``。

    cache key 同时依赖 schema、typed source content 与 source URDF bytes；这保证在 source
    sidecar、URDF 或 canonical schema 任一项改变时，不会静默复用旧派生产物。
    """

    schema = schema or CANONICAL_HAND_SCHEMA_V1
    source_content_hash = _stable_digest(hand.to_dict())
    source_urdf_hash = _sha256_file(source_urdf_path) if source_urdf_path is not None else _stable_digest("")
    cache_key = _stable_digest(
        {"schema_version": schema.version, "schema_digest": schema.digest, "source_content_hash": source_content_hash, "source_urdf_hash": source_urdf_hash}
    )
    artifact_dir = output_root / "canonical_runtime" / schema.version / cache_key
    manifest_path = artifact_dir / "canonical_runtime.json"
    urdf_path = artifact_dir / "hand.urdf"
    if manifest_path.exists() and urdf_path.exists():
        artifact = CanonicalHandArtifact.from_manifest(manifest_path)
        if artifact.schema_digest == schema.digest and artifact.source_content_hash == source_content_hash:
            return artifact

    canonical, routing = lower_hand_to_canonical(
        hand,
        asset_id=asset_id,
        schema=schema,
        asset_row=asset_row,
        topology=topology,
        q_home=q_home,
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result = UrdfWriter(UrdfWriterCfg(overwrite=True)).export(canonical, artifact_dir)
    if result.errors:
        raise RuntimeError(f"canonical URDF export failed: {result.errors}")
    artifact = CanonicalHandArtifact(
        schema_version=schema.version,
        schema_digest=schema.digest,
        asset_id=asset_id,
        source_content_hash=source_content_hash,
        source_urdf_hash=source_urdf_hash,
        physical_geometry_hash=_stable_digest(canonical.to_dict()),
        canonical_urdf_hash=_sha256_file(urdf_path),
        canonical_urdf_path=str(urdf_path),
        manifest_path=str(manifest_path),
        routing=routing,
    )
    manifest_path.write_text(json.dumps(artifact.to_manifest(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _sha256_file(path: Path | None) -> str:
    r"""计算文件内容 SHA-256；不存在的 source URDF 由调用方显式传入 None。"""

    if path is None:
        return _stable_digest("")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


CANONICAL_HAND_SCHEMA_V1 = CanonicalHandSchemaCfg()


__all__ = [
    "CANONICAL_HAND_SCHEMA_VERSION",
    "CANONICAL_HAND_SCHEMA_V1",
    "CanonicalHandSchemaCfg",
    "CanonicalHandRouting",
    "CanonicalHandArtifact",
    "CanonicalHandGroupManifest",
    "lower_hand_to_canonical",
    "materialize_canonical_artifact",
]
