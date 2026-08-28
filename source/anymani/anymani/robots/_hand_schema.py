r"""Hand spawn selection 的 articulation-schema 校验。

本模块只消费 assets 交付的 ``HandContainer`` 与 canonical artifact manifest，不 import IsaacLab、
tasks 或 distill。校验目标是保证一个 batched articulation 的 joint/body 顺序确实同构；几何参数、
材质与任务 reward 不属于此边界。
"""

from __future__ import annotations

from anymani.assets.bank import HandContainer
from anymani.assets.canonical_runtime import (
    CANONICAL_HAND_SCHEMA_V1,
    CanonicalHandArtifact,
    validate_canonical_artifact,
)


def validate_canonical_hand_schema(
    containers: tuple[HandContainer, ...],
    artifacts: tuple[CanonicalHandArtifact, ...],
) -> None:
    r"""验证 canonical selection 的统一 schema、ordered names 与 row routing。"""

    schema = CANONICAL_HAND_SCHEMA_V1  # v1 固定 16-DOF / 25-body importer contract
    if len(containers) == 0 or len(containers) != len(artifacts):
        raise ValueError("canonical selection and artifact manifest must be non-empty and same length")
    for expected_row, (container, artifact) in enumerate(zip(containers, artifacts, strict=True)):
        validate_canonical_artifact(artifact, schema=schema)  # public adapter boundary 再核对 runtime 文件
        if artifact.routing.asset_id != container.asset_id:
            raise ValueError(
                f"canonical routing asset mismatch: manifest={artifact.routing.asset_id!r}, "
                f"container={container.asset_id!r}"
            )
        if artifact.routing.asset_row != expected_row:
            raise ValueError(
                f"canonical routing row must follow selection order: asset={container.asset_id!r}, "
                f"row={artifact.routing.asset_row}, expected={expected_row}"
            )
        if tuple(artifact.to_manifest()["schema"]["joint_names"]) != schema.joint_names:
            raise ValueError(f"canonical asset {container.asset_id!r} has invalid importer joint order manifest")
        if tuple(artifact.to_manifest()["schema"]["body_names"]) != schema.body_names:
            raise ValueError(f"canonical asset {container.asset_id!r} has invalid body order manifest")


def validate_same_hand_schema(containers: tuple[HandContainer, ...]) -> None:
    r"""检查 native ``MultiAssetSpawner`` selection 是否共享 articulation schema。

    签名包含 handedness-invariant topology、DOF、slot 顺序、每指 revolute DOF 和完整有序 joint
    sequence。``preserve_order=True`` 只保留 importer 已有顺序，不会自动重排，因此 joint sequence
    不同的资产不能进入同一个 batched articulation。
    """

    if len(containers) == 0:
        raise ValueError("HandSpawnAdapter requires at least one selected hand asset")
    reference = hand_schema_signature(containers[0])  # 第一项是 same-schema 参照
    for container in containers[1:]:
        signature = hand_schema_signature(container)  # 当前 sidecar 的有序 schema 摘要
        if signature != reference:
            raise ValueError(
                "selected hand assets are not same-schema: "
                f"reference={containers[0].asset_id}:{reference!r}, "
                f"offender={container.asset_id}:{signature!r}"
            )


def hand_schema_signature(container: HandContainer) -> tuple[object, ...]:
    r"""从 ``hand.yaml`` sidecar 抽取 same-schema 有序签名。"""

    sidecar = container.sidecar  # generated hand sidecar；保持 dict 以兼容资产 schema 演化
    finger_signature = tuple(
        (finger.get("name"), finger.get("revolute_dof")) for finger in sidecar.get("fingers", [])
    )  # 有序 finger schema，拒绝同 DOF 但 finger routing 不同的资产
    joint_sequence = ordered_revolute_joint_names(sidecar, asset_id=container.asset_id)  # `[J]` action 轴
    return (
        handedness_invariant_topology_key(sidecar.get("topology_name")),
        sidecar.get("dof"),
        tuple(sidecar.get("surviving_slots", [])),
        finger_signature,
        joint_sequence,
    )


def handedness_invariant_topology_key(topology_name: object) -> object:
    r"""仅移除 topology 名开头的一个物理 handedness token。"""

    if isinstance(topology_name, str) and topology_name.startswith(("left_", "right_")):
        return topology_name.split("_", maxsplit=1)[1]  # family/DOF/missing/mixed/connectivity tokens 全部保留
    return topology_name  # 非标准名称不猜测，交给完整签名精确比较


def ordered_revolute_joint_names(sidecar: dict[str, object], *, asset_id: str) -> tuple[str, ...]:
    r"""按 exporter 的 finger/joint 顺序提取 revolute articulation joint names。

    $$
    \mathcal J=(j_0,j_1,\ldots,j_{J-1}),\qquad J=\texttt{sidecar.dof}.
    $$

    fixed joints 只建立 link hierarchy，不占 policy action slot。
    """

    hand_cfg = sidecar.get("hand_cfg")  # 完整 generated schema；顶层 summary 不含 joint names
    if not isinstance(hand_cfg, dict):
        raise ValueError(f"asset {asset_id!r} sidecar must provide mapping hand_cfg for joint-order validation")
    fingers = hand_cfg.get("fingers")  # 有序 finger 轴，与 URDF exporter 一致
    if not isinstance(fingers, list):
        raise ValueError(f"asset {asset_id!r} sidecar hand_cfg.fingers must be a list")

    joint_names: list[str] = []  # 只收集 policy 可控 revolute joints
    for finger_index, finger_cfg in enumerate(fingers):
        if not isinstance(finger_cfg, dict) or not isinstance(finger_cfg.get("joints"), list):
            raise ValueError(f"asset {asset_id!r} hand_cfg.fingers[{finger_index}].joints must be a list")
        for joint_index, joint_cfg in enumerate(finger_cfg["joints"]):
            if not isinstance(joint_cfg, dict):
                raise ValueError(
                    f"asset {asset_id!r} hand_cfg.fingers[{finger_index}].joints[{joint_index}] must be a mapping"
                )
            if joint_cfg.get("joint_type") != "revolute":
                continue  # fixed joints 不进入 action/observation joint axis
            joint_name = joint_cfg.get("name")  # 必须等于 URDF `<joint name=...>`
            if not isinstance(joint_name, str) or not joint_name:
                raise ValueError(f"asset {asset_id!r} has a revolute joint without a non-empty name")
            joint_names.append(joint_name)

    expected_dof = sidecar.get("dof")  # exporter summary 中的可控 DOF 数 $J$
    if not isinstance(expected_dof, int) or len(joint_names) != expected_dof:
        raise ValueError(
            f"asset {asset_id!r} ordered revolute-joint count {len(joint_names)} does not match dof={expected_dof!r}"
        )
    if len(set(joint_names)) != len(joint_names):
        raise ValueError(f"asset {asset_id!r} ordered revolute-joint sequence contains duplicate names: {joint_names!r}")
    return tuple(joint_names)  # tuple 可哈希且适合精确 schema signature 比较


__all__ = ["validate_canonical_hand_schema", "validate_same_hand_schema"]
