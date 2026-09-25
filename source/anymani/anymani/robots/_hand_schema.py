(
    'Validate articulation schema for hand-spawn selections. This module consumes '
    'HandContainer and the canonical artifact manifest from assets; it imports no '
    'Isaac Lab, tasks, or distill. It ensures batched articulations share '
    'joint/body ordering. Geometry, materials, and task rewards are outside this '
    'contract.'
)

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
    'Validate one schema, ordered names, and row routing across a canonical selection.'

    schema = CANONICAL_HAND_SCHEMA_V1  # V1 importer contract: fixed 16 DoFs and 25 bodies.
    if len(containers) == 0 or len(containers) != len(artifacts):
        raise ValueError("canonical selection and artifact manifest must be non-empty and same length")
    for expected_row, (container, artifact) in enumerate(zip(containers, artifacts, strict=True)):
        validate_canonical_artifact(artifact, schema=schema)  # Revalidate runtime files at the public adapter boundary.
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
    (
        'Check whether a native MultiAssetSpawner selection shares one articulation '
        'schema. The signature includes handedness-invariant topology, DoF, slot '
        'order, per-finger revolute DoFs, and the full ordered joint sequence. '
        'preserve_order=True keeps importer order; it does not reorder assets, so '
        'different joint sequences cannot share one batched articulation.'
    )

    if len(containers) == 0:
        raise ValueError("HandSpawnAdapter requires at least one selected hand asset")
    reference = hand_schema_signature(containers[0])  # The first item is the same-schema reference.
    for container in containers[1:]:
        signature = hand_schema_signature(container)  # Ordered schema summary from the current sidecar.
        if signature != reference:
            raise ValueError(
                "selected hand assets are not same-schema: "
                f"reference={containers[0].asset_id}:{reference!r}, "
                f"offender={container.asset_id}:{signature!r}"
            )


def hand_schema_signature(container: HandContainer) -> tuple[object, ...]:
    'Extract the ordered same-schema signature from the hand.yaml sidecar.'

    sidecar = container.sidecar  # Generated-hand sidecar; retain dict form for asset-schema evolution.
    finger_signature = tuple(
        (finger.get("name"), finger.get("revolute_dof")) for finger in sidecar.get("fingers", [])
    )  # Ordered finger schema rejects assets with matching DoF but different finger routing.
    joint_sequence = ordered_revolute_joint_names(sidecar, asset_id=container.asset_id)  # Action axis [J].
    return (
        handedness_invariant_topology_key(sidecar.get("topology_name")),
        sidecar.get("dof"),
        tuple(sidecar.get("surviving_slots", [])),
        finger_signature,
        joint_sequence,
    )


def handedness_invariant_topology_key(topology_name: object) -> object:
    'Remove one physical handedness token only when it prefixes a topology name.'

    if isinstance(topology_name, str) and topology_name.startswith(("left_", "right_")):
        return topology_name.split("_", maxsplit=1)[1]  # Retain family, DoF, missing, mixed, and connectivity tokens.
    return topology_name  # Do not guess nonstandard names; compare the complete signature exactly.


def ordered_revolute_joint_names(sidecar: dict[str, object], *, asset_id: str) -> tuple[str, ...]:
    (
        'Extract revolute articulation joint names in exporter finger/joint order. '
        'Fixed joints create link hierarchy but no policy action slots.'
    )

    hand_cfg = sidecar.get("hand_cfg")  # Complete generated schema; top-level summary omits joint names.
    if not isinstance(hand_cfg, dict):
        raise ValueError(f"asset {asset_id!r} sidecar must provide mapping hand_cfg for joint-order validation")
    fingers = hand_cfg.get("fingers")  # Ordered finger axis, matching the URDF exporter.
    if not isinstance(fingers, list):
        raise ValueError(f"asset {asset_id!r} sidecar hand_cfg.fingers must be a list")

    joint_names: list[str] = []  # Collect only policy-controlled revolute joints.
    for finger_index, finger_cfg in enumerate(fingers):
        if not isinstance(finger_cfg, dict) or not isinstance(finger_cfg.get("joints"), list):
            raise ValueError(f"asset {asset_id!r} hand_cfg.fingers[{finger_index}].joints must be a list")
        for joint_index, joint_cfg in enumerate(finger_cfg["joints"]):
            if not isinstance(joint_cfg, dict):
                raise ValueError(
                    f"asset {asset_id!r} hand_cfg.fingers[{finger_index}].joints[{joint_index}] must be a mapping"
                )
            if joint_cfg.get("joint_type") != "revolute":
                continue  # Fixed joints are excluded from action/observation joint axes.
            joint_name = joint_cfg.get("name")  # Must match the URDF joint name exactly.
            if not isinstance(joint_name, str) or not joint_name:
                raise ValueError(f"asset {asset_id!r} has a revolute joint without a non-empty name")
            joint_names.append(joint_name)

    expected_dof = sidecar.get("dof")  # Controlled DoF count from the exporter summary: J.
    if not isinstance(expected_dof, int) or len(joint_names) != expected_dof:
        raise ValueError(
            f"asset {asset_id!r} ordered revolute-joint count {len(joint_names)} does not match dof={expected_dof!r}"
        )
    if len(set(joint_names)) != len(joint_names):
        raise ValueError(f"asset {asset_id!r} ordered revolute-joint sequence contains duplicate names: {joint_names!r}")
    return tuple(joint_names)  # Tuple form is hashable for exact schema-signature comparison.


__all__ = ["validate_canonical_hand_schema", "validate_same_hand_schema"]
