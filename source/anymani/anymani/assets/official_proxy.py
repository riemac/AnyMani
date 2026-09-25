"Builds a canonical proxy from explicitly parsed official-hand geometry without modifying source URDF data."

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import PurePosixPath
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from .asset_schema_geometry import (
    AnchorSeedSemanticsCfg,
    CollisionComponentSemanticsCfg,
    HandGeometrySemanticsCfg,
    _content_hash,
)
from .bank.hand_container import HandContainer
from .bank.official_hand import OfficialHandSemanticsCfg, forward_kinematics, load_official_hand_semantics

OFFICIAL_SEMANTIC_ROTATION = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
OFFICIAL_PROXY_VERSION = "owner-boxes-native-tips-v1"


@dataclass(frozen=True)
class OfficialProxyBundle:
    "Proxy geometry, kinematic semantics, and source hashes derived from one official hand."

    native: OfficialHandSemanticsCfg
    container: HandContainer
    palm_bounds_h: tuple[tuple[float, ...], tuple[float, ...]]
    owner_bounds: dict[str, list[list[float]]]

    def manifest(self) -> dict[str, Any]:
        "Serializes the official proxy hand and its source geometry identity."

        semantics = self.container.geometry_semantics
        assert semantics is not None
        return {
            "proxy_recipe": OFFICIAL_PROXY_VERSION,
            "native": self.native.to_dict(),
            "proxy_geometry_semantics": asdict(semantics),
            "palm_bounds_h_m": self.palm_bounds_h,
            "owner_bounds_reference_m": self.owner_bounds,
            "physics_urdf": self.native.source_urdf_path,
            "physics_urdf_sha256": self.native.source_digest,
            "representation_proxy_only": True,
        }


def _pose(position, rpy) -> np.ndarray:

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    transform[:3, 3] = position
    return transform


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:

    return points @ transform[:3, :3].T + transform[:3, 3]


def _component_vertices(component: CollisionComponentSemanticsCfg, description: OfficialHandSemanticsCfg) -> np.ndarray:

    payload = component.geometry_payload
    if component.geometry_kind == "box":
        signs = np.asarray(tuple(product((-0.5, 0.5), repeat=3)), dtype=np.float64)
        return signs * np.asarray(payload["size"], dtype=np.float64)
    if component.geometry_kind == "mesh":
        uri = str(payload.get("source_uri", payload.get("file_path")))
        match = next((item for item in description.mesh_digests if item.source_uri == uri), None)
        if match is None:
            raise ValueError(f"component {component.component_id} mesh URI is absent from source metadata")
        mesh = trimesh.load(match.resolved_path, force="mesh", process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        return vertices * np.asarray(payload.get("scale", (1.0, 1.0, 1.0)), dtype=np.float64)
    raise ValueError(f"official proxy has no declared rule for {component.geometry_kind!r}")


def _owner_vertices(description: OfficialHandSemanticsCfg) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:

    transforms = {name: np.asarray(value).reshape(4, 4) for name, value in forward_kinematics(description).items()}
    components = {component.component_id: component for component in description.geometry_semantics.components}
    vertices_by_owner: dict[str, np.ndarray] = {}
    for owner in description.geometry_semantics.owners:
        inverse_reference = np.linalg.inv(transforms[owner.reference_link])
        parts: list[np.ndarray] = []
        for component_id in owner.component_ids:
            component = components[component_id]
            local = _pose(component.origin_pos_m, component.origin_rpy_rad)
            reference_from_component = inverse_reference @ transforms[component.carrier_link] @ local
            parts.append(_transform_points(_component_vertices(component, description), reference_from_component))
        if not parts:
            raise ValueError(f"official representation owner {owner.owner_id!r} has no physical component")
        vertices_by_owner[owner.owner_id] = np.concatenate(parts, axis=0)
    return vertices_by_owner, transforms


def build_official_proxy(description: OfficialHandSemanticsCfg) -> OfficialProxyBundle:
    'Builds official proxy.'

    raw_vertices, raw_transforms = _owner_vertices(description)
    palm = next(owner for owner in description.geometry_semantics.owners if owner.role == "palm")
    rotation = np.asarray(OFFICIAL_SEMANTIC_ROTATION).reshape(3, 3)
    palm_root = _transform_points(raw_vertices[palm.owner_id], raw_transforms[palm.reference_link])
    palm_rotated = palm_root @ rotation.T
    bounds = np.stack((palm_rotated.min(axis=0), palm_rotated.max(axis=0)))
    center = bounds.mean(axis=0)
    translation = -np.asarray((center[0], bounds[0, 1], center[2]))
    native = load_official_hand_semantics(
        description.source_urdf_path,
        asset_id=description.asset_id,
        semantic_R_ha=OFFICIAL_SEMANTIC_ROTATION,
        semantic_p_ha=tuple(translation),
    )
    semantics = native.geometry_semantics
    by_id = {component.component_id: component for component in semantics.components}
    owner_bounds: dict[str, list[list[float]]] = {}
    components: list[CollisionComponentSemanticsCfg] = []
    owners = []
    for owner in semantics.owners:
        points = raw_vertices[owner.owner_id]
        low, high = points.min(axis=0), points.max(axis=0)
        owner_bounds[owner.owner_id] = [low.tolist(), high.tolist()]
        if owner.role == "tip":
            components.extend(by_id[key] for key in owner.component_ids)
            owners.append(owner)
            continue
        component_id = f"proxy:{owner.owner_id}:box"
        size = high - low
        if (size <= 0).any():
            raise ValueError(f"degenerate official proxy owner {owner.owner_id!r}")
        components.append(
            CollisionComponentSemanticsCfg(
                component_id=component_id,
                owner_id=owner.owner_id,
                carrier_link=owner.reference_link,
                collision_index=0,
                collision_name=component_id,
                geometry_kind="box",
                geometry_payload={"type": "box", "size": tuple(size)},
                origin_pos_m=tuple((low + high) / 2),
                origin_rpy_rad=(0.0, 0.0, 0.0),
                source_joint_name=owner.joint_name,
            )
        )
        owners.append(replace(owner, component_ids=(component_id,)))


    anchors = []
    for joint in native.joints:
        if joint.depth == 0:
            transform = raw_transforms[joint.child_link]
            anchors.append(
                AnchorSeedSemanticsCfg(
                    seed_id=f"{joint.finger_name}:root",
                    finger_name=joint.finger_name,
                    first_active_joint_name=joint.source_name,
                    support_owner_id=palm.owner_id,
                    position_a_m=tuple(transform[:3, 3]),
                    rotation_a=tuple(transform[:3, :3].reshape(-1)),
                )
            )
    payload = {name: getattr(semantics, name) for name in semantics.__dataclass_fields__ if name != "content_hash"}
    payload.update(
        migration_version=OFFICIAL_PROXY_VERSION,
        components=tuple(components),
        owners=tuple(owners),
        anchor_seeds=tuple(anchors),
    )
    proxy_semantics = HandGeometrySemanticsCfg(**payload, content_hash=_content_hash(payload))


    from pathlib import Path

    virtual_to_real = {PurePosixPath("hand.urdf"): Path(native.source_urdf_path)}
    virtual_to_real.update({PurePosixPath(item.source_uri): Path(item.resolved_path) for item in native.mesh_digests})
    container = HandContainer(
        asset_id=native.asset_id,
        virtual_to_real=virtual_to_real,
        real_to_virtual={path: name for name, path in virtual_to_real.items()},
        source_kind="official",
        sidecar={
            "representation_proxy_only": True,
            "proxy_recipe": OFFICIAL_PROXY_VERSION,
            "native_source_sha256": native.source_digest,
        },
        geometry_semantics=proxy_semantics,
    )
    calibrated_bounds = bounds + translation
    return OfficialProxyBundle(native, container, tuple(tuple(row) for row in calibrated_bounds), owner_bounds)


def build_native_tip_regions(description: OfficialHandSemanticsCfg) -> dict[str, dict[str, Any]]:
    'Builds native tip regions.'

    components = {component.component_id: component for component in description.geometry_semantics.components}
    regions: dict[str, dict[str, Any]] = {}
    for owner in description.geometry_semantics.owners:
        if owner.role != "tip":
            continue
        body = description.tip_collision_body_names[owner.finger_name]
        points = []
        for name in owner.component_ids:
            component = components[name]
            if component.carrier_link != body:
                raise ValueError("native TIP region requires a single declared carrier body")
            points.append(
                _transform_points(
                    _component_vertices(component, description),
                    _pose(component.origin_pos_m, component.origin_rpy_rad),
                )
            )
        hull = ConvexHull(np.concatenate(points, axis=0))
        regions[owner.finger_name] = {
            "carrier_link": body,
            "component_ids": list(owner.component_ids),
            "planes_body": hull.equations.tolist(),
            "hull_vertex_count": len(hull.vertices),
            "source_urdf_sha256": description.source_digest,
            "collision_approximation": "convexHull",
        }
    return regions
