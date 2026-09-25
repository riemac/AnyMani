"Combines structural, joint, and post-mutation geometry gates."

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, cast

from ..asset_base import AssetCfgBase, HandCfg
from ..asset_schema_core import PoseCfg
from ..asset_schema_embodiment import PalmCfg
from ._base import ValidationResult, ValidatorBase
from ._finger_length import FingerLengthConfig, evaluate_finger_axial_length
from ._sdf_clearance import SdfClearanceConfig
from ._sdf_service import CentralSdfServiceError, evaluate_finger_sdf_clearance_routed
from .finger_rules import FingerValidator, FingerValidatorCfg

# ============================================================================

# ============================================================================


@dataclass
class _HandValidatorStageCfg(AssetCfgBase):
    "Private intermediate record used by the validator implementation."

    dof_min: int | None = 1
    "Smallest allowed active degree-of-freedom count."

    dof_max: int | None = None
    "Largest allowed active degree-of-freedom count."

    finger_count_min: int | None = 3
    "Minimum number of surviving fingers after topology lowering."

    finger_count_max: int | None = 4
    "Count or linear dimension in the units declared by the associated schema."

    require_thumb: bool = True
    "Whether this generation or validation condition is active."

    thumb_min_revolute_dof: int | None = 3
    "Minimum active revolute joints required for the thumb."

    require_non_thumb_with_min_revolute_dof: int | None = 3
    "Minimum active revolute joints required on a surviving non-thumb finger."

    check_global_uniqueness: bool = True
    "Whether this generation or validation condition is active."

    check_mount_consistency: bool = True
    "Whether this generation or validation condition is active."

    check_finger_spacing: bool = True
    "Whether this generation or validation condition is active."

    min_finger_spacing: float = 0.015
    "Minimum accepted finger surface clearance in meters. The legacy recipe key is retained for compatibility."

    # In post-mutation mode this recipe field gates sampled surface-to-surface clearance, not mount-origin distance.

    finger: FingerValidatorCfg = field(default_factory=FingerValidatorCfg)
    "Finger configuration or semantic slot owned by the current operation."

    strict: bool = False
    "Whether missing or unverified semantic fields cause validation to fail."


@dataclass
class HandValidatorPreMadeCfg(_HandValidatorStageCfg):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    check_palm_thumb_binding: bool = True
    "Whether this generation or validation condition is active."


@dataclass
class HandValidatorPostMutateCfg(_HandValidatorStageCfg):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    check_finger_length: bool = False
    "Whether home-pose finger length is measured and gated."

    max_thumb_length: float | None = None
    "Maximum accepted thumb length in meters."

    max_non_thumb_length: float | None = None
    "Maximum accepted non-thumb length in meters."

    sdf_surface_samples_per_axis: int = 5
    "Unit direction expressed in the owning local frame."

    sdf_unsupported_geometry_policy: Literal["fail", "warn_skip"] = "fail"
    "Action for geometry not supported by the selected SDF backend."

    sdf_threshold_tolerance: float = 1e-9
    "Distance tolerance for signed-distance validation, in meters."

    sdf_device: Literal["auto", "cuda", "cpu"] = "auto"
    "Compute device used for signed-distance clearance validation."

    sdf_mesh_backend: Literal["auto", "warp", "trimesh"] = "auto"
    "Mesh signed-distance implementation; backend failures remain explicit."

    sdf_mesh_surface_samples: int = 4096
    "Mesh-surface sample count used by signed-distance checks."


@dataclass
class HandValidatorCfg(AssetCfgBase):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    PreMadeCfg = HandValidatorPreMadeCfg
    PostMutateCfg = HandValidatorPostMutateCfg

    class_type: type[HandValidator] | None = None
    "Associated runtime implementation for this configuration class."

    pre_made: HandValidatorPreMadeCfg = field(default_factory=HandValidatorPreMadeCfg)
    "Pre-made topology validation settings."

    post_mutate: HandValidatorPostMutateCfg = field(default_factory=HandValidatorPostMutateCfg)
    "Post-mutation validator and proposal configuration."

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = HandValidator


# ============================================================================

# ============================================================================


class HandValidator(ValidatorBase):
    "Checks generated geometry against explicit acceptance gates."

    cfg: HandValidatorCfg

    def __init__(self, cfg: HandValidatorCfg):
        self.cfg = cfg

    def validate(
        self,
        target: HandCfg,
        *,
        stage: str = "post_mutate",
    ) -> ValidationResult:  # type: ignore[override]
        "Runs the configured acceptance gates and returns explicit candidate evidence."

        if stage == "pre_made":
            return self._validate_with_stage_cfg(target, stage_cfg=self.cfg.pre_made, stage=stage)
        if stage == "post_mutate":
            return self._validate_with_stage_cfg(target, stage_cfg=self.cfg.post_mutate, stage=stage)
        raise ValueError(f"Unsupported validation stage {stage!r}; expected 'pre_made' or 'post_mutate'.")

    def validate_pre_made(self, target: HandCfg) -> ValidationResult:
        'Validates pre-made hand topology and geometry.'

        return self.validate(target, stage="pre_made")

    def validate_post_mutate(self, target: HandCfg) -> ValidationResult:
        'Validates post-mutation hand geometry and connectivity.'

        return self.validate(target, stage="post_mutate")

    def _validate_with_stage_cfg(
        self,
        target: HandCfg,
        *,
        stage_cfg: _HandValidatorStageCfg,
        stage: str,
    ) -> ValidationResult:

        result = ValidationResult()
        finger_validator = FingerValidator(stage_cfg.finger)

        for finger in target.fingers:
            result.merge(finger_validator.validate(finger))

        if isinstance(stage_cfg, HandValidatorPreMadeCfg) and stage_cfg.check_palm_thumb_binding:
            self._validate_palm_thumb_binding(target, result=result)

        if stage_cfg.check_global_uniqueness:
            finger_names = [finger.name for finger in target.fingers]
            if len(finger_names) != len(set(finger_names)):
                result.add_error(
                    f"hand '{target.name}'[{stage}]: duplicate finger names",
                    code="hand.duplicate_finger_names",
                )

            joint_names = [joint.name for joint in target.iter_joints()]
            if len(joint_names) != len(set(joint_names)):
                result.add_error(
                    f"hand '{target.name}'[{stage}]: duplicate joint names",
                    code="hand.duplicate_joint_names",
                )

            palm = cast(PalmCfg, target.palm)
            link_names = [palm.name] + [joint.child for joint in target.iter_joints()]
            if len(link_names) != len(set(link_names)):
                result.add_error(
                    f"hand '{target.name}'[{stage}]: duplicate link names",
                    code="hand.duplicate_link_names",
                )

        dof = target.dof_count
        if stage_cfg.dof_min is not None and dof < stage_cfg.dof_min:
            result.add_error(
                f"hand '{target.name}'[{stage}]: dof {dof} < min {stage_cfg.dof_min}",
                code="hand.dof_below_min",
            )
        if stage_cfg.dof_max is not None and dof > stage_cfg.dof_max:
            result.warnings.append(f"hand '{target.name}'[{stage}]: dof {dof} > max {stage_cfg.dof_max}")

        finger_count = len(target.fingers)
        if stage_cfg.finger_count_min is not None and finger_count < stage_cfg.finger_count_min:
            result.add_error(
                f"hand '{target.name}'[{stage}]: finger count {finger_count} < min {stage_cfg.finger_count_min}",
                code="hand.finger_count_below_min",
            )
        if stage_cfg.finger_count_max is not None and finger_count > stage_cfg.finger_count_max:
            result.add_error(
                f"hand '{target.name}'[{stage}]: finger count {finger_count} > max {stage_cfg.finger_count_max}",
                code="hand.finger_count_above_max",
            )

        thumb_finger = next((finger for finger in target.fingers if finger.name == "thumb"), None)
        if stage_cfg.require_thumb and thumb_finger is None:
            result.add_error(
                f"hand '{target.name}'[{stage}]: missing required thumb finger",
                code="hand.missing_required_thumb",
            )

        if thumb_finger is not None and stage_cfg.thumb_min_revolute_dof is not None:
            thumb_dof = _revolute_dof_count(thumb_finger)
            if thumb_dof < stage_cfg.thumb_min_revolute_dof:
                result.add_error(
                    f"hand '{target.name}'[{stage}]: thumb revolute dof {thumb_dof} < min {stage_cfg.thumb_min_revolute_dof}",
                    code="hand.thumb_revolute_dof_below_min",
                )

        if stage_cfg.require_non_thumb_with_min_revolute_dof is not None:
            threshold = stage_cfg.require_non_thumb_with_min_revolute_dof
            non_thumb_dofs = [
                (finger.name, _revolute_dof_count(finger))
                for finger in target.fingers
                if finger.name != "thumb"
            ]
            if not any(dof >= threshold for _, dof in non_thumb_dofs):
                result.add_error(
                    f"hand '{target.name}'[{stage}]: expected at least one non-thumb finger with revolute dof >= {threshold}, "
                    f"got {non_thumb_dofs!r}",
                    code="hand.non_thumb_revolute_dof_below_min",
                )

        if stage_cfg.check_mount_consistency:
            for finger in target.fingers:
                if finger.parent_link != palm.name:
                    result.add_error(
                        f"finger '{finger.name}'[{stage}] parent_link '{finger.parent_link}' != palm '{palm.name}'",
                        code="hand.finger_mount_parent_mismatch",
                    )

        if stage_cfg.check_finger_spacing:
            if isinstance(stage_cfg, HandValidatorPostMutateCfg):
                self._validate_post_mutate_sdf_clearance(target, stage_cfg=stage_cfg, result=result)
            else:
                self._validate_mount_origin_spacing(target, stage_cfg=stage_cfg, stage=stage, result=result)

        if isinstance(stage_cfg, HandValidatorPostMutateCfg) and stage_cfg.check_finger_length:
            self._validate_post_mutate_finger_length(target, stage_cfg=stage_cfg, result=result)

        if stage_cfg.strict:
            result = result.as_strict()
        result.passed = len(result.errors) == 0
        return result

    def _validate_mount_origin_spacing(
        self,
        target: HandCfg,
        *,
        stage_cfg: _HandValidatorStageCfg,
        stage: str,
        result: ValidationResult,
    ) -> None:

        mounts = [(finger.name, cast(PoseCfg, finger.mount).pos) for finger in target.fingers]
        for idx in range(len(mounts)):
            for jdx in range(idx + 1, len(mounts)):
                name_i, pos_i = mounts[idx]
                name_j, pos_j = mounts[jdx]
                distance = math.sqrt(sum((lhs - rhs) ** 2 for lhs, rhs in zip(pos_i, pos_j)))
                if distance < stage_cfg.min_finger_spacing:
                    result.warnings.append(
                        f"finger spacing '{name_i}'-'{name_j}'[{stage}, mount_origin_approx]: "
                        f"{distance * 100.0:.2f} cm < min {stage_cfg.min_finger_spacing * 100.0:.2f} cm"
                    )

    def _validate_post_mutate_sdf_clearance(
        self,
        target: HandCfg,
        *,
        stage_cfg: HandValidatorPostMutateCfg,
        result: ValidationResult,
    ) -> None:

        try:
            clearance = evaluate_finger_sdf_clearance_routed(
                target,
                SdfClearanceConfig(
                    min_clearance=stage_cfg.min_finger_spacing,
                    surface_samples_per_axis=stage_cfg.sdf_surface_samples_per_axis,
                    unsupported_policy=stage_cfg.sdf_unsupported_geometry_policy,
                    tolerance=stage_cfg.sdf_threshold_tolerance,
                    device=stage_cfg.sdf_device,
                    mesh_backend=stage_cfg.sdf_mesh_backend,
                    mesh_surface_samples=stage_cfg.sdf_mesh_surface_samples,
                ),
            )
        except CentralSdfServiceError:


            raise
        except (RuntimeError, ValueError) as exc:
            result.add_error(
                f"finger spacing sdf[post_mutate]: {exc}",
                code="hand.finger_spacing_sdf_evaluation_failed",
            )
            result.metadata["finger_spacing_certificate"] = {
                "pose_scope": "post_mutate_home_pose",
                "geometry_scope": "collision_geometry_only",
                "sdf_kind": "sampled_surface_sdf_approx",
                "complete": False,
                "device": stage_cfg.sdf_device,
                "mesh_sdf": {
                    "requested_backend": stage_cfg.sdf_mesh_backend,
                    "actual_backend": "none",
                    "mesh_query_count": 0,
                    "mesh_sample_count": 0,
                    "fallback_events": [],
                },
                "skipped_bodies": [],
                "not_certified": [
                    "all_pose_collision_free",
                    "mesh_exact_clearance",
                    "trajectory_safety",
                    "physics_runtime_safety",
                ],
            }
            return

        certificate = clearance.certificate.to_dict()
        result.metadata["finger_spacing_certificate"] = certificate
        if not clearance.certificate.complete:
            result.add_error(
                "finger spacing sdf[post_mutate]: incomplete certificate; "
                f"skipped_bodies={certificate.get('skipped_bodies', [])!r}",
                code="hand.finger_spacing_sdf_certificate_incomplete",
            )

        for pair in clearance.violations:
            result.add_error(
                f"finger spacing '{pair.finger_i}'-'{pair.finger_j}'[post_mutate, sdf_clearance]: "
                f"{pair.clearance * 100.0:.2f} cm < min {stage_cfg.min_finger_spacing * 100.0:.2f} cm "
                f"(i_to_j={pair.direction_i_to_j * 100.0:.2f} cm, "
                f"j_to_i={pair.direction_j_to_i * 100.0:.2f} cm)",
                code="hand.finger_spacing_sdf_below_min",
            )

    def _validate_post_mutate_finger_length(
        self,
        target: HandCfg,
        *,
        stage_cfg: HandValidatorPostMutateCfg,
        result: ValidationResult,
    ) -> None:

        try:
            finger_length = evaluate_finger_axial_length(
                target,
                FingerLengthConfig(
                    max_thumb_length=stage_cfg.max_thumb_length,
                    max_non_thumb_length=stage_cfg.max_non_thumb_length,
                ),
            )
        except (RuntimeError, ValueError) as exc:
            result.add_error(
                f"finger length[post_mutate]: {exc}",
                code="hand.finger_length_evaluation_failed",
            )
            result.metadata["finger_length_certificate"] = {
                "pose_scope": "post_mutate_home_pose",
                "geometry_scope": "collision_geometry_only",
                "length_kind": "axial_projection_extent",
                "complete": False,
                "skipped_bodies": [],
                "not_certified": [
                    "all_pose_length",
                    "bent_chain_geodesic_length",
                    "physics_runtime_safety",
                ],
                "thresholds": {
                    "thumb": stage_cfg.max_thumb_length,
                    "non_thumb": stage_cfg.max_non_thumb_length,
                },
                "measurements": [],
                "violations": [],
            }
            return

        certificate = finger_length.certificate.to_dict()
        result.metadata["finger_length_certificate"] = certificate
        if not finger_length.certificate.complete:
            result.add_error(
                "finger length[post_mutate]: incomplete certificate; "
                f"skipped_bodies={certificate.get('skipped_bodies', [])!r}",
                code="hand.finger_length_certificate_incomplete",
            )

        for measurement in finger_length.violations:
            threshold = measurement.threshold if measurement.threshold is not None else float("nan")
            result.add_error(
                f"finger '{measurement.finger_name}'[post_mutate, axial_length]: "
                f"{measurement.axial_length * 100.0:.2f} cm > max {threshold * 100.0:.2f} cm "
                f"(role={measurement.role}, axis_source={measurement.axis_source})",
                code="hand.finger_axial_length_above_max",
            )

    def _validate_palm_thumb_binding(self, target: HandCfg, *, result: ValidationResult) -> None:

        thumb_finger = next((finger for finger in target.fingers if finger.name == "thumb"), None)
        if thumb_finger is None:
            return

        metadata = dict(target.metadata or {})
        premade_metadata = metadata.get("premade_connectivity")
        if not isinstance(premade_metadata, dict):
            premade_metadata = metadata.get("premade_topology")
        if not isinstance(premade_metadata, dict):
            return

        slot_family_map = premade_metadata.get("slot_family_map")
        if not isinstance(slot_family_map, dict):
            return

        thumb_family = slot_family_map.get("thumb")
        if thumb_family is None:
            return

        palm_family = target.family
        if str(thumb_family) != str(palm_family):
            result.add_error(
                f"hand '{target.name}'[pre_made]: palm family {palm_family!r} requires thumb family to match, "
                f"got thumb family {thumb_family!r}",
                code="hand.palm_thumb_family_mismatch",
            )


def _revolute_dof_count(finger) -> int:

    return sum(1 for joint in finger.joints if joint.joint_type == "revolute")


__all__ = ["HandValidatorCfg", "HandValidator"]
