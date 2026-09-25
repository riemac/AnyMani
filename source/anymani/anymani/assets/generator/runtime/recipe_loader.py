"Loads declarative mutation recipes into typed configuration objects."

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from ...asset_base import AssetCfgBase
from ...asset_physics import AssetPhysicsCfg
from ...builder.hand_builders import GripperLikeHandBuilderCfg
from ...exporter.hand_exporter import HandExporterCfg
from ...exporter.sidecar import SidecarCfg
from ...exporter.urdf_writer import UrdfWriterCfg
from ...presets import make_human_like_builder_cfg
from ...validator.finger_rules import FingerValidatorCfg
from ...validator.hand_rules import HandValidatorCfg
from ...validator.joint_rules import JointValidatorCfg
from ..hand_generator import HandGeneratorCfg
from ..mutate import (
    HandMutatorCfg,
    LimitTweakCfg,
    LinkScaleCfg,
    MountPerturbCfg,
    TipReplaceCfg,
)

# ============================================================================
#  Recipe Loader
# ============================================================================


class RecipeLoader:
    "Converts declarative mutation recipes into validated typed operator configs."

    @staticmethod
    def load(path: str | Path) -> HandGeneratorCfg:
        "Loads and validates a mutation recipe into typed operator configs."

        recipe_path = Path(path)
        if not recipe_path.exists():
            raise FileNotFoundError(recipe_path)

        raw = yaml.safe_load(recipe_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"recipe root must be a mapping, got {type(raw).__name__}")
        return RecipeLoader.load_dict(raw)

    @staticmethod
    def load_dict(raw: dict[str, Any]) -> HandGeneratorCfg:
        'Loads and validates typed configuration from a dictionary.'

        data = deepcopy(raw)



        if "export_dir" in data and "output_dir" not in data:
            data["output_dir"] = data.pop("export_dir")



        #


        #

        data.pop("sampling_strategy", None)

        removed_root_fields = [
            field_name
            for field_name in (
                "output_layout",
                "run_name",
                "run_policy",
                "layout",
            )
            if field_name in data
        ]
        if removed_root_fields:
            raise ValueError(
                "Removed HandGeneratorCfg fields in recipe: "
                f"{removed_root_fields}. "
                "Use the fixed topology-root contract instead: "
                "pre-made -> <group>/<topology>/, mutate-only -> <topology>/<mutate_timestamp>/<sample_id>/."
            )

        if "Made" in data and isinstance(data["Made"], dict):
            data["Made"] = _build_made_cfg(data["Made"])
        if "Mutate" in data and isinstance(data["Mutate"], dict):
            data["Mutate"] = _build_mutate_cfg(data["Mutate"])
        if "Validate" in data and isinstance(data["Validate"], dict):
            data["Validate"] = _build_validate_cfg(data["Validate"])
        if "Export" in data and isinstance(data["Export"], dict):
            data["Export"] = _build_export_cfg(data["Export"])
        if "Physics" in data and isinstance(data["Physics"], dict):
            data["Physics"] = _build_physics_cfg(data["Physics"])

        return HandGeneratorCfg(**data)

    @staticmethod
    def dump(cfg: HandGeneratorCfg) -> dict[str, Any]:
        "Writes a typed mutator recipe in the stable YAML format."

        dumped = _dump_value(cfg)
        if not isinstance(dumped, dict):
            raise TypeError("HandGeneratorCfg dump result must be a mapping")
        return dumped

    @staticmethod
    def save(cfg: HandGeneratorCfg, path: str | Path) -> None:
        "Saves the normalized mutation recipe to the requested path."

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            yaml.safe_dump(RecipeLoader.dump(cfg), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )


# ============================================================================

# ============================================================================


def _build_mutate_cfg(raw: dict[str, Any]) -> HandMutatorCfg:

    data = deepcopy(raw)
    legacy_tool_cfg_map = {
        "link_scale": LinkScaleCfg,
        "tip_replace": TipReplaceCfg,
        "limit_tweak": LimitTweakCfg,
        "mount_perturb": MountPerturbCfg,
    }
    removed_tool_names = {"joint_delete", "finger_replace"}
    tuple_fields = {
        "link_scale": ("link_scale", "clip"),
        "tip_replace": ("target_fingers",),
        "limit_tweak": ("joint_range",),
        "mount_perturb": (
            "pos_radius",
            "rot_radius",
            "thumb_pos_radius",
            "thumb_rot_radius",
            "mirror_yaw_range",
            "mirror_x_range",
        ),
    }

    mutate_cfg = HandMutatorCfg()
    for key in list(data.keys()):
        if key == "class_type":
            continue
        if key in {"terms", "order", "on_reject", "step_validate", "prefer_cuda_sampling"}:
            raise ValueError(f"Mutate.{key} is not supported by IsaacLab-style post-mutate cfg.")
        if key in removed_tool_names:
            raise ValueError(f"Mutate.{key} has been removed from post-mutate; move it out of Mutate.")

        payload = data.pop(key)
        if not isinstance(payload, dict):
            continue

        if "cfg" in payload:
            setattr(mutate_cfg, key, _build_named_mutator_term_cfg(key, payload))
            continue

        if key in legacy_tool_cfg_map:
            setattr(
                mutate_cfg,
                key,
                _build_legacy_mutator_cfg(
                    key,
                    payload,
                    cfg_cls=legacy_tool_cfg_map[key],
                    tuple_field_names=tuple_fields.get(key, ()),
                ),
            )
            continue

        raise ValueError(f"Unknown mutate term: {key!r}")

    return mutate_cfg


def _build_named_mutator_term_cfg(term_name: str, raw: dict[str, Any]) -> AssetCfgBase:

    cfg_type_name = raw.get("cfg_type")
    cfg_payload = raw.get("cfg")
    if not isinstance(cfg_type_name, str) or not isinstance(cfg_payload, dict):
        raise ValueError(f"Mutate term {term_name!r} must provide cfg_type and cfg")

    cfg_type_map = {
        "LinkScaleCfg": LinkScaleCfg,
        "TipReplaceCfg": TipReplaceCfg,
        "LimitTweakCfg": LimitTweakCfg,
        "MountPerturbCfg": MountPerturbCfg,
    }
    if cfg_type_name not in cfg_type_map:
        raise ValueError(f"Unsupported mutate cfg_type: {cfg_type_name!r}")

    tuple_fields = {
        "LinkScaleCfg": ("target_joints",),
        "TipReplaceCfg": ("target_fingers",),
        "LimitTweakCfg": ("target_joints",),
        "MountPerturbCfg": (
            "pos_radius",
            "rot_radius",
            "thumb_pos_radius",
            "thumb_rot_radius",
            "mirror_yaw_range",
            "mirror_x_range",
        ),
    }
    return _build_legacy_mutator_cfg(
        term_name,
        cfg_payload,
        cfg_cls=cfg_type_map[cfg_type_name],
        tuple_field_names=tuple_fields[cfg_type_name],
    )


def _build_legacy_mutator_cfg(
    term_name: str,
    raw: dict[str, Any],
    *,
    cfg_cls: type[Any],
    tuple_field_names: tuple[str, ...],
) -> AssetCfgBase:

    payload = deepcopy(raw)
    for field_name in tuple_field_names:
        if field_name in payload and isinstance(payload[field_name], list):
            payload[field_name] = tuple(payload[field_name])

    try:
        return cfg_cls(**payload)
    except TypeError as exc:
        raise TypeError(f"Failed to build mutate term {term_name!r}: {exc}") from exc


def _build_validate_stage_cfg(raw: dict[str, Any], *, stage_cfg_cls: type[Any]) -> Any:

    data = deepcopy(raw)
    data.pop("class_type", None)
    finger_raw = data.get("finger")
    if isinstance(finger_raw, dict):
        finger_data = deepcopy(finger_raw)
        joint_raw = finger_data.get("joint")
        if isinstance(joint_raw, dict):
            finger_data["joint"] = JointValidatorCfg(**joint_raw)
        data["finger"] = FingerValidatorCfg(**finger_data)
    return stage_cfg_cls(**data)


def _build_validate_cfg(raw: dict[str, Any]) -> HandValidatorCfg:

    data = deepcopy(raw)
    pre_made_raw = data.get("pre_made")
    post_mutate_raw = data.get("post_mutate")

    if isinstance(pre_made_raw, dict) or isinstance(post_mutate_raw, dict):
        if isinstance(pre_made_raw, dict):
            data["pre_made"] = _build_validate_stage_cfg(
                pre_made_raw,
                stage_cfg_cls=HandValidatorCfg.PreMadeCfg,
            )
        if isinstance(post_mutate_raw, dict):
            data["post_mutate"] = _build_validate_stage_cfg(
                post_mutate_raw,
                stage_cfg_cls=HandValidatorCfg.PostMutateCfg,
            )
        return HandValidatorCfg(**data)

    legacy_stage_raw = deepcopy(data)
    legacy_stage_raw.pop("class_type", None)
    return HandValidatorCfg(
        pre_made=_build_validate_stage_cfg(
            legacy_stage_raw,
            stage_cfg_cls=HandValidatorCfg.PreMadeCfg,
        ),
        post_mutate=_build_validate_stage_cfg(
            legacy_stage_raw,
            stage_cfg_cls=HandValidatorCfg.PostMutateCfg,
        ),
    )


def _build_export_cfg(raw: dict[str, Any]) -> HandExporterCfg:

    data = deepcopy(raw)
    if "Urdf" in data and isinstance(data["Urdf"], dict):
        data["Urdf"] = UrdfWriterCfg(**data["Urdf"])
    if "Sidecar" in data and isinstance(data["Sidecar"], dict):
        data["Sidecar"] = SidecarCfg(**data["Sidecar"])
    return HandExporterCfg(**data)


def _build_physics_cfg(raw: dict[str, Any]) -> AssetPhysicsCfg:

    data = deepcopy(raw)
    return AssetPhysicsCfg(**data)


def _build_made_cfg(raw: dict[str, Any]) -> Any:

    data = deepcopy(raw)
    builder_type = data.pop("builder_type", "human_like")
    if builder_type == "gripper_like":
        return GripperLikeHandBuilderCfg(**data)
    if builder_type != "human_like":
        raise ValueError(f"Unsupported builder_type: {builder_type!r}")
    return make_human_like_builder_cfg(**data)


def _dump_value(value: Any) -> Any:

    if isinstance(value, HandMutatorCfg):
        return _dump_value(value.to_dict())
    if is_dataclass(value):
        return {
            obj_field.name: _dump_value(getattr(value, obj_field.name))
            for obj_field in fields(value)
            if obj_field.name != "class_type" and not obj_field.name.startswith("_")
        }
    if hasattr(value, "to_dict"):
        return _dump_value(value.to_dict())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        dumped: dict[str, Any] = {}
        for key, item in value.items():
            if key == "class_type" or key.startswith("_"):
                continue
            dumped[key] = _dump_value(item)
        return dumped
    if isinstance(value, tuple):
        return [_dump_value(item) for item in value]
    if isinstance(value, list):
        return [_dump_value(item) for item in value]
    return value


__all__ = ["RecipeLoader"]
