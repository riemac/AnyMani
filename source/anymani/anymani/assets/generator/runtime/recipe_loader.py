r"""YAML recipe 解析器：YAML ↔ HandGeneratorCfg 双向转换。

本模块解决的核心问题是：`HandGeneratorCfg` 是一个多层嵌套 dataclass，直接用
Python 构造时配置量大；有了 YAML loader，用户只需编写声明式 recipe 文件，就能
描述完整的生成空间与导出约定。

YAML Recipe 格式约定
--------------------

顶层字段与 `HandGeneratorCfg` 的 Python 字段名一一对应（snake_case）。
大写首字母的 Cfg 字段（`Made`, `Mutate`, `Validate`, `Export`）在 YAML 中保持大写，
以区分“子配置块”和普通标量字段。

典型 recipe 示例::

    name: allegro_variant
    family: allegro
    handedness: right
    n_samples: 8
    mode: full
    artifact_level: bundle
    output_dir: assets/generated/
    Made:
      palm_cfg: com_allegro
      finger_cfg: allegro_non_thumb_v1
      thumb_cfg: allegro_thumb_v1

设计说明
--------

### class_type 字段处理

所有 Cfg 中的 `class_type` 字段（`type[SomeClass]`）在 YAML 中**不出现**，
由 loader 在各 Cfg 的 `__post_init__` 中自动注入默认值。

### 元组字段

Python dataclass 里的 `tuple[str, ...]` 字段（如 `order`、`target_joints`）
在 YAML 中写成 YAML list，loader 自动转为 Python `tuple`。

### 当前首轮支持范围

当前优先服务已经打通的 pre-made 主链，因此 loader 的强支持路径是：

1. `HandGeneratorCfg`
2. `HumanLikeHandBuilderCfg`
3. `ComPalmBuilderCfg` / `SinglePalmBuilderCfg`
4. regular finger / thumb preset
5. `Validate` / `Export` / `Mutate` 的声明式反序列化

也就是说，这里先把“真实能跑通的 recipe 路线”做扎实，而不是提前把尚未落地的
post-mutate 运行时复杂度塞进 loader。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from ...asset_base import AssetCfgBase
from ...asset_physics import AssetPhysicsCfg
from ...builder.hand_builders import GripperLikeHandBuilderCfg
from ...exporter.hand_exporter import HandExporterCfg
from ...exporter.sidecar import SidecarCfg
from ...exporter.urdf_writer import UrdfWriterCfg
from ..hand_generator import HandGeneratorCfg
from ..mutate import (
    HandMutatorCfg,
    LimitTweakCfg,
    LinkScaleCfg,
    MountPerturbCfg,
    TipReplaceCfg,
)
from ...presets import make_human_like_builder_cfg
from ...validator.finger_rules import FingerValidatorCfg
from ...validator.hand_rules import HandValidatorCfg
from ...validator.joint_rules import JointValidatorCfg


# ============================================================================
#  Recipe Loader
# ============================================================================


class RecipeLoader:
    r"""YAML recipe 解析器。

    提供从 YAML 文件（或 dict）加载 `HandGeneratorCfg` 的静态方法，
    以及把 `HandGeneratorCfg` 序列化回 YAML 兼容 dict 的反向接口。
    """

    @staticmethod
    def load(path: str | Path) -> HandGeneratorCfg:
        r"""从 YAML 文件加载 `HandGeneratorCfg`。

        Args:
            path (str | Path): YAML recipe 文件路径。

        Returns:
            HandGeneratorCfg: 解析后的生成器配置对象。

        Raises:
            FileNotFoundError: 文件不存在时。
            ValueError: YAML 根节点不是 mapping 时。
        """

        recipe_path = Path(path)
        if not recipe_path.exists():
            raise FileNotFoundError(recipe_path)

        raw = yaml.safe_load(recipe_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"recipe root must be a mapping, got {type(raw).__name__}")
        return RecipeLoader.load_dict(raw)

    @staticmethod
    def load_dict(raw: dict[str, Any]) -> HandGeneratorCfg:
        r"""从已解析的 Python dict 加载 `HandGeneratorCfg`。

        比 ``load()`` 更轻量，适合在 Python 测试代码中直接构造 recipe。

        Args:
            raw (dict[str, Any]): 与 YAML recipe 格式相同的 Python dict。

        Returns:
            HandGeneratorCfg: 解析后的生成器配置对象。
        """

        data = deepcopy(raw)

        # 历史 recipe 曾使用 `export_dir`；当前统一收口为 `output_dir`。
        # 这里做兼容桥接，避免旧实验描述在 generator recipe 边界直接报废。
        if "export_dir" in data and "output_dir" not in data:
            data["output_dir"] = data.pop("export_dir")

        # 历史 recipe 里 `sampling_strategy` 曾显式区分 sample / enumerate。
        # 当前语义已经固定：
        #
        # - pre-made topology = 离散枚举
        # - post-mutate = 对每个 pre-made topology 做 Monte Carlo 采样
        #
        # 因而旧字段在 loader 层直接吞掉，避免老 recipe 因多余键崩溃。
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
        r"""把 `HandGeneratorCfg` 序列化为 YAML 兼容的 Python dict。

        返回的 dict 可直接用 ``yaml.safe_dump()`` 写入文件，也可用于日志记录
        或 experiment tracking。

        Args:
            cfg (HandGeneratorCfg): 待序列化的生成器配置。

        Returns:
            dict[str, Any]: 与 YAML recipe 格式对应的 Python dict。
        """

        dumped = _dump_value(cfg)
        if not isinstance(dumped, dict):
            raise TypeError("HandGeneratorCfg dump result must be a mapping")
        return dumped

    @staticmethod
    def save(cfg: HandGeneratorCfg, path: str | Path) -> None:
        r"""将 `HandGeneratorCfg` 序列化并写入 YAML 文件。

        Args:
            cfg (HandGeneratorCfg): 待保存的生成器配置。
            path (str | Path): 目标 YAML 文件路径；父目录不存在时自动创建。
        """

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            yaml.safe_dump(RecipeLoader.dump(cfg), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )


# ============================================================================
#  内部辅助
# ============================================================================


def _build_mutate_cfg(raw: dict[str, Any]) -> HandMutatorCfg:
    r"""把 Mutate YAML 块递归实例化为 `HandMutatorCfg`。

    虽然当前首轮运行时仍不真正执行 post-mutate，但 recipe 层先把结构读通，
    这样后续真正落地 mutate 时，不需要再回头改 YAML 约定。
    """

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
    r"""解析新式 `term_name: {cfg_type, cfg}` 结构。"""

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
    r"""兼容 `Mutate.link_scale: {...}` 结构，并保持 cfg 字段原样。"""

    payload = deepcopy(raw)
    for field_name in tuple_field_names:
        if field_name in payload and isinstance(payload[field_name], list):
            payload[field_name] = tuple(payload[field_name])

    try:
        return cfg_cls(**payload)
    except TypeError as exc:
        raise TypeError(f"Failed to build mutate term {term_name!r}: {exc}") from exc


def _build_validate_stage_cfg(raw: dict[str, Any], *, stage_cfg_cls: type[Any]) -> Any:
    r"""递归实例化单个 validator 阶段块。"""

    data = deepcopy(raw)
    data.pop("class_type", None)  # stage cfg 只是纯规则容器，不需要 runtime class 入口
    finger_raw = data.get("finger")
    if isinstance(finger_raw, dict):
        finger_data = deepcopy(finger_raw)
        joint_raw = finger_data.get("joint")
        if isinstance(joint_raw, dict):
            finger_data["joint"] = JointValidatorCfg(**joint_raw)
        data["finger"] = FingerValidatorCfg(**finger_data)
    return stage_cfg_cls(**data)


def _build_validate_cfg(raw: dict[str, Any]) -> HandValidatorCfg:
    r"""递归实例化 Validate 块。

    当前同时兼容两种 YAML 形状：

    1. 新的显式阶段形状：
       `Validate: {pre_made: {...}, post_mutate: {...}}`
    2. 旧的平面形状：
       `Validate: {strict: false, finger: {...}, ...}`

    对旧形状，loader 会把同一组规则同时复制到 `pre_made` 与 `post_mutate`，
    这样历史 recipe 不会因为这轮 staged-validator 重构直接失效。
    """

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
    r"""递归实例化 Export 块。"""

    data = deepcopy(raw)
    if "Urdf" in data and isinstance(data["Urdf"], dict):
        data["Urdf"] = UrdfWriterCfg(**data["Urdf"])
    if "Sidecar" in data and isinstance(data["Sidecar"], dict):
        data["Sidecar"] = SidecarCfg(**data["Sidecar"])
    return HandExporterCfg(**data)


def _build_physics_cfg(raw: dict[str, Any]) -> AssetPhysicsCfg:
    r"""递归实例化 Physics 块。"""

    data = deepcopy(raw)
    return AssetPhysicsCfg(**data)


def _build_made_cfg(raw: dict[str, Any]) -> Any:
    r"""递归实例化 Made 块。

    当前首轮默认优先走 human-like hand 路线，因为这是已经实现并测试覆盖的主路径。
    若未来 recipe 需要显式切到 gripper-like，可通过 `builder_type=gripper_like`
    做明确声明，而不是在 loader 里猜测用户意图。
    """

    data = deepcopy(raw)
    builder_type = data.pop("builder_type", "human_like")
    if builder_type == "gripper_like":
        return GripperLikeHandBuilderCfg(**data)
    if builder_type != "human_like":
        raise ValueError(f"Unsupported builder_type: {builder_type!r}")
    return make_human_like_builder_cfg(**data)


def _dump_value(value: Any) -> Any:
    r"""递归把 cfg / dataclass 值规约为 YAML 兼容对象。

    序列化时显式移除 `class_type`，并把 tuple / Path 压成 YAML 友好格式。
    """

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
