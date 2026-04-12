r"""YAML recipe 解析器：YAML ↔ HandGeneratorCfg 双向转换。

本模块解决的核心问题是：`HandGeneratorCfg` 是一个多层嵌套 dataclass，直接用
Python 构造时配置量大；有了 YAML loader，用户只需编写声明式 recipe 文件，就能
描述完整的生成空间和采样策略。

YAML Recipe 格式约定
--------------------

顶层字段与 `HandGeneratorCfg` 的 Python 字段名一一对应（snake_case）。
大写首字母的 Cfg 字段（`Made`, `Mutate`, `Validate`, `Export`）在 YAML 中保持大写，
以区分“子配置块”和普通标量字段。

典型 recipe 示例::

    name: allegro_variant
    family: allegro
    handedness: right
    sampling_strategy: sample
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

from ..builder.finger_buiders import (
    AllegroFingerBuilderCfg,
    LeapFingerBuilderCfg,
    RegularThumbBuilderCfg,
    get_finger_builder_preset,
)
from ..builder.hand_builders import GripperLikeHandBuilderCfg, HumanLikeHandBuilderCfg
from ..builder.palm_builders import (
    ComPalmBuilderCfg,
    SinglePalmBuilderCfg,
    get_com_palm_preset,
    get_single_palm_box_preset,
)
from ..exporter.hand_exporter import HandExporterCfg
from ..exporter.sidecar import SidecarCfg
from ..exporter.urdf_writer import UrdfWriterCfg
from ..generator.hand_generator import HandGeneratorCfg
from ..generator.mutate import (
    FingerReplaceCfg,
    HandMutatorCfg,
    JointDeleteCfg,
    LimitTweakCfg,
    LinkScaleCfg,
    MountPerturbCfg,
    TipReplaceCfg,
)
from ..validator.finger_rules import FingerValidatorCfg
from ..validator.hand_rules import HandValidatorCfg
from ..validator.joint_rules import JointValidatorCfg


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
        # 这里做兼容桥接，避免旧实验描述在 tooling 层直接报废。
        if "export_dir" in data and "output_dir" not in data:
            data["output_dir"] = data.pop("export_dir")

        if "Made" in data and isinstance(data["Made"], dict):
            data["Made"] = _build_made_cfg(data["Made"])
        if "Mutate" in data and isinstance(data["Mutate"], dict):
            data["Mutate"] = _build_mutate_cfg(data["Mutate"])
        if "Validate" in data and isinstance(data["Validate"], dict):
            data["Validate"] = _build_validate_cfg(data["Validate"])
        if "Export" in data and isinstance(data["Export"], dict):
            data["Export"] = _build_export_cfg(data["Export"])

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
    if "order" in data and isinstance(data["order"], list):
        data["order"] = tuple(data["order"])

    tool_cfg_map = {
        "joint_delete": JointDeleteCfg,
        "link_scale": LinkScaleCfg,
        "tip_replace": TipReplaceCfg,
        "limit_tweak": LimitTweakCfg,
        "mount_perturb": MountPerturbCfg,
        "finger_replace": FingerReplaceCfg,
    }
    tuple_fields = {
        "joint_delete": ("deleted_joints",),
        "link_scale": ("target_joints",),
        "tip_replace": ("target_fingers",),
        "limit_tweak": ("target_joints",),
        "mount_perturb": ("target_fingers",),
        "finger_replace": (),
    }

    for key, cfg_cls in tool_cfg_map.items():
        payload = data.get(key)
        if not isinstance(payload, dict):
            continue
        payload = deepcopy(payload)
        for field_name in tuple_fields[key]:
            if field_name in payload and isinstance(payload[field_name], list):
                payload[field_name] = tuple(payload[field_name])
        data[key] = cfg_cls(**payload)

    return HandMutatorCfg(**data)


def _build_validate_cfg(raw: dict[str, Any]) -> HandValidatorCfg:
    r"""递归实例化 Validate 块。"""

    data = deepcopy(raw)
    finger_raw = data.get("finger")
    if isinstance(finger_raw, dict):
        finger_data = deepcopy(finger_raw)
        joint_raw = finger_data.get("joint")
        if isinstance(joint_raw, dict):
            finger_data["joint"] = JointValidatorCfg(**joint_raw)
        data["finger"] = FingerValidatorCfg(**finger_data)
    return HandValidatorCfg(**data)


def _build_export_cfg(raw: dict[str, Any]) -> HandExporterCfg:
    r"""递归实例化 Export 块。"""

    data = deepcopy(raw)
    if "Urdf" in data and isinstance(data["Urdf"], dict):
        data["Urdf"] = UrdfWriterCfg(**data["Urdf"])
    if "Sidecar" in data and isinstance(data["Sidecar"], dict):
        data["Sidecar"] = SidecarCfg(**data["Sidecar"])
    return HandExporterCfg(**data)


def _build_made_cfg(raw: dict[str, Any]) -> Any:
    r"""递归实例化 Made 块。

    当前首轮默认优先走 human-like hand 路线，因为这是已经实现并测试覆盖的主路径。
    若未来 recipe 需要显式切到 gripper-like，可通过 `builder_type=gripper_like`
    做明确声明，而不是在 loader 里猜测用户意图。
    """

    data = deepcopy(raw)
    if "palm_cfg" in data:
        data["palm_cfg"] = _build_palm_cfg(data["palm_cfg"])
    if "finger_cfg" in data:
        data["finger_cfg"] = _build_finger_slot_cfg(data["finger_cfg"])
    if "thumb_cfg" in data:
        data["thumb_cfg"] = _build_finger_cfg(data["thumb_cfg"])

    builder_type = data.pop("builder_type", "human_like")
    if builder_type == "gripper_like":
        return GripperLikeHandBuilderCfg(**data)
    if builder_type != "human_like":
        raise ValueError(f"Unsupported builder_type: {builder_type!r}")
    return HumanLikeHandBuilderCfg(**data)


def _build_palm_cfg(raw: Any) -> Any:
    r"""构造 palm builder cfg。

    palm 当前支持两类声明风格：

    1. preset 字符串：如 `com_allegro`、`single_box_leap`
    2. 直接 dict：由字段形状推断是 `ComPalmBuilderCfg` 还是 `SinglePalmBuilderCfg`
    """

    if isinstance(raw, (ComPalmBuilderCfg, SinglePalmBuilderCfg)):
        return raw
    if isinstance(raw, str):
        if raw.startswith("com_"):
            return get_com_palm_preset(raw.removeprefix("com_"))
        if raw.startswith("single_box_"):
            return get_single_palm_box_preset(raw.removeprefix("single_box_"))
        raise ValueError(f"Unsupported palm preset string: {raw!r}")
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported palm cfg payload: {raw!r}")
    if "preset" in raw:
        return ComPalmBuilderCfg(**raw)
    return SinglePalmBuilderCfg(**raw)


def _build_finger_slot_cfg(raw: Any) -> Any:
    r"""构造非拇指 finger 槽位配置。

    `HumanLikeHandBuilderCfg.finger_cfg` 有两层语义：

    1. 一份共享 finger cfg：所有非拇指共用
    2. 一个按 slot 分配的 dict：如 `index/middle/ring/little` 各自独立

    因此这里必须先判断传入 dict 是“单个 cfg 描述”还是“按 finger 名分配的映射”。
    """

    if isinstance(raw, dict):
        slot_names = {"index", "middle", "ring", "little"}
        # 这里不再用“是否长得像 cfg 字段集合”来猜，因为 dump 后的 cfg 会携带
        # `_mesh_offsets_6d` 这类内部规范化字段；若继续做负向猜测，round-trip
        # 时就会把单个 finger cfg 误判成“按手指槽位分配的映射”。
        if raw and set(raw).issubset(slot_names):
            return {name: _build_finger_cfg(cfg) for name, cfg in raw.items()}
    return _build_finger_cfg(raw)


def _build_finger_cfg(raw: Any) -> Any:
    r"""构造 regular finger cfg。

    当前首轮优先支持两种输入：

    1. preset 字符串：最稳，也是最符合现在 pre-made slice 的入口
    2. 直接 cfg dict：通过字段特征推断 thumb / leap / allegro
    """

    if isinstance(raw, (AllegroFingerBuilderCfg, LeapFingerBuilderCfg, RegularThumbBuilderCfg)):
        return raw
    if isinstance(raw, str):
        return get_finger_builder_preset(raw)
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported finger cfg payload: {raw!r}")

    data = deepcopy(raw)
    preset_name = data.pop("preset_name", data.pop("preset", None))
    if isinstance(preset_name, str):
        return get_finger_builder_preset(preset_name).replace(**data)

    thumb_keys = {"lengths", "cmc1_width", "cmc1_height", "cmc1_offset", "non_cmc1_offset"}
    if thumb_keys & set(data):
        return RegularThumbBuilderCfg(**data)
    if "fixed_part" in data:
        return LeapFingerBuilderCfg(**data)
    return AllegroFingerBuilderCfg(**data)


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
