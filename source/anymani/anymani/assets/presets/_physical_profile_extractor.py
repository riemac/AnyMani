r"""官方 URDF joint physical profile 的离线提取工具。

该模块不是 pre-made 运行时路径的一部分。它的用途是当官方 URDF 或映射假设变化时，
帮助研究者从指定 URDF 中重新提取 `limit / effort / velocity / friction`，再人工核对
并同步到 `physical_presets.py`。

# NOTE:
这里故意不提供“自动写回 preset 文件”的能力。原因是 physical profile 是科研数值锚点，
每次更新都应经过人工 diff 和可视化巡检，而不是让脚本静默改动 committed preset。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import argparse
import xml.etree.ElementTree as ET


@dataclass
class ExtractedJointPhysics:
    r"""从官方 URDF 中提取出的单个 source joint 物理属性。"""

    name: str
    r"""官方 URDF joint 名。"""

    lower: float
    r"""`<limit lower>`，广义坐标下界 $q_{\min}$。"""

    upper: float
    r"""`<limit upper>`，广义坐标上界 $q_{\max}$。"""

    effort: float | None
    r"""`<limit effort>`，驱动力矩 / 力上界。"""

    velocity: float | None
    r"""`<limit velocity>`，速度上界。"""

    friction: float | None
    r"""`<joint_properties friction>`；官方未提供时为 `None`。"""


_LEAP_NON_THUMB_MAPPING: dict[str, tuple[str, ...]] = {
    "mcp1": ("1", "5", "9"),
    "mcp2": ("0", "4", "8"),
    "pip": ("2", "6", "10"),
    "dip": ("3", "7", "11"),
}
r"""LEAP 非拇指 child-link suffix 到官方 source joints 的默认映射。

该映射按 parent-child 串联顺序，而不是按 joint 名数字顺序。
官方 LEAP 的 `joint 1/5/9` 才是 palm 侧第一个 MCP slot；
`joint 0/4/8` 则是从 `mcp_joint*` 接到 `pip*` 的第二个 MCP slot。
"""


_LEAP_THUMB_MAPPING: dict[str, tuple[str, ...]] = {
    "cmc1": ("12",),
    "cmc2": ("13",),
    "mcp": ("14",),
    "dip": ("15",),
}
r"""LEAP 拇指 child-link suffix 到官方 source joints 的默认映射。"""


_ALLEGRO_NON_THUMB_MAPPING: dict[str, tuple[str, ...]] = {
    "mcp1": ("joint_0.0", "joint_4.0", "joint_8.0"),
    "mcp2": ("joint_1.0", "joint_5.0", "joint_9.0"),
    "pip": ("joint_2.0", "joint_6.0", "joint_10.0"),
    "dip": ("joint_3.0", "joint_7.0", "joint_11.0"),
}
r"""Allegro 非拇指 child-link suffix 到官方 source joints 的默认映射。"""


_ALLEGRO_THUMB_MAPPING: dict[str, tuple[str, ...]] = {
    "cmc1": ("joint_12.0",),
    "cmc2": ("joint_13.0",),
    "mcp": ("joint_14.0",),
    "dip": ("joint_15.0",),
}
r"""Allegro 拇指 child-link suffix 到官方 source joints 的默认映射。"""


_MAPPING_PRESETS: dict[tuple[str, str], dict[str, tuple[str, ...]]] = {
    ("leap", "non_thumb"): _LEAP_NON_THUMB_MAPPING,
    ("leap", "thumb"): _LEAP_THUMB_MAPPING,
    ("allegro", "non_thumb"): _ALLEGRO_NON_THUMB_MAPPING,
    ("allegro", "thumb"): _ALLEGRO_THUMB_MAPPING,
}
r"""命令行参数到默认 source-joint 映射的索引。"""


def read_joint_physics(urdf_path: Path) -> dict[str, ExtractedJointPhysics]:
    r"""读取 URDF 中所有 revolute joint 的 joint-level 物理属性。

    Args:
        urdf_path (Path): 官方 URDF 路径。

    Returns:
        dict[str, ExtractedJointPhysics]: 以 source joint 名为 key 的属性表。
    """

    root = ET.parse(urdf_path).getroot()  # 该工具离线运行，因此可直接解析 XML
    records: dict[str, ExtractedJointPhysics] = {}
    for joint in root.findall("joint"):
        if joint.attrib.get("type") == "fixed":
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        joint_properties = joint.find("joint_properties")
        friction = None if joint_properties is None else float(joint_properties.attrib["friction"])
        records[joint.attrib["name"]] = ExtractedJointPhysics(
            name=joint.attrib["name"],
            lower=float(limit.attrib["lower"]),
            upper=float(limit.attrib["upper"]),
            effort=float(limit.attrib["effort"]) if "effort" in limit.attrib else None,
            velocity=float(limit.attrib["velocity"]) if "velocity" in limit.attrib else None,
            friction=friction,
        )
    return records


def _lookup_record(records: Mapping[str, ExtractedJointPhysics], source_joint: str) -> ExtractedJointPhysics:
    r"""按 source joint 名查记录，并兼容 TRO-Grasp LEAP 的 `a_0` 命名。"""

    if source_joint in records:
        return records[source_joint]
    alias = f"a_{source_joint}"
    if alias in records:
        return records[alias]
    raise KeyError(f"source joint {source_joint!r} not found in URDF")


def extract_profile(
    urdf_path: Path,
    mapping: Mapping[str, tuple[str, ...]],
) -> dict[str, list[ExtractedJointPhysics]]:
    r"""按 canonical child-link suffix 聚合官方 source joint 物理属性。

    Args:
        urdf_path (Path): 官方 URDF 路径。
        mapping (Mapping[str, tuple[str, ...]]): child suffix 到 source joints 的映射。

    Returns:
        dict[str, list[ExtractedJointPhysics]]: 每个 canonical slot 的 source 记录列表。
    """

    records = read_joint_physics(urdf_path)
    return {
        child_suffix: [_lookup_record(records, source_joint) for source_joint in source_joints]
        for child_suffix, source_joints in mapping.items()
    }


def _format_profile(profile: Mapping[str, list[ExtractedJointPhysics]]) -> str:
    r"""把提取结果格式化为便于人工复制到 `physical_presets.py` 的文本。"""

    lines: list[str] = []
    for child_suffix, records in profile.items():
        first = records[0]
        source_names = tuple(record.name for record in records)
        lines.append(
            f"{child_suffix}: source_joints={source_names}, "
            f"limit=({first.lower}, {first.upper}, effort={first.effort}, velocity={first.velocity}), "
            f"friction={first.friction}"
        )
    return "\n".join(lines)


def main() -> None:
    r"""命令行入口：从指定 URDF 打印 profile 草稿。"""

    parser = argparse.ArgumentParser(description="Extract AnyMani official joint physical profile draft.")
    parser.add_argument("urdf", type=Path, help="官方 URDF 路径")
    parser.add_argument("--family", choices=("leap", "allegro"), required=True, help="官方手型 family")
    parser.add_argument("--kind", choices=("non_thumb", "thumb"), required=True, help="finger profile 类型")
    args = parser.parse_args()

    profile = extract_profile(args.urdf, _MAPPING_PRESETS[(args.family, args.kind)])
    print(_format_profile(profile))


if __name__ == "__main__":
    main()


__all__ = ["ExtractedJointPhysics", "read_joint_physics", "extract_profile"]
