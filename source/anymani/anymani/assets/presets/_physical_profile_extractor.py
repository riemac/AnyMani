"Reads official joint limits and actuator values for manual review. It never writes preset files automatically because these values are research anchors."

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import argparse
import xml.etree.ElementTree as ET


@dataclass
class ExtractedJointPhysics:
    "Joint limits, effort, velocity, and friction read from one official source joint."

    name: str
    "Stable semantic identifier preserved in the output metadata."

    lower: float
    "Lower legal or sampling bound in the units of the associated field."

    upper: float
    "Upper legal or sampling bound in the units of the associated field."

    effort: float | None
    "Joint effort limit in newton meters."

    velocity: float | None
    "Joint velocity limit in radians per second."

    friction: float | None
    "Joint friction coefficient copied from the reviewed family profile."


_LEAP_NON_THUMB_MAPPING: dict[str, tuple[str, ...]] = {
    "mcp1": ("1", "5", "9"),
    "mcp2": ("0", "4", "8"),
    "pip": ("2", "6", "10"),
    "dip": ("3", "7", "11"),
}
"Mapping from official LEAP non-thumb source joints to anatomical slots."


_LEAP_THUMB_MAPPING: dict[str, tuple[str, ...]] = {
    "cmc1": ("12",),
    "cmc2": ("13",),
    "mcp": ("14",),
    "dip": ("15",),
}
"Mapping from official LEAP thumb source joints to anatomical slots."


_ALLEGRO_NON_THUMB_MAPPING: dict[str, tuple[str, ...]] = {
    "mcp1": ("joint_0.0", "joint_4.0", "joint_8.0"),
    "mcp2": ("joint_1.0", "joint_5.0", "joint_9.0"),
    "pip": ("joint_2.0", "joint_6.0", "joint_10.0"),
    "dip": ("joint_3.0", "joint_7.0", "joint_11.0"),
}
"Mapping from Allegro source joints to canonical non-thumb slots."


_ALLEGRO_THUMB_MAPPING: dict[str, tuple[str, ...]] = {
    "cmc1": ("joint_12.0",),
    "cmc2": ("joint_13.0",),
    "mcp": ("joint_14.0",),
    "dip": ("joint_15.0",),
}
"Mapping from Allegro source joints to canonical thumb slots."


_MAPPING_PRESETS: dict[tuple[str, str], dict[str, tuple[str, ...]]] = {
    ("leap", "non_thumb"): _LEAP_NON_THUMB_MAPPING,
    ("leap", "thumb"): _LEAP_THUMB_MAPPING,
    ("allegro", "non_thumb"): _ALLEGRO_NON_THUMB_MAPPING,
    ("allegro", "thumb"): _ALLEGRO_THUMB_MAPPING,
}
"Registry of exact official-to-canonical joint mappings."


def read_joint_physics(urdf_path: Path) -> dict[str, ExtractedJointPhysics]:
    'Loads and validates joint physics.'

    root = ET.parse(urdf_path).getroot()
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
    "Extracts official joint limits and actuator properties into reviewed preset records."

    records = read_joint_physics(urdf_path)
    return {
        child_suffix: [_lookup_record(records, source_joint) for source_joint in source_joints]
        for child_suffix, source_joints in mapping.items()
    }


def _format_profile(profile: Mapping[str, list[ExtractedJointPhysics]]) -> str:

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
    "Dispatches the selected asset command and returns its process status."

    parser = argparse.ArgumentParser(description="Extract AnyMani official joint physical profile draft.")
    parser.add_argument("urdf", type=Path, help='Path to the official URDF.')
    parser.add_argument("--family", choices=("leap", "allegro"), required=True, help='Official hand family.')
    parser.add_argument("--kind", choices=("non_thumb", "thumb"), required=True, help='Finger profile kind.')
    args = parser.parse_args()

    profile = extract_profile(args.urdf, _MAPPING_PRESETS[(args.family, args.kind)])
    print(_format_profile(profile))


if __name__ == "__main__":
    main()


__all__ = ["ExtractedJointPhysics", "read_joint_physics", "extract_profile"]
