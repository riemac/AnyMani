"Builds palm geometry and finger mounts in the declared palm frame."

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from ..asset_base import PalmCfg
from ..asset_builders import PalmBuilder, PalmBuilderCfg
from ..asset_schema_core import CollisionGeometryCfg, InertialCfg, PoseCfg, VisualGeometryCfg
from ..presets.mount_presets import get_mount_preset
from ..presets.palm_presets import get_com_palm_preset_data


_DEFAULT_PALM_DENSITY = 700.0
"Default palm density used by physics closure."




def _box_inertia(width: float, length: float, height: float, mass: float) -> dict[str, float]:
    return {
        "ixx": mass * (length * length + height * height) / 12.0,
        "iyy": mass * (width * width + height * height) / 12.0,
        "izz": mass * (width * width + length * length) / 12.0,
    }


def _cylinder_inertia(radius: float, height: float, mass: float) -> dict[str, float]:
    return {
        "ixx": mass * (3.0 * radius * radius + height * height) / 12.0,
        "iyy": mass * (3.0 * radius * radius + height * height) / 12.0,
        "izz": mass * radius * radius / 2.0,
    }


def _sphere_inertia(radius: float, mass: float) -> dict[str, float]:
    moment = 2.0 * mass * radius * radius / 5.0
    return {"ixx": moment, "iyy": moment, "izz": moment}


def _estimate_mass(volume: float) -> float:
    return max(volume * _DEFAULT_PALM_DENSITY, 1e-5)


@dataclass
class SinglePalmBuilderCfg(PalmBuilderCfg):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    shape: Literal["box", "cylinder", "sphere", "ellipse"] = "box"
    "Primitive dimensions in meters or mesh reference, as declared by the geometry type."

    length: float | None = None
    "Primary link span along the declared anatomical or joint-local axis, in meters."

    width: float | None = None
    "Link cross-section dimension in meters."

    height: float | None = None
    "Link cross-section dimension in meters."

    radius: float | None = None
    "Primitive radius in meters."

    a: float | None = None
    "Lower bound of a scalar or vector sampling interval."

    b: float | None = None
    "Upper bound of a scalar or vector sampling interval."

    def __post_init__(self):
        super().__post_init__()
        if self.shape == "box":
            for field_name in ("width", "length", "height"):
                value = getattr(self, field_name)
                if value is None or float(value) <= 0.0:
                    raise ValueError(f"{field_name} must be positive for box palms")
        elif self.shape in {"cylinder", "sphere"}:
            if self.radius is None or float(self.radius) <= 0.0:
                raise ValueError(f"radius must be positive for {self.shape} palms")
            if self.height is None or float(self.height) <= 0.0:
                raise ValueError(f"height must be positive for {self.shape} palms")
        elif self.shape == "ellipse":
            if self.a is None or float(self.a) <= 0.0 or self.b is None or float(self.b) <= 0.0:
                raise ValueError("ellipse palms require positive a and b")
            if self.height is None or float(self.height) <= 0.0:
                raise ValueError("ellipse palms require positive height")
        else:
            raise ValueError(f"unsupported palm shape: {self.shape}")
        self.class_type = SinglePalmBuilder


@dataclass
class ComPalmBuilderCfg(PalmBuilderCfg):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    preset: Literal["leap", "allegro"] = "allegro"
    "Named builder or geometry preset used by this component."

    def __post_init__(self):
        super().__post_init__()
        self.class_type = ComPalmBuilder


@dataclass
class CustomPalmBuilderCfg(PalmBuilderCfg):
    "Declares unit-aware inputs and validation policy for the asset pipeline."




class SinglePalmBuilder(PalmBuilder):
    "Builds a configured hand component from typed geometry and local frames."

    cfg: SinglePalmBuilderCfg

    def __init__(self, cfg: SinglePalmBuilderCfg):
        super().__init__(cfg)
        self.cfg = cfg

    def build(self) -> PalmCfg:
        "Builds the configured geometry component from typed dimensions and local frames."

        # ================================================================

        # ================================================================




        #



        #



        #



        #



        # ================================================================
        if self.cfg.shape == "box":

            width = float(self.cfg.width)
            length = float(self.cfg.length)
            height = float(self.cfg.height)
            origin = PoseCfg(pos=(0.0, length / 2.0, 0.0))
            mass = _estimate_mass(width * length * height)
            inertia = _box_inertia(width, length, height, mass)
            geometry = {"type": "box", "size": (width, length, height)}
        elif self.cfg.shape == "cylinder":

            radius = float(self.cfg.radius)
            height = float(self.cfg.height)
            origin = PoseCfg()
            mass = _estimate_mass(math.pi * radius * radius * height)
            inertia = _cylinder_inertia(radius, height, mass)
            geometry = {"type": "cylinder", "radius": radius, "length": height}
        elif self.cfg.shape == "sphere":

            radius = float(self.cfg.radius)
            origin = PoseCfg(pos=(0.0, radius, 0.0))
            mass = _estimate_mass(4.0 * math.pi * radius**3 / 3.0)
            inertia = _sphere_inertia(radius, mass)
            geometry = {"type": "sphere", "radius": radius}
        else:

            a = float(self.cfg.a)
            b = float(self.cfg.b)
            c = float(self.cfg.height) / 2.0
            radius = max(a, b, c)
            origin = PoseCfg(pos=(0.0, b, 0.0))
            mass = _estimate_mass(4.0 * math.pi * a * b * c / 3.0)
            inertia = {
                "ixx": mass * (b * b + c * c) / 5.0,
                "iyy": mass * (a * a + c * c) / 5.0,
                "izz": mass * (a * a + b * b) / 5.0,
            }
            geometry = {"type": "sphere", "radius": radius}

        collision = CollisionGeometryCfg(name="palm_collision", geometry=geometry, origin=origin)
        visual = VisualGeometryCfg(name="palm_visual", geometry=geometry, origin=origin)
        metadata = {"shape": self.cfg.shape}
        if self.cfg.shape == "ellipse":
            metadata["ellipse_axes"] = {
                "a": float(self.cfg.a),
                "b": float(self.cfg.b),
                "c": float(self.cfg.height) / 2.0,
            }
            metadata["approximation"] = "sphere_envelope"
        return PalmCfg(
            name="palm",
            inertial=InertialCfg(mass=mass, origin=origin, inertia=inertia),
            collisions=[collision],
            visuals=[visual],
            metadata=metadata,
        )


class ComPalmBuilder(PalmBuilder):
    "Builds a configured hand component from typed geometry and local frames."

    cfg: ComPalmBuilderCfg

    def __init__(self, cfg: ComPalmBuilderCfg):
        super().__init__(cfg)
        self.cfg = cfg

    def build(self) -> PalmCfg:
        "Builds the configured geometry component from typed dimensions and local frames."



        preset = get_com_palm_preset_data(self.cfg.preset)
        collisions = [
            CollisionGeometryCfg(
                name=f"{self.cfg.preset}_col_{index}",
                geometry={"type": "box", "size": entry["size"]},
                origin=PoseCfg(pos=entry["origin"], rpy=entry.get("rpy", (0.0, 0.0, 0.0))),
            )
            for index, entry in enumerate(preset["collisions"])
        ]
        visuals = [
            VisualGeometryCfg(
                name=f"{self.cfg.preset}_vis_{index}",
                geometry={"type": "box", "size": entry["size"]},
                origin=PoseCfg(pos=entry["origin"], rpy=entry.get("rpy", (0.0, 0.0, 0.0))),
            )
            for index, entry in enumerate(preset["collisions"])
        ]
        mount_preset_name = str(preset["mount_preset"])
        mounts = get_mount_preset(mount_preset_name)
        metadata = {
            "preset": self.cfg.preset,
            "mount_preset": mount_preset_name,
            "finger_mounts": mounts,
        }
        return PalmCfg(
            name="palm",
            inertial=InertialCfg(**preset["inertial"]),
            collisions=collisions,
            visuals=visuals,
            metadata=metadata,
        )


__all__ = [
    "SinglePalmBuilderCfg",
    "ComPalmBuilderCfg",
    "CustomPalmBuilderCfg",
    "SinglePalmBuilder",
    "ComPalmBuilder",
]
