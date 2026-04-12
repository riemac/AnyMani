r"""Palm builders for the pre-made hand asset pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from ..asset_base import PalmCfg
from ..asset_builders import PalmBuilder, PalmBuilderCfg
from ..asset_schema_core import CollisionGeometryCfg, InertialCfg, PoseCfg, VisualGeometryCfg


_DEFAULT_PALM_DENSITY = 700.0
"""A slightly denser default than finger links to keep inertial values positive."""


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
    r"""Single primitive palm configuration.

    The palm frame follows the drawing in ``assets/doc/Single-Palm.jpg``:
    the origin lies at the bottom center, ``+y`` points toward the fingers,
    ``+x`` spans palm width, and ``+z`` spans thickness.
    """

    shape: Literal["box", "cylinder", "sphere", "ellipse"] = "box"
    """Primitive palm family."""

    length: float | None = None
    """Palm length along ``+y`` for box / ellipse palms."""

    width: float | None = None
    """Palm width along ``+x`` for box palms."""

    height: float | None = None
    """Palm thickness along ``+z`` for every supported shape."""

    radius: float | None = None
    """Palm radius for cylinder / sphere palms."""

    a: float | None = None
    """Ellipse semi-axis along ``+x``."""

    b: float | None = None
    """Ellipse semi-axis along ``+y``."""

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
    r"""Preset composite palm configuration."""

    preset: Literal["leap", "allegro"] = "allegro"
    """Composite palm preset family."""

    def __post_init__(self):
        super().__post_init__()
        self.class_type = ComPalmBuilder


@dataclass
class CustomPalmBuilderCfg(PalmBuilderCfg):
    r"""Placeholder config for future mesh-authored palm presets."""


class SinglePalmBuilder(PalmBuilder):
    r"""Builder for palms made from one primitive."""

    cfg: SinglePalmBuilderCfg

    def __init__(self, cfg: SinglePalmBuilderCfg):
        super().__init__(cfg)
        self.cfg = cfg

    def build(self) -> PalmCfg:
        r"""Build one primitive palm.

        # Question:
        The current canonical schema does not support a scaled primitive directly,
        so ``ellipse`` is exported as a sphere-envelope approximation while keeping
        the ellipsoid inertia formula. This is sufficient for the first vertical
        slice but should be revisited if ellipsoid palms become a training target.
        """

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
            metadata["ellipse_axes"] = {"a": float(self.cfg.a), "b": float(self.cfg.b), "c": float(self.cfg.height) / 2.0}
            metadata["approximation"] = "sphere_envelope"
        return PalmCfg(
            name="palm",
            inertial=InertialCfg(mass=mass, origin=origin, inertia=inertia),
            collisions=[collision],
            visuals=[visual],
            metadata=metadata,
        )


_COM_PALM_PRESETS: dict[str, dict[str, object]] = {
    "allegro": {
        "collisions": [
            {"size": (0.0414, 0.1120, 0.0448), "origin": (-0.0090, 0.0000, -0.0230)},
            {"size": (0.0414, 0.0538, 0.0428), "origin": (-0.0090, -0.0253, -0.0667)},
            {"size": (0.0414, 0.0720, 0.0130), "origin": (-0.0093, -0.00557, -0.08874)},
        ],
        "inertial": {
            "mass": 0.4154,
            "origin": (0.0, 0.0, 0.0),
            "inertia": {"ixx": 1.0e-4, "iyy": 1.0e-4, "izz": 1.0e-4},
        },
        "finger_mounts": {
            "index": {"pos": (0.0, 0.0435, -0.001542), "rpy": (-0.0873, 0.0, 0.0)},
            "middle": {"pos": (0.0, 0.0, 0.0007), "rpy": (0.0, 0.0, 0.0)},
            "ring": {"pos": (0.0, -0.0435, -0.001542), "rpy": (0.0873, 0.0, 0.0)},
            "thumb": {"pos": (-0.0182, 0.019333, -0.045987), "rpy": (0.0, -1.6581, -1.5708)},
        },
    },
    "leap": {
        "collisions": [
            {"size": (0.022, 0.026, 0.034), "origin": (-0.009, 0.008, -0.011)},
            {"size": (0.022, 0.026, 0.034), "origin": (-0.009, -0.037, -0.011)},
            {"size": (0.022, 0.026, 0.034), "origin": (-0.009, -0.082, -0.011)},
            {"size": (0.058, 0.020, 0.046), "origin": (-0.066, -0.078, -0.0115), "rpy": (0.0, 0.0, -0.2967)},
            {"size": (0.020, 0.120, 0.030), "origin": (-0.030, -0.035, -0.003)},
            {"size": (0.010, 0.120, 0.020), "origin": (-0.032, -0.035, -0.024), "rpy": (0.0, 0.785, 0.0)},
            {"size": (0.024, 0.116, 0.046), "origin": (-0.048, -0.033, -0.0115)},
            {"size": (0.044, 0.052, 0.046), "origin": (-0.078, -0.053, -0.0115)},
            {"size": (0.004, 0.036, 0.034), "origin": (-0.098, -0.009, -0.006)},
            {"size": (0.044, 0.056, 0.004), "origin": (-0.078, -0.003, 0.010)},
        ],
        "inertial": {
            "mass": 0.237,
            "origin": (0.0, 0.0, 0.0),
            "inertia": {
                "ixx": 3.54094e-4,
                "ixy": -1.193e-6,
                "ixz": -2.445e-6,
                "iyy": 2.60915e-4,
                "iyz": -2.905e-6,
                "izz": 5.29257e-4,
            },
        },
        "finger_mounts": {
            "index": {"pos": (-0.0070, 0.0230, -0.0187), "rpy": (1.5708, 1.5708, 0.0)},
            "middle": {"pos": (-0.0071, -0.0224, -0.0187), "rpy": (1.5708, 1.5708, 0.0)},
            "ring": {"pos": (-0.00709, -0.0678, -0.0187), "rpy": (1.5708, 1.5708, 0.0)},
            "thumb": {"pos": (-0.0693, -0.0012, -0.0216), "rpy": (0.0, 1.5708, 0.0)},
        },
    },
}


class ComPalmBuilder(PalmBuilder):
    r"""Builder for composite preset palms."""

    cfg: ComPalmBuilderCfg

    def __init__(self, cfg: ComPalmBuilderCfg):
        super().__init__(cfg)
        self.cfg = cfg

    def build(self) -> PalmCfg:
        r"""Build a preset palm with multiple primitive collisions."""

        preset = _COM_PALM_PRESETS[self.cfg.preset]
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
        mounts = {name: PoseCfg.from_value(value) for name, value in preset["finger_mounts"].items()}
        metadata = {"preset": self.cfg.preset, "finger_mounts": mounts}
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
