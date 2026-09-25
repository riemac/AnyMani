"Connects hand configurations to palm, finger, and joint builders without changing their physical values."

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .asset_base import AssetCfgBase, FingerCfg, HandCfg, JointCfg, PalmCfg

if TYPE_CHECKING:
    from .asset_schema_core import WristJointSpec


@dataclass
class BuilderCfg(AssetCfgBase):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    class_type: type["Builder"] | None = None
    "Associated runtime implementation for this configuration class."


@dataclass
class JointBuilderCfg(BuilderCfg):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    class_type: type["Builder"] | None = None
    "Associated runtime implementation for this configuration class."

    is_customized: bool = None
    "Whether this generation or validation condition is active."

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = JointBuilder


@dataclass
class FingerBuilderCfg(BuilderCfg):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    class_type: type["Builder"] | None = None
    "Associated runtime implementation for this configuration class."

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = FingerBuilder


@dataclass
class PalmBuilderCfg(BuilderCfg):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    class_type: type["Builder"] | None = None
    "Associated runtime implementation for this configuration class."

    wrist_joints: list["WristJointSpec"] | None = None
    "Optional wrist articulation entries in proximal-to-distal order."

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = PalmBuilder


@dataclass
class HandBuilderCfg(BuilderCfg):
    "Declares unit-aware inputs and validation policy for the asset pipeline."

    name: str = "hand"
    "Stable semantic identifier preserved in the output metadata."

    family: str = "generic"
    "Base-palm family; surviving slot families are stored separately for mixed hands."

    palm_cfg: "PalmBuilderCfg | None" = None
    "Typed palm geometry and mount configuration."

    class_type: type["Builder"] | None = None
    "Associated runtime implementation for this configuration class."

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = HandBuilder


class Builder:
    "Builds a configured hand component from typed geometry and local frames."

    cfg: BuilderCfg

    def __init__(self, cfg: BuilderCfg):
        self.cfg = cfg

    def build(self) -> AssetCfgBase:
        "Builds the configured geometry component from typed dimensions and local frames."

        raise NotImplementedError('Builder is a scaffold; its concrete construction algorithm is not implemented.')


class JointBuilder(Builder):
    "Builds a configured hand component from typed geometry and local frames."

    def __init__(self, cfg: JointBuilderCfg):
        super().__init__(cfg)

    def build(self) -> JointCfg:
        "Builds the configured geometry component from typed dimensions and local frames."

        raise NotImplementedError('JointBuilder is a scaffold; joint-level construction is not implemented.')


class FingerBuilder(Builder):
    "Builds a configured hand component from typed geometry and local frames."

    def __init__(self, cfg: FingerBuilderCfg):
        super().__init__(cfg)

    def build(self) -> FingerCfg:
        "Builds the configured geometry component from typed dimensions and local frames."

        raise NotImplementedError('FingerBuilder is a scaffold; finger-level construction is not implemented.')


class PalmBuilder(Builder):
    "Builds a configured hand component from typed geometry and local frames."

    def __init__(self, cfg: PalmBuilderCfg):
        super().__init__(cfg)

    def build(self) -> PalmCfg:
        "Builds the configured geometry component from typed dimensions and local frames."

        raise NotImplementedError('PalmBuilder is a scaffold; palm-level construction is not implemented.')


class HandBuilder(Builder):
    "Builds a configured hand component from typed geometry and local frames."

    def __init__(self, cfg: HandBuilderCfg):
        super().__init__(cfg)

    def build(self) -> HandCfg:
        "Builds the configured geometry component from typed dimensions and local frames."

        raise NotImplementedError('HandBuilder is a scaffold; hand-level construction is not implemented.')


__all__ = [
    "BuilderCfg",
    "JointBuilderCfg",
    "FingerBuilderCfg",
    "PalmBuilderCfg",
    "HandBuilderCfg",
    "Builder",
    "JointBuilder",
    "FingerBuilder",
    "PalmBuilder",
    "HandBuilder",
]
