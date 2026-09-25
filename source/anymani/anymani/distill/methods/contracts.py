"""Typed lifecycle and objective contracts shared by embodiment methods."""


from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch

from anymani.distill.objectives.contracts import AdditiveStatistic, ObjectiveTermResult


@dataclass(frozen=True)
class FeatureSpec:


    entity_width: int
    entity_axis: str = "PALM/JOINT/TIP owner sequence"
    joint_view: str = "gather entities with joint_entity_index"
    frame_contract: str = "hand frame {h}; in-plane SO(2) gauge"
    coordinate_rewrite_contract: str = "density invariant; kappa sign-equivariant"

    def __post_init__(self) -> None:


        if self.entity_width < 1:
            raise ValueError("entity_width must be positive")


@dataclass(frozen=True)
class MethodParameterGroup:


    name: str
    parameters: tuple[torch.nn.Parameter, ...]

    def __post_init__(self) -> None:


        if not self.name or not self.parameters:
            raise ValueError("method parameter groups require a name and at least one parameter")


@dataclass(frozen=True)
class MethodStep:


    objectives: dict[str, ObjectiveTermResult]
    sample_count: int


@dataclass(frozen=True)
class MethodUpdate:


    terms: dict[str, float]
    sample_count: int
    denominators: dict[str, float] = field(default_factory=dict)
    gradient_evidence: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class MethodEvaluationReport:


    metrics: dict[str, float]
    strata: dict[str, object]
    teacher_baselines: dict[str, object] = field(default_factory=dict)
    ablations: dict[str, object] | None = None


@runtime_checkable
class MethodSplitSession(Protocol):


    @property
    def asset_count(self) -> int:

        ...

    def realize(self, schedule_item: Any, *, schedule: Any, step: int) -> Any:

        ...

    def realize_units(self, schedule_item: Any, *, schedule: Any, step: int) -> Iterable[Any]:

        ...

    def state_dict(self) -> dict[str, object]:

        ...

    def load_state_dict(self, state: Mapping[str, object]) -> None:

        ...

    def close(self) -> None:

        ...

    def drain_runtime_events(self) -> tuple[dict[str, object], ...]:

        ...


@runtime_checkable
class EmbodimentMethod(Protocol):


    def prepare(self, catalog: Any, *, role: str, device: torch.device, dtype: torch.dtype) -> None:

        ...

    def split_names(self, role: str) -> tuple[str, ...]:

        ...

    def split_asset_count(self, role: str, *, suite: str = "") -> int:

        ...

    def asset_manifest(self, catalog: Any) -> dict[str, Any]:

        ...

    def open_session(
        self,
        role: str,
        *,
        suite: str = "",
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
        max_resident_assets: int,
        window_factory: Any,
        resource_profile: bool = False,
    ) -> MethodSplitSession:

        ...

    def initialize_model(self, *, device: torch.device, dtype: torch.dtype) -> Any:

        ...

    def parameters(self) -> Iterable[torch.nn.Parameter]:

        ...

    def train_mode(self) -> None:

        ...

    def eval_mode(self) -> None:

        ...

    def forward_objectives(
        self,
        batch: Any,
        *,
        step: int,
        mode: str = "train",
        microbatch_size: int | None = None,
    ) -> MethodStep:

        ...

    def reduce_update(self, steps: tuple[MethodStep, ...]) -> MethodUpdate:

        ...

    def stage_replay_unit(self, unit: Any) -> Any:

        ...

    def restore_replay_unit(self, unit: Any, *, device: torch.device) -> Any:

        ...

    def evaluate_session(
        self,
        session: MethodSplitSession,
        schedule: Any,
        *,
        include_ablations: bool = False,
    ) -> MethodEvaluationReport:

        ...

    def analyze_ablations(
        self,
        evidence: Mapping[str, Any],
        *,
        bootstrap_replicates: int,
        seed: int,
    ) -> dict[str, Any]:

        ...

    def feature_spec(self) -> FeatureSpec:

        ...

    def training_state_dict(self) -> dict[str, torch.Tensor]:

        ...

    def load_training_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:

        ...

    def retained_artifact_payload(self, *, metadata: Mapping[str, Any], source_checkpoint: Path) -> dict[str, Any]:

        ...

    def close(self) -> None:

        ...

__all__ = [
    "AdditiveStatistic",
    "EmbodimentMethod",
    "FeatureSpec",
    "MethodStep",
    "MethodEvaluationReport",
    "MethodParameterGroup",
    "MethodSplitSession",
    "MethodUpdate",
    "ObjectiveTermResult",
]
