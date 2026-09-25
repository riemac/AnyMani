"""Shared LEAP and Allegro student actor. Inputs include 30-frame history, [B,21,128] owner geometry, and [B,16,15] joint kinematics. N040, No-Z, and FK differ only in geometry-token use and the FK auxiliary head; the action mean is [B,16]."""


from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from .palm_rotation_policy import (
    GEOMETRY_WIDTH,
    JOINT_COUNT,
    PalmRotationActorObservation,
    PalmRotationDirectActor,
    PalmRotationGeometry,
)

FAMILY_ROTATION_VARIANTS = ("n040", "no_z", "fk")
FamilyRotationVariant = Literal["n040", "no_z", "fk"]
JOINT_KINEMATICS_WIDTH = 15
JOINT_ORIGIN_WIDTH = 3
LINK_LENGTH_M = 0.1


@dataclass(frozen=True)
class FamilyRotationActorOutput:


    mean: torch.Tensor
    film_modulation_rms: torch.Tensor
    log_std: torch.Tensor
    fk_prediction: torch.Tensor | None = None

    @property
    def direct_mean(self) -> torch.Tensor:


        return self.mean


class FamilyRotationStudentActor(PalmRotationDirectActor):


    def __init__(
        self,
        variant: FamilyRotationVariant,
        *,
        initial_log_std: float = -0.5,
        max_log_std: float = -0.43,
    ) -> None:


        if variant not in FAMILY_ROTATION_VARIANTS:
            raise ValueError(f"family rotation variant must be one of {FAMILY_ROTATION_VARIANTS}, got {variant!r}")
        super().__init__(
            initial_log_std=initial_log_std,
            max_log_std=max_log_std,
            history_encoder="tcn",
            local_skip=False,
            sigma_mode="global",
            phase_clock_enabled=False,
        )
        self.variant: FamilyRotationVariant = variant
        self.representation: FamilyRotationVariant = variant
        self.joint_kinematics_width = JOINT_KINEMATICS_WIDTH
        self.link_length_m = LINK_LENGTH_M

        self.joint_kinematics_adapter = nn.Linear(JOINT_KINEMATICS_WIDTH, GEOMETRY_WIDTH)
        nn.init.zeros_(self.joint_kinematics_adapter.weight)
        nn.init.zeros_(self.joint_kinematics_adapter.bias)
        self.fk_head: nn.Module | None = None
        if variant == "fk":

            self.fk_head = nn.Sequential(
                nn.Linear(GEOMETRY_WIDTH, 64),
                nn.GELU(),
                nn.Linear(64, JOINT_ORIGIN_WIDTH),
            )
        self.family_actor_config: dict[str, object] = {
            "variant": variant,
            "representation": variant,
            "history_encoder": "tcn",
            "history_length": 30,
            "local_skip": False,
            "sigma_mode": "global",
            "phase_clock_enabled": False,
            "joint_kinematics_width": JOINT_KINEMATICS_WIDTH,
            "joint_origin_width": JOINT_ORIGIN_WIDTH,
            "link_length_m": LINK_LENGTH_M,
            "initial_log_std": float(initial_log_std),
            "max_log_std": float(max_log_std),
        }

        self.global_log_std.requires_grad_(False)

    def _effective_geometry(self, geometry: PalmRotationGeometry) -> PalmRotationGeometry:


        if self.variant in {"no_z", "fk"}:

            return PalmRotationGeometry(
                tokens=torch.zeros_like(geometry.tokens),
                owner_valid=geometry.owner_valid,
                shortest_path=geometry.shortest_path,
                parent_direction=geometry.parent_direction,
                child_direction=geometry.child_direction,
            )
        return geometry

    def _tip_only_observation(self, observation: PalmRotationActorObservation) -> PalmRotationActorObservation:


        current = observation.jnt_current.clone()
        history = observation.jnt_history.clone()
        current[..., 3] = 0.0
        history[..., 3] = 0.0
        owner_contact = observation.owner_contact.clone()
        owner_contact[:, :17] = 0.0
        return PalmRotationActorObservation(
            jnt_current=current,
            jnt_history=history,
            jnt_limits=observation.jnt_limits,
            owner_contact=owner_contact,
            jnt_valid=observation.jnt_valid,
            tip_valid=observation.tip_valid,
            owner_valid=observation.owner_valid,
        )

    def _contextual_tokens_with_kinematics(
        self,
        observation: PalmRotationActorObservation,
        geometry: PalmRotationGeometry,
        local: torch.Tensor,
        finger: torch.Tensor,
        hand: torch.Tensor,
        joint_kinematics: torch.Tensor,
    ) -> torch.Tensor:


        dynamic = torch.zeros_like(geometry.tokens)
        dynamic[:, 0] = self.palm_dynamic_projection(hand)  # PALM hand summary
        dynamic[:, 1:17] = self.joint_dynamic_projection(local)  # JOINT local FiLM state
        dynamic[:, 17:21] = self.tip_dynamic_projection(finger)  # TIP finger summary
        kinematic_delta = torch.zeros_like(geometry.tokens)
        joint_embedding = self.joint_kinematics_adapter(joint_kinematics)  # `[B,16,128]`
        joint_mask = observation.jnt_valid.unsqueeze(-1).to(dtype=joint_embedding.dtype)
        kinematic_delta[:, 1:17] = joint_embedding * joint_mask
        tokens = self.geometry_adapter(geometry.tokens) + dynamic
        tokens = tokens + self.owner_contact_projection(observation.owner_contact)  # TIP-only contact ABI
        tokens = tokens + kinematic_delta
        return self.global_backbone(
            tokens,
            geometry.shortest_path,
            geometry.parent_direction,
            geometry.child_direction,
            geometry.owner_valid,
        )

    def forward(
        self,
        observation: PalmRotationActorObservation,
        geometry: PalmRotationGeometry,
        *,
        joint_kinematics: torch.Tensor | None = None,
        phase_clock: torch.Tensor | None = None,
        _validated: bool = False,
    ) -> FamilyRotationActorOutput:


        del _validated
        if phase_clock is not None:
            raise ValueError("family rotation student requires phase_clock_enabled=False")
        if joint_kinematics is None:
            raise ValueError("family rotation student requires joint_kinematics with shape [B,16,15]")
        batch = observation.jnt_current.shape[0]
        expected_shape = (batch, JOINT_COUNT, JOINT_KINEMATICS_WIDTH)
        if tuple(joint_kinematics.shape) != expected_shape:
            raise ValueError(
                f"joint_kinematics must have shape {expected_shape}, got {tuple(joint_kinematics.shape)}"
            )
        if joint_kinematics.dtype != observation.jnt_current.dtype:
            raise ValueError(
                f"joint_kinematics must have dtype {observation.jnt_current.dtype}, got {joint_kinematics.dtype}"
            )
        if joint_kinematics.device != observation.jnt_current.device:
            raise ValueError(
                f"joint_kinematics must be on device {observation.jnt_current.device}, got {joint_kinematics.device}"
            )
        if geometry.tokens.shape[0] != batch:
            raise ValueError("family student observation and geometry batch sizes disagree")
        torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
            torch.isfinite(joint_kinematics).all(),
            "joint_kinematics must contain finite values",
        )
        torch._assert_async(  # pyright: ignore[reportPrivateImportUsage]
            torch.all(observation.owner_valid == geometry.owner_valid),
            "family student actor/geometry masks disagree",
        )
        effective_geometry = self._effective_geometry(geometry)
        effective_observation = self._tip_only_observation(observation)
        local, finger, hand, film_modulation_rms = self._local_and_hand(effective_observation, effective_geometry)
        contextual = self._contextual_tokens_with_kinematics(
            effective_observation,
            effective_geometry,
            local,
            finger,
            hand,
            joint_kinematics,
        )
        contextual_joint = contextual[:, 1:17]
        raw_direct = self.direct_head(contextual_joint).squeeze(-1)
        mean = torch.tanh(raw_direct)
        mean = torch.where(effective_observation.jnt_valid, mean, torch.zeros_like(mean))
        fk_prediction = None
        if self.fk_head is not None:

            fk_prediction = self.fk_head(contextual_joint)
            fk_prediction = torch.where(
                observation.jnt_valid.unsqueeze(-1),
                fk_prediction,
                torch.zeros_like(fk_prediction),
            )
        return FamilyRotationActorOutput(
            mean=mean,
            film_modulation_rms=film_modulation_rms,
            log_std=self._policy_log_std(contextual, effective_observation),
            fk_prediction=fk_prediction,
        )


def build_family_rotation_policy(
    variant: FamilyRotationVariant,
    *,
    device: torch.device | str | None = None,
    initial_log_std: float = -0.5,
    max_log_std: float = -0.43,
) -> FamilyRotationStudentActor:


    actor = FamilyRotationStudentActor(
        variant,
        initial_log_std=initial_log_std,
        max_log_std=max_log_std,
    )
    return actor.to(device=device, dtype=torch.float32)


FamilyRotationPolicy = FamilyRotationStudentActor


__all__ = [
    "FAMILY_ROTATION_VARIANTS",
    "FamilyRotationActorOutput",
    "FamilyRotationStudentActor",
    "FamilyRotationPolicy",
    "FamilyRotationVariant",
    "JOINT_KINEMATICS_WIDTH",
    "JOINT_ORIGIN_WIDTH",
    "LINK_LENGTH_M",
    "build_family_rotation_policy",
]
