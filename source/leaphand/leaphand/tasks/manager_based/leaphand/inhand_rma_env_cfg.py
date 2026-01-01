# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""LeapHand in-hand rotation (RMA) environment config - ManagerBasedRLEnv.

This config is intentionally isolated from other task configs (e.g. float / round-tip)
so that RMA-specific observation groups can be enabled without affecting other uses.

RMA expects three observation groups:
- policy: base policy observation (may include privileged kinematics in sim)
- priv_info: privileged extrinsics / object properties for μ(priv_info)
- proprio_hist: proprioception + action history for φ(proprio_hist)

The env wrapper can be configured to return a dict of obs groups by setting
`obs_groups` and `concate_obs_group=False` in the rl_games env config.
"""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp

from . import mdp as leap_mdp
from .inhand_base_env_cfg import InHandObjectEnvCfg as _BaseInHandObjectEnvCfg
from .inhand_base_env_cfg import ObservationsCfg as _BaseObservationsCfg


@configclass
class ObservationsCfg(_BaseObservationsCfg):
    """RMA observations: add `priv_info` and `proprio_hist` groups."""

    @configclass
    class PrivInfoCfg(ObsGroup):
        """RMA privileged object properties (for μ)."""

        obj_mass_scale = ObsTerm(
            func=leap_mdp.object_mass_scale,
            params={"asset_cfg": SceneEntityCfg("object"), "scale_range": (0.25, 1.2)},
        )
        obj_com_offset = ObsTerm(
            func=leap_mdp.object_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "com_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
            },
        )
        obj_material = ObsTerm(
            func=leap_mdp.object_material_properties,
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "static_friction_range": (0.2, 1.0),
                "dynamic_friction_range": (0.15, 0.6),
                "restitution_range": (0.0, 0.1),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ProprioHistCfg(ObsGroup):
        """RMA adaptation input: proprioception + action history (for φ)."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos_limit_normalized,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    priv_info: ObsGroup = PrivInfoCfg()
    proprio_hist: ObsGroup = ProprioHistCfg(history_length=30)


@configclass
class InHandRmaEnvCfg(_BaseInHandObjectEnvCfg):
    """LeapHand continuous rotation with RMA observation groups."""

    observations: ObservationsCfg = ObservationsCfg()
