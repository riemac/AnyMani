r"""Generated canonical hand与DexCube共享scene，不绑定pregrasp cache。

Formal task与pregrasp搜索必须观察同一articulation、object material、solver、scale和24-sensor ABI。Scene构造只解析
asset physical identity，不查询pregrasp records；因此尚未发布cache的新资产也能先进入搜索/认证，而正式task仍在自己的
event配置中fail closed解析exact basin。
"""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path

from anymani.pregrasp.isaac_runtime import file_sha256

from ...contact_sensors import install_contact_sensors
from .asset_binding import GeneratedAssetBinding, build_generated_asset_binding
from .pregrasp_identity import (
    DEX_CUBE_SHA256,
    FORMAL_OBJECT_DENSITY_KG_M3,
    FORMAL_OBJECT_SCALE,
    FORMAL_SOLVER_POSITION_ITERATIONS,
    FORMAL_SOLVER_VELOCITY_ITERATIONS,
    FormalPregraspCatalogIdentity,
)

OBJECT_SCALE = FORMAL_OBJECT_SCALE  # DexCube absolute USD scale；其它anchor由独立prestartup process覆盖spawn cfg
DEX_CUBE_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"
RESOLVED_DEX_CUBE_PATH = Path(retrieve_file_path(DEX_CUBE_USD_PATH)).resolve(strict=True)
RESOLVED_DEX_CUBE_SHA256 = file_sha256(RESOLVED_DEX_CUBE_PATH)
if RESOLVED_DEX_CUBE_SHA256 != DEX_CUBE_SHA256:
    raise RuntimeError("resolved DexCube USD bytes disagree with formal pregrasp identity")
FORMAL_PREGRASP_IDENTITY = FormalPregraspCatalogIdentity.build(
    object_scale=OBJECT_SCALE,
    cube_sha256=RESOLVED_DEX_CUBE_SHA256,
)
ASSET_BINDING: GeneratedAssetBinding = build_generated_asset_binding()  # 当前process唯一ordered physical axis
ASSET_COUNT = ASSET_BINDING.asset_count  # selection-local prototype数$A$
NUM_ENVS = int(os.environ.get("ANYMANI_HETERO_NUM_ENVS", str(ASSET_COUNT)))  # round-robin scene环境数$N$
if NUM_ENVS < ASSET_COUNT:
    raise ValueError("ANYMANI_HETERO_NUM_ENVS must be at least the selected asset count")
ACTIVE_MASK_BY_ENV = ASSET_BINDING.active_joint_mask_by_env(NUM_ENVS)  # $[N,16]$ canonical validity
CONTACT_LAYOUT = ASSET_BINDING.contact_layout  # 固定TIP4+non-tip19+PALM1 sensor ABI


@configclass
class GeneratedHeterogeneousSceneCfg(InteractiveSceneCfg):
    r"""Canonical generated hands、fixed-scale DexCube、ground/light与24 contact sensors。

    Object scale $s=1.2$、density 400 kg/m3与solver position iterations 8是P0001实测physics identity的一部分。
    搜索其它scale时必须在scene startup前覆盖``object.spawn.scale``，不能通过episode event修改collision geometry。
    """

    robot = ASSET_BINDING.hand_adapter.build_articulation_cfg(prim_path="{ENV_REGEX_NS}/Robot")
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=DEX_CUBE_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=FORMAL_SOLVER_POSITION_ITERATIONS,
                solver_velocity_iteration_count=FORMAL_SOLVER_VELOCITY_ITERATIONS,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=FORMAL_OBJECT_DENSITY_KG_M3),
            scale=(OBJECT_SCALE, OBJECT_SCALE, OBJECT_SCALE),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.6), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.1)),
    )
    light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )

    def __post_init__(self) -> None:
        r"""安装与formal task完全相同的object-filtered sensor集合。"""

        super().__post_init__()  # pyright: ignore[reportAttributeAccessIssue]
        install_contact_sensors(self, CONTACT_LAYOUT)


__all__ = [
    "ACTIVE_MASK_BY_ENV",
    "ASSET_BINDING",
    "ASSET_COUNT",
    "CONTACT_LAYOUT",
    "DEX_CUBE_USD_PATH",
    "FORMAL_PREGRASP_IDENTITY",
    "GeneratedHeterogeneousSceneCfg",
    "NUM_ENVS",
    "OBJECT_SCALE",
    "RESOLVED_DEX_CUBE_PATH",
    "RESOLVED_DEX_CUBE_SHA256",
]
