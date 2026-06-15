r"""Pure config tests for heterogeneous URDF visual material restoration.

这些测试不启动 Isaac Sim，也不导入真实 USD / Omni binding。它们只锁住本轮
低侵入修复的两个 contract：

1. generated URDF 中 `<visual name="..."><material><color rgba="..."/>` 能被解析成
   `visual_name -> RGBA`；
2. opt-in `restore_urdf_visual_materials=True` 只替换 child `UrdfFileCfg.func`，不改变
   `MultiAssetSpawnerCfg` 的资产选择语义。
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _load_heterogeneous_cfg_module():
    r"""用最小 IsaacLab stub 加载 heterogeneous env cfg 文件。

    Returns:
        module: 包含 URDF 颜色解析 helper 与 `HeterogeneousHandSetCfg` 的临时模块。
    """

    module_path = Path(__file__).resolve().parents[1] / "heterogeneous_test_env_cfg.py"
    spec = importlib.util.spec_from_file_location(
        "anymani.tasks.gm.heterogeneous_test_env_cfg_under_test",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load heterogeneous test env cfg module from {module_path}")

    previous_modules = _install_isaaclab_cfg_stubs()
    module = importlib.util.module_from_spec(spec)
    previous_target_module = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_target_module is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous_target_module
        _restore_modules(previous_modules)
    return module


def _install_isaaclab_cfg_stubs() -> dict[str, types.ModuleType | None]:
    r"""安装加载 heterogeneous cfg 所需的 IsaacLab stub module tree。

    Returns:
        dict[str, types.ModuleType | None]: 被本测试临时接管模块的旧状态。
    """

    class _Cfg:
        r"""接受任意关键字的配置占位类。"""

        def __init__(self, **kwargs):
            r"""保存字段，模拟 IsaacLab configclass 实例的开放属性语义。"""

            self.__dict__.update(kwargs)

        def __post_init__(self) -> None:
            r"""ManagerBasedRLEnvCfg stub 的 no-op post init。"""

    class _UrdfFileCfg(_Cfg):
        r"""`sim_utils.UrdfFileCfg` 的最小近似，只保留 `func` 与传入字段。"""

        func = object()

    class _MultiAssetSpawnerCfg(_Cfg):
        r"""`sim_utils.MultiAssetSpawnerCfg` 的最小近似。"""

    class _ArticulationCfg(_Cfg):
        r"""`ArticulationCfg` stub，含 nested `InitialStateCfg`。"""

        InitialStateCfg = _Cfg

    class _AssetBaseCfg(_Cfg):
        r"""`AssetBaseCfg` stub，含 nested `InitialStateCfg`。"""

        InitialStateCfg = _Cfg

    class _JointDriveCfg(_Cfg):
        r"""`UrdfConverterCfg.JointDriveCfg` stub。"""

        class PDGainsCfg(_Cfg):
            r"""PD gains stub，保留 stiffness / damping 字段。"""

    class _UrdfConverterCfg:
        r"""`UrdfConverterCfg` namespace stub。"""

        JointDriveCfg = _JointDriveCfg

    def configclass(cls):
        r"""极小 `configclass`：按 annotated fields 生成 keyword-only init。"""

        annotations = getattr(cls, "__annotations__", {})

        def __init__(self, **kwargs):
            r"""用 class body default 与用户 kwargs 初始化配置对象。"""

            for name in annotations:
                if name in kwargs:
                    value = kwargs[name]
                elif hasattr(cls, name):
                    value = getattr(cls, name)
                else:
                    value = None
                setattr(self, name, value)
            for name, value in kwargs.items():
                if name not in annotations:
                    setattr(self, name, value)

        cls.__init__ = __init__
        return cls

    isaaclab = types.ModuleType("isaaclab")
    actuators = types.ModuleType("isaaclab.actuators")
    assets = types.ModuleType("isaaclab.assets")
    envs = types.ModuleType("isaaclab.envs")
    envs_common = types.ModuleType("isaaclab.envs.common")
    envs_mdp = types.ModuleType("isaaclab.envs.mdp")
    managers = types.ModuleType("isaaclab.managers")
    scene = types.ModuleType("isaaclab.scene")
    sim = types.ModuleType("isaaclab.sim")
    converters = types.ModuleType("isaaclab.sim.converters")
    physics_materials_cfg = types.ModuleType("isaaclab.sim.spawners.materials.physics_materials_cfg")
    utils = types.ModuleType("isaaclab.utils")
    utils_assets = types.ModuleType("isaaclab.utils.assets")
    anymani = types.ModuleType("anymani")
    tasks = types.ModuleType("anymani.tasks")
    gm = types.ModuleType("anymani.tasks.gm")
    asset_binding = types.ModuleType("anymani.tasks.gm.asset_binding")

    for package in (isaaclab, envs, sim, anymani, tasks, gm):
        package.__path__ = []

    actuators.ImplicitActuatorCfg = _Cfg
    assets.ArticulationCfg = _ArticulationCfg
    assets.AssetBaseCfg = _AssetBaseCfg
    envs.ManagerBasedRLEnvCfg = _Cfg
    envs_common.ViewerCfg = _Cfg
    envs_mdp.RelativeJointPositionActionCfg = _Cfg
    envs_mdp.joint_pos = object()
    envs_mdp.joint_vel = object()
    envs_mdp.is_alive = object()
    envs_mdp.time_out = object()
    managers.ObservationGroupCfg = _Cfg
    managers.ObservationTermCfg = _Cfg
    managers.RewardTermCfg = _Cfg
    managers.TerminationTermCfg = _Cfg
    scene.InteractiveSceneCfg = _Cfg
    sim.ArticulationRootPropertiesCfg = _Cfg
    sim.DomeLightCfg = _Cfg
    sim.GroundPlaneCfg = _Cfg
    sim.MultiAssetSpawnerCfg = _MultiAssetSpawnerCfg
    sim.PhysxCfg = _Cfg
    sim.PreviewSurfaceCfg = _Cfg
    sim.RigidBodyPropertiesCfg = _Cfg
    sim.SimulationCfg = _Cfg
    sim.UrdfFileCfg = _UrdfFileCfg
    converters.UrdfConverterCfg = _UrdfConverterCfg
    physics_materials_cfg.RigidBodyMaterialCfg = _Cfg
    utils.configclass = configclass
    utils_assets.ISAAC_NUCLEUS_DIR = "/Isaac/Nucleus"
    asset_binding.DEFAULT_HAND_INIT_POS = (0.0, 0.0, 0.0)
    asset_binding.DEFAULT_HAND_INIT_ROT = (1.0, 0.0, 0.0, 0.0)

    replacements = {
        "isaaclab": isaaclab,
        "isaaclab.actuators": actuators,
        "isaaclab.assets": assets,
        "isaaclab.envs": envs,
        "isaaclab.envs.common": envs_common,
        "isaaclab.envs.mdp": envs_mdp,
        "isaaclab.managers": managers,
        "isaaclab.scene": scene,
        "isaaclab.sim": sim,
        "isaaclab.sim.converters": converters,
        "isaaclab.sim.spawners.materials.physics_materials_cfg": physics_materials_cfg,
        "isaaclab.utils": utils,
        "isaaclab.utils.assets": utils_assets,
        "anymani": anymani,
        "anymani.tasks": tasks,
        "anymani.tasks.gm": gm,
        "anymani.tasks.gm.asset_binding": asset_binding,
    }
    previous = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)
    return previous


def _restore_modules(previous: dict[str, types.ModuleType | None]) -> None:
    r"""恢复测试 stub 安装前的 `sys.modules` 状态。"""

    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def test_generated_hand_urdf_visual_rgba_is_parsed_by_visual_name() -> None:
    r"""generated URDF 的 palm/finger/tip debug color 必须能按 visual name 解析。"""

    module = _load_heterogeneous_cfg_module()
    urdf_path = module.DEFAULT_HETEROGENEOUS_HAND_SET.resolve_urdf_path(
        module.DEFAULT_HETEROGENEOUS_HAND_SET.variants[0]
    )

    visual_rgba_by_name = module._parse_urdf_visual_rgba_by_name(urdf_path)

    assert visual_rgba_by_name["palm_visual"] == (0.603921569, 0.149019608, 0.149019608, 1.0)
    assert visual_rgba_by_name["index_j0_vis"] == (0.866666667, 0.866666667, 0.0509803922, 1.0)
    assert visual_rgba_by_name["index_tip_mesh_vis"] == (0.92, 0.88, 0.78, 1.0)


def test_restore_visual_materials_is_opt_in_on_urdf_child_cfg(tmp_path: Path) -> None:
    r"""只有 opt-in hand set 会把 child `UrdfFileCfg.func` 替换成颜色恢复 wrapper。"""

    module = _load_heterogeneous_cfg_module()
    bundle_dir = tmp_path / "variant_a"
    bundle_dir.mkdir()
    (bundle_dir / "hand.urdf").write_text("<robot name='stub'/>", encoding="utf-8")
    (bundle_dir / "hand.yaml").write_text("variant: a\n", encoding="utf-8")
    variant = module.HeterogeneousHandVariantCfg(variant_id="variant_a", bundle_dir="variant_a")

    default_hand_set = module.HeterogeneousHandSetCfg(
        topology_name="stub",
        base_dir=str(tmp_path),
        variants=(variant,),
        restore_urdf_visual_materials=False,
        validate_mesh_relpaths=False,
    )
    restored_hand_set = module.HeterogeneousHandSetCfg(
        topology_name="stub",
        base_dir=str(tmp_path),
        variants=(variant,),
        restore_urdf_visual_materials=True,
        validate_mesh_relpaths=False,
    )

    default_child_cfg = default_hand_set.build_multi_urdf_spawn_cfg().assets_cfg[0]
    restored_child_cfg = restored_hand_set.build_multi_urdf_spawn_cfg().assets_cfg[0]

    assert default_child_cfg.func is not module._spawn_urdf_with_restored_visual_materials
    assert restored_child_cfg.func is module._spawn_urdf_with_restored_visual_materials
    assert module.DEFAULT_HETEROGENEOUS_HAND_SET.restore_urdf_visual_materials is True


def test_heterogeneous_scene_uses_clear_sky_dome_light() -> None:
    r"""异构 GUI smoke 应复用 DexSuite 的清天 HDRI，而不是纯灰 dome light。"""

    module = _load_heterogeneous_cfg_module()
    texture_file = module.HeterogeneousHandTestSceneCfg.light.spawn.texture_file

    assert module.HeterogeneousHandTestSceneCfg.light.prim_path == "/World/skyLight"
    assert module.HeterogeneousHandTestSceneCfg.light.spawn.intensity == 750.0
    assert texture_file.endswith("/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr")


def test_heterogeneous_hand_root_pose_aligns_semantic_frame_to_world() -> None:
    r"""异构 smoke 的 generated hand 初态应表达 $R_{wh}=I$。"""

    module = _load_heterogeneous_cfg_module()
    robot_init_state = module.HeterogeneousHandTestSceneCfg.robot.init_state

    assert module.HETEROGENEOUS_HAND_INIT_ROT == (1.0, 0.0, 0.0, 0.0)
    assert robot_init_state.rot == module.HETEROGENEOUS_HAND_INIT_ROT
