r"""异构资产手内操作环境配置文件。

这里仅用最简单的观察、动作、重置等 MDP 组件（尽量 IsaacLab 官方预定义即可）来测试同拓扑异构资产在同一环境训练的可能性，做低侵入式 MVP。分两步：

1. 测试同拓扑异构资产是否可以通过类似 `random_agent.py` 的随机运行测试，这里不做训练
2. 在确认第一步可行后，测试同拓扑异构资产是否可以通过类似 `train.py` 的训练测试，用最简单的 MLP 网络即可，仅判断训练的可行性，不追求表现。

目标 urdf 用：

- AnyMani/source/anymani/anymani/assets/generated/2026-06-10_11-30-08/single_palm_leap/right_t4_i4_m4_r4/2026-06-11_14-20-22/0b6fbfce/hand.urdf
- AnyMani/source/anymani/anymani/assets/generated/2026-06-10_11-30-08/single_palm_leap/right_t4_i4_m4_r4/2026-06-11_14-20-22/0bdf0eca/hand.urdf
- AnyMani/source/anymani/anymani/assets/generated/2026-06-10_11-30-08/single_palm_leap/right_t4_i4_m4_r4/2026-06-11_14-20-22/00d68163/hand.urdf

这 3 个 urdf 即可。
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import MISSING
from pathlib import Path

import isaaclab.envs.mdp as isaac_mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.converters import UrdfConverterCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .asset_binding import DEFAULT_HAND_INIT_POS

logger = logging.getLogger(__name__)
_UrdfRgba = tuple[float, float, float, float]

_ANYMANI_ROOT = Path(__file__).resolve().parents[5]
r"""AnyMani repository root.

本文件位于 `source/anymani/anymani/tasks/gm/`，向上 5 层到达仓库根目录。
显式锚定 repo root 是为了让测试环境与当前工作目录无关；IsaacLab 的 URDF importer
拿到的是绝对 `asset_path`，URDF 内部 `../meshes/...` 仍按 URDF 自身目录解析。
"""

_HETEROGENEOUS_RUN_ROOT = _ANYMANI_ROOT / (
    "source/anymani/anymani/assets/generated/2026-06-10_11-30-08/"
    "single_palm_leap/right_t4_i4_m4_r4/2026-06-11_14-20-22"
)
r"""本 MVP 固定使用的 post-mutate run 目录。

目录结构约定：

```text
<run_root>/
  0b6fbfce/hand.urdf
  0b6fbfce/hand.yaml
  ...
  meshes/*.stl|*.obj
```

因此 `hand.urdf` 中的 `../meshes/foo.stl` 应解析到 `<run_root>/meshes/foo.stl`。
"""

HETEROGENEOUS_HAND_INIT_ROT = (1.0, 0.0, 0.0, 0.0)
r"""异构可视化 smoke 的 hand root 初始姿态 `(w,x,y,z)`。

该测试环境的目的不是复刻旧 LeapHand object-in-hand 任务的腕部姿态，而是核对
same-schema generated hand 在世界系中的可视语义。因此这里显式采用单位四元数，
表达第一版假设 `{a} \approx {h}` 且 $R_{wh}=I$：手心语义法向 $z^h$ 指向
世界上方 $z^w$，$x^h,y^h$ 与 $x^w,y^w$ 同向。若后续资产 metadata 给出严格
`{a}->{h}` 校准矩阵，应在本地 smoke cfg 中组合该固定校准，而不是回退到旧
LeapHand 视觉锚点。
"""


@configclass
class HeterogeneousHandVariantCfg:
    r"""一个 same-schema post-mutate hand variant 的声明式路径引用。

    Args:
        variant_id (str): post-mutate 样本短 hash，例如 `0b6fbfce`。
        bundle_dir (str): 单个 variant bundle 目录；相对路径按 hand set 的
            `base_dir` 解析，绝对路径则直接使用。
        urdf_file (str): bundle 内 URDF 文件名，默认 `hand.urdf`。
        sidecar_file (str): bundle 内 sidecar 文件名，默认 `hand.yaml`。

    NOTE:
        这里没有引入 asset bank 抽象；它只是未来 bank manifest 的最小字段原型。
    """

    variant_id: str = MISSING
    bundle_dir: str = MISSING
    urdf_file: str = "hand.urdf"
    sidecar_file: str = "hand.yaml"


@configclass
class HeterogeneousHandSetCfg:
    r"""同一 articulation schema 下的一组异构 hand variants。

    这个 cfg 负责从声明式路径解析到 IsaacLab 可消费的 `UrdfFileCfg` 列表。
    第一版只支持 same-schema post-mutate variants；跨拓扑 mixed assets 留接口，
    但不在此测试环境内实现。
    """

    topology_name: str = MISSING
    base_dir: str = MISSING
    variants: tuple[HeterogeneousHandVariantCfg, ...] = MISSING
    random_choice: bool = False
    validate_mesh_relpaths: bool = True
    restore_urdf_visual_materials: bool = False
    r"""是否在 URDF spawn 后恢复 generated URDF 的 per-visual RGB debug color。

    Isaac Sim 5.1 的 URDF importer 在当前 generated hand 上没有稳定保留
    `<visual><material><color rgba=...>`：primitive box visual 常常没有 bound
    material，mesh tip 则可能得到 importer 默认白色 material。这个开关只修复
    GUI / render 语义，不改变 collision、mass、joint、drive 或 RL MDP。
    """

    def resolve_bundle_dir(self, variant: HeterogeneousHandVariantCfg) -> Path:
        r"""解析单个 variant bundle 目录。

        Args:
            variant (HeterogeneousHandVariantCfg): 待解析的 variant 声明。

        Returns:
            Path: 绝对 bundle 目录。
        """

        raw_bundle_dir = Path(variant.bundle_dir).expanduser()
        if raw_bundle_dir.is_absolute():
            return raw_bundle_dir.resolve()
        return (Path(self.base_dir).expanduser().resolve() / raw_bundle_dir).resolve()

    def resolve_urdf_path(self, variant: HeterogeneousHandVariantCfg) -> Path:
        r"""解析并返回 variant 的绝对 URDF 路径。"""

        return self.resolve_bundle_dir(variant) / variant.urdf_file

    def resolve_sidecar_path(self, variant: HeterogeneousHandVariantCfg) -> Path:
        r"""解析并返回 variant 的绝对 sidecar 路径。"""

        return self.resolve_bundle_dir(variant) / variant.sidecar_file

    def validate(self) -> None:
        r"""验证 hand set 的文件闭包。

        检查内容：

        1. hand set 非空；
        2. 每个 variant 有 `hand.urdf` 与 `hand.yaml`；
        3. URDF 中所有相对 mesh 路径均能从 `urdf.parent` 解析到真实文件。

        Raises:
            ValueError: hand set 为空或存在不支持的 mesh URI。
            FileNotFoundError: 必需文件或 mesh 文件不存在。
        """

        if len(self.variants) == 0:
            raise ValueError("Heterogeneous hand set must contain at least one variant.")

        for variant in self.variants:
            urdf_path = self.resolve_urdf_path(variant)
            sidecar_path = self.resolve_sidecar_path(variant)
            _require_file(urdf_path, label=f"{variant.variant_id}/hand.urdf")
            _require_file(sidecar_path, label=f"{variant.variant_id}/hand.yaml")
            if self.validate_mesh_relpaths:
                _validate_urdf_mesh_relpaths(urdf_path, variant_id=variant.variant_id)

    def build_multi_urdf_spawn_cfg(self) -> sim_utils.MultiAssetSpawnerCfg:
        r"""构造 IsaacLab `MultiAssetSpawnerCfg`。

        Returns:
            sim_utils.MultiAssetSpawnerCfg: 包含多个 `UrdfFileCfg` 的异构 spawn cfg。
        """

        self.validate()
        return sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                _build_hand_urdf_file_cfg(
                    self.resolve_urdf_path(variant),
                    restore_visual_materials=self.restore_urdf_visual_materials,
                )
                for variant in self.variants
            ],
            random_choice=self.random_choice,
        )


def _require_file(path: Path, *, label: str) -> Path:
    r"""要求给定路径是文件，并返回绝对路径。"""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Heterogeneous hand test asset is missing {label}: {resolved}")
    return resolved


def _validate_urdf_mesh_relpaths(urdf_path: Path, *, variant_id: str) -> None:
    r"""检查 URDF 内 mesh filename 是否能从 `urdf.parent` 闭合解析。

    Args:
        urdf_path (Path): 已存在的 URDF 文件路径。
        variant_id (str): 错误消息中的 variant 标识。

    Raises:
        ValueError: 遇到 `package://` 等第一版不支持的 mesh URI。
        FileNotFoundError: 相对或绝对 mesh 文件不存在。
    """

    root = ET.parse(urdf_path).getroot()
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if filename is None:
            continue

        if "://" in filename:
            raise ValueError(
                f"Unsupported mesh URI in heterogeneous test asset {variant_id}: {filename!r}. "
                "First MVP expects local relative or absolute mesh paths."
            )

        mesh_path = Path(filename).expanduser()
        if not mesh_path.is_absolute():
            mesh_path = urdf_path.parent / mesh_path
        if not mesh_path.resolve().is_file():
            raise FileNotFoundError(
                f"Mesh referenced by heterogeneous test asset {variant_id} is missing: "
                f"{filename!r} -> {mesh_path.resolve()}"
            )


def _parse_urdf_visual_rgba_by_name(urdf_path: Path) -> dict[str, _UrdfRgba]:
    r"""解析 URDF named visual 对应的 RGBA debug color。

    generated hand 的颜色语义写在 URDF 层：每个 `<visual name="...">`
    下面通常有局部 `<material><color rgba="r g b a"/>`。Isaac Sim importer
    当前没有稳定把这层信息落到 USD material binding，因此这里把它抽取成一个
    纯 Python contract，供 spawn 后的 USD wrapper 恢复视觉语义。

    Args:
        urdf_path (Path): generated hand 的 `hand.urdf` 路径。

    Returns:
        dict[str, _UrdfRgba]: `visual_name -> (r,g,b,a)`；匿名 visual 或无颜色
            material 的 visual 被跳过。

    Raises:
        ValueError: `rgba` 不是四个浮点数时抛出，说明 asset 颜色规格本身损坏。
    """

    root = ET.parse(urdf_path).getroot()
    named_material_rgba: dict[str, _UrdfRgba] = {}
    visual_rgba_by_name: dict[str, _UrdfRgba] = {}

    # 先收集 URDF 顶层 material。虽然当前 generated hand 多数使用 visual-local
    # material，这里保留 URDF 标准 reference 路径，避免未来 exporter 做去重时破坏恢复逻辑。
    for material in root.findall("./material"):
        material_name = material.attrib.get("name")
        color = material.find("color")
        if material_name is None or color is None:
            continue
        if rgba_text := color.attrib.get("rgba"):
            named_material_rgba[material_name] = _parse_urdf_rgba(rgba_text, urdf_path=urdf_path)

    # 再逐个 named visual 读取局部 color；若 visual 只引用顶层 material，则退回顶层表。
    for visual in root.findall(".//visual"):
        visual_name = visual.attrib.get("name")
        material = visual.find("material")
        if visual_name is None or material is None:
            continue

        color = material.find("color")
        if color is not None and (rgba_text := color.attrib.get("rgba")):
            visual_rgba_by_name[visual_name] = _parse_urdf_rgba(rgba_text, urdf_path=urdf_path)
            continue

        material_name = material.attrib.get("name")
        if material_name is not None and material_name in named_material_rgba:
            visual_rgba_by_name[visual_name] = named_material_rgba[material_name]

    return visual_rgba_by_name


def _parse_urdf_rgba(rgba_text: str, *, urdf_path: Path) -> _UrdfRgba:
    r"""把 URDF `rgba="r g b a"` 解析成四元浮点颜色。

    Args:
        rgba_text (str): URDF material color 字符串。
        urdf_path (Path): 错误消息中的资产路径锚点。

    Returns:
        _UrdfRgba: 四个通道值，范围按 URDF exporter 约定通常在 $[0,1]$。

    Raises:
        ValueError: 通道数不是 4，或无法解析成浮点数。
    """

    try:
        channels = tuple(float(channel) for channel in rgba_text.split())
    except ValueError as exc:
        raise ValueError(f"Invalid URDF rgba color in {urdf_path}: {rgba_text!r}") from exc

    if len(channels) != 4:
        raise ValueError(f"URDF rgba color must have four channels in {urdf_path}: {rgba_text!r}")

    return channels


def _spawn_urdf_with_restored_visual_materials(
    prim_path: str,
    cfg: sim_utils.UrdfFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    r"""官方 URDF spawn 之后，恢复 generated hand 的 per-visual debug color。

    这个 wrapper 是有意保持低侵入的配置层补丁：动力学资产仍由 IsaacLab 官方
    `spawn_from_urdf` 创建；本函数只在返回的 USD prim 子树上补充 visual material
    binding，使 GUI 中看到的 palm / phalange / tip 颜色重新对应 URDF 语义。

    Args:
        prim_path (str): IsaacLab 传入的目标 prim path；在 `MultiAssetSpawnerCfg`
            中通常是 `/World/Template/Asset_000*` prototype。
        cfg (sim_utils.UrdfFileCfg): 官方 URDF importer 配置，`asset_path` 指向
            generated `hand.urdf`。
        translation (tuple[float, float, float] | None): 官方 spawn 位移参数。
        orientation (tuple[float, float, float, float] | None): 官方 spawn 姿态参数，
            采用 IsaacLab `(w,x,y,z)` 四元数约定。
        **kwargs: IsaacLab wrapper 透传的 `clone_in_fabric`、`replicate_physics` 等参数。

    Returns:
        Usd.Prim: 官方 `spawn_from_urdf` 返回的 root prim。
    """

    from isaaclab.sim.spawners.from_files import spawn_from_urdf

    spawned_prim = spawn_from_urdf(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    visual_rgba_by_name = _parse_urdf_visual_rgba_by_name(Path(cfg.asset_path))
    _restore_visual_materials_on_spawned_prim(spawned_prim, visual_rgba_by_name)
    return spawned_prim


def _restore_visual_materials_on_spawned_prim(spawned_prim, visual_rgba_by_name: dict[str, _UrdfRgba]) -> None:
    r"""在一个已 spawn 的 hand prim 子树上绑定 URDF visual colors。

    USD binding 写在 prototype 阶段最有利：`MultiAssetSpawnerCfg` 随后用
    `Sdf.CopySpec` 把 prototype 复制到各个 cloned env，因此每个 asset variant
    只需要恢复一次颜色。若某个 visual 无法匹配或单个 binding 失败，只 warning
    并保留官方导入结果，避免可视化补丁阻断 RL smoke。

    Args:
        spawned_prim: 官方 URDF spawn 返回的 root prim。
        visual_rgba_by_name (dict[str, _UrdfRgba]): URDF visual name 到 RGBA 的映射。
    """

    if len(visual_rgba_by_name) == 0:
        return

    visual_prims = _find_spawned_visual_prims_by_name(spawned_prim, set(visual_rgba_by_name))
    bound_target_by_path: dict[str, str] = {}
    missing_visual_names: list[str] = []

    for visual_name, rgba in visual_rgba_by_name.items():
        visual_prim = visual_prims.get(visual_name)
        if visual_prim is None:
            missing_visual_names.append(visual_name)
            continue

        target_prim = _nearest_editable_material_binding_prim(visual_prim)
        target_path = str(target_prim.GetPath())
        previous_visual_name = bound_target_by_path.get(target_path)
        if previous_visual_name is not None and visual_rgba_by_name[previous_visual_name][:3] != rgba[:3]:
            logger.warning(
                "Skip URDF visual color for %s because editable USD target %s was already bound for %s.",
                visual_name,
                target_path,
                previous_visual_name,
            )
            continue

        try:
            _bind_urdf_preview_surface(spawned_prim, target_prim, visual_name, rgba)
        except Exception as exc:
            logger.warning("Failed to restore URDF visual color for %s on %s: %s", visual_name, target_path, exc)
            continue

        bound_target_by_path[target_path] = visual_name

    if missing_visual_names:
        logger.warning(
            "Could not find %d URDF visual prims under spawned hand %s; examples: %s",
            len(missing_visual_names),
            spawned_prim.GetPath(),
            missing_visual_names[:5],
        )


def _find_spawned_visual_prims_by_name(spawned_prim, visual_names: set[str]) -> dict[str, object]:
    r"""在 spawned hand 子树内查找与 URDF visual name 同名的 USD prim。

    Isaac Sim 的 instanceable URDF USD 会把具体 geometry 放到 prototype 中；普通
    `Stage.Traverse()` 看不到这些 instance proxy。这里显式使用
    `Usd.TraverseInstanceProxies()`，才能命中 `/visuals/<visual_name>` 这类 prim。

    Args:
        spawned_prim: hand root prim。
        visual_names (set[str]): 需要恢复颜色的 URDF visual name 集合。

    Returns:
        dict[str, object]: `visual_name -> Usd.Prim`；同名重复时保留第一次命中。
    """

    from pxr import Usd

    visual_prims: dict[str, object] = {}
    prim_range = Usd.PrimRange(spawned_prim, Usd.TraverseInstanceProxies())
    for prim in prim_range:
        prim_name = prim.GetName()
        prim_path = str(prim.GetPath())
        if prim_name in visual_names and "/visuals/" in prim_path and prim_name not in visual_prims:
            visual_prims[prim_name] = prim

    return visual_prims


def _nearest_editable_material_binding_prim(visual_prim):
    r"""为 material binding 选择最近的非 instance-proxy ancestor。

    当前 generated hand 的 `/visuals/<visual_name>` 往往是 instance proxy，直接
    author material binding 可能被 USD 拒绝；它的父节点 `/visuals` 是普通 prim，且
    每个 link 只有一个 visual，因此在父节点绑定 `strongerThanDescendants` 能同时覆盖
    primitive box 和 mesh tip 的默认材质。

    Args:
        visual_prim: `Usd.PrimRange(..., TraverseInstanceProxies())` 找到的 visual prim。

    Returns:
        Usd.Prim: 可 author material binding 的最近 ancestor。
    """

    target_prim = visual_prim
    while target_prim.IsInstanceProxy():
        parent_prim = target_prim.GetParent()
        if not parent_prim.IsValid():
            break
        target_prim = parent_prim
    return target_prim


def _bind_urdf_preview_surface(spawned_prim, target_prim, visual_name: str, rgba: _UrdfRgba) -> None:
    r"""创建并绑定一个表示 URDF RGB 的 USD PreviewSurface material。

    Args:
        spawned_prim: hand root prim；material 会创建在其 `Looks/` 子树下。
        target_prim: 需要接受 material binding 的 USD prim。
        visual_name (str): URDF visual name，用于生成稳定 material path。
        rgba (_UrdfRgba): URDF 颜色；本轮只使用 RGB，alpha 暂不映射到 USD opacity。
    """

    from pxr import UsdShade

    stage = spawned_prim.GetStage()
    root_path = str(spawned_prim.GetPath())
    looks_path = f"{root_path}/Looks"
    material_path = f"{looks_path}/{_sanitize_usd_prim_name('urdf_' + visual_name)}"

    if not stage.GetPrimAtPath(looks_path).IsValid():
        stage.DefinePrim(looks_path, "Scope")

    if not stage.GetPrimAtPath(material_path).IsValid():
        material_cfg = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(rgba[0], rgba[1], rgba[2]),
            roughness=0.5,
            metallic=0.0,
        )
        material_cfg.func(material_path, material_cfg)

    material = UsdShade.Material(stage.GetPrimAtPath(material_path))
    if target_prim.HasAPI(UsdShade.MaterialBindingAPI):
        material_binding_api = UsdShade.MaterialBindingAPI(target_prim)
    else:
        material_binding_api = UsdShade.MaterialBindingAPI.Apply(target_prim)
    material_binding_api.Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)


def _sanitize_usd_prim_name(raw_name: str) -> str:
    r"""把 URDF visual name 转成保守合法的 USD prim name 片段。"""

    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", raw_name)
    if sanitized == "" or sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized


def _build_hand_urdf_file_cfg(
    urdf_path: Path,
    *,
    restore_visual_materials: bool = False,
) -> sim_utils.UrdfFileCfg:
    r"""为一个 generated hand URDF 构造 importer cfg。

    这些参数与 `asset_binding.build_hand_articulation_cfg` 的单资产 debug 路线保持一致，
    但此处作为 `MultiAssetSpawnerCfg.assets_cfg` 的元素使用。
    """

    urdf_file_cfg = sim_utils.UrdfFileCfg(
        asset_path=str(urdf_path.resolve()),
        fix_base=True,
        merge_fixed_joints=False,
        force_usd_conversion=False,
        make_instanceable=True,
        collision_from_visuals=False,
        self_collision=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            drive_type="force",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=3.0, damping=0.1),
        ),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            enable_gyroscopic_forces=False,
            angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=64.0 / 3.141592653589793 * 180.0,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
            fix_root_link=True,
        ),
    )
    if restore_visual_materials:
        urdf_file_cfg.func = _spawn_urdf_with_restored_visual_materials
    return urdf_file_cfg


DEFAULT_HETEROGENEOUS_HAND_SET = HeterogeneousHandSetCfg(
    topology_name="right_t4_i4_m4_r4",
    base_dir=str(_HETEROGENEOUS_RUN_ROOT),
    random_choice=False,
    validate_mesh_relpaths=True,
    restore_urdf_visual_materials=True,
    variants=(
        HeterogeneousHandVariantCfg(variant_id="0b6fbfce", bundle_dir="0b6fbfce"),
        HeterogeneousHandVariantCfg(variant_id="0bdf0eca", bundle_dir="0bdf0eca"),
        HeterogeneousHandVariantCfg(variant_id="00d68163", bundle_dir="00d68163"),
    ),
)
r"""第一版异构 smoke 固定使用的 3 个 same-schema post-mutate variants。

`random_choice=False` 时 IsaacLab 会按 `assets_cfg[index % len(assets_cfg)]` 轮转，
因此 `num_envs=9` 应在 GUI 中看到 A/B/C/A/B/C/A/B/C。
"""


def build_heterogeneous_hand_articulation_cfg(
    hand_set: HeterogeneousHandSetCfg,
    *,
    prim_path: str,
) -> ArticulationCfg:
    r"""将声明式 hand set 绑定为一个 batched `ArticulationCfg`。

    Args:
        hand_set (HeterogeneousHandSetCfg): same-schema hand variant 集合。
        prim_path (str): IsaacLab scene 中 robot articulation 的 prim path。

    Returns:
        ArticulationCfg: 可作为 `scene.robot` 的异构 articulation 配置。
    """

    return ArticulationCfg(
        prim_path=prim_path,
        spawn=hand_set.build_multi_urdf_spawn_cfg(),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=DEFAULT_HAND_INIT_POS,
            rot=HETEROGENEOUS_HAND_INIT_ROT,
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=0.95,
                velocity_limit_sim=8.48,
                stiffness=3.0,
                damping=0.1,
                friction=0.01,
                armature=0.001,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )


@configclass
class HeterogeneousHandTestSceneCfg(InteractiveSceneCfg):
    r"""只包含异构 hand articulation 的最小 scene。

    没有 object；目标是隔离验证同一 `Articulation` batch 能否持有多个
    same-schema post-mutate hand variants。
    """

    robot: ArticulationCfg = build_heterogeneous_hand_articulation_cfg(
        DEFAULT_HETEROGENEOUS_HAND_SET,
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.1)),
    )

    light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class HeterogeneousHandTestActionsCfg:
    r"""官方相对关节位置动作。

    第一版不使用 GM clamp action，避免把异构 articulation smoke 与自定义 MDP
    组件耦合。`scale=0.05` 让随机 agent 的 joint target 做较慢随机游走。
    """

    joint_pos = isaac_mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.05,
        preserve_order=True,
    )


@configclass
class HeterogeneousHandTestObservationsCfg:
    r"""最小 policy observation：关节位置与速度。"""

    @configclass
    class PolicyCfg(ObsGroup):
        r"""Actor-facing flat observation group。"""

        joint_pos = ObsTerm(func=isaac_mdp.joint_pos)
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel)

        def __post_init__(self) -> None:
            r"""关闭噪声并拼接成单个 flat tensor。"""

            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class HeterogeneousHandTestRewardsCfg:
    r"""最小 alive reward，只保证 RL env reward manager 有合法输出。"""

    alive = RewTerm(func=isaac_mdp.is_alive, weight=1.0)


@configclass
class HeterogeneousHandTestTerminationsCfg:
    r"""最小 termination：只按 episode time limit reset。"""

    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)


@configclass
class HeterogeneousHandTestEnvCfg(ManagerBasedRLEnvCfg):
    r"""异构 hand articulation smoke env。

    该环境不是正式 GM teacher，也不表达 object manipulation 任务；它只验证
    IsaacLab 能否在一个 batched `Articulation` 中承载 3 个 same-schema URDF hand
    variants，并能被 `scripts/random_agent.py` reset/step。
    """

    scene: HeterogeneousHandTestSceneCfg = HeterogeneousHandTestSceneCfg(
        num_envs=9,
        env_spacing=0.75,
        replicate_physics=False,
        clone_in_fabric=False,
    )
    viewer: ViewerCfg = ViewerCfg()
    sim: SimulationCfg = SimulationCfg(
        physics_material=RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
        ),
    )

    observations: HeterogeneousHandTestObservationsCfg = HeterogeneousHandTestObservationsCfg()
    actions: HeterogeneousHandTestActionsCfg = HeterogeneousHandTestActionsCfg()
    rewards: HeterogeneousHandTestRewardsCfg = HeterogeneousHandTestRewardsCfg()
    terminations: HeterogeneousHandTestTerminationsCfg = HeterogeneousHandTestTerminationsCfg()
    commands = None
    curriculum = None

    def __post_init__(self) -> None:
        r"""设置随机可视化 smoke 的仿真时序。"""

        super().__post_init__()
        self.decimation = 4
        self.episode_length_s = 4.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (2.0, 2.0, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.4)


__all__ = [
    "DEFAULT_HETEROGENEOUS_HAND_SET",
    "HeterogeneousHandSetCfg",
    "HeterogeneousHandTestActionsCfg",
    "HeterogeneousHandTestEnvCfg",
    "HeterogeneousHandTestObservationsCfg",
    "HeterogeneousHandTestRewardsCfg",
    "HeterogeneousHandTestSceneCfg",
    "HeterogeneousHandTestTerminationsCfg",
    "HeterogeneousHandVariantCfg",
    "build_heterogeneous_hand_articulation_cfg",
]
