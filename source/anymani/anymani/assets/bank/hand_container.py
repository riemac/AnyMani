r"""单个 generated hand asset 的虚拟标准视图。

本模块只描述“一个 hand asset container 长什么样”，不负责资产集合发现、随机采样、
train / heldout split，也不负责 IsaacLab spawn。它是 `HandBank` 输出给下游的最小
资产单元。

设计目标：把当前真实产物目录

```text
<post_mutate_root>/
  meshes/<mesh-file>.obj|stl
  <sample_id>/
    hand.urdf
    hand.yaml
```

以及 pre-made topology 根目录，统一呈现为“作伪”的虚拟标准 bundle：

```text
<virtual-hand-bundle>/
  hand.urdf
  hand.yaml
  meshes/<mesh-file>.obj|stl
```

该视图不复制文件、不创建软链接、不读取 mesh 几何内容，只维护
`virtual path <-> real path` 的双射。下游只需声明消费虚拟标准视图，不再重复处理
post-mutate run root 与 shared meshes 的历史目录细节。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, TypeAlias

import yaml

from ..asset_schema_geometry import HandGeometrySemanticsCfg
from .geometry_semantics import HandAssetSourceKind, resolve_hand_geometry_semantics
from .path_utils import resolve_container_entry_path

UrdfRgba = tuple[float, float, float, float]
"""URDF `<color rgba="r g b a"/>` 的四通道颜色表示，通道通常在 $[0,1]$。"""

@dataclass(frozen=True)
class UrdfMeshRef:
    r"""URDF mesh 引用的轻量路径记录。

    该结构只记录路径，不读取 mesh 顶点、面片或物理几何内容。它服务于两个目的：

    1. 保留 URDF 中原始 `<mesh filename="..."/>`，便于报错和反查；
    2. 记录该 URI 在当前 hand container 虚拟视图下对应的真实磁盘路径。
    """

    raw_uri: str
    """URDF 内原始 mesh filename，例如 `../meshes/cs_tip_xxx.obj`。"""

    virtual_path: PurePosixPath
    """虚拟标准视图中的 mesh 路径，例如 `meshes/cs_tip_xxx.obj`。"""

    real_path: Path
    """解析到本机文件系统的真实 mesh 路径。"""


@dataclass
class HandContainerCfg:
    r"""单个 hand asset 的声明式入口。

    这个 cfg 是 `HandBankCfg.containers` 中的一项，面向用户 / manifest 手写使用。
    它不要求用户列出 sidecar、mesh 或 visual material 信息；这些由 `HandBank`
    在运行时解析成 `HandContainer`。
    """

    path: str | Path
    r"""单个 hand asset 的入口路径。

    允许三类输入：

    - post-mutate sample 目录，如 `1314999e`；
    - 绝对或相对 bundle 目录，如 `/.../1314999e`；
    - 具体 URDF 文件，如 `/.../1314999e/hand.urdf`。

    相对路径的基准由 `HandBank` 根据 resolved source root 决定，而不是由本 cfg
    在 `__post_init__` 中自行做文件 IO。
    """

    asset_id: str | None = None
    """可选资产 ID；未提供时后续运行时优先从 sidecar `id` 推断，再退回目录名。"""

    urdf_file: str = "hand.urdf"
    """当 `path` 指向目录时，目录内默认 URDF 文件名。当前 generated contract 固定为 `hand.urdf`。"""

    sidecar_file: str = "hand.yaml"
    """当 `path` 指向目录时，目录内默认 sidecar 文件名。当前 generated contract 固定为 `hand.yaml`。"""

    source_kind: HandAssetSourceKind = "generated"
    """资产来源类型；official 缺人工核验的几何语义时不得使用 generated 迁移规则。"""

    topology_key: str | None = None
    """可选 morphology 身份；未提供时几何语义解析器读取 sidecar ``topology_name``。"""


HandContainerLike: TypeAlias = str | Path | HandContainerCfg
"""用户可写的单 hand asset 入口；字符串 / Path 会规范化为 `HandContainerCfg(path=...)`。"""


def coerce_hand_container_cfg(value: HandContainerLike) -> HandContainerCfg:
    r"""把用户简写的 container 入口规范化为 `HandContainerCfg`。

    Args:
        value (HandContainerLike): 可以是已有 cfg、sample id 字符串、相对/绝对路径。

    Returns:
        HandContainerCfg: 统一后的声明式单资产入口。
    """

    if isinstance(value, HandContainerCfg):
        return value
    if isinstance(value, str | Path):
        return HandContainerCfg(path=value)
    raise TypeError(f"Unsupported hand container entry: {value!r}")


@dataclass(frozen=True)
class HandContainer:
    r"""一个 resolved hand asset 的虚拟标准视图。

    `HandContainer` 是资产银行对下游暴露的最小资产单元。它不是 IsaacLab wrapper，
    也不是训练样本；它只是把真实磁盘上的 URDF / sidecar / shared meshes 组织成
    一个下游中立的虚拟 bundle。
    """

    asset_id: str
    """稳定资产 ID；post-mutate 主线通常等于 8 位 sample hash。"""

    virtual_to_real: dict[PurePosixPath, Path]
    """虚拟标准路径到真实磁盘路径的映射，如 `meshes/x.obj -> /abs/.../meshes/x.obj`。"""

    real_to_virtual: dict[Path, PurePosixPath]
    """真实磁盘路径到虚拟标准路径的反向映射；用于日志、manifest 和错误定位。"""

    sidecar: dict[str, Any] = field(default_factory=dict)
    """解析后的 `hand.yaml` 内容。第一版先保持 `dict`，避免过早固化完整 schema。"""

    source_kind: HandAssetSourceKind = "generated"
    """当前 container 的资产来源类型。"""

    geometry_semantics: HandGeometrySemanticsCfg | None = None
    """按需解析的静态几何语义；tasks 默认不要求，distill 必须显式要求。"""

    mesh_refs: tuple[UrdfMeshRef, ...] = ()
    """从 URDF `<mesh filename=...>` 解析出的 mesh 路径引用表。"""

    visual_rgba_by_name: dict[str, UrdfRgba] = field(default_factory=dict)
    """URDF named visual 到 RGBA debug color 的映射，供可视化适配层选择性消费。"""

    @classmethod
    def from_cfg(
        cls,
        cfg: HandContainerCfg,
        *,
        source_root: Path | None = None,
        require_sidecar: bool = True,
        validate_mesh_relpaths: bool = True,
        parse_visual_rgba: bool = True,
        require_geometry_semantics: bool = False,
        allow_legacy_left_handedness: bool = False,
    ) -> HandContainer:
        r"""把单资产声明式入口解析为虚拟标准视图。

        Args:
            cfg (HandContainerCfg): 单个 hand asset 的用户入口。
            source_root (Path | None): 相对 sample id 的解析基准，通常为 post-mutate run root。
            require_sidecar (bool): 是否要求 `hand.yaml` 必须存在。
            validate_mesh_relpaths (bool): 是否要求 URDF mesh 引用全部闭合到真实文件。
            parse_visual_rgba (bool): 是否解析 named visual 的 RGBA debug color。
            require_geometry_semantics (bool): 是否解析/迁移类型化几何语义；official 缺字段时严格拒绝。
            allow_legacy_left_handedness (bool): 是否显式允许缺少严格镜像证书的
                generated left；默认 ``False``，只供历史审计。

        Returns:
            HandContainer: 下游中立的虚拟标准 bundle。
        """

        entry_path = resolve_container_entry_path(cfg.path, source_root=source_root)
        urdf_path, sidecar_path = _resolve_urdf_and_sidecar_paths(
            entry_path,
            urdf_file=cfg.urdf_file,
            sidecar_file=cfg.sidecar_file,
        )
        if not urdf_path.is_file():
            raise FileNotFoundError(f"hand URDF does not exist: {urdf_path}")
        sidecar = _load_sidecar(sidecar_path, require_sidecar=require_sidecar)
        _validate_generated_handedness_contract(
            sidecar,
            source_kind=cfg.source_kind,
            allow_legacy_left_handedness=allow_legacy_left_handedness,
        )  # 在解析 mesh/geometry 前 fail-fast，避免无效 left bundle进入下游容器

        # 延迟导入，避免 `urdf_utils` 与本模块的类型定义形成 import-time 循环。
        from .urdf_utils import parse_urdf_mesh_refs, parse_urdf_visual_rgba_by_name

        mesh_refs = parse_urdf_mesh_refs(urdf_path, require_existing=validate_mesh_relpaths)
        visual_rgba_by_name = parse_urdf_visual_rgba_by_name(urdf_path) if parse_visual_rgba else {}
        asset_id = str(cfg.asset_id or sidecar.get("id") or urdf_path.parent.name)
        geometry_semantics = (
            resolve_hand_geometry_semantics(
                sidecar,
                source_kind=cfg.source_kind,
                asset_id=asset_id,
                topology_key=cfg.topology_key,
            )
            if require_geometry_semantics
            else None
        )
        virtual_to_real, real_to_virtual = _build_virtual_path_bijection(
            urdf_path=urdf_path,
            sidecar_path=sidecar_path if sidecar_path.is_file() else None,
            mesh_refs=mesh_refs,
        )
        return cls(
            asset_id=asset_id,
            virtual_to_real=virtual_to_real,
            real_to_virtual=real_to_virtual,
            sidecar=sidecar,
            source_kind=cfg.source_kind,
            geometry_semantics=geometry_semantics,
            mesh_refs=mesh_refs,
            visual_rgba_by_name=visual_rgba_by_name,
        )


    @property
    def urdf_path(self) -> Path:
        r"""虚拟标准 `hand.urdf` 对应的真实路径。"""

        return self.real_path("hand.urdf")

    @property
    def sidecar_path(self) -> Path:
        r"""虚拟标准 `hand.yaml` 对应的真实路径。"""

        return self.real_path("hand.yaml")

    def real_path(self, virtual_path: str | PurePosixPath) -> Path:
        r"""把虚拟标准路径映射到真实磁盘路径。

        Raises:
            NotImplementedError: 当前只落 scaffold，后续实现虚拟路径查表与错误消息。
        """

        key = _normalize_virtual_path(virtual_path)
        try:
            return self.virtual_to_real[key]
        except KeyError as exc:
            raise KeyError(f"unknown virtual hand asset path {str(key)!r} for asset {self.asset_id!r}") from exc

    def virtual_path(self, real_path: str | Path) -> PurePosixPath:
        r"""把真实磁盘路径映射回虚拟标准路径。

        Raises:
            NotImplementedError: 当前只落 scaffold，后续实现反向查表与路径规范化。
        """

        key = Path(real_path).expanduser().resolve(strict=False)
        try:
            return self.real_to_virtual[key]
        except KeyError as exc:
            raise KeyError(f"real path {key} is not part of hand asset {self.asset_id!r}") from exc


def _validate_generated_handedness_contract(
    sidecar: dict[str, Any],
    *,
    source_kind: HandAssetSourceKind,
    allow_legacy_left_handedness: bool,
) -> None:
    r"""拒绝缺少严格整手镜像证书的 generated left bundle。

    安全门只作用于 ``source_kind="generated"`` 且顶层 ``handedness="left"``。
    新合同要求：

    - ``version == HANDEDNESS_CONTRACT_VERSION``；
    - canonical 真源为 ``right``，目标为 ``left``；
    - 反射平面为 ``palm_yz``；
    - ``same_q`` 与 ``physical_lowering_complete`` 均为真。

    Args:
        sidecar: 已解析的 ``hand.yaml`` 顶层 mapping。
        source_kind: ``generated`` 或 ``official`` 的权威来源边界。
        allow_legacy_left_handedness: 历史审计用显式 override。

    Raises:
        ValueError: generated left 缺少或伪造/损坏严格合同，且未显式 override。
    """

    if source_kind != "generated":
        return  # official 资产由自身人工合同管理，不套用 generated lowering 规则

    from ..handedness import validate_generated_handedness_contract  # 延迟导入，保持 bank 初始化轻量

    validate_generated_handedness_contract(
        sidecar,
        allow_legacy_left_handedness=allow_legacy_left_handedness,
    )  # Bank 与 mutate-only restore 共享版本/字段真源


def _resolve_urdf_and_sidecar_paths(entry_path: Path, *, urdf_file: str, sidecar_file: str) -> tuple[Path, Path]:
    r"""从目录或 URDF 文件入口推出 `hand.urdf` 与 `hand.yaml` 真实路径。"""

    if entry_path.suffix == ".urdf":
        urdf_path = entry_path.resolve(strict=False)
        bundle_dir = urdf_path.parent
    else:
        bundle_dir = entry_path.resolve(strict=False)
        urdf_path = (bundle_dir / urdf_file).resolve(strict=False)
    return urdf_path, (bundle_dir / sidecar_file).resolve(strict=False)


def _load_sidecar(sidecar_path: Path, *, require_sidecar: bool) -> dict[str, Any]:
    r"""读取 sidecar YAML；缺失时按配置选择失败或返回空 dict。"""

    if not sidecar_path.is_file():
        if require_sidecar:
            raise FileNotFoundError(f"hand sidecar does not exist: {sidecar_path}")
        return {}
    data = yaml.safe_load(sidecar_path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"hand sidecar must be a YAML mapping: {sidecar_path}")
    return data


def _build_virtual_path_bijection(
    *,
    urdf_path: Path,
    sidecar_path: Path | None,
    mesh_refs: tuple[UrdfMeshRef, ...],
) -> tuple[dict[PurePosixPath, Path], dict[Path, PurePosixPath]]:
    r"""建立虚拟标准 bundle 路径与真实磁盘路径的双射。"""

    virtual_to_real: dict[PurePosixPath, Path] = {}
    real_to_virtual: dict[Path, PurePosixPath] = {}
    _add_virtual_mapping(virtual_to_real, real_to_virtual, PurePosixPath("hand.urdf"), urdf_path)
    if sidecar_path is not None:
        _add_virtual_mapping(virtual_to_real, real_to_virtual, PurePosixPath("hand.yaml"), sidecar_path)
    for mesh_ref in mesh_refs:
        _add_virtual_mapping(virtual_to_real, real_to_virtual, mesh_ref.virtual_path, mesh_ref.real_path)
    return virtual_to_real, real_to_virtual


def _add_virtual_mapping(
    virtual_to_real: dict[PurePosixPath, Path],
    real_to_virtual: dict[Path, PurePosixPath],
    virtual_path: PurePosixPath,
    real_path: Path,
) -> None:
    r"""向双射表加入一条路径映射，并拒绝非双射冲突。"""

    normalized_virtual = _normalize_virtual_path(virtual_path)
    normalized_real = Path(real_path).expanduser().resolve(strict=False)
    existing_real = virtual_to_real.get(normalized_virtual)
    if existing_real is not None and existing_real != normalized_real:
        raise ValueError(
            f"virtual hand asset path {normalized_virtual} maps to both {existing_real} and {normalized_real}"
        )
    existing_virtual = real_to_virtual.get(normalized_real)
    if existing_virtual is not None and existing_virtual != normalized_virtual:
        raise ValueError(
            f"real hand asset path {normalized_real} maps to both {existing_virtual} and {normalized_virtual}"
        )
    virtual_to_real[normalized_virtual] = normalized_real
    real_to_virtual[normalized_real] = normalized_virtual


def _normalize_virtual_path(path: str | PurePosixPath) -> PurePosixPath:
    r"""规范化虚拟 bundle 路径，拒绝绝对路径和父目录逃逸。"""

    virtual_path = PurePosixPath(path)
    if virtual_path.is_absolute() or ".." in virtual_path.parts:
        raise ValueError(f"virtual hand asset path must stay inside the virtual bundle: {path!r}")
    return virtual_path


__all__ = [
    "HandContainerLike",
    "HandContainer",
    "HandContainerCfg",
    "UrdfMeshRef",
    "UrdfRgba",
    "coerce_hand_container_cfg",
]
