r"""手资产银行（Hand Asset Bank）的集合与选择实现。

本模块承接 `assets.asset_bank` 的通用 Asset Bank façade，专门描述 generated hand
资产集合的声明式配置、运行时索引、以及单个 hand URDF bundle 的“虚拟标准视图”。

调研当前 generator / exporter 后，真实产物 contract 暂定如下：

```text
# pre-made topology 根；它本身就是一个可消费的 hand bundle，且与后变异样本同级进入 bank
generated/<premade_timestamp>/<group>/<topology>/
  hand.urdf
  hand.yaml
  tree.txt
  meshes/

# independent post-mutate run 根
generated/<premade_timestamp>/<group>/<topology>/<mutate_timestamp>/
  summary.yaml
  meshes/
  <sample_id>/
    hand.urdf
    hand.yaml
    tree.txt
```

`HandContainer` 的定义放在 `hand_container.py`。本文件只描述“如何从资产来源中
发现、解析并选择一组 hand containers”。对于 `source_mode="post_mutate"`，这里默认把
母体 pre-made topology 与 run root 下的 post-mutate leaf samples 一起拉平成同级候选；
单个 container 对外呈现一个“作伪”的虚拟标准 bundle 视图：

```text
<virtual-hand-bundle>/
  hand.urdf
  hand.yaml
  meshes/<mesh-file>.obj|stl
```

该虚拟视图不复制文件、不创建软链接、不把 mesh 几何读进内存，只维护
`virtual path <-> real path` 的双射。母体 topology 自身的 `meshes/...` 按 topology 根解析；
post-mutate leaf 的 `../meshes/...` 按 run root 解析。下游 `tasks/gm` 和 `distill` 只消费
这个标准视图，再各自适配 IsaacLab spawn、geometry observation、训练 manifest 等语义。

当前实现边界：

- `source_mode="post_mutate"` 已落地，默认包含 source topology 母体；
- `source_mode="pre_made"` / `"mixed"` 先保留接口，resolve 时显式报
  `NotImplementedError`；其中 `mixed` 指 Leap / Allegro 等跨形态拓扑产物组织，
  不表示“母体 + post-mutate variants”的同源 topology 内部拉平；
- `selection_mode="explicit"` 服务验证阶段的手写 URDF / bundle 列表；
- `selection_mode="sample"` 服务 teacher training 的固定 seed 随机子集；
- `selection_mode="all"` 服务整批 smoke / 统计检查；
- 文件 IO、XML/YAML 解析、采样算法和虚拟路径映射均集中在 `HandBank.resolve()` 阶段，
  cfg 构造阶段仍保持无 IO。

TOAGENT: 注释不可删，但可根据实际情况润色、重构、精炼、优化。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..asset_bank import AssetBank, AssetBankCfg
from .hand_container import HandContainer, HandContainerCfg, HandContainerLike, coerce_hand_container_cfg
from .path_utils import resolve_post_mutate_root

HandSourceMode = Literal["post_mutate", "pre_made", "mixed"]
"""手资产来源模式。

`post_mutate` 对应当前 teacher specialist policy 的主线资产；`pre_made` 与
`mixed` 暂作为未来扩展锚点保留。注意 `mixed` 是跨手型 / 跨 family 产物组织语义，
不是“pre-made 母体 + post-mutate leaf”的集合包含关系。
"""

HandSelectionMode = Literal["explicit", "sample", "all"]
"""手资产选择模式。

`explicit` 用于人工列出少量验证资产；`sample` 用于固定 seed 的训练子集；
`all` 用于整批检查或无需抽样的 smoke。
"""

@dataclass(frozen=True)
class HandSelection:
    r"""一次 hand bank resolve 后的选择结果。

    该结构是下游 `tasks/gm` / `distill` 消费的批量入口。它记录“选中了哪些资产”
    以及足以复现选择过程的轻量元数据，但不包含 IsaacLab env runtime 或训练状态。
    """

    assets: tuple[HandContainer, ...]
    """已选中的 hand containers，顺序应与后续 env-id / asset-id routing 保持一致。"""

    source_mode: HandSourceMode
    """本次选择使用的资产来源模式。"""

    selection_mode: HandSelectionMode
    """本次选择使用的资产筛选 / 采样模式。"""

    sample_seed: int | None = None
    """若 `selection_mode="sample"`，记录复现采样的 seed；否则可为 `None`。"""

    source_root: Path | None = None
    """本次选择对应的 resolved source root；explicit 绝对路径且无 source 时可为 `None`。"""


@dataclass
class HandBankCfg(AssetBankCfg):
    r"""手资产银行配置类。

    这是下游用户主要面对的 façade：它声明从哪里读取 generated hand assets，以及
    如何从候选集合中选择一批 hand containers。具体路径解析、目录扫描、YAML/XML
    解析和采样执行均由运行时 `HandBank.resolve()` 承担。
    """

    class_type: type[HandBank] | None = None
    """关联的运行时类；默认在 `__post_init__` 中设为 `HandBank`。"""

    source_mode: HandSourceMode = "post_mutate"
    """资产来源模式。当前只实现 `post_mutate`。"""

    selection_mode: HandSelectionMode = "explicit"
    """资产选择模式：手写列表、固定 seed 随机采样、或全量选择。"""

    pre_made_path: str | Path | None = None
    r"""pre-made topology 根或其上级路径。

    该字段用于两类情况：

    - 未来 `source_mode="pre_made"`；
    - 当前 `post_mutate_path` 只给 timestamp / run name 时，与其组合成 post-mutate run root。

    第一版不把路径强行绑定到 `assets/generated`，允许普通相对路径或绝对路径。
    """

    post_mutate_path: str | Path | None = None
    r"""post-mutate run 根路径，或在给定 `pre_made_path` 时的 run 名称。

    典型真实结构为：
    `.../<topology>/<post_mutate_timestamp>/`，其中直接子目录 `1314999e/` 等为
    sample bundle，run 根下的 `meshes/` 为共享 mesh 边界。
    """

    post_mutate_name: str | None = None
    """post-mutate run 名称的显式字段；当不想重载 `post_mutate_path` 语义时使用。"""

    include_source_topology: bool = True
    """是否把 post-mutate run 的母体 pre-made topology 作为同级候选纳入集合。

    当该字段为 `True` 时，`post_mutate_path.parent` 必须是包含 `hand.urdf` 的 source
    topology 根目录；母体的 mesh URI 继续按该 topology 根解析，而不是按 run root 解析。
    """

    containers: tuple[HandContainerLike, ...] = ()
    """`selection_mode="explicit"` 时用户手写的单资产入口列表。

    为了让声明式配置不被 `HandContainerCfg(path=...)` 重复淹没，本字段允许直接写：

    ```python
    containers=("066b6272", "0bdf0eca", "00d68163")
    ```

    `__post_init__` 会把字符串 / Path 统一 lower 成 `HandContainerCfg`。
    """

    sample_count: int | None = None
    """`selection_mode="sample"` 时需要抽取的资产数量。"""

    sample_seed: int = 0
    """`selection_mode="sample"` 的可复现随机种子。"""

    require_sidecar: bool = True
    """是否要求每个 hand container 必须有 `hand.yaml`。默认必需。"""

    validate_mesh_relpaths: bool = True
    """是否检查 URDF 中所有 mesh filename 都能闭合解析到真实文件。"""

    parse_visual_rgba: bool = True
    """是否解析 URDF visual material color，供后续可视化 / material restore adapter 使用。"""

    def __post_init__(self) -> None:
        r"""只做无 IO 的 cfg 归一化。

        注意：不要在配置对象初始化阶段扫描目录或读取 `hand.urdf` / `hand.yaml`。
        文件系统相关解析应留给 `HandBank.resolve()`。
        """

        if self.class_type is None:
            self.class_type = HandBank
        self.containers = tuple(coerce_hand_container_cfg(container) for container in self.containers)


class HandBank(AssetBank):
    r"""手资产银行运行时索引器。

    `HandBank` 消费 `HandBankCfg`，在显式调用 `resolve()` 时执行候选发现、路径解析、
    虚拟视图构建、bundle 校验与 selection 记录。构造函数本身不做 IO。
    """

    cfg: HandBankCfg

    def __init__(self, cfg: HandBankCfg):
        r"""保存 cfg；不扫描目录、不读取文件。"""

        super().__init__(cfg)

    def resolve(self) -> HandSelection:
        r"""解析配置并返回 hand selection。

        Raises:
            NotImplementedError: 当 `source_mode` 不是当前已实现的 `post_mutate` 时抛出。
        """

        if self.cfg.source_mode != "post_mutate":
            raise NotImplementedError(f"HandBank source_mode={self.cfg.source_mode!r} is not implemented yet")
        candidates = self.discover()
        return self.select(candidates)

    def discover(self) -> tuple[HandContainer, ...]:
        r"""发现当前 source root 下的候选 hand containers。

        对 `source_mode="post_mutate"`，候选集合默认等于：

        $$
        \mathcal{A}=\{\text{source topology}\}\cup\{\text{post-mutate leaf samples}\}.
        $$

        这里的“同级”是虚拟集合语义，不要求真实目录同级：母体在 run root 父目录，
        leaf samples 在 run root 子目录。

        Raises:
            FileNotFoundError: 当 post-mutate run root 或必需的 source topology bundle 不存在时抛出。
        """

        source_root = self._resolve_optional_source_root()
        if self.cfg.selection_mode == "explicit":
            if not self.cfg.containers:
                raise ValueError("selection_mode='explicit' requires at least one HandContainerCfg")
            return tuple(self._container_from_cfg(container_cfg, source_root=source_root) for container_cfg in self.cfg.containers)

        if source_root is None:
            raise ValueError(f"selection_mode={self.cfg.selection_mode!r} requires a post_mutate source root")
        if not source_root.is_dir():
            raise FileNotFoundError(f"post-mutate source root does not exist: {source_root}")
        candidates = []
        if self.cfg.include_source_topology:
            # 母体 topology 与 run root 的 leaf samples 在虚拟视图中同级；这里只把它当作普通候选。
            source_topology_root = source_root.parent  # 母体 topology 根；其 `meshes/` 与 run root 分离
            if not self._has_hand_bundle_contract(source_topology_root):
                raise FileNotFoundError(
                    "include_source_topology=True requires a source topology bundle at "
                    f"post_mutate_path.parent: {source_topology_root / 'hand.urdf'}"
                )
            candidates.append(
                self._container_from_cfg(HandContainerCfg(path=source_topology_root), source_root=None)
            )
        # run root 下的每个 sample 目录仍按 post-mutate leaf bundle 解析；它们共享 run root/meshes。
        candidates.extend(
            self._container_from_cfg(HandContainerCfg(path=child), source_root=None)
            for child in source_root.iterdir()
            if child.is_dir() and (child / "hand.urdf").is_file()
        )
        return tuple(sorted(candidates, key=lambda container: container.asset_id))

    def select(self, candidates: tuple[HandContainer, ...]) -> HandSelection:
        r"""从候选 containers 中执行 explicit / sample / all 选择。

        Args:
            candidates (tuple[HandContainer, ...]): 已解析、已校验的候选资产。

        Raises:
            ValueError: 当 `sample_count` 越界或 `selection_mode` 未知时抛出。
        """

        source_root = self._resolve_optional_source_root()
        if self.cfg.selection_mode == "explicit":
            return HandSelection(
                assets=candidates,
                source_mode=self.cfg.source_mode,
                selection_mode=self.cfg.selection_mode,
                sample_seed=None,
                source_root=source_root,
            )
        if self.cfg.selection_mode == "all":
            return HandSelection(
                assets=tuple(sorted(candidates, key=lambda container: container.asset_id)),
                source_mode=self.cfg.source_mode,
                selection_mode=self.cfg.selection_mode,
                sample_seed=None,
                source_root=source_root,
            )
        if self.cfg.selection_mode == "sample":
            if self.cfg.sample_count is None:
                raise ValueError("selection_mode='sample' requires sample_count")
            if self.cfg.sample_count < 0:
                raise ValueError("sample_count must be non-negative")
            sorted_candidates = tuple(sorted(candidates, key=lambda container: container.asset_id))
            if self.cfg.sample_count > len(sorted_candidates):
                raise ValueError(
                    f"sample_count={self.cfg.sample_count} exceeds available hand assets={len(sorted_candidates)}"
                )
            selected = tuple(random.Random(self.cfg.sample_seed).sample(sorted_candidates, self.cfg.sample_count))
            return HandSelection(
                assets=selected,
                source_mode=self.cfg.source_mode,
                selection_mode=self.cfg.selection_mode,
                sample_seed=self.cfg.sample_seed,
                source_root=source_root,
            )
        raise ValueError(f"unknown HandBank selection_mode: {self.cfg.selection_mode!r}")

    def _resolve_optional_source_root(self) -> Path | None:
        r"""解析 post-mutate source root；explicit 绝对路径场景允许缺省。"""

        try:
            return resolve_post_mutate_root(self.cfg)
        except ValueError:
            if self.cfg.selection_mode == "explicit" and all(Path(cfg.path).expanduser().is_absolute() for cfg in self.cfg.containers):
                return None
            raise

    def _container_from_cfg(self, cfg: HandContainerCfg, *, source_root: Path | None) -> HandContainer:
        r"""按当前 bank 解析选项构造单个 `HandContainer`。"""

        return HandContainer.from_cfg(
            cfg,
            source_root=source_root,
            require_sidecar=self.cfg.require_sidecar,
            validate_mesh_relpaths=self.cfg.validate_mesh_relpaths,
            parse_visual_rgba=self.cfg.parse_visual_rgba,
        )

    @staticmethod
    def _has_hand_bundle_contract(candidate_root: Path) -> bool:
        r"""判断目录是否具备一个可作为 hand bundle 的最小 contract。"""

        return (candidate_root / "hand.urdf").is_file()


__all__ = [
    "HandBank",
    "HandBankCfg",
    "HandSelection",
    "HandSelectionMode",
    "HandSourceMode",
]
