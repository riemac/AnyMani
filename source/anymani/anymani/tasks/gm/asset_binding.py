r"""Bind one generated hand asset into the `gm` task scene.

这一层不是 asset bank 管理器。它只描述“给定一个已经由 `distill`
选中的 hand asset，`tasks/gm` 需要怎样把它变成 Isaac Lab 的
`ArticulationCfg`”。

最低输入 contract：

- `hand.urdf`：当前 hand 的可加载 URDF，mesh 路径应相对自身目录闭合；
- `hand.yaml`：sidecar 元数据，至少包含 `family`、`handedness`、`dof`、
  `finger_count`、`topology_name`、`surviving_slots`、`slot_family_map`、
  `per_finger_connectivity`；
- same-topology RL 主线要求一批资产共享 action joint schema。跨拓扑 padding
  / mask / token 化是后续 `distill/models` 问题，不在本文件解决。

TODO:
    后续实现 `build_hand_articulation_cfg(...)` 时，应只做薄绑定：
    路径校验、URDF importer 参数、初始位姿、actuator 默认值、joint order
    contract 提取。不要在这里扫描 generated root，也不要决定 train/heldout split。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


@dataclass(frozen=True)
class GmHandAssetRef:
    r"""A selected generated hand asset consumed by `gm`.

    Args:
        root_dir (Path): 单个 hand bundle 目录；通常包含 `hand.urdf` 与 `hand.yaml`。
        topology_name (str | None): 可选拓扑名，用于训练 manifest 与日志核对。
        asset_id (str | None): 可选样本 ID；post-mutate 样本通常有短 hash。

    NOTE:
        这个 dataclass 表达的是“已选资产引用”，不是“资产库”。如果未来需要
        64 / 128 个 assets 的采样、分段训练或 heldout eval，应在 `distill`
        侧生成一组 `GmHandAssetRef`，再逐段交给环境配置。
    """

    root_dir: Path
    topology_name: str | None = None
    asset_id: str | None = None

    @property
    def urdf_path(self) -> Path:
        r"""Return the expected URDF path for the selected hand."""

        return self.root_dir / "hand.urdf"

    @property
    def sidecar_path(self) -> Path:
        r"""Return the expected sidecar metadata path for the selected hand."""

        return self.root_dir / "hand.yaml"


def build_hand_articulation_cfg(asset: GmHandAssetRef, *, prim_path: str) -> ArticulationCfg:
    r"""Build an Isaac Lab articulation cfg for one selected generated hand.

    这是轻量 scaffold，不是最终实现。函数保留签名是为了固定后续数据流：

    $$
    \texttt{GmHandAssetRef} \rightarrow \texttt{ArticulationCfg}
    \rightarrow \texttt{GmInHandSceneCfg.robot}
    $$

    Args:
        asset (GmHandAssetRef): `distill` 或 debug cfg 已经选好的单个 hand bundle。
        prim_path (str): Isaac Lab scene 中 robot articulation 的 prim path。

    Returns:
        ArticulationCfg: 可被 `scene.robot` 消费的 articulation config。

    Raises:
        NotImplementedError: 当前阶段只打脚手架，避免假装环境已可运行。

    TODO:
        实现时应使用 `sim_utils.UrdfFileCfg(asset_path=..., fix_base=True, ...)`
        或先离线转 USD 后用 `UsdFileCfg`。若追求 4096/8192 env 并行，建议优先
        评估离线 USD cache；在线 URDF conversion 只适合早期 debug。
    """

    _ = asset
    _ = prim_path
    _ = sim_utils
    _ = ImplicitActuatorCfg
    raise NotImplementedError("gm hand asset binding is scaffolded but not implemented yet.")


__all__ = [
    "GmHandAssetRef",
    "build_hand_articulation_cfg",
]
