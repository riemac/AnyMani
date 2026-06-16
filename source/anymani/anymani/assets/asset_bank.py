r"""TODO: 资产银行（Asset Bank）顶层 facade 与基础 Schema。

本模块是 `assets` 子项目对外暴露“资产集合接口”的稳定入口。它面向已经由 generator / exporter 物化到磁盘的资产 bundle，例如 pre-made topology 资产、
post-mutate 资产，以及未来可能出现的 object / robot / mixed asset collection。

Asset Bank 的核心职责不是生成资产，而是把“磁盘上已经有什么”整理成下游可消费的声明式集合：
$$
\text{generated bundles}
\rightarrow
\text{path routing / symlink resolution / bundle validation / mesh closure}
\rightarrow
\text{topology grouping / generic selection / generic sampling}
\rightarrow
\text{downstream-neutral asset references}.
$$
边界约定：

- `assets` 负责：资产路径解析、集合聚合、bundle 完整性校验、mesh 引用闭包、通用筛选 / 划分 / 采样机制。
- `tasks/gm` 负责：把已经选中的 hand asset reference 适配成 IsaacLab `ArticulationCfg` / `MultiAssetSpawnerCfg`，并处理异构资产在 env runtime 中的 spawn。
- `distill` 负责：把资产集合接入训练配置、geometry observation、模型输入 schema、teacher / student / heldout 等实验语义。

因此本层应保持 IsaacLab-free、trainer-free：它描述“有哪些资产、选中了哪些资产”，但不描述“这些资产如何进入仿真”或“这些资产服务哪个训练阶段”。
这些会有 distill、tasks/gm 各自负责适配接口。

TOAGENT: 注释不可删，但可根据实际情况润色、重构、精炼、优化
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any


@dataclass
class AssetBankCfg:
    r"""资产银行配置基类。

    这是所有具体 asset bank 配置的最小公共父类。当前不急于在这里放入字段，
    因为 hand asset bank、object asset bank、未来 mixed asset bank 对路径结构、
    sidecar schema、mesh closure 和采样语义的要求可能不同。

    设计意图：

    - 顶层基类只承载“这是一个 asset bank config”的类型锚点；
    - 具体字段应优先落在 `bank/hand_bank.py` 等子模块的专用配置类中；
    - 若未来多个 bank 类型沉淀出真正共享字段，再上移到本基类。
    """


class AssetBank:
    r"""资产银行运行时基类。

    运行时类对应配置类，负责把声明式配置解析成可查询的只读资产索引。所谓“重活”
    主要指路径扫描、软链接解析、bundle 校验、sidecar / mesh 元数据读取、拓扑分组、
    以及通用 selection / sampling 的执行结果缓存。

    这不是训练 runtime，也不是 IsaacLab env runtime。它的输出应是下游中立的资产引用
    或资产选择结果；`tasks/gm` 和 `distill` 再各自把这些结果适配到自身语义中。
    """

    def __init__(self, cfg: AssetBankCfg):
        self.cfg = cfg


if TYPE_CHECKING:
    from .bank.hand_bank import HandBank, HandBankCfg, HandSelection, HandSelectionMode, HandSourceMode
    from .bank.hand_container import HandContainer, HandContainerCfg, HandContainerLike, UrdfMeshRef, UrdfRgba


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "HandBank": ("anymani.assets.bank.hand_bank", "HandBank"),
    "HandBankCfg": ("anymani.assets.bank.hand_bank", "HandBankCfg"),
    "HandSelection": ("anymani.assets.bank.hand_bank", "HandSelection"),
    "HandSelectionMode": ("anymani.assets.bank.hand_bank", "HandSelectionMode"),
    "HandSourceMode": ("anymani.assets.bank.hand_bank", "HandSourceMode"),
    "HandContainer": ("anymani.assets.bank.hand_container", "HandContainer"),
    "HandContainerCfg": ("anymani.assets.bank.hand_container", "HandContainerCfg"),
    "HandContainerLike": ("anymani.assets.bank.hand_container", "HandContainerLike"),
    "UrdfMeshRef": ("anymani.assets.bank.hand_container", "UrdfMeshRef"),
    "UrdfRgba": ("anymani.assets.bank.hand_container", "UrdfRgba"),
}
r"""顶层 facade 的 lazy re-export 表。

这里刻意不用普通 eager import：`bank/hand_bank.py` 需要从本模块导入
`AssetBankCfg` / `AssetBank` 基类；若顶层 facade 又在文件头部急切导入 hand bank，
就会把一个本可很薄的基础层变成循环初始化热点。lazy re-export 同时满足：

- 下游可以使用稳定入口 `anymani.assets.asset_bank.HandBankCfg`；
- 真实实现仍保留在 `assets.bank.*` 子包；
- import `AssetBankCfg` 时不会顺带加载全部 hand-bank scaffold。
"""


def __getattr__(name: str) -> Any:
    r"""Lazy-load concrete asset-bank symbols exposed by this facade.

    Args:
        name (str): 需要从 facade 读取的符号名。

    Returns:
        Any: 对应子模块中的真实对象。

    Raises:
        AttributeError: 当符号不属于本 facade 的公开 contract 时抛出。
    """

    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    return getattr(import_module(module_name), attr_name)


__all__ = [
    "AssetBankCfg",
    "AssetBank",
    "HandBank",
    "HandBankCfg",
    "HandSelection",
    "HandSelectionMode",
    "HandSourceMode",
    "HandContainer",
    "HandContainerCfg",
    "HandContainerLike",
    "UrdfMeshRef",
    "UrdfRgba",
]
