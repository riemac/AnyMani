r"""后序变异工具的公共基础协议。

本模块只定义 `MutatorBase`——一个最小协议类，供所有后序变异工具壳继承。
它不承载任何配置逻辑，只确保所有工具都能被 `HandMutator` 流水线以统一接口
调度。
"""

from __future__ import annotations

from typing import Any

from ...asset_base import HandCfg


# TODO: post-mutator 所包含的通用属性，后续变异配置类都复用
@dataclass
class MutatorBaseCfg(AssetCfgBase):
    """所有后序变异算子配置的最小公共基类。

    为 MutatorTerm.cfg 提供类型收窄——
    所有合法的 mutator cfg 都必须有 class_type，
    而 class_type 的内部复杂行为由各子类自行掌控。

    各子类负责自己声明：
    - 变异作用于 HandCfg 的哪些属性路径
    - 采用什么分布及裁剪约束
    - per-entity 精调机制（如 per_joint_delta_distribution）
    - 统一暴露什么接口给用户配置使用
        - 一个约定，内部属性用 "_" 下划线前缀，用户配置接口后，可由 `__post_init__()` 解析
        - 用户好友的对外接口属性则不用 "_"
    """
    class_type: type["MutatorBase"] | None = None


class MutatorBase:
    r"""所有后序变异算子的最小基类。

    位于 Sample 层级，负责：读原始 cfg 和 HandCfg 相关属性，返回要修改的属性，以及采样到的参数，后续交由 HandMutator 并行 apply
    """

    cfg: MutatorBaseCfg

    def __init__(self, cfg: MutatorBaseCfg) -> None:
        self.cfg = cfg


__all__ = ["MutatorBase"]
