"""post-mutate 联合采样使用的基础分布描述。

这层模块只承载两件事：

1. 用声明式 dataclass 描述某个独立标量采样维度的分布；
2. 在 generator / sampler 层按这些描述真正抽样。

它**故意**不承载任何 hand-specific 语义：

- 不知道某个值是 link length delta 还是 mount translation；
- 不知道某个值最终写回哪根 finger / 哪个 joint；
- 也不直接调用任何 mutator。

这样每个具体 mutator 就可以退化成：

1. 给出一组 `ScalarDistributionCfg`
2. 接收上游采样好的数值，做一次确定性变换
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Literal

from ...asset_base import AssetCfgBase


@dataclass
class ScalarDistributionCfg(AssetCfgBase):
    r"""单个独立标量采样维度的声明式分布配置。

    当前先只支持 post-mutate 已经足够的三类分布：

    - ``fixed``：固定值，不引入随机性
    - ``normal``：高斯分布
    - ``uniform``：均匀分布

    之所以限制在这三类，是因为当前已实现的连续参数变异
    （link scale / limit tweak / mount perturb / tip perturb）
    都能自然落在这个集合里；后续若出现离散或分段分布，再扩充即可。
    """

    kind: Literal["fixed", "normal", "uniform"] = "fixed"
    """分布类型。"""

    value: float = 0.0
    """`fixed` 模式下直接返回的常数值。"""

    mean: float = 0.0
    """`normal` 模式下的均值 $\mu$。"""

    sigma: float = 0.0
    """`normal` 模式下的标准差 $\sigma$。"""

    low: float = 0.0
    """`uniform` 模式下的下界。"""

    high: float = 0.0
    """`uniform` 模式下的上界。"""

    clip_min: float | None = None
    """采样后可选的下界裁剪；为 `None` 时不裁剪。"""

    clip_max: float | None = None
    """采样后可选的上界裁剪；为 `None` 时不裁剪。"""

    def __post_init__(self):
        if self.sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {self.sigma}")
        if self.kind == "uniform" and self.low > self.high:
            raise ValueError(f"uniform low must be <= high, got low={self.low}, high={self.high}")
        if self.clip_min is not None and self.clip_max is not None and self.clip_min > self.clip_max:
            raise ValueError(
                f"clip_min must be <= clip_max, got clip_min={self.clip_min}, clip_max={self.clip_max}"
            )


def sample_scalar_distribution(cfg: ScalarDistributionCfg, *, rng: random.Random | None = None) -> float:
    r"""按声明式分布配置采样一个标量值。

    Args:
        cfg (ScalarDistributionCfg): 当前标量维度的分布描述。
        rng (random.Random | None): 可选随机数发生器；为 `None` 时退回全局 `random`。

    Returns:
        float: 当前维度的采样值。
    """

    random_source = rng or random
    if cfg.kind == "fixed":
        value = float(cfg.value)
    elif cfg.kind == "normal":
        value = float(random_source.gauss(cfg.mean, cfg.sigma))
    else:
        value = float(random_source.uniform(cfg.low, cfg.high))

    if cfg.clip_min is not None:
        value = max(value, float(cfg.clip_min))
    if cfg.clip_max is not None:
        value = min(value, float(cfg.clip_max))
    return value


__all__ = ["ScalarDistributionCfg", "sample_scalar_distribution"]
