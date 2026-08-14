r"""Geometry SSL 的声明式实验组合。

实验模块拥有完整科学语义组合；``ssl.runtime`` 只执行已 resolve 的资产、q、window、optimizer
和 checkpoint 轴，不在运行时猜测模型或损失。当前正式候选是 21-asset 同 topology canonical
residual family pilot。
"""

from anymani.distill.ssl.config import GeometrySSLExperimentCfg

from .canonical_residual_family import (
    FORMAL_FAMILY,
    FORMAL_MOTHER,
    FORMAL_VARIANT_IDS,
    MOTHER_ASSET_ID,
    canonical_residual_family_experiment,
)


def experiment_configs() -> dict[str, GeometrySSLExperimentCfg]:
    r"""返回 Hydra ConfigStore 使用的完整实验名到 frozen config 映射。"""

    return {
        "geometry_ssl_canonical_residual_family": canonical_residual_family_experiment(),
    }

__all__ = [
    "FORMAL_FAMILY",
    "FORMAL_MOTHER",
    "FORMAL_VARIANT_IDS",
    "MOTHER_ASSET_ID",
    "canonical_residual_family_experiment",
    "experiment_configs",
]
