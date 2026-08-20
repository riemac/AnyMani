r"""多锚点 Gaussian 隐式场实验的 schema 3 顶层配置。"""

from omegaconf import MISSING

from ..data import HandAssetCatalogCfg
from ..experiment import EmbodimentPretrainCfg

# Dataset manifest 已完整冻结 train/validation/evaluation；实验层不再重复声明 partition。
DATA_CFG = HandAssetCatalogCfg(
    manifest="source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1/ssl.yaml",
    expected_sha256="f1398417888e7c237cbb2583dcf8e9cd10bef7fee792b307c67dfa74fb6e0698",
)


# 各 role 先独立完成配置，再由最高实验 façade 统一装配。
EXPERIMENT = EmbodimentPretrainCfg(
    data=DATA_CFG,
    # 其余 role 暂由既有 Hydra groups 注入，等待后续逐项科研审计。
    method=MISSING,
    trainer=MISSING,
    evaluation=MISSING,
    run=MISSING,
)


__all__ = ["DATA_CFG", "EXPERIMENT"]
