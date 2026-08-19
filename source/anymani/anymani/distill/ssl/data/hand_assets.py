r"""HandAssetDataset 到平等 embodiment catalog 的 data role。

Data role 只回答“本次实验有哪些资产以及它们属于哪个 partition”。它不采样 $q$，不生成 query、
sigma 或监督目标，也不把 mother/family 等 provenance 转成隐藏训练权重。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from anymani.assets.bank.dataset import HandAssetDataset, ResolvedHandAssetDataset
from anymani.assets.bank.hand_container import HandContainer


@dataclass(frozen=True)
class EmbodimentCatalog:
    r"""一份 resolved dataset 的平等资产轴与完整 partition/provenance 证据。"""

    dataset: ResolvedHandAssetDataset  # YAML identity、typed config 与全部 partition provenance

    @property
    def train(self) -> tuple[HandContainer, ...]:
        r"""返回 Trainer 唯一可用于参数更新的有序资产轴。"""

        return self.dataset.train.assets

    @property
    def validation(self) -> dict[str, tuple[HandContainer, ...]]:
        r"""返回 checkpoint selection 使用的两条具名 held-out 资产轴。

        Assets schema 2.0 保留 ``unseen_variant_set`` 与 ``unseen_mother``，训练侧
        必须先在各 suite 内聚合，再等权形成 checkpoint score，不能按资产数量扁平加权。
        """

        return {name: partition.assets for name, partition in self.dataset.validation.items()}

    @property
    def evaluation(self) -> dict[str, tuple[HandContainer, ...]]:
        r"""返回训练冻结后使用的具名 evaluation suites。"""

        return {name: partition.assets for name, partition in self.dataset.evaluation.items()}


class HandAssetCatalog:
    r"""解析一份 hand asset dataset；构造阶段不读取文件系统。"""

    def __init__(self, config: HandAssetCatalogCfg) -> None:
        r"""保存 manifest identity 与安全选项。"""

        self.config = config

    def resolve(self) -> EmbodimentCatalog:
        r"""读取 manifest、展开固定 roles 并验证可选的预期 SHA-256。"""

        dataset = HandAssetDataset.from_yaml(self.config.manifest).resolve(
            require_geometry_semantics=True,
            allow_legacy_left_handedness=self.config.allow_legacy_left_handedness,
        )
        if self.config.expected_sha256 and dataset.source_sha256 != self.config.expected_sha256:
            raise ValueError(
                "hand asset dataset SHA-256 mismatch: "
                f"expected={self.config.expected_sha256}, actual={dataset.source_sha256}"
            )
        return EmbodimentCatalog(dataset)


@dataclass(frozen=True)
class HandAssetCatalogCfg:
    r"""固定消费一份 HandAssetDataset 的 data role 配置。"""

    runtime_type: ClassVar[type[HandAssetCatalog]] = HandAssetCatalog  # Hydra 不序列化 runtime 绑定
    manifest: str = ""  # 相对 AnyMani 根或绝对 dataset YAML 路径
    expected_sha256: str = ""  # 正式 recipe 可钉住原始 YAML bytes；空值只记录实际 hash
    allow_legacy_left_handedness: bool = False  # 历史审计专用，正式训练保持 false

    def __post_init__(self) -> None:
        r"""拒绝空 manifest 和格式错误的显式 SHA-256。"""

        if not self.manifest:
            raise ValueError("hand asset catalog requires one dataset manifest")
        if self.expected_sha256 and (
            len(self.expected_sha256) != 64 or any(char not in "0123456789abcdef" for char in self.expected_sha256)
        ):
            raise ValueError("expected_sha256 must be an empty string or one lowercase SHA-256 digest")


__all__ = ["EmbodimentCatalog", "HandAssetCatalog", "HandAssetCatalogCfg"]
