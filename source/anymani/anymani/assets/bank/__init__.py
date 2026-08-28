r"""Asset-bank 子包入口。

当前只落 hand asset bank 的 scaffold。实现细节放在 `hand_bank.py`，本文件仅提供
稳定 re-export，便于后续下游使用：

```python
from anymani.assets.bank import HandBankCfg
```
"""

from .dataset import (
    HAND_ASSET_DATASET_SCHEMA_VERSION,
    HandAssetDataset,
    HandAssetDatasetCfg,
    HandAssetEvaluationCfg,
    HandAssetLineageCfg,
    HandAssetOfficialPartitionCfg,
    HandAssetPartitionCfg,
    HandAssetProvenance,
    HandAssetRunCfg,
    ResolvedHandAssetDataset,
    ResolvedHandAssetPartition,
    ResolvedHandAssetRecord,
)
from .geometry_semantics import HandAssetSourceKind
from .hand_bank import (
    HandBank,
    HandBankCfg,
    HandSelection,
    HandSelectionMode,
    HandSourceMode,
)
from .hand_container import (
    HandContainer,
    HandContainerCfg,
    HandContainerLike,
    UrdfMeshRef,
    UrdfRgba,
    coerce_hand_container_cfg,
)
from .path_utils import (
    resolve_anymani_root,
    resolve_bank_path,
    resolve_container_entry_path,
    resolve_post_mutate_root,
)
from .prepared_train import PREPARED_TRAIN_CACHE_SCHEMA_VERSION, resolve_prepared_train

__all__ = [
    "HAND_ASSET_DATASET_SCHEMA_VERSION",
    "PREPARED_TRAIN_CACHE_SCHEMA_VERSION",
    "HandAssetDataset",
    "HandAssetDatasetCfg",
    "HandAssetEvaluationCfg",
    "HandAssetLineageCfg",
    "HandAssetOfficialPartitionCfg",
    "HandAssetPartitionCfg",
    "HandAssetProvenance",
    "HandAssetRunCfg",
    "HandBank",
    "HandBankCfg",
    "HandContainer",
    "HandContainerCfg",
    "HandContainerLike",
    "HandSelection",
    "HandSelectionMode",
    "HandSourceMode",
    "ResolvedHandAssetDataset",
    "ResolvedHandAssetPartition",
    "ResolvedHandAssetRecord",
    "HandAssetSourceKind",
    "UrdfMeshRef",
    "UrdfRgba",
    "resolve_anymani_root",
    "resolve_bank_path",
    "resolve_container_entry_path",
    "resolve_post_mutate_root",
    "resolve_prepared_train",
    "coerce_hand_container_cfg",
]
