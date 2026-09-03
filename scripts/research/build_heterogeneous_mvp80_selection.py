r"""从formal 2048 PPO train生成80手MVP的确定性left/right pair候选清单。"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from anymani.assets.bank.dataset import HandAssetDataset
from anymani.assets.bank.path_utils import resolve_anymani_root
from anymani.assets.bank.prepared_train import resolve_prepared_train
from anymani.assets.bank.representative_selection import representative_selection_document

DEFAULT_DATASET = Path("source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1/ppo.yaml")
"""Formal 2048资产父manifest，相对AnyMani根。"""

DEFAULT_OUTPUT = Path(
    "source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1/ppo_mvp80_candidates.yaml"
)
"""版本化候选pair artifact；最终80 rows将在pregrasp通过后另行发布。"""


def _parse_args() -> argparse.Namespace:
    r"""解析父manifest、输出路径与每cell候选pair数量。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pairs-per-cell", type=int, default=10)
    parser.add_argument("--candidate-pairs-per-cell", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    r"""解析一次prepared train并原子发布候选selection YAML。"""

    args = _parse_args()
    root = resolve_anymani_root()  # Hydra/shell cwd无关的项目根
    dataset_path = args.dataset if args.dataset.is_absolute() else root / args.dataset
    output_path = args.output if args.output.is_absolute() else root / args.output
    dataset = HandAssetDataset.from_yaml(dataset_path)
    partition, _ = resolve_prepared_train(dataset, require_geometry_semantics=True)
    document = representative_selection_document(
        partition,
        parent_dataset_path=str(dataset.source_path.relative_to(root)),  # 项目内可迁移路径，不写本机绝对前缀
        parent_dataset_sha256=dataset.source_sha256,
        pairs_per_cell=args.pairs_per_cell,
        candidate_pairs_per_cell=args.candidate_pairs_per_cell,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8")
    temporary.replace(output_path)  # 同文件系统replace只发布完整旧版或完整新版
    print(
        {
            "output": str(output_path),
            "selected_asset_count": document["selected_asset_count"],
            "candidate_pairs_per_cell": document["candidate_pairs_per_cell"],
        },
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
