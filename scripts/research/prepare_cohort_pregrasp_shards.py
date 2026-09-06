r"""按训练同一exact key核对cohort的strict预抓取，并只为缺失资产发布生成分片。

输入是已完成真实canonical lowering的完整训练集合。缓存命中同时要求hand configuration、物理几何、
canonical/routing、object、scale、physics和generation身份一致，并重新验证全部Top-8硬门。已有条目只读，
不重新搜索或替换旧初态；缺失成员按形态cell交错后分片，交给现有strict生成器执行。

本脚本启动Kit以构造训练同路径的asset binding，但不创建scene或推进PhysX。默认每片16资产，现有生成器
每资产同时验证32候选，因此每片512个物理环境。分片只划分生成任务，不缩小最终训练/评价集合。
每次检查使用新的输出目录；目录内的source/canonical子集合及preparation.json共同保留完整父集合映射。
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

import yaml


def main() -> None:
    r"""核对完整父集合，生成缺失成员的有界任务，不改变搜索分布或接触/稳定准入门。

    只有GoodPregraspMissError表示需要补建；损坏缓存、身份不符和不满足strict门均保留为错误。
    子集合的canonical证书来自本次实际binding，不能从源bundle内容hash猜测配置或物理身份。
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-lock", type=Path, required=True, help="完整训练集合的canonical-final lock。")
    parser.add_argument("--output-dir", type=Path, required=True, help="本次检查独占的新输出目录。")
    parser.add_argument("--shard-assets", type=int, default=16, help="每个生成任务的最大资产数，默认16。")
    parser.add_argument("--catalog", type=Path, default=None, help="缺省与训练入口使用相同strict目录。")
    args = parser.parse_args()
    if args.shard_assets < 1:
        parser.error("--shard-assets must be positive")
    parent_path = args.cohort_lock.expanduser().resolve(strict=True)  # 最终训练成员轴，不在本工具中重选资产。
    output_root = args.output_dir.expanduser().resolve()  # 独占的准备证据目录。
    if output_root.exists():
        raise FileExistsError(f"pregrasp preparation output already exists: {output_root}")
    parent = yaml.safe_load(parent_path.read_text(encoding="utf-8"))
    if not isinstance(parent, dict) or parent.get("schema_version") != "1.2.0":
        raise ValueError("pregrasp preparation requires a canonical-final cohort lock")

    # 必须在任务模块导入前固定支持集；这里只创建静态binding，环境轴不进入物理仿真。
    os.environ.pop("ANYMANI_HETERO_ASSET_ROWS", None)
    os.environ["ANYMANI_HETERO_COHORT_LOCK"] = str(parent_path)
    os.environ["ANYMANI_HETERO_NUM_ENVS"] = "1"
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True)
    try:
        from anymani.assets.bank.cohort import load_hand_asset_cohort, write_hand_asset_cohort_subset
        from anymani.pregrasp.good_catalog import GoodPregraspCatalog, GoodPregraspMissError
        from anymani.pregrasp.strict_gate import MVP80_STRICT_GOOD_PREGRASP_GATE
        from anymani.tasks.hetero.config.generated.asset_binding import build_generated_asset_binding

        parent_cohort = load_hand_asset_cohort(parent_path)  # 所有分片共享这份已验证证据，不在每片内重复解析源库。
        binding = build_generated_asset_binding()  # 与PPO完全相同的source/canonical/routing真源。
        assert parent_cohort.lock_sha256 == binding.cohort_lock_sha256
        reset_cfg = binding.build_good_pregrasp_reset_cfg(
            num_envs=binding.asset_count,
            catalog_root=args.catalog.expanduser().resolve() if args.catalog is not None else None,
        )  # 每资产一个逻辑副本，只借用runtime正向构造查询键。
        catalog = GoodPregraspCatalog(reset_cfg.catalog_root)
        cached_indices: list[int] = []  # 已经具有合法Top-8的完整父集合索引。
        missing_by_cell: dict[int, list[int]] = defaultdict(list)  # 缺失项按形态cell组织，不改变policy标签。
        keys = tuple(item.resolve_key() for item in reset_cfg.bindings)  # exact key包含全部生成和物理语义。
        for index, key in enumerate(keys):
            try:
                entry = catalog.resolve(key)
            except GoodPregraspMissError:
                missing_by_cell[binding.morphology_cell_ids[index]].append(index)
            else:
                MVP80_STRICT_GOOD_PREGRASP_GATE.validate_entry(entry)  # 命中也重验Top-8；不把坏缓存视作缺失。
                cached_indices.append(index)

        # 交错各cell，使第一个小分片就包含不同指尖数/拇指自由度。尾部不补重复资产。
        ordered_missing = [
            index
            for row in zip_longest(*(missing_by_cell[cell] for cell in sorted(missing_by_cell)))
            for index in row
            if index is not None
        ]
        assert len(cached_indices) + len(ordered_missing) == binding.asset_count
        assert len(set(cached_indices + ordered_missing)) == binding.asset_count  # 完整父分母无遗漏或重复。
        output_root.mkdir(parents=True)
        inspection = {
            "parent_cohort_lock": str(parent_path),
            "parent_cohort_lock_sha256": binding.cohort_lock_sha256,
            "catalog_root": str(catalog.root),
            "expected_keys": [key.to_dict() for key in keys],
            "cached_parent_asset_indices": cached_indices,
            "ordered_missing_parent_asset_indices": ordered_missing,
        }  # 先保存完整查询结果，发布分片途中退出也不丢失已核对的成员轴。
        (output_root / "inspection.json").write_text(
            json.dumps(inspection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        shards = []
        for start in range(0, len(ordered_missing), args.shard_assets):
            indices = ordered_missing[start : start + args.shard_assets]  # 子集合到完整训练轴的显式映射。
            shard_id = f"{output_root.name}-shard-{len(shards):03d}"  # 新检查批次使用新名称，不与旧部分输出混称。
            source_path = output_root / f"{shard_id}.lock.yaml"
            canonical_path = output_root / f"{shard_id}.canonical.lock.yaml"
            write_hand_asset_cohort_subset(
                parent_cohort,
                source_path,
                canonical_path,
                cohort_id=shard_id,
                member_indices=indices,
                selection={
                    "algorithm": "missing-exact-pregrasp-cell-interleaved-v1",
                    "purpose": "pregrasp-generation-only",
                    "catalog_root": str(catalog.root),  # 后续仅给分片锁时，也保持本次明确选择的数据位置。
                },
            )  # 子集发布不重新解析完整训练库；实际生成器加载时仍执行全部source/canonical核对。
            shards.append(
                {
                    "cohort_id": shard_id,
                    "cohort_lock": str(canonical_path),
                    "asset_count": len(indices),
                    "parent_asset_indices": indices,
                }
            )

        # 发布的是准备清单，不是学习结果。只有所有缺失项生成完，完整cohort才可进入训练。
        document = {
            "artifact_type": "anymani.good_pregrasp.cohort_preparation",
            "schema_version": "1.0.0",
            "parent_cohort_lock": str(parent_path),
            "parent_cohort_lock_sha256": binding.cohort_lock_sha256,
            "catalog_root": str(catalog.root),
            "asset_count": binding.asset_count,
            "cached_count": len(cached_indices),
            "cached_parent_asset_indices": cached_indices,
            "missing_count": len(ordered_missing),
            "shard_assets_max": args.shard_assets,
            "shards": shards,
        }
        plan_path = output_root / "preparation.json"
        temporary = plan_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(plan_path)
        print(
            json.dumps(
                {
                    "preparation": str(plan_path),
                    "assets": binding.asset_count,
                    "cached": len(cached_indices),
                    "missing": len(ordered_missing),
                    "shards": len(shards),
                }
            ),
            flush=True,
        )
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
