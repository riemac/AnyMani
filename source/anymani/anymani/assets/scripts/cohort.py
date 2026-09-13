r"""从已发布训练源构建纯家族32×4的预抓取准备候选。

选择单位为母体系，每条谱系由母体与三个几何代表变体组成。LEAP/Allegro各自的源配额由typed recipe
交付，源坐标保留manifest身份；输出仍属于source-level候选，canonical物理身份与初态准入分别验证。

此入口不读取策略成绩。输出原子新建，已有集合可通过新cohort ID与新路径表达后续准备修订。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path

from anymani.assets.bank import cohort_selection
from anymani.assets.bank.cohort import load_hand_asset_cohort, write_hand_asset_cohort_lock
from anymani.assets.bank.path_utils import resolve_anymani_root, resolve_bank_path


def main() -> None:
    r"""解析家族及来源，执行分层选择，并发布可追溯的准备候选。

    源manifest可以显式覆盖，默认读取已有跨形态数据集。每组均为8母体×4代表，故总资产数为128。
    ``--exclude-mother``追加保护谱系，已在recipe内记录的留出始终保留；不足源配额时明确报告缺口。
    """

    # 输入只描述资产来源与成员身份，不引入任务奖励或策略评价信息。
    root = resolve_anymani_root()  # AnyMani根目录，避免shell工作目录改变默认来源
    dataset_root = root / "source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("leap", "allegro"), required=True)
    parser.add_argument("--cohort-id", required=True, help="本次准备候选的独立名称。")
    parser.add_argument("--output", type=Path, default=None, help="显式候选lock输出，可覆盖分层默认路径。")
    parser.add_argument("--ppo-manifest", type=Path, default=dataset_root / "ppo.yaml")
    parser.add_argument("--ssl-manifest", type=Path, default=dataset_root / "ssl.yaml")
    parser.add_argument("--exclude-mother", action="append", default=[], help="追加整条研究留出谱系。")
    args = parser.parse_args()

    # 显式源配额决定家族内四cell覆盖；新名称与追加排除写入独立recipe，常量本身不变。
    base = (
        cohort_selection.PURE_LEAP_RIGHT_A128_RECIPE
        if args.family == "leap"
        else cohort_selection.PURE_ALLEGRO_RIGHT_A128_RECIPE
    )
    exclusions = tuple(dict.fromkeys((*base.excluded_mother_names, *args.exclude_mother)))
    recipe = replace(base, cohort_id=args.cohort_id, excluded_mother_names=exclusions)
    sources = {"ppo": resolve_bank_path(args.ppo_manifest), "ssl": resolve_bank_path(args.ssl_manifest)}
    output = resolve_bank_path(
        args.output or dataset_root / "cohorts" / "pure_family" / f"{recipe.cohort_id}.lock.yaml"
    )  # 默认按稳定主题嵌套，命令可指定完整case目录
    if output.exists():
        raise FileExistsError(output)

    # Selector只输出源坐标及选择证据；统一writer再次验证完整源、内容与成员顺序。
    resolved = cohort_selection.resolve_lineage_cohort_selection(recipe, source_manifests=sources)
    selection = {
        **resolved.selection_document,
        "purpose": "strict-pregrasp-preparation-candidates",
        "policy_exposure": "selection-does-not-read-policy-results",
        "selector_source_sha256": hashlib.sha256(Path(cohort_selection.__file__).read_bytes()).hexdigest(),
    }  # 后续最终发布需另加canonical/strict证据，不能把源存在视为已通过准入
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".family-cohort-", dir=output.parent) as directory:
        staged = write_hand_asset_cohort_lock(
            Path(directory) / "source.lock.yaml",
            cohort_id=recipe.cohort_id,
            source_manifests=sources,
            member_coordinates=resolved.member_coordinates,
            selection=selection,
            require_geometry_semantics=True,
        )
        os.link(staged, output)  # 同文件系统原子新建，保护检查后并发出现的目标

    # 独立consumer重读发布文件，摘要只报告实际来源和数量，不赋予训练准备完成状态。
    cohort = load_hand_asset_cohort(output, require_geometry_semantics=True)
    mothers = Counter(member.provenance.mother_path for member in cohort.members)
    source_counts = Counter(member.source_alias for member in cohort.members)
    print(
        json.dumps(
            {
                "cohort_id": cohort.cohort_id,
                "family": recipe.family,
                "asset_count": len(cohort.members),
                "mother_count": len(mothers),
                "members_per_mother": sorted(set(mothers.values())),
                "source_member_counts": dict(sorted(source_counts.items())),
                "lock_path": str(cohort.lock_path),
                "lock_sha256": cohort.lock_sha256,
                "canonical_physics": "pending",
                "strict_pregrasp": "pending",
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
