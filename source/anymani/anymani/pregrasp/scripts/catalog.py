r"""复制已认证的完整Top-8条目，建立独立准备池或按exact key发布角色快照。

源目录经过schema、payload及键校验后，再以共同strict门检查全部条目。所有检查完成后才调用一次
``publish_many``，使目标目录只显示完整旧版或完整新版。复制保留原物理和生成身份，不重新搜索初态。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from anymani.pregrasp.good_catalog import GoodPregraspCatalog, GoodPregraspIndexEntry, GoodPregraspKey
from anymani.pregrasp.strict_gate import MVP80_STRICT_GOOD_PREGRASP_GATE


def copy_strict_catalog(
    source: GoodPregraspCatalog,
    target: GoodPregraspCatalog,
    *,
    keys: Sequence[GoodPregraspKey] | None = None,
) -> tuple[GoodPregraspIndexEntry, ...]:
    r"""将源中完整且满足strict门的条目发布到另一目录。

    Args:
        source: 已有schema-3目录，始终只读。
        target: 独立目标目录；同key同内容幂等，不同内容由publisher拒绝。
        keys: 明确的角色key集合；None读取全部源条目，显式集合中任一缺失即终止。

    Returns:
        与请求顺序一致的已发布index引用，原Top-8内容身份保持。
    """

    if source.root.resolve() == target.root.resolve():
        raise ValueError("source and target catalogs must be distinct")
    if not source.index_path.is_file():
        raise FileNotFoundError(source.index_path)  # 源路径错误与合法的空请求分开解释

    # Resolver已经验证来源内容；所有物理准入检查先于目标目录的任何可见提交。
    entries = source.read_entries() if keys is None else source.resolve_many(tuple(keys))
    for entry in entries:
        MVP80_STRICT_GOOD_PREGRASP_GATE.validate_entry(entry)  # Top-8逐项满足相同cold-reset准入
    return target.publish_many(entries)  # 只复制完整entry，不从部分NPZ重建或追加候选


def main() -> None:
    r"""解析源/目标目录和可选角色keys，执行一次受校验的复制。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="已有认证catalog根。")
    parser.add_argument("--target", type=Path, required=True, help="独立准备池或角色catalog根。")
    parser.add_argument(
        "--keys-file",
        type=Path,
        default=None,
        help="exact-key列表，或prepare工具产生的含expected_keys的inspection JSON。",
    )
    args = parser.parse_args()

    # 角色key由绑定/准备工具产生；其物理、routing、scale等字段通过统一key schema解析。
    keys = None
    if args.keys_file is not None:
        document = json.loads(args.keys_file.read_text(encoding="utf-8"))
        records = document.get("expected_keys") if isinstance(document, Mapping) else document
        if not isinstance(records, list):
            raise ValueError("keys file must contain an exact-key list or expected_keys")
        keys = tuple(GoodPregraspKey.from_dict(record) for record in records)
    source, target = GoodPregraspCatalog(args.source), GoodPregraspCatalog(args.target)
    copied = copy_strict_catalog(source, target, keys=keys)

    # 摘要复用publisher交付的内容身份，不再进行额外的文件级重复校验。
    print(
        json.dumps(
            {
                "source_catalog": str(source.root.resolve()),
                "target_catalog": str(target.root.resolve()),
                "copied_count": len(copied),
                "entry_digests": [entry.entry_digest for entry in copied],
                "strict_top8_validated": True,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
