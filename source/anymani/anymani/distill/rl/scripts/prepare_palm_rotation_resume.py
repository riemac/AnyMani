r"""为同方法完整续训建立独立目录，保留source checkpoint与不可变Parquet分片。

新目录只链接checkpoint声明的分片，不复制可能领先于checkpoint的后续训练数据，也不链接可重新合并的
``metrics.parquet``。Actor、Critic、optimizer、RNG和课程仍由训练入口从同一checkpoint统一恢复。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


def prepare_resume(checkpoint: Path, destination: Path) -> Path:
    r"""验证checkpoint声明的完整分片库存，再在同一arm目录创建独立续训分支。

    返回新目录中的checkpoint路径。硬链接只用于不可变输入；source路径及内容保持不变。所有分片先按
    SHA-256验证，缺失或漂移时不创建目标目录；训练入口还会独立重验方法身份和库存。
    """

    checkpoint = checkpoint.expanduser().resolve(strict=True)
    destination = destination.expanduser().resolve()  # 新run与source共享arm根，但拥有独立的后续产物
    source_run = checkpoint.parent.parent
    if checkpoint.parent.name != "nn" or destination.parent != source_run.parent:
        raise ValueError("resume destination must be a sibling run under the same arm root")
    if destination.exists():
        raise FileExistsError(destination)
    document = torch.load(checkpoint, map_location="cpu", weights_only=False)
    inventory = document["anymani_metrics_recorder"]  # checkpoint时刻的训练证据边界
    identity = document["anymani_identity"]["identity_digest"]
    if inventory["identity_digest"] != identity or int(inventory["last_recorded_update"]) != int(document["epoch"]):
        raise ValueError("checkpoint and metrics inventory have different method/update identities")
    links = [(checkpoint, Path("nn") / checkpoint.name)]
    for shard in inventory["shards"]:
        name = shard["name"]
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("metrics inventory must contain plain shard basenames")
        source = source_run / "metrics_shards" / name
        with source.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != shard["sha256"]:
            raise ValueError(f"checkpoint-declared shard SHA mismatch: {name}")
        links.append((source, Path("metrics_shards") / name))
    if len({relative for _, relative in links}) != len(links):
        raise ValueError("metrics inventory contains duplicate shards")

    # 验证完成后才发布新目录；不链接mutable params或合并表，不改写checkpoint字节或内部计数。
    destination.mkdir()
    for source, relative in links:
        target = destination / relative
        target.parent.mkdir(exist_ok=True)
        os.link(source, target)
    with checkpoint.open("rb") as stream:
        checkpoint_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    (destination / "resume_source.json").write_text(
        json.dumps(
            {
                "source_checkpoint": str(checkpoint),
                "source_checkpoint_sha256": checkpoint_sha,
                "method_identity_digest": identity,
                "epoch": int(document["epoch"]),
                "frame": int(document["frame"]),
                "linked_shard_count": len(links) - 1,
                "source_checkpoint_inode": checkpoint.stat().st_ino,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination / "nn" / checkpoint.name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(prepare_resume(args.checkpoint, args.destination))
