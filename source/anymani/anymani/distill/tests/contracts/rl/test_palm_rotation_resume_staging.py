r"""独立续训目录只包含checkpoint声明的不可变证据，不吸收其后shards。"""

import hashlib
from pathlib import Path

import pytest
import torch
from anymani.distill.rl.scripts.prepare_palm_rotation_resume import prepare_resume


@pytest.mark.parametrize("corrupted", (False, True))
def test_resume_links_only_declared_shards_and_preserves_source(tmp_path: Path, corrupted: bool) -> None:
    r"""库存漂移先失败；成功分支共享只读inode而不共享合并表，source后续片保留原处。"""

    source = tmp_path / "source"
    (source / "nn").mkdir(parents=True)
    (source / "metrics_shards").mkdir()
    shard = source / "metrics_shards" / "metrics-000000-u00000001-u00000040.parquet"
    shard.write_bytes(b"immutable-shard")
    later = source / "metrics_shards" / "metrics-000001-u00000041-u00000050.parquet"
    later.write_bytes(b"not-backed-by-checkpoint")
    (source / "metrics.parquet").write_bytes(b"mutable-merge")
    checkpoint = source / "nn" / "epoch40.pth"
    torch.save(
        {
            "epoch": 40,
            "frame": 307200,
            "anymani_identity": {"identity_digest": "a" * 64},
            "anymani_metrics_recorder": {
                "identity_digest": "a" * 64,
                "last_recorded_update": 40,
                "shards": [{"name": shard.name, "sha256": hashlib.sha256(shard.read_bytes()).hexdigest()}],
            },
        },
        checkpoint,
    )
    target = tmp_path / "continuation"
    if corrupted:
        shard.write_bytes(b"changed")
        with pytest.raises(ValueError, match="SHA mismatch"):
            prepare_resume(checkpoint, target)
        assert not target.exists()
        return
    linked = prepare_resume(checkpoint, target)
    assert linked.stat().st_ino == checkpoint.stat().st_ino
    assert (target / "metrics_shards" / shard.name).stat().st_ino == shard.stat().st_ino
    assert not (target / "metrics.parquet").exists()
    assert not (target / "metrics_shards" / later.name).exists()
    assert later.read_bytes() == b"not-backed-by-checkpoint"
    with pytest.raises(FileExistsError):
        prepare_resume(checkpoint, target)
