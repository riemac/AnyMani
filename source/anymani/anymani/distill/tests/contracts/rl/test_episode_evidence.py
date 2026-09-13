r"""逐回合证据的CPU文件合同：保留联合分布，不把右删失或缺失数据记为成功。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from anymani.distill.diagnostics.recording.rl.episode_evidence import write_episode_evidence


def test_episode_round_trip_preserves_rewards_versions_and_censoring(tmp_path: Path, episode_columns) -> None:
    r"""真实奖励积分不再归一化；跨策略版本与未完成回合保持可辨别。"""
    destination = tmp_path / "episodes-0000.parquet"
    write_episode_evidence(
        destination,
        episode_columns,
        reward_sums={"rotation_progress": np.array([6.0, -0.2]), "failure": np.array([0.0, 0.0])},
        identity_digest="a" * 64,
        segment_id="segment-0",
    )
    table = pl.read_parquet(destination)
    assert table.height == 2
    assert table["reward_sum/rotation_progress"].to_list() == [6.0, -0.2]
    assert table["policy_version_start"].to_list() == [0, 3]
    assert table["policy_version_end"].to_list() == [4, 8]
    assert table["censored"].to_list() == [False, True]
    assert table.filter(~pl.col("censored"))["net_turns"].median() == 1.25
    assert table["identity_digest"].unique().to_list() == ["a" * 64]
    assert not list(tmp_path.glob("*.tmp"))


def test_existing_evidence_cannot_be_overwritten(tmp_path: Path, episode_columns) -> None:
    r"""错误复用分片路径不能改写先前证据。"""
    destination = tmp_path / "episodes.parquet"
    destination.write_bytes(b"previous-evidence")
    with pytest.raises(ValueError, match="new .parquet"):
        write_episode_evidence(
            destination, episode_columns, reward_sums={}, identity_digest="a" * 64, segment_id="segment-0"
        )
    assert destination.read_bytes() == b"previous-evidence"


@pytest.mark.parametrize(
    "column,value,message",
    [
        ("duration_s", np.array([120.0, 5.0]), "duration"),
        ("episode_id", np.array([0.0, 0.0]), "integer"),
        ("policy_version_end", np.array([4, 2]), "monotone"),
        ("absolute_path_turns", np.array([0.1, 0.3]), "absolute path"),
        ("max_positive_net_turns", np.array([0.1, 0.1]), "frontier"),
        ("net_turns", np.array([np.nan, -0.1]), "finite"),
        ("censored", np.array([True, True]), "terminal or censored"),
        ("censored", np.array([False, False]), "terminal or censored"),
        ("termination_drop", np.array([0, 0]), "boolean"),
    ],
)
def test_invalid_episode_facts_fail_before_publication(
    tmp_path: Path, episode_columns, column: str, value: np.ndarray, message: str
) -> None:
    r"""时间、身份、几何充分统计或终止语义不一致时拒绝发布。"""
    episode_columns[column] = value
    destination = tmp_path / "episodes.parquet"
    with pytest.raises(ValueError, match=message):
        write_episode_evidence(
            destination, episode_columns, reward_sums={}, identity_digest="a" * 64, segment_id="segment-0"
        )
    assert not destination.exists()


def test_duplicate_episode_is_rejected(tmp_path: Path, episode_columns) -> None:
    r"""一个segment内同一环境/episode只允许出现一次。"""
    episode_columns["env_id"][:] = 0
    with pytest.raises(ValueError, match="duplicate"):
        write_episode_evidence(
            tmp_path / "episodes.parquet",
            episode_columns,
            reward_sums={},
            identity_digest="a" * 64,
            segment_id="segment-0",
        )


def test_invalid_reward_array_is_not_silently_zero_filled(tmp_path: Path, episode_columns) -> None:
    r"""缺测或非有限奖励贡献必须显式处理，不生成伪造的零贡献。"""
    with pytest.raises(ValueError, match="finite"):
        write_episode_evidence(
            tmp_path / "episodes.parquet",
            episode_columns,
            reward_sums={"torque_l2": np.array([0.0, np.nan])},
            identity_digest="a" * 64,
            segment_id="segment-0",
        )


def test_write_failure_does_not_publish_partial_evidence(tmp_path: Path, episode_columns, monkeypatch) -> None:
    r"""写入失败清理临时文件，读者只看到完整分片。"""

    def fail_write(*args, **kwargs):
        raise OSError("simulated storage failure")

    monkeypatch.setattr(pl.DataFrame, "write_parquet", fail_write)
    with pytest.raises(OSError, match="storage failure"):
        write_episode_evidence(
            tmp_path / "episodes.parquet",
            episode_columns,
            reward_sums={},
            identity_digest="a" * 64,
            segment_id="segment-0",
        )
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def episode_columns() -> dict[str, np.ndarray]:
    r"""一个完整30秒回合和一个5秒右删失回合，ID相同但来自不同环境。"""
    return {
        "env_id": np.array([0, 1]),
        "episode_id": np.array([0, 0]),
        "asset_index": np.array([0, 1]),
        "policy_version_start": np.array([0, 3]),
        "policy_version_end": np.array([4, 8]),
        "policy_steps": np.array([600, 100]),
        "duration_s": np.array([30.0, 5.0]),
        "net_turns": np.array([1.25, -0.1]),
        "absolute_path_turns": np.array([1.5, 0.3]),
        "max_positive_net_turns": np.array([1.25, 0.1]),
        "goal_count": np.array([12, 0]),
        "frontier_count": np.array([15, 1]),
        "termination_drop": np.array([False, False]),
        "termination_axis": np.array([False, False]),
        "termination_timeout": np.array([True, False]),
        "censored": np.array([False, True]),
    }
