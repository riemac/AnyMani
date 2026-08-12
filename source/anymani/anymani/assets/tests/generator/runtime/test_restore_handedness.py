r"""Independent post-mutate 来源恢复的 handedness 安全门测试。

``load_post_mutate_source`` 直接恢复 topology 根的 ``hand.yaml.hand_cfg``，不会经过
``HandBank``。因此 legacy generated left 的 fail-closed 规则必须在 restore 边界
再次执行，否则错误左手仍可作为 post-mutate 母体继续派生。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from assets.asset_schema_embodiment import FingerCfg, HandCfg, JointCfg, PalmCfg
from assets.generator.runtime.restore import load_post_mutate_source


def _write_topology_sidecar(
    topology_dir: Path,
    *,
    handedness: str,
    include_contract: bool,
) -> Path:
    r"""写出一份可由 restore 恢复的最小 generated topology sidecar。"""

    topology_dir.mkdir(parents=True, exist_ok=True)
    hand = HandCfg(
        name=f"restore_{handedness}",
        family="unit_test",
        handedness=handedness,
        palm=PalmCfg(name="palm"),
        fingers=[
            FingerCfg(
                name="index",
                parent_link="palm",
                joints=[
                    JointCfg(
                        name="index_j0",
                        parent="palm",
                        child="index_link",
                        joint_type="revolute",
                        limit=(-1.0, 1.0),
                    )
                ],
            )
        ],
    )  # 最小一自由度 hand 足以隔离 restore/contract，不引入 mesh 与 physics 变量
    sidecar = {
        "id": f"restore_{handedness}_source",
        "handedness": handedness,
        "topology_name": f"{handedness}_i1",
        "hand_cfg": hand.to_dict(),
    }
    if include_contract:
        sidecar["handedness_contract"] = {
            "version": "1.0",
            "canonical_handedness": "right",
            "target_handedness": handedness,
            "reflection_plane": "palm_yz",
            "same_q": True,
            "physical_lowering_complete": True,
        }  # 模拟当前 exporter 写出的顶层严格合同
    sidecar_path = topology_dir / "hand.yaml"
    sidecar_path.write_text(
        yaml.safe_dump(sidecar, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return sidecar_path


def test_restore_rejects_legacy_generated_left_by_default(tmp_path: Path) -> None:
    r"""Legacy left 缺少严格合同，不得作为 mutate-only 母体。"""

    topology_dir = tmp_path / "legacy_left"
    _write_topology_sidecar(topology_dir, handedness="left", include_contract=False)

    with pytest.raises(ValueError, match="legacy generated left"):
        load_post_mutate_source(topology_dir)


def test_restore_allows_strict_left_and_explicit_legacy_audit(tmp_path: Path) -> None:
    r"""新 strict left 默认可恢复；legacy override 只用于显式历史审计。"""

    strict_dir = tmp_path / "strict_left"
    legacy_dir = tmp_path / "legacy_left"
    _write_topology_sidecar(strict_dir, handedness="left", include_contract=True)
    _write_topology_sidecar(legacy_dir, handedness="left", include_contract=False)

    strict_source = load_post_mutate_source(strict_dir)
    legacy_source = load_post_mutate_source(legacy_dir, allow_legacy_left_handedness=True)

    assert strict_source.hand_cfg.handedness == "left"
    assert legacy_source.hand_cfg.handedness == "left"
