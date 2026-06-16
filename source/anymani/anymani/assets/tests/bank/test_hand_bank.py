r"""Hand asset bank 的路径与 selection contract tests。

测试只在 `tmp_path` 中构造最小 post-mutate 产物形状，不调用 generator，避免在
`assets/generated/` 下产生新资产堆积。真实 generated 产物结构已经由 generator / exporter
测试覆盖；这里关注 bank 对产物目录 contract 的消费语义。
"""

from __future__ import annotations

import textwrap
from pathlib import Path, PurePosixPath

import pytest
from anymani.assets.bank import HandBank, HandBankCfg, HandContainer, HandContainerCfg


def _write_sample(
    run_root: Path,
    sample_id: str,
    *,
    mesh_name: str = "finger_tip_soft.stl",
    include_sidecar: bool = True,
) -> Path:
    r"""写出一个最小 post-mutate sample bundle。

    Args:
        run_root (Path): post-mutate run 根目录，内部持有共享 `meshes/`。
        sample_id (str): sample 目录名，同时默认写入 sidecar `id`。
        mesh_name (str): 共享 mesh 文件名。
        include_sidecar (bool): 是否写 `hand.yaml`。

    Returns:
        Path: sample bundle 目录。
    """

    mesh_dir = run_root / "meshes"
    sample_dir = run_root / sample_id
    mesh_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    (mesh_dir / mesh_name).write_text("solid fake\nendsolid fake\n", encoding="utf-8")
    (sample_dir / "hand.urdf").write_text(
        textwrap.dedent(
            f"""
            <robot name="fake_hand">
              <link name="tip">
                <visual name="tip_visual">
                  <geometry><mesh filename="../meshes/{mesh_name}" /></geometry>
                  <material name="tip_color"><color rgba="0.92 0.88 0.78 1" /></material>
                </visual>
                <collision name="tip_collision">
                  <geometry><mesh filename="../meshes/{mesh_name}" /></geometry>
                </collision>
              </link>
            </robot>
            """
        ).strip(),
        encoding="utf-8",
    )
    if include_sidecar:
        (sample_dir / "hand.yaml").write_text(
            textwrap.dedent(
                f"""
                id: {sample_id}
                topology_name: right_t4_i4_m4_r4
                dof: 16
                hand_cfg: {{}}
                """
            ).strip(),
            encoding="utf-8",
        )
    return sample_dir


def test_hand_container_exposes_virtual_bundle_bijection(tmp_path: Path) -> None:
    r"""post-mutate shared mesh layout 应被作伪成标准 `hand.urdf + meshes/` 视图。"""

    run_root = tmp_path / "post_mutate"
    sample_dir = _write_sample(run_root, "066b6272")

    container = HandContainer.from_cfg(HandContainerCfg(path="066b6272"), source_root=run_root)

    assert container.asset_id == "066b6272"
    assert container.real_path("hand.urdf") == (sample_dir / "hand.urdf").resolve(strict=False)
    assert container.real_path("hand.yaml") == (sample_dir / "hand.yaml").resolve(strict=False)
    assert container.real_path("meshes/finger_tip_soft.stl") == (run_root / "meshes" / "finger_tip_soft.stl").resolve(
        strict=False
    )
    assert container.virtual_path(run_root / "meshes" / "finger_tip_soft.stl") == PurePosixPath(
        "meshes/finger_tip_soft.stl"
    )
    assert container.mesh_refs[0].raw_uri == "../meshes/finger_tip_soft.stl"
    assert container.visual_rgba_by_name["tip_visual"] == (0.92, 0.88, 0.78, 1.0)


def test_hand_bank_explicit_resolves_sample_id_relative_to_post_mutate_root(tmp_path: Path) -> None:
    r"""explicit mode 下，字符串 sample id 简写应相对 run root 解析。"""

    run_root = tmp_path / "post_mutate"
    _write_sample(run_root, "066b6272")

    selection = HandBank(
        HandBankCfg(
            post_mutate_path=run_root,
            selection_mode="explicit",
            containers=("066b6272",),
        )
    ).resolve()

    assert [asset.asset_id for asset in selection.assets] == ["066b6272"]
    assert selection.source_root == run_root.resolve(strict=False)


def test_hand_bank_all_and_sample_are_stable(tmp_path: Path) -> None:
    r"""all mode 稳定排序；sample mode 在固定 seed 下可复现且无放回。"""

    run_root = tmp_path / "post_mutate"
    for sample_id in ("b_sample", "a_sample", "c_sample"):
        _write_sample(run_root, sample_id, mesh_name=f"{sample_id}.stl")

    all_selection = HandBank(HandBankCfg(post_mutate_path=run_root, selection_mode="all")).resolve()
    sample_cfg = HandBankCfg(post_mutate_path=run_root, selection_mode="sample", sample_count=2, sample_seed=17)
    first_sample = HandBank(sample_cfg).resolve()
    second_sample = HandBank(sample_cfg).resolve()

    assert [asset.asset_id for asset in all_selection.assets] == ["a_sample", "b_sample", "c_sample"]
    assert [asset.asset_id for asset in first_sample.assets] == [asset.asset_id for asset in second_sample.assets]
    assert len({asset.asset_id for asset in first_sample.assets}) == 2


def test_hand_bank_sample_rejects_oversized_request(tmp_path: Path) -> None:
    r"""sample_count 大于候选数时应 fail-fast，而不是静默重复资产。"""

    run_root = tmp_path / "post_mutate"
    _write_sample(run_root, "only_one")

    with pytest.raises(ValueError, match="exceeds available"):
        HandBank(HandBankCfg(post_mutate_path=run_root, selection_mode="sample", sample_count=2)).resolve()


def test_hand_container_requires_sidecar_by_default(tmp_path: Path) -> None:
    r"""`hand.yaml` 默认属于 hand container contract，缺失时应报错。"""

    run_root = tmp_path / "post_mutate"
    _write_sample(run_root, "missing_sidecar", include_sidecar=False)

    with pytest.raises(FileNotFoundError, match="sidecar"):
        HandContainer.from_cfg(HandContainerCfg(path="missing_sidecar"), source_root=run_root)

    relaxed = HandContainer.from_cfg(
        HandContainerCfg(path="missing_sidecar"),
        source_root=run_root,
        require_sidecar=False,
    )
    assert relaxed.asset_id == "missing_sidecar"


def test_hand_container_rejects_missing_mesh_reference(tmp_path: Path) -> None:
    r"""URDF mesh filename 默认必须闭合到真实文件。"""

    run_root = tmp_path / "post_mutate"
    _write_sample(run_root, "missing_mesh")
    (run_root / "meshes" / "finger_tip_soft.stl").unlink()

    with pytest.raises(FileNotFoundError, match="mesh reference"):
        HandContainer.from_cfg(HandContainerCfg(path="missing_mesh"), source_root=run_root)


def test_hand_container_rejects_virtual_mesh_path_conflict(tmp_path: Path) -> None:
    r"""两个不同真实 mesh 不允许映射到同一个虚拟 `meshes/<basename>`。"""

    run_root = tmp_path / "post_mutate"
    sample_dir = run_root / "conflict"
    (run_root / "meshes_a").mkdir(parents=True)
    (run_root / "meshes_b").mkdir(parents=True)
    sample_dir.mkdir(parents=True)
    (run_root / "meshes_a" / "same.stl").write_text("solid a\nendsolid a\n", encoding="utf-8")
    (run_root / "meshes_b" / "same.stl").write_text("solid b\nendsolid b\n", encoding="utf-8")
    (sample_dir / "hand.yaml").write_text("id: conflict\nhand_cfg: {}\n", encoding="utf-8")
    (sample_dir / "hand.urdf").write_text(
        textwrap.dedent(
            """
            <robot name="fake_hand">
              <link name="tip_a"><visual><geometry><mesh filename="../meshes_a/same.stl" /></geometry></visual></link>
              <link name="tip_b"><visual><geometry><mesh filename="../meshes_b/same.stl" /></geometry></visual></link>
            </robot>
            """
        ).strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="maps to both"):
        HandContainer.from_cfg(HandContainerCfg(path="conflict"), source_root=run_root)


@pytest.mark.parametrize("source_mode", ["pre_made", "mixed"])
def test_unimplemented_source_modes_fail_explicitly(tmp_path: Path, source_mode: str) -> None:
    r"""pre-made / mixed 接口先保留，但第一版 resolve 必须显式拒绝。"""

    with pytest.raises(NotImplementedError, match=source_mode):
        HandBank(HandBankCfg(source_mode=source_mode, post_mutate_path=tmp_path, selection_mode="all")).resolve()
