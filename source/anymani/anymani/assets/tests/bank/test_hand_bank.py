r"""Hand asset bank 的路径与 selection contract tests。

测试只在 `tmp_path` 中构造最小 post-mutate 产物形状，不调用 generator，避免在
`assets/generated/` 下产生新资产堆积。真实 generated 产物结构已经由 generator / exporter
测试覆盖；这里关注 bank 对产物目录 contract 的消费语义。
"""

from __future__ import annotations

import textwrap
from pathlib import Path, PurePosixPath

import pytest
import yaml
from anymani.assets.bank import HandBank, HandBankCfg, HandContainer, HandContainerCfg


def _write_sample(
    run_root: Path,
    sample_id: str,
    *,
    mesh_name: str = "finger_tip_soft.stl",
    include_sidecar: bool = True,
    handedness: str = "right",
    include_handedness_contract: bool = False,
) -> Path:
    r"""写出一个最小 post-mutate sample bundle。

    Args:
        run_root (Path): post-mutate run 根目录，内部持有共享 `meshes/`。
        sample_id (str): sample 目录名，同时默认写入 sidecar `id`。
        mesh_name (str): 共享 mesh 文件名。
        include_sidecar (bool): 是否写 `hand.yaml`。
        handedness (str): sidecar 顶层物理 handedness。
        include_handedness_contract (bool): 是否写入严格整手镜像 contract v1。

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
        sidecar = {
            "id": sample_id,
            "handedness": handedness,
            "topology_name": f"{handedness}_t4_i4_m4_r4",
            "dof": 16,
            "hand_cfg": {},
        }  # fixture 显式模拟 exporter 的顶层 bundle contract
        if include_handedness_contract:
            sidecar["handedness_contract"] = {
                "version": "1.0",
                "canonical_handedness": "right",
                "target_handedness": handedness,
                "reflection_plane": "palm_yz",
                "same_q": True,
                "physical_lowering_complete": True,
            }  # 新 generated bundle 由该证书声明完整物理 lowering 已完成
        (sample_dir / "hand.yaml").write_text(
            yaml.safe_dump(sidecar, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    return sample_dir


def _write_source_topology(
    topology_root: Path,
    *,
    asset_id: str = "source_topology",
    mesh_name: str = "source_tip.stl",
) -> Path:
    r"""写出 post-mutate run 的母体 pre-made topology bundle。

    Args:
        topology_root (Path): pre-made topology 根目录，也就是 post-mutate run 的父目录。
        asset_id (str): 母体在 asset bank 中暴露的稳定 ID，真实产物来自 `hand.yaml.id`。
        mesh_name (str): 母体自己 `meshes/` 目录下的 mesh 文件名。

    Returns:
        Path: topology 根目录；它本身就是一个可消费的虚拟 hand bundle。
    """

    mesh_dir = topology_root / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    (mesh_dir / mesh_name).write_text("solid source\nendsolid source\n", encoding="utf-8")
    (topology_root / "hand.urdf").write_text(
        textwrap.dedent(
            f"""
            <robot name="source_hand">
              <link name="tip">
                <visual name="source_tip_visual">
                  <geometry><mesh filename="meshes/{mesh_name}" /></geometry>
                  <material name="source_color"><color rgba="0.1 0.2 0.3 1" /></material>
                </visual>
                <collision name="source_tip_collision">
                  <geometry><mesh filename="meshes/{mesh_name}" /></geometry>
                </collision>
              </link>
            </robot>
            """
        ).strip(),
        encoding="utf-8",
    )
    (topology_root / "hand.yaml").write_text(
        textwrap.dedent(
            f"""
            id: {asset_id}
            topology_name: right_t4_i4_m4_r4
            dof: 16
            validation:
              pre_made: {{}}
            hand_cfg: {{}}
            """
        ).strip(),
        encoding="utf-8",
    )
    return topology_root


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

    all_selection = HandBank(
        HandBankCfg(post_mutate_path=run_root, selection_mode="all", include_source_topology=False)
    ).resolve()
    sample_cfg = HandBankCfg(
        post_mutate_path=run_root,
        selection_mode="sample",
        sample_count=2,
        sample_seed=17,
        include_source_topology=False,
    )
    first_sample = HandBank(sample_cfg).resolve()
    second_sample = HandBank(sample_cfg).resolve()

    assert [asset.asset_id for asset in all_selection.assets] == ["a_sample", "b_sample", "c_sample"]
    assert [asset.asset_id for asset in first_sample.assets] == [asset.asset_id for asset in second_sample.assets]
    assert len({asset.asset_id for asset in first_sample.assets}) == 2


def test_post_mutate_discovery_includes_source_topology_as_peer_candidate(tmp_path: Path) -> None:
    r"""post-mutate source 应把母体 topology 与后变异样本拉平成同级候选。"""

    topology_root = tmp_path / "right_t4_i4_m4_r4"
    run_root = topology_root / "2026-06-11_14-20-22"
    _write_source_topology(topology_root, asset_id="premade_root", mesh_name="source_tip.stl")
    _write_sample(run_root, "variant_b", mesh_name="sample_tip.stl")
    _write_sample(run_root, "variant_a", mesh_name="sample_tip.stl")

    selection = HandBank(HandBankCfg(post_mutate_path=run_root, selection_mode="all")).resolve()
    assets_by_id = {asset.asset_id: asset for asset in selection.assets}

    assert [asset.asset_id for asset in selection.assets] == ["premade_root", "variant_a", "variant_b"]
    assert selection.source_root == run_root.resolve(strict=False)
    assert assets_by_id["premade_root"].real_path("hand.urdf") == (topology_root / "hand.urdf").resolve(strict=False)
    assert assets_by_id["premade_root"].real_path("meshes/source_tip.stl") == (
        topology_root / "meshes" / "source_tip.stl"
    ).resolve(strict=False)
    assert assets_by_id["variant_a"].real_path("meshes/sample_tip.stl") == (
        run_root / "meshes" / "sample_tip.stl"
    ).resolve(strict=False)
    assert assets_by_id["premade_root"].mesh_refs[0].raw_uri == "meshes/source_tip.stl"
    assert assets_by_id["variant_a"].mesh_refs[0].raw_uri == "../meshes/sample_tip.stl"


def test_post_mutate_source_topology_inclusion_can_be_disabled(tmp_path: Path) -> None:
    r"""调试旧实验时可显式关闭母体候选，恢复纯 post-mutate leaf 集合。"""

    topology_root = tmp_path / "right_t4_i4_m4_r4"
    run_root = topology_root / "2026-06-11_14-20-22"
    _write_source_topology(topology_root, asset_id="premade_root")
    _write_sample(run_root, "variant_a")

    selection = HandBank(
        HandBankCfg(post_mutate_path=run_root, selection_mode="all", include_source_topology=False)
    ).resolve()

    assert [asset.asset_id for asset in selection.assets] == ["variant_a"]


def test_post_mutate_source_topology_inclusion_requires_parent_bundle(tmp_path: Path) -> None:
    r"""默认包含母体时，非标准 run root 应 fail-fast，而不是退回旧式纯 leaf 语义。"""

    run_root = tmp_path / "standalone_post_mutate"
    _write_sample(run_root, "variant_a")

    with pytest.raises(FileNotFoundError, match="source topology bundle"):
        HandBank(HandBankCfg(post_mutate_path=run_root, selection_mode="all")).resolve()


def test_post_mutate_sample_count_sees_source_topology_candidate(tmp_path: Path) -> None:
    r"""固定 seed 采样的候选池应包含母体，因此容量上限是 $1+N_{variant}$。"""

    topology_root = tmp_path / "right_t4_i4_m4_r4"
    run_root = topology_root / "2026-06-11_14-20-22"
    _write_source_topology(topology_root, asset_id="premade_root")
    _write_sample(run_root, "variant_a")

    selection = HandBank(
        HandBankCfg(post_mutate_path=run_root, selection_mode="sample", sample_count=2, sample_seed=23)
    ).resolve()

    assert {asset.asset_id for asset in selection.assets} == {"premade_root", "variant_a"}


def test_hand_bank_sample_rejects_oversized_request(tmp_path: Path) -> None:
    r"""sample_count 大于候选数时应 fail-fast，而不是静默重复资产。"""

    topology_root = tmp_path / "right_t4_i4_m4_r4"
    run_root = topology_root / "post_mutate"
    _write_source_topology(topology_root, asset_id="source_topology")
    _write_sample(run_root, "only_one")

    with pytest.raises(ValueError, match="exceeds available"):
        HandBank(HandBankCfg(post_mutate_path=run_root, selection_mode="sample", sample_count=3)).resolve()


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


def test_generated_left_without_handedness_contract_is_rejected_by_default(tmp_path: Path) -> None:
    r"""Legacy generated left 缺少严格镜像证书时必须 fail-closed。

    旧资产只把 ``handedness: left`` 当标签，并未保证 palm、全部 mounts、joint
    axes、mesh 与惯量满足完整反射合同。默认拒绝可防止它们重新进入训练集合。
    """

    run_root = tmp_path / "post_mutate"
    _write_sample(run_root, "legacy_left", handedness="left", include_handedness_contract=False)

    with pytest.raises(ValueError, match="legacy generated left"):
        HandBank(
            HandBankCfg(
                post_mutate_path=run_root,
                selection_mode="explicit",
                containers=("legacy_left",),
            )
        ).resolve()


def test_generated_left_legacy_override_is_explicit_and_local(tmp_path: Path) -> None:
    r"""研究者显式开启 legacy override 后，旧 left 可用于审计但不改变默认值。"""

    run_root = tmp_path / "post_mutate"
    _write_sample(run_root, "legacy_left", handedness="left", include_handedness_contract=False)

    selection = HandBank(
        HandBankCfg(
            post_mutate_path=run_root,
            selection_mode="explicit",
            containers=("legacy_left",),
            allow_legacy_left_handedness=True,
        )
    ).resolve()

    assert [asset.asset_id for asset in selection.assets] == ["legacy_left"]  # override 只放行当前 bank 实例


def test_generated_right_without_handedness_contract_remains_readable(tmp_path: Path) -> None:
    r"""Legacy generated right 是 canonical 真源，不因缺少新 left-lowering 证书被拒绝。"""

    run_root = tmp_path / "post_mutate"
    _write_sample(run_root, "legacy_right", handedness="right", include_handedness_contract=False)

    selection = HandBank(
        HandBankCfg(
            post_mutate_path=run_root,
            selection_mode="explicit",
            containers=("legacy_right",),
        )
    ).resolve()

    assert [asset.asset_id for asset in selection.assets] == ["legacy_right"]


def test_official_left_without_generated_contract_is_not_rejected(tmp_path: Path) -> None:
    r"""Official left 使用自身人工资产合同，不套用 generated legacy gate。"""

    run_root = tmp_path / "official"
    sample_dir = _write_sample(run_root, "official_left", handedness="left", include_handedness_contract=False)

    container = HandContainer.from_cfg(
        HandContainerCfg(path=sample_dir, source_kind="official"),
    )

    assert container.source_kind == "official"  # source_kind 是 gate 的权威边界，不从目录名猜测


def test_generated_left_with_strict_handedness_contract_is_readable(tmp_path: Path) -> None:
    r"""新 generated left 携带完整 contract v1 时应通过默认安全门。"""

    run_root = tmp_path / "post_mutate"
    _write_sample(run_root, "strict_left", handedness="left", include_handedness_contract=True)

    selection = HandBank(
        HandBankCfg(
            post_mutate_path=run_root,
            selection_mode="explicit",
            containers=("strict_left",),
        )
    ).resolve()

    assert selection.assets[0].sidecar["handedness_contract"]["same_q"] is True  # 证书保留给下游审计


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


def test_pre_made_discovers_only_self_contained_topology_bundles(tmp_path: Path) -> None:
    r"""pre-made 上级 root 可发现多个母体，但不得误收共享 mesh 的 post-mutate leaf。"""

    first = _write_source_topology(tmp_path / "leap_topology", asset_id="leap_root", mesh_name="leap.stl")
    second = _write_source_topology(tmp_path / "allegro_topology", asset_id="allegro_root", mesh_name="allegro.stl")
    run_root = first / "post_mutate_run"
    _write_sample(run_root, "variant", mesh_name="variant.stl")

    selection = HandBank(
        HandBankCfg(source_mode="pre_made", pre_made_path=tmp_path, selection_mode="all")
    ).resolve()

    assert [asset.asset_id for asset in selection.assets] == ["allegro_root", "leap_root"]
    assert all(asset.real_path("hand.urdf").parent in {first, second} for asset in selection.assets)


def test_mixed_uses_explicit_cross_family_manifest_and_stable_sampling(tmp_path: Path) -> None:
    r"""mixed 不猜共同目录；任意 family bundle manifest 可 all 或固定 seed 无放回采样。"""

    leap = _write_source_topology(tmp_path / "leap", asset_id="leap_root", mesh_name="leap.stl")
    allegro = _write_source_topology(tmp_path / "allegro", asset_id="allegro_root", mesh_name="allegro.stl")
    all_selection = HandBank(
        HandBankCfg(source_mode="mixed", selection_mode="all", containers=(leap, allegro))
    ).resolve()
    sample_selection = HandBank(
        HandBankCfg(
            source_mode="mixed",
            selection_mode="sample",
            sample_count=1,
            sample_seed=61,
            containers=(leap, allegro),
        )
    ).resolve()

    assert [asset.asset_id for asset in all_selection.assets] == ["allegro_root", "leap_root"]
    assert sample_selection.assets[0].asset_id in {"allegro_root", "leap_root"}
