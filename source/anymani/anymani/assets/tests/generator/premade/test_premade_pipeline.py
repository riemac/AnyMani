"""pre-made 闭环的 validator / exporter / generator 测试。

这组测试把首轮最关键的纵向契约锁住：

1. `HandCfg -> HandValidator`
2. `HandCfg -> UrdfWriter`
3. `HandGenerator.generate()` 在不启用 mutate 时能稳定产出 bundle

测试设计上尽量复用同一份 Allegro 锚点 hand，避免因为测试样本漂移掩盖接口问题。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import assets.generator.hand_generator as hand_generator_module
import assets.generator.premade.batch as premade_batch_module
import pytest
import yaml
from assets.bank import HandContainer, HandContainerCfg
from assets.builder.hand_builders import HumanLikeHandBuilder, HumanLikeHandBuilderCfg
from assets.exporter.urdf_writer import UrdfWriter, UrdfWriterCfg
from assets.generator.hand_generator import HandGenerator, HandGeneratorCfg
from assets.presets import make_human_like_builder_cfg
from assets.procedural_meshes import materialize_hand_procedural_meshes
from assets.validator.hand_rules import HandValidator, HandValidatorCfg


def _make_allegro_builder_cfg() -> HumanLikeHandBuilderCfg:
    """与 builder 测试共用的一份 Allegro hand recipe。"""

    return make_human_like_builder_cfg(
        name="allegro_demo",
        family="allegro",
        handedness="right",
        palm_cfg="com_allegro",
        finger_cfg="allegro_non_thumb_v1",
        thumb_cfg="allegro_thumb_v1",
    )


def _build_allegro_hand():
    """构造一份稳定的整手 `HandCfg`，供纵向测试复用。"""

    return HumanLikeHandBuilder(_make_allegro_builder_cfg()).build()


def _parse_triplet(text: str) -> tuple[float, float, float]:
    r"""把 URDF 里的 `\"x y z\"` / `\"r p y\"` 三元串解析成浮点 tuple。"""

    return tuple(float(value) for value in text.split())  # type: ignore[return-value]


def test_post_mutate_hand_validator_rejects_sdf_clearance_violation(tmp_path):
    """post-mutate spacing 现在是 SDF clearance 硬闸门，不再 warning 放行。"""

    hand, _written = materialize_hand_procedural_meshes(
        _build_allegro_hand(),
        mesh_root_dir=tmp_path / "meshes",
    )
    validator = HandValidator(
        HandValidatorCfg(
            post_mutate=HandValidatorCfg.PostMutateCfg(min_finger_spacing=0.05)
        )
    )

    result = validator.validate(hand)

    assert result.passed is False
    assert any("sdf_clearance" in error for error in result.errors)
    assert result.metadata["finger_spacing_certificate"]["pose_scope"] == "post_mutate_home_pose"


def test_urdf_writer_always_folds_mount_into_first_joint_origin():
    r"""整手 URDF 应始终用第一关节 `origin` 表达挂载位姿。

    这里锁住的是你刚确认过的官方语义：

    - 不允许再出现 `*_mount_link` / `*_mount_joint`；
    - `FingerCfg.mount` 必须直接折叠进 finger 链第一个 joint 的 `origin`。
    """

    hand = _build_allegro_hand()
    writer = UrdfWriter(UrdfWriterCfg())

    root = ET.fromstring(writer.to_urdf_string(hand))
    joint_elems = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    link_names = {link.attrib["name"] for link in root.findall("link")}
    index_finger = next(finger for finger in hand.fingers if finger.name == "index")
    first_joint = index_finger.joints[0]

    expected_pos = tuple(
        index_finger.mount.pos[axis] + first_joint.origin.pos[axis]
        for axis in range(3)
    )  # 挂载平移应直接吸收到第一关节局部 origin
    expected_rpy = tuple(
        index_finger.mount.rpy[axis] + first_joint.origin.rpy[axis]
        for axis in range(3)
    )  # 挂载姿态同样应折叠到第一关节的 RPY 上

    assert "index_mount_link" not in link_names
    assert "index_mount_joint" not in joint_elems
    assert joint_elems["index_j0"].find("parent").attrib["link"] == hand.palm.name
    assert _parse_triplet(joint_elems["index_j0"].find("origin").attrib["xyz"]) == pytest.approx(expected_pos)
    assert _parse_triplet(joint_elems["index_j0"].find("origin").attrib["rpy"]) == pytest.approx(expected_rpy)


def test_urdf_writer_serializes_joint_properties_friction_for_leap_profile():
    r"""LEAP official profile 中的 joint friction 应写成 `<joint_properties>`。"""

    hand = HumanLikeHandBuilder(
        make_human_like_builder_cfg(
            name="leap_joint_properties_demo",
            family="leap",
            handedness="right",
            palm_cfg="single_box_leap",
            finger_cfg="leap_non_thumb_v1",
            thumb_cfg="leap_thumb_v1",
        )
    ).build()
    writer = UrdfWriter(UrdfWriterCfg())

    root = ET.fromstring(writer.to_urdf_string(hand))
    link_names = {link.attrib["name"] for link in root.findall("link")}
    joint_elems = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    index_j0 = joint_elems["index_j0"]
    index_j1 = joint_elems["index_j1"]

    assert "index_mount_link" not in link_names
    assert "index_mount_joint" not in joint_elems
    assert index_j0.find("limit").attrib == {
        "lower": "-0.314",
        "upper": "2.23",
        "effort": "0.95",
        "velocity": "8.48",
    }
    assert index_j0.find("joint_properties").attrib == {"friction": "0"}
    assert index_j1.find("limit").attrib == {
        "lower": "-1.047",
        "upper": "1.047",
        "effort": "0.95",
        "velocity": "8.48",
    }
    assert index_j1.find("joint_properties").attrib == {"friction": "0"}
    assert index_j0.find("parent").attrib["link"] == "index_root_fixed_link"


def test_hand_generator_rejects_full_mode_until_topology_root_migration_is_finished(tmp_path):
    """`mode=\"full\"` 当前应显式报不支持，而不是悄悄沿用旧目录语义。"""

    cfg = HandGeneratorCfg(
        mode="full",
        artifact_level="bundle",
        output_dir=tmp_path,
        Made=_make_allegro_builder_cfg(),
    )

    with pytest.raises(NotImplementedError, match="mode='full' is temporarily unsupported"):
        HandGenerator(cfg).generate()


def test_hand_generator_can_explicitly_skip_hand_level_validator(tmp_path):
    r"""`Validate=None` 时，generator 应跳过 hand-level validator，而不是隐式启用默认规则。

    这里故意构造一只“缺拇指”的 hand：

    - 若 pre-made validator 开着，当前 contract 下它必须被拒绝；
    - 若 `Validate=None`，则应沿着“只看 builder / exporter，不做 hand-level 合法性闸门”的
      研究排查路径继续产出。
    """

    invalid_made_cfg = _make_allegro_builder_cfg().replace(thumb_cfg=None)

    validated_result = HandGenerator(
        HandGeneratorCfg(
            mode="made",
            artifact_level="hand_cfg",
            output_dir=tmp_path / "validated",
            Made=invalid_made_cfg,
            Validate=HandValidatorCfg(
                pre_made=HandValidatorCfg.PreMadeCfg(
                    check_finger_spacing=False,
                )
            ),
        )
    ).generate()

    skipped_result = HandGenerator(
        HandGeneratorCfg(
            mode="made",
            artifact_level="hand_cfg",
            output_dir=tmp_path / "skipped",
            Made=invalid_made_cfg,
            Validate=None,
        )
    ).generate()

    assert validated_result is None
    assert skipped_result is not None
    assert skipped_result.hand_cfg is not None


def _single_family_full_pool(hand_preset: str, family: str) -> dict[str, dict[str, list[str]]]:
    r"""构造一份单 topology 的 full-chain connectivity pool。

    这个小 helper 用于把并行测试的变量压到最低：

    - 只测 generator 的任务级并行；
    - 不把 mixed / missing / connectivity registry 全空间也卷进来。
    """

    thumb_recipe = f"{family}_thumb_full"  # thumb 与 palm family 绑定，避免 validator 因 family 错配拒绝
    non_thumb_recipe = f"{family}_non_thumb_full"  # non-thumb 保持完整三指，形成一个稳定合法样本
    return {
        hand_preset: {
            "thumb": [thumb_recipe],
            "index": [non_thumb_recipe],
            "middle": [non_thumb_recipe],
            "ring": [non_thumb_recipe],
        }
    }


def _single_family_low_dof_pool(hand_preset: str, family: str) -> dict[str, dict[str, list[str]]]:
    r"""构造所有 non-thumb 均低于 3 revolute DOF 的确定性拒绝样本。

    thumb 保留 full chain，避免引入无关的 thumb 规则；三根 non-thumb 都删除
    ``j2/j3``，使每根只剩 2 revolute DOF，从而只命中
    ``hand.non_thumb_revolute_dof_below_min``。
    """

    low_dof_recipe = f"{family}_non_thumb_drop_j2_j3"  # 每根 non-thumb 剩余 2 个 revolute DOF
    return {
        hand_preset: {
            "thumb": [f"{family}_thumb_full"],
            "index": [low_dof_recipe],
            "middle": [low_dof_recipe],
            "ring": [low_dof_recipe],
        }
    }


def test_premade_generate_batch_parallelizes_sample_level_and_main_process_writes_summary(tmp_path):
    r"""pre-made 默认并行应保持“worker 产样本、主进程写 summary”的边界。

    这里锁住的不是绝对速度，而是并行语义：

    - worker 可以独立完成 build / validator / export；
    - run-level `summary.yaml` 只能由主进程汇总写出；
    - 成功数、attempted 数与产物数一致。
    """

    cfg = HandGeneratorCfg(
        mode="made",
        artifact_level="bundle",
        output_dir=tmp_path,
        handedness="all",
        hand_presets=["single_palm_leap"],
        connectivity_presets=_single_family_full_pool("single_palm_leap", "leap"),
        mixed=False,
        missing=False,
        max_enumerate=2,
        premade_parallel=True,
        premade_parallel_workers=2,
    )

    results = list(HandGenerator(cfg).generate_batch())

    assert len(results) == 2
    assert all(result.urdf_path is not None and result.urdf_path.is_file() for result in results)
    summary_paths = list(tmp_path.glob("*/summary.yaml"))
    assert len(summary_paths) == 1
    summary = yaml.safe_load(summary_paths[0].read_text(encoding="utf-8"))
    assert summary["stats"]["attempted"] == 2
    assert summary["stats"]["succeeded"] == 2
    assert summary["stats"]["rejected"] == 0


@pytest.mark.parametrize(
    ("premade_parallel", "handedness", "expected_rejections"),
    ((False, "right", 1), (True, "all", 2)),
)
def test_premade_rejection_records_reason_and_removes_materialized_meshes(
    tmp_path,
    *,
    premade_parallel: bool,
    handedness: str,
    expected_rejections: int,
):
    r"""串行和并行 pre-made rejection 都不得留下候选期 OBJ 或空目录。

    canonical LEAP builder 会生成 procedural `cs` fingertip；因此这个测试真实经过
    ``materialize -> physics closure -> validator rejection``，而不是用人工空文件
    模拟。左右手并行时两个 worker 应各拒绝一次，主进程按同一稳定原因代码汇总。
    """

    cfg = HandGeneratorCfg(
        mode="made",
        artifact_level="bundle",
        output_dir=tmp_path,
        handedness=handedness,
        hand_presets=["single_palm_leap"],
        connectivity_presets=_single_family_low_dof_pool("single_palm_leap", "leap"),
        mixed=False,
        missing=False,
        Validate=HandValidatorCfg(
            pre_made=HandValidatorCfg.PreMadeCfg(
                check_finger_spacing=False,
                require_non_thumb_with_min_revolute_dof=3,
            )
        ),
        premade_parallel=premade_parallel,
        premade_parallel_workers=2 if premade_parallel else None,
    )

    results = list(HandGenerator(cfg).generate_batch())
    run_root = next(path for path in tmp_path.iterdir() if path.is_dir())
    summary = yaml.safe_load((run_root / "summary.yaml").read_text(encoding="utf-8"))

    assert results == []
    assert summary["stats"]["attempted"] == expected_rejections
    assert summary["stats"]["rejected"] == expected_rejections
    assert summary["stats"]["rejected_by_reason"] == {
        "hand.non_thumb_revolute_dof_below_min": expected_rejections
    }
    assert list(run_root.rglob("*.obj")) == []
    assert [path for path in run_root.rglob("*") if path.is_dir()] == []


@pytest.mark.parametrize("failure_stage", ("physics", "validator", "export"))
def test_premade_exception_rolls_back_new_materialized_meshes(monkeypatch, tmp_path, failure_stage: str):
    r"""physics、validator 或 export 异常都必须回滚本候选新物化的 OBJ。

    三个异常点覆盖 materialization 后的完整生命周期。异常本身继续向调用者传播，
    这里只锁住文件事务：失败候选不能伪装成 generated asset，也不能留下空 topology
    层级。
    """

    def _raise_synthetic_failure(*_args, **_kwargs):
        raise RuntimeError(f"synthetic {failure_stage} failure")

    def _write_partial_urdf_then_raise(_self, _result, output_dir, *_args, **_kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "hand.urdf").write_text("<robot name='partial'/>\n", encoding="utf-8")
        raise RuntimeError("synthetic export failure")

    if failure_stage == "physics":
        monkeypatch.setattr(hand_generator_module, "close_hand_physics", _raise_synthetic_failure)
    elif failure_stage == "validator":
        monkeypatch.setattr(hand_generator_module.HandValidator, "validate_pre_made", _raise_synthetic_failure)
    else:
        monkeypatch.setattr(hand_generator_module.HandExporter, "export", _write_partial_urdf_then_raise)

    cfg = HandGeneratorCfg(
        mode="made",
        artifact_level="bundle",
        output_dir=tmp_path,
        handedness="right",
        hand_presets=["single_palm_leap"],
        connectivity_presets=_single_family_full_pool("single_palm_leap", "leap"),
        mixed=False,
        missing=False,
        Validate=HandValidatorCfg(
            pre_made=HandValidatorCfg.PreMadeCfg(check_finger_spacing=False)
        ),
        premade_parallel=False,
    )

    with pytest.raises(RuntimeError, match=f"synthetic {failure_stage} failure"):
        list(HandGenerator(cfg).generate_batch())

    run_root = next(path for path in tmp_path.iterdir() if path.is_dir())
    assert list(run_root.rglob("*.obj")) == []
    assert [path for path in run_root.rglob("*") if path.is_dir()] == []


def test_left_custom_mesh_is_rolled_back_when_physics_fails(monkeypatch, tmp_path):
    r"""Left custom mesh 在 physics 前首次镜像后，失败候选必须回滚该共享文件。

    该测试与 procedural ``cs`` 回滚不同：输入是项目内已有的非对称 custom STL，
    materializer 会在当前 run 的 ``meshes/`` 下新建内容哈希镜像文件。physics 失败
    后只删除本候选首次发布的镜像副本，canonical source mesh 必须保持存在。
    """

    def _raise_synthetic_failure(*_args, **_kwargs):
        raise RuntimeError("synthetic reflected mesh physics failure")

    custom_finger_cfg = make_human_like_builder_cfg(
        name="left_custom_mesh_rollback",
        family="leap",
        handedness="left",
        palm_cfg="single_box_leap",
        finger_cfg="leap_non_thumb_v1",
        thumb_cfg="leap_thumb_v1",
    )
    # 用 typed finger cfg 覆盖默认 cs tip，确保本测试经过普通 custom mesh reflection 分支。
    custom_finger_cfg = custom_finger_cfg.replace(
        finger_cfg=custom_finger_cfg.finger_cfg.replace(
            tip={"type": "mesh", "tip_type": "wedge", "scale": 1.0},
        ),
        thumb_cfg=custom_finger_cfg.thumb_cfg.replace(
            tip={"type": "mesh", "tip_type": "wedge", "scale": 1.0},
        ),
    )
    built_left = HumanLikeHandBuilder(custom_finger_cfg).build()  # 读取 builder 实际解析后的默认 custom tip source
    source_mesh = Path(
        next(finger for finger in built_left.fingers if finger.name == "index")
        .tip_joint.collisions[0]
        .geometry.file_path
    )
    assert source_mesh.is_file()  # 本测试必须真实经过普通 custom STL reflection 分支
    monkeypatch.setattr(hand_generator_module, "close_hand_physics", _raise_synthetic_failure)

    cfg = HandGeneratorCfg(
        mode="made",
        artifact_level="bundle",
        output_dir=tmp_path,
        Made=custom_finger_cfg,
        Validate=None,
    )

    with pytest.raises(RuntimeError, match="synthetic reflected mesh physics failure"):
        HandGenerator(cfg).generate()

    run_root = next(path for path in tmp_path.iterdir() if path.is_dir())
    assert list(run_root.rglob("*_yz_reflect_v1_*")) == []  # 候选期镜像 mesh 已由 written_paths 回滚
    assert source_mesh.is_file()  # canonical source 不属于 run 回滚边界


def test_left_custom_mesh_bundle_closes_handedness_contract_end_to_end(tmp_path):
    r"""Generator 应把 left custom mesh、physics、URDF、sidecar 与 Bank 闭合为同一事实。

    该用例覆盖完整顺序：

    $$
    \text{canonical build}\to\text{left lowering}\to\text{mesh materialize}
    \to\text{physics closure}\to\text{URDF/sidecar}\to\text{HandBank}.
    $$

    最终 sidecar 不得残留待处理反射标记，URDF 必须引用内容哈希镜像 mesh，且
    Bank 在默认 ``allow_legacy_left_handedness=False`` 下可直接读取新证书资产。
    """

    builder_cfg = make_human_like_builder_cfg(
        name="left_custom_mesh_bundle",
        family="leap",
        handedness="left",
        palm_cfg="single_box_leap",
        finger_cfg="leap_non_thumb_v1",
        thumb_cfg="leap_thumb_v1",
    )
    builder_cfg = builder_cfg.replace(
        finger_cfg=builder_cfg.finger_cfg.replace(
            tip={"type": "mesh", "tip_type": "wedge", "scale": 1.0},
        ),
        thumb_cfg=builder_cfg.thumb_cfg.replace(
            tip={"type": "mesh", "tip_type": "wedge", "scale": 1.0},
        ),
    )  # 全部 fingertips 使用非对称 custom STL，覆盖 non-thumb/thumb 功能相位
    generator = HandGenerator(
        HandGeneratorCfg(
            mode="made",
            artifact_level="bundle",
            output_dir=tmp_path,
            Made=builder_cfg,
            Validate=None,
        )
    )

    result = generator.generate()

    assert result is not None and result.urdf_path is not None and result.sidecar_path is not None
    sidecar = yaml.safe_load(result.sidecar_path.read_text(encoding="utf-8"))
    urdf_text = result.urdf_path.read_text(encoding="utf-8")
    assert sidecar["handedness_contract"]["target_handedness"] == "left"
    assert sidecar["handedness_contract"]["same_q"] is True
    assert "_yz_reflect_v1_" in urdf_text  # URDF 引用最终镜像 mesh，而非 canonical source basename

    hand_snapshot = sidecar["hand_cfg"]
    all_mesh_geometries = [
        element["geometry"]
        for finger in hand_snapshot["fingers"]
        for joint in finger["joints"]
        for element in [*joint["collisions"], *joint["visuals"]]
        if "file_path" in element["geometry"]
    ]
    assert all(geometry["reflected_about_yz"] is False for geometry in all_mesh_geometries)
    assert any("_yz_reflect_v1_" in geometry["file_path"] for geometry in all_mesh_geometries)

    container = HandContainer.from_cfg(HandContainerCfg(path=result.urdf_path.parent))
    assert container.sidecar["handedness"] == "left"  # 新 strict left 通过默认 legacy 安全门


def test_premade_parallel_failure_falls_back_to_serial(monkeypatch, tmp_path):
    r"""进程池路径异常时，generator 应回退到原串行枚举。

    这个测试模拟的是 worker / pickle / executor 环境失败，而不是样本本身非法。
    样本非法应由 validator 记录拒绝；并行基础设施失败才触发 serial fallback。
    """

    def _raise_parallel_failure(self, *, tasks):
        raise RuntimeError("synthetic parallel executor failure")

    monkeypatch.setattr(premade_batch_module, "run_premade_parallel", _raise_parallel_failure)

    cfg = HandGeneratorCfg(
        mode="made",
        artifact_level="hand_cfg",
        output_dir=tmp_path,
        handedness="right",
        hand_presets=["single_palm_allegro"],
        connectivity_presets=_single_family_full_pool("single_palm_allegro", "allegro"),
        mixed=False,
        missing=False,
        max_enumerate=1,
        premade_parallel=True,
        premade_parallel_fallback="serial",
    )

    results = list(HandGenerator(cfg).generate_batch())

    assert len(results) == 1
    summary_paths = list(tmp_path.glob("*/summary.yaml"))
    assert len(summary_paths) == 1
    summary = yaml.safe_load(summary_paths[0].read_text(encoding="utf-8"))
    assert summary["stats"]["attempted"] == 1
    assert summary["stats"]["succeeded"] == 1
