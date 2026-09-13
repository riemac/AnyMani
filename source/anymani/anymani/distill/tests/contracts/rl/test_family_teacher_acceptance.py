r"""家族教师重复验收的最小契约测试。

测试把“单次 family_teacher 归约”和“两次结果的重复归约”分开。
纯 reducer 夹具不启动 Isaac，也不构造 reward 或逐步轨迹；文件 wrapper
只用临时 JSON、canonical lock 和可选 HDF5 路径验证身份/去重边界。

固定参照是原始 A128、32 个拓扑、每拓扑 4 个成员、每资产 R16。
新的重复门是两次一圈可靠 row 交集至少 86；旧 103/29 统计必须逐次原样保留。
两圈仅用于描述，不能把两圈结果替代一圈重复门。
"""

from __future__ import annotations

import copy  # 证明发布报告不会原地改写旧的强门对象。
import hashlib  # 绑定临时 canonical lock 的原始字节身份。
import importlib.util  # 直接加载被测文件，绕开 anymani 包初始化。
import json  # 构造有限的 evaluation identity。
from pathlib import Path  # 所有测试写入 pytest 临时目录。
from types import SimpleNamespace  # wrapper 测试用轻量 fake family_teacher。
from typing import Any  # 合成 JSON 报告的异构叶节点。

import pytest  # 契约边界与非法证据断言。
import torch  # 仅保存CPU metadata夹具，不构造模型或启动Isaac。

# 测试模块与实现文件使用稳定的相对位置，并不依赖 Python 包搜索路径。
_SOURCE = (
    Path(__file__).resolve().parents[3] / "diagnostics/analysis/rl/family_teacher_acceptance.py"
)  # 两个新文件的固定相对路径。
_SPEC = importlib.util.spec_from_file_location("family_teacher_acceptance_contract", _SOURCE)  # raw-module 导入边界。
assert _SPEC is not None and _SPEC.loader is not None, "重复验收实现必须可独立加载"
acceptance = importlib.util.module_from_spec(_SPEC)  # 不经过 anymani.__init__ 或 Isaac 注册。
_SPEC.loader.exec_module(acceptance)  # 只执行纯文件分析入口。

_METHOD = "a" * 64  # 测试用 method identity SHA-256。
_CHECKPOINT = "b" * 64  # 测试用冻结 checkpoint SHA-256。


def _thresholds() -> dict[str, float | int]:
    r"""返回固定 R 的阈值字典；一圈与两圈只改变净圈目标。"""

    return {
        "horizon_s": 30.0,  # 固定首轨迹的时间窗口，单位秒。
        "replicas_per_asset": 16,  # 安全比例分母。
        "net_turns_min": 1.0,  # 一圈可靠门的净圈下界。
        "directional_consistency_min": 0.7,  # 中位净圈/路径圈的方向门。
        "safe_replica_fraction_min": 0.75,  # 12/16 副本联合安全。
        "topology_representative_fraction_min": 0.5,  # 4代表拓扑至少2个。
    }  # 原始固定评价的数值锚点。


def _asset_results(
    one_rows: set[int],
    two_rows: set[int],
    *,
    bad_direction_rows: set[int] | None = None,
    bad_safe_rows: set[int] | None = None,
) -> list[dict[str, Any]]:
    r"""生成 128 行有限 R16 资产中心，不生成任何 reward 或原始轨迹。

    通过 row 的净圈取 1 或 2 圈，失败 row 取 0.4 圈；方向与安全默认
    为固定门边界值。指定坏方向/坏安全行用于证伪“仅凭 row 集合计数”。
    """

    bad_direction_rows = set() if bad_direction_rows is None else set(bad_direction_rows)  # 不改变调用方集合。
    bad_safe_rows = set() if bad_safe_rows is None else set(bad_safe_rows)  # 不改变调用方集合。
    assets: list[dict[str, Any]] = []  # 128 个 selection-local 资产结果。
    for row in range(128):  # 原始 A128 分母不能被测试夹具动态缩小。
        net = 2.0 if row in two_rows else 1.0 if row in one_rows else 0.4  # 两圈仅提高净圈目标。
        direction = 0.69 if row in bad_direction_rows else 0.8  # 方向不足不能成为可靠资产。
        safe = 11 / 16 if row in bad_safe_rows else 0.75  # 少于12/16不能成为可靠资产。
        assets.append(
            {
                "dataset_row": row,  # 交集的唯一 selection-local 键。
                "topology_id": f"family/mother-{row // 4:02d}",  # 32拓扑，每个4代表。
                "mother_id": f"mother-{row // 4:02d}",  # 与旧 family_teacher 结果一致的母体标签。
                "replica_count": 16,  # 每资产固定副本分母。
                "finite": True,  # 所有数值是有限 CPU 标量。
                "net_turns_median": net,  # 副本中位后的有符号净圈。
                "absolute_path_turns_median": net / 0.8,  # 使默认方向性为0.8。
                "directional_consistency": direction,  # 归约后的方向性。
                "safe_replica_fraction": safe,  # 联合安全副本比例。
            }
        )  # 测试只构造已归约的资产中心。
    return assets  # 返回完整有序资产轴。


def _coverage(rows: set[int], turns: int) -> dict[str, Any]:
    r"""构造既有 family_teacher 结果的 coverage 外壳。"""

    thresholds = _thresholds()  # 一圈阈值是旧可靠门的固定值。
    thresholds["net_turns_min"] = float(turns)  # two-turn 只提高净圈目标。
    ordered = sorted(rows)  # JSON 通过集合使用稳定 row 顺序。
    table = [
        {
            "topology_id": f"family/mother-{group:02d}",  # 每组4个row。
            "asset_count": 4,  # 拓扑代表分母。
            "passed_asset_count": sum(row in rows for row in range(group * 4, group * 4 + 4)),  # 组内通过代表票数。
            "required_asset_count": 2,  # ceil(4×0.5)=2。
            "passed": sum(row in rows for row in range(group * 4, group * 4 + 4)) >= 2,  # 仅供报告可读性。
        }
        for group in range(32)  # 保留完整32拓扑表。
    ]
    return {
        "schema_version": "1.0.0",  # family_teacher coverage schema。
        "thresholds": thresholds,  # 一圈/两圈的独立门声明。
        "finite": True,  # 归约统计有限。
        "asset_count": 128,  # 原始资产分母。
        "topology_count": 32,  # 原始拓扑分母。
        "passed_asset_count": len(ordered),  # 必须与通过row集合闭合。
        "passed_asset_rows": ordered,  # 新入口按此集合做交集。
        "passed_topology_count": sum(item["passed"] for item in table),  # 仅保留完整统计。
        "asset_fraction": len(ordered) / 128,  # 固定分母比例。
        "topology_fraction": sum(item["passed"] for item in table) / 32,  # 固定拓扑分母比例。
        "topology_results": table,  # 旧报告的逐拓扑事实。
    }  # 不包含任何未审计的原始奖励字段。


def _report(
    one_rows: set[int],
    two_rows: set[int] | None = None,
    *,
    bad_direction_rows: set[int] | None = None,
    bad_safe_rows: set[int] | None = None,
    historical_asset_passed: bool = False,
    historical_topology_passed: bool = False,
) -> dict[str, Any]:
    r"""构造最小完整 family_teacher 报告供纯 reducer 测试。"""

    two_rows = set(one_rows) if two_rows is None else set(two_rows)  # 默认两圈集合与一圈相同。
    assets = _asset_results(
        one_rows, two_rows, bad_direction_rows=bad_direction_rows, bad_safe_rows=bad_safe_rows
    )  # 完整有限资产轴。
    return {
        "artifact_type": "anymani.family_teacher.fixed_gate_analysis",  # 旧分析产物类型。
        "schema_version": "1.0.0",  # 旧分析报告版本。
        "strong_gate": {
            "required_assets": 103,  # 历史一圈资产门。
            "required_topologies": 29,  # 历史一圈拓扑门。
            "asset_gate_passed": historical_asset_passed,  # 旧报告原布尔事实。
            "topology_gate_passed": historical_topology_passed,  # 旧报告原布尔事实。
        },  # 新入口不会用此门代替86交集门。
        "one_turn": _coverage(one_rows, 1),  # 旧一圈可靠集合。
        "two_turn": _coverage(two_rows, 2),  # 旧两圈描述集合。
        "asset_results": assets,  # 128×R16 已归约的物理中心。
    }  # 夹具故意不放 reward/trace 原始内容。


def test_85_common_assets_fail_and_86_common_assets_pass() -> None:
    r"""共同一圈 row 为85时失败，达到86时通过新重复门。"""

    report_85 = _report(set(range(85)))  # 两次都只有85个共同可靠资产。
    failed = acceptance.reduce_repeat_acceptance(report_85, report_85)  # 交集仍是85。
    assert failed["repeat_gate"]["common_passed_assets"] == 85, failed  # 原128分母不被缩小。
    assert failed["repeat_gate"]["passed"] is False, failed  # 85不能晋级。

    report_86 = _report(set(range(86)))  # 两次冻结评价共同有86个可靠资产。
    passed = acceptance.reduce_repeat_acceptance(report_86, report_86)  # 相同内容但此处是纯报告归约。
    assert passed["repeat_gate"]["common_passed_assets"] == 86, passed  # 恰好边界应包含等号。
    assert passed["repeat_gate"]["passed"] is True, passed  # 新门的默认required_assets=86。
    assert len(passed["common_reliable_asset_rows"]) == 86, passed  # 返回可复核的具体row集合。


def test_each_run_86_but_intersection_85_fails() -> None:
    r"""两次各自86个通过但共同只有85个时仍失败。"""

    first = _report(set(range(86)))  # 第一次通过0..85。
    second = _report(set(range(85)) | {86})  # 第二次替换最后一个row，单独仍为86。
    result = acceptance.reduce_repeat_acceptance(first, second)  # 只按dataset_row求交集。
    assert result["repeat_gate"]["first_passed_assets"] == 86, result  # 第一次单独门。
    assert result["repeat_gate"]["second_passed_assets"] == 86, result  # 第二次单独门。
    assert result["repeat_gate"]["common_passed_assets"] == 85, result  # 共同可重复能力只有85。
    assert result["repeat_gate"]["passed"] is False, result  # 不以两次单独结果的并集冒充重复门。


def test_direction_or_safe_deficit_cannot_be_counted() -> None:
    r"""一圈净圈足够但方向或安全不足的资产不能进入通过集合。"""

    with pytest.raises(
        acceptance.AcceptanceError, match="N/D/S|passed_asset_rows"
    ):  # 失败应是证据不一致而不是隐式计数。
        acceptance.reduce_repeat_acceptance(
            _report(set(range(86)), bad_direction_rows={0}),  # 自报row0通过但D=0.69。
            _report(set(range(86)), bad_direction_rows={0}),
        )
    with pytest.raises(acceptance.AcceptanceError, match="N/D/S|passed_asset_rows"):  # 11/16安全同样不能计数。
        acceptance.reduce_repeat_acceptance(
            _report(set(range(86)), bad_safe_rows={0}),  # 自报row0通过但S=11/16。
            _report(set(range(86)), bad_safe_rows={0}),
        )


def test_two_turn_intersection_is_descriptive_and_never_replaces_one_turn() -> None:
    r"""两圈集合即使存在，也不能把85个一圈共同通过提升为86个。"""

    first = _report(set(range(85)), set(range(80)))  # 一圈85、两圈是其真子集。
    second = _report(set(range(85)), set(range(80)))  # 两次的two-turn都独立保留。
    result = acceptance.reduce_repeat_acceptance(first, second)  # 新门只看one_turn交集。
    assert result["two_turn_intersection"]["passed_asset_count"] == 80, result  # two-turn交集单独可见。
    assert result["repeat_gate"]["common_passed_assets"] == 85, result  # 一圈分母和集合不被替换。
    assert result["repeat_gate"]["passed"] is False, result  # two-turn不补一圈缺口。
    assert result["two_turn_intersection"]["substitutes_one_turn"] is False, result  # 输出明确声明边界。


def test_old_103_29_statistics_are_copied_without_mutation() -> None:
    r"""旧103/29强门逐次保留，不能被新86门覆盖或改写。"""

    first = _report(
        set(range(86)), historical_asset_passed=True, historical_topology_passed=False
    )  # 旧门是独立历史事实。
    second = _report(set(range(86)), historical_asset_passed=False, historical_topology_passed=True)  # 第二次旧门相反。
    before_first = copy.deepcopy(first["strong_gate"])  # 记录调用前的旧字段。
    before_second = copy.deepcopy(second["strong_gate"])  # 记录调用前的旧字段。
    result = acceptance.reduce_repeat_acceptance(first, second)  # 新入口只读并复制。
    assert first["strong_gate"] == before_first, first  # 输入对象未被原地更新。
    assert second["strong_gate"] == before_second, second  # 输入对象未被原地更新。
    assert result["historical_103_29_gate"]["runs"] == [before_first, before_second], (
        result
    )  # 两次历史门在唯一命名下分别保留。
    assert result["repeat_gate"]["required_assets"] == 86, result  # 新门使用自己的阈值字段。


@pytest.mark.parametrize("fault", ["missing", "duplicate"])
def test_missing_or_duplicate_asset_rows_are_rejected(fault: str) -> None:
    r"""完整128资产轴缺失或重复row时拒绝，而不是缩小分母。"""

    report = _report(set(range(86)))  # 先构造完整合法资产轴。
    if fault == "missing":  # 删除一行会形成127行。
        report["asset_results"].pop()  # 只修改临时夹具。
    else:  # duplicate row 保持128行但破坏一一映射。
        report["asset_results"][-1]["dataset_row"] = 0  # 0已经存在，必须失败。
    with pytest.raises(acceptance.AcceptanceError, match="128|重复|dataset_row"):  # 错误指出原始分母/row边界。
        acceptance.reduce_repeat_acceptance(report, report)  # 纯 reducer 不过滤坏行。


def _protocol() -> dict[str, Any]:
    r"""返回文件 wrapper 需要的固定30秒/R16/mean协议。"""

    protocol: dict[str, Any] = {
        "num_assets": 128,  # 固定资产宇宙。
        "policy_steps": 600,  # 600个20Hz策略步。
        "policy_dt_s": 0.05,  # 秒/步，合计30秒。
        "horizon_s": 30.0,  # 固定首轨迹时间窗。
        "replicas_per_asset": 16,  # R16。
        "deterministic_actor_mean": True,  # 确定性均值动作。
        "first_trajectory_only": True,  # automatic reset后不增补首轨迹。
        "pregrasp_rank": 0,  # strict rank-0初态。
        "adr_enabled": False,  # 固定ADR0。
        "actor_contact": "tip-only-binary",  # actor物理观察权限。
        "evaluation_role": "capability",  # 能力评价角色。
        "actor_relay": None,  # 单冻结Actor。
        "actor_contact_intervention": "none",  # 无观察干预。
        "residual_off_intervention": False,  # 无动作分支干预。
        "direct_logit_gain_intervention": 1.0,  # 无动作幅值干预。
        "cohort_transfer": False,  # 使用训练canonical cohort。
        "reliable_topology_coverage_protocol_matched": True,  # 正式R协议。
        "goal_advance": "qualified-pose",  # fullKD目标推进。
        "goal_reference": "current_object",  # 当前物体目标参考。
        "reliable_topology_coverage_thresholds": _thresholds(),  # 一圈R阈值。
        "trace_stride": 1,  # 正式dense trace逐策略步记录。
        "trace_rewards": True,  # 正式trace包含加权reward terms。
    }  # 动作权威改由checkpoint identity的policy字段提供。
    return protocol  # 每次调用返回独立对象，避免测试之间共享修改。


def _checkpoint_identity(cohort_sha: str, *, authority: float = 1 / 24) -> dict[str, Any]:
    r"""构造CPU checkpoint内的最小method identity，含policy和N040证据。"""

    payload: dict[str, Any] = {
        "identity_schema_version": "4.0.0",  # 与真实PPO checkpoint版本一致。
        "manifest": {
            "sha256": cohort_sha,  # 与显式canonical lock绑定。
            "support_asset_count": 128,  # 原始A128分母。
            "selected_rows": list(range(128)),  # 成员排序证据。
        },
        "policy": {
            "actor_contact": "tip-only-binary",  # policy物理观察权限。
            "action_authority_rad_per_policy_step": authority,  # rad/策略步。
        },
        "geometry_provider": {
            "retained_artifact": {
                "schema_version": "5.0.0",  # 冻结N040 artifact schema。
                "artifact_type": "retained_geometry_encoder",  # N040类型。
                "sha256": "d" * 64,  # 仅作为metadata identity字段。
            }
        },
    }  # 只保留被测CPU metadata边界。
    payload["identity_digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()  # 与生产stable method identity规则一致。
    return payload  # checkpoint保存时复制该JSON-safe identity。


def _write_checkpoint(
    tmp_path: Path,
    cohort_sha: str,
    *,
    authority: float = 1 / 24,
    name: str = "teacher.pth",
) -> tuple[Path, str, str]:
    r"""写入CPU-only checkpoint并返回路径、字节SHA和method identity SHA。"""

    identity = _checkpoint_identity(cohort_sha, authority=authority)  # policy/N040/manifest元数据。
    path = tmp_path / name  # wrapper从evaluation根字段读取此路径。
    torch.save({"anymani_identity": identity, "model": {}}, path)  # 不构造神经网络权重。
    digest = hashlib.sha256(path.read_bytes()).hexdigest()  # expected checkpoint SHA。
    return path, digest, identity["identity_digest"]  # 三个外部测试锚点。


def _write_cohort(tmp_path: Path) -> tuple[Path, str]:
    r"""写入完整 schema-1.2 lock，验证 wrapper 的128成员顺序边界。"""

    path = tmp_path / "canonical.lock.yaml"  # 后缀保持真实入口常用命名。
    document = {
        "schema_version": "1.2.0",
        "members": [
            {
                "cohort_index": row,
                "provenance": {
                    "group_name": "family",
                    "mother_name": f"mother-{row // 4:02d}",
                },
            }
            for row in range(128)
        ],
    }  # wrapper 要求完整128成员并逐项核对排序。
    data = json.dumps(document, separators=(",", ":")).encode("utf-8")  # hash与实际写入字节一致。
    path.write_bytes(data)  # 测试不使用 YAML 解析器。
    return path, hashlib.sha256(data).hexdigest()  # evaluation identity绑定此lock字节。


def _write_evaluation(
    tmp_path: Path,
    name: str,
    cohort_sha: str,
    *,
    method: str = _METHOD,
    checkpoint_sha: str = _CHECKPOINT,
    checkpoint_path: Path | None = None,
    trajectory: Path | None = None,
    trace: Path | None = None,
) -> Path:
    r"""写入含terminal和dense trace声明的已发布evaluation摘要。"""

    document: dict[str, Any] = {
        "artifact_type": "anymani.palm_rotation_support_fixed_evaluation",  # fake family_teacher 不依赖其余字段。
        "schema_version": "1.4.0",  # 与现有固定评价输出一致。
        "evaluation_identity": {
            "manifest_sha256": cohort_sha,  # 显式绑定canonical lock。
            "method_identity_digest": method,  # 固定method identity。
            "checkpoint_sha256": checkpoint_sha,  # wrapper expected会另验。
            "protocol": _protocol(),  # 固定30秒/R16协议。
        },  # 不伪造HDF5内部字段。
    }
    if checkpoint_path is not None:  # 正式wrapper必须从evaluation读取真实checkpoint路径。
        document["checkpoint"] = str(checkpoint_path)  # CPU metadata核验的文件来源。
    if trajectory is not None:  # terminal HDF5测试显式声明文件。
        document["trajectory_hdf5"] = str(trajectory)  # wrapper只读取path/stat，audit负责内容。
    if trace is not None:  # dense step trace测试显式声明文件。
        document["step_trace"] = {
            "path": str(trace),
            "samples": 1,
            "sha256": "e" * 64,
        }  # fake auditor不解析字节，但wrapper强制声明完整结构。
    path = tmp_path / name  # 两个evaluation JSON使用不同inode。
    path.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")  # 同内容不同run也允许。
    return path  # 返回已发布临时路径。


def _fake_report_for(path: Path) -> dict[str, Any]:
    r"""按文件名给出两次 wrapper 使用的已验证 family_teacher 报告。"""

    rows = set(range(86)) if path.name.startswith("first") else set(range(85)) | {86}  # 两次单独86、交集85。
    return _report(rows, rows)  # fake只替代旧一次分析，不替代新交集逻辑。


def _fake_family_teacher(expected_method: str = _METHOD, expected_checkpoint: str = _CHECKPOINT) -> SimpleNamespace:
    r"""构造只含公开 evaluate_teacher 的 raw-module替身。"""

    def evaluate_teacher(
        path: Path, cohort: Path, *, expected_method_identity_digest: str, expected_checkpoint_sha256: str
    ) -> dict[str, Any]:
        r"""验证wrapper确实把显式expected身份传入旧一次评价入口。"""

        assert cohort.is_file()  # fake的输入路径仍需是显式canonical文件。
        assert expected_method_identity_digest == expected_method  # 禁止从JSON自报推导method。
        assert expected_checkpoint_sha256 == expected_checkpoint  # 禁止从JSON自报推导checkpoint。
        return _fake_report_for(path)  # 返回不含模型的旧分析报告。

    return SimpleNamespace(evaluate_teacher=evaluate_teacher)  # raw loader只需此公开函数。


def _fake_trace_auditor() -> SimpleNamespace:
    r"""构造只报告passed的trace auditor，单测不重复实现HDF5语义。"""

    def audit_palm_rotation_trace(path: Path, *, action_mode: str) -> dict[str, Any]:
        r"""验证正式wrapper传入mean模式，然后交还最小完整性证书。"""

        assert action_mode == "mean"  # 正式重复门必须是确定性Actor均值。
        return {
            "status": "passed",
            "evaluation": str(path),
            "scope": "test trace auditor",
        }  # 真实HDF5语义由既有模块负责。

    return SimpleNamespace(audit_palm_rotation_trace=audit_palm_rotation_trace)  # raw loader替身。


def _raw_loader_for(expected_method: str, expected_checkpoint: str):
    r"""按源文件名区分旧一次归约与独立trace audit的raw loader。"""

    family = _fake_family_teacher(expected_method, expected_checkpoint)  # 旧一次family_teacher替身。
    trace = _fake_trace_auditor()  # 独立trace audit替身。

    def load(_module_name: str, source: Path) -> Any:
        r"""保留被测入口的两个raw-module装载边界。"""

        return family if source.name == "family_teacher.py" else trace  # 不导入包初始化。

    return load  # monkeypatch后仍能验证正式调用顺序。


def test_wrapper_rejects_different_checkpoint_before_reduction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r"""两份evaluation若不是同一checkpoint，不能拼接两次结果。"""

    cohort, cohort_sha = _write_cohort(tmp_path)  # 一个显式canonical lock。
    checkpoint_path, checkpoint_sha, method = _write_checkpoint(tmp_path, cohort_sha)  # CPU metadata的合法基准。
    first_trajectory = tmp_path / "first.h5"  # 第一次terminal HDF5。
    first_trace = tmp_path / "first.trace.h5"  # 第一次dense trace HDF5。
    second_trajectory = tmp_path / "second.h5"  # 第二次terminal HDF5。
    second_trace = tmp_path / "second.trace.h5"  # 第二次dense trace HDF5。
    for path in (
        first_trajectory,
        first_trace,
        second_trajectory,
        second_trace,
    ):  # 只需普通文件供fake auditor验证路径。
        path.write_bytes(path.name.encode())  # 不构造HDF5；独立trace边界由fake替身隔离。
    first = _write_evaluation(
        tmp_path,
        "first.json",
        cohort_sha,
        method=method,
        checkpoint_sha=checkpoint_sha,
        checkpoint_path=checkpoint_path,
        trajectory=first_trajectory,
        trace=first_trace,
    )  # 合法基准。
    second = _write_evaluation(
        tmp_path,
        "second.json",
        cohort_sha,
        method=method,
        checkpoint_sha="c" * 64,
        checkpoint_path=checkpoint_path,
        trajectory=second_trajectory,
        trace=second_trace,
    )  # 不同冻结权重声明。
    monkeypatch.setattr(
        acceptance, "_load_raw_module", _raw_loader_for(method, checkpoint_sha)
    )  # 隔离旧分析/trace实现，保留wrapper边界。
    with pytest.raises(acceptance.AcceptanceError, match="checkpoint|expected"):  # 身份错误不是科学失败。
        acceptance.evaluate_repeat_acceptance(
            [first, second],
            cohort,
            expected_method_identity_digest=method,
            expected_checkpoint_sha256=checkpoint_sha,
        )


def test_wrapper_rejects_duplicate_evaluation_path(tmp_path: Path) -> None:
    r"""同一路径即使被传入两次，也不能被当作两次独立运行。"""

    cohort, _ = _write_cohort(tmp_path)  # 先写canonical lock。
    evaluation = _write_evaluation(
        tmp_path, "evaluation.json", hashlib.sha256(cohort.read_bytes()).hexdigest()
    )  # 一份JSON。
    with pytest.raises(acceptance.AcceptanceError, match="同一路径"):  # resolve后立刻拒绝。
        acceptance.evaluate_repeat_acceptance(
            [evaluation, evaluation],
            cohort,
            expected_method_identity_digest=_METHOD,
            expected_checkpoint_sha256=_CHECKPOINT,
        )


@pytest.mark.parametrize("hard_link", [False, True])
def test_wrapper_rejects_same_trajectory_hdf5_path_or_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hard_link: bool
) -> None:
    r"""同一trajectory HDF5的resolve路径或inode不能扩大重复运行数。"""

    cohort, cohort_sha = _write_cohort(tmp_path)  # canonical lock与两份JSON身份相同。
    checkpoint_path, checkpoint_sha, method = _write_checkpoint(
        tmp_path, cohort_sha
    )  # 先闭合checkpoint action/N040 metadata。
    hdf5 = tmp_path / "evaluation.h5"  # 内容只作inode哨兵，不启动HDF5库。
    hdf5.write_bytes(b"same trajectory entity")  # 普通文件足以验证路径/inode边界。
    second_hdf5 = hdf5 if not hard_link else tmp_path / "evaluation-copy.h5"  # hard link分支使用不同路径。
    if hard_link:  # 两条路径仍指向同一inode。
        second_hdf5.hardlink_to(hdf5)  # 测试临时目录内的可逆文件操作。
    first_trace = tmp_path / "first.trace.h5"  # 与trajectory分离的dense文件。
    second_trace = tmp_path / "second.trace.h5"  # 第二次独立trace文件。
    first_trace.write_bytes(b"trace-one")  # fake auditor不解析HDF5内容。
    second_trace.write_bytes(b"trace-two")  # 只验证跨run trajectory实体重复。
    first = _write_evaluation(
        tmp_path,
        "first.json",
        cohort_sha,
        method=method,
        checkpoint_sha=checkpoint_sha,
        checkpoint_path=checkpoint_path,
        trajectory=hdf5,
        trace=first_trace,
    )  # 第一次HDF5。
    second = _write_evaluation(
        tmp_path,
        "second.json",
        cohort_sha,
        method=method,
        checkpoint_sha=checkpoint_sha,
        checkpoint_path=checkpoint_path,
        trajectory=second_hdf5,
        trace=second_trace,
    )  # 同路径或同inode。
    monkeypatch.setattr(
        acceptance, "_load_raw_module", _raw_loader_for(method, checkpoint_sha)
    )  # HDF5边界在旧分析前检查。
    with pytest.raises(acceptance.AcceptanceError, match="HDF5|路径|inode"):  # 不能进入86归约。
        acceptance.evaluate_repeat_acceptance(
            [first, second],
            cohort,
            expected_method_identity_digest=method,
            expected_checkpoint_sha256=checkpoint_sha,
        )


def test_different_run_hdf5_content_copy_is_allowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r"""不同inode的同内容HDF5可表示确定性重复，不按字节内容误判为同一次。"""

    cohort, cohort_sha = _write_cohort(tmp_path)  # 同一个canonical成员轴。
    checkpoint_path, checkpoint_sha, method = _write_checkpoint(tmp_path, cohort_sha)  # CPU checkpoint identity。
    first_hdf5 = tmp_path / "first.h5"  # 第一次独立文件。
    second_hdf5 = tmp_path / "second.h5"  # 第二次独立文件。
    first_hdf5.write_bytes(b"deterministic trajectory bytes")  # 内容可以完全相同。
    second_hdf5.write_bytes(first_hdf5.read_bytes())  # inode仍不同。
    first_trace = tmp_path / "first.trace.h5"  # dense trace独立文件。
    second_trace = tmp_path / "second.trace.h5"  # dense trace独立文件。
    first_trace.write_bytes(b"same dense trace bytes")  # 内容相同也不能成为同一inode。
    second_trace.write_bytes(first_trace.read_bytes())  # 不同inode的确定性重复。
    first = _write_evaluation(
        tmp_path,
        "first.json",
        cohort_sha,
        method=method,
        checkpoint_sha=checkpoint_sha,
        checkpoint_path=checkpoint_path,
        trajectory=first_hdf5,
        trace=first_trace,
    )  # 第一次评价。
    second = _write_evaluation(
        tmp_path,
        "second.json",
        cohort_sha,
        method=method,
        checkpoint_sha=checkpoint_sha,
        checkpoint_path=checkpoint_path,
        trajectory=second_hdf5,
        trace=second_trace,
    )  # 第二次评价。
    monkeypatch.setattr(
        acceptance, "_load_raw_module", _raw_loader_for(method, checkpoint_sha)
    )  # 使用可控旧分析与trace报告。
    result = acceptance.evaluate_repeat_acceptance(
        [first, second],
        cohort,
        expected_method_identity_digest=method,
        expected_checkpoint_sha256=checkpoint_sha,
    )  # 正式入口仍强制调用trace auditor。
    assert result["repeat_gate"]["common_passed_assets"] == 85, result  # 文件去重通过后仍按真实row交集。
    assert (
        result["inputs"]["hdf5_entities"][0]["trajectory"]["entity"]
        != result["inputs"]["hdf5_entities"][1]["trajectory"]["entity"]
    ), result  # inode确实不同。


def test_formal_wrapper_rejects_missing_dense_trace(tmp_path: Path) -> None:
    r"""缺少terminal或stride-1 reward trace时，正式file入口必须拒绝。"""

    cohort, cohort_sha = _write_cohort(tmp_path)  # canonical lock仍然显式提供。
    first = _write_evaluation(tmp_path, "first.json", cohort_sha)  # 故意没有HDF5路径。
    second = _write_evaluation(tmp_path, "second.json", cohort_sha)  # 两份summary都不完整。
    with pytest.raises(acceptance.AcceptanceError, match="trajectory|trace"):  # 不能在未审计dense证据上输出passed。
        acceptance.evaluate_repeat_acceptance(
            [first, second],
            cohort,
            expected_method_identity_digest=_METHOD,
            expected_checkpoint_sha256=_CHECKPOINT,
        )


def test_checkpoint_policy_authority_must_be_one_over_24(tmp_path: Path) -> None:
    r"""checkpoint metadata中的物理动作权限不是1/24时拒绝正式身份。"""

    cohort, cohort_sha = _write_cohort(tmp_path)  # manifest身份来源。
    checkpoint_path, checkpoint_sha, method = _write_checkpoint(
        tmp_path, cohort_sha, authority=1 / 12, name="bad-authority.pth"
    )  # 合法重新计算identity digest，单独触发action authority门。
    with pytest.raises(acceptance.AcceptanceError, match="authority|1/24"):  # 不能以evaluation自报替代policy真源。
        acceptance._read_checkpoint_identity(
            checkpoint_path,
            expected_method_identity_digest=method,
            expected_checkpoint_sha256=checkpoint_sha,
            cohort_sha256=cohort_sha,
        )
