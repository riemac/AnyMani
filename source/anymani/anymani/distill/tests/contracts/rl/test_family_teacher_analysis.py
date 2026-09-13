r"""家族教师只读分析合同：证伪候选泄漏、分母漂移和强门伪报。

所有输入均为临时目录中的小型 CPU 产物；checkpoint 内容故意不是模型。
正式覆盖的独立手算参照为 128 资产、32 拓扑、每拓扑 4 代表、每资产 16 副本。
资产门与拓扑门分别为 103/128 和 29/32；测试不代表任何真实教师已经达标。
直接按文件加载分析模块，使本测试的导入闭包不经过 anymani 的环境注册。
"""

from __future__ import annotations

import hashlib  # 测试绑定的是输入文件原始字节。
import importlib.util  # 文件入口绕开顶层包的环境注册。
import json  # 合成正式 JSON 层级。
import subprocess  # 独立解释器验证导入边界。
import sys  # 使用 pytest 本身的 CPU Python。
from pathlib import Path  # 所有可写产物位于 tmp_path。
from typing import Any  # JSON 的异构叶节点。

import polars as pl  # 只生成小型 Parquet 事实表。
import pytest  # 契约断言与非法输入参数化。

_SOURCE = Path(__file__).resolve().parents[3] / "diagnostics/analysis/rl/family_teacher.py"  # 两文件间的稳定相对路径。
_SPEC = importlib.util.spec_from_file_location("family_teacher_analysis_contract", _SOURCE)  # 无包初始化。
assert _SPEC is not None and _SPEC.loader is not None, "只读分析文件必须可独立加载"
analysis = importlib.util.module_from_spec(_SPEC)  # 模块不持有模型、MDP 或训练对象。
_SPEC.loader.exec_module(analysis)  # 只执行被测新文件。
_METHOD = "a" * 64  # 显式指定的预期方法身份。
_CHECKPOINT = "b" * 64  # 由调用者另验的 checkpoint SHA-256。


def _row(update: int, **changes: Any) -> dict[str, Any]:
    r"""构造合法 global 行；预算为 $u\times61440$ transitions，净转单位为圈。"""
    row = {
        "schema_version": "2.6.0",  # 当前 recorder 的首窗宽表版本。
        "scope": "global",  # 候选查询的唯一统计层级。
        "identity_digest": _METHOD,  # 同一个训练方法。
        "update": update,  # PPO 更新坐标。
        "transitions": update * 61440,  # 每次更新的新环境转移数。
        "first30_asset_count": 128,  # 固定资产宇宙。
        "first30_observed_assets": 128,  # 所有资产均有首窗。
        "first30_qualified_assets": 128,  # 所有资产满足配置资格门。
        "first30_window_count": 128 * 16,  # 资产内窗口总数，分母为 2048。
        "first30_windows_min": 16,  # 最弱资产也有 16 个窗口。
        "first30_windows_max": 16,  # 此夹具各资产等样本量。
        "first30_net_median": 1.5,  # 资产内中位后再取资产中位，单位圈。
        "first30_goal_median": 10.0,  # 严格目标命中次数，仅作合法性校验。
        "first30_direction_median": 0.8,  # 训练逐窗方向归约，允许范围 [-1,1]。
        "first30_safe_fraction": 0.75,  # 资产等权的首窗安全比例。
        "first30_one_turn_proxy_assets": 100,  # 一圈训练代理计数。
        "first30_two_turn_proxy_assets": 50,  # 两圈代理是上述集合的子集。
        "first30_policy_start_min": 0,  # 留存窗口的策略版本下界。
        "first30_policy_end_max": update * 61440,  # 版本坐标也是 transitions。
        "reward_mean": 0.0,  # 故意保留可操纵奖励，以证伪奖励选优。
    }
    row.update(changes)  # 每个反例只改变所声明的事实。
    return row  # 原生 Python 标量可直接进入 Polars。


def _run(tmp_path: Path, rows: list[dict[str, Any]], saved: tuple[int, ...]) -> tuple[Path, Path]:
    r"""写入完成表及显式 nn；非 global 行含坏身份，用来检验统计层级隔离。"""
    metrics = tmp_path / "metrics.parquet"  # 完成产物的正式文件名。
    noise = {**rows[0], "scope": "asset", "identity_digest": "非 global 身份", "transitions": -1}  # 禁止参与选优。
    pl.DataFrame([*rows, noise]).write_parquet(metrics)  # 小表不依赖现有 recorder 实现。
    nn = tmp_path / "nn"  # 调用者必须显式提供的目录。
    nn.mkdir()  # 测试自己的临时目录。
    for update in saved:  # 文件名定位与权重解析应完全分离。
        (nn / f"evaluation_teacher_ep_{update:05d}.pth").write_bytes(b"not a model")  # 不可反序列化的哨兵。
    return metrics, nn  # 两个输入不通过训练 run 自动发现。


def _dump(path: Path, value: dict[str, Any]) -> None:
    r"""写测试事实；允许生成 NaN 反例，生产解析器必须拒绝它。"""
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")  # 只写 pytest 临时产物。


def _case(tmp_path: Path, counts: list[int]) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    r"""以每拓扑通过代表数直接构造覆盖真值，不调用生产归约函数。

    通过资产取 $N=2$ 圈、$P=2.5$ 圈、$D=0.8$、$S=12/16$。
    不通过资产取 $N=0.4$ 圈、$P=0.5$ 圈，故仅净圈不足。
    """
    assert len(counts) == 32 and all(0 <= count <= 4 for count in counts), "独立夹具必须保持 32×4 分母"
    cohort = {
        "schema_version": "1.2.0",  # canonical-final lock 的真实版本。
        "members": [
            {
                "cohort_index": row,  # 正式评价使用 canonical 成员位置轴。
                "provenance": {"group_name": "allegro", "mother_name": f"mother-{row // 4:02d}"},  # 完整拓扑身份。
            }
            for row in range(128)  # 128 个唯一成员。
        ],
    }
    cohort_path = tmp_path / "training.canonical.lock.yaml"  # 扩展名为 YAML，内容确为 JSON。
    _dump(cohort_path, cohort)  # hash 必须在写入之后计算。
    thresholds = {
        "horizon_s": 30.0,  # 首轨迹最长模拟时间，秒。
        "replicas_per_asset": 16,  # 安全比例的分母。
        "net_turns_min": 1.0,  # 可靠一圈门。
        "directional_consistency_min": 0.7,  # 一圈与两圈使用相同方向门。
        "safe_replica_fraction_min": 0.75,  # 至少 12/16 副本安全。
        "topology_representative_fraction_min": 0.5,  # 至少 2/4 代表。
    }
    document = {
        "artifact_type": "anymani.palm_rotation_support_fixed_evaluation",  # 正式能力产物。
        "schema_version": "1.4.0",  # 当前 JSON 文档层版本。
        "evaluation_identity": {
            "schema_version": "1.6.0",  # 当前身份层版本。
            "manifest_sha256": hashlib.sha256(cohort_path.read_bytes()).hexdigest(),  # 精确字节绑定。
            "method_identity_digest": _METHOD,  # 方法身份由调用者给定。
            "checkpoint_sha256": _CHECKPOINT,  # checkpoint 摘要由调用者给定。
            "protocol": {
                "num_assets": 128,  # 资产分母固定。
                "policy_steps": 600,  # 600×0.05=30 秒。
                "policy_dt_s": 0.05,  # 20 Hz 控制周期。
                "horizon_s": 30.0,  # 首轨迹窗口上限。
                "replicas_per_asset": 16,  # 每资产 R16。
                "deterministic_actor_mean": True,  # 固定 Actor 均值。
                "first_trajectory_only": True,  # reset 后轨迹不进入统计。
                "pregrasp_rank": 0,  # 固定 rank-0 起点。
                "adr_enabled": False,  # 无自适应课程。
                "actor_contact": "tip-only-binary",  # 指尖二值触觉。
                "evaluation_role": "capability",  # 能力测量角色。
                "actor_relay": None,  # 单个冻结 Actor。
                "actor_contact_intervention": "none",  # 无观察干预。
                "residual_off_intervention": False,  # 无动作分支干预。
                "direct_logit_gain_intervention": 1.0,  # 无幅值干预。
                "cohort_transfer": False,  # 绑定训练集合本身。
                "reliable_topology_coverage_protocol_matched": True,  # 正式 R 协议声明。
                "reliable_topology_coverage_thresholds": dict(thresholds),  # 双位置阈值必须一致。
                "goal_advance": "qualified-pose",  # fullKD 的目标推进法。
                "goal_reference": "current_object",  # fullKD 的目标参考法。
            },
        },
        "physical_rotation": {
            "finite_and_identity_valid": True,  # 全资产有效；禁止删除失败资产。
            "asset_results": [
                {
                    "dataset_row": row,  # selection-local 轴 0..127。
                    "mother_id": f"mother-{row // 4:02d}",  # 与 cohort 的母体逐项对应。
                    "replica_count": 16,  # 每资产分母。
                    "finite": True,  # 逐资产有限性声明。
                    "net_turns_median": 2.0 if row % 4 < counts[row // 4] else 0.4,  # 正向物理净圈。
                    "absolute_path_turns_median": 2.5 if row % 4 < counts[row // 4] else 0.5,  # 路径圈。
                    "directional_consistency": 0.8,  # ratio-of-medians 的独立手算值。
                    "safe_replica_fraction": 0.75,  # 12/16，无量纲。
                    "viability_passed": False,  # 与可靠门故意相反，验证不读取此字段。
                    "scale_ready_passed": False,  # D=0.8 在旧 .85 门下确实失败。
                }
                for row in range(128)  # 不能池化成 2048 个独立资产。
            ],
        },
        "reliable_topology_coverage": {
            "schema_version": "1.0.0",  # 真实 R 归约版本。
            "thresholds": thresholds,  # 一圈可靠覆盖阈值。
            "finite": True,  # 完整轴上的有限性声明。
            "asset_count": 128,  # 全体资产分母。
            "passed_asset_count": sum(counts),  # 独立整数加和真值。
            "passed_asset_rows": [row for row in range(128) if row % 4 < counts[row // 4]],  # 真实通过集合。
            "topology_count": 32,  # 不按副本或短母体名称合并。
            "passed_topology_count": sum(count >= 2 for count in counts),  # 拓扑内半数代表。
            "topology_results": [
                {
                    "topology_id": f"allegro/mother-{index:02d}",  # group 限定的母体。
                    "asset_count": 4,  # 此拓扑四代表。
                    "passed_asset_count": count,  # 手算通过代表数。
                    "required_asset_count": 2,  # ceil(4/2)=2。
                    "passed": count >= 2,  # 各拓扑独立门。
                }
                for index, count in enumerate(counts)  # 32 行拓扑表。
            ],
        },
        "scale_ladder": {"passed": False},  # 不参与新强门的旁路字段。
        "support": {"closure_passed": False},  # 不参与新强门的旁路字段。
    }
    evaluation = tmp_path / "evaluation.json"  # 已完成的真实 schema 形状。
    _dump(evaluation, document)  # 不生成 HDF5，也不运行评价器。
    return evaluation, cohort_path, document, cohort  # 修改反例后由测试显式重写临时 JSON。


def _evaluate(evaluation: Path, cohort: Path) -> dict[str, Any]:
    r"""每次调用都显式绑定预期方法身份与 checkpoint 哈希。"""
    return analysis.evaluate_teacher(
        evaluation, cohort, expected_method_identity_digest=_METHOD, expected_checkpoint_sha256=_CHECKPOINT
    )  # 从不从待核验 JSON 反向生成 expected 值。


def test_select_saved_proxy_best_not_reward_or_unsaved_update(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r"""高奖励和未保存的更高代理均不能成为候选；checkpoint 只允许 stat。"""
    rows = [
        _row(50, reward_mean=1e9),  # 有保存但奖励高不构成优势。
        _row(100, first30_one_turn_proxy_assets=110, reward_mean=-1e9),  # 真正可用的训练最佳。
        _row(150, first30_one_turn_proxy_assets=128),  # 全曲线最佳没有保存点。
        _row(2000),  # 完整终点必须独立保留。
    ]
    metrics, nn = _run(tmp_path, rows, (50, 100, 2000))  # 明确可用 update 集合。
    (nn / "last_teacher_ep_100_rew100000.pth").write_bytes(b"not a model either")  # 同 update 低优先级文件。
    original_open = Path.open  # 保留非 checkpoint 产物读取。

    def guarded_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        r"""禁止任何 checkpoint 内容读取，包括为求 hash 而读取。"""
        assert path.suffix != ".pth", f"checkpoint 只能文件名定位：{path}"
        return original_open(path, *args, **kwargs)  # Parquet hash 与 CPU 元数据可以读取。

    monkeypatch.setattr(Path, "open", guarded_open)  # 为整个选择调用建立内容访问哨兵。
    result = analysis.select_candidates(metrics, nn)  # 默认 u2000、A128、61440 transitions/update。
    assert [item["update"] for item in result["candidates"]] == [100, 2000], result  # 未保存 u150 与高奖励 u50 均不进入候选。
    assert Path(result["candidates"][0]["checkpoint"]["path"]).name.startswith("evaluation_"), result  # 同点优先 evaluation 类。
    assert result["checkpoint_metadata_verified"] is False, "文件名不能验证模型身份"  # 命名定位不承诺内容身份。
    assert len(result["input_fingerprint_sha256"]) == 64, "输入事实集必须有确定指纹"  # 明确固定输入快照。


@pytest.mark.parametrize("winner", ["one", "two", "net", "safe", "earlier"])  # 字典序的每一级都需独立证伪。
def test_proxy_lexicographic_priority_and_earlier_tie(tmp_path: Path, winner: str) -> None:
    r"""排序严格是 $(n_1,n_2,N,S,-u)$；后位极大值不能越过前位。"""
    early = _row(10)  # 完全平局时应选择更早 update。
    later = _row(20)  # 每个反例最多改变一处优先键和后续键。
    changes = {
        "one": {"first30_one_turn_proxy_assets": 101, "first30_two_turn_proxy_assets": 0},  # 一圈优先。
        "two": {"first30_two_turn_proxy_assets": 51, "first30_net_median": 1.0},  # 两圈优先于较低的合法净圈中心。
        "net": {"first30_net_median": 1.6, "first30_safe_fraction": 0.7},  # 中心优先于较低的合法资产安全均值。
        "safe": {"first30_safe_fraction": 0.8},  # 最后一个正向排序键。
        "earlier": {},  # 完全平局。
    }
    later.update(changes[winner])  # 保持其余字段与合法分母。
    metrics, nn = _run(tmp_path, [early, later, _row(2000, first30_one_turn_proxy_assets=0, first30_two_turn_proxy_assets=0)], (10, 20, 2000))
    result = analysis.select_candidates(metrics, nn)  # 末点特意降低，避免掩盖排序反例。
    assert result["candidates"][0]["update"] == (10 if winner == "earlier" else 20), result  # 后位改善不能越过前位，完全平局取早。


def test_final_best_deduplicates_and_last_filename_is_supported(tmp_path: Path) -> None:
    r"""最佳与终点相同只返回一份，同时保留两个选择理由。"""
    metrics, nn = _run(tmp_path, [_row(2000)], ())  # 此处只有 last 类文件。
    (nn / "last_teacher_ep_2000_rew-123.4.pth").write_bytes(b"filename only")  # 奖励后缀不解析为排序指标。
    result = analysis.select_candidates(metrics, nn)  # 不要求同 update 必有 evaluation 文件。
    assert len(result["candidates"]) == 1, result  # 正式评价预算不重复花在同一 update。
    assert result["candidates"][0]["roles"] == ["best_first30", "final"], result  # 去重必须保留两个选择理由。


def test_unique_periodic_checkpoint_takes_priority_over_multiple_last_spellings(tmp_path: Path) -> None:
    r"""训练终点可产生两种last后缀；唯一周期保存点提供明确来源，无需猜两份last是否相同。"""
    metrics, nn = _run(tmp_path, [_row(128)], (128,))  # 周期保存点是该更新的明确优先来源。
    (nn / "last_teacher_ep_128_rew__-14.75_.pth").write_bytes(b"last at budget stop")  # 正常退出保存名。
    (nn / "last_teacher_ep_128_rew_-14.75.pth").write_bytes(b"periodic last")  # rl_games周期last保存名。
    result = analysis.select_candidates(metrics, nn, final_update=128)  # 不删除或改写两份原始文件。
    assert len(result["candidates"]) == 1  # 同更新不会被当作两个科学候选。
    assert Path(result["candidates"][0]["checkpoint"]["path"]).name == "evaluation_teacher_ep_00128.pth"
    assert len(result["inputs"]["checkpoint_inventory"]) == 3  # 低优先级来源仍完整保留在审计清单。


@pytest.mark.parametrize("missing", ["checkpoint", "global", "endpoint", "completed_table"])  # 四种不完整证据。
def test_final_endpoint_is_mandatory(tmp_path: Path, missing: str) -> None:
    r"""最终候选必须有完成表中的终点行和对应保存文件，不能用较早点补位。"""
    rows = [_row(100), _row(2000)] if missing != "global" else [_row(100)]  # 缺 global 不可补造。
    saved = (100,) if missing == "checkpoint" else (100, 2000)  # 缺保存文件不可补造。
    metrics, nn = _run(tmp_path, rows, saved)  # 标准完成表输入。
    if missing == "endpoint":  # 含更晚 update 的表不属于所声明的完整终点。
        pl.DataFrame([*rows, _row(2001)]).write_parquet(metrics)  # 不能默默截断成 u2000。
    if missing == "completed_table":  # shard 或实时日志不是这个入口的完成证据。
        metrics.unlink()  # 仅删除测试自己的临时文件。
    with pytest.raises((analysis.AnalysisError, FileNotFoundError), match="final|终点|完成|metrics.parquet"):
        analysis.select_candidates(metrics, nn)  # 每一种都应明确拒绝。


def test_low_samples_return_only_final(tmp_path: Path) -> None:
    r"""15 窗即使代理数很好也不能成为最佳；合法无观测终点仍可文件名定位。"""
    early = _row(100, first30_windows_min=15, first30_windows_max=15, first30_window_count=128 * 15, first30_qualified_assets=0, first30_one_turn_proxy_assets=0, first30_two_turn_proxy_assets=0)
    final = _row(2000, first30_observed_assets=0, first30_qualified_assets=0, first30_window_count=0, first30_windows_min=0, first30_windows_max=0, first30_one_turn_proxy_assets=0, first30_two_turn_proxy_assets=0)
    for key in ("net_median", "goal_median", "direction_median", "safe_fraction", "policy_start_min", "policy_end_max"):
        final[f"first30_{key}"] = None  # 官方无观测 schema 的合法 null。
    metrics, nn = _run(tmp_path, [early, final], (100, 2000))  # 低样本不等于输入损坏。
    result = analysis.select_candidates(metrics, nn)  # 不制造“训练最佳”。
    assert [item["update"] for item in result["candidates"]] == [2000], result  # 合法无观测不阻止完整终点定位。
    assert result["candidates"][0]["roles"] == ["final"], result  # 不把样本不足点冠以最佳。
    assert result["eligible_best_count"] == 0, result  # 15 窗与 0 窗均不够 16 窗门。


@pytest.mark.parametrize("denominator", ["asset", "observed", "qualified"])  # 三项 A128 要求分别验证。
def test_each_first30_asset_denominator_is_required(tmp_path: Path, denominator: str) -> None:
    r"""低样本资产宇宙、缺测资产与资格不足不能被高代理计数掩盖。"""
    changes = {
        "asset": {"first30_asset_count": 127, "first30_observed_assets": 127, "first30_qualified_assets": 127, "first30_window_count": 127 * 16},  # 127 个等样本资产。
        "observed": {"first30_observed_assets": 127, "first30_qualified_assets": 127, "first30_window_count": 127 * 16, "first30_windows_min": 0},  # 第 128 资产缺测。
        "qualified": {"first30_qualified_assets": 127, "first30_windows_max": 32, "first30_window_count": 127 * 32 + 16},  # 配置资格门可高于代理所需 16 窗。
    }
    early = _row(100, **changes[denominator])  # 高代理但不具备完整 A128 资格。
    final = _row(2000, first30_one_turn_proxy_assets=0, first30_two_turn_proxy_assets=0)  # 较低代理的合法完整终点。
    metrics, nn = _run(tmp_path, [early, final], (100, 2000))  # 两个点都实际有保存文件。
    result = analysis.select_candidates(metrics, nn)  # 第一项必须因分母被排除。
    assert [item["update"] for item in result["candidates"]] == [2000], result  # 三项分母条件分别阻止高代理点获选。
    assert result["eligible_best_count"] == 1 and result["saved_update_audit"][0]["exclusion_reasons"], result  # 排除条件必须可解释。


@pytest.mark.parametrize("fault", ["duplicate", "identity", "transitions", "nan", "count", "ambiguous", "orphan"])
def test_select_rejects_corrupt_or_ambiguous_facts(tmp_path: Path, fault: str) -> None:
    r"""身份、预算、唯一性和数值合法性错误不能降格成科学未达标。"""
    rows = [_row(100), _row(2000)]  # 两个有效保存点的基准。
    if fault == "duplicate":  # 一个 update 不能对应两条 global。
        rows.append(_row(100))  # 内容相同也必须拒绝。
    if fault in {"identity", "transitions", "nan", "count"}:  # 各错误相互独立。
        key, value = {"identity": ("identity_digest", "c" * 64), "transitions": ("transitions", 1), "nan": ("first30_net_median", float("nan")), "count": ("first30_two_turn_proxy_assets", 129)}[fault]
        rows[0][key] = value  # 只改变一个必要事实。
    metrics, nn = _run(tmp_path, rows, (100, 2000))  # 此前都是真实存在的路径。
    if fault == "ambiguous":  # 没有周期evaluation时，两个last缺少可辨别的优先来源。
        (nn / "evaluation_teacher_ep_00100.pth").unlink()  # 只移除测试临时目录内的优先来源。
        for suffix in ("1", "2"):  # 不按奖励后缀消解歧义。
            (nn / f"last_teacher_ep_100_rew{suffix}.pth").write_bytes(b"ambiguous")  # 仅测试临时保存点。
    if fault == "orphan":  # 每个识别出的保存点都须有唯一 global。
        (nn / "evaluation_teacher_ep_00150.pth").write_bytes(b"orphan")  # 没有 update150 行。
    with pytest.raises(analysis.AnalysisError):  # 错误类别与合法 passed=False 分离。
        analysis.select_candidates(metrics, nn)  # 非法证据不得得到貌似合法的候选列表。


def test_custom_budget_and_asset_denominator(tmp_path: Path) -> None:
    r"""显式参数允许小型 CPU 数据，但预算仍须逐 update 精确闭合。"""
    row = _row(7, transitions=7 * 120, first30_asset_count=4, first30_observed_assets=4, first30_qualified_assets=4, first30_window_count=64, first30_one_turn_proxy_assets=4, first30_two_turn_proxy_assets=2, first30_policy_end_max=840)
    metrics, nn = _run(tmp_path, [row], (7,))  # 此测试不改变正式强门的 128/32 分母。
    result = analysis.select_candidates(metrics, nn, final_update=7, asset_count=4, transitions_per_update=120)
    assert result["candidates"][0]["transitions"] == 840, result  # 7×120 新 transitions。


@pytest.mark.parametrize(
    ("counts", "assets", "topologies", "passed"),
    [
        ([4] * 22 + [3] + [2] * 6 + [0] * 3, 103, 29, True),  # 两门恰好通过。
        ([4] * 22 + [2] * 7 + [0] * 3, 102, 29, False),  # 只缺一个资产。
        ([4] * 23 + [2] * 5 + [1] + [0] * 3, 103, 28, False),  # 只缺一个拓扑。
    ],
)
def test_strong_gate_has_independent_asset_and_topology_requirements(tmp_path: Path, counts: list[int], assets: int, topologies: int, passed: bool) -> None:
    r"""103/128 与 29/32 是交集门；两圈 D=0.8 应按 .7 通过而非 .85 失败。"""
    evaluation, cohort, _, _ = _case(tmp_path, counts)  # 真值来自手算整数分配。
    result = _evaluate(evaluation, cohort)  # 不读取 legacy support/scale 布尔值。
    assert result["passed"] is passed, result  # 一个强门通过不能补偿另一个强门失败。
    for level in ("one_turn", "two_turn"):  # 同一批通过资产净转均为两圈。
        coverage = result[level]  # 两个层级必须分别重算。
        assert (coverage["passed_asset_count"], coverage["passed_topology_count"]) == (assets, topologies), coverage  # 计数由独立代表分配真值给出。
        assert coverage["asset_fraction"] == assets / 128, coverage  # 资产分母固定为 128。
        assert coverage["topology_fraction"] == topologies / 32, coverage  # 拓扑分母固定为 32。
    assert result["gaps"]["assets_to_103"] == max(103 - assets, 0), result  # 达门后资产缺口下限为零。
    assert result["gaps"]["topologies_to_29"] == max(29 - topologies, 0), result  # 拓扑缺口与资产缺口独立。
    assert result["asset_summary"]["net_turns_median"] == 2.0, result  # 资产等权中心，不是副本池化。
    assert result["asset_summary"]["safe_replica_count"] == 128 * 12, result  # 安全分母 128×16。


def test_two_turn_coverage_is_recomputed_independently(tmp_path: Path) -> None:
    r"""前 16 拓扑仅一圈、后 16 拓扑两圈，两个覆盖表不能相互复用。"""
    evaluation, cohort, document, _ = _case(tmp_path, [4] * 32)  # 一圈 R 仍为 128/128、32/32。
    for asset in document["physical_rotation"]["asset_results"][:64]:  # 前 16×4 代表只转一圈。
        asset.update(net_turns_median=1.0, absolute_path_turns_median=1.25)  # 方向仍为 .8，安全仍为 12/16。
    _dump(evaluation, document)  # R 一圈表无需改变，因为这些资产仍满足可靠门。
    result = _evaluate(evaluation, cohort)  # 两圈数据必须直接从原始 128 行再判门。
    assert result["one_turn"]["passed_asset_count"] == 128, result  # 完整一圈覆盖。
    assert (result["two_turn"]["passed_asset_count"], result["two_turn"]["passed_topology_count"]) == (64, 16), result  # 只有后半数拓扑达到两圈。
    assert result["asset_summary"]["net_turns_median"] == 1.5, result  # 两个中心资产为 1 与 2 圈。


@pytest.mark.parametrize(
    ("net", "path", "safe", "one", "two"),
    [
        (1.0, 1.25, 0.75, 1, 0),  # 一圈净转边界包含等号。
        (2.0, 2.5, 0.75, 1, 1),  # 两圈净转边界包含等号。
        (7.0, 10.0, 0.75, 1, 1),  # D=.7 的边界也包含等号。
        (7.0, 10.000001, 0.75, 0, 0),  # D 略低于 .7，不能用复算容差放宽科学门。
        (2.0, 2.5, 11 / 16, 0, 0),  # 安全副本必须至少 12/16。
        (0.999999, 1.25, 0.75, 0, 0),  # 略不足一圈仍属合法科学失败。
    ],
)
def test_physical_gate_boundary_conjunction(tmp_path: Path, net: float, path: float, safe: float, one: int, two: int) -> None:
    r"""净圈、方向与安全三门各自独立，并使用精确阈值而非近似通过。"""
    evaluation, cohort, document, _ = _case(tmp_path, [one] + [0] * 31)  # 独立手算的一圈通过集合。
    document["physical_rotation"]["asset_results"][0].update(net_turns_median=net, absolute_path_turns_median=path, directional_consistency=net / path, safe_replica_fraction=safe)  # 正式比值由两个资产中位数给出。
    _dump(evaluation, document)  # 其它 127 行保持合法失败。
    result = _evaluate(evaluation, cohort)  # 数据有效与强门结果分开。
    assert (result["one_turn"]["passed_asset_count"], result["two_turn"]["passed_asset_count"]) == (one, two), result  # 三个阈值的交集不添加模糊通过带。


@pytest.mark.parametrize("field", ["manifest_sha256", "method_identity_digest", "checkpoint_sha256"])
def test_evaluation_identity_must_bind_external_expectations(tmp_path: Path, field: str) -> None:
    r"""三个身份锚点分别核验；自报哈希不能成为自己的 expected 值。"""
    evaluation, cohort, document, _ = _case(tmp_path, [4] * 32)  # 科学指标全部达标仍需身份有效。
    document["evaluation_identity"][field] = "c" * 64  # 保持合法哈希形状但改变身份。
    _dump(evaluation, document)  # 仅重写临时输入。
    with pytest.raises(analysis.AnalysisError, match=field):  # 错误必须指出具体身份锚点。
        _evaluate(evaluation, cohort)  # 任一身份锚点错配均属于非法输入。


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_assets", 127),  # 资产分母。
        ("policy_steps", 599),  # 首轨迹时间预算。
        ("policy_dt_s", 0.1),  # 物理单位为秒。
        ("horizon_s", 60.0),  # 不能把耐久复查当作主门。
        ("replicas_per_asset", 15),  # R16 分母。
        ("deterministic_actor_mean", False),  # 采样动作不等价于固定均值。
        ("first_trajectory_only", False),  # 自动 reset 不能扩充样本。
        ("pregrasp_rank", True),  # bool 不能借 Python 的整数等价通过。
        ("adr_enabled", True),  # 正式门固定 ADR0。
        ("actor_contact", "all-owner-binary-no-force"),  # 观察合同改变。
        ("evaluation_role", "diagnostic"),  # 角色不符。
        ("actor_relay", {}),  # 不能拼接两个 Actor 的能力。
        ("actor_contact_intervention", "tip-only-mask"),  # 观察干预。
        ("residual_off_intervention", True),  # 动作分支干预。
        ("direct_logit_gain_intervention", True),  # bool 也不能冒充浮点 1。
        ("cohort_transfer", True),  # 非训练集合的迁移评价。
        ("reliable_topology_coverage_protocol_matched", False),  # R 协议声明无效。
        ("goal_advance", "angle-only"),  # fullKD 目标推进不符。
        ("goal_reference", "previous_goal"),  # fullKD 参考不符。
    ],
)
def test_protocol_mismatch_is_invalid_not_failed_science(tmp_path: Path, field: str, value: Any) -> None:
    r"""固定协议逐字段证伪；输出错误必须带字段路径。"""
    evaluation, cohort, document, _ = _case(tmp_path, [4] * 32)  # 科学成功不能掩盖协议改变。
    document["evaluation_identity"]["protocol"][field] = value  # 单字段反例。
    _dump(evaluation, document)  # 明确的真实 schema 路径。
    with pytest.raises(analysis.AnalysisError, match=field):  # 不返回含混的 passed=False。
        _evaluate(evaluation, cohort)


@pytest.mark.parametrize("fault", ["missing_protocol", "replicas", "safe_grid", "duplicate_rows", "missing_asset", "mother", "direction", "nan", "infinity", "finite", "negative_path", "coverage_finite", "threshold", "protocol_threshold", "reported_assets", "reported_topologies", "reported_rows", "reported_topology_table", "asset_denominator", "topology_denominator"])
def test_invalid_data_and_forged_coverage_are_rejected(tmp_path: Path, fault: str) -> None:
    r"""原始资产、比例分母、R 计数/集合/拓扑表须构成相互一致的证据。"""
    evaluation, cohort, document, _ = _case(tmp_path, [4] * 32)  # 完全通过的可核对基准。
    asset = document["physical_rotation"]["asset_results"][0]  # 所有资产错例局限于第一行。
    reported = document["reliable_topology_coverage"]  # 只改派生报告时原始事实保持不变。
    asset_changes = {"replicas": ("replica_count", 15), "safe_grid": ("safe_replica_fraction", 0.76), "duplicate_rows": ("dataset_row", 1), "mother": ("mother_id", "wrong"), "direction": ("directional_consistency", 0.7), "nan": ("net_turns_median", float("nan")), "infinity": ("absolute_path_turns_median", float("inf")), "finite": ("finite", False), "negative_path": ("absolute_path_turns_median", -1.0)}
    if fault in asset_changes:  # 单个原始资产声明与正式定义冲突。
        key, value = asset_changes[fault]  # 选择反例的唯一变更。
        asset[key] = value  # 不自动修补派生表。
    if fault == "missing_protocol":  # 对旧缺字段评价必须明确拒绝。
        del document["evaluation_identity"]["protocol"]["actor_relay"]  # missing 不等于 JSON null。
    if fault == "missing_asset":  # 失败样本不能被删除后缩小分母。
        document["physical_rotation"]["asset_results"].pop()  # 127 行伪装 A128。
    report_changes = {"coverage_finite": ("finite", False), "reported_assets": ("passed_asset_count", 127), "reported_topologies": ("passed_topology_count", 31), "reported_rows": ("passed_asset_rows", [0] * 128), "asset_denominator": ("asset_count", 127), "topology_denominator": ("topology_count", 31)}
    if fault in report_changes:  # 各类 R 伪报均需独立发现。
        key, value = report_changes[fault]  # 不依赖某一个布尔总开关。
        reported[key] = value  # 与 128 原始行的重计结果不符。
    if fault == "reported_topology_table":  # 总通过数没变，但具体拓扑表造假。
        reported["topology_results"][0]["passed_asset_count"] = 3  # 必须逐拓扑核对。
    if fault == "threshold":  # R 不能使用两圈或 .85 方向门。
        reported["thresholds"]["directional_consistency_min"] = 0.85  # 与固定 .7 不同。
    if fault == "protocol_threshold":  # identity 内重复阈值声明也要核对。
        document["evaluation_identity"]["protocol"]["reliable_topology_coverage_thresholds"]["net_turns_min"] = 2.0
    _dump(evaluation, document)  # 把反例写成独立 CPU 输入。
    with pytest.raises(analysis.AnalysisError):  # 任何一种不合法都不能被当作正常门失败。
        _evaluate(evaluation, cohort)


def test_inflated_pass_counts_cannot_promote_failed_assets(tmp_path: Path) -> None:
    r"""真实 0/128、0/32 的资产不能靠伪报 103、29 两个数字晋升强门。"""
    evaluation, cohort, document, _ = _case(tmp_path, [0] * 32)  # 所有原始资产仅 0.4 圈。
    document["reliable_topology_coverage"].update(passed_asset_count=103, passed_topology_count=29)  # 直接伪造强门数字。
    _dump(evaluation, document)  # 身份、协议和原始资产仍保持合法。
    with pytest.raises(analysis.AnalysisError, match="passed_asset_count"):  # 必须重计，不能直接判自报计数。
        _evaluate(evaluation, cohort)


@pytest.mark.parametrize("fault", ["index", "representatives", "group", "schema"])
def test_canonical_topology_axis_is_rebuilt_from_members(tmp_path: Path, fault: str) -> None:
    r"""即使 manifest hash 随内容更新，错误的 canonical 轴或拓扑仍不能通过。"""
    evaluation, cohort_path, document, cohort = _case(tmp_path, [4] * 32)  # 所有资产本来通过。
    if fault == "index":  # evaluator 使用成员位置，cohort_index 必须逐项对齐。
        cohort["members"][0]["cohort_index"] = 1  # 不能借排序悄悄重绑定。
    if fault == "representatives":  # 3/5 代表错配不能套用 2/4 分母。
        cohort["members"][0]["provenance"]["mother_name"] = "mother-01"  # 总资产数仍为 128。
    if fault == "group":  # mother_name 不变时也必须保留 group 身份。
        for member in cohort["members"][:4]:  # 整个拓扑改 group，仍是合法 32×4。
            member["provenance"]["group_name"] = "different-family"  # R 表的 topology_id 将不匹配。
    if fault == "schema":  # 非 canonical-final schema 不能承担成员位置绑定。
        cohort["schema_version"] = "1.1.0"  # 必须明确拒绝。
    _dump(cohort_path, cohort)  # 模拟调用者提供另一个完整集合。
    document["evaluation_identity"]["manifest_sha256"] = hashlib.sha256(cohort_path.read_bytes()).hexdigest()  # 排除纯 hash 错例。
    _dump(evaluation, document)  # 保持三个身份锚点本身合法。
    with pytest.raises(analysis.AnalysisError):  # 还必须通过拓扑级科学绑定。
        _evaluate(evaluation, cohort_path)


def test_direction_uses_official_epsilon_and_negative_clipping(tmp_path: Path) -> None:
    r"""$D=clip(max(N,0)/max(P,2^{-23}),0,1)$ 在短路径和反向旋转时保持正式定义。"""
    evaluation, cohort, document, _ = _case(tmp_path, [0] * 32)  # 这些边界行都不可能满足一圈门。
    assets = document["physical_rotation"]["asset_results"]  # 修改已失败资产，不改变 R 的通过集合。
    assets[0].update(net_turns_median=2**-24, absolute_path_turns_median=2**-25, directional_consistency=0.5)  # epsilon 控制分母。
    assets[1].update(net_turns_median=-1.0, absolute_path_turns_median=2.0, directional_consistency=0.0)  # 反向净圈裁到零方向性。
    assets[2].update(net_turns_median=0.8, absolute_path_turns_median=0.5, directional_consistency=1.0)  # 上端 clip 正式存在。
    _dump(evaluation, document)  # 这些数值可单独验证比值公式。
    result = _evaluate(evaluation, cohort)  # 不能改为逐轨方向比值中位数。
    assert result["passed"] is False, result  # 合法但科学未达标。
    assert [row["directional_consistency"] for row in result["asset_results"][:3]] == [0.5, 0.0, 1.0], result  # epsilon、下端和上端裁剪三种边界。


def test_cli_outputs_are_exclusive_and_evaluation_is_bound(tmp_path: Path) -> None:
    r"""两个子命令都可直接调用，--output 只创建新文件且强制 expected 身份参数。"""
    evaluation, cohort, _, _ = _case(tmp_path, [0] * 32)  # 合法科学失败也应正常输出报告。
    output = tmp_path / "gate.json"  # 原目录中的新结果路径。
    args = ["evaluate", str(evaluation), str(cohort), "--expected-method-identity-digest", _METHOD, "--expected-checkpoint-sha256", _CHECKPOINT, "--output", str(output)]
    assert analysis.main(args) == 0, "合法门失败的 CLI 状态仍是成功完成分析"
    original = output.read_bytes()  # 用字节守恒证明没有覆盖。
    with pytest.raises(SystemExit) as error:  # 排他创建冲突是输入/输出错误。
        analysis.main(args)  # 同一路径再次运行。
    assert error.value.code == 2 and output.read_bytes() == original, "已有结果必须原样保留"
    with pytest.raises(SystemExit):  # 无 expected 绑定不能启动固定门分析。
        analysis.main(["evaluate", str(evaluation), str(cohort)])
    metrics, nn = _run(tmp_path, [_row(2000)], (2000,))  # 第二子命令独立消费训练指标。
    selection = tmp_path / "selection.json"  # 不覆盖 gate 报告。
    assert analysis.main(["select", str(metrics), str(nn), "--output", str(selection)]) == 0, "选择 CLI 应正常完成"
    assert len(json.loads(selection.read_text())["candidates"]) == 1, "终点最佳应去重"


def test_file_entrypoint_has_no_model_or_mdp_imports(tmp_path: Path) -> None:
    r"""在全新 CPU 解释器中阻断模型/MDP 加载，实际执行两个产物分析入口。"""
    evaluation, cohort, _, _ = _case(tmp_path, [0] * 32)  # 只用合成 JSON，不读任何真实训练日志。
    metrics, nn = _run(tmp_path, [_row(2000)], (2000,))  # 只用合成 Parquet 和非模型保存点。
    arguments = [
        ["--help"],  # 注册的两个命令应可见。
        ["select", str(metrics), str(nn), "--output", str(tmp_path / "guard-selection.json")],  # 真正执行 Polars collect。
        ["evaluate", str(evaluation), str(cohort), "--expected-method-identity-digest", _METHOD, "--expected-checkpoint-sha256", _CHECKPOINT, "--output", str(tmp_path / "guard-gate.json")],  # 真正执行固定门复算。
    ]
    probe = r'''
import importlib.abc  # 导入边界探针只使用标准库。
import importlib.util  # find_spec 可以探测依赖，但不能执行其模块。
import json  # 接收两个真实 CLI 的有限输入。
import runpy  # 执行真实文件入口。
import sys  # 安装禁止运行时依赖的导入哨兵。
class BlockedLoader(importlib.abc.Loader):
    """只在真正执行模块时拒绝，允许 Polars 的可用性探测。"""
    def create_module(self, spec):
        """使用 Python 缺省模块创建；执行前仍有硬边界。"""
        return None  # 此时还没有执行目标模块。
    def exec_module(self, module):
        """禁止读取/执行 Torch 或环境模块的实现。"""
        raise AssertionError("禁止模型/MDP 导入: " + module.__name__)
class Guard(importlib.abc.MetaPathFinder):
    """把运行时依赖的真实加载替换为拒绝加载器。"""
    def find_spec(self, fullname, path=None, target=None):
        """区分 find_spec 元数据探测和真正模块执行。"""
        if fullname.split(".")[0] in {"torch", "anymani", "isaaclab", "isaacsim", "omni"}:
            return importlib.util.spec_from_loader(fullname, BlockedLoader())  # 元数据探测本身不是导入执行。
sys.meta_path.insert(0, Guard())  # 禁令涵盖间接导入。
source = sys.argv[1]  # 唯一被测源码路径。
arguments = json.loads(sys.argv[2])  # 固定三次有界调用。
for args in arguments:  # 模块导入与真正读取产物都必须满足边界。
    sys.argv = [source, *args]  # CLI 参数来自测试自己的临时输入。
    try:
        runpy.run_path(source, run_name="__main__")  # 与推荐文件执行方式相同。
    except SystemExit as exit_status:
        assert exit_status.code == 0, exit_status.code  # 所有调用都应合法完成。
assert not any(name.split(".")[0] in {"torch", "anymani", "isaaclab", "isaacsim", "omni"} for name in sys.modules)  # 不得留有真实运行时模块。
'''
    process = subprocess.run([sys.executable, "-B", "-c", probe, str(_SOURCE), json.dumps(arguments)], capture_output=True, text=True, timeout=30)  # 有界 CPU 子进程。
    assert process.returncode == 0, process.stderr  # 任何禁止导入都会产生非零退出。
    assert "select" in process.stdout and "evaluate" in process.stdout, process.stdout  # 两个窄入口均已注册。
