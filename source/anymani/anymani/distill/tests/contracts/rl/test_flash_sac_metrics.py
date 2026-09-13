r"""训练回合首30秒统计的 CPU 合同：实际净圈、物理失败、删失与资产等权。

所有输入是 N=4 的 pre-reset 事实，不启动 Isaac 或 GPU。数值期望用明确的
episode 列表计算，区分每资产均值、窗口池均值和安全比例，不能解释为冻结 R16 评价。
"""

from __future__ import annotations

import copy
import io
import math
from types import MappingProxyType

import pytest
import torch
from anymani.distill.rl.flash_sac.metrics import FirstThirtySecondsMetrics


def _metrics(limit: int = 32, assets=(0, 0, 1, 1)) -> FirstThirtySecondsMetrics:
    r"""构造两个资产、四个副本的小型统计器；允许测试非连续资产索引。"""
    return FirstThirtySecondsMetrics(torch.tensor(assets), max_episodes_per_asset=limit)  # 默认每资产最近32个窗口。


def _snapshot(
    durations=(0.0, 0.0, 0.0, 0.0),  # 四个环境实际已经过的秒数。
    first30=(0.0, 0.0, 0.0, 0.0),  # 上游冻结的首30秒净圈。
    terminal_turns=(0.0, 0.0, 0.0, 0.0),  # 测试给定净圈，转为输入需要的终止弧度。
    *,
    drop=(),  # 当前步drop副本索引。
    axis=(),  # 当前步axis失败副本索引。
    timeout=(),  # 当前步timeout副本索引。
) -> dict[str, torch.Tensor]:
    r"""生成独立的七字段事实，净圈与终止弧度刻意分开，防止误用全回合终点。"""
    duration = torch.tensor(durations, dtype=torch.float64)  # 当前回合实际经过秒数，不是计划时域。
    snapshot = {  # 双精度测试避免夹杂输入量化误差；实现也须接受真实上游 FP32。
        "episode_duration_s": duration,  # [N] elapsed seconds。
        "net_rotation_rad": torch.tensor(terminal_turns, dtype=torch.float64) * (2 * math.pi),  # 有符号弧度。
        "net_turns_first30": torch.tensor(first30, dtype=torch.float64),  # 上游冻结的首30秒净圈。
        "first30_complete": duration >= 30.0,  # 与 command 的时窗完成事实一致。
        "termination_object_out_of_anchor": torch.zeros(4, dtype=torch.bool),  # 物体掉落/脱离锚点。
        "termination_goal_axis_misaligned": torch.zeros(4, dtype=torch.bool),  # 旋转轴偏离。
        "termination_time_out": torch.zeros(4, dtype=torch.bool),  # 实际计划时域结束。
    }
    for name, indices in (  # 同一行允许多种终止同时成立，统计必须只计一个 episode。
        ("termination_object_out_of_anchor", drop),  # drop 物理失败。
        ("termination_goal_axis_misaligned", axis),  # axis 物理失败。
        ("termination_time_out", timeout),  # timeout 不自动等于首30秒失败。
    ):
        snapshot[name][list(indices)] = True  # 所有标志都是 pre-reset 当前步的事实。
    return snapshot  # 只有简单 CPU tensor，无模拟器依赖。


def _assert_state_equal(actual: dict, expected: dict) -> None:
    r"""递归核对恢复状态，tensor 逐位一致，窗口保持 Python 标量。"""
    assert actual.keys() == expected.keys(), "状态字段集合发生变化"
    for name, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[name], value, rtol=0, atol=0)  # 包含 shape 与 dtype。
        elif isinstance(value, dict):
            _assert_state_equal(actual[name], value)  # 资产窗口/删失计数或配置。
        else:
            assert actual[name] == value, f"恢复内容不同：{name}"


def test_normal_first30_is_once_per_episode_and_uses_frozen_window_value() -> None:
    r"""29.95秒无结论；首30秒第一次完成立即记录，120秒终点不能改写该记录。"""
    metrics = _metrics()  # 空统计不能把未知能力输出为零。
    assert metrics.summary() == {  # 计数是已知的零；只有统计估计量是 None。
        "assets_with_window": 0,  # 尚无覆盖资产。
        "first30_episode_count": 0,  # 尚无有效窗口。
        "first30_asset_mean_net": None,  # 未定义资产等权均值。
        "first30_episode_mean_net": None,  # 未定义池均值。
        "first30_safe_fraction": None,  # 未定义安全比例。
        "censored_early_timeouts": 0,  # 累计删失事件数。
    }
    metrics.observe(_snapshot((29.95, 0, 0, 0)), policy_version=4)  # 未完成、未失败，继续等待。
    assert metrics.summary()["first30_episode_count"] == 0, "不能提前把不足30秒的在途回合计入窗口"
    metrics.observe(_snapshot((30, 0, 0, 0), first30=(1.25, 0, 0, 0), terminal_turns=(90, 0, 0, 0)), 5)
    expected = metrics.summary()  # 首30秒结算时的真实值为1.25圈，90圈是不同的输入字段。
    assert expected["first30_asset_mean_net"] == 1.25 and expected["first30_safe_fraction"] == 1.0
    metrics.observe(_snapshot((120, 0, 0, 0), first30=(1.25, 0, 0, 0), terminal_turns=(99, 0, 0, 0), timeout=(0,)), 8)
    assert metrics.summary() == expected, "后续120秒回合终点不能重复结算或覆盖首30秒结果"
    metrics.observe(_snapshot((30, 0, 0, 0), first30=(2.75, 0, 0, 0)), 11)  # done 已清 seen，下一回合可再次记录。
    assert metrics.summary()["first30_episode_count"] == 2, "timeout 后应允许下一回合"
    assert metrics.state_dict()["windows"][0] == [(1.25, True, 5), (2.75, True, 11)], "必须保存结算时策略版本"


@pytest.mark.parametrize("failure", ["drop", "axis"])
def test_late_failure_cannot_pollute_an_already_recorded_first30(failure: str) -> None:
    r"""30秒已经安全完成的回合，后来失败只清 seen，不回写安全标志或净圈。"""
    metrics = _metrics()  # 未达30秒的其他副本始终不贡献统计。
    metrics.observe(_snapshot((30, 0, 0, 0), first30=(2, 0, 0, 0)), 1)  # 已有安全首30秒事实。
    before = metrics.summary()  # 对照包含窗口池安全比例。
    late = _snapshot((45, 0, 0, 0), first30=(2, 0, 0, 0), terminal_turns=(-100, 0, 0, 0), **{failure: (0,)})
    metrics.observe(late, 2)  # 后期失败的-100圈与首30秒结果无关。
    assert metrics.summary() == before, "后期失败污染了已经完成的首30秒结果"
    metrics.observe(_snapshot((4, 0, 0, 0), terminal_turns=(-0.5, 0, 0, 0), drop=(0,)), 3)  # 下一 episode 的早期失败。
    assert metrics.state_dict()["windows"][0] == [(2.0, True, 1), (-0.5, False, 3)], "失败必须只影响自己的回合"


def test_early_failures_keep_actual_signed_turns_and_coincident_bits_count_once() -> None:
    r"""5秒失败的一圈仍是一圈；drop+axis+timeout 同时发生也只有一个失败窗口。"""
    metrics = _metrics()  # 每资产保留自身最近窗口，先检验物理归因。
    metrics.observe(  # 环境3只有早 timeout；其他三行均有真实物理失败。
        _snapshot(
            (5, 10, 20, 29),  # 所有环境均未到30秒。
            first30=(999, 999, 999, 999),  # 未完成标志下此字段不能被当作有效结果。
            terminal_turns=(1, -0.5, 0.25, 100),  # 真实有符号终点转角对应的圈数。
            drop=(0, 1),  # 两行drop。
            axis=(1, 2),  # 环境1同时有drop和axis，只应计一次。
            timeout=(1, 3),  # 环境1有真实失败，只有环境3是纯早timeout。
        ),
        policy_version=7,
    )
    state = metrics.state_dict()  # 逐资产的 Python 标量窗口。
    assert state["windows"][0] == [(1.0, False, 7), (-0.5, False, 7)], "提前失败不能按30/duration放大"
    assert state["windows"][1] == [(0.25, False, 7)], "同时终止位不能重复记账，纯早timeout必须排除"
    summary = metrics.summary()  # 三个有效失败窗口，另有一个删失回合。
    assert summary["first30_episode_count"] == 3 and summary["first30_safe_fraction"] == 0.0
    assert summary["censored_early_timeouts"] == 1, "真实失败与timeout同时发生时不得再算删失"
    assert metrics.summary([0])["censored_early_timeouts"] == 0, "删失计数也必须按资产筛选"
    assert metrics.summary([1])["censored_early_timeouts"] == 1, "删失应归属实际发生的资产"


@pytest.mark.parametrize(
    "flags", [(), ("drop",), ("axis",), ("drop", "axis"), ("timeout",), ("drop", "axis", "timeout")]
)
def test_completion_step_uses_first30_value_and_physical_safety(flags: tuple[str, ...]) -> None:
    r"""恰在30秒完成时用首30净圈；该步物理失败则 unsafe，纯 timeout 仍可安全完成。"""
    metrics = _metrics()  # 单窗口即可明确验证完成/物理失败的优先关系。
    snapshot = _snapshot(  # 完成步可以同时触发任意组合的物理结束位。
        (30, 0, 0, 0), first30=(2, 0, 0, 0), terminal_turns=(50, 0, 0, 0), **{name: (0,) for name in flags}
    )
    metrics.observe(snapshot, 9)  # 首30冻结值与该步总转角字段故意不等。
    safe = not ("drop" in flags or "axis" in flags)  # timeout 只结束 episode，不伪造物理失败。
    assert metrics.state_dict()["windows"][0] == [(2.0, safe, 9)], "完成步应只结算一次首30秒窗口"
    assert metrics.summary()["censored_early_timeouts"] == 0, "已经到30秒的timeout不是早期删失"


def test_early_timeout_is_censored_and_cannot_evict_a_valid_window() -> None:
    r"""删失回合只增加累计计数，不进入近期窗口、均值、安全分母或deque容量。"""
    metrics = _metrics(limit=1)  # 若删失错误进入deque，会挤掉唯一有效记录。
    metrics.observe(_snapshot((30, 0, 0, 0), first30=(3, 0, 0, 0), timeout=(0,)), 1)  # 有效完成并立即结束。
    for duration in (5, 20):
        snapshot = _snapshot((duration, 0, 0, 0), terminal_turns=(100, 0, 0, 0), timeout=(0,))  # 纯早timeout。
        metrics.observe(snapshot, 2)  # 每次均为一个独立短回合，100圈也不能加入有效窗口。
    summary = metrics.summary()  # 原有效窗口仍是唯一估计依据。
    assert summary["first30_episode_count"] == 1 and summary["first30_asset_mean_net"] == 3.0
    assert summary["first30_safe_fraction"] == 1.0 and summary["censored_early_timeouts"] == 2
    assert metrics.summary([1])["first30_asset_mean_net"] is None, "未覆盖资产不可填成零能力"


def test_asset_equal_mean_differs_from_episode_pool_and_subsets_are_sets() -> None:
    r"""资产7快速失败三次、资产2完成一次：等权均值5，池均值3，安全比例1/4。"""
    metrics = _metrics(assets=(7, 7, 2, 9))  # 非连续索引，资产9完全没有有效数据。
    for version in (1, 2, 3):
        metrics.observe(_snapshot((1, 0, 0, 0), terminal_turns=(1, 0, 0, 0), drop=(0,)), version)  # 每次均是新回合。
    metrics.observe(_snapshot((0, 0, 30, 0), first30=(0, 0, 9, 0)), 4)  # 慢完成资产仅一个窗口。
    summary = metrics.summary()  # 不按资产终止速度为主指标加权。
    assert summary["assets_with_window"] == 2 and summary["first30_episode_count"] == 4
    assert summary["first30_asset_mean_net"] == 5.0, "资产均值应先分别为1和9，再等权平均"
    assert summary["first30_episode_mean_net"] == 3.0, "池均值仅作不同采样速度的诊断"
    assert summary["first30_safe_fraction"] == 0.25, "安全比例应使用同一窗口池分母"
    assert metrics.summary([7, 7, 2]) == summary, "重复资产索引不能隐式增加组权重"
    assert metrics.summary([2, 9])["first30_asset_mean_net"] == 9.0, "组内未覆盖资产不应当成零样本"
    assert metrics.summary([])["first30_episode_mean_net"] is None, "空组均值未定义"
    with pytest.raises(ValueError, match="asset"):
        metrics.summary([99])  # 不允许把未知资产拼写错误静默解释为未覆盖。


def test_deque_truncates_per_asset_and_has_deterministic_simultaneous_order() -> None:
    r"""每资产只留最近M个有效窗口；同一步多个副本按环境索引顺序入队。"""
    metrics = _metrics(limit=2)  # 两个副本共享同一个资产deque，而非各有M个窗口。
    for version, net in ((11, -1), (12, 2), (13, 5)):
        metrics.observe(_snapshot((30, 0, 0, 0), first30=(net, 0, 0, 0), timeout=(0,)), version)  # 连续三个完整回合。
    assert metrics.state_dict()["windows"][0] == [(2.0, True, 12), (5.0, True, 13)], "deque必须丢弃最早窗口"
    assert metrics.summary()["first30_asset_mean_net"] == 3.5, "近期窗口均值不应混入被淘汰的-1"
    metrics.observe(_snapshot((30, 30, 0, 0), first30=(10, 20, 0, 0), timeout=(0, 1)), 14)  # 同资产同时完成两个副本。
    assert metrics.state_dict()["windows"][0] == [(10.0, True, 14), (20.0, True, 14)], "同一步的入队次序应可恢复"


def test_checkpoint_preserves_seen_windows_versions_and_subset_stream_reset() -> None:
    r"""恢复后不重复计入已完成窗口；只清指定流的seen，历史窗口与删失计数继续保留。"""
    metrics = _metrics(limit=3)  # 两个完成但尚未物理结束的回合。
    metrics.observe(_snapshot((30, 10, 30, 5), first30=(1, 0, 4, 0), timeout=(3,)), 6)  # 资产1另有一个早timeout。
    stream = io.BytesIO()  # 内存序列化，避免写入拥有范围之外的测试文件。
    torch.save(metrics.state_dict(), stream)  # 状态包括已结算 seen、Python窗口和累计删失。
    stream.seek(0)  # 新对象恢复相同配置与映射。
    state = torch.load(stream, map_location="cpu", weights_only=True)  # 仅CPU反序列化。
    restored = _metrics(limit=3)  # 不能依赖之前对象的deque或tensor别名。
    restored.load_state_dict(state)  # 完整验证后恢复。
    restored.observe(_snapshot((31, 11, 31, 0), first30=(1, 0, 4, 0)), 7)  # 不应重复计算已完成的两个窗口。
    assert restored.summary() == metrics.summary(), "恢复丢失seen会把同回合窗口计两次"
    before = restored.summary()  # reset_streams 不删历史统计。
    restored.reset_streams(torch.tensor([0, 0]))  # 重建环境0，重复索引只清一次。
    restored.reset_streams(torch.tensor([], dtype=torch.long))  # 空子集无操作。
    assert restored.summary() == before, "流重建不能清除已有窗口或累计删失"
    restored.observe(_snapshot((30, 12, 32, 0), first30=(3, 0, 4, 0)), 8)  # 新环境0完成，环境2仍是旧episode。
    assert restored.state_dict()["windows"][0] == [(1.0, True, 6), (3.0, True, 8)], "指定新流应允许新回合结算"
    assert restored.state_dict()["windows"][1] == [(4.0, True, 6)], "未重置环境不能重复结算"
    assert restored.summary()["censored_early_timeouts"] == 1, "流重置不能清除删失计数"
    before = restored.summary()  # 默认 reset 清全部seen，仍不改累计统计。
    restored.reset_streams()  # 下一份新物理环境快照可正常结算。
    assert not restored.state_dict()["seen"].any() and restored.summary() == before
    state["windows"][0][0] = (999.0, False, 999)  # 加载源被外部复用，不能改写已恢复deque。
    assert restored.summary() == before, "load_state_dict 不得长期别名外部窗口列表"


def test_state_snapshot_is_independent_and_deques_contain_only_python_scalars() -> None:
    r"""checkpoint 是小状态快照；后续observe不修改旧快照，窗口不持有tensor或梯度图。"""
    metrics = _metrics()  # 初始资产映射也须脱离外部可变tensor。
    snapshot = _snapshot((30, 0, 0, 0), first30=(1.5, 0, 0, 0), timeout=(0,))  # 一个已完成episode。
    snapshot["net_turns_first30"].requires_grad_()  # 故意提供带梯度数值，统计不应延长计算图生命周期。
    metrics.observe(snapshot, 10)  # 只保存Python float/bool/int。
    saved = metrics.state_dict()  # 窗口与seen应是独立小副本。
    expected = copy.deepcopy(saved)  # 逐位比较基准。
    metrics.observe(_snapshot((30, 0, 0, 0), first30=(2.5, 0, 0, 0), timeout=(0,)), 11)
    _assert_state_equal(saved, expected)
    for windows in metrics.state_dict()["windows"].values():
        for net, safe, version in windows:
            assert type(net) is float and type(safe) is bool and type(version) is int, "deque只能保存Python标量"


def test_snapshot_superset_and_cpu_copy_path_only_transfers_new_event_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""用CPU路径审计搬运粒度：仅新窗口/删失事件行，不搬整个N轴快照或额外字段。"""
    metrics = _metrics()  # 先完成一次性的资产metadata准备，再监视observe期间的cpu搬运入口。
    transfers = []  # 只记录显式Tensor.cpu调用的输入shape，不启动GPU。
    original_cpu = torch.Tensor.cpu  # 保留真实CPU张量行为。

    def record_cpu(tensor, *args, **kwargs):
        r"""监视稀疏事件搬运的行数，同时执行原Tensor.cpu实现。"""
        transfers.append(tuple(tensor.shape))  # 输入形状可证明是否提前选出了事件子集。
        return original_cpu(tensor, *args, **kwargs)  # 数值路径不改写。

    monkeypatch.setattr(torch.Tensor, "cpu", record_cpu)  # 审计的是生产observe所经过的真实搬运入口。
    snapshot = _snapshot((30, 29, 5, 2), first30=(1, 0, 0, 0), terminal_turns=(0, 0, -1, 0), axis=(2,), timeout=(3,))
    snapshot["unrelated_diagnostic"] = torch.zeros(4, 21, 3)  # 上游完整snapshot的其他字段不需要搬运。
    metrics.observe(MappingProxyType(snapshot), 1)  # Mapping接口与完整快照超集都必须可用。
    assert sorted(transfers) == [(3,), (3, 3)], "只应搬运3个新事件的索引及(net,safe,censored)载荷"
    transfers.clear()  # 下一步没有新结算事件。
    metrics.observe(_snapshot((31, 29.5, 0, 0), first30=(1, 0, 0, 0)), 2)  # 已seen的完整窗口不再搬运。
    assert not transfers, "无新窗口时不应搬运任何完整快照行"


@pytest.mark.parametrize("case", ["missing", "shape", "bool", "integer", "nan", "negative_duration", "completion"])
def test_invalid_snapshot_is_rejected_before_seen_or_windows_change(case: str) -> None:
    r"""不一致的时长/完成标志、错误shape/dtype或非有限数值必须在统计前整体拒绝。"""
    metrics = _metrics()  # 已有一条事实，用于检测非法输入的部分写入。
    metrics.observe(_snapshot((30, 0, 0, 0), first30=(1, 0, 0, 0)), 1)  # env0已seen。
    before = metrics.state_dict()  # 小状态独立快照。
    snapshot = _snapshot((31, 30, 0, 0), first30=(1, 2, 0, 0))  # 本来会新增env1窗口。
    if case == "missing":
        del snapshot["net_rotation_rad"]  # 提前失败必需的弧度真值不能缺失。
    elif case == "shape":
        snapshot["net_rotation_rad"] = torch.zeros(4, 1)  # [N,1]不能广播成[N]。
    elif case == "bool":
        snapshot["first30_complete"] = snapshot["first30_complete"].float()  # 完成标志不是连续数值。
    elif case == "integer":
        snapshot["episode_duration_s"] = snapshot["episode_duration_s"].long()  # 秒数必须是浮点输入。
    elif case == "nan":
        snapshot["net_turns_first30"][1] = float("nan")  # 新窗口不能污染近期均值。
    elif case == "negative_duration":
        snapshot["episode_duration_s"][3] = -1  # 实际回合时长不能为负。
    else:
        snapshot["first30_complete"][1] = False  # 已达30秒却未完成，不符合首30秒producer合同。
    with pytest.raises((ValueError, TypeError)):
        metrics.observe(snapshot, 2)  # 应在更新seen和deque之前失败。
    _assert_state_equal(metrics.state_dict(), before)


@pytest.mark.parametrize("version", [-1, True, 1.5])
def test_policy_version_is_an_explicit_nonnegative_integer(version) -> None:
    r"""窗口的策略版本保留精确整数身份，不截断浮点或把bool当版本。"""
    metrics = _metrics()  # 无效版本不能触发新窗口。
    with pytest.raises((ValueError, TypeError)):
        metrics.observe(_snapshot((30, 0, 0, 0)), version)  # 校验优先于统计副作用。
    assert metrics.summary()["first30_episode_count"] == 0, "非法版本仍产生了窗口"


@pytest.mark.parametrize(
    "indices",
    [torch.tensor([0, 4]), torch.tensor([-1]), torch.tensor([0.0]), torch.tensor([[0]]), torch.tensor([True])],
)
def test_reset_streams_rejects_bad_subset_without_partial_clear(indices: torch.Tensor) -> None:
    r"""错误子集不能先清一部分seen，再因越界或dtype错误失败。"""
    metrics = _metrics()  # 四环境均已seen，任意错误清零都可被发现。
    metrics.observe(_snapshot((30, 30, 30, 30)), 1)  # 每资产两个有效窗口。
    before = metrics.state_dict()  # 包含完整去重状态。
    with pytest.raises((ValueError, TypeError)):
        metrics.reset_streams(indices)  # 索引不是bool mask，也不能负索引回绕。
    _assert_state_equal(metrics.state_dict(), before)


@pytest.mark.parametrize(
    "case", ["config", "mapping", "seen", "window_count", "net", "safe", "version", "censored", "assets", "extra"]
)
def test_checkpoint_validation_is_strict_and_atomic(case: str) -> None:
    r"""恢复必须验证映射、窗口上限、逐窗口标量及计数；拒绝静默截断或部分加载。"""
    metrics = _metrics(limit=2)  # checkpoint必须匹配每资产容量。
    metrics.observe(_snapshot((30, 0, 0, 0), first30=(1, 0, 0, 0)), 4)  # 有一条窗口及seen状态。
    before = metrics.state_dict()  # 失败前的完整状态。
    bad = copy.deepcopy(before)  # 每次只破坏一种合同。
    if case == "config":
        bad["max_episodes_per_asset"] = 3  # 同数据、不同窗口宽度也改变统计语义。
    elif case == "mapping":
        bad["asset_index_by_env"] = torch.tensor([1, 0, 1, 0])  # 相同资产集合但副本归属不同。
    elif case == "seen":
        bad["seen"] = bad["seen"].float()  # 不能隐式转换去重标志。
    elif case == "window_count":
        bad["windows"][0] *= 3  # 超出容量必须报错，不得由deque自动丢样本。
    elif case == "net":
        bad["windows"][0][0] = (float("nan"), True, 4)  # 池均值不能含非有限值。
    elif case == "safe":
        bad["windows"][0][0] = (1.0, 1, 4)  # 安全标志必须是Python bool。
    elif case == "version":
        bad["windows"][0][0] = (1.0, True, -4)  # 策略版本不能为负。
    elif case == "censored":
        bad["censored_early_timeouts"][0] = -1  # 累计计数不允许负数。
    elif case == "assets":
        bad["windows"][9] = []  # 未声明资产不能被偷偷加入分母。
    else:
        bad["unrecognized"] = 1  # checkpoint字段集合严格匹配。
    with pytest.raises((ValueError, TypeError)):
        metrics.load_state_dict(bad)  # 所有检查必须先于任何状态写入。
    _assert_state_equal(metrics.state_dict(), before)


@pytest.mark.parametrize(
    "assets,limit", [((), 32), ((0, -1, 1, 1), 32), ((0.0, 0.0, 1.0, 1.0), 32), ((0, 0, 1, 1), 0), ((0, 0, 1, 1), True)]
)
def test_invalid_asset_metadata_or_window_capacity_is_rejected(assets, limit) -> None:
    r"""资产索引需非空一维非负整数，每资产窗口上限需正整数。"""
    with pytest.raises((ValueError, TypeError)):
        _metrics(limit=limit, assets=assets)  # 不自动补资产、截断float索引或修正窗口长度。


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_physical_snapshot_precision_and_elapsed_boundary(dtype: torch.dtype) -> None:
    r"""真实上游FP32与审计FP64都按输入数值统计，29.95秒在途回合仍不得提前完成。"""
    metrics = _metrics()  # 环境0完整、环境1早失败、环境2在途、环境3删失。
    snapshot = _snapshot(  # 各行分别用于完成、早失败、在途和删失的数值审计。
        (30, 5, 29.95, 15), first30=(1.25, 0, 0, 0), terminal_turns=(0, -0.75, 0, 0), drop=(1,), timeout=(3,)
    )
    snapshot = {name: value.to(dtype) if value.is_floating_point() else value for name, value in snapshot.items()}
    expected_failure = float(snapshot["net_rotation_rad"][1]) / (2 * math.pi)  # 先保留真实输入精度，再换算圈数。
    metrics.observe(snapshot, 0)  # policy_version=0合法，浮点输入不能改变版本身份。
    summary = metrics.summary()  # 只有同一资产的两个有效窗口。
    assert summary["first30_episode_count"] == 2 and summary["assets_with_window"] == 1
    assert summary["first30_episode_mean_net"] == pytest.approx((1.25 + expected_failure) / 2, abs=1e-14)
    assert summary["first30_safe_fraction"] == 0.5 and summary["censored_early_timeouts"] == 1
