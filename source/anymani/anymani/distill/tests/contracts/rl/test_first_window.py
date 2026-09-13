r"""首30秒训练代理的纯CPU合同：证伪窗口池偏斜、样本门失效与恢复时的部分写入。

本文件直接加载被测叶模块，避免包初始化牵入任务注册、Torch或Isaac运行时。
所有轨迹事实均为手工构造的NumPy数组；通过这些测试只建立统计合同，不建立策略能力结论。
"""

from __future__ import annotations

import importlib.util
import json
import random
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

# 只执行统计叶模块；--confcutdir同样让本测试不依赖父级训练fixture。
_SOURCE = Path(__file__).resolve().parents[3] / "diagnostics/recording/rl/first_window.py"  # distill内固定入口
_SPEC = importlib.util.spec_from_file_location("_first_window_cpu_contract", _SOURCE)  # 不运行anymani.__init__
assert _SPEC is not None and _SPEC.loader is not None, f"无法加载CPU统计模块：{_SOURCE}"  # 明确失败边界
_MODULE = importlib.util.module_from_spec(_SPEC)  # 独立叶模块命名空间
_SPEC.loader.exec_module(_MODULE)  # 被测模块只可依赖NumPy和标准库
FirstWindowStatistics = _MODULE.FirstWindowStatistics  # 测试正式类接口，不构造替代实现

# 这些固定字段同时用于空资产、单资产与资产组；缺测和真实零必须可区分。
_FIELDS = {
    "first30_asset_count",  # 所选资产总数
    "first30_observed_assets",  # 至少一个窗口的资产数
    "first30_qualified_assets",  # 达到minimum_windows的资产数
    "first30_window_count",  # 当前FIFO内窗口总数
    "first30_windows_min",  # 包含缺测资产的最小样本量
    "first30_windows_max",  # 包含缺测资产的最大样本量
    "first30_net_median",  # 两层净转中位数，单位圈
    "first30_goal_median",  # 两层严格目标计数中位数
    "first30_direction_median",  # 两层有符号方向中位数
    "first30_safe_fraction",  # 资产等权安全比例
    "first30_one_turn_proxy_assets",  # 至少16窗口的一圈训练代理
    "first30_two_turn_proxy_assets",  # 至少16窗口的两圈训练代理
    "first30_policy_start_min",  # 留存窗口的最早起始策略版本
    "first30_policy_end_max",  # 留存窗口的最晚结束策略版本
}


def _columns(
    assets: Sequence[int],
    net: Sequence[float],
    *,
    path: Sequence[float] | None = None,
    goals: Sequence[int] | None = None,
    safe: Sequence[bool] | None = None,
    start: Sequence[int] | None = None,
    end: Sequence[int] | None = None,
) -> dict[str, np.ndarray]:
    r"""构造多个合同共用的已结算窗口；默认绝对路径为净转绝对值、安全且版本0→1。"""
    count = len(assets)  # $E$为已完成首窗或提前drop/axis结算的事件数
    net_array = np.asarray(net, dtype=np.float64)  # 有符号净转，单位圈
    return {
        "asset_index": np.asarray(assets, dtype=np.int64),  # 资产索引必须真正为整数
        "net_turns": net_array,  # 不把反向运动截成零
        "absolute_path_turns": np.abs(net_array) if path is None else np.asarray(path, dtype=np.float64),  # 路径圈数
        "goal_count": np.asarray([0] * count if goals is None else goals, dtype=np.int64),  # 严格目标事件计数
        "safe": np.asarray([True] * count if safe is None else safe, dtype=np.bool_),  # 环境已判定的安全事实
        "policy_version_start": np.asarray([0] * count if start is None else start, dtype=np.int64),  # 首窗起点
        "policy_version_end": np.asarray([1] * count if end is None else end, dtype=np.int64),  # 首窗结算点
    }


def test_assets_are_equal_weight_despite_fast_failure_pool() -> None:
    r"""一只手1个成功窗，另一只手32个快速失败窗；全局安全率应为1/2而非1/33。"""
    stats = FirstWindowStatistics(3)  # 第三只手未观测，不能补成失败窗口
    columns = _columns([0] + [1] * 32, [2.0] + [0.0] * 32, goals=[10] + [0] * 32, safe=[True] + [False] * 32)
    stats.add_batch(columns)  # 故意使回合事件池严重偏向易失败资产
    result = stats.summary()  # 先逐资产归约，再等权汇总有数据的资产
    assert result["first30_net_median"] == 1.0, result  # median([2,0])，池化中位数则为0
    assert result["first30_goal_median"] == 5.0, result  # 目标数也使用同样的资产测度
    assert result["first30_direction_median"] == 0.5, result  # 零路径方向为0
    assert result["first30_safe_fraction"] == 0.5, result  # mean([1,0])
    assert result["first30_safe_fraction"] != np.mean(columns["safe"]), result  # 明确排除1/33
    assert result["first30_window_count"] == 33, result  # 容量约束按资产独立应用
    assert (result["first30_observed_assets"], result["first30_qualified_assets"]) == (2, 1), result
    assert (result["first30_windows_min"], result["first30_windows_max"]) == (0, 32), result
    assert result["first30_one_turn_proxy_assets"] == 0, result  # 成功资产仅1窗，不能越过16窗门


def test_window_medians_are_not_means_or_ratio_of_medians() -> None:
    r"""同资产逐窗方向取中位数，不能用净转中位数除以路径中位数代替。"""
    stats = FirstWindowStatistics(1)  # 四个窗口仍应报告描述量，但不足以建立训练代理
    stats.add_batch(_columns([0] * 4, [0.0, 1.0, 3.0, 6.0], path=[1.0, 10.0, 3.0, 12.0], goals=[0, 100, 4, 8]))
    result = stats.per_asset()[0]  # 净转排序[0,1,3,6]，目标排序[0,4,8,100]
    assert result["first30_net_median"] == 2.0, result  # 偶数样本取中央两项平均
    assert result["first30_goal_median"] == 6.0, result  # 不受目标计数100这个离群窗支配
    assert result["first30_direction_median"] == pytest.approx(0.3), result  # median([0,.1,1,.5])
    assert result["first30_direction_median"] != pytest.approx(2.0 / 6.5), result  # 排除ratio-of-medians
    assert result["first30_qualified_assets"] == result["first30_two_turn_proxy_assets"] == 0, result


def test_sixteenth_window_opens_proxy_with_inclusive_thresholds() -> None:
    r"""净转2、方向0.7、安全12/16恰好达到门槛；15窗时仍不能声称代理通过。"""
    stats = FirstWindowStatistics(1)  # 固定proxy门为16，默认资格门也为16
    stats.add_batch(
        _columns([0] * 15, [0.0] * 7 + [2.0] * 8, path=[1.0] * 7 + [2.0 / 0.7] * 8, safe=[True] * 12 + [False] * 3)
    )  # 净转/方向中位数已过门，但样本仍不足
    assert stats.summary()["first30_one_turn_proxy_assets"] == 0, stats.summary()  # 15窗不能冒充16窗
    stats.add_batch(_columns([0], [2.0], path=[2.0 / 0.7], safe=[False], start=[5], end=[9]))  # 第16窗跨版本
    result = stats.summary()  # 12/16=.75，比较必须包含边界
    assert result["first30_safe_fraction"] == 0.75, result  # 安全率是窗口事实的比例
    assert result["first30_qualified_assets"] == 1, result  # 样本资格门已达到
    assert result["first30_one_turn_proxy_assets"] == result["first30_two_turn_proxy_assets"] == 1, result
    assert (result["first30_policy_start_min"], result["first30_policy_end_max"]) == (0, 9), result


@pytest.mark.parametrize(
    "net,path,safe_count,one,two",
    [
        (1.0, 1.0, 16, 1, 0),  # 恰好一圈仅计一圈代理
        (0.99, 1.0, 16, 0, 0),  # 大量goal也不能弥补真实净转不足
        (2.0, 4.0, 16, 0, 0),  # 正反抵消后方向仅0.5
        (2.0, 2.0, 11, 0, 0),  # 11/16低于安全门0.75
        (-2.0, 2.0, 16, 0, 0),  # 反向两圈不能当成正向两圈
    ],
)
def test_proxy_requires_joint_net_direction_and_safety(net, path, safe_count, one, two) -> None:
    r"""三个物理判据必须同时成立；goal_count不是轴向旋转的替代量。"""
    stats = FirstWindowStatistics(1)  # 每个反例均已有16窗，单独检验物理判据
    stats.add_batch(
        _columns(
            [0] * 16,  # 同一资产已达到16窗门
            [net] * 16,  # 固定净转，隔离其他判据的作用
            path=[path] * 16,  # 固定路径，方向为net/path
            goals=[1000] * 16,  # 再多goal也不能代替物理净转
            safe=[True] * safe_count + [False] * (16 - safe_count),  # 经验安全概率
        )
    )
    result = stats.summary()  # 高目标计数不进入一圈/两圈代理公式
    assert result["first30_one_turn_proxy_assets"] == one, result  # 一圈门的反例
    assert result["first30_two_turn_proxy_assets"] == two, result  # 两圈门的反例


@pytest.mark.parametrize("minimum,qualified", [(1, 1), (16, 1), (20, 0)])
def test_configurable_qualification_does_not_redefine_sixteen_window_proxy(minimum, qualified) -> None:
    r"""minimum_windows只标记描述量资格，proxy的至少16窗定义固定不变。"""
    stats = FirstWindowStatistics(1, minimum_windows=minimum)  # 显式区分可配置资格与固定proxy协议
    stats.add_batch(_columns([0] * 16, [2.0] * 16))  # 已满足16窗与所有物理门
    result = stats.summary()  # minimum=20时仍可观察到16窗训练代理，但qualified为0
    assert result["first30_qualified_assets"] == qualified, result  # 资格按显式配置
    assert result["first30_two_turn_proxy_assets"] == 1, result  # 不暗中修改proxy定义


def test_zero_and_negative_net_keep_signed_direction_and_ignore_ghost_columns() -> None:
    r"""静止、净零往返与反向运动分别给出0、0、负方向；ghost附加列没有统计作用。"""
    stats = FirstWindowStatistics(4)  # 第四资产没有任何完成窗口
    columns = _columns([0, 1, 2], [0.0, 0.0, -1.0], path=[0.0, 3.0, 2.0])  # 显式构造三种物理事实
    columns["ghost_goal_count"] = np.array([10000, 10000, 10000])  # 未消费的额外目标列
    columns["ghost_net_turns"] = np.array([np.nan])  # 附加诊断的形状/缺测不改变必需列合同
    stats.add_batch(columns)  # 接受必需列超集，绝不读取ghost来补充代理
    rows = stats.per_asset()  # 包含0..A-1全部资产
    assert [rows[i]["first30_direction_median"] for i in range(4)] == [0.0, 0.0, -0.5, None], rows
    assert rows[2]["first30_net_median"] == -1.0, rows[2]  # 负净转保持符号
    assert stats.summary()["first30_goal_median"] == 0.0, stats.summary()  # ghost不能伪造真实目标


@pytest.mark.parametrize(
    "selection,assets,observed,windows,minimum,maximum,net",
    [
        ([], 0, 0, 0, 0, 0, None),  # 空组
        ([2], 1, 0, 0, 0, 0, None),  # 单个缺测资产
        ([0], 1, 1, 1, 1, 1, 2.0),  # 单个有观测资产
        ([1, 0], 2, 2, 2, 1, 1, 0.0),  # 次序不赋予权重
        ([3, 0, 2], 3, 2, 3, 0, 2, 2.0),  # 组内含缺测资产
        (None, 4, 3, 4, 0, 2, 2.0),  # None等价全部资产
    ],
)
def test_empty_single_and_full_groups_share_exact_schema(
    selection, assets, observed, windows, minimum, maximum, net
) -> None:
    r"""组大小和窗口量包括缺测资产；浮点缺测使用None，不能伪造成物理零。"""
    stats = FirstWindowStatistics(4)  # 资产2缺测；资产3有两个窗口
    stats.add_batch(_columns([0, 1, 3, 3], [2.0, -2.0, 1.0, 3.0], start=[2, 5, 7, 1], end=[3, 6, 8, 9]))
    result = stats.summary(selection)  # 逐组重算，不从全局预聚合量隐式重加权
    assert set(result) == _FIELDS, result  # 锁定writer所需的14个标量字段
    assert (result["first30_asset_count"], result["first30_observed_assets"]) == (assets, observed), result
    assert (result["first30_window_count"], result["first30_windows_min"], result["first30_windows_max"]) == (
        windows,  # 当前窗口总数
        minimum,  # 含缺测资产的最小计数
        maximum,  # 含缺测资产的最大计数
    ), result
    assert result["first30_net_median"] == net, result  # 资产中位数分别为2、-2、2
    assert all(type(value) in (int, float, type(None)) for value in result.values()), result  # 只返回Python标量
    if not observed:  # 无观测时四个统计量与两个策略版本界均未定义
        for name in (
            "net_median",  # 无观测不等于零净转
            "goal_median",  # 未观察到任何目标事件不等于观察到零目标
            "direction_median",  # 无窗口时方向未定义
            "safe_fraction",  # 零分母不能伪装成安全率0或1
            "policy_start_min",  # 版本0是有效版本，不能用作缺测值
            "policy_end_max",  # 没有已结算窗口就没有结束版本
        ):
            assert result[f"first30_{name}"] is None, result  # 缺测不同于零运动或版本0
    if selection is not None and len(selection) == 1:  # 两个公开接口必须使用同一归约定义
        assert result == stats.per_asset()[selection[0]], result  # 单资产schema完全一致


@pytest.mark.parametrize("selection", [[0, 0], [-1], [2], [0.0], [True], [np.bool_(False)], "", {0}, np.array([[0]])])
def test_invalid_asset_selection_cannot_duplicate_or_truncate_weights(selection) -> None:
    r"""拒绝重复、越界、浮点/布尔索引与非一维序列，防止隐藏资产重加权。"""
    stats = FirstWindowStatistics(2)  # 合法集合仅由索引0、1组成
    stats.add_batch(_columns([0], [1.0]))  # 读请求出错也不应破坏既有事实
    before = stats.state_dict()  # 用完整有序窗口检查原子性
    with pytest.raises(ValueError):  # 不接受把0.0、False隐式转成0
        stats.summary(selection)
    assert stats.state_dict() == before, "非法资产选择改变了窗口历史"  # 读接口无副作用


def test_fifo_is_per_asset_ordered_and_detached_from_caller() -> None:
    r"""单批超容量与交错资产均按输入行顺序淘汰；调用方复用数组不改变留存证据。"""
    stats = FirstWindowStatistics(2, max_windows_per_asset=3, minimum_windows=2)  # 缩小FIFO便于人工核对
    columns = _columns(
        [0, 1, 0, 0, 1, 0], [0.0, 10.0, 2.0, 4.0, 20.0, 6.0], start=[0, 2, 4, 6, 8, 10], end=[1, 3, 5, 7, 9, 11]
    )  # 两个资产交错；最早的资产0窗口必须被淘汰
    stats.add_batch(columns)  # 资产0仅保留2、4、6；资产1保留10、20
    before = stats.state_dict()  # 保存与输入数组无关的标量快照
    for array in columns.values():  # recorder后续可能原地复用CPU缓冲区
        array[...] = 0  # 用破坏性修改证伪隐藏数组引用
    assert stats.state_dict() == before, "统计器持有了调用方数组的别名"  # 不持有tensor/array
    rows = stats.per_asset()  # 各资产独立队列，不共享全局容量
    assert rows[0]["first30_net_median"] == 4.0 and rows[1]["first30_net_median"] == 15.0, rows
    assert (stats.summary()["first30_policy_start_min"], stats.summary()["first30_policy_end_max"]) == (2, 11), rows
    stats.add_batch(_columns([1], [30.0]))  # 另一资产的新窗口不淘汰资产0
    assert stats.per_asset()[0] == rows[0], stats.per_asset()  # 独立FIFO边界


@pytest.mark.parametrize(
    "column,value",
    [
        ("asset_index", np.array([0.0, 1.0])),  # float索引不可截断
        ("asset_index", np.array([False, True])),  # bool不能冒充整数
        ("asset_index", np.array([0, 2])),  # 第二行越界，第一行也不能提交
        ("asset_index", np.array([0, -1])),  # 负索引不能访问末资产
        ("net_turns", np.array([1, 2])),  # 物理浮点列必须保持声明dtype
        ("net_turns", np.array([1.0, np.nan])),  # 非有限净转
        ("net_turns", np.array([1.0, np.inf])),  # 无穷净转
        ("net_turns", np.array([1.0, -np.inf])),  # 负无穷同样非法
        ("net_turns", np.array([1.0, 2.0], dtype=object)),  # object列不能绕过数值合同
        ("net_turns", np.array([1.0, 2.0], dtype=np.complex128)),  # 复数不属于实角位移
        ("net_turns", np.ma.array([1.0, 2.0], mask=[False, True])),  # 掩码缺测不能成为真实窗口
        ("absolute_path_turns", np.array([1.0, -2.0])),  # 绝对路径必须非负
        ("absolute_path_turns", np.array([1.0, 1.9])),  # 明显小于净转绝对值
        ("absolute_path_turns", np.array([1.0, 0.0])),  # 非零净转不允许零路径
        ("absolute_path_turns", np.array([1.0, np.inf])),  # 路径缺测不能化成方向0
        ("absolute_path_turns", np.array([1.0, np.nan])),  # 非有限路径
        ("goal_count", np.array([0.0, 0.0])),  # goal计数必须整数
        ("goal_count", np.array([False, True])),  # bool计数应被拒绝
        ("goal_count", np.array([0, -1])),  # 负计数非法
        ("safe", np.array([1, 0])),  # 安全事实必须bool
        ("policy_version_start", np.array([0, -1])),  # 版本非负
        ("policy_version_start", np.array([0.0, 0.0])),  # 版本不能隐式截断
        ("policy_version_end", np.array([1, -1])),  # end不可早于start
        ("policy_version_end", np.array([False, True])),  # bool版本非法
        ("safe", np.array([True])),  # 所有必需列均须[E]对齐
        ("safe", np.array([[True, True]])),  # 二维列不隐式flatten
        ("asset_index", [0, 1]),  # 非ndarray输入不隐式转为CPU事实
    ],
)
def test_invalid_batch_is_rejected_atomically(column, value) -> None:
    r"""错误放在后续行/列，验证拒绝批次后旧窗口、FIFO淘汰和版本范围都不变。"""
    stats = FirstWindowStatistics(2, max_windows_per_asset=1)  # 有效首行一旦提交就会覆盖旧窗
    stats.add_batch(_columns([0, 1], [8.0, 9.0], start=[8, 9], end=[8, 9]))  # 初始已满的FIFO
    before = stats.state_dict()  # 完整状态比仅比较中位数更能检测部分写入
    columns = _columns([0, 1], [1.0, 2.0])  # 第一行原则上合法
    columns[column] = value  # 将单个合同破坏注入真实批形状
    with pytest.raises(ValueError):  # 所有错误统一在更新前暴露
        stats.add_batch(columns)
    assert stats.state_dict() == before, f"非法列{column}导致了部分提交或淘汰"  # 原子拒绝


def test_missing_columns_empty_batches_and_monotone_versions() -> None:
    r"""空批是合法无操作；缺键及非负但倒序的版本都不能形成窗口。"""
    stats = FirstWindowStatistics(1)  # 单资产无历史
    before = stats.state_dict()  # 空状态也必须保持完整配置
    stats.add_batch(_columns([], []))  # E=0，所有必需列仍有正确dtype和形状
    assert stats.state_dict() == before, "空批创建了伪窗口"  # 无观测不是零转窗口
    columns = _columns([0], [1.0], start=[2], end=[1])  # 两个版本非负但发生倒序
    with pytest.raises(ValueError):  # 必须比较start/end，而非只检查非负
        stats.add_batch(columns)
    del columns["safe"]  # 超集可接受，但必需键不可缺失
    with pytest.raises(ValueError):  # 缺失安全事实不默认True
        stats.add_batch(columns)
    assert stats.state_dict() == before, "非法版本或缺列写入了窗口"  # 保持原子性


def test_direction_roundoff_clip_is_not_a_physics_repair() -> None:
    r"""只把浮点舍入造成的微小越界clip到±1；零路径不允许用绝对容差掩盖非零净转。"""
    stats = FirstWindowStatistics(2)  # 分别验证正/负方向的上界
    one = np.nextafter(np.float32(1.0), np.float32(2.0))  # float32中大于1的相邻可表示数
    columns = _columns([0, 1], [float(one), -float(one)], path=[1.0, 1.0])  # 仅约1.2e-7相对偏差
    columns["net_turns"] = columns["net_turns"].astype(np.float32)  # 保留生产者常见精度
    stats.add_batch(columns)  # 舍入范围内方向可做有界修正
    assert [row["first30_direction_median"] for row in stats.per_asset().values()] == [1.0, -1.0], stats.per_asset()
    restored = FirstWindowStatistics(2)  # 状态转成Python浮点后仍使用相同容差
    restored.load_state_dict(stats.state_dict())  # 浮点容差的含义不能随序列化改变
    assert restored.summary() == stats.summary(), restored.summary()  # clip与恢复一致
    for net, path in [(1.001, 1.0), (1e-20, 0.0), (0.0, -1e-20)]:  # 明显越界或数学上不可能的路径
        with pytest.raises(ValueError):  # 不用无条件clip把非法事实修成合法
            stats.add_batch(_columns([0], [net], path=[path]))


def test_large_finite_values_and_unsigned_counters_do_not_overflow() -> None:
    r"""有限极值仍给出有限中位数；uint64策略版本与goal计数不能被截成负int64。"""
    stats = FirstWindowStatistics(2)  # 同时覆盖资产内与资产间偶数中位数
    largest = float(np.finfo(np.float64).max)  # 原始输入有限，中央两值求和会溢出
    columns = _columns([0, 0, 1, 1], [largest] * 4)  # 四个数学上合法的同值窗口
    counter = 2**64 - 1  # NumPy无符号64位可合法表达的整数
    for key in ("goal_count", "policy_version_start", "policy_version_end"):  # 保留计数和版本的整数事实
        columns[key] = np.full(4, counter, dtype=np.uint64)  # 检测有符号转换溢出
    stats.add_batch(columns)  # 不要求生产者预先缩窄整数dtype
    result = stats.summary()  # 中位数需要稳定计算，不能生成inf
    assert result["first30_net_median"] == largest, result  # 有限最大值保持有限
    assert result["first30_goal_median"] == float(counter), result  # 浮点报告按声明精度输出
    assert result["first30_policy_start_min"] == result["first30_policy_end_max"] == counter, result  # 版本精确


@pytest.mark.skipif(np.finfo(np.longdouble).maxexp <= np.finfo(np.float64).maxexp, reason="平台无更宽浮点指数范围")
@pytest.mark.parametrize("value", ["1e-4000", "1e4000"])
def test_wide_float_cannot_silently_change_physical_facts(value) -> None:
    r"""更宽浮点在原dtype下有限，但不能在纯Python状态中下溢为静止或上溢为无穷运动。"""
    stats = FirstWindowStatistics(1, max_windows_per_asset=1)  # 原队列已有窗口，检测非法批次是否提前淘汰
    stats.add_batch(_columns([0], [2.0]))  # 一条有限的既有事实
    before = stats.state_dict()  # 要求非法转换后精确保持原对象
    columns = _columns([0, 0], [1.0, 1.0])  # 第一行有效，第二行无法表达为Python float
    columns["net_turns"] = np.array(["1.0", value], dtype=np.longdouble)  # 避免构造时先经过binary64
    columns["absolute_path_turns"] = columns["net_turns"].copy()  # 在原精度下net==path完全合法
    assert np.isfinite(columns["net_turns"]).all(), value  # 失败应来自状态精度边界，而非源数据NaN/inf
    with pytest.raises(ValueError):  # 不能将第二行偷偷写成0/0或inf/inf
        stats.add_batch(columns)
    assert stats.state_dict() == before, "宽浮点不可表示值破坏了原子拒绝"  # 第一行也不能提交


@pytest.mark.parametrize(
    "configuration",
    [
        {"asset_count": -1},  # 资产宇宙大小不可为负
        {"asset_count": 1.0},  # 数值相等也不可截断浮点配置
        {"asset_count": True},  # bool不代表资产总数
        {"asset_count": 1, "max_windows_per_asset": 0},  # 零容量会吞掉全部观测
        {"asset_count": 1, "max_windows_per_asset": 33},  # 合同固定每资产最多32窗
        {"asset_count": 1, "max_windows_per_asset": False},  # bool容量非法
        {"asset_count": 1, "minimum_windows": 0},  # 缺测资产不能因零样本门成为qualified
        {"asset_count": 1, "minimum_windows": 16.0},  # 资格门同样是整数配置
    ],
)
def test_invalid_configuration_fails_explicitly(configuration) -> None:
    r"""容量和样本门的错误配置在构造时失败，不靠空deque或隐式转换产生歧义统计。"""
    with pytest.raises(ValueError):  # 所有配置错误都在建立可消费对象前暴露
        FirstWindowStatistics(**configuration)


def test_empty_asset_universe_is_a_valid_empty_group() -> None:
    r"""A=0无需伪造资产；空批、空组及状态恢复仍有完备的14字段合同。"""
    stats = FirstWindowStatistics(0)  # 空支持集是明确的集合边界
    stats.add_batch(_columns([], []))  # 空集合不会产生窗口
    result = stats.summary()  # 所有计数为0，所有未定义量为None
    assert set(result) == _FIELDS and all(value is None or value == 0 for value in result.values()), result
    assert stats.per_asset() == {} and result == stats.summary([]), result  # 两种空组表达一致
    stats.load_state_dict(stats.state_dict())  # 空历史恢复同样满足原始schema
    assert stats.summary() == result, "空状态恢复改变了统计"  # 无物理seen/RNG依赖


def test_state_round_trip_preserves_fifo_and_uses_only_python_values() -> None:
    r"""JSON往返保留配置、有序原始窗口及后续淘汰语义；状态快照与恢复输入都不形成别名。"""
    stats = FirstWindowStatistics(2, max_windows_per_asset=3, minimum_windows=2)  # 非默认配置须显式恢复匹配
    stats.add_batch(
        _columns([0, 1, 0, 0], [-1.0, 8.0, 2.0, 4.0], path=[2.0, 8.0, 2.0, 5.0], safe=[False, True, True, True])
    )  # 状态中同时包含负净转、非单位方向与不安全事实
    state = stats.state_dict()  # 顶层只有配置与每资产有序窗口
    assert set(state) == {"asset_count", "max_windows_per_asset", "minimum_windows", "windows"}, state  # 无环境seen位
    encoded = json.dumps(state, allow_nan=False)  # ndarray/tensor/NumPy整数或非有限值不能混入状态
    decoded = json.loads(encoded)  # JSON可无损保存本合同的标量/列表结构
    assert decoded == state, state  # 不用tuple或对象身份承载窗口顺序
    restored = FirstWindowStatistics(2, max_windows_per_asset=3, minimum_windows=2)  # 配置相同的独立对象
    restored.load_state_dict(decoded)  # 恢复原始证据，归约量由窗口重新计算
    assert restored.per_asset() == stats.per_asset(), restored.per_asset()  # 四个统计量和版本同时对齐
    decoded["windows"][0][0][0] = 1234.0  # 外部修改恢复输入不应改变内部窗口
    assert restored.state_dict() == state, "恢复后仍持有外部状态列表"  # 无可变别名
    state["windows"][0].clear()  # 修改导出快照不应清空原对象
    new = _columns([0, 1], [6.0, 10.0])  # 下一批触发同样的资产0淘汰
    stats.add_batch(new)  # 继续原序列
    restored.add_batch(new)  # 继续恢复序列
    assert restored.state_dict() == stats.state_dict(), "恢复改变了FIFO时间顺序"  # 验证后续行为而非仅快照


@pytest.mark.parametrize(
    "field,value",
    [
        ("asset_count", 3),  # 资产宇宙改变
        ("max_windows_per_asset", 4),  # 容量改变会改变历史截断
        ("minimum_windows", 3),  # 资格门改变
        ("minimum_windows", 2.0),  # 数值相等的float也不是合法整数配置
        ("asset_count", True),  # bool配置非法
        ("windows", []),  # 缺资产列表
        ("windows", [[], [[1.0, 1.0, 0, True, 0, 1]] * 4]),  # 超容量坏状态不能静默截断
        ("windows", [[], [[1.0, 1.0, 0, True, 0]]]),  # 窗口字段不完整
        ("windows", [[], [[float("nan"), 1.0, 0, True, 0, 1]]]),  # 恢复也拒绝非有限物理值
        ("windows", [[], [[1.0, 0.5, 0, True, 0, 1]]]),  # 路径小于净转
        ("windows", [[], [[1.0, 1.0, True, True, 0, 1]]]),  # bool不能变为goal计数1
        ("windows", [[], [[1.0, 1.0, 0, 1, 0, 1]]]),  # 数字1不能变为安全事实
        ("windows", [[], [[1.0, 1.0, -1, True, 0, 1]]]),  # 非负计数约束
        ("windows", [[], [[1.0, 1.0, 10**400, True, 0, 1]]]),  # 巨大整数不能形成有限浮点goal中位数
        ("windows", [[], [[1.0, 1.0, 0, True, 2, 1]]]),  # 版本倒序
        ("windows", [[], [[1.0, 1.0, 0, True, 0.0, 1]]]),  # 版本必须真整数
        ("windows", [[], [np.array([1.0] * 6)]]),  # 纯Python状态合同禁止数组窗口
    ],
)
def test_bad_restore_is_atomic_even_after_valid_earlier_assets(field, value) -> None:
    r"""坏记录放在最后资产，先前资产即便已经解析也不得提交，更不得静默淘汰超容量历史。"""
    stats = FirstWindowStatistics(2, max_windows_per_asset=3, minimum_windows=2)  # 恢复目标含既有观测
    stats.add_batch(_columns([0, 1], [7.0, 9.0]))  # 非空目标使部分清空也可检测
    before = stats.state_dict()  # 保留精确窗口序列
    bad = deepcopy(before)  # 独立坏状态，不通过别名提前改变对象
    bad[field] = value  # 单独破坏配置、形状或数值合同
    with pytest.raises(ValueError):  # 解析、校验通过之前不能替换任何资产历史
        stats.load_state_dict(bad)
    assert stats.state_dict() == before, f"坏恢复字段{field}改变了原对象"  # 恢复事务原子性


def test_state_schema_and_all_operations_leave_training_rng_unchanged() -> None:
    r"""窗口状态不接管物理环境seen位；记录、查询与恢复均不推进Python/NumPy训练随机流。"""
    python_before = random.getstate()  # 只读取当前随机流，不重设训练seed
    numpy_before = np.random.get_state(legacy=True)  # legacy全局NumPy随机流也是训练进程可用资源
    stats = FirstWindowStatistics(1)  # 构造也不采样
    stats.add_batch(_columns([0], [1.0]))  # 添加确定性事实
    stats.per_asset()  # 归约没有bootstrap或随机子采样
    stats.summary(np.array([0], dtype=np.int64))  # 接受一维NumPy整数选择器
    state = stats.state_dict()  # 状态只含窗口，不保存RNG或环境seen
    stats.load_state_dict(state)  # 合法恢复没有随机初始化
    for bad in ({**state, "seen": [True]}, {key: value for key, value in state.items() if key != "windows"}):
        with pytest.raises(ValueError):  # 未声明字段与缺字段都不能偷偷改变恢复合同
            stats.load_state_dict(bad)
    assert random.getstate() == python_before, "统计改变了Python训练随机流"  # 不消耗随机数
    numpy_after = np.random.get_state(legacy=True)  # 同样只读取状态
    np.testing.assert_equal(numpy_after, numpy_before, err_msg="NumPy随机状态改变")  # 逐层核对随机键、流位置与缓存
