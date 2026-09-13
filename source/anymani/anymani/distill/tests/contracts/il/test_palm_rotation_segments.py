r"""离线示范按完整30秒段收录；时钟、恢复过程与episode级隔离均可在CPU证伪。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from anymani.distill.il.palm_rotation_segments import select_demonstration_segments, split_demonstration_episodes


def _stream(steps: int = 1200, envs: int = 2) -> dict:
    r"""构造一位冻结教师的连续20Hz采集；默认每30秒净转.6圈。"""
    clock = np.repeat(np.arange(1, steps + 1, dtype=np.int64)[:, None], envs, axis=1)
    net = clock.astype(np.float64) * 0.6 / 600  # 有符号累计圈数，不是瞬时速度。
    return {
        "net_turns": net,  # 采集器每个物理结果的累计有符号圈数。
        "absolute_path_turns": net / 0.8,  # 逐段方向.8，与终点净圈共同判断。
        "goal_count": clock // 60,  # 目标计数仅作为记录，不替代真实净转。
        "episode_id": np.zeros_like(clock),  # 首个回合，后续测试显式模拟终止与reset。
        "episode_step": clock,  # 从1开始计数的已完成动作数。
        "physical_failure": np.zeros_like(clock, dtype=bool),  # 默认全程无drop/axis。
        "timeout": np.zeros_like(clock, dtype=bool),  # 默认尚未到回合时域终点。
        "asset_index_by_env": np.arange(envs, dtype=np.int64),  # 静态资产来源，不作模型输入。
        "stochastic_by_env": np.arange(envs) % 2 == 1,  # 两种采集行为都有测试覆盖。
    }


def test_window_uses_differences_and_keeps_every_frame_of_stall_recovery() -> None:
    r"""第二段使用累计量之差；段中暂时静止的100帧仍包含在接受区间。"""
    stream = _stream()  # 两资产，各有两个完整30秒段。
    net = stream["net_turns"]  # 按时间逐步积累，不能与窗口增量混淆。
    net[:200, 0] = np.linspace(0.001, 0.2, 200)  # 前10秒逐步推转。
    net[200:300, 0] = 0.2  # 教师暂时卡住，后续恢复而不是把这100帧裁掉。
    net[300:600, 0] = np.linspace(0.2, 0.6, 300)  # 后15秒恢复并达到有效数据门。
    stream["absolute_path_turns"][:, 0] = net[:, 0] / 0.8  # 保持方向性约.8。
    segments = select_demonstration_segments(**stream)  # 输出完整区间，不只输出运动帧。
    assert len(segments) == 4  # 2资产×2段；不产生重叠滑窗扩大分母。
    assert all(segment.accepted for segment in segments)  # 完整安全、.6圈和.8方向均合格。
    first = next(segment for segment in segments if segment.env_id == 0 and segment.start_frame == 0)  # 首段。
    second = next(segment for segment in segments if segment.env_id == 0 and segment.start_frame == 600)  # 后段。
    assert first.stop_frame == 600 and first.net_turns == pytest.approx(0.6)  # 闭合600动作观察机会。
    assert second.net_turns == pytest.approx(0.6)  # 不能把累计1.2圈当作后30秒的增量。
    assert first.goal_count == second.goal_count == 10  # 目标累计量也必须逐段取差。
    assert first.start_frame <= 200 < 300 <= first.stop_frame  # 停滞帧在同一返回区间内部。


def test_late_recovery_can_be_accepted_without_asset_level_reliable_gate() -> None:
    r"""弱资产的一个有效段也收录；此前低于.5圈的段独立保留拒绝原因。"""
    stream = _stream(envs=1)  # 仅一个尚未通过正式可靠门的资产。
    stream["net_turns"][:600] *= 0.5  # 前30秒只有.3圈。
    stream["net_turns"][600:] = 0.3 + np.arange(1, 601)[:, None] * 0.7 / 600  # 后段真实增加.7圈。
    stream["absolute_path_turns"] = stream["net_turns"] / 0.8  # 两段方向均合格。
    first, second = select_demonstration_segments(**stream)  # 不读取任何资产级可靠布尔值。
    assert not first.accepted and "net-below-0p5" in first.rejection_reasons  # 低转动段不被高转动段掩盖。
    assert second.accepted and second.net_turns == pytest.approx(0.7)  # 弱资产的有效经验保留。


def test_physical_failure_at_window_end_rejects_but_complete_timeout_is_safe() -> None:
    r"""第600步掉落与纯timeout必须区分；.5圈不能抵消物理失败。"""
    stream = _stream(steps=600)  # 精确到第30秒边界。
    stream["physical_failure"][-1, 0] = True  # 边界当步掉落仍属于本段结果。
    stream["timeout"][-1] = True  # 第一副本两旗同时，第二副本仅完整时域结束。
    failed, timed_out = select_demonstration_segments(**stream)  # 相同净圈，不同物理安全事实。
    assert not failed.accepted and not failed.safe  # 不能把边界失败划给下一个窗口。
    assert "physical-failure" in failed.rejection_reasons  # 失败原因保持可审计。
    assert timed_out.accepted and timed_out.safe  # 完整30秒timeout没有额外drop/axis。


def test_short_failed_episode_is_not_extrapolated_or_joined_to_next_episode() -> None:
    r"""300步短失败不外推到30秒，也不与reset后的前300步拼成一个成功段。"""
    stream = _stream(steps=900, envs=1)  # 45秒bank内包含一个15秒失败回合与一个30秒回合。
    stream["physical_failure"][299, 0] = True  # 第300步结束旧episode。
    stream["episode_id"][300:] = 1  # 下一动作属于新的物理回合。
    stream["episode_step"][300:, 0] = np.arange(1, 601)  # 新回合时钟从1开始。
    stream["net_turns"][300:, 0] = np.arange(1, 601) * 0.6 / 600  # 累计净圈在reset后重新起算。
    stream["absolute_path_turns"] = stream["net_turns"] / 0.8  # 保持联合轨迹量对应。
    stream["goal_count"][300:, 0] = np.arange(1, 601) // 60  # 目标计数同样重置。
    segments = select_demonstration_segments(**stream)  # 不能跨reset拼接成600帧。
    assert len(segments) == 1  # 15秒旧回合不贡献一个完整30秒样本。
    assert segments[0].episode_id == 1 and segments[0].start_frame == 300  # 新段索引精确对应bank。
    assert segments[0].stop_frame == 900 and segments[0].accepted  # 接受整条新回合。


@pytest.mark.parametrize("fault", ["clock-gap", "reset-without-done", "float-asset", "nan", "reverse-path"])
def test_invalid_recording_is_rejected_instead_of_silently_repaired(fault: str) -> None:
    r"""时间或物理事实失配必须暴露，不能通过筛掉坏行伪装为有效数据集。"""
    stream = _stream()  # 先建立完整合法数据，每种fault只破坏一个合同。
    if fault == "clock-gap":
        stream["episode_step"][200, 0] += 1  # 模拟漏记一步而数组长度未变。
    elif fault == "reset-without-done":
        stream["episode_id"][600:, 0] = 1  # 没有对应terminal的隐式换回合。
    elif fault == "float-asset":
        stream["asset_index_by_env"] = np.array([0.0, 1.0])  # 索引即使取整数值也不能依赖浮点截断。
    elif fault == "nan":
        stream["net_turns"][1, 0] = np.nan  # 非有限物理事实不能按零运动处理。
    else:
        stream["absolute_path_turns"][200, 0] = 0  # 正净转对应零路径，违反物理自洽。
    with pytest.raises(ValueError):
        select_demonstration_segments(**stream)  # 坏bank整体拒绝，不部分生成训练数据。


def test_episode_split_keeps_all_windows_together_and_preserves_global_rng() -> None:
    r"""同回合两个窗口不可跨train/val；本地分组随机流不改变全局NumPy状态。"""
    stream = _stream(envs=16)  # 一个资产的多条独立采集episode。
    stream["asset_index_by_env"][:] = 0  # 一资产的16个独立采集副本。
    stream["stochastic_by_env"][:] = np.arange(16) >= 8  # 两种教师行为各8副本。
    segments = select_demonstration_segments(**stream)  # 每副本两段，共32个合格段。
    before: Any = np.random.get_state()  # legacy状态元组；检查split无训练随机流副作用。
    split = split_demonstration_episodes(segments, seed=42)  # 在episode而非时间步层随机分配。
    after: Any = np.random.get_state()  # 显式标量/数组逐项核对，不重新采样全局RNG。
    assert np.array_equal(before[1], after[1]) and before[2:] == after[2:]  # 全局采样状态不变。
    assert len(split.training_indices) == 24 and len(split.validation_indices) == 8  # 完整episode维持3:1。
    train_keys = {(segments[i].env_id, segments[i].episode_id) for i in split.training_indices}  # 训练来源集合。
    val_keys = {(segments[i].env_id, segments[i].episode_id) for i in split.validation_indices}  # 独立验证来源。
    assert train_keys.isdisjoint(val_keys)  # 相邻30秒段没有跨集合泄漏。
    assert {segments[i].stochastic for i in split.validation_indices} == {False, True}  # 验证覆盖两种采集行为。
    assert split == split_demonstration_episodes(segments, seed=42)  # 同seed严格复现相同划分。


def test_single_available_episode_stays_in_training_with_honest_validation_gap() -> None:
    r"""稀少但合格的弱资产数据保留给训练，不拆同一回合制造验证集。"""
    segments = select_demonstration_segments(**_stream(envs=1))  # 两个段同属唯一episode。
    split = split_demonstration_episodes(segments)  # 不以段数2冒充独立episode数2。
    assert split.training_indices == (0, 1) and split.validation_indices == ()  # 保留有效监督，不伪造验证。
    assert split.assets_without_validation == (0,)  # 稀少资产的证据缺口可下游读取。
