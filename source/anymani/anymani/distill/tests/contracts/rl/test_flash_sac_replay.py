r"""紧凑回放的 CPU 数值合同：绝对时间、回合边界、均衡抽样和物理终点。

所有张量保持真实的 16-JOINT / 21-owner ABI；只缩小环境数和时间容量。
参考轨迹直接保存完整历史，作为独立于环形索引实现的逐帧审计真值。
运行时使用 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1，不启动模拟器。
"""

from __future__ import annotations

import copy
import io
import math
from typing import Any

import pytest
import torch
from anymani.distill.rl.flash_sac.replay import DYNAMIC_SHAPES, CompactReplay

# 非连续且交错排列的标签同时审计“资产身份”和“环境副本”两条轴。
ASSET_IDS = torch.tensor([9, 2, 9, 2], dtype=torch.long)  # N=4，A=2，每资产两个副本。
EXPECTED_SHAPES = {  # 这是任务规定的独立 ABI，不能仅从被测常量生成期望。
    "actor_jnt_current": (16, 5),  # 含当前关节状态与触觉的五通道帧。
    "actor_owner_contact": (21, 1),  # Actor 的二元接触，不含接触强度。
    "critic_jnt_state": (16, 4),  # Critic 的四通道关节状态。
    "critic_owner_contact": (21, 2),  # Critic 的特权接触信息保持 FP32。
    "critic_obj": (1, 15),  # 物体状态。
    "critic_task": (1, 8),  # 任务状态。
    "critic_reward_release": (1,),  # 奖励释放状态。
}


def _replay(**kwargs) -> CompactReplay:
    r"""构造 N=4、A=2、L=3、n=3 的小回放，可逐项覆盖边界参数。"""
    config: dict[str, Any] = dict(  # 异质配置且测试包含故意非法值，40 transitions 即十个 vector-step。
        capacity_transitions=40, num_envs=4, asset_ids_by_env=ASSET_IDS, history_steps=3, n_step=3, gamma=0.5
    )
    config.update(kwargs)  # 边界测试只修改明确指定的配置。
    return CompactReplay(**config)  # 默认 CPU、非锁页内存。


def _observation(time: int) -> dict[str, torch.Tensor]:
    r"""用环境、绝对时间、字段、通道的唯一 FP32 标签生成紧凑状态。"""
    result = {}  # 每个字段都有完整 ABI shape，无 history 或静态几何。
    for field, (name, shape) in enumerate(EXPECTED_SHAPES.items()):
        # /128 是精确二进制小数，可辨认通道置换而不引入表示误差。
        features = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape) / 128
        envs = torch.arange(4, dtype=torch.float32).reshape(4, *([1] * len(shape)))  # 广播环境轴。
        result[name] = envs * 10000 + time * 10 + field * 100 + features  # 每个实数位置独立可辨。
    # 二元接触使用独立奇偶模式，所有其他字段仍保持一般实数而不是常数零。
    owners = torch.arange(21).reshape(1, 21, 1)  # owner 槽位轴。
    envs = torch.arange(4).reshape(4, 1, 1)  # 环境轴。
    result["actor_owner_contact"] = ((owners + envs + time) % 2).float()  # 合法的 0/1 接触。
    return result  # 所有返回 tensor 都是 CPU FP32。


def _packet(time: int, terminated=(), truncated=()) -> dict:
    r"""生成一步输入；结束环境的 pre-reset 终点与下回合初态明显不同。"""
    term = torch.zeros(4, dtype=torch.bool)  # 真终止向量。
    trunc = torch.zeros(4, dtype=torch.bool)  # 有限时域 timeout 向量。
    term[list(terminated)] = True  # 可同时在若干环境发生终止。
    trunc[list(truncated)] = True  # 两种结束标志可在同一环境同时成立。
    current = _observation(time)  # append 入口的当前状态。
    next_observation = _observation(time + 1)  # 非终止时与下一次 current 完全一致。
    for name, value in next_observation.items():
        # 终止后下一次 append 的 current 没有此偏移，直接暴露 reset 污染。
        if name == "actor_owner_contact":
            value[term | trunc] = 1 - value[term | trunc]  # 接触也区分物理终点和 reset。
        else:
            value[term | trunc] += 100000  # FP32 精确表示的终点标签。
    # 动作和原始奖励的时间标签允许检查 n-step 是否读错行或跨回合累加。
    actions = torch.arange(64, dtype=torch.float32).reshape(4, 16) + time * 100  # [N,16]。
    rewards = torch.arange(4, dtype=torch.float32) + time * 10 + 1  # [N]，不做归一化。
    return dict(  # 关键字与正式 append 接口一致。
        current=current,  # 当前动态帧。
        actions=actions,  # [N,16] 起点动作。
        rewards=rewards,  # [N] 原始奖励。
        terminated=term,  # [N] bool 真终止。
        truncated=trunc,  # [N] bool 有限时域 timeout。
        next_observation=next_observation,  # reset 之前的明确终点。
    )


def _trajectory(replay: CompactReplay, count: int, *, reset_streams: bool = False) -> list[dict]:
    r"""保存完整轨迹参考；物理 done 与显式流重启分别驱动历史的逐帧 reset。"""
    records = []  # 仅测试的小容量参考，不属于 replay 的持久内存。
    history = None  # 第一次 append 对应 CircularBuffer 首帧填满。
    previous_done = torch.ones(4, dtype=torch.bool)  # 四个环境均从新回合开始。
    for time in range(count):
        # 四副本以不同周期终止，保证两次以上覆盖后仍有多种回合边界。
        terminated = tuple(env for env in (0, 2) if time % (env + 4) == env + 1)
        truncated = tuple(env for env in (1, 3) if time % (env + 4) == env + 1)
        # 流重启与物理结束使用独立日程，覆盖连续重启以及环槽 9→0 的边界。
        stream_reset = torch.zeros(4, dtype=torch.bool)  # 本行前是否重建了物理环境。
        if reset_streams and time in (3, 9, 10, 20, 29, 31):
            indices = torch.tensor([time % 4, (time + 1) % 4])  # 两个副本的新历史段。
            replay.reset_streams(indices)  # 只改变下一次 append 的历史起点。
            stream_reset[indices] = True  # oracle 独立记录断点，不能从 replay 读取标签。
            previous_done |= stream_reset  # 历史 reset 既可由物理 done，也可由流重启触发。
        packet = _packet(time, terminated, truncated)  # current/next 保留精确终点差异。
        frame = packet["current"]["actor_jnt_current"]  # [N,16,5]。
        if history is None:
            history = frame[:, None].repeat(1, replay.history_steps, 1, 1)  # reset 首帧重复 L 次。
        else:
            history = torch.cat((history[:, 1:], frame[:, None]), dim=1)  # oldest-to-latest 左移。
            history[previous_done] = frame[previous_done, None]  # 只重置刚刚结束的副本。
        # 独立参考先用物理终点形成 next_history，再在下一步应用 reset。
        next_frame = packet["next_observation"]["actor_jnt_current"]  # pre-reset 终点。
        next_history = torch.cat((history[:, 1:], next_frame[:, None]), dim=1)  # [N,L,16,5]。
        records.append(  # 独立参考同时保留流边界，但不把它写成 packet 中的物理结束标志。
            dict(packet=packet, history=history.clone(), next_history=next_history, stream_reset=stream_reset)
        )
        replay.append(**packet)  # 被测实现只看到紧凑帧，不能使用参考 history。
        previous_done = packet["terminated"] | packet["truncated"]  # 下一次 append 的 reset 标志。
    return records  # 时间轴未覆盖，供标量 oracle 审计环形实现。


def _assert_tree_equal(actual, expected) -> None:
    r"""递归严格比较 checkpoint 与 sample，浮点逐位相同而非近似相同。"""
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), "checkpoint/sample 字段集合不一致"
        for key in expected:
            _assert_tree_equal(actual[key], expected[key])  # 递归到真实 tensor 内容。
    elif isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)  # 包含 dtype 与 shape 检查。
    else:
        assert actual == expected, f"metadata 不一致：{actual!r} != {expected!r}"


@pytest.mark.parametrize("length", [3, 10, 11, 23, 34])
@pytest.mark.parametrize("reset_streams", [False, True])
def test_gather_matches_full_history_oracle_through_multiple_wraps(length: int, reset_streams: bool) -> None:
    r"""独立 oracle 核对全部合法起点，覆盖物理/流分段与超过三圈的 ring 写入。"""
    replay = _replay()  # T=10，L=3，n=3。
    records = _trajectory(replay, length, reset_streams=reset_streams)  # 至少三步才有成熟起点。
    oldest = max(0, length - 10)  # 真实保留的最早绝对时间。
    lower = oldest + 2 if oldest else 0  # 覆盖后保守舍弃 L-1 个起点。
    pairs = [(env, time) for time in range(lower, length - 2) for env in range(4)]  # 全部合法组合。
    envs, times = torch.tensor(pairs, dtype=torch.long).T  # [B]、[B]。
    batch = replay.gather(envs, times)  # 一次调用同时审计全部环境/时间组合。
    assert replay.total_transitions == length * 4, "计数必须包含所有预热与被覆盖的 transition"
    assert set(batch) == {  # 索引是诊断 metadata，不混入 Actor 观察。
        "obs",  # current 及完整历史。
        "next_obs",  # 物理终点及其历史。
        "actions",  # 起点动作。
        "rewards",  # n-step 原始折扣回报。
        "discounts",  # bootstrap 权重。
        "terminated",  # 终点真终止。
        "truncated",  # 终点 timeout。
        "steps",  # 实际 target 长度。
        "asset_index",  # 稠密资产路由。
        "environment_index",  # 副本身份。
        "sequence_index",  # 绝对起点。
        "next_sequence_index",  # 物理终点的绝对状态时间。
    }
    for row, (env, start) in enumerate(pairs):
        # 标量 oracle 逐转移累加；物理结束停止 bootstrap，流边界保留 bootstrap。
        reward, discount, endpoint = 0.0, 1.0, start  # 累积量与绝对终点。
        for endpoint in range(start, start + 3):
            packet = records[endpoint]["packet"]  # 当前候选 transition。
            reward += discount * float(packet["rewards"][env])  # 原始奖励折扣和。
            discount *= 0.5  # 每经历一个 transition 乘一次 gamma。
            if packet["terminated"][env] or packet["truncated"][env]:
                discount = 0.0  # 两种有限时域结束都禁止 bootstrap。
                break  # 不读取新回合奖励。
            if endpoint + 1 < length and records[endpoint + 1]["stream_reset"][env]:
                break  # 流重启只有数据断点含义：停在旧转移 next，保留 gamma^实际步数。
        first, last = records[start]["packet"], records[endpoint]["packet"]  # 首状态与真实终点。
        for output, expected in (("obs", first["current"]), ("next_obs", last["next_observation"])):
            assert set(batch[output]) == set(EXPECTED_SHAPES) | {"actor_jnt_history"}
            for name in EXPECTED_SHAPES:
                torch.testing.assert_close(batch[output][name][row], expected[name][env], atol=0, rtol=0)
        # 完整历史参考同时检测帧顺序、episode 首帧补齐、环境隔离和终点 reset 污染。
        torch.testing.assert_close(batch["obs"]["actor_jnt_history"][row], records[start]["history"][env])
        torch.testing.assert_close(batch["next_obs"]["actor_jnt_history"][row], records[endpoint]["next_history"][env])
        torch.testing.assert_close(batch["actions"][row], first["actions"][env])  # 动作来自起点。
        assert batch["rewards"][row].item() == reward, f"原始 n-step 回报错误：env={env}, start={start}"
        assert batch["discounts"][row].item() == discount, f"bootstrap 权重错误：env={env}, start={start}"
        assert batch["steps"][row].item() == endpoint - start + 1, "实际长度应截止首个 done 或流边界"
        assert batch["terminated"][row] == last["terminated"][env], "必须返回终点 terminated"
        assert batch["truncated"][row] == last["truncated"][env], "必须返回终点 truncated"
        assert batch["asset_index"][row].item() == (1 if env % 2 == 0 else 0), "资产稠密索引应按 ID 排序"
        assert batch["environment_index"][row].item() == env, "环境副本不能串线"
        assert batch["sequence_index"][row].item() == start, "起点索引必须保持绝对时间"
        assert batch["next_sequence_index"][row].item() == endpoint + 1, "next 索引指终点状态，不是 ring 槽位"


@pytest.mark.parametrize("end_step", [0, 1, 2])
@pytest.mark.parametrize("flags", ["terminated", "truncated", "both", "neither"])
def test_n_step_discount_and_finite_horizon_terminal(end_step: int, flags: str) -> None:
    r"""显式验证 gamma=.99、一步/两步/三步结束及两种结束同时成立。"""
    replay = _replay(gamma=0.99)  # 正式默认折扣，不限于二进制精确的 0.5。
    for time in range(3):
        # 结束之后的奖励仍很大，若 mask 漏掉 reset 行就会明显改变期望值。
        term = (0,) if time == end_step and flags in ("terminated", "both") else ()
        trunc = (0,) if time == end_step and flags in ("truncated", "both") else ()
        replay.append(**_packet(time, term, trunc))  # 首个起点恰好在三步后成熟。
    batch = replay.gather(torch.tensor([0]), torch.tensor([0]))  # 固定环境和起点。
    steps = 3 if flags == "neither" else end_step + 1  # 有结束则截断，否则固定 n=3。
    expected_reward = sum(0.99**k * (10 * k + 1) for k in range(steps))  # 独立标量公式。
    assert batch["rewards"].item() == pytest.approx(expected_reward, rel=2e-7), "FP32 折扣和错误"
    assert batch["discounts"].item() == pytest.approx(0.99**3 if flags == "neither" else 0.0)
    assert batch["steps"].item() == steps, "n-step 长度错误"
    assert batch["terminated"].item() == (flags in ("terminated", "both")), "terminated 来源错误"
    assert batch["truncated"].item() == (flags in ("truncated", "both")), "timeout 来源错误"


def test_capacity_maturity_and_conservative_history_boundary() -> None:
    r"""容量向下取整，完整未来成熟前拒绝采样；覆盖后不虚构旧历史。"""
    replay = _replay(capacity_transitions=43)  # floor(43/4)*4=40。
    assert replay.actual_capacity == 40, "实际容量必须是完整 vector-step 的倍数"
    assert replay.total_transitions == 0 and not replay.ready, "空回放不可采样"
    for time in range(2):
        replay.append(**_packet(time, terminated=(0, 1, 2, 3)))  # 即使已 done 也保守等待完整 n 行。
        assert not replay.ready, "尚未具备 n-step 未来"
        with pytest.raises(ValueError, match="mature"):
            replay.sample(4)  # 预热期间不能抽到未写入槽位。
        with pytest.raises(ValueError, match="mature"):
            replay.gather(torch.tensor([0]), torch.tensor([0]))  # 确定性接口使用同一成熟条件。
    for time in range(2, 11):
        replay.append(**_packet(time))  # T=10，写到绝对时间 10 后 oldest=1。
    assert replay.ready, "覆盖后仍有上下文和未来完整的起点"
    for time in (0, 1, 2, 9, 10, 11, -1):
        with pytest.raises(ValueError, match="mature"):
            replay.gather(torch.tensor([0]), torch.tensor([time]))  # 合法区间仅为 [3,8]。
    # 小到容不下 L+n-1 的池仍按请求分配，但覆盖后 ready 必须反映无合法窗口。
    tiny = _replay(capacity_transitions=16)  # T=4 < L+n-1=5。
    _trajectory(tiny, 5)  # 覆盖之后无法同时拥有三帧历史和三步未来。
    assert not tiny.ready, "不足以支持稳态历史的容量不能伪称 ready"


def test_sample_is_exactly_asset_balanced_and_reproducible() -> None:
    r"""每个 minibatch 严格资产等额，副本与绝对起点均在合法支持集中抽取。"""
    replay = _replay()  # 非连续资产 ID，交错副本布局。
    _trajectory(replay, 34)  # 三圈覆盖后合法起点 [26,31]。
    generator = torch.Generator().manual_seed(431)  # CPU 随机源由调用者持有。
    rng = generator.get_state()  # 保存外部随机状态供确定性审计。
    batch = replay.sample(120, generator=generator)  # 每资产正好 60 个样本，可有放回。
    assert torch.bincount(batch["asset_index"]).tolist() == [60, 60], "每资产样本数必须完全相同"
    assert set(batch["environment_index"].tolist()) == {0, 1, 2, 3}, "固定随机种子应覆盖全部副本"
    assert set(batch["sequence_index"].tolist()) == set(range(26, 32)), "抽样支持必须是合法绝对时间"
    _assert_tree_equal(batch, replay.gather(batch["environment_index"], batch["sequence_index"]))
    generator.set_state(rng)  # Replay 本身不保存、重置或消费其他隐藏 RNG。
    _assert_tree_equal(batch, replay.sample(120, generator=generator))
    for size in (0, -2, 3):
        with pytest.raises(ValueError, match="batch_size"):
            replay.sample(size)  # 空 batch 或不整除资产数均不能静默修正。


def test_checkpoint_round_trip_restores_cursor_episode_and_external_rng() -> None:
    r"""真实序列化后恢复环布局；随后 append 仍延续正确的回合起点。"""
    replay = _replay()  # 内容经过多次覆盖，不能只恢复前缀或相对 cursor。
    _trajectory(replay, 34)  # 最新时刻包含各环境不同的 episode_start。
    replay.append(**_packet(34, terminated=(0,), truncated=(3,)))  # checkpoint 恰在两个环境结束之后。
    stream = io.BytesIO()  # 内存序列化，不产生测试文件或修改其他拥有路径。
    torch.save(replay.state_dict(), stream)  # state_dict 自身不负责 clone 整池。
    stream.seek(0)  # torch.load 读取保存时刻的独立内容。
    state = torch.load(stream, map_location="cpu", weights_only=True)  # 仅 CPU 恢复。
    restored = _replay()  # 构造严格匹配布局。
    restored.load_state_dict(state)  # load 必须 copy 到自己的数组。
    _assert_tree_equal(restored.state_dict(), replay.state_dict())
    # 相同外部 Generator state 必须给出相同抽样，且不依赖全局 torch 随机源。
    rng = torch.Generator().manual_seed(81).get_state()  # 调用者管理的唯一采样随机状态。
    expected = replay.sample(16, generator=torch.Generator().set_state(rng))  # 原池抽样。
    actual = restored.sample(16, generator=torch.Generator().set_state(rng))  # 恢复池抽样。
    _assert_tree_equal(actual, expected)
    for time in range(35, 38):
        replay.append(**_packet(time))  # 新回合首帧位于绝对 step=35。
        restored.append(**_packet(time))  # 恢复后保持完全相同的更新轨迹。
    _assert_tree_equal(restored.state_dict(), replay.state_dict())
    batch = restored.gather(torch.tensor([0, 3]), torch.tensor([35, 35]))  # 审计 checkpoint 后的 reset。
    expected_history = _observation(35)["actor_jnt_current"][[0, 3], None].repeat(1, 3, 1, 1)
    torch.testing.assert_close(batch["obs"]["actor_jnt_history"], expected_history)  # 新回合首帧重复。
    state["current"]["actor_jnt_current"].fill_(-123)  # 修改载入源，检验 load 后无整池别名。
    _assert_tree_equal(restored.state_dict(), replay.state_dict())


@pytest.mark.parametrize("case", ["config", "metadata", "cursor", "episode", "dtype", "shape", "extra"])
def test_checkpoint_rejects_incompatible_or_corrupt_layout_before_copy(case: str) -> None:
    r"""配置、路由或时间轴损坏必须整体拒绝，不能部分覆盖仍可用的回放。"""
    replay = _replay()  # 目的池已有真实数据。
    _trajectory(replay, 13)  # 已覆盖一次。
    before = copy.deepcopy(replay.state_dict())  # 仅在小容量测试中显式 clone。
    bad = copy.deepcopy(before)  # 构造单一明确破坏。
    if case == "config":
        bad["config"]["gamma"] = 0.99  # 同 shape 但折扣语义不同也不兼容。
    elif case == "metadata":
        bad["asset_ids_by_env"] = ASSET_IDS.flip(0)  # 资产到副本映射改变。
    elif case == "cursor":
        bad["write_index"] = 0  # 13 % 10 = 3，cursor 不可独立漂移。
    elif case == "episode":
        bad["episode_start"][2, 0] = 999  # 绝对时间 12 的回合起点不能在未来。
    elif case == "dtype":
        bad["next_observation"]["critic_obj"] = bad["next_observation"]["critic_obj"].half()  # 禁止精度漂移。
    elif case == "shape":
        bad["actions"] = bad["actions"][:, :, :15]  # 16-JOINT ABI 不可改变。
    else:
        bad["geometry_tokens"] = torch.zeros(1)  # checkpoint 也必须严格限制字段。
    with pytest.raises((ValueError, TypeError)):
        replay.load_state_dict(bad)  # 必须在任何持久 copy 之前完成验证。
    _assert_tree_equal(replay.state_dict(), before)


@pytest.mark.parametrize("location", ["current", "next_observation"])
@pytest.mark.parametrize("case", ["extra", "missing", "shape", "binary", "nan", "dtype"])
def test_append_rejects_invalid_observation_without_partial_write(location: str, case: str) -> None:
    r"""严格验证两端 compact ABI；非法终点同样不能推进 cursor 或 episode_start。"""
    replay = _replay()  # 保留已写入行以检测失败时覆盖副作用。
    replay.append(**_packet(0, terminated=(0,)))  # 下一 append 应开始新回合。
    before = copy.deepcopy(replay.state_dict())  # 小容量快照。
    packet = _packet(1)  # 有待检验的下一行。
    if case == "extra":
        packet[location]["actor_jnt_history"] = torch.zeros(4, 30, 16, 5)  # 明确禁止持久历史。
    elif case == "missing":
        del packet[location]["critic_task"]  # 不能静默用零填缺失字段。
    elif case == "shape":
        packet[location]["critic_reward_release"] = torch.zeros(4)  # [N] 与 [N,1] 不等价。
    elif case == "binary":
        packet[location]["actor_owner_contact"][0, 0, 0] = 0.25  # bool 转换会丢失此非法数值。
    elif case == "nan":
        packet[location]["critic_obj"][0, 0, 0] = float("nan")  # 污染数值不得进入持久池。
    else:
        packet[location]["critic_jnt_state"] = packet[location]["critic_jnt_state"].half()  # FP32 合同。
    with pytest.raises((ValueError, TypeError)):
        replay.append(**packet)  # 全字段预检查后才允许复制。
    _assert_tree_equal(replay.state_dict(), before)


@pytest.mark.parametrize("field", ["actions", "rewards", "terminated", "truncated"])
def test_append_rejects_nonvector_reward_action_or_nonbool_end_flags(field: str) -> None:
    r"""transition 辅助字段保持严格形状；done 不能用任意数值隐式转 bool。"""
    replay = _replay()  # 空池状态也应在验证失败后保持完整。
    packet = _packet(0)  # 合法基准输入。
    if field in ("terminated", "truncated"):
        packet[field] = packet[field].float()  # 结束标志明确要求 bool dtype。
    else:
        packet[field] = packet[field].unsqueeze(-1)  # 禁止广播掩盖形状错误。
    with pytest.raises((ValueError, TypeError)):
        replay.append(**packet)  # 无效 transition 不能计入预热。
    assert replay.total_transitions == 0, "非法输入推进了回放计数"


def test_append_owns_fp32_data_and_bool_contact_without_autograd_aliases() -> None:
    r"""写入采用复制和断开梯度；返回 contact 恢复 FP32 的网络输入 ABI。"""
    replay = _replay(n_step=1)  # 单步即可检查写入所有权。
    packet = _packet(0)  # 数值、通道和 owner 均有标签。
    packet["current"]["actor_owner_contact"] = packet["current"]["actor_owner_contact"].bool()  # bool 合法。
    packet["actions"].requires_grad_()  # 采集动作可能仍携带 actor 的梯度图。
    packet["next_observation"]["critic_obj"].requires_grad_()  # 终点也必须脱离外部图。
    replay.append(**packet)  # append 不得把外部 storage/计算图长期留在池中。
    before = replay.gather(torch.tensor([0]), torch.tensor([0]))  # 独立返回 batch。
    with torch.no_grad():
        for value in packet.values():
            values = value.values() if isinstance(value, dict) else (value,)  # 覆盖全部采集源。
            for tensor in values:
                tensor.zero_()  # 模拟 collector 重用临时 buffer。
    _assert_tree_equal(replay.gather(torch.tensor([0]), torch.tensor([0])), before)
    state = replay.state_dict()  # 持久 dtype 与返回 dtype 可以不同。
    for side in ("current", "next_observation"):
        assert state[side]["actor_owner_contact"].dtype == torch.bool, "binary contact 应精确紧凑保存"
        for value in state[side].values():
            assert not value.requires_grad and value.grad_fn is None, "回放不能持有计算图"
    assert before["obs"]["actor_owner_contact"].dtype == torch.float32, "Actor 输入应恢复 FP32 接触"
    assert not state["actions"].requires_grad, "动作存储不能保留 Actor 梯度"


def test_storage_bytes_are_exact_and_independent_of_history_or_static_tokens() -> None:
    r"""实际持久 tensor 总字节数只随 transition 数与资产 metadata 增长。"""
    short = _replay(history_steps=3)  # 相同池容量，短历史。
    long = _replay(history_steps=30)  # History30 不产生 [T,N,L,...] 数组。
    assert DYNAMIC_SHAPES == EXPECTED_SHAPES, "动态字段常量与任务 ABI 不一致"
    assert short.storage_bytes == long.storage_bytes, "history_steps 不得放大持久回放"
    state = short.state_dict()  # 所有持久数组均应有可恢复 checkpoint 表达。
    tensors = []  # 遍历 checkpoint，不依赖实现自己的 storage_bytes 计算。
    for value in state.values():
        if isinstance(value, torch.Tensor):
            tensors.append(value)  # 动作、奖励、结束标志、时间与路由 metadata。
        elif isinstance(value, dict):
            tensors.extend(item for item in value.values() if isinstance(item, torch.Tensor))  # 两端动态状态。
    storages = {value.untyped_storage().data_ptr(): value.untyped_storage().nbytes() for value in tensors}
    assert short.storage_bytes == sum(storages.values()), "storage_bytes 必须统计实际持久数组"
    # 每端 210 个 FP32 + 21 个 bool = 861 B；动作64、奖励4、done2、episode_start8。
    assert short.storage_bytes == 40 * 1800 + 4 * 32 + 2 * 8, "每 transition 应恰为 1800 B，另加路由 metadata"
    other_state = short.state_dict()  # state_dict 的两次调用必须引用同一池而不是各 clone 一份。
    assert state["current"]["actor_jnt_current"].data_ptr() == other_state["current"]["actor_jnt_current"].data_ptr()
    # 阻止 geometry、limits、静态图被误装进 compact packet；输入均为小 tensor。
    for name in ("geometry_tokens", "joint_limits", "static_graph"):
        packet = _packet(0)  # 保持其他字段正确，单独证伪额外 key。
        packet["current"][name] = torch.zeros(1)  # 大字段的语义由 key 决定，不按 numel 猜测。
        with pytest.raises(ValueError, match="fields"):
            short.append(**packet)  # 无论实际大小都必须拒绝额外字段。


@pytest.mark.parametrize(
    "kwargs",
    [
        {"capacity_transitions": 3},  # 不够一整行 N=4。
        {"num_envs": 0},  # 同步轴必须非空。
        {"history_steps": 0},  # 历史至少包含当前帧。
        {"n_step": 0},  # target 至少经历一步。
        {"gamma": -0.1},  # 折扣不允许负数。
        {"gamma": 1.1},  # 折扣不能放大未来奖励。
        {"gamma": float("nan")},  # 非有限配置不可形成可解释 target。
        {"asset_ids_by_env": torch.tensor([0, 0, 0, 1])},  # 副本数不相等。
        {"asset_ids_by_env": torch.tensor([0, 1])},  # 资产标签轴不匹配 N。
        {"asset_ids_by_env": ASSET_IDS.float()},  # 资产身份禁止浮点隐式截断。
    ],
)
def test_constructor_rejects_invalid_capacity_or_asset_layout(kwargs: dict) -> None:
    r"""非完整 vector-step、非法折扣、资产不等副本等配置必须在分配前拒绝。"""
    with pytest.raises((ValueError, TypeError)):
        _replay(**kwargs)  # 不自动补副本、不扩容量、不截断 metadata。


@pytest.mark.parametrize(
    "envs,times",
    [
        (torch.tensor([4]), torch.tensor([0])),  # 副本上界越界。
        (torch.tensor([-1]), torch.tensor([0])),  # 禁止 Python 负索引回绕。
        (torch.tensor([0.5]), torch.tensor([0])),  # 副本不能为浮点。
        (torch.tensor([0]), torch.tensor([0.5])),  # 时间不能静默截断为整数。
        (torch.tensor([[0]]), torch.tensor([0])),  # 禁止二维隐式广播。
        (torch.tensor([0, 1]), torch.tensor([0])),  # 两个索引必须同样本数。
        (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)),  # 空 batch 无训练语义。
    ],
)
def test_gather_rejects_invalid_audit_indices(envs: torch.Tensor, times: torch.Tensor) -> None:
    r"""确定性审计拒绝浮点截断、负索引、越界副本、空 batch 与广播。"""
    replay = _replay()  # 合法时间起点已成熟，因此只检验索引本身。
    _trajectory(replay, 3)  # 当前唯一合法绝对时间为零。
    with pytest.raises((ValueError, TypeError)):
        replay.gather(envs, times)  # 不能让 PyTorch 负索引或广播改变科研查询含义。


@pytest.mark.parametrize("gamma", [0.0, 1.0])
@pytest.mark.parametrize("n_step", [1, 3])
def test_single_frame_history_and_discount_endpoints(gamma: float, n_step: int) -> None:
    r"""L=1 的空左移片段仍追加真实 next；gamma=0/1 保持一步权重为一。"""
    replay = _replay(history_steps=1, n_step=n_step, gamma=gamma)  # 历史退化为当前帧。
    for time in range(23):
        replay.append(**_packet(time))  # 超过两次环覆盖，没有任何回合结束。
    start = 23 - n_step  # 最新成熟起点，终点 transition 固定为 22。
    batch = replay.gather(torch.tensor([0]), torch.tensor([start]))  # 一次确定性查询。
    current = _observation(start)["actor_jnt_current"][:1, None]  # L=1 的 current 真值。
    next_frame = _observation(23)["actor_jnt_current"][:1, None]  # 无需等第 23 次 append 即可读 next。
    torch.testing.assert_close(batch["obs"]["actor_jnt_history"], current)  # 不出现空历史。
    torch.testing.assert_close(batch["next_obs"]["actor_jnt_history"], next_frame)  # 只含显式终点帧。
    expected = sum(gamma**k * ((start + k) * 10 + 1) for k in range(n_step))  # gamma^0=1，包括 gamma=0。
    assert batch["rewards"].item() == expected, "折扣边界处的零次幂或奖励累计错误"
    assert batch["discounts"].item() == gamma**n_step, "无结束时 bootstrap 应为 gamma^n"


@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_checkpoint_can_resume_empty_or_warmup_history30(count: int) -> None:
    r"""空池与预热 checkpoint 均可恢复，初期 L>T 时仍重复真实 episode 首帧。"""
    replay = _replay(history_steps=30)  # L=30>T=10，仅检查未覆盖的初期补齐。
    for time in range(count):
        replay.append(**_packet(time, terminated=(0,) if time == 0 else ()))  # 环境零在第一步结束。
    restored = _replay(history_steps=30)  # 完全匹配的配置，不把 L30 误恢复成 L3。
    restored.load_state_dict(replay.state_dict())  # 源为共享引用，load 仍必须复制到自己的池。
    assert restored.ready == (count >= 3), "预热期 ready 状态恢复错误"
    _assert_tree_equal(restored.state_dict(), replay.state_dict())
    for time in range(count, 4):
        packet = _packet(time, terminated=(0,) if time == 0 else ())  # 延续同一采集轨迹。
        replay.append(**packet)  # 原池推进。
        restored.append(**packet)  # 恢复池推进。
    _assert_tree_equal(restored.state_dict(), replay.state_dict())
    batch = restored.gather(torch.tensor([0, 1]), torch.tensor([1, 1]))  # 同时查询 reset 和未 reset 的副本。
    frames = _observation(1)["actor_jnt_current"]  # absolute step 1 的 current。
    expected = frames[:2, None].repeat(1, 30, 1, 1)  # 环境零的新回合全部重复首帧。
    expected[1, :29] = _observation(0)["actor_jnt_current"][1]  # 环境一旧回合首帧补齐前 29 帧。
    torch.testing.assert_close(batch["obs"]["actor_jnt_history"], expected)  # L>T 也不使用负槽位旧数据。


@pytest.mark.parametrize("boundary", [6, 10, 30])
@pytest.mark.parametrize("subset", [False, True])
def test_reset_streams_after_load_preserves_old_tail_bootstrap_and_new_history(boundary: int, subset: bool) -> None:
    r"""保存→load→流重启→新初态：旧尾部可 bootstrap，新历史只属于重建后的环境。"""
    original = _replay(gamma=0.99)  # 原池作为未重启副本的连续流对照。
    for time in range(boundary):
        original.append(**_packet(time))  # 无物理 done，不能将人为恢复断点解释成有限时域结束。
    stream = io.BytesIO()  # 真正序列化，但只占测试进程内存。
    torch.save(original.state_dict(), stream)  # 旧物理环境结束运行时的 checkpoint。
    stream.seek(0)  # 恢复到新创建的回放对象。
    replay = _replay(gamma=0.99)  # 严格匹配布局与折扣配置。
    replay.load_state_dict(torch.load(stream, map_location="cpu", weights_only=True))  # 原经验全部保留。
    before = copy.deepcopy(replay.state_dict())  # 审计 reset_streams 本身的持久写入范围。
    selected = torch.tensor([0, 2]) if subset else torch.arange(4)  # 子集重启或默认全部重启。
    replay.reset_streams(selected if subset else None)  # 下一次 append 才写入新段首帧。
    expected_pending = copy.deepcopy(before)  # 所有已存经验和计数都应逐位相同。
    expected_pending["next_episode_start"][selected] = boundary  # 复用原有 [N] int64 保存待生效起点。
    _assert_tree_equal(replay.state_dict(), expected_pending)
    assert replay.total_transitions == boundary * 4 and replay.ready, "流重启不应改变采样预算或成熟窗口"
    assert replay.storage_bytes == 40 * 1800 + 4 * 32 + 2 * 8, "流重启不应增加持久数组"

    # 待重置状态必须在第一次新 append 之前即可序列化并严格恢复。
    pending_stream = io.BytesIO()  # 验证 pending 状态不是只存在 Python 调用栈中。
    torch.save(replay.state_dict(), pending_stream)  # 此时新段尚无任何 transition。
    pending_stream.seek(0)  # 载入一个不同的池实例。
    restored = _replay(gamma=0.99)  # 再次恢复应保留未消费的流边界。
    restored.load_state_dict(torch.load(pending_stream, map_location="cpu", weights_only=True))
    _assert_tree_equal(restored.state_dict(), replay.state_dict())
    for time in range(boundary, boundary + 4):
        packet = _packet(time)  # 未选中的副本继续原环境轨迹。
        recreated = _packet(time + 101)  # 新物理环境的初态、接触、奖励均有明显不同的标签。
        for name, value in packet.items():
            if isinstance(value, dict):
                for field, tensor in value.items():
                    tensor[selected] = recreated[name][field][selected]  # 只替换重建副本的动态观察。
            else:
                value[selected] = recreated[name][selected]  # 动作/奖励的跳变不能泄漏到旧 target。
        original.append(**packet)  # 连续流对照；未选中的副本与 restored 物理数据完全相同。
        replay.append(**packet)  # 未经第二次恢复的流边界，作为 pending 恢复一致性对照。
        restored.append(**packet)  # 待重置起点应在首 append 生效，并在后续 append 保持。
    _assert_tree_equal(restored.state_dict(), replay.state_dict())

    # 旧段末尾三个起点分别应累积 3、2、1 步；都落在相同的旧物理终点。
    for start in range(boundary - 3, boundary):
        batch = restored.gather(selected, torch.full_like(selected, start))  # 批量查询所有重建副本。
        steps = boundary - start  # 已限定在 1..3，最后一条旧 transition 为 boundary-1。
        rewards = [sum(0.99**k * (10 * (start + k) + env + 1) for k in range(steps)) for env in selected.tolist()]
        torch.testing.assert_close(batch["rewards"], torch.tensor(rewards), rtol=3e-7, atol=0)  # FP32 折扣和。
        discounts = torch.full_like(batch["discounts"], 0.99**steps)  # 流边界保留 gamma^实际步数 的 bootstrap。
        torch.testing.assert_close(batch["discounts"], discounts)  # 与 FP32 输出逐副本比较。
        assert batch["steps"].tolist() == [steps] * len(selected), "跨段之前的实际长度错误"
        assert not batch["terminated"].any() and not batch["truncated"].any(), "流断点不得伪造物理结束"
        assert batch["next_sequence_index"].tolist() == [boundary] * len(selected), "应返回旧转移 next 的概念时间"
        for name, value in _observation(boundary).items():
            torch.testing.assert_close(batch["next_obs"][name], value[selected])  # 显式 next，绝非重建后的新初态。
        frames = [_observation(time)["actor_jnt_current"][selected] for time in range(boundary - 2, boundary + 1)]
        torch.testing.assert_close(batch["next_obs"]["actor_jnt_history"], torch.stack(frames, dim=1))  # 旧段终点历史。

    # 新段首帧重复 L 次；第二帧也只可包含新段首帧与本帧。
    for offset in (0, 1):
        batch = restored.gather(selected, torch.full_like(selected, boundary + offset))  # 两起点都已成熟。
        first = _observation(boundary + 101)["actor_jnt_current"][selected]  # 新物理环境首帧。
        expected_history = first[:, None].repeat(1, 3, 1, 1)  # L=3，先全部补齐新段首帧。
        expected_history[:, -1] = _observation(boundary + offset + 101)["actor_jnt_current"][selected]
        torch.testing.assert_close(batch["obs"]["actor_jnt_history"], expected_history)  # 不引用旧段或反复 reset。
    if subset:
        envs, times = torch.tensor([(env, time) for env in (1, 3) for time in range(boundary - 3, boundary + 2)]).T
        _assert_tree_equal(restored.gather(envs, times), original.gather(envs, times))  # 其余副本逐字段不受影响。

    # 已实现的段边界也必须可再次恢复；同一 RNG 下，资产与时间采样支持保持完全相同。
    realized = _replay(gamma=0.99)  # 第三次恢复验证已写入主数组的 segment 起点。
    realized.load_state_dict(restored.state_dict())  # 不能把合法人工分段误判为损坏的 done 递推。
    batch = realized.sample(40, generator=torch.Generator().manual_seed(53))  # 每资产严格 20 行。
    baseline = original.sample(40, generator=torch.Generator().manual_seed(53))  # 同一同步成熟窗口。
    assert torch.bincount(batch["asset_index"]).tolist() == [20, 20], "流重启不应改变资产均衡"
    for key in ("asset_index", "environment_index", "sequence_index"):
        torch.testing.assert_close(batch[key], baseline[key])  # 没有因分段引入额外拒绝抽样或 RNG 消费。
    _assert_tree_equal(batch, restored.sample(40, generator=torch.Generator().manual_seed(53)))  # 恢复采样一致。


@pytest.mark.parametrize("flags", ["neither", "terminated", "truncated", "both"])
@pytest.mark.parametrize("offset", [0, 1])
def test_reset_streams_respects_real_done_before_or_at_boundary(flags: str, offset: int) -> None:
    r"""真实结束优先关闭 bootstrap；新段中的 done 不能污染旧段尚未结束的 target。"""
    replay = _replay()  # 旧段 0..3，重建边界在 4，查询起点固定为 2。
    for time in range(4):
        term = (0,) if time == 2 + offset and flags in ("terminated", "both") else ()  # 旧段真实终止。
        trunc = (0,) if time == 2 + offset and flags in ("truncated", "both") else ()  # 旧段真实 timeout。
        replay.append(**_packet(time, term, trunc))  # 首个结束与恢复断点可以分离或重合。
    replay.reset_streams()  # 所有环境下一行进入新历史段，不改写已保存的两种结束标志。
    replay.append(**_packet(100, terminated=(0,), truncated=(0,)))  # 新段奖励和 done 都不能进入旧 target。
    batch = replay.gather(torch.tensor([0]), torch.tensor([2]))  # n=3 的名义未来已完整写入。
    steps = 2 if flags == "neither" else offset + 1  # 无物理 done 时只在流边界前停止。
    expected_reward = sum(0.5**k * (21 + 10 * k) for k in range(steps))  # 旧段 raw reward。
    assert batch["rewards"].item() == expected_reward, "真实结束或流边界后的奖励进入了旧 target"
    assert batch["discounts"].item() == (0.5**steps if flags == "neither" else 0.0), "物理结束与恢复截断混淆"
    assert batch["steps"].item() == steps, "应以最早真实结束/流边界决定长度"
    assert batch["terminated"].item() == (flags in ("terminated", "both")), "只可读取旧段终点的 terminated"
    assert batch["truncated"].item() == (flags in ("truncated", "both")), "只可读取旧段终点的 truncated"


def test_reset_streams_pending_is_idempotent_and_survives_invalid_append() -> None:
    r"""重复索引与空子集有明确语义；append 验证失败不能消费待重置状态。"""
    replay = _replay(n_step=1)  # 立即审计 reset 前的已存经验及 reset 后首帧。
    replay.reset_streams()  # 空池重启等价于初始历史段起点零。
    replay.reset_streams(torch.tensor([], dtype=torch.long))  # 空整数子集是无副作用操作。
    replay.append(**_packet(0))  # S=1，已有合法的一步 target。
    replay.reset_streams(torch.tensor([0, 2, 2], dtype=torch.int32))  # 重复副本只标记一次。
    replay.reset_streams(torch.tensor([1]))  # 同一 append 前可累加不同子集。
    expected = copy.deepcopy(replay.state_dict())  # 三个副本待重置，副本三继续原段。
    replay.reset_streams(torch.tensor([2, 0]))  # 同一绝对 S 上重复调用应幂等。
    replay.reset_streams(torch.tensor([], dtype=torch.long))  # 不撤销已存在的 pending 标记。
    _assert_tree_equal(replay.state_dict(), expected)
    packet = _packet(1)  # 新初态输入的数值验证仍然先于写入。
    packet["rewards"][0] = float("nan")  # 单一非法奖励触发 append 拒绝。
    with pytest.raises(ValueError, match="finite"):
        replay.append(**packet)  # 失败后下一次合法 append 仍是新段首帧。
    _assert_tree_equal(replay.state_dict(), expected)
    for time in (1, 2):
        replay.append(**_packet(100 + time))  # 第一次消费段起点，第二次应继续同一新段。
    state = replay.state_dict()  # 直接审计逐环境绝对段起点。
    assert state["episode_start"][1:3].tolist() == [[1, 1, 1, 0], [1, 1, 1, 0]], "pending 必须只触发一次"
    assert replay.total_transitions == 12, "reset_streams 和失败 append 都不能计为采样"


@pytest.mark.parametrize(
    "indices",
    [
        torch.tensor([0, 4]),  # 前半合法、后半越界，必须先全部验证而非部分标记。
        torch.tensor([-1]),  # 禁止负索引回绕到最后一个副本。
        torch.tensor([0.0]),  # 整值浮点也不能作为环境身份。
        torch.tensor([[0]]),  # 环境子集必须是一维。
        torch.tensor([True]),  # bool 是 mask 语义，不是本接口的环境索引。
    ],
)
def test_reset_streams_rejects_invalid_subset_atomically(indices: torch.Tensor) -> None:
    r"""恢复子集索引失败不能改变任何副本的 pending 状态或旧经验。"""
    replay = _replay()  # 有效数据不能被失败的恢复接口污染。
    replay.append(**_packet(0))  # 非空流保证错误标记可被检测。
    before = copy.deepcopy(replay.state_dict())  # 小池的逐位审计基准。
    with pytest.raises((ValueError, TypeError)):
        replay.reset_streams(indices)  # 必须在写入 pending 起点前完整校验。
    _assert_tree_equal(replay.state_dict(), before)


@pytest.mark.parametrize("case", ["pending", "rewind", "lost_done"])
def test_checkpoint_rejects_invalid_segment_recurrence(case: str) -> None:
    r"""允许人工分段仍须拒绝任意改写历史：新起点只能是当前绝对步，物理 done 必须开新段。"""
    replay = _replay()  # 用同时包含物理结束与流重启的 checkpoint 审计时间不变量。
    for time in range(4):
        replay.append(**_packet(time, terminated=(1,) if time == 2 else ()))  # 副本一在绝对步 3 开新回合。
    replay.reset_streams()  # 绝对步 4 开始人工新段。
    replay.append(**_packet(100))  # 段起点 4，而非观察数值的时间标签 100。
    replay.append(**_packet(101))  # S=6，下一行只能继续起点 4 或待重置为 6。
    before = copy.deepcopy(replay.state_dict())  # 合法已实现段边界。
    bad = copy.deepcopy(before)  # 分别破坏三类独立时序约束。
    if case == "pending":
        bad["next_episode_start"][0] = 5  # 既不是当前段首 4，也不是下一步 6。
    elif case == "rewind":
        bad["episode_start"][5, 0] = 3  # 新段不能倒退到更早的无关首帧。
    else:
        bad["episode_start"][3, 1] = 0  # 真实 terminated 后不能继续原段。
    with pytest.raises(ValueError, match="episode_start"):
        replay.load_state_dict(bad)  # 加载失败必须早于任何持久复制。
    _assert_tree_equal(replay.state_dict(), before)
