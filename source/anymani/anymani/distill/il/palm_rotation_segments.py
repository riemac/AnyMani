r"""冻结教师采集的30秒示范段与episode级训练/验证划分，仅处理CPU数组。

一个输入bank属于一位冻结教师。行t保存第t次动作之后、自动reset之前的物理结果；
观察、动作中心标签及真实执行动作由采集器另存于同一(t,env)轴。这里返回完整时间区间，
不按瞬时转速裁掉停滞帧，因此合格段中的回中、卡住与后续恢复仍可供学生学习。
固定20Hz、600步、净圈>=.5、方向>=.7且无物理失败的片段进入模仿数据，资产是否达到
正式1圈/R16门不参与此筛选。两个教师的bank应分别划分，再通过显式资产映射合并。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

WINDOW_STEPS = 600  # 30秒×20Hz；区间包含600个真实动作及对应物理结果。
MINIMUM_NET_TURNS = 0.5  # 数据门，独立于正式可靠资产的1圈门。
MINIMUM_DIRECTION = 0.7  # 每段signed net / absolute path，不是资产级ratio-of-medians。


@dataclass(frozen=True)
class DemonstrationSegment:
    r"""一段完整连续轨迹及其联合物理质量；frame索引用于取全部600帧。

    start_frame包含、stop_frame不包含；两者索引bank时间轴，而非本回合的局部step。
    safe只描述本段是否发生物理失败，timeout结束的完整段仍可安全。accepted是数据准入，
    不等同于资产可靠认证；不合格段保留真实统计和所有拒绝原因。
    """

    env_id: int  # 固定采集环境索引，仅作数据定位。
    asset_index: int  # 本teacher bank的资产索引，不进入Actor输入。
    episode_id: int  # 在对应环境内递增的回合ID。
    start_frame: int  # bank时间轴包含式起点。
    stop_frame: int  # bank时间轴排除式终点。
    stochastic: bool  # 教师原随机策略或确定性动作中心的行为来源。
    net_turns: float  # 本段有符号净转，不使用全episode累计量代替。
    absolute_path_turns: float  # 同一时间段的累计绝对转动路径。
    goal_count: int  # 本段严格子目标数，独立于物理圈数。
    directional_consistency: float  # signed net/path，零路径为0。
    safe: bool  # 本段内无drop/axis；不把纯timeout当掉落。
    accepted: bool  # .5圈、方向与物理安全的联合数据门。
    rejection_reasons: tuple[str, ...]  # 所有未满足的数据门，保留原因而非删除事实。


@dataclass(frozen=True)
class EpisodeDataSplit:
    r"""索引输入segment序列的确定性划分，完整episode不跨训练和验证集合。"""

    training_indices: tuple[int, ...]  # 已接受片段中的训练条目。
    validation_indices: tuple[int, ...]  # 已接受片段中的独立episode验证条目。
    assets_without_validation: tuple[int, ...]  # 有训练数据却无独立验证episode的资产。


def _array(value: Any, name: str, shape: tuple[int, ...], kinds: str) -> np.ndarray:
    r"""保持采集轴与dtype语义，拒绝隐式转型、masked缺测和广播。"""
    if not isinstance(value, np.ndarray) or np.ma.isMaskedArray(value):  # 缺测掩码与真实物理零值有不同含义。
        raise ValueError(f"{name} must be an unmasked numpy array")  # 输入是原始CPU事实。
    if value.shape != shape or value.dtype.kind not in kinds:  # 每列必须指向同一(t,env)物理时刻。
        raise ValueError(f"{name} must have shape {shape} and dtype kind {kinds}")  # 不flatten记录轴。
    if kinds == "f" and (value.dtype.itemsize not in (4, 8) or not np.isfinite(value).all()):  # 半精度累计误差不属于本bank合同。
        raise ValueError(f"{name} must contain finite float32/float64 facts")  # 保持采集的数值精度。
    if kinds == "iu" and (np.any(value < 0) or np.any(value >= 2**63)):  # 对齐持久化有符号64位索引范围。
        raise ValueError(f"{name} must contain nonnegative int64-representable values")  # bool由kind检查拒绝。
    return value  # 不复制大bank，不把缺测或坏ID悄悄修正。


def select_demonstration_segments(
    *,
    net_turns: np.ndarray,  # [T,N]post-physics本回合累计有符号圈数。
    absolute_path_turns: np.ndarray,  # [T,N]相同起点的累计绝对路径圈数。
    goal_count: np.ndarray,  # [T,N]包含当前动作结果的严格目标累计数。
    episode_id: np.ndarray,  # [T,N]动作执行时所属回合，reset后下一行才递增。
    episode_step: np.ndarray,  # [T,N]本回合已完成的策略步数，首动作后为1。
    physical_failure: np.ndarray,  # bool[T,N]，drop OR axis。
    timeout: np.ndarray,  # bool[T,N]，可与physical_failure同时发生。
    asset_index_by_env: np.ndarray,  # int[N]，整个bank内环境到资产固定。
    stochastic_by_env: np.ndarray,  # bool[N]，两种冻结教师行为由副本固定分配。
) -> tuple[DemonstrationSegment, ...]:
    r"""按回合内不重叠600步窗口筛选示范，输出合格和不合格完整段。

    累计量记为N_t、P_t、G_t。区间[s,e]的量为(N_e-N_{s-1},P_e-P_{s-1},G_e-G_{s-1})，
    本回合第一段的前值为0。时间和reset先整bank验证，避免把不同回合接成一条示范。
    未满600步的尾段没有完整30秒观察机会，不输出为失败或成功，也不按时长外推。

    Returns:
        tuple: 按结束时间、环境索引升序的完整段；accepted只由数据门决定。
    """
    if not isinstance(net_turns, np.ndarray) or net_turns.ndim != 2 or min(net_turns.shape) < 1:  # 首列定义共同T/N轴。
        raise ValueError("teacher bank needs nonempty [time,environment] arrays")  # 不接受空运行伪产物。
    shape = net_turns.shape  # T个连续策略步、N个静态采集环境。
    _, env_count = shape  # 环境轴保持原顺序，不按成绩重排。
    net = _array(net_turns, "net_turns", shape, "f")  # 单位圈，允许反向净转。
    path = _array(absolute_path_turns, "absolute_path_turns", shape, "f")  # 单位圈，非负路径。
    goals = _array(goal_count, "goal_count", shape, "iu")  # 严格事件计数，不以角度换算。
    ids = _array(episode_id, "episode_id", shape, "iu").astype(np.int64, copy=False)  # 精确时钟比较。
    steps = _array(episode_step, "episode_step", shape, "iu").astype(np.int64, copy=False)  # 已完成动作数。
    failure = _array(physical_failure, "physical_failure", shape, "b")  # 物理安全事实。
    timed_out = _array(timeout, "timeout", shape, "b")  # 有限时域结束事实。
    assets = _array(asset_index_by_env, "asset_index_by_env", (env_count,), "iu")  # 资产可重复对应多个副本。
    stochastic = _array(stochastic_by_env, "stochastic_by_env", (env_count,), "b")  # 不作为学生特征。

    # 全bank来自一次显式reset后的连续采集；任何漏帧/错配reset都影响History30的重建。
    done = failure | timed_out  # 两旗同时只造成一次episode递增。
    if np.any(steps[0] != 1) or np.any(steps < 1):  # 采集起点必须有可定义的episode初始历史。
        raise ValueError("teacher recording must start at episode step1")  # 首段历史可合法重复初帧。
    if np.any(ids[1:] != ids[:-1] + done[:-1].astype(np.int64)):  # 终止事实和下一行回合身份逐环境对应。
        raise ValueError("episode IDs must advance exactly once after a recorded terminal")  # 不容许隐式拼接。
    if np.any(steps[1:] != np.where(done[:-1], 1, steps[:-1] + 1)):  # 连续帧不能跨越未记录的动作。
        raise ValueError("episode step clock has a gap or inconsistent reset")  # 600条记录必须确为600步。
    if np.any(path < 0) or np.any(np.abs(net.astype(np.float64)) > path.astype(np.float64) * (1 + 1e-6)):  # 有符号位移受路径约束。
        raise ValueError("cumulative absolute path must bound signed net turns")  # 全episode三角不等式。
    same_episode = ~done[:-1]  # reset后的累计量允许归零，回合内路径/计数只增不减。
    if np.any((path[1:] < path[:-1]) & same_episode) or np.any((goals[1:] < goals[:-1]) & same_episode):  # 回合内累计量的单调性。
        raise ValueError("absolute path and goal counts must be monotone inside an episode")

    # 每600个episode步产生一个完整候选段；窗口起点位于同一bank，不移动真实物理初态。
    end_frames, environments = np.nonzero(steps % WINDOW_STEPS == 0)  # 行优先顺序给出确定性的事件排列。
    epsilon = max(np.finfo(net.dtype).eps, np.finfo(path.dtype).eps)  # 原始累计量的舍入精度。
    result: list[DemonstrationSegment] = []  # 仅存小型区间/质量表，动态帧仍在原bank。
    for end, env in zip(end_frames.tolist(), environments.tolist(), strict=True):  # 每个事件都是一段完整观察机会。
        start = end - WINDOW_STEPS + 1  # 闭区间[s,e]含精确600帧。
        if start < 0 or ids[start, env] != ids[end, env]:  # 目标动作与物体状态不能来自不同回合。
            raise ValueError("complete window lacks its own episode prefix")  # 全局时钟合法时不应发生。
        initial = steps[start, env] == 1  # 本回合第一窗口使用零累计基线。
        base_net = 0.0 if initial else float(net[start - 1, env])  # 前一动作之后的净转。
        base_path = 0.0 if initial else float(path[start - 1, env])  # 与净转完全相同的时间边界。
        base_goals = 0 if initial else int(goals[start - 1, env])  # 不重复计入前一段的目标。
        net_delta = float(net[end, env]) - base_net  # 后30秒的增量，不是累计到60/90/120秒的总量。
        path_delta = float(path[end, env]) - base_path  # 同一段绝对路径。
        goal_delta = int(goals[end, env]) - base_goals  # 精确整数目标数。

        # FP32累计器相减的误差随前缀尺度和窗口步数增长；该界只验证物理自洽，不放宽.5圈门。
        scale = max(1.0, abs(base_net), abs(float(net[end, env])), base_path, float(path[end, env]))
        tolerance = 4 * WINDOW_STEPS * float(epsilon) * scale  # 两累计器舍入的保守尺度界，单位圈。
        if abs(net_delta) > path_delta + tolerance:  # 仅容许原累计精度能解释的数值偏差。
            raise ValueError("segment net/path violates the cumulative roundoff bound")  # 不用clip隐藏明显坏数据。
        direction = 0.0 if path_delta == 0 else float(np.clip(net_delta / path_delta, -1.0, 1.0))  # 无量纲有符号方向。
        safe = not bool(failure[start : end + 1, env].any())  # 末帧失败也排除；纯timeout保持安全。
        reasons = tuple(  # 联合门保留多个失败原因，不以单一排序标签遮蔽安全问题。
            reason for rejected, reason in (
                (not safe, "physical-failure"),  # 安全条件不能被较大净圈抵消。
                (net_delta < MINIMUM_NET_TURNS, "net-below-0p5"),  # 对实际记录量作固定阈值判断。
                (direction < MINIMUM_DIRECTION, "direction-below-0p7"),  # 不奖励往返累计路径。
            ) if rejected
        )  # 低质量数据保留全部拒绝原因，原始帧不删除。
        result.append(DemonstrationSegment(  # 输出区间索引，不复制或改变教师控制过程。
            env_id=env, asset_index=int(assets[env]), episode_id=int(ids[end, env]),  # 唯一来源与统计归属。
            start_frame=start, stop_frame=end + 1, stochastic=bool(stochastic[env]),  # Python切片边界。
            net_turns=net_delta, absolute_path_turns=path_delta, goal_count=goal_delta,  # 同时间区间的三种真实累计差。
            directional_consistency=direction, safe=safe, accepted=not reasons, rejection_reasons=reasons,  # 固定数据门裁决。
        ))  # 一条记录代表完整600帧，包括其中的暂时停滞与恢复。
    return tuple(result)  # 不持有可变bank数组，也不触发物理运行。


def split_demonstration_episodes(
    segments: tuple[DemonstrationSegment, ...], *, seed: int = 42,
) -> EpisodeDataSplit:
    r"""在资产与采集行为内按完整episode约3:1划分，单episode组全部留作训练。

    同一个episode的所有合格段共享标签。每组至少有两个episode时取max(1,floor(n/4))个
    验证episode，其余用于训练；数据少时实际比例随整数取整变化并保留缺口。局部default_rng
    不改变教师或学生使用的全局NumPy/Torch随机状态，文件中的初态与物理轨迹也保持不变。
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("split seed must be a nonnegative integer")  # 不隐式截断浮点种子。
    grouped: dict[tuple[int, bool], dict[tuple[int, int], list[int]]] = defaultdict(lambda: defaultdict(list))  # 资产/行为→episode→段。
    episode_groups: dict[tuple[int, int], tuple[int, bool]] = {}  # 一个bank内env/episode唯一定位数据来源。
    for index, segment in enumerate(segments):  # 输入顺序仅定义索引，不作为质量排序。
        if not segment.accepted:  # 未过数据门的区间不提供模仿标签。
            continue  # 只划分已通过数据门的段；拒绝段仍留在原始段表。
        key = (segment.env_id, segment.episode_id)  # 同一episode可贡献多个连续30秒段。
        group = (segment.asset_index, segment.stochastic)  # 同时兼顾资产和教师行为模式。
        if key in episode_groups and episode_groups[key] != group:  # 一个真实回合有固定的资产和行为策略。
            raise ValueError("an episode cannot change its asset or collection behavior")
        episode_groups[key] = group  # 来源不进入Actor，只影响数据隔离。
        grouped[group][key].append(index)  # 保留该episode的全部已接受区间。
    generator = np.random.default_rng(seed)  # 与训练全局随机流分开的可复现局部生成器。
    train: list[int] = []  # 返回输入segment序列的索引，不复制动态观察数据。
    validation: list[int] = []  # 验证索引按完整episode组形成。
    for group in sorted(grouped):  # 稳定分组次序使显式seed具有可复现实义。
        episodes = grouped[group]  # 一个资产/行为分层内的独立episode集合。
        keys = sorted(episodes)  # 先固定输入顺序，再应用显式seed的排列。
        validation_count = max(1, len(keys) // 4) if len(keys) >= 2 else 0  # 单episode不能制造独立验证。
        permutation = generator.permutation(len(keys))  # 只在本组episode级抽样。
        held_out = {keys[int(i)] for i in permutation[:validation_count]}  # 所有窗口随episode一起划分。
        for key in keys:  # 一个episode所有600步段共同进入同一集合。
            (validation if key in held_out else train).extend(episodes[key])  # 不逐帧随机拆分。
    training_assets = {segments[i].asset_index for i in train}  # 实际有监督支持的资产。
    validation_assets = {segments[i].asset_index for i in validation}  # 实际存在独立验证episode的资产。
    return EpisodeDataSplit(
        training_indices=tuple(sorted(train)), validation_indices=tuple(sorted(validation)),
        assets_without_validation=tuple(sorted(training_assets - validation_assets)),  # 缺口显式报告。
    )  # 返回不可变索引，后续采样器再执行族/资产等权抽样。
