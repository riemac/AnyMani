r"""首30秒已结算窗口的资产等权统计，仅消费CPU事实，不生成MDP真值。

调用方在首30秒完成或此前drop/axis时提供一行；早timeout不进入本统计器。
净转与绝对路径均以圈为单位，严格goal计数与轴向转动分别报告，ghost目标不参与计算。
窗口可跨PPO策略版本，因此一圈/两圈判据仅称跨版本训练proxy，不构成正式R16能力结论。
物理环境的首窗seen位、事件去重与reset生命周期由调用方管理；这里没有环境身份或训练RNG。
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np

# 只检查这些必需列，允许调用方传入带有其他诊断列的超集；顺序也是批量行解包顺序。
_COLUMN_KINDS = {
    "asset_index": "iu",  # $[E]$整数资产索引，有符号/无符号均可，bool不可
    "net_turns": "f",  # $[E]$浮点有符号净转，单位圈，允许负值
    "absolute_path_turns": "f",  # $[E]$浮点绝对路径圈数，理论上不小于净转绝对值
    "goal_count": "iu",  # $[E]$非负整数严格目标计数，不以ghost目标代替
    "safe": "b",  # $[E]$布尔安全事实，不用整数0/1隐式推断
    "policy_version_start": "iu",  # $[E]$非负整数起始策略版本
    "policy_version_end": "iu",  # $[E]$非负整数结束版本，逐窗end>=start
}

# 比值容差约为8个float32机器epsilon；固定容差使Python浮点状态恢复不改变验收边界。
# 只允许$|net/path|\le1+10^{-6}$的舍入越界，绝不使用绝对圈数容差掩盖零路径上的非零净转。
_DIRECTION_TOLERANCE = 1e-6  # 无量纲，明显的路径不一致仍在clip之前拒绝
_PROXY_MINIMUM_WINDOWS = 16  # 一圈/两圈训练proxy的固定样本门，与描述量资格配置分开
_Window = tuple[float, float, int, bool, int, int]  # net、path、goal、safe、start、end；仅Python标量
_Metrics = dict[str, int | float | None]  # 单资产与资产组使用完全相同的标量schema


def _integer(value: object, name: str, minimum: int) -> int:
    r"""验证配置或选择器的真正整数；拒绝bool及任何浮点到索引的隐式截断。"""
    # np.integer可无损转成Python整数，包含uint64；先排除bool，因为Python中bool继承int。
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):  # 布尔真值不能成为资产数或索引1
        raise ValueError(f"{name} must be an integer, not boolean or floating point")  # 保留索引语义
    result = int(value)  # 经dtype验证后的精确整数转换，不缩窄到int64
    if result < minimum:  # 资产数可为0，容量/资格门至少为1
        raise ValueError(f"{name} must be >= {minimum}, got {result}")  # 拒绝负索引等边界
    return result  # 公开配置与状态始终为Python整数


def _validated_window(values: object) -> _Window:
    r"""验证一行Python事实并返回不可变窗口；批量添加和状态恢复共享同一物理约束。"""
    # 固定六元原始窗口不保存方向的冗余副本，恢复时可由原始事实重新归约。
    expected = (float, float, int, bool, int, int)  # 每个位置的Python标量类型
    if not isinstance(values, (list, tuple)) or len(values) != len(expected):  # 单窗必须保留六项联合事实
        raise ValueError("a window must contain [net, path, goal, safe, policy_start, policy_end]")  # 完整窗口
    if any(type(value) is not kind for value, kind in zip(values, expected, strict=True)):  # 恢复也遵守原始标量类型
        raise ValueError("window scalar types must be float, float, int, bool, int, int")  # bool不可冒充计数
    net, path, goals, _, start, end = cast(_Window, tuple(values))  # 已验证类型，不进行有损强转

    # 有限性在比值前检查；尤其inf路径不能把缺测净转伪装成方向0。
    if not math.isfinite(net) or not math.isfinite(path):  # 同时拒绝转换为Python浮点后溢出的宽浮点输入
        raise ValueError("net_turns and absolute_path_turns must be finite Python floats")  # 缺测不补零
    if path < 0 or goals < 0 or start < 0 or end < start:  # end>=start>=0同时保证两个版本非负
        raise ValueError("path/goal_count/policy versions must be nonnegative and policy end >= start")  # 事实边界
    try:  # NumPy整数均可形成有限浮点报告；手工状态中的任意精度巨大整数也须满足该输出合同
        float(goals)  # goal原始整数仍精确保留，只检查报告层能否表达中位数
    except OverflowError as error:  # 在恢复阶段拒绝无法归约的整数，而非延迟到summary才失败
        raise ValueError("goal_count must admit a finite Python float summary") from error  # 防止中位数报告溢出
    if path == 0.0:  # 零路程只允许净转严格为零，包括IEEE有符号零
        if net != 0.0:  # 零路径的三角不等式要求净转恰为零，不引入绝对圈数容差
            raise ValueError("zero absolute path requires exactly zero net_turns")  # 不以容差修复不可能的运动
    elif abs(net / path) > 1.0 + _DIRECTION_TOLERANCE:  # 先验证三角不等式，再允许方向clip
        raise ValueError("absolute path is smaller than abs(net_turns) beyond floating-point tolerance")  # 不修补物理矛盾
    return cast(_Window, tuple(values))  # 不保留调用方可变列表或NumPy数组


def _direction(window: _Window) -> float:
    r"""已验证窗口的方向：零路径为0，否则为net/path，仅修正容差内的±1浮点越界。"""
    net, path = window[:2]  # 角位移/路程的量纲相同，比值无量纲
    return 0.0 if path == 0.0 else max(-1.0, min(1.0, net / path))  # $d\in[-1,1]$，反向运动保持负号


def _median(values: Sequence[int | float]) -> float:
    r"""非空有限数据的中位数；偶数样本稳定求中点，避免两个最大有限浮点值相加溢出。"""
    ordered = sorted(values)  # 输入可为单资产至多32窗或组内资产中位数；排序不改变原始窗口时间顺序
    middle = len(ordered) // 2  # 中央位置，奇数时为唯一中位数
    if len(ordered) % 2:  # 奇数样本无需插值
        return float(ordered[middle])  # 报告层统一返回Python浮点数
    low, high = float(ordered[middle - 1]), float(ordered[middle])  # 偶数样本的两项中央顺序统计量
    if low <= 0.0 <= high:  # 异号相加不会溢出；避免先计算可能溢出的high-low
        return (low + high) / 2.0  # $m=(x_{n/2}+x_{n/2+1})/2$
    return low + (high - low) / 2.0  # 同号差值有限，也保留相等次正规数的中位数


class FirstWindowStatistics:
    r"""每资产独立FIFO上的首30秒训练描述量与固定样本门proxy。

    对资产$a$的留存窗口$W_a$，分别计算$\tilde n_a=median(net)$、
    $\tilde g_a=median(goal)$、$\tilde d_a=median(net/path)$和
    $s_a=\sum_{w\in W_a}safe_w/|W_a|$；零路径的方向定义为0。
    对所选且有窗口的资产，再取三个资产中位数的中位数与安全率的等权平均。
    因而快速失败资产即便产生更多窗口，也不会以窗口数量重加权其他资产。
    窗口计数与最小/最大样本量仍包含所选的缺测资产；低样本描述量照常报告并单列资格数。

    一圈/两圈训练proxy分别要求净转中位数至少1圈/至少2圈；共同要求至少16窗、方向中位数>=0.7、安全率>=0.75。
    方向是有符号的$[-1,1]$量，不是概率；安全率才是$[0,1]$的经验伯努利比例。
    概率参考：若16窗真为独立同分布且真实安全概率为1/2，至少12窗安全的概率为
    $\sum_{k=12}^{16}\binom{16}{k}2^{-16}=2517/65536\approx0.0384$。
    这仅描述单个安全门；不能作为三个门联合通过或多资产筛选的显著性水平。
    在$n=16,\hat p=0.75$时，独立近似标准误$\sqrt{\hat p(1-\hat p)/n}\approx0.108$，
    实际训练窗又有共享策略、时间相关与跨版本非平稳性，故16窗仍属低样本训练proxy，不能称R16。

    Args:
        asset_count: 资产宇宙大小$A>=0$，所有公开资产索引属于0..A-1。
        max_windows_per_asset: 每资产FIFO容量，范围1..32，默认32；按事件输入顺序保留最近窗口。
        minimum_windows: qualified计数门，正整数、默认16；可以大于容量，此时没有qualified资产。

    Raises:
        ValueError: 配置不是有效整数，或容量超出1..32。
    """

    def __init__(self, asset_count: int, max_windows_per_asset: int = 32, minimum_windows: int = 16) -> None:
        r"""建立空窗口历史；不读取环境、创建随机流或初始化物理seen位。"""
        self.asset_count = _integer(asset_count, "asset_count", 0)  # 固定资产宇宙，可以为空
        self.max_windows_per_asset = _integer(max_windows_per_asset, "max_windows_per_asset", 1)  # 最近历史容量
        self.minimum_windows = _integer(minimum_windows, "minimum_windows", 1)  # 描述量资格门
        if self.max_windows_per_asset > 32:  # 合同规定每资产最多留存32个首窗
            raise ValueError("max_windows_per_asset must be <= 32")  # 不扩展历史储存协议
        self._windows: list[deque[_Window]] = [  # 每个deque独立淘汰，只拥有不可变Python窗口
            deque(maxlen=self.max_windows_per_asset) for _ in range(self.asset_count)  # A个独立空队列，缺测资产也占一项
        ]

    def add_batch(self, columns: Mapping[str, np.ndarray]) -> None:
        r"""验证整批$[E]$CPU事实后按输入顺序追加；非法行不会部分写入或触发FIFO淘汰。

        Args:
            columns: 七个必需列见模块_COLUMN_KINDS；可含超集，附加列完全不参与统计。
                E=0是合法无操作。浮点物理列、整数索引/计数/版本和bool安全列必须符合声明dtype。
                状态采用Python float；更宽浮点若溢出或把非零量下溢成零，则整批拒绝。

        Raises:
            ValueError: 缺列、非NumPy数组、非一维/不对齐、非法dtype、非有限值或物理事实不一致。
        """
        # 先锁定必需列，既不使用np.asarray隐式接收tensor，也不读取ghost等附加诊断。
        if not isinstance(columns, Mapping) or any(name not in columns for name in _COLUMN_KINDS):
            raise ValueError(f"columns must supply all required arrays: {tuple(_COLUMN_KINDS)}")  # 完整CPU事实
        arrays = [columns[name] for name in _COLUMN_KINDS]  # 固定七列顺序，与窗口解包对应
        for (name, kinds), array in zip(_COLUMN_KINDS.items(), arrays, strict=True):  # 空批也须满足七列类型合同
            if not isinstance(array, np.ndarray) or np.ma.isMaskedArray(array) or array.ndim != 1:  # 每行对应一个真实事件
                raise ValueError(f"{name} must be an unmasked CPU numpy [E] array")  # 掩码缺测不能当成真实值
            if array.dtype.kind not in kinds:  # 先检查dtype，禁止bool计数或float索引静默转换
                raise ValueError(f"{name} has invalid dtype {array.dtype}; expected numpy kind {kinds}")  # 不隐式截断列值
            if array.dtype.kind == "f" and not np.isfinite(array).all():  # 原始精度下先检查NaN/inf
                raise ValueError(f"{name} must contain only finite values")  # 后续转换也再次检查有限性
        count = arrays[0].size  # $E$是本批已结算窗口数，不是物理环境总数
        if any(array.shape != (count,) for array in arrays):  # 不flatten、不广播缺失行
            raise ValueError("all required columns must be aligned [E] arrays")  # 联合窗口必须逐行对齐

        # 所有行先物化为Python标量并验证；即使最后一行非法，也不会淘汰任一资产的旧窗口。
        pending: list[tuple[int, _Window]] = []  # 局部事务，仅在全部通过后写入历史
        for asset, net, path, goals, safe, start, end in zip(*arrays, strict=True):  # 七列按同一事件行联合验证
            index = int(asset)  # 整数dtype已验证，uint64也保留精确数值
            if not 0 <= index < self.asset_count:  # 明确拒绝负索引与资产宇宙外的索引
                raise ValueError(f"asset_index {index} outside [0, {self.asset_count})")  # 不允许负索引指向末资产
            net_value, path_value = float(net), float(path)  # Python状态使用binary64，不能把宽浮点非零运动下溢成0
            if (net != 0.0 and net_value == 0.0) or (path != 0.0 and path_value == 0.0):  # 序列化不能把微小运动变成静止
                raise ValueError("physical values must not underflow to zero in Python float state")  # 拒绝静默丢失事实
            window = _validated_window((net_value, path_value, int(goals), bool(safe), int(start), int(end)))  # 原始六元窗口
            pending.append((index, window))  # 保持输入行顺序，重复资产不是重复事件的判据
        for index, window in pending:  # 提交阶段没有数据校验分支，容量淘汰只发生在这一阶段
            self._windows[index].append(window)  # FIFO逐资产独立；对象不持有NumPy/tensor引用

    def per_asset(self) -> dict[int, dict[str, int | float | None]]:
        r"""返回0..A-1全部资产的新标量字典；缺测资产也按单资产同schema报告。"""
        return {index: self.summary([index]) for index in range(self.asset_count)}  # 单资产定义复用同一归约路径

    def summary(self, asset_indices: Sequence[int] | None = None) -> _Metrics:
        r"""汇总显式资产集合；有窗口资产等权，缺测资产仅进入资产数与窗口数范围。

        Args:
            asset_indices: None表示全部资产；接受一维整数序列，包含NumPy整数数组；空序列表示空组。

        Returns:
            14个first30_*字段的Python标量字典；无观测的浮点描述量及策略版本界为None。
            first30_asset_count仍是所选资产数，其余观测/资格/proxy/窗口计数无数据时为0。

        Raises:
            ValueError: 非序列、非整数/布尔索引、重复资产或越界；绝不按重复次数隐式重加权。
        """
        # 选择器保持集合语义，但接口要求有序的一维Sequence，避免误读字符串、映射或二维数组。
        if asset_indices is None:  # 固定资产宇宙，不以本批出现的资产动态改写组大小
            selected = list(range(self.asset_count))  # 全部0..A-1，包含缺数据资产
        else:
            if (
                not isinstance(asset_indices, (Sequence, np.ndarray))  # 接受显式序列及NumPy选择器
                or isinstance(asset_indices, (str, bytes))  # 字符串不是资产索引序列
                or (isinstance(asset_indices, np.ndarray) and asset_indices.ndim != 1)  # 禁止展开二维选择器
            ):
                raise ValueError("asset_indices must be a one-dimensional integer sequence")  # 拒绝集合等隐式转换
            selected = [_integer(index, "asset_index", 0) for index in asset_indices]  # 禁止float/bool索引
            if any(index >= self.asset_count for index in selected) or len(set(selected)) != len(selected):  # 重复索引会改变权重
                raise ValueError("asset_indices must be unique and inside the configured asset range")  # 无隐藏权重

        # 第一层是资产内窗口分布：方向先逐窗算net/path，不能用两个中位数的比值替代。
        counts = [len(self._windows[index]) for index in selected]  # 缺测资产贡献0个窗口
        observed = [self._windows[index] for index in selected if self._windows[index]]  # 仅有观测资产参与均值
        values = [  # 每项为(n, median_net, median_goal, median_direction, safe_fraction)
            (
                len(windows),  # 当前留存窗口数，最多32
                _median([window[0] for window in windows]),  # 净转中位数，单位圈
                _median([window[2] for window in windows]),  # 严格目标计数中位数
                _median([_direction(window) for window in windows]),  # 无量纲有符号方向中位数
                sum(window[3] for window in windows) / len(windows),  # $s_a=\sum safe/|W_a|\in[0,1]$
            )
            for windows in observed  # 每项恰对应一只已观测手，所有资产内分母均非零
        ]

        # 第二层中位数/均值对资产等权；低样本资产描述量照常参与，资格与proxy样本门单独计数。
        medians = [_median([row[column] for row in values]) if values else None for column in (1, 2, 3)]  # 组内三个二层中位数
        safe_fraction = sum(row[4] for row in values) / len(values) if values else None  # 每只已观测手权重1/A_obs
        eligible_net = [  # 固定16窗并且同时满足方向与安全门的资产，再检查一圈/两圈净转门
            net  # 已满足窗口数、方向与安全门的资产净转中位数
            for n, net, _, direction, safe in values  # goal计数独立报告，不进入净转能力代理
            if n >= _PROXY_MINIMUM_WINDOWS and direction >= 0.7 and safe >= 0.75  # 样本、方向与安全三门取交集
        ]
        return {
            "first30_asset_count": len(selected),  # 所选资产全集大小，不等于已观测资产数
            "first30_observed_assets": len(observed),  # 至少1窗的资产数
            "first30_qualified_assets": sum(count >= self.minimum_windows for count in counts),  # 配置资格门
            "first30_window_count": sum(counts),  # 当前留存窗口总数，不是累计历史数
            "first30_windows_min": min(counts, default=0),  # 包含缺测资产的0
            "first30_windows_max": max(counts, default=0),  # 空组返回0
            "first30_net_median": medians[0],  # median_a(median_w(net))，单位圈
            "first30_goal_median": medians[1],  # median_a(median_w(goal))，不能替代净转
            "first30_direction_median": medians[2],  # median_a(median_w(direction))，范围[-1,1]
            "first30_safe_fraction": safe_fraction,  # mean_a(safe_fraction_a)，范围[0,1]
            "first30_one_turn_proxy_assets": sum(net >= 1.0 for net in eligible_net),  # 跨版本一圈训练proxy
            "first30_two_turn_proxy_assets": sum(net >= 2.0 for net in eligible_net),  # 跨版本两圈训练proxy
            "first30_policy_start_min": min((window[4] for windows in observed for window in windows), default=None),  # 留存窗起始版本下界
            "first30_policy_end_max": max((window[5] for windows in observed for window in windows), default=None),  # 留存窗结束版本上界
        }

    def state_dict(self) -> dict[str, object]:
        r"""返回独立的纯Python状态；windows[a]按最旧→最新保存[net,path,goal,safe,start,end]。

        只保存asset_count、max_windows_per_asset、minimum_windows及每资产有序原始窗口。
        不保存派生中位数、物理环境seen位或RNG；修改返回列表不会改写内部历史。
        """
        return {
            "asset_count": self.asset_count,  # 固定资产宇宙
            "max_windows_per_asset": self.max_windows_per_asset,  # 恢复必须保持相同历史截断规则
            "minimum_windows": self.minimum_windows,  # 恢复必须保持相同资格规则
            "windows": [[list(window) for window in windows] for windows in self._windows],  # 两层列表复制，无别名
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        r"""完整验证配置和有序窗口后一次性替换状态；失败时原对象不变。

        Args:
            state: state_dict生成的纯Python标量/列表映射；配置必须与本对象完全一致。

        Raises:
            ValueError: schema缺失/多余、配置不匹配、资产/窗口列表形状错误、超容量或非法原始事实。
                超容量状态属于坏数据，恢复不会通过FIFO静默截断它。
        """
        # 精确schema让环境seen/RNG等不属于本统计器的生命周期状态无法混入恢复协议。
        expected = {"asset_count", "max_windows_per_asset", "minimum_windows", "windows"}  # 四个固定顶层字段
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError(f"first-window state must contain exactly {sorted(expected)}")  # 不猜测状态缺省值
        for name in ("asset_count", "max_windows_per_asset", "minimum_windows"):  # 三项配置均属于统计语义
            if type(state[name]) is not int or state[name] != getattr(self, name):  # 同值bool/float也不可充当配置
                raise ValueError(f"state configuration {name} does not match this statistics object")  # 不在恢复时重定义资产测度
        histories = state["windows"]  # 列表下标就是资产索引，缺测资产必须显式保留空列表
        if type(histories) is not list or len(histories) != self.asset_count:
            raise ValueError("state windows must be a list with one history per configured asset")  # 保持资产宇宙

        # 临时构造所有资产；即使最后资产最后窗口失败，也不会清空或替换已存在的历史。
        restored: list[deque[_Window]] = []  # 恢复事务的候选状态，与目标对象分离
        for history in histories:  # 外层顺序固定对应资产0..A-1
            if type(history) is not list or len(history) > self.max_windows_per_asset:
                raise ValueError("state asset histories must be lists within the configured window capacity")  # 超容量不能静默淘汰
            if any(type(window) is not list for window in history):  # 拒绝tuple/array等非声明状态表示
                raise ValueError("state windows must contain Python lists of scalar facts")  # 恢复协议不持有设备对象
            restored.append(deque((_validated_window(window) for window in history), maxlen=self.max_windows_per_asset))  # 原顺序逐窗复验
        self._windows = restored  # 全部验证通过后的唯一状态替换，不改动配置、物理seen或训练RNG
