r"""FlashSAC 随机训练回合的首30秒统计，按资产维护近期窗口。

观测来自 command.post_physics_evaluation_snapshot：当前物理步已经完成，而环境
尚未 reset；episode_duration_s 是本回合实际经过秒数，并非计划的120秒时域。
首次 first30_complete=True 时读取上游冻结的 net_turns_first30；此前若发生真实
drop/axis 失败，则只记终止净转角除以 2π，不用 30/duration 放大短存活轨迹。
提前纯 timeout 是删失：增加累计删失计数，但不进入净圈均值或安全比例的分母。

每资产独立保留最近 M=32 个 (net, safe, policy_version) 窗口。主净圈指标先求
每资产的近期 episode 均值，再对有窗口的资产等权；同时报告池均值与池安全比例。
这些是跨策略版本的随机训练窗口，policy_version 仅表示结算时的策略版本，不能
断言整段轨迹由同一版本产生。本指标不是冻结 checkpoint 的 R16 副本中位数能力。
资产身份仅用于统计分组，不进入 Actor 观察。
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

import torch

# 只要求这些字段存在；上游完整 snapshot 的其他诊断字段既不参与计算，也不搬到 CPU。
_FLOAT_FIELDS = ("episode_duration_s", "net_rotation_rad", "net_turns_first30")  # 秒、弧度、有符号圈数。
_BOOL_FIELDS = (  # 不把连续数值或任意非零整数隐式解释为物理事件。
    "first30_complete",  # 上游已经冻结首30秒窗口。
    "termination_object_out_of_anchor",  # 物体脱离锚点/掉落。
    "termination_goal_axis_misaligned",  # 目标旋转轴偏离。
    "termination_time_out",  # 计划回合时域结束。
)
_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)  # 环境/资产索引dtype。


def _integer(name: str, value: int, minimum: int = 0) -> int:
    r"""保持计数和版本的精确 Python 整数身份，拒绝 bool、浮点截断及负数。"""
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value  # 只验证，不修正调用者的配置或版本。


class FirstThirtySecondsMetrics:
    r"""从逐步 pre-reset 事实结算训练回合首30秒窗口，不估计冻结策略能力。

    Args:
        asset_index_by_env: 非空 [N] 非负整数资产索引；允许非连续索引及不同副本数。
        max_episodes_per_asset: 每资产独立 deque 上限 M，默认32，只计有效窗口。

    每个物理步调用一次 observe。每环境 seen 在首次有效结算后置位，物理 done
    当步先处理本回合结果、再清 seen，因此后期失败不会覆盖30秒已经结算的结果。
    重新创建物理环境时，在首个新快照前调用 reset_streams 清相应 seen。

    动态事件筛选与 seen 在快照设备上完成；仅把新窗口/删失事件的少量索引与
    (net,safe,censored) 行复制到 CPU。deque 中只有 Python float/bool/int。
    相同步同时结算的同资产副本按环境索引升序入队，保证截窗和恢复可重复。
    """

    def __init__(self, asset_index_by_env: torch.Tensor, max_episodes_per_asset: int = 32) -> None:
        r"""冻结统计资产路由，分配逐环境去重位与逐资产的有限近期窗口。"""
        self.max_episodes_per_asset = _integer("max_episodes_per_asset", max_episodes_per_asset, 1)  # M>=1。
        if not isinstance(asset_index_by_env, torch.Tensor) or asset_index_by_env.ndim != 1:
            raise ValueError("asset_index_by_env must be a one-dimensional tensor")
        if asset_index_by_env.layout != torch.strided or asset_index_by_env.dtype not in _INTEGER_DTYPES:
            raise TypeError("asset_index_by_env must be a dense integer tensor")
        if asset_index_by_env.numel() == 0 or bool((asset_index_by_env < 0).any()):
            raise ValueError("asset_index_by_env must be nonempty and nonnegative")

        # 静态路由仅在初始化时复制全N轴；Python tuple 不会别名调用者可变的输入tensor。
        self._assets_by_env = tuple(asset_index_by_env.detach().cpu().tolist())  # [N] 原始资产索引，不再稠密重编号。
        self._assets = tuple(sorted(set(self._assets_by_env)))  # 实际存在的资产集合，供组筛选和恢复验证。
        self.num_envs = len(self._assets_by_env)  # N，所有必需快照字段的精确轴长度。
        device = asset_index_by_env.device  # 初始化去重位使用资产metadata所在设备。
        self._seen = torch.zeros(self.num_envs, dtype=torch.bool, device=device)  # 当前回合是否已结算。
        self._windows: dict[int, deque[tuple[float, bool, int]]] = {
            asset: deque(maxlen=self.max_episodes_per_asset) for asset in self._assets
        }  # 每资产最多M个(net,safe,结算时policy_version)，不含删失回合。
        self._censored_early_timeouts = dict.fromkeys(self._assets, 0)  # 每资产累计删失数，不随deque淘汰。

    def _check_snapshot(self, snapshot: Mapping[str, torch.Tensor]) -> torch.device:
        r"""验证七个[N]字段及30秒时窗语义；所有实数检查在输入设备上归约。"""
        if not isinstance(snapshot, Mapping) or not set((*_FLOAT_FIELDS, *_BOOL_FIELDS)).issubset(snapshot):
            raise ValueError("snapshot is missing required first30 fields")
        device = None  # 必需字段必须位于同一设备，避免隐式搬运全环境快照。
        for name in (*_FLOAT_FIELDS, *_BOOL_FIELDS):
            value = snapshot[name]  # 只读取统计使用的字段，不遍历完整snapshot的其余大字段。
            if not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.shape != (self.num_envs,):
                raise ValueError(f"snapshot {name} must be a dense tensor with shape ({self.num_envs},)")
            if name in _FLOAT_FIELDS and not value.is_floating_point():
                raise TypeError(f"snapshot {name} must be floating point")
            if name in _BOOL_FIELDS and value.dtype != torch.bool:
                raise TypeError(f"snapshot {name} must have bool dtype")
            if device is not None and value.device != device:
                raise ValueError("all required snapshot fields must share the same device")
            device = value.device  # 原始物理事实的设备，不在此阶段搬到CPU。

        # command按elapsed>=30锁存完成位；此检查能拒绝把其他参考时长的字段误标为首30秒。
        duration = snapshot["episode_duration_s"]  # [N]，真实已经过秒数。
        valid = (duration >= 0) & (snapshot["first30_complete"] == (duration >= 30.0))  # 时长/完成位一致。
        for name in _FLOAT_FIELDS:
            valid &= torch.isfinite(snapshot[name])  # 未定义或污染的数值不能进入近期估计量。
        if not bool(valid.all()):
            raise ValueError("snapshot needs finite values, duration >= 0, first30_complete == (duration >= 30)")
        return duration.device  # 只传回设备身份，不复制任何全N轴数值到CPU。

    @torch.no_grad()
    def observe(self, snapshot: Mapping[str, torch.Tensor], policy_version: int) -> None:
        r"""结算首次完成/提前物理失败的窗口，并排除30秒前的纯timeout。

        Args:
            snapshot: 七个必需[N]字段，允许传入上游完整Mapping超集；每物理步调用一次。
            policy_version: 非负整数，记录本次结算时的策略版本，不代表整个轨迹的冻结版本。

        记 C 为first30_complete，F=drop OR axis，T为timeout，U=NOT seen。
        有效窗口 W=U AND (C OR F)，删失 Z=U AND T AND NOT(C OR F)。
        W 的净圈取 C ? net_turns_first30 : net_rotation_rad/(2π)，safe=C AND NOT F。
        更新 seen'=(seen OR W) AND NOT(F OR T)，因此done步仍先消费pre-reset事实。
        """
        _integer("policy_version", policy_version)  # 无效版本必须在任何统计状态写入之前拒绝。
        device = self._check_snapshot(snapshot)  # 全字段验证先于seen、deque和删失计数更新。
        seen = self._seen.to(device=device)  # 只迁移小型bool去重状态；通常首次接入快照时迁移一次。
        complete = snapshot["first30_complete"]  # C，窗口已经由上游锁存。
        failure = snapshot["termination_object_out_of_anchor"] | snapshot["termination_goal_axis_misaligned"]  # F。
        timeout = snapshot["termination_time_out"]  # T，纯早timeout属于删失。
        window = ~seen & (complete | failure)  # W，只结算当前回合第一次有效结果。
        censored = ~seen & timeout & ~complete & ~failure  # Z，物理失败和timeout同时出现优先计失败。
        event_envs = (window | censored).nonzero(as_tuple=False).flatten()  # [K]，按环境索引升序的新事件。

        # 先索引出K个事件，再形成小载荷；动态[N]实数与额外诊断字段不会整体跨设备搬运。
        if event_envs.numel():
            reached = complete[event_envs]  # [K]，本次结算是否已经到达30秒。
            full_net = snapshot["net_turns_first30"][event_envs].double()  # 只将事件行转FP64，保持输入数值精度。
            terminal_rad = snapshot["net_rotation_rad"][event_envs].double()  # [K]，保留实际有符号终点弧度。
            net = torch.where(reached, full_net, terminal_rad / (2.0 * math.pi))  # 早失败不作30/duration外推。
            safe = reached & ~failure[event_envs]  # [K]，恰在完成步发生drop/axis也必须unsafe。
            payload = torch.stack((net, safe.double(), censored[event_envs].double()), dim=1)  # [K,3]小事件载荷。
            host_envs = event_envs.cpu().tolist()  # 只复制K个索引，静态资产路由已在CPU。
            host_rows = payload.cpu().tolist()  # 只复制K个(net,safe,censored)，无tensor进入deque。
            for env, (net_value, safe_value, is_censored) in zip(host_envs, host_rows, strict=True):
                asset = self._assets_by_env[env]  # 原始资产索引，只参与统计分组。
                if is_censored:
                    self._censored_early_timeouts[asset] += 1  # 累计删失，不占deque容量或有效窗口分母。
                else:
                    record = (float(net_value), bool(safe_value), policy_version)  # 只有Python标量，不持有物理tensor。
                    self._windows[asset].append(record)  # 超过M时仅淘汰该资产最早的有效窗口。
        self._seen = (seen | window) & ~(failure | timeout)  # 结算之后再清done行，允许下一episode重新开始。

    def _selected_assets(self, asset_indices: Sequence[int] | None) -> tuple[int, ...]:
        r"""将统计组解释为已声明资产的集合；重复索引合并，空组合法，未知资产报错。"""
        if asset_indices is None:
            return self._assets  # 默认对全部实际存在资产统计，并明确报告有效覆盖数。
        if not isinstance(asset_indices, Sequence) or isinstance(asset_indices, (str, bytes)):
            raise TypeError("asset_indices must be a sequence of integer asset indices")
        selected = tuple(sorted({_integer("asset_index", asset) for asset in asset_indices}))  # 每资产只贡献一次权重。
        if not set(selected).issubset(self._windows):
            raise ValueError("asset_indices contains an unknown asset")
        return selected  # 原始索引不进行再次稠密映射，适合LEAP/Allegro显式分组。

    def summary(self, asset_indices: Sequence[int] | None = None) -> dict[str, int | float | None]:
        r"""汇总指定资产的近期训练窗口；无有效窗口时均值/比例为None，计数为零。

        设所选资产中有窗口的集合为 A_v，各资产deque长度为 m_a，净圈为 x_ai：
        $$\mu_{asset}=\frac1{|A_v|}\sum_{a\in A_v}\frac1{m_a}\sum_i x_{ai},\quad
        \mu_{pool}=\frac{\sum_{a,i}x_{ai}}{\sum_a m_a},\quad
        f_{safe}=\frac{\sum_{a,i}safe_{ai}}{\sum_a m_a}.$$

        assets_with_window报告有效资产覆盖；first30_episode_count是当前deque中的
        有效窗口总数。没有窗口的资产不当作零能力加入均值。censored_early_timeouts
        是指定组自初始化/恢复以来的累计删失数，不属于deque或上述任何分母。

        这些量描述跨策略版本的随机训练窗口；不能作为冻结checkpoint/R16中位数。
        """
        selected = self._selected_assets(asset_indices)  # 组是资产集合，而不是可重复加权的索引列表。
        active = [self._windows[asset] for asset in selected if self._windows[asset]]  # A_v，只包含有窗口的资产。
        count = sum(len(windows) for windows in active)  # sum_a m_a，当前有效窗口池大小。
        censored = sum(self._censored_early_timeouts[asset] for asset in selected)  # 分组累计删失计数。
        asset_mean = None  # 没有数据时未定义，不能显示为“能力零”。
        episode_mean = None  # 同上，窗口池均值未定义。
        safe_fraction = None  # 无有效分母时安全比例未定义。
        if count:
            means = [math.fsum(net for net, _, _ in windows) / len(windows) for windows in active]  # 各资产近期均值。
            asset_mean = math.fsum(means) / len(active)  # 主指标：先求每资产均值，再资产等权。
            episode_mean = math.fsum(net for windows in active for net, _, _ in windows) / count  # 诊断用池均值。
            safe_count = sum(safe for windows in active for _, safe, _ in windows)  # 同一有效窗口池内的安全完成数。
            safe_fraction = safe_count / count  # 池比例，不是先按资产求比例再平均。
        return {  # 只返回Python int/float/None，可直接供终端或结构化记录消费。
            "assets_with_window": len(active),  # 所选组中真正有首30秒结果的资产数。
            "first30_episode_count": count,  # 当前所有被保留有效窗口数，不是终止次数。
            "first30_asset_mean_net": asset_mean,  # 资产等权平均净圈。
            "first30_episode_mean_net": episode_mean,  # 窗口池平均净圈，仅用于诊断样本权重差异。
            "first30_safe_fraction": safe_fraction,  # 当前窗口池的安全完成比例。
            "censored_early_timeouts": censored,  # 累计的纯早timeout删失事件数。
        }

    @torch.no_grad()
    def reset_streams(self, env_ids: torch.Tensor | None = None) -> None:
        r"""重新创建物理环境时只清指定副本seen；保留过去窗口、版本与累计删失。

        Args:
            env_ids: 一维整数副本索引；None清全部，空子集无操作，重复索引幂等。
                必须位于[0,N)，不接受bool mask、负索引或浮点截断。
        """
        if env_ids is None:
            self._seen.zero_()  # 全部新物理流均允许下一回合结算，历史统计不变。
            return  # 不为在途被重建的回合伪造失败或timeout统计。
        if not isinstance(env_ids, torch.Tensor) or env_ids.ndim != 1 or env_ids.layout != torch.strided:
            raise ValueError("env_ids must be a dense one-dimensional integer tensor")
        if env_ids.dtype not in _INTEGER_DTYPES:
            raise TypeError("env_ids must have an integer dtype, not a bool mask")
        if not bool(((env_ids >= 0) & (env_ids < self.num_envs)).all()):
            raise ValueError(f"env_ids must lie in [0,{self.num_envs})")
        indices = env_ids.to(device=self._seen.device, dtype=torch.long)  # 只搬运指定索引到seen所在设备。
        self._seen.index_fill_(0, indices, False)  # 所有索引验证通过后一次性清除，重复/空索引合法。

    def state_dict(self) -> dict[str, Any]:
        r"""返回独立的小状态快照：精确资产映射、窗口配置、seen、Python窗口及删失计数。"""
        windows = {asset: list(windows) for asset, windows in self._windows.items()}  # 独立小列表，内含不可变标量。
        return {  # 没有物理快照或学习图；CPU状态可在不同快照设备之间恢复。
            "version": 1,  # 首30秒训练统计的状态布局版本。
            "max_episodes_per_asset": self.max_episodes_per_asset,  # 每资产近期窗口宽度M。
            "asset_index_by_env": torch.tensor(self._assets_by_env, dtype=torch.long, device="cpu"),  # 冻结副本归属。
            "seen": self._seen.detach().cpu().clone(),  # checkpoint时保存完整[N]去重位，不在observe逐步搬运。
            "windows": windows,  # 每资产按结算时间排序的近期窗口。
            "censored_early_timeouts": dict(self._censored_early_timeouts),  # 累计删失的独立Python整数副本。
        }

    @torch.no_grad()
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        r"""完整验证后恢复小状态，严格要求资产到副本映射与每资产窗口配置相同。

        seen可从CPU checkpoint恢复到当前设备。窗口长度超出M、非法标量、负计数或
        资产集合不匹配均拒绝，不借deque自动截断来隐藏配置或数据损坏。
        """
        required = {  # checkpoint同时恢复统计内容、配置和当前回合去重状态。
            "version",  # 状态布局版本。
            "max_episodes_per_asset",  # 每资产窗口宽度M。
            "asset_index_by_env",  # 精确副本归属。
            "seen",  # 当前回合已结算标志。
            "windows",  # Python近期窗口。
            "censored_early_timeouts",  # 累计删失计数。
        }
        if not isinstance(state, Mapping) or set(state) != required:
            raise ValueError("metrics checkpoint fields do not match")
        if type(state["version"]) is not int or state["version"] != 1:
            raise ValueError("metrics checkpoint version does not match")
        if _integer("max_episodes_per_asset", state["max_episodes_per_asset"], 1) != self.max_episodes_per_asset:
            raise ValueError("metrics checkpoint window configuration does not match")

        # 先验证映射与去重位的shape/dtype，GPU→CPU仅发生在显式checkpoint恢复阶段。
        for name, dtype in (("asset_index_by_env", torch.long), ("seen", torch.bool)):
            value = state[name]  # 两个[N]的小状态数组。
            if not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.shape != (self.num_envs,):
                raise ValueError(f"metrics checkpoint {name} must have shape ({self.num_envs},)")
            if value.dtype != dtype:
                raise TypeError(f"metrics checkpoint {name} must have dtype {dtype}")
        if tuple(state["asset_index_by_env"].detach().cpu().tolist()) != self._assets_by_env:
            raise ValueError("metrics checkpoint asset mapping does not match")
        for name in ("windows", "censored_early_timeouts"):
            values = state[name]  # 每种实际资产都必须显式出现，不能漏掉零计数资产。
            if (
                not isinstance(values, Mapping)  # 资产到状态的显式映射。
                or set(values) != set(self._assets)  # 不能丢失或引入任何资产。
                or any(type(asset) is not int for asset in values)  # bool等不能伪装成资产索引。
            ):
                raise ValueError(f"metrics checkpoint {name} asset set does not match")

        # 在临时Python容器中完成所有窗口与计数验证，失败时本对象保持原样。
        windows = {}  # 已验证的新deque；总规模至多A*M个三标量记录。
        censored = {}  # 已验证的累计删失数，不从窗口长度反推。
        for asset in self._assets:
            records = state["windows"][asset]  # 单个资产按时间升序保存的近期窗口列表。
            if not isinstance(records, list) or len(records) > self.max_episodes_per_asset:
                raise ValueError("metrics checkpoint window list exceeds configured capacity or has wrong type")
            parsed = []  # 验证全部记录后才交给有maxlen的deque。
            for record in records:
                if not isinstance(record, (tuple, list)) or len(record) != 3:
                    raise ValueError("metrics checkpoint window must contain (net, safe, policy_version)")
                net, safe, policy_version = record  # 严格的Python标量合同。
                if type(net) is not float or not math.isfinite(net) or type(safe) is not bool:
                    raise ValueError("metrics checkpoint window needs a finite Python float and bool safe flag")
                _integer("policy_version", policy_version)  # 版本是结算时的非负整数身份。
                parsed.append((net, safe, policy_version))  # 新tuple避免别名外部可变记录列表。
            windows[asset] = deque(parsed, maxlen=self.max_episodes_per_asset)  # 长度已验证，不会隐式丢记录。
            count = state["censored_early_timeouts"][asset]  # 该资产的累计删失次数。
            censored[asset] = _integer("censored_early_timeouts", count)  # Python整数且>=0。
        seen = state["seen"].detach().to(device=self._seen.device, copy=True)  # 最后准备自己的去重状态副本。
        self._seen = seen  # 全部验证成功后统一替换，保留正确的回合去重状态。
        self._windows = windows  # 独立deque，不别名传入checkpoint。
        self._censored_early_timeouts = censored  # 独立累计删失字典。
