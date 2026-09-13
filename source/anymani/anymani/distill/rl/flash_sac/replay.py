r"""FlashSAC 的紧凑同步回放：只保存动态单帧，在采样时恢复时序状态。

记 N 为同步环境数，T=floor(capacity_transitions/N) 为时间槽数，S 为已 append
的绝对 vector-step 数。绝对 transition s 写入槽 s mod T，包含 current_s、动作、
原始奖励、两种结束标志和显式 next_s。next_s 是 post-physics/pre-reset 状态；
即使其概念时间为 s+1，也不能用 auto-reset 后的 current_{s+1} 替代。

History30 的物理合同是 oldest-to-latest 且含当前帧。每个历史段首次 append 将首帧
重复填满 L；因此历史索引为 max(s-L+1+j, episode_start_s)，j=0,...,L-1。
episode_start 同时描述初始段、物理 done 后的新回合和 reset_streams 发起的新段。
显式流重启用于重新创建物理环境：旧段 n-step 在边界前停止，仍从旧转移的显式
next bootstrap；只有真实 terminated/truncated 才将 bootstrap 权重置零。
历史、geometry_tokens、静态图、joint limits 均不进入持久 transition 数组。
资产索引仅用于采样/路由 metadata；不将身份标签添加为 Actor 的观察输入。

每条 transition 的持久张量恰为 1800 B：两端各 210 个 FP32 与 21 个 bool，
加 16 个 FP32 动作、一个 FP32 原始奖励、两个 bool 和一个 int64 历史段起点。
另有 32N+8A B 的路由与下一段起点 metadata；待重置状态也在该起点中表达。
L 只影响临时采样 batch 内存。
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import torch

# 这是紧凑动态状态的布局 metadata，不是需要拼接进 Actor 的额外观察。
DYNAMIC_SHAPES: dict[str, tuple[int, ...]] = {
    "actor_jnt_current": (16, 5),  # JOINT 当前五通道帧，是重建历史的唯一持久帧源。
    "actor_owner_contact": (21, 1),  # 精确二元 owner 接触；持久 bool、返回 FP32。
    "critic_jnt_state": (16, 4),  # Critic 特权关节状态，FP32。
    "critic_owner_contact": (21, 2),  # Critic 接触量包含实数通道，不能整体压为 bool。
    "critic_obj": (1, 15),  # 物体动态状态，FP32。
    "critic_task": (1, 8),  # 有限时域任务动态状态，FP32。
    "critic_reward_release": (1,),  # 奖励释放动态量，FP32。
}
_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)  # 无浮点截断。


def _positive_integer(name: str, value: int) -> int:
    r"""检查计数参数，拒绝 bool、浮点自动取整与非正容量。"""
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value  # 保留调用者明确给出的整数，不做隐式修正。


def _check_tensor(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> None:
    r"""核对真实 tensor 与精确轴布局；禁止 PyTorch copy 的隐式广播。"""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.layout != torch.strided or tuple(value.shape) != shape:
        raise ValueError(f"{name} expected dense shape {shape}, got {tuple(value.shape)} / {value.layout}")


def _check_fp32(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> None:
    r"""实数动态量保持 FP32 与有限数值，禁止写入时隐式降精度或污染回报。"""
    _check_tensor(name, value, shape)  # 形状失败必须早于任何持久 copy。
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must be FP32, got {value.dtype}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


class CompactReplay:
    r"""按同步 [T,N,...] 布局保存动态 transition，并进行严格逐资产均衡抽样。

    默认 CPU、L=30、n=3、gamma=.99。device 指持久池及返回 batch 的设备；
    GPU collector 可将输入复制到 CPU 池，append 返回时复制已经完成。
    每种资产拥有相同数量的环境副本；任意整数标签按升序映射为 [0,A) 索引。

    保守成熟窗口定义为：oldest=max(0,S-T)，上界 S-n；若 oldest=0，下界为零，
    否则下界为 oldest+L-1。物理结束或历史段边界截断回报，但不提前放宽成熟窗口。
    T>=L+n-1 才能保证覆盖后的稳态可采性；更小池仍按请求分配，由 ready 报告。

    Args:
        capacity_transitions: 请求的 transition 数，实际为 floor(capacity/N)*N。
        num_envs: 每个同步 vector-step 的 N 个环境，必须为正整数。
        asset_ids_by_env: [N] 整数资产标签，允许非连续标签与交错副本排列。
        history_steps: oldest-to-latest 历史长度 L，含当前帧，默认 30。
        n_step: 回报最多累积的 transition 数 n，默认 3。
        gamma: [0,1] 原始奖励折扣，默认 .99。
        device: 池和返回 tensor 的设备，默认 'cpu'。
        pin_memory: 是否为 CPU transition 数组分配锁页内存，默认 False。

    Notes:
        该对象不负责 collector 与 learner 的并发互斥；append/reset_streams/gather/checkpoint
        应由调用者按序使用。state_dict 返回共享 tensor 引用，须在下一次写入前序列化。
    """

    def __init__(
        self,
        capacity_transitions: int,
        num_envs: int,
        asset_ids_by_env: torch.Tensor,
        history_steps: int = 30,
        n_step: int = 3,
        gamma: float = 0.99,
        device: torch.device | str = "cpu",
        pin_memory: bool = False,
    ) -> None:
        r"""先验证时序与资产布局，再分配完整 vector-step 的紧凑数组。"""
        self.num_envs = _positive_integer("num_envs", num_envs)  # 同步环境轴 N。
        self.capacity_transitions = _positive_integer("capacity_transitions", capacity_transitions)  # 请求行数。
        self.history_steps = _positive_integer("history_steps", history_steps)  # 含当前帧的历史长度 L。
        self.n_step = _positive_integer("n_step", n_step)  # 完整未来需要 n 行。
        self.capacity_steps = capacity_transitions // num_envs  # T=floor(C_requested/N)。
        self.actual_capacity = self.capacity_steps * num_envs  # 实际 transition 数 C=T*N。
        if self.capacity_steps == 0:
            raise ValueError("capacity_transitions must fit at least one complete vector-step")
        if isinstance(gamma, bool) or not isinstance(gamma, Real) or not math.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError(f"gamma must be finite and in [0,1], got {gamma!r}")
        if type(pin_memory) is not bool:
            raise TypeError("pin_memory must be bool")
        self.gamma = float(gamma)  # 保存原始折扣配置，不预缩放奖励。
        self.device = torch.device(device)  # 只在调用者明确指定时选择非 CPU 设备。
        self.pin_memory = pin_memory  # 锁页影响搬运方式，不改变 tensor dtype。
        if pin_memory and self.device.type != "cpu":
            raise ValueError("pin_memory requires a CPU replay device")

        # 资产路由使用独立副本，collector 重用输入 metadata 时不会改写池的身份。
        _check_tensor("asset_ids_by_env", asset_ids_by_env, (num_envs,))  # [N]。
        if asset_ids_by_env.dtype not in _INTEGER_DTYPES:
            raise TypeError("asset_ids_by_env must have an integer dtype")
        self.asset_ids_by_env = asset_ids_by_env.detach().to(device=self.device, dtype=torch.long, copy=True)
        self.device = self.asset_ids_by_env.device  # 规范化 cpu:0 / cuda 的实际设备编号。
        ids, inverse, counts = torch.unique(self.asset_ids_by_env, sorted=True, return_inverse=True, return_counts=True)
        if not bool((counts == counts[0]).all()):
            raise ValueError("all assets must have the same number of environment replicas")
        self.asset_ids = ids  # [A]，asset_index 到原始标签的升序字典。
        self.asset_index_by_env = inverse  # [N]，每个环境的稠密资产索引。
        self.num_assets = ids.numel()  # A 从 metadata 推断。
        self.replicas_per_asset = num_envs // self.num_assets  # R=N/A，各资产相同。
        ordered_envs = torch.argsort(inverse, stable=True)  # 按资产分组，同资产内保留原环境顺序。
        self.environments_by_asset = ordered_envs.reshape(self.num_assets, self.replicas_per_asset)  # [A,R]。

        # 每端只分配动态单帧；zeros 使尚未写入的 checkpoint 槽也有确定且有限的内容。
        self._current = {}  # 每个 value 为 [T,N,*dynamic_shape]。
        self._next_observation = {}  # 显式物理终点，不能省略或替换为下一行 current。
        leading = (self.capacity_steps, num_envs)  # 同步时间轴与环境轴。
        for name, shape in DYNAMIC_SHAPES.items():
            dtype = torch.bool if name == "actor_owner_contact" else torch.float32  # 只有二元接触压缩。
            self._current[name] = self._zeros((*leading, *shape), dtype)  # current_s。
            self._next_observation[name] = self._zeros((*leading, *shape), dtype)  # pre-reset next_s。
        self._actions = self._zeros((*leading, 16), torch.float32)  # 原始动作 [T,N,16]。
        self._rewards = self._zeros(leading, torch.float32)  # 原始、未折扣奖励 [T,N]。
        self._terminated = self._zeros(leading, torch.bool)  # 真终止 [T,N]。
        self._truncated = self._zeros(leading, torch.bool)  # 有限时域 timeout [T,N]。
        self._episode_start = self._zeros(leading, torch.long)  # 每个 current 所属历史段的绝对起点。
        self._next_episode_start = self._zeros((num_envs,), torch.long)  # 下一 append 的起点，包含待流重启状态。
        self._total_steps = 0  # S：所有 append 次数，不因预热、覆盖或 reset 而归零。

    def _zeros(self, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        r"""分配属于回放的独立连续数组；pin_memory 只用于 CPU 持久 transition。"""
        return torch.zeros(shape, dtype=dtype, device=self.device, pin_memory=self.pin_memory)  # 精确布局。

    @property
    def total_transitions(self) -> int:
        r"""返回全部已 append 的 S*N 条 transition，包含预热与已经被覆盖的行。"""
        return self._total_steps * self.num_envs  # 不能用当前池占用量代替采集预算。

    def _mature_bounds(self) -> tuple[int, int]:
        r"""返回共同合法的绝对起点闭区间 [lower,upper]；lower>upper 表示未成熟。"""
        oldest = max(0, self._total_steps - self.capacity_steps)  # 仍在环中的最早绝对 transition。
        lower = oldest + self.history_steps - 1 if oldest else 0  # 覆盖后保守排除 L-1 个起点。
        return lower, self._total_steps - self.n_step  # s+n-1 <= S-1，显式终点无需再等 current_{s+n}。

    @property
    def ready(self) -> bool:
        r"""是否每种资产都存在上下文和完整 n-step 未来；同步布局共享同一合法时间窗。"""
        lower, upper = self._mature_bounds()  # 物理 done/流边界只截断 target，不改变成熟支持集。
        return lower <= upper  # 每资产至少一个副本，因此存在起点即可严格均衡抽样。

    @torch.no_grad()
    def reset_streams(self, environment_indices: torch.Tensor | None = None) -> None:
        r"""让指定副本的下一次 append 开始新历史段，供重新创建物理环境后的恢复使用。

        Args:
            environment_indices: 一维整数环境索引，必须位于 [0,N)；None 表示全部环境。
                重复索引幂等，空整数 tensor 表示无操作，不接受 bool mask。

        设当前已采集 S 行，将所选副本的 next_episode_start 设为 S。该已有 [N]
        int64 数组直接保存 pending reset 状态，因此 state_dict/load_state_dict
        可以在新初态尚未 append 时完整恢复。重复调用只合并待重置的副本集合。

        下一次 append 将 current_S 标为新段首帧，历史不足 L 时重复该首帧；已有
        transition、绝对计数、物理结束标志均不改动。旧段跨边界的 target 使用旧段
        最后一条 transition 的显式 next；无真实 done 时保留 gamma^实际步数 的 bootstrap 权重。
        """
        if environment_indices is None:
            self._next_episode_start.fill_(self._total_steps)  # 所有副本在下一绝对步 S 开新段。
            return  # 只改 pending metadata，没有新增 transition。

        # 先完整核对索引再一次性标记，防止无效子集对一部分环境产生恢复副作用。
        envs = self._indices("environment_indices", environment_indices, allow_empty=True)  # 合法空集合也保留语义。
        if not bool(((envs >= 0) & (envs < self.num_envs)).all()):
            raise ValueError(f"environment_indices must lie in [0,{self.num_envs})")
        self._next_episode_start.index_fill_(0, envs, self._total_steps)  # 指定副本的新段首帧将是 current_S。

    def _check_observation(self, name: str, observation: dict[str, torch.Tensor]) -> None:
        r"""检查精确动态字段集、轴、FP32 与二元接触，拒绝历史或静态大字段。"""
        if not isinstance(observation, dict) or observation.keys() != DYNAMIC_SHAPES.keys():
            raise ValueError(f"{name} fields must be exactly {tuple(DYNAMIC_SHAPES)}")
        for field, shape in DYNAMIC_SHAPES.items():
            value = observation[field]  # 当前待检查的动态物理量。
            expected = (self.num_envs, *shape)  # 第一轴必须是完整同步 N。
            if field != "actor_owner_contact":
                _check_fp32(f"{name}.{field}", value, expected)  # 不隐式降精度。
            else:
                _check_tensor(f"{name}.{field}", value, expected)  # binary 检查前先验证 tensor。
                if value.dtype not in (torch.bool, torch.float32, *_INTEGER_DTYPES):
                    raise TypeError(f"{name}.{field} must be bool, FP32, or integer binary contact")
                if not bool(((value == 0) | (value == 1)).all()):
                    raise ValueError(f"{name}.{field} must be exactly binary (0 or 1)")

    @torch.no_grad()
    def append(
        self,
        current: dict[str, torch.Tensor],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        next_observation: dict[str, torch.Tensor],
    ) -> None:
        r"""复制一行同步 transition，严格保留原始奖励和 reset 之前的物理终点。

        Args:
            current: 仅含 DYNAMIC_SHAPES，各字段第一轴为 N。
            actions: FP32 [N,16]，对应 current 上执行的动作。
            rewards: FP32 [N] 原始奖励；不在存储时折扣、裁剪或归一化。
            terminated: bool [N] 真终止标志。
            truncated: bool [N] 有限时域 timeout；同样禁止 bootstrap。
            next_observation: 仅含 DYNAMIC_SHAPES 的 post-physics/pre-reset 终点。

        全部输入验证通过后才写池。复制为阻塞式且不保留 autograd 图，调用者可在
        返回后立即复用采集 tensor。初始 episode_start=0；上一行 done 或显式
        reset_streams 指定的环境在本行使用新段起点，其他副本继续原历史段。
        """
        self._check_observation("current", current)  # 验证全部 current 字段。
        self._check_observation("next_observation", next_observation)  # 终点也执行同样严格验证。
        _check_fp32("actions", actions, (self.num_envs, 16))  # 禁止动作广播或降精度。
        _check_fp32("rewards", rewards, (self.num_envs,))  # raw reward 形状严格为 [N]。
        for name, value in (("terminated", terminated), ("truncated", truncated)):
            _check_tensor(name, value, (self.num_envs,))  # 两种结束标志使用同一环境轴。
            if value.dtype != torch.bool:
                raise TypeError(f"{name} must have bool dtype")

        # s mod T 是唯一写槽；所有 copy 完成前不推进绝对 cursor 或回合状态。
        slot = self._total_steps % self.capacity_steps  # 当前绝对 transition 的环槽。
        for name in DYNAMIC_SHAPES:
            self._current[name][slot].copy_(current[name])  # GPU 输入到 CPU 也等待复制完成。
            self._next_observation[name][slot].copy_(next_observation[name])  # 显式物理终点。
        self._actions[slot].copy_(actions)  # [N,16]，阻塞复制且断开外部梯度。
        self._rewards[slot].copy_(rewards)  # [N]，原始奖励数值不变。
        self._terminated[slot].copy_(terminated)  # 真终止。
        self._truncated[slot].copy_(truncated)  # timeout。
        self._episode_start[slot].copy_(self._next_episode_start)  # 本行 current 的历史段起点，含待生效流重启。
        done = self._terminated[slot] | self._truncated[slot]  # 任一结束使下一行进入新回合。
        self._next_episode_start.masked_fill_(done, self._total_steps + 1)  # 新回合从下一次 append 起算。
        self._total_steps += 1  # 全部复制与 episode metadata 更新完成后，S 才增加。

    def sample(self, batch_size: int, *, generator: torch.Generator | None = None) -> dict[str, Any]:
        r"""每资产严格抽 B/A 个样本，有放回均匀选副本和合法绝对起点。

        Args:
            batch_size: 正整数且必须整除 A；不能静默补齐或丢弃样本。
            generator: 与池设备匹配的外部 Generator；其 state 由调用者独立保存。

        Returns:
            与 gather 完全相同的 batch，行按稠密 asset_index 分组。

        因同步布局和保守成熟窗不依赖环境，副本与时间可独立均匀抽样；
        P(e,s|a)=1/(R*K)，K=upper-lower+1。全过程不逐样本循环或拒绝重抽。
        """
        _positive_integer("batch_size", batch_size)  # 空 batch 不构成合法训练样本。
        if batch_size % self.num_assets:
            raise ValueError(f"batch_size must be divisible by num_assets={self.num_assets}")
        lower, upper = self._mature_bounds()  # 所有资产的共同支持集。
        if lower > upper:
            raise ValueError("replay has no mature starting step with complete history and n-step future")
        if generator is not None and torch.device(generator.device) != self.device:
            raise ValueError(f"generator device must match replay device {self.device}")

        # 每种资产确定性放入 B/A 行，只将其内部副本与时间作为随机变量。
        assets = torch.arange(self.num_assets, device=self.device).repeat_interleave(batch_size // self.num_assets)
        replicas = torch.randint(self.replicas_per_asset, (batch_size,), generator=generator, device=self.device)
        envs = self.environments_by_asset[assets, replicas]  # [B]，严格属于对应资产。
        sequences = torch.randint(lower, upper + 1, (batch_size,), generator=generator, device=self.device)
        return self.gather(envs, sequences)  # 确定性审计与随机采样共用唯一 target 构造路径。

    def _indices(self, name: str, values: torch.Tensor, *, allow_empty: bool = False) -> torch.Tensor:
        r"""核对一维整数索引；审计 batch 非空，流重启子集允许为空，均禁止浮点截断。"""
        if not isinstance(values, torch.Tensor) or values.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional tensor")
        if not allow_empty and values.numel() == 0:
            raise ValueError(f"{name} must be nonempty")
        if values.layout != torch.strided or values.dtype not in _INTEGER_DTYPES:
            raise TypeError(f"{name} must be a dense integer tensor")
        return values.to(device=self.device, dtype=torch.long, copy=True)  # [B]，返回值不别名调用者索引。

    def _history(self, envs: torch.Tensor, sequences: torch.Tensor) -> torch.Tensor:
        r"""向量化恢复 H_s[j]=current[max(s-L+1+j,episode_start_s)]，形状 [B,L,16,5]。"""
        starts = self._episode_start[sequences % self.capacity_steps, envs]  # 每个 current 的历史段起点 [B]。
        offsets = torch.arange(1 - self.history_steps, 1, device=self.device)  # oldest-to-latest，末位为 0。
        absolute = torch.maximum(sequences[:, None] + offsets, starts[:, None])  # reset 首帧重复补齐。
        return self._current["actor_jnt_current"][absolute % self.capacity_steps, envs[:, None]]  # [B,L,16,5]。

    def gather(self, environment_indices: torch.Tensor, sequence_indices: torch.Tensor) -> dict[str, Any]:
        r"""按 [B] 环境及绝对起点构造历史和 n-step target，允许重复、非均衡审计索引。

        记 p_s=episode_start_s。物理长度 k_phys 为首个 done 的位置加一；段长度
        k_seg 为第一个满足 p_{s+j} != p_s 的 j>=1；未遇到对应事件时各取 n。
        实际长度 k=min(n,k_phys,k_seg)，终点 transition 为 e=s+k-1。目标定义为：
        $$R_s=\sum_{j=0}^{k-1}\gamma^j r_{s+j},\quad d_s=\gamma^k(1-done_e).$$
        本项目固定 120 秒有限时域对照中，两种物理结束均使 d_s=0；仅因流重启
        而提前停止累加时，d_s=gamma^k，bootstrap 状态仍为旧段显式物理 next_e。
        next_history=[H_e[1:], explicit_next_e]，而不是 reset 之后的 H_{e+1}。

        Returns:
            obs、next_obs: 动态 FP32 字段及 actor_jnt_history [B,L,16,5]。
            actions [B,16]、rewards [B]、discounts [B]: 均为 FP32。
            terminated、truncated [B]: bool，取实际终点 e 的原始标志；流重启不伪造标志。
            steps [B]: int64 实际长度 k。
            asset_index、environment_index、sequence_index、next_sequence_index [B]:
                int64 metadata；资产为稠密索引，sequence 为绝对时间，next_sequence=e+1。
                遇到物理结束或流边界时，next_sequence 不授权读取该行新段的 current。

        Raises:
            ValueError/TypeError: 索引非法，或起点缺少上下文/完整 n 行未来。
        """
        envs = self._indices("environment_indices", environment_indices)  # [B]，副本索引。
        sequences = self._indices("sequence_indices", sequence_indices)  # [B]，绝对起点。
        if envs.shape != sequences.shape:
            raise ValueError("environment_indices and sequence_indices must have identical [B] shapes")
        if not bool(((envs >= 0) & (envs < self.num_envs)).all()):
            raise ValueError(f"environment_indices must lie in [0,{self.num_envs})")
        lower, upper = self._mature_bounds()  # 与 sample、ready 共用严格成熟合同。
        if not bool(((sequences >= lower) & (sequences <= upper)).all()):
            raise ValueError(f"sequence_indices must be mature absolute starts in [{lower},{upper}]")

        # [B,n] 时间网格同时定位物理结束和历史段边界，不逐样本或逐 n-step 循环。
        offsets = torch.arange(self.n_step, device=self.device)  # j=0,...,n-1。
        future_slots = (sequences[:, None] + offsets) % self.capacity_steps  # [B,n] 环形时间槽。
        future_envs = envs[:, None]  # [B,1] 广播同一副本，避免跨环境串线。
        done = self._terminated[future_slots, future_envs] | self._truncated[future_slots, future_envs]
        physical_steps = torch.where(done, offsets + 1, self.n_step).amin(dim=1)  # done 行奖励必须计入。
        starts = self._episode_start[future_slots, future_envs]  # [B,n]，每个候选 current 的历史段起点。
        segment_steps = torch.where(starts != starts[:, :1], offsets, self.n_step).amin(dim=1)  # 新段首行不得计入。
        steps = torch.minimum(physical_steps, segment_steps)  # k=min(k_phys,k_seg)，首列必属当前段，故 k>=1。
        endpoints = sequences + steps - 1  # [B]，真实终点 transition 的绝对时间 e。
        endpoint_slots = endpoints % self.capacity_steps  # 终点在环内的槽位。
        terminated = self._terminated[endpoint_slots, envs]  # 实际终点的真终止，流边界可能早于 n。
        truncated = self._truncated[endpoint_slots, envs]  # 同上，保留 timeout 原始标志。

        # mask 只保留当前物理/历史段的有效前缀，不把任何新段奖励加入旧 target。
        raw_rewards = self._rewards[future_slots, future_envs]  # [B,n] 原始奖励。
        active_rewards = torch.where(offsets < steps[:, None], raw_rewards, 0.0)  # j<k。
        gamma = torch.tensor(self.gamma, dtype=torch.float32, device=self.device)  # FP32 标量。
        rewards = (active_rewards * gamma.pow(offsets)).sum(dim=1)  # R_s=sum_{j<k} gamma^j r_{s+j}。
        discounts = gamma.pow(steps) * ~(terminated | truncated)  # d_s=gamma^k * ~done_e。

        # 返回网络观察时恢复 FP32 接触；资产身份与静态证据由调用者在观察之外路由。
        slots = sequences % self.capacity_steps  # 起点 current 与动作使用同一环槽。
        obs = {name: values[slots, envs].to(torch.float32) for name, values in self._current.items()}
        next_obs = {
            name: values[endpoint_slots, envs].to(torch.float32) for name, values in self._next_observation.items()
        }  # 每个字段均来自显式 next_e，不借用下一 append 的 current。
        obs["actor_jnt_history"] = self._history(envs, sequences)  # H_s，含当前帧。
        endpoint_history = self._history(envs, endpoints)  # H_e，与 n-step 终点对齐。
        next_frame = next_obs["actor_jnt_current"][:, None]  # [B,1,16,5]，pre-reset 物理终点。
        next_obs["actor_jnt_history"] = torch.cat((endpoint_history[:, 1:], next_frame), dim=1)  # 左移并追加。
        return {  # 所有索引及实际长度均为 [B] int64，结束标志为 [B] bool。
            "obs": obs,  # 当前状态及 H_s。
            "next_obs": next_obs,  # 显式物理终点及左移追加后的历史。
            "actions": self._actions[slots, envs],  # 起点执行的动作 [B,16]。
            "rewards": rewards,  # 原始折扣和 R_s。
            "discounts": discounts,  # bootstrap 权重 d_s。
            "terminated": terminated,  # 终点真终止。
            "truncated": truncated,  # 终点有限时域 timeout。
            "steps": steps,  # 实际累积 transition 数 k。
            "asset_index": self.asset_index_by_env[envs],  # 独立于 Actor 观察的资产路由。
            "environment_index": envs,  # 环境副本的原始索引。
            "sequence_index": sequences,  # 起点绝对时间 s。
            "next_sequence_index": endpoints + 1,  # 物理终点状态的概念绝对时间 e+1。
        }

    def _tensor_state(self) -> dict[str, Any]:
        r"""列出全部持久数组的共享引用，确保字节统计和 checkpoint 覆盖同一完整状态。"""
        return {  # 字典是新的容器，tensor 仍是本池的原有数组，不额外 clone。
            "current": dict(self._current),  # 起点动态帧 [T,N,...]。
            "next_observation": dict(self._next_observation),  # 显式物理终点 [T,N,...]。
            "actions": self._actions,  # 原始动作 [T,N,16]。
            "rewards": self._rewards,  # 原始奖励 [T,N]。
            "terminated": self._terminated,  # 真终止标志 [T,N]。
            "truncated": self._truncated,  # timeout 标志 [T,N]。
            "episode_start": self._episode_start,  # 每个 current 所属历史段绝对起点 [T,N]。
            "next_episode_start": self._next_episode_start,  # 下一 append 的段起点 [N]，含 pending 流重启。
            "asset_ids_by_env": self.asset_ids_by_env,  # 环境到原始资产标签 [N]。
            "asset_ids": self.asset_ids,  # 稠密索引到原始标签 [A]。
            "asset_index_by_env": self.asset_index_by_env,  # 环境到稠密资产索引 [N]。
            "environments_by_asset": self.environments_by_asset,  # 严格均衡采样的副本表 [A,R]。
        }

    @property
    def storage_bytes(self) -> int:
        r"""返回实际持久 tensor storage 字节数；不含 Python 容器、返回 batch 或调用者 RNG。"""
        storages = {}  # 按地址去重，统计底层真实 storage 而不是视图的逻辑 numel。
        for value in self._tensor_state().values():
            tensors = value.values() if isinstance(value, dict) else (value,)  # 两端动态字典与独立 metadata。
            for tensor in tensors:
                storage = tensor.untyped_storage()  # 实际 tensor 分配，包含元素 dtype 的真实宽度。
                storages[storage.data_ptr()] = storage.nbytes()  # 相同 storage 只计一次。
        return sum(storages.values())  # 1800*C + 32*N + 8*A，与 history_steps 无关。

    def state_dict(self) -> dict[str, Any]:
        r"""返回完整恢复状态，不复制整池；调用者须在下次 append/reset_streams 前完成序列化。

        包含精确布局、资产 metadata、绝对 cursor、每行与下一 append 的段起点，
        以及所有已写/未写槽的确定内容。next_episode_start[e]=S 表示该副本下次
        从 S 开新段，既可由物理 done，也可由 reset_streams 发起；这完整保存了
        未消费的流重启状态。外部 Generator 状态不属于本对象。
        """
        config = {  # 科学参数与内存布局均严格匹配，不隐式迁移容量、精度或历史长度。
            "capacity_transitions": self.capacity_transitions,  # 请求容量，保留实验配置原值。
            "actual_capacity": self.actual_capacity,  # 向下取整后的真实 transition 数。
            "num_envs": self.num_envs,  # N，决定同步行宽度。
            "history_steps": self.history_steps,  # L，决定重建的时序语义。
            "n_step": self.n_step,  # n，决定成熟窗口与 target 长度。
            "gamma": self.gamma,  # 原始奖励折扣。
            "device": str(self.device),  # 实际持久设备。
            "pin_memory": self.pin_memory,  # 恢复目标的内存分配方式。
            "dynamic_shapes": dict(DYNAMIC_SHAPES),  # 防止 ABI 同名字段被改 shape 后仍加载。
        }
        return {  # 标量 cursor 冗余提供交叉检查；无任何池 tensor 的 clone。
            "version": 1,  # 紧凑回放 checkpoint schema。
            "config": config,  # 严格匹配的布局与数值配置。
            "total_steps": self._total_steps,  # 已采集的绝对 vector-step 数 S。
            "write_index": self._total_steps % self.capacity_steps,  # 下一次写入的环槽。
            "total_transitions": self.total_transitions,  # 全部采集预算 S*N，含预热。
            **self._tensor_state(),  # 全部持久数组，与 storage_bytes 同源。
        }

    def _check_episode_state(self, state: dict[str, Any], total_steps: int) -> None:
        r"""验证历史段递推：物理 done 必须开新段，无 done 只可延续原段或在当前步显式开新段。"""
        pending = state["next_episode_start"]  # [N]，下一次写入的段起点，包含 pending 流重启。
        if total_steps == 0:
            if bool((pending != 0).any()):
                raise ValueError("empty checkpoint must have next_episode_start=0")
            return  # 未写入行没有物理时间语义。

        # 只形成小型时间/episode metadata 网格，不复制占主要内存的动态特征。
        oldest = max(0, total_steps - self.capacity_steps)  # checkpoint 中仍存在的绝对起点。
        absolute = torch.arange(oldest, total_steps, device=self.device)  # [K]，K<=T。
        slots = absolute % self.capacity_steps  # 按物理先后顺序排列环槽。
        starts = state["episode_start"][slots]  # [K,N]，绝对历史段起点。
        done = state["terminated"][slots] | state["truncated"][slots]  # [K,N]，任一结束均 reset。
        if not bool(((starts >= 0) & (starts <= absolute[:, None])).all()):
            raise ValueError("checkpoint episode_start must lie between zero and its current absolute step")
        if oldest == 0 and bool((starts[0] != 0).any()):
            raise ValueError("checkpoint initial episode_start must be zero")
        # 没有物理 done 的边界可由 reset_streams 创建，但新首帧必须就是当前绝对步。
        expected = torch.where(done[:-1], absolute[1:, None], starts[:-1])  # 仅按物理生命周期推导的段起点。
        valid = (starts[1:] == expected) | (starts[1:] == absolute[1:, None])  # 延续原段或合法显式新段。
        if not bool(valid.all()):
            raise ValueError("checkpoint episode_start is inconsistent with termination/reset order")
        expected_pending = torch.where(done[-1], total_steps, starts[-1])  # 仅按最后一条物理转移推导。
        if not bool(((pending == expected_pending) | (pending == total_steps)).all()):
            raise ValueError("checkpoint next_episode_start is inconsistent with the last transition")

    @torch.no_grad()
    def load_state_dict(self, state: dict[str, Any]) -> None:
        r"""完整验证后复制 checkpoint，严格恢复资产路由、绝对时间、历史段和待流重启状态。

        只接受与本对象一致的版本、字段、配置、设备、dense 连续 shape/dtype。
        加载源的锁页属性不作要求；恢复后使用本对象构造时分配的存储。
        任一验证失败均发生在持久 copy 前；成功加载后不与传入整池建立 storage 别名。
        """
        expected = self.state_dict()  # 仅共享引用与小字典，无整池快照。
        if not isinstance(state, dict) or state.keys() != expected.keys():
            raise ValueError("checkpoint fields do not match CompactReplay state")
        if type(state["version"]) is not int or state["version"] != 1 or state["config"] != expected["config"]:
            raise ValueError("checkpoint version/config does not match replay layout")
        total_steps = state["total_steps"]  # S，必须是非负绝对 vector-step 计数。
        if type(total_steps) is not int or total_steps < 0:
            raise ValueError("checkpoint total_steps must be a nonnegative integer")
        for name, value in (  # 冗余 cursor 标量必须由同一个绝对时间 S 导出。
            ("write_index", total_steps % self.capacity_steps),  # 环槽 S mod T。
            ("total_transitions", total_steps * self.num_envs),  # 采集行数 S*N。
        ):
            if type(state[name]) is not int or state[name] != value:
                raise ValueError(f"checkpoint {name} is inconsistent with total_steps")

        # 先收集已验证的源/目的张量对；所有数组检查通过后才进入 copy 阶段。
        copies = []  # 仅 tensor 引用对，不额外分配动态池。
        for name, target in self._tensor_state().items():
            source = state[name]  # 同一 checkpoint key 的源数组或动态字典。
            if isinstance(target, dict):
                if not isinstance(source, dict) or source.keys() != target.keys():
                    raise ValueError(f"checkpoint {name} fields do not match compact observation")
                pairs = [(f"{name}.{field}", source[field], tensor) for field, tensor in target.items()]  # 固定七字段。
            else:
                pairs = [(name, source, target)]  # 单个数值/metadata 数组。
            for field, value, tensor in pairs:
                _check_tensor(field, value, tuple(tensor.shape))  # 精确 [T,N,...] 布局。
                if value.dtype != tensor.dtype or value.device != tensor.device or not value.is_contiguous():
                    raise ValueError(f"checkpoint {field} must match dtype/device/contiguous layout")
                if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                    raise ValueError(f"checkpoint {field} must contain only finite values")
                copies.append((tensor, value))  # 留待验证全部完成后，复制进入自己的 storage。
        # 身份必须逐元素相等，不能只比较资产数/副本数后重解释既有动态样本。
        for name in ("asset_ids_by_env", "asset_ids", "asset_index_by_env", "environments_by_asset"):
            if not torch.equal(state[name], expected[name]):
                raise ValueError(f"checkpoint {name} does not match asset metadata")
        self._check_episode_state(state, total_steps)  # 检查时间边界与 done 的可解释性。
        for target, source in copies:
            target.copy_(source)  # 阻塞复制，恢复后不保留外部 checkpoint storage 的别名。
        self._total_steps = total_steps  # 所有内容复制完成后才更新绝对 cursor。
