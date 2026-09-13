r"""逐环境、按组件启用的初态ADR；首个组件只随机化物体掌面位置。

沿用GM/inhand的25档幅度尺度：w(k)=0.01*k/25 m。max_level仅限制可到达档位，
不改变插值分母。第1档半宽0.4mm，第5档2mm。等级属于环境的课程状态，绝不作为Actor特征。
每回合只在reset采样一次位置偏移，并由后续command reset捕获新的位置参考。

升降证据使用每个环境自己的首30秒实际净圈；提前失败不按短时长放大成绩。
至少3个本档回合后，EMA>=1圈且最近首30秒安全可升1档，EMA<0.5圈可退1档。
分数在2圈饱和；该课程是鲁棒性软开关，不奖励超过2圈的额外难度进展。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .task_math import quaternion_apply_wxyz


@dataclass
class ObjectPositionAdrCfg:
    r"""位置组件的幅度谱与独立环境调度；长度单位m，参考时间s。"""

    enabled: bool = False  # 未启用时不产生随机数或状态变化。
    reference_levels: int = 25  # 原幅度谱分母，与最高允许档位分离。
    max_level: int = 5  # 本轮最多开放原谱前5档。
    initial_level: int = 1  # 从原第1档0.4mm轻扰动开始；弱环境可以退回0。
    full_half_width_m: float = 0.01  # 原第25档的位置半宽。
    reference_seconds: float = 30.0  # 对齐1圈/30s的主任务窗口。
    promote_turns: float = 1.0  # 在当前扰动下能够完成一圈才增难。
    demote_turns: float = 0.5  # 保守回退边界，避免弱环境被困在高档。
    score_cap_turns: float = 2.0  # 2圈以上不加快晋档。
    ema_alpha: float = 0.5  # 每个环境按自身回合更新，首样本无零初始化偏差。
    min_episodes: int = 3  # 独立环境的本档回合数，不是全局reset-hook次数。

    def __post_init__(self) -> None:
        r"""拒绝错误档位、单位或没有滞回区间的调度配置。"""
        if not 0 <= self.initial_level <= self.max_level <= self.reference_levels or self.reference_levels < 1:
            raise ValueError('ADR levels must preserve 0 <= initial <= max <= reference')
        if not 0 < self.ema_alpha <= 1 or self.min_episodes < 1:
            raise ValueError('ADR requires a positive episode window and valid EMA alpha')
        if not 0 <= self.demote_turns < self.promote_turns <= self.score_cap_turns:
            raise ValueError('ADR turn thresholds need a nonempty hysteresis interval')
        if self.full_half_width_m < 0 or self.reference_seconds <= 0:
            raise ValueError('ADR position/time ranges are invalid')


@dataclass
class HeterogeneousAdrCfg:
    r"""嵌套组件配置；只声明已经实现的位置组件，不建立空的随机化模块。"""

    object_position: ObjectPositionAdrCfg = field(default_factory=ObjectPositionAdrCfg)


class PerEnvironmentPositionAdr:
    r"""每个环境独立持有档位、回合EMA和已执行扰动。"""

    def __init__(self, cfg: ObjectPositionAdrCfg, num_envs: int, device: torch.device | str):
        self.cfg = cfg  # 对象为已验证的组件配置。
        self.level = torch.full((num_envs,), cfg.initial_level, dtype=torch.long, device=device)
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.ema = torch.zeros(num_envs, device=device)  # 各环境独立的30秒净圈分数。
        self.trials = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.offset_h = torch.zeros(num_envs, 3, device=device)  # 本回合实际掌面扰动，z始终0。

    @property
    def half_width(self) -> torch.Tensor:
        r"""实际半宽按原25档谱计算，不按本轮max_level重新放大。"""
        return self.level.float() * (self.cfg.full_half_width_m / self.cfg.reference_levels)

    def observe(self, ids: torch.Tensor, turns: torch.Tensor, safe_window: torch.Tensor) -> None:
        r"""只消费刚结束回合的对应环境，其他环境的课程状态保持。"""
        valid = self.initialized[ids]  # 首次reset不是一个已完成回合。
        selected = ids[valid]
        if selected.numel() == 0:
            return
        score = turns[valid].clamp(0, self.cfg.score_cap_turns)  # 不为负转或过快上尾提高课程分数。
        first = self.trials[selected] == 0
        updated = torch.where(first, score, (1 - self.cfg.ema_alpha) * self.ema[selected] + self.cfg.ema_alpha * score)
        self.ema[selected] = updated
        self.trials[selected] += 1
        ready = self.trials[selected] >= self.cfg.min_episodes
        up = ready & (updated >= self.cfg.promote_turns) & safe_window[valid] & (self.level[selected] < self.cfg.max_level)
        down = ready & (updated < self.cfg.demote_turns) & (self.level[selected] > 0)
        self.level[selected] += up.long() - down.long()  # 每次最多变化一档，逐环境独立。
        changed = selected[up | down]
        self.trials[changed] = 0
        self.ema[changed] = 0  # 新难度重新积累证据。

    def sample(self, ids: torch.Tensor) -> torch.Tensor:
        r"""在手掌语义xy平面采样，不扰动高度或初始姿态。"""
        noise = torch.rand((ids.numel(), 3), device=self.level.device) * 2 - 1
        noise[:, 2] = 0
        self.offset_h[ids] = noise * self.half_width[ids, None]
        self.initialized[ids] = True
        return self.offset_h[ids]

    def state_dict(self) -> dict[str, torch.Tensor]:
        r"""保存真实课程状态，供同方法resume恢复。"""
        return {name: getattr(self, name).detach().clone() for name in ('level', 'initialized', 'ema', 'trials', 'offset_h')}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        r"""恢复同轴环境状态，防止把一个环境的难度广播给所有环境。"""
        for name, value in state.items():
            target = getattr(self, name)
            if value.shape != target.shape:
                raise ValueError(f'ADR state shape mismatch: {name}')
            target.copy_(value.to(target.device))


def get_position_adr(env: Any) -> PerEnvironmentPositionAdr | None:
    r"""按nested配置按需实例化启用组件；关闭组件不创建运行态。"""
    cfg = getattr(env.cfg, 'adr', None)
    if cfg is None or not cfg.object_position.enabled:
        return None
    runtime = getattr(env, '_hetero_position_adr', None)
    if runtime is None:
        runtime = PerEnvironmentPositionAdr(cfg.object_position, env.num_envs, env.device)
        env._hetero_position_adr = runtime
    return runtime


def perturb_reset_position(env: Any, ids: torch.Tensor, position_w: torch.Tensor, hand_quat_w: torch.Tensor) -> torch.Tensor:
    r"""在基准预抓取之后、写入物理与捕获新anchor之前叠加本回合位置扰动。"""
    runtime = get_position_adr(env)
    if runtime is None:
        return position_w
    if bool(runtime.initialized[ids].any()):
        command = env.command_manager.get_term('goal_pose')
        duration = env.episode_length_buf[ids].float() * float(env.step_dt)
        complete = command.reference_window_complete[ids]
        turns = torch.where(complete, command.reference_window_net_turns[ids], command.net_rotation_turns[ids])
        failed = env.termination_manager.get_term('object_out_of_anchor')[ids] | env.termination_manager.get_term('goal_axis_misaligned')[ids]
        safe_window = complete & ((duration > runtime.cfg.reference_seconds + 1e-4) | ~failed)
        runtime.observe(ids, turns, safe_window)
    offset = runtime.sample(ids)  # 无关环境不重新采样。
    return position_w + quaternion_apply_wxyz(hand_quat_w, offset)  # 语义掌面坐标转到world。
