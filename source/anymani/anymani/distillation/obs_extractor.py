from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils.buffers import CircularBuffer
from isaaclab.utils import noise


@dataclass
class _TermRuntime:
    """观测项运行时数据结构
    
    存储单个观测项的配置和历史缓冲区信息
    """
    name: str  # 观测项名称
    cfg: ObservationTermCfg  # 观测项配置
    history_len: int  # 历史长度
    flatten_history_dim: bool  # 是否展平历史维度
    history: CircularBuffer | None  # 循环缓冲区，用于存储历史观测


class ObsGroupExtractor:
    """在任意环境上计算Isaac Lab观测组
    
    该类镜像了Isaac Lab的ObservationManager.compute_group()的核心逻辑，
    但在内部保持历史缓冲区，因此可用于*教师*策略，其观测组与学生环境的
    活动组不同。

    注意:
        - 默认情况下禁用损坏/噪声，因为教师应该尽可能确定性。
        - 该类假设每个term函数都可以用提供的env调用。
    """

    def __init__(
        self,
        env: Any,
        group_cfg: ObservationGroupCfg,
        device: str | torch.device,
        *,
        apply_corruption: bool = False,
    ) -> None:
        """初始化观测组提取器
        
        Args:
            env: 环境实例
            group_cfg: 观测组配置
            device: 计算设备
            apply_corruption: 是否应用噪声/损坏（默认False，保持教师策略确定性）
        """
        self._env = env
        self._group_cfg = group_cfg
        self._device = str(device)
        self._apply_corruption = apply_corruption

        # 获取是否拼接观测项的配置，默认为True
        self._concatenate_terms = bool(getattr(group_cfg, "concatenate_terms", True))
        # 获取拼接维度，默认为-1（最后一维）
        self._concatenate_dim = int(getattr(group_cfg, "concatenate_dim", -1))
        if self._concatenate_dim >= 0:
            # 考虑batch维度，需要+1
            self._concatenate_dim += 1

        # 存储所有观测项的运行时信息
        self._terms: list[_TermRuntime] = []
        self._build_terms()

    def _build_terms(self) -> None:
        """构建观测项列表
        
        遍历观测组配置中的所有项，为每个有效的ObservationTermCfg创建
        _TermRuntime实例，并初始化历史缓冲区（如果需要）。
        """
        group_items = self._group_cfg.__dict__.items()
        for term_name, term_cfg in group_items:
            # 跳过组级别的配置项（非观测项）
            if term_name in [
                "enable_corruption",
                "concatenate_terms",
                "history_length",
                "flatten_history_dim",
                "concatenate_dim",
            ]:
                continue
            # 跳过None值
            if term_cfg is None:
                continue
            # 只处理ObservationTermCfg类型的配置
            if not isinstance(term_cfg, ObservationTermCfg):
                continue

            # 组级别的历史配置会覆盖单个观测项的配置
            group_hist = getattr(self._group_cfg, "history_length", None)
            group_flat = getattr(self._group_cfg, "flatten_history_dim", True)

            # 获取观测项的历史长度和展平配置
            history_len = int(term_cfg.history_length)
            flatten_hist = bool(term_cfg.flatten_history_dim)
            # 如果组级别有配置，则使用组级别的配置覆盖
            if group_hist is not None:
                history_len = int(group_hist)
                flatten_hist = bool(group_flat)

            # 如果需要历史，创建循环缓冲区
            history: CircularBuffer | None = None
            if history_len and history_len > 0:
                history = CircularBuffer(max_len=history_len, batch_size=self._env.num_envs, device=self._device)

            # 添加到观测项列表
            self._terms.append(
                _TermRuntime(
                    name=term_name,
                    cfg=term_cfg,
                    history_len=history_len,
                    flatten_history_dim=flatten_hist,
                    history=history,
                )
            )

    def reset(self, env_ids: list[int] | torch.Tensor | None = None) -> None:
        """重置观测历史缓冲区
        
        Args:
            env_ids: 要重置的环境ID列表。如果为None，则重置所有环境。
        """
        for term in self._terms:
            if term.history is not None:
                if env_ids is None:
                    term.history.reset()
                else:
                    term.history.reset(env_ids)

    @torch.no_grad()
    def compute(self, *, update_history: bool = True) -> torch.Tensor | dict[str, torch.Tensor]:
        """计算观测组
        
        对每个观测项执行以下步骤：
        1. 调用观测函数获取原始观测
        2. 应用修改器（如果有）
        3. 应用噪声/损坏（如果启用）
        4. 裁剪和缩放
        5. 更新历史缓冲区（如果需要）
        
        Args:
            update_history: 是否更新历史缓冲区。设为False可用于获取观测维度而不影响状态。
            
        Returns:
            如果concatenate_terms=True，返回拼接后的张量；
            否则返回字典，键为观测项名称，值为对应的观测张量。
        """
        group_obs: dict[str, torch.Tensor] = {}
        for term in self._terms:
            # 1. 调用观测函数获取原始观测值
            obs: torch.Tensor = term.cfg.func(self._env, **term.cfg.params).clone()

            # 2. 应用修改器（例如归一化、变换等）
            if term.cfg.modifiers is not None:
                for modifier in term.cfg.modifiers:
                    obs = modifier.func(obs, **modifier.params)

            # 3. 应用噪声/损坏（默认禁用以保持教师策略的确定性）
            if self._apply_corruption and getattr(self._group_cfg, "enable_corruption", False):
                if isinstance(term.cfg.noise, noise.NoiseCfg):
                    obs = term.cfg.noise.func(obs, term.cfg.noise)
                elif isinstance(term.cfg.noise, noise.NoiseModelCfg) and term.cfg.noise.func is not None:
                    obs = term.cfg.noise.func(obs)

            # 4. 裁剪和缩放
            if term.cfg.clip:
                obs = obs.clip_(min=term.cfg.clip[0], max=term.cfg.clip[1])
            if term.cfg.scale is not None:
                obs = obs.mul_(term.cfg.scale)

            # 5. 处理历史缓冲区
            if term.history is not None and term.history_len > 0:
                if update_history:
                    # 更新历史缓冲区
                    term.history.append(obs)
                elif term.history._buffer is None:  # type: ignore[attr-defined]
                    # 保护：在首次外部调用时初始化历史
                    term.history.append(obs)

                # 获取历史缓冲区
                buf = term.history.buffer
                if term.flatten_history_dim:
                    # 展平历史维度：(num_envs, history_len, obs_dim) -> (num_envs, history_len * obs_dim)
                    group_obs[term.name] = buf.reshape(self._env.num_envs, -1)
                else:
                    # 保持历史维度
                    group_obs[term.name] = buf
            else:
                # 无历史，直接使用当前观测
                group_obs[term.name] = obs

        # 根据配置决定返回拼接的张量还是字典
        if self._concatenate_terms:
            return torch.cat(list(group_obs.values()), dim=self._concatenate_dim)
        return group_obs

    @property
    def obs_dim(self) -> int:
        """获取观测维度
        
        注意：仅在concatenate_terms=True时有效
        
        Returns:
            观测向量的维度
            
        Raises:
            RuntimeError: 如果返回的是字典观测（concatenate_terms=False）
        """
        obs = self.compute(update_history=False)
        if isinstance(obs, dict):
            raise RuntimeError("ObsGroupExtractor returns Dict obs; obs_dim is undefined.")
        return int(obs.shape[-1])
