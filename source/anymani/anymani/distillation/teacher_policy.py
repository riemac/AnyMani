from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gym
import torch

from rl_games.algos_torch.players import PpoPlayerContinuous


def _clone_tree(x: Any) -> Any:
    """递归深拷贝嵌套的字典、列表或元组结构
    
    Args:
        x: 要克隆的对象，可以是字典、列表、元组或其他类型
        
    Returns:
        克隆后的对象，保持原有的嵌套结构
    """
    if isinstance(x, dict):
        return {k: _clone_tree(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_clone_tree(v) for v in x)
    return x


def _zero_rnn_states(states: Any, done_ids: torch.Tensor) -> Any:
    """将指定环境的RNN隐藏状态清零
    
    当某些环境完成episode时，需要重置其对应的RNN隐藏状态。
    该函数递归处理各种可能的状态结构（Tensor、list、tuple等）。
    
    Args:
        states: RNN隐藏状态，可以是Tensor、list、tuple或None
        done_ids: 需要重置的环境索引张量
        
    Returns:
        更新后的状态，指定环境的隐藏状态已清零
    """
    if states is None:
        return None
    if isinstance(states, torch.Tensor):
        # 常见形状: (num_layers, num_envs, hidden) 或 (num_envs, hidden)
        # 如果倒数第二维是环境维度，则将对应环境的状态清零
        if states.ndim >= 2 and states.shape[-2] >= int(done_ids.max().item()) + 1:
            states[..., done_ids, :] = 0
        # 如果第一维是环境维度，则将对应环境的状态清零
        elif states.ndim >= 1 and states.shape[0] >= int(done_ids.max().item()) + 1:
            states[done_ids, ...] = 0
        return states
    # 递归处理列表或元组类型的状态
    if isinstance(states, (list, tuple)):
        out = []
        for s in states:
            out.append(_zero_rnn_states(s, done_ids))
        return type(states)(out)
    return states


@dataclass
class RLGamesTeacherPolicy:
    """最小化的rl_games教师策略包装器
    
    使用rl_games的PpoPlayerContinuous加载检查点并执行推理。
    
    如果底层网络是基于RNN的，则策略是有状态的；
    使用reset_done()为刚刚重置的环境清零隐藏状态。
    
    Attributes:
        player: rl_games的PPO玩家实例，用于策略推理
        action_scale: 动作缩放因子
    """

    player: PpoPlayerContinuous
    action_scale: float

    @classmethod
    def from_agent_cfg(
        cls,
        teacher_params: dict,
        *,
        checkpoint_path: str,
        obs_dim: int,
        action_dim: int,
        device_name: str,
        action_scale: float,
        num_envs: int | None = None,
    ) -> "RLGamesTeacherPolicy":
        """从配置创建教师策略实例
        
        Args:
            teacher_params: 教师策略的参数字典（将被深拷贝）
            checkpoint_path: 检查点文件路径
            obs_dim: 观测空间维度
            action_dim: 动作空间维度
            device_name: 设备名称（如'cuda:0'或'cpu'）
            action_scale: 动作缩放因子
            num_envs: 环境数量，用于批量推理
            
        Returns:
            初始化好的RLGamesTeacherPolicy实例
        """
        # 深拷贝参数以避免修改原始配置
        params = _clone_tree(teacher_params)
        cfg = params.setdefault("config", {})
        cfg["device_name"] = device_name
        
        # 设置环境信息，定义观测和动作空间
        cfg["env_info"] = {
            "observation_space": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(obs_dim,)),
            "action_space": gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,)),
            "state_space": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(0,)),
        }
        
        # 创建PPO玩家并加载检查点
        player = PpoPlayerContinuous(params)
        player.restore(checkpoint_path)

        # 当为多个环境提供观测时启用批量推理
        # 否则rl_games会将(num_envs, obs_dim)展开并压平为(1, num_envs*obs_dim)
        if num_envs is not None and int(num_envs) > 1:
            player.has_batch_dimension = True
            player.batch_size = int(num_envs)

        # 重置玩家状态（包括RNN隐藏状态）
        player.reset()
        return cls(player=player, action_scale=action_scale)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, *, deterministic: bool = True) -> torch.Tensor:
        """根据观测生成动作
        
        Args:
            obs: 观测张量，形状为(num_envs, obs_dim)或(obs_dim,)
            deterministic: 是否使用确定性策略（True）或随机策略（False）
            
        Returns:
            动作张量，形状与输入观测的批次维度匹配
        """
        # PpoPlayerContinuous期望Tensor或dict类型的观测，这里使用Tensor
        action = self.player.get_action(obs, is_deterministic=deterministic)
        # 确保返回值是torch.Tensor类型
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, device=self.player.device)
        return action

    def reset_all(self) -> None:
        """重置所有环境的策略状态
        
        清零所有RNN隐藏状态（如果策略使用RNN）
        """
        self.player.reset()

    def reset_done(self, done_ids: torch.Tensor) -> None:
        """重置指定环境的策略状态
        
        仅清零已完成episode的环境对应的RNN隐藏状态。
        
        Args:
            done_ids: 已完成环境的索引张量
        """
        if done_ids.numel() == 0:
            return
        self.player.states = _zero_rnn_states(self.player.states, done_ids)
