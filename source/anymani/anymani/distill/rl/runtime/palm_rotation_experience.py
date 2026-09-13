r"""环境连续底层存储：保持rl_games的[H,N,...]接口，让env-major展开成为视图。

底层S的形状为[N,H,...]且连续，对外V=S.transpose(0,1)。因此
swap_and_flatten01(V)=S.reshape(N*H,...)，无需再复制整份历史和几何特征。
本模块只改变存储stride，不改变轨迹时间轴、样本值、dtype、GAE或分层排列。
"""

from __future__ import annotations

from typing import Any

import torch
from rl_games.common.experience import ExperienceBuffer


class EnvMajorExperienceBuffer(ExperienceBuffer):
    r"""普通非RNN连续PPO的经验分配器；复用原始update_data与get_transformed_list。

递归Dict保持同一[H,N]逻辑前缀，底层仅交换该前缀的存储次序。
原父类负责空间dtype、零初始化及CPU pinned-memory约定，外部rl_games不修改。
    """

    def _create_tensor_from_space(self, space: Any, base_shape: tuple[int, ...]) -> Any:
        r"""分配[N,H,...]后暴露[H,N,...]视图；Dict在本层递归以防重复交换。"""
        if type(space).__name__ == "Dict":
            return {key: self._create_tensor_from_space(value, base_shape) for key, value in space.spaces.items()}
        if len(base_shape) != 2:
            raise ValueError("env-major rollout storage requires a [horizon,environments] prefix")
        horizon, environments = base_shape  # 逻辑时间步数H与并行环境数N。
        backing = super()._create_tensor_from_space(space, (environments, horizon))  # 唯一一份连续底层。
        assert isinstance(backing, torch.Tensor)  # Dict已在上层递归，此处必须是叶子张量。
        return backing.transpose(0, 1)  # 只改视图，写入和GAE仍使用V[t,env,...]。

    def side_channel(self, width: int) -> torch.Tensor:
        r"""为动作/FiLM诊断分配相同存储顺序，防止旁路再产生完整展开副本。"""
        horizon, environments = self.obs_base_shape
        return torch.zeros(environments, horizon, width, dtype=torch.float32, device=self.device).transpose(0, 1)
