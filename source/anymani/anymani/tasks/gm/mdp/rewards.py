r"""Reward terms for `tasks.gm`.

本模块只承载任务奖励与正则项的函数实现。奖励设计应描述“物体是否完成了
手内操作目标”，不要在 reward 中偷偷编码资产采样偏好。

# TODO：
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv