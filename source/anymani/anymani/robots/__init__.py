# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""机器人 runtime 配置模块。

`robots` 是 AnyMani 的 embodiment adapter 层：它消费 `assets` 生成的 hand bundle，
并把它们 lower 成 Isaac Lab 可以 spawn 的 robot / articulation 配置。任务环境
只消费这里的 robot cfg，不应把 spawn/importer 细节复制到 `tasks` 或 `distill`。

本包刻意保持 lazy facade：contract tests 会用 IsaacLab stub 导入
`anymani.robots.hand_spawn`，若在包初始化阶段 eager import `leap.py`，就会把真实
IsaacLab robot cfg 拉进纯 Python 测试路径，破坏“默认 pytest 不启动 Isaac Sim”的约定。
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DEFAULT_HAND_ANCHOR_POS_E",
    "HandActuatorSpawnCfg",
    "HandFrameCfg",
    "HandJointInitCfg",
    "HandSpawnAdapter",
    "HandSpawnCfg",
    "HandUrdfSpawnCfg",
    "LEAP_HAND_CFG",
    "LEAP_HAND_URDF_CFG",
    "LEAP_HAND_URDF_PATH",
]


def __getattr__(name: str) -> Any:
    r"""按需导出 robot cfg 与 generated hand spawn adapter。

    Args:
        name (str): 被外部请求的公开符号名。

    Returns:
        Any: 对应子模块中的配置类或常量。

    Raises:
        AttributeError: 当请求的符号不属于本 facade 时抛出。
    """

    if name == "LEAP_HAND_CFG":
        from .leap import LEAP_HAND_CFG

        return LEAP_HAND_CFG
    if name in {"LEAP_HAND_URDF_CFG", "LEAP_HAND_URDF_PATH"}:
        from . import leap_urdf

        return getattr(leap_urdf, name)
    if name in {
        "DEFAULT_HAND_ANCHOR_POS_E",
        "HandActuatorSpawnCfg",
        "HandFrameCfg",
        "HandJointInitCfg",
        "HandSpawnAdapter",
        "HandSpawnCfg",
        "HandUrdfSpawnCfg",
    }:
        from . import hand_spawn

        return getattr(hand_spawn, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
