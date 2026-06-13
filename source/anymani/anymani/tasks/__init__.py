# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""Package containing task implementations for the extension.

本包的运行时职责是注册 Isaac Lab / Gym task；但局部 contract tests 只需要把
`anymani.tasks.gm.tests` 当作普通测试目录导入，不应因此启动 Isaac Sim / USD
binding。Isaac Sim 的 `pxr` / `omni` 模块通常要在 `AppLauncher` 之后才可用，
所以这里采用“可注册则注册，否则保持轻量 import”的边界。
"""

##
# Register Gym environments.
##

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils", ".mdp"]


def _register_task_packages_if_runtime_available() -> None:
    r"""Register task packages when Isaac Sim runtime bindings are available.

    纯 Python contract tests 会 import `anymani.tasks` 的父包路径；若此时尚未通过
    `AppLauncher` 加载 Isaac Sim，`isaaclab_tasks.utils` 可能间接导入 `pxr` 失败。
    这不是任务配置错误，而是测试层不应触发重型仿真注册。
    """

    try:
        from isaaclab_tasks.utils import import_packages  # Isaac Lab task registry helper
    except ModuleNotFoundError as exc:
        if exc.name in {"pxr", "omni"}:
            return  # 未启动 Isaac Sim：允许轻量导入父包，具体 env 注册留给运行时入口
        raise

    import_packages(__name__, _BLACKLIST_PKGS)  # Isaac Sim 可用时沿用原有自动注册语义


_register_task_packages_if_runtime_available()
