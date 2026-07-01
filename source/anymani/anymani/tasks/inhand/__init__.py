# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""In-hand manipulation tasks for AnyMani.

This module contains in-hand object manipulation tasks using dexterous hands.

MDP 组件库:
    - inhand_env_cfg: 可复用的 MDP 配置组件（Obs/Actions/Rewards 等）

手型配置:
    - config/leaphand/: LeapHand 配置
    - config/leaphand_round/: 半球指尖 LeapHand 配置
"""

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils", ".mdp"]


def _register_inhand_configs_if_runtime_available() -> None:
    r"""Register in-hand Gym configs only when Isaac Sim bindings are importable.

    默认 contract tests 会收集 `tasks/inhand/tests`，pytest 在进入测试文件前会先
    import `anymani.tasks.inhand`。普通 Python 环境里 `pxr` 只能在 Isaac Sim / Kit
    runtime 启动后使用，因此这里沿用顶层 `anymani.tasks` 的轻量导入边界：
    runtime 可用时注册 Gym env，runtime 不可用时允许纯文本/纯数学测试继续收集。
    """

    try:
        # Isaac Lab 的 registry helper 会间接导入 env/action/controller 模块，进而需要 `pxr`。
        from isaaclab_tasks.utils import import_packages

        # 导出 MDP 组件库；这些配置类依赖 IsaacLab runtime 类型，只在 runtime 可用时注入 globals。
        from .inhand_env_cfg import (
            CommonEventCfg,
            CommonRewardsCfg,
            CommonTerminationsCfg,
            EmptyCurriculumCfg,
            InHandObjectSceneCfg,
            JointSpaceActionsCfg,
            JointSpaceObservationsCfg,
            JointSpaceObsGroupCfg,
            ProprioceptionObsGroupCfg,
            ReorientationCommandsCfg,
            TactileCriticObsGroupCfg,
            TactileObservationsCfg,
            TactileObsGroupCfg,
            TactileRewardsCfg,
            TactileSceneCfg,
        )
    except ModuleNotFoundError as exc:
        if exc.name in {"pxr", "omni"}:
            return  # 未启动 Isaac Sim：允许 contract tests 以轻量方式 import 父包。
        raise

    # 将原先的公开配置名写回模块全局命名空间，保持 runtime 下的外部 import 兼容性。
    globals().update(
        {
            "CommonEventCfg": CommonEventCfg,
            "CommonRewardsCfg": CommonRewardsCfg,
            "CommonTerminationsCfg": CommonTerminationsCfg,
            "EmptyCurriculumCfg": EmptyCurriculumCfg,
            "InHandObjectSceneCfg": InHandObjectSceneCfg,
            "JointSpaceActionsCfg": JointSpaceActionsCfg,
            "JointSpaceObsGroupCfg": JointSpaceObsGroupCfg,
            "JointSpaceObservationsCfg": JointSpaceObservationsCfg,
            "ProprioceptionObsGroupCfg": ProprioceptionObsGroupCfg,
            "ReorientationCommandsCfg": ReorientationCommandsCfg,
            "TactileCriticObsGroupCfg": TactileCriticObsGroupCfg,
            "TactileObsGroupCfg": TactileObsGroupCfg,
            "TactileObservationsCfg": TactileObservationsCfg,
            "TactileRewardsCfg": TactileRewardsCfg,
            "TactileSceneCfg": TactileSceneCfg,
        }
    )

    # Import all configs in this package so each sub-package can register its Gym ids.
    import_packages(__name__, _BLACKLIST_PKGS)


_register_inhand_configs_if_runtime_available()
