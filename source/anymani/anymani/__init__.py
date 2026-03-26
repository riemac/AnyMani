# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

# NOTE:
#   `anymani` 既承载 Isaac Lab runtime 集成，也承载如 `assets/` 这类前序工程子项目。
#   在仅做 schema / builder / validator 测试时，当前环境未必安装了完整的
#   Isaac Sim / pxr 依赖，因此这里对 runtime-facing 注册逻辑做软导入处理。
#   若相关依赖齐全，行为与原来一致；若不齐全，则允许轻量子模块独立被导入和测试。
try:
    # Register Gym environments.
    from .tasks import *

    # Register UI extensions.
    from .ui_extension_example import *
except ModuleNotFoundError:
    # 缺少可选 runtime 依赖时，允许轻量子模块继续工作。
    pass
