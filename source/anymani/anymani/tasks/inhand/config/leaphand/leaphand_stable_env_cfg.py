# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stable (frozen) environment configurations for reproducibility.

This file contains verified, stable environment configurations that have been
validated with trained models. These configurations should NOT be modified
to ensure reproducibility of previous training results.

Usage:
    1. Train a model using configurations from leaphand_env_cfg.py
    2. When satisfied with results, copy the configuration here with a version suffix
    3. Record the git commit, model path, and obs/action dimensions in the docstring
    4. Import from this file when replaying old models

Example:
    from anymani.tasks.inhand.config.leaphand.leaphand_stable_env_cfg import (
        LeapHandJointStableV1Cfg
    )
"""

# TODO: 当有训练好的稳定模型时，将配置复制到这里
#
# 格式示例：
#
# @configclass
# class LeapHandJointStableV1Cfg(InHandObjectEnvCfg):
#     '''Stable configuration V1 for LeapHand joint-space control.
#
#     Frozen at: 2026-02-02
#     Git commit: abc123
#     Model path: outputs/2026-02-02/leaphand_joint.pth
#     Obs dim: 85
#     Action dim: 16
#
#     ⚠️ DO NOT MODIFY - This configuration is frozen for reproducibility.
#     '''
#     pass  # 完整配置定义
