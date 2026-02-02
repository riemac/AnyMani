# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration sub-package for in-hand manipulation tasks.

This package contains robot-specific configurations (LeapHand, Allegro, etc.).
"""

from isaaclab_tasks.utils import import_packages

# Import all configs in this package
import_packages(__name__, blacklist=[])
