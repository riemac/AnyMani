# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""In-hand manipulation tasks for AnyMani.

This module contains in-hand object manipulation tasks using dexterous hands.
"""

from isaaclab_tasks.utils import import_packages

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils", ".mdp"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
