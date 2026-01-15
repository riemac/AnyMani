"""Online distillation utilities for LeapHand.

This package provides lightweight building blocks for teacher-student online distillation
used with Isaac Lab + rl_games.

- Teacher policy inference (rl_games checkpoints)
- Teacher observation extraction from an Isaac Lab env state
- Gymnasium wrapper that injects teacher labels into env `extras`
"""

from .teacher_policy import RLGamesTeacherPolicy
from .obs_extractor import ObsGroupExtractor
from .label_wrapper import DistillInfoWrapper
