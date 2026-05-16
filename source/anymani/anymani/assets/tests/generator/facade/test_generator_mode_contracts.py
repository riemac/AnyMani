"""`HandGenerator` façade 的 mode 边界回归测试。"""

from __future__ import annotations

import pytest

from assets.generator.hand_generator import HandGenerator, HandGeneratorCfg
from assets.generator.mutate import HandMutatorCfg, MountPerturbCfg


def _single_full_pool() -> dict[str, dict[str, list[str]]]:
    """提供一个只有 canonical full topology 的 pre-made pool。"""

    return {
        "single_palm_allegro": {
            "thumb": ["allegro_thumb_full"],
            "index": ["allegro_non_thumb_full"],
            "middle": ["allegro_non_thumb_full"],
            "ring": ["allegro_non_thumb_full"],
        }
    }


class DemoMountOnlyMutatorCfg(HandMutatorCfg):
    """只启用 mount perturb，供 generator full 模式边界测试使用。"""

    mount = MountPerturbCfg(
        self_mode="general",
        pos_radius=0.001,
    )


def test_hand_generator_full_mode_is_explicitly_blocked_during_layout_migration():
    """`mode=\"full\"` 目前应显式拒绝，避免旧 full 语义悄悄混入新目录 contract。"""

    with pytest.raises(NotImplementedError, match="mode='full' is temporarily unsupported"):
        list(
            HandGenerator(
                HandGeneratorCfg(
                    mode="full",
                    artifact_level="hand_cfg",
                    handedness="right",
                    hand_presets=["single_palm_allegro"],
                    connectivity_presets=_single_full_pool(),
                    mixed=False,
                    missing=False,
                    max_enumerate=3,
                    n_samples=3,
                    Mutate=DemoMountOnlyMutatorCfg(),
                )
            ).generate_batch()
        )
