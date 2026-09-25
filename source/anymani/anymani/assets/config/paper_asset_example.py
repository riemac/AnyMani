"""Small pre-made LEAP recipe for CPU asset generation and bundle checks."""

from pathlib import Path

from .asset_gen_cfg import PRE_MADE_CFG as BASE_PRE_MADE_CFG


PRE_MADE_CFG = BASE_PRE_MADE_CFG.replace(
    output_dir=Path("outputs/paper-asset-example"),
    hand_presets=["single_palm_leap"],
    connectivity_presets={
        "single_palm_leap": {
            "thumb": ["leap_thumb_full"],
            "index": ["leap_non_thumb_full"],
            "middle": ["leap_non_thumb_full"],
            "ring": ["leap_non_thumb_full"],
        }
    },
    handedness="right",
    mixed=False,
    missing=False,
    max_enumerate=1,
    premade_parallel=False,
)

PRE_MADE_SHOW_REGISTRY = False
PRE_MADE_PRINT_RESULT_LIMIT = 1
