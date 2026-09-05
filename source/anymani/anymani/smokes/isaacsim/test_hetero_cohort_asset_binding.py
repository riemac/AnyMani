r"""Member-level cohort canonical lowering与exact pregrasp binding的IsaacSim smoke。

Canonical USD lowering依赖``pxr``，因此本文件先启动Kit，再导入AnyMani robot/task模块。测试不创建simulation
world、不推进physics，只验证A16成员顺序、source provenance、physical hash、env routing与pregrasp key。
运行命令：

```
timeout --kill-after=20s 180s /home/hac/isaac/IsaacLab/isaaclab.sh -p -m pytest \
  source/anymani/anymani/smokes/isaacsim/test_hetero_cohort_asset_binding.py -q -s
```
"""

from __future__ import annotations

# ruff: noqa: E402, I001
# Runtime routing必须在AppLauncher前冻结；Isaac/AnyMani runtime import必须位于launcher之后。

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

ANYMANI_ROOT = Path(__file__).resolve().parents[5]
COHORT_LOCK = (
    ANYMANI_ROOT
    / "source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1/cohorts/pure-leap-right-a16.lock.yaml"
)
os.environ.pop("ANYMANI_HETERO_ASSET_ROWS", None)
os.environ["ANYMANI_HETERO_COHORT_LOCK"] = str(COHORT_LOCK)
os.environ["ANYMANI_HETERO_NUM_ENVS"] = "128"

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from anymani.tasks.hetero.config.generated.asset_binding import build_generated_asset_binding
from anymani.tasks.hetero.config.generated.cohort_good_pregrasp_identity import (
    COHORT_GOOD_PREGRASP_CATALOG_ROOT,
    COHORT_GOOD_PREGRASP_GENERATION_DIGEST,
)


def teardown_module() -> None:
    r"""模块内唯一canonical-lowering断言完成后关闭Kit。"""

    simulation_app.close()


def test_a16_cohort_lowers_to_dense_runtime_and_cohort_pregrasp_identity() -> None:
    r"""A16保持lock顺序，并把dense local routing与source provenance严格分离。"""

    binding = build_generated_asset_binding()
    assert binding.asset_count == 16
    assert binding.dataset_rows == tuple(range(16))
    assert binding.source_rows == tuple(range(848, 864))
    assert binding.source_member_keys == tuple(f"ppo#{row}" for row in range(848, 864))
    assert binding.cohort_id == "pure-leap-right-a16"
    assert binding.cohort_lock_sha256 == hashlib.sha256(COHORT_LOCK.read_bytes()).hexdigest()
    assert len({artifact.physical_geometry_hash for artifact in binding.canonical_artifacts}) == 16
    assert Counter(binding.asset_index_by_env(128)) == Counter({index: 8 for index in range(16)})

    reset_cfg = binding.build_good_pregrasp_reset_cfg(num_envs=128)
    assert Path(reset_cfg.catalog_root) == ANYMANI_ROOT / COHORT_GOOD_PREGRASP_CATALOG_ROOT
    assert reset_cfg.require_strict is True
    assert len(reset_cfg.bindings) == 16
    assert {
        item.resolve_key().generation_identity_digest for item in reset_cfg.bindings
    } == {COHORT_GOOD_PREGRASP_GENERATION_DIGEST}
    print(
        json.dumps(
            {
                "asset_count": binding.asset_count,
                "cohort_lock_sha256": binding.cohort_lock_sha256,
                "replicas_per_asset": 8,
                "catalog_root": reset_cfg.catalog_root,
                "generation_identity_digest": COHORT_GOOD_PREGRASP_GENERATION_DIGEST,
            },
            sort_keys=True,
        ),
        flush=True,
    )
