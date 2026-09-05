r"""通过真实canonical runtime lowering发布schema-1.2 cohort lock。

输入1.0/1.1 source-resolved lock，先在AppLauncher内构造与训练相同的``HandSpawnAdapter``，取得逐成员
``physical_geometry_hash``与``canonical_schema_digest``，再写到独立``*.canonical.lock.yaml``。本入口不创建
Isaac scene、不推进PhysX；输出lock供正式PPO/evaluation，source lock继续保留原有pregrasp证据链。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    r"""在任何Isaac/task import前解析source与目标lock路径。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


ARGS = _parse_args()
SOURCE_LOCK = ARGS.source_lock.expanduser().resolve(strict=True)  # source lock exact bytes
OUTPUT_LOCK = (
    ARGS.output.expanduser().resolve()
    if ARGS.output is not None
    else SOURCE_LOCK.with_name(f"{SOURCE_LOCK.name.removesuffix('.lock.yaml')}.canonical.lock.yaml")
)
os.environ.pop("ANYMANI_HETERO_ASSET_ROWS", None)
os.environ["ANYMANI_HETERO_COHORT_LOCK"] = str(SOURCE_LOCK)  # config import前冻结同序prototype轴
os.environ["ANYMANI_HETERO_NUM_ENVS"] = "1"  # 只构造binding；不实例化依赖env轴的task cfg

from isaaclab.app import AppLauncher  # noqa: E402  # Kit必须先于pxr/robot adapter import启动

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


def main() -> None:
    r"""执行同训练路径lowering，并发布/重载canonical-final lock。"""

    from anymani.assets.bank.cohort import finalize_hand_asset_cohort_lock
    from anymani.tasks.hetero.config.generated.asset_binding import build_generated_asset_binding

    binding = build_generated_asset_binding()
    artifacts = binding.canonical_artifacts
    schema_versions = {artifact.schema_version for artifact in artifacts}
    if len(schema_versions) != 1:
        raise RuntimeError(f"cohort canonical artifacts disagree on schema version: {sorted(schema_versions)}")
    output = finalize_hand_asset_cohort_lock(
        SOURCE_LOCK,
        OUTPUT_LOCK,
        canonical_identities=tuple(
            (artifact.source_content_hash, artifact.physical_geometry_hash, artifact.schema_digest)
            for artifact in artifacts
        ),
        canonical_schema_version=next(iter(schema_versions)),
    )
    print(
        json.dumps(
            {
                "source_lock": str(SOURCE_LOCK),
                "source_lock_sha256": binding.cohort_lock_sha256,
                "canonical_lock": str(output),
                "asset_count": binding.asset_count,
                "physical_geometry_hashes_unique": len({artifact.physical_geometry_hash for artifact in artifacts}),
                "canonical_schema_version": next(iter(schema_versions)),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
