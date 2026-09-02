r"""把P0001已认证records迁移到task-owned formal physics identity。

迁移不改变candidate、metrics、gate、tier或scale certificate，只补录搜索时已经由scene配置和三scale probe确定的
density/material/solver/mass/inertia/contact/controller字段。Source cache保留不动；destination使用新lookup/content digests，
使formal task无法命中旧的欠约束identity。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from anymani.pregrasp import AtomicPregraspCache, PregraspRecord
from anymani.pregrasp.schema import stable_digest
from anymani.tasks.hetero.config.generated.pregrasp_identity import (
    DEX_CUBE_SHA256,
    FORMAL_PREGRASP_GATE,
    FORMAL_SEARCH_PROTOCOL_DIGEST,
    FormalPregraspCatalogIdentity,
    formal_physics_identity,
    validate_formal_search_identity,
)

LEGACY_PHYSICS_IDENTITY = {
    "isaac_sim": "5.1",
    "object_sha256": DEX_CUBE_SHA256,
    "absolute_prestartup_scale": True,
    "object_mass_policy": "fixed_mass_from_usd",
    "object_inertia_scale_law": "approximately_s_squared",
    "physics_dt_s": 1.0 / 120.0,
    "policy_dt_s": 0.05,
    "solver_position_iterations": 8,
    "solver_velocity_iterations": 0,
    "contact_force_threshold_N": 0.25,
    "contact_ema_alpha": 0.5,
    "effort_source": "implicit_actuator_computed_torque",
}
r"""待迁移source record必须exact匹配的P0001 identity，防止脚本替任意旧证据背书。"""


def _parse_args() -> argparse.Namespace:
    r"""解析source/destination cache与migration manifest路径。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/pregrasp/schema_v2/cache"))
    parser.add_argument("--destination", type=Path, default=Path("outputs/pregrasp/schema_v2/formal-cache-v2"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/pregrasp/schema_v2/formal-cache-v2-migration.json"),
    )
    return parser.parse_args()


def _load_record(cache: AtomicPregraspCache, entry: Any) -> PregraspRecord:
    r"""从source index entry严格恢复record与所有content digests。"""

    payload = json.loads(cache.payload_path(entry).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("pregrasp source payload must be a JSON object")
    return PregraspRecord.from_dict(payload)


def _validate_source(record: PregraspRecord) -> None:
    r"""只允许迁移当前formal gate、Cube与完整basin protocol下的P0001 records。"""

    key = record.lookup_key
    if key.cube_asset_sha256 != DEX_CUBE_SHA256 or key.cube_asset_id != "DexCube":
        raise ValueError("source record does not bind the measured DexCube bytes")
    if key.gate_digest != FORMAL_PREGRASP_GATE.digest or record.gate != FORMAL_PREGRASP_GATE:
        raise ValueError("source record gate differs from current formal gate")
    if stable_digest(key.physics_identity) != stable_digest(LEGACY_PHYSICS_IDENTITY):
        raise ValueError("source record is not from the exact P0001 physics identity")
    validate_formal_search_identity(key.search_identity)


def main() -> int:
    r"""幂等发布18条增强identity records并写old→new provenance。"""

    args = _parse_args()
    source = AtomicPregraspCache(args.source)
    destination = AtomicPregraspCache(args.destination)
    source_index = source.load_index()
    migrated: list[dict[str, Any]] = []
    for entry in source_index.entries:
        record = _load_record(source, entry)
        _validate_source(record)
        scale = record.candidate.object_scale
        physics_identity = formal_physics_identity(object_scale=scale, cube_sha256=DEX_CUBE_SHA256)
        new_key = replace(record.lookup_key, physics_identity=physics_identity)
        expected_identity = FormalPregraspCatalogIdentity.build(object_scale=scale, cube_sha256=DEX_CUBE_SHA256)
        expected_identity.validate_lookup_key(new_key)
        migrated_record = replace(record, lookup_key=new_key)
        published = destination.publish(migrated_record)
        migrated.append(
            {
                "asset_id": record.lookup_key.asset_id,
                "tier": record.tier.value,
                "scale": scale,
                "source_record_digest": record.digest,
                "source_lookup_digest": record.lookup_key.digest,
                "record_digest": migrated_record.digest,
                "lookup_digest": migrated_record.lookup_key.digest,
                "payload_relpath": published.payload_relpath,
            }
        )
    destination_index = destination.load_index()
    if len(destination_index.entries) != len(source_index.entries):
        raise RuntimeError("formal destination cache does not contain exactly the migrated source records")
    artifact = {
        "artifact_type": "anymani.pregrasp.formal_identity_migration",
        "schema_version": "1.0.0",
        "source_cache": str(source.root),
        "source_index_digest": source_index.digest,
        "destination_cache": str(destination.root),
        "destination_index_digest": destination_index.digest,
        "record_count": len(migrated),
        "formal_gate_digest": FORMAL_PREGRASP_GATE.digest,
        "formal_search_protocol_digest": FORMAL_SEARCH_PROTOCOL_DIGEST,
        "records": migrated,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: artifact[key] for key in artifact if key != "records"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
