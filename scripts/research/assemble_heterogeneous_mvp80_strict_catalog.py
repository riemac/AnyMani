r"""从同一strict-v5协议的多次物理筛选证据组装最终80×Top-8 catalog。

Left/right pair fallback天然发生在首次80手run之后。该脚本不重跑物理，也不修改门限；它读取候选manifest和
一个或多个strict summaries/NPZ，按每row严格通过数建立资格集合，再调用原确定性pair finalizer选出每个
neutral cell的前10个完整pairs。每个最终entry从对应物理证据重建、重放strict gate后才原子发布。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATES = ROOT / (
    "source/anymani/anymani/assets/datasets/cross_embodiment_balanced_v1/ppo_mvp80_candidates.yaml"
)
DEFAULT_CATALOG = ROOT / "outputs/pregrasp/catalogs/heterogeneous_rotation_mvp80_dexcube_s1p1_v5"
DEFAULT_SUMMARY = ROOT / "outputs/pregrasp/search/heterogeneous_rotation_mvp80_dexcube_s1p1_v5/final-summary.json"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
parser.add_argument("--source", type=Path, action="append", required=True, help="Strict generation summary JSON.")
parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
AppLauncher.add_app_launcher_args(parser)
args, unknown = parser.parse_known_args()
sys.argv = [sys.argv[0], *unknown]


def _resolved(path: Path) -> Path:
    r"""把CLI相对路径锚定到AnyMani root。"""

    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    r"""流式计算summary/catalog provenance摘要。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


candidate_path = _resolved(args.candidates)
source_paths = tuple(_resolved(path) for path in args.source)
candidate_document = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
source_documents = tuple(json.loads(path.read_text(encoding="utf-8")) for path in source_paths)

# AppLauncher之前只做JSON/YAML选择：通过row集合决定ASSET_BINDING必须构造的最终80轴。
qualified_rows = {
    int(item["dataset_row"])
    for document in source_documents
    for item in document["asset_candidate_counts"]
    if int(item["strict_passed_candidates"]) >= 8
}
from anymani.assets.bank.representative_selection import finalize_representative_selection  # noqa: E402

selection = finalize_representative_selection(
    candidate_document,
    passed_rows=tuple(sorted(qualified_rows)),
    pregrasp_catalog_root=str(_resolved(args.catalog).relative_to(ROOT)),
    pregrasp_summary_paths=[str(_resolved(args.summary).relative_to(ROOT))],
)
selected_rows = tuple(int(row) for row in selection["selected_rows"])
if len(selected_rows) != 80 or len(set(selected_rows)) != 80:
    raise RuntimeError("strict evidence did not yield exactly 10 complete pairs per neutral cell")
os.environ["ANYMANI_HETERO_ASSET_ROWS"] = ",".join(str(row) for row in selected_rows)
os.environ["ANYMANI_HETERO_NUM_ENVS"] = "80"

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def main() -> int:
    r"""重建并验证80个entries，随后发布catalog index与final summary。"""

    from anymani.pregrasp import active_mask_digest
    from anymani.pregrasp.good_catalog import (
        GoodPregraspCandidate,
        GoodPregraspCatalog,
        GoodPregraspEntry,
        GoodPregraspKey,
        GoodPregraspMember,
        GoodPregraspMetrics,
    )
    from anymani.pregrasp.strict_gate import MVP80_STRICT_GOOD_PREGRASP_GATE
    from anymani.tasks.hetero.config.generated.scene import ASSET_BINDING, RESOLVED_DEX_CUBE_SHA256
    from anymani.tasks.hetero.config.generated.strict_good_pregrasp_identity import (
        STRICT_GOOD_PREGRASP_GENERATION_DIGEST,
        STRICT_GOOD_PREGRASP_GENERATION_IDENTITY,
        STRICT_GOOD_PREGRASP_PHYSICS_DIGEST,
    )

    # 所有source必须来自当前同一generation/physics/gate协议；混入旧v4或中间算法立即拒绝。
    source_data: list[tuple[Path, dict[str, Any], Any]] = []
    for path, document in zip(source_paths, source_documents, strict=True):
        if document.get("generation_identity_digest") != STRICT_GOOD_PREGRASP_GENERATION_DIGEST:
            raise RuntimeError(f"source generation identity mismatch: {path}")
        if document.get("physics_identity_digest") != STRICT_GOOD_PREGRASP_PHYSICS_DIGEST:
            raise RuntimeError(f"source physics identity mismatch: {path}")
        if document.get("strict_gate_digest") != MVP80_STRICT_GOOD_PREGRASP_GATE.digest:
            raise RuntimeError(f"source strict gate mismatch: {path}")
        candidate_npz = Path(document["candidate_npz"])
        candidate_npz = candidate_npz if candidate_npz.is_absolute() else ROOT / candidate_npz
        source_data.append((path, document, np.load(candidate_npz, allow_pickle=False)))

    # 同一row若出现在多个source，选择strict通过数更多者；tie以summary path排序，保持确定性。
    evidence_by_row: dict[int, tuple[Path, dict[str, Any], Any, int, int]] = {}
    for path, document, arrays in sorted(source_data, key=lambda item: str(item[0])):
        rows = [int(value) for value in arrays["dataset_rows"]]
        summary_count = {int(item["dataset_row"]): int(item["strict_passed_candidates"]) for item in document["asset_candidate_counts"]}
        for asset_index, row in enumerate(rows):
            count = int(arrays["passed"][asset_index].sum())
            if count != summary_count[row]:
                raise RuntimeError(f"summary/NPZ strict count mismatch for row {row} in {path}")
            previous = evidence_by_row.get(row)
            if count >= 8 and (previous is None or count > previous[4]):
                evidence_by_row[row] = (path, document, arrays, asset_index, count)
    missing = sorted(set(selected_rows) - set(evidence_by_row))
    if missing:
        raise RuntimeError(f"final pair selection lacks strict Top-8 evidence for rows {missing}")

    finger_names = ("index", "middle", "ring", "thumb")
    entries: list[GoodPregraspEntry] = []
    source_usage: dict[str, list[int]] = {}
    for final_index, dataset_row in enumerate(selected_rows):
        path, document, arrays, asset_index, _ = evidence_by_row[dataset_row]
        passed_indices = np.flatnonzero(arrays["passed"][asset_index])
        quality = (
            1000.0
            - 100.0 * arrays["violation"][asset_index, passed_indices]
            + arrays["palm_fraction"][asset_index, passed_indices]
            + arrays["joint_margin"][asset_index, passed_indices]
            - arrays["displacement_m"][asset_index, passed_indices] / 0.005
            - arrays["peak_angular_rad_s"][asset_index, passed_indices] / 2.0
        )
        ranked = passed_indices[np.argsort(-quality, kind="stable")[:8]]
        members: list[GoodPregraspMember] = []
        for rank, candidate_index_value in enumerate(ranked):
            candidate_index = int(candidate_index_value)
            pair = arrays["pair"][asset_index, candidate_index]
            distances = arrays["distances"][asset_index, candidate_index]
            q = arrays["q"][asset_index, candidate_index]
            position = arrays["position"][asset_index, candidate_index]
            metrics = GoodPregraspMetrics(
                joint_limit_margin_fraction=float(arrays["joint_margin"][asset_index, candidate_index]),
                envelope_fingers=("thumb", finger_names[int(pair[0])], finger_names[int(pair[1])]),
                envelope_sector_min_deg=float(arrays["sector_deg"][asset_index, candidate_index]),
                envelope_tip_center_distance_m=(float(distances[0]), float(distances[1]), float(distances[2])),
                penetration_depth_max_m=float(arrays["penetration_m"][asset_index, candidate_index]),
                object_displacement_max_m=float(arrays["displacement_m"][asset_index, candidate_index]),
                object_tilt_max_deg=float(arrays["tilt_deg"][asset_index, candidate_index]),
                peak_linear_velocity_m_s=float(arrays["peak_linear_m_s"][asset_index, candidate_index]),
                peak_off_axis_angular_velocity_rad_s=float(
                    arrays["peak_off_axis_angular_rad_s"][asset_index, candidate_index]
                ),
                palm_contact_fraction=float(arrays["palm_fraction"][asset_index, candidate_index]),
                owner_contact_fraction=tuple(
                    float(value) for value in arrays["owner_contact_fraction"][asset_index, candidate_index]
                ),
                peak_angular_velocity_rad_s=float(arrays["peak_angular_rad_s"][asset_index, candidate_index]),
            )
            candidate = GoodPregraspCandidate(
                q_state_rad=tuple(float(value) for value in q),
                q_target_rad=tuple(float(value) for value in q),
                active_joint_mask=tuple(bool(value) for value in ASSET_BINDING.active_joint_masks[final_index]),
                object_position_h_m=(float(position[0]), float(position[1]), float(position[2])),
            )
            score = (
                metrics.palm_contact_fraction,
                metrics.joint_limit_margin_fraction,
                metrics.envelope_sector_min_deg / 180.0,
                -max(metrics.envelope_tip_center_distance_m),
                -metrics.object_displacement_max_m,
                -metrics.object_tilt_max_deg / 10.0,
                -metrics.peak_linear_velocity_m_s,
                -float(metrics.peak_angular_velocity_rad_s or 0.0),
                -metrics.penetration_depth_max_m,
            )
            members.append(GoodPregraspMember(rank=rank, candidate=candidate, metrics=metrics, selection_score=score))

        artifact = ASSET_BINDING.canonical_artifacts[final_index]
        source_asset = ASSET_BINDING.source_assets[final_index]
        source_asset_ids = {
            int(item["dataset_row"]): str(item["asset_id"]) for item in document["asset_candidate_counts"]
        }
        if source_asset.asset_id != source_asset_ids[dataset_row]:
            raise RuntimeError("candidate NPZ asset identity disagrees with final binding")
        key = GoodPregraspKey(
            asset_id=source_asset.asset_id,
            source_content_hash=artifact.source_content_hash,
            physical_geometry_hash=artifact.physical_geometry_hash,
            canonical_schema_digest=artifact.schema_digest,
            routing_digest=active_mask_digest(artifact.routing.active_joint_mask),
            object_asset_id="DexCube",
            object_asset_sha256=RESOLVED_DEX_CUBE_SHA256,
            object_scale=1.1,
            physics_identity_digest=STRICT_GOOD_PREGRASP_PHYSICS_DIGEST,
            generation_identity_digest=STRICT_GOOD_PREGRASP_GENERATION_DIGEST,
        )
        entry = GoodPregraspEntry(key=key, members=tuple(members))
        MVP80_STRICT_GOOD_PREGRASP_GATE.validate_entry(entry)
        entries.append(entry)
        source_usage.setdefault(str(path), []).append(dataset_row)

    # 全部80×8先通过内存验证，再开始写最终catalog；失败不会留下部分selection。
    catalog_root = _resolved(args.catalog)
    catalog = GoodPregraspCatalog(catalog_root)
    published = []
    for dataset_row, entry in zip(selected_rows, entries, strict=True):
        index_entry = catalog.publish(entry)
        published.append(
            {
                "dataset_row": dataset_row,
                "asset_id": entry.key.asset_id,
                "key_digest": index_entry.key_digest,
                "entry_digest": index_entry.entry_digest,
                "strict_passed_candidates": evidence_by_row[dataset_row][4],
            }
        )

    final_summary = {
        "artifact_type": "anymani.good_pregrasp.strict_generation_summary",
        "schema_version": "1.1.0",
        "assembly_algorithm": "strict-evidence-pair-fallback-v1",
        "dataset_rows": list(selected_rows),
        "selected_asset_count": 80,
        "rejected_pairs": selection["rejected_pairs"],
        "generation_identity": STRICT_GOOD_PREGRASP_GENERATION_IDENTITY,
        "generation_identity_digest": STRICT_GOOD_PREGRASP_GENERATION_DIGEST,
        "physics_identity_digest": STRICT_GOOD_PREGRASP_PHYSICS_DIGEST,
        "strict_gate_digest": MVP80_STRICT_GOOD_PREGRASP_GATE.digest,
        "source_summaries": [
            {"path": str(path), "sha256": _sha256(path), "selected_rows": source_usage.get(str(path), [])}
            for path in source_paths
        ],
        "catalog_root": str(catalog_root),
        "published_count": len(published),
        "failed_count": 0,
        "failed": [],
        "published": published,
        "formal_all_80_top8_passed": len(published) == 80,
    }
    summary_path = _resolved(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(final_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    print(
        {
            "summary": str(summary_path),
            "catalog": str(catalog_root),
            "published": len(published),
            "rejected_pairs": len(selection["rejected_pairs"]),
        },
        flush=True,
    )
    return 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
