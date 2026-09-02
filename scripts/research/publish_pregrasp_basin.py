r"""把严格point replay与local-perturbation evidence发布为production pregrasp basin record。

本脚本不运行物理仿真，也不从“最佳candidate”猜测缺失字段。它只接受schema-2.1 nominal point与
schema-3 point-search basin artifact，重新验证所有内嵌record、中心零扰动、binomial成功数、gate、
physical identity和source hashes。首版scale certificate只发布实际测试anchor的退化闭区间$[s,s]$；
没有额外scale stress时绝不外推到相邻scale。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

from anymani.pregrasp import (
    AtomicPregraspCache,
    FilePregraspProvider,
    PregraspCoverage,
    PregraspQuery,
    PregraspRecord,
    PregraspTier,
    ScaleCertificate,
    ScaleStressSample,
    certify_pregrasp,
    tier_satisfies,
)


def _parse_args() -> argparse.Namespace:
    r"""解析只读evidence路径、目标asset row与production cache root。"""

    parser = argparse.ArgumentParser(description=__doc__)  # CLI同时作为可审计复现入口
    parser.add_argument("--nominal-artifact", type=Path, required=True)  # 至少两次exact replay后的point artifact
    parser.add_argument("--basin-artifact", type=Path, required=True)  # 含中心+随机扰动trials的artifact
    parser.add_argument("--dataset-row", type=int, required=True)  # 只作artifact selection，不进入lookup identity
    parser.add_argument("--cache-root", type=Path, required=True)  # AtomicPregraspCache根目录
    parser.add_argument(
        "--minimum-tier",
        choices=("support_basin", "contact_basin"),
        default="contact_basin",
    )  # 认证的binomial成功事件；必须与basin search identity一致
    return parser.parse_args()


def _sha256(path: Path) -> str:
    r"""返回输入evidence原始bytes的SHA-256，绑定格式与内容。"""

    digest = hashlib.sha256()  # 分块读取避免未来大artifact产生额外完整bytes副本
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_document(path: Path, *, artifact_type: str) -> dict[str, Any]:
    r"""读取JSON object并验证外层artifact type，不容忍静默格式转换。"""

    document = json.loads(path.read_text(encoding="utf-8"))  # JSON parser拒绝语法损坏
    if not isinstance(document, dict) or document.get("artifact_type") != artifact_type:
        raise ValueError(f"{path} is not a {artifact_type} document")
    return document


def _select_row(items: list[dict[str, Any]], row: int, *, label: str) -> dict[str, Any]:
    r"""从per-asset列表选择唯一dataset row，拒绝缺失与重复。"""

    matches = [item for item in items if int(item["dataset_row"]) == row]  # row只定位evidence来源
    if len(matches) != 1:
        raise ValueError(f"{label} must contain exactly one dataset_row={row} item")
    return matches[0]


def _max_abs_difference(left: list[float], right: tuple[float, ...]) -> float:
    r"""计算两个固定维向量的$L_\infty$差，供中心trial identity复核。"""

    if len(left) != len(right):
        return math.inf
    return max(abs(float(a) - float(b)) for a, b in zip(left, right))  # 量纲沿输入保持rad或m


def main() -> int:
    r"""验证point/basin证据，构造窄scale证书并原子发布。

    发布记录使用nominal artifact中的reset candidate$(q_s,q_t,T_{ho})$，因为basin trials正是围绕该输入
    扰动；point metrics使用basin的零扰动中心trial，保证最终证书引用同一独立复核。Local basin成功定义为
    trial tier至少达到显式``--minimum-tier``，统计量为$k/n$，并再次与gate的$p_{\min}=0.8$比较。

    Returns:
        int: 成功发布并由provider反向解析后返回0。

    Raises:
        ValueError: 任一identity、中心状态、trial统计、tier或scale证据不一致。
    """

    args = _parse_args()  # 用户提供的路径、row、tier与cache边界不在脚本内隐式搜索
    nominal_path = args.nominal_artifact.resolve()  # evidence hash绑定resolved文件内容
    basin_path = args.basin_artifact.resolve()
    nominal = _load_document(nominal_path, artifact_type="anymani.pregrasp.point_search")
    basin = _load_document(basin_path, artifact_type="anymani.pregrasp.point_search")
    if basin.get("schema_version") != "3.0.0" or basin.get("portfolio") != "basin":
        raise ValueError("basin evidence must use point-search schema 3.0.0 and basin portfolio")

    minimum_tier = PregraspTier(args.minimum_tier)  # support/contact使用同一物理门但不同接触成功事件

    # Nominal reset candidate必须来自独立exact replay，并达到本次要求的point tier。
    nominal_item = _select_row(nominal["gate_frontier"], args.dataset_row, label="nominal gate_frontier")
    nominal_record = PregraspRecord.from_dict(nominal_item["record"])  # 重算record与lookup digests
    if nominal_record.coverage != PregraspCoverage.POINT or not tier_satisfies(nominal_record.tier, minimum_tier):
        raise ValueError("nominal record does not satisfy the requested replayed point tier")

    # Basin trials逐条重新解析，防止只信任outer summary或tier histogram。
    trial_items = [point for point in basin["points"] if int(point["dataset_row"]) == args.dataset_row]
    if not trial_items:
        raise ValueError("basin artifact contains no trials for requested dataset row")
    trial_records = [PregraspRecord.from_dict(point["record"]) for point in trial_items]
    if any(record.coverage == PregraspCoverage.BASIN for record in trial_records):
        raise ValueError("raw basin trials cannot carry pre-existing basin certificates")
    if any(record.gate != nominal_record.gate for record in trial_records):
        raise ValueError("nominal and basin trials disagree on gate")
    if any(record.lookup_key.physics_identity != nominal_record.lookup_key.physics_identity for record in trial_records):
        raise ValueError("nominal and basin trials disagree on physics identity")

    # Env 0是协议显式保留的零扰动中心；状态、pose和twist必须与nominal candidate严格相符。
    center_items = [point for point in trial_items if int(point["env_id"]) == 0]
    if len(center_items) != 1:
        raise ValueError("single-asset basin artifact must contain exactly one env-0 center trial")
    center_item = center_items[0]
    center_index = trial_items.index(center_item)
    center_record = trial_records[center_index]
    candidate = nominal_record.candidate  # production reset输入就是local perturbation分布的中心
    if _max_abs_difference(center_item["initial_q_state_rad"], candidate.q_state_rad) > 1.0e-7:
        raise ValueError("basin center q_state does not match nominal candidate")
    if _max_abs_difference(center_item["initial_object_position_h_m"], candidate.object_position_h_m) > 1.0e-7:
        raise ValueError("basin center object position does not match nominal candidate")
    center_quaternion = tuple(float(value) for value in center_item["initial_object_orientation_h_wxyz"])
    quaternion_dot = abs(sum(left * right for left, right in zip(center_quaternion, candidate.object_orientation_wxyz)))
    if 1.0 - quaternion_dot > 1.0e-7:  # $q$与$-q$表示同一SO(3) orientation
        raise ValueError("basin center object orientation does not match nominal candidate")
    if any(float(value) != 0.0 for value in center_item["initial_linear_velocity_h_m_s"]):
        raise ValueError("basin center linear velocity must be zero")
    if any(float(value) != 0.0 for value in center_item["initial_angular_velocity_h_rad_s"]):
        raise ValueError("basin center angular velocity must be zero")
    if center_record.candidate.q_target_rad != candidate.q_target_rad:
        raise ValueError("basin center PD preload target does not match nominal candidate")
    if not tier_satisfies(center_record.tier, minimum_tier):
        raise ValueError("zero-perturbation center trial did not reproduce requested tier")

    # 从逐trial严格record重算$k/n$，并与outer sufficient statistics双向核对。
    tier_successes = sum(tier_satisfies(record.tier, minimum_tier) for record in trial_records)
    # $k$：完整minimum-tier point gate通过数；support与contact都先通过相同稳定物理门
    trial_count = len(trial_records)  # $n$：中心+随机local perturbations总数
    summary = _select_row(basin["basin_summary"], args.dataset_row, label="basin summary")
    if "minimum_tier" in summary:
        if str(summary["minimum_tier"]) != minimum_tier.value:
            raise ValueError("basin summary minimum tier disagrees with publication request")
        reported_successes = int(summary["tier_successes"])
    else:
        if minimum_tier != PregraspTier.CONTACT_BASIN:
            raise ValueError("legacy basin summary can only certify contact tier")
        reported_successes = int(summary["contact_successes"])
    if int(summary["trials"]) != trial_count or reported_successes != tier_successes:
        raise ValueError("basin outer summary disagrees with strict trial records")
    success_fraction = tier_successes / trial_count  # binomial sufficient statistic $\hat p=k/n$
    if not bool(summary["passed"]) or success_fraction < nominal_record.gate.min_basin_success_fraction:
        raise ValueError("local perturbation basin does not satisfy configured success fraction")

    # 当前只测试一个absolute prestartup scale，因此发布退化闭区间$[s,s]$，不外推scale邻域。
    scale = float(basin["scale"])
    anchor = format(scale, ".12g")
    if anchor not in {"1.1", "1.2", "1.25"} or abs(candidate.object_scale - scale) > 1.0e-8:
        raise ValueError("candidate and basin evidence disagree on sealed scale anchor")
    physics_snapshot = {
        "object_mass_kg": basin["actual_object_mass_kg"][0],
        "object_inertia_kg_m2": basin["actual_object_inertia_kg_m2"][0],
        "cube_sha256": basin["cube_sha256"],
        "physics_identity": basin["physics_identity"],
        "nominal_artifact_sha256": _sha256(nominal_path),
        "basin_artifact_sha256": _sha256(basin_path),
    }  # actual PhysX mass/inertia绑定该prestartup scale，不从density理论外推
    scale_sample = ScaleStressSample(
        scale=scale,
        passed=True,
        reason_codes=(),
        physics_snapshot=physics_snapshot,
    )
    certificate = ScaleCertificate(
        anchor=anchor,  # type: ignore[arg-type]  # 上方集合检查已把str缩窄到三个sealed Literal
        scale_min=scale,
        scale_max=scale,
        scale_samples=(scale_sample,),
        perturbation_trials=trial_count,
        perturbation_successes=tier_successes,
        gravity_directions_passed=0,  # 当前论文palm-up contact tier不宣称六轴gravity robustness
    )

    # 新lookup identity显式绑定point lineage与basin protocol；dataset row仍只保留provenance，不进入digest。
    basin_search_identity = trial_records[0].lookup_key.to_dict()["search_identity"]
    search_identity = {
        "algorithm": "hetero-contact-basin-certification-v1",
        "candidate_state_semantics": "separate_actual_q_state_and_pd_preload_target",
        "nominal_record_digest": nominal_record.digest,
        "nominal_artifact_sha256": _sha256(nominal_path),
        "basin_artifact_sha256": _sha256(basin_path),
        "basin_protocol": basin_search_identity,
        "minimum_tier": minimum_tier.value,
        "perturbation_trials": trial_count,
        "perturbation_successes": tier_successes,
    }
    lookup_key = replace(nominal_record.lookup_key, search_identity=search_identity)
    certified = certify_pregrasp(
        lookup_key=lookup_key,
        candidate=candidate,
        metrics=center_record.metrics,
        gate=nominal_record.gate,
        coverage=PregraspCoverage.BASIN,
        scale_certificate=certificate,
    )
    if certified.tier != center_record.tier or not tier_satisfies(certified.tier, minimum_tier):
        raise ValueError("published basin unexpectedly changed the independently replayed center tier")

    # Payload先于index原子发布；随后从provider边界按exact identity/tier/coverage/scale反向查询。
    cache = AtomicPregraspCache(args.cache_root)
    entry = cache.publish(certified)
    resolution = FilePregraspProvider(args.cache_root).resolve(
        PregraspQuery(
            lookup_key=certified.lookup_key,
            requested_scale=scale,
            min_tier=minimum_tier,
            require_basin=True,
        )
    )
    if resolution.record.digest != certified.digest or resolution.index_entry != entry:
        raise RuntimeError("provider read-back disagrees with atomically published record")

    print(
        json.dumps(
            {
                "cache_root": str(cache.root),
                "payload": str(cache.payload_path(entry)),
                "record_digest": certified.digest,
                "lookup_digest": certified.lookup_key.digest,
                "tier": certified.tier.value,
                "coverage": certified.coverage.value,
                "scale_interval": [certificate.scale_min, certificate.scale_max],
                "minimum_tier": minimum_tier.value,
                "basin_successes": tier_successes,
                "basin_trials": trial_count,
                "basin_success_fraction": success_fraction,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
