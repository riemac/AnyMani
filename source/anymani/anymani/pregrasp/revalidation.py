r"""Allegro128 局部关节限位修订的有界纯 CPU 输入合同。

本模块只处理一个独立的 original-top8-with-projection 候选 artifact：

* 候选来自旧 catalog 每个资产已经排序的 Top-8，扩限的 non-thumb j1
  才允许做一次向上的投影；它不是 Sobol/CEM 搜索结果。
* generation identity 明确复用原 strict 1 s 物理门，但候选本身仍是
  candidates-only-not-physically-certified，不能进入已认证 catalog。
* NPZ 中的 hand state 使用 canonical 16 槽、弧度与 zero ghost；object
  orientation 固定为 hand-frame upright 的 wxyz=(1,0,0,0)。

这里故意不导入 tasks、robots、Isaac Lab、Torch 或 CUDA。调用方负责
把返回候选绑定到修订后资产并运行真实 PhysX strict gate；本模块的职责是让
文件身份、来源、重排和输入语义在进入物理编排前可重复、可审计地失败关闭。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

from .schema import canonical_json_bytes

# 独立 artifact/kind/schema 三元组，避免被历史 c82 catalog 当作同一种生成协议。
REVALIDATION_GENERATION_ARTIFACT_TYPE = "anymani.pregrasp.revalidation_generation_identity"
REVALIDATION_GENERATION_KIND = "allegro128.local_joint_limit_revalidation"
REVALIDATION_GENERATION_SCHEMA_VERSION = "1.0.0"
REVALIDATION_GENERATION_PROTOCOL = "allegro128-local-joint-limit-revalidation-v1"
REVALIDATION_REFINEMENT_GENERATION_ARTIFACT_TYPE = "anymani.pregrasp.revalidation_refinement_generation_identity"
REVALIDATION_REFINEMENT_GENERATION_KIND = "allegro128.local_joint_limit_revalidation_with_cem"
REVALIDATION_REFINEMENT_GENERATION_SCHEMA_VERSION = "2.0.0"
REVALIDATION_REFINEMENT_GENERATION_PROTOCOL = "allegro128-local-joint-limit-revalidation-cem-v2"

# 候选来源和发布基数是固定协议边界；每个资产必须保留 parent rank 0..7。
REVALIDATION_CANDIDATE_SOURCE = "original-top8-with-projection"
REVALIDATION_CANDIDATE_COUNT = 8
REVALIDATION_RANKING = "preserve-parent-rank"
REVALIDATION_MAX_ASSET_COUNT = 128
REVALIDATION_SCENE_SEED = 20260902
REVALIDATION_REFINED_RANKING = "preserve-parent-rank-when-all-initial-pass-else-physical-quality"

# 投影进入新 upper limit 后仍要求归一化余量达到 10.1%，而原 strict gate
# 仍是 10%。最大差值是本轮已生成 fixture 的可审计数值锚点，单位 rad。
REVALIDATION_PROJECTION_MARGIN_FRACTION = 0.101
REVALIDATION_STRICT_MARGIN_FRACTION = 0.10
REVALIDATION_MAX_PROJECTION_DELTA_RAD = 0.06453248217813295

# 原 strict cold-reset 物理门的观测窗口；这里记录协议语义，不执行仿真。
REVALIDATION_STRICT_PHYSICS_WINDOW_SECONDS = 1.0
REVALIDATION_UPRIGHT_QUATERNION_WXYZ = (1.0, 0.0, 0.0, 0.0)

# canonical hand layout 中只有这些 non-thumb j1 槽允许发生 upward projection。
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REQUIRED_NPZ_KEYS = frozenset(
    {
        "original_q_rad",
        "projected_q_rad",
        "object_position_h_m",
        "object_orientation_h_wxyz",
        "active_joint_mask",
        "parent_asset_id",
        "new_asset_id",
        "parent_entry_digest",
        "canonical_joint_names",
    }
)
_TARGET_KEY_MARKERS = ("target", "qtarget")


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    r"""计算字段排序且拒绝 NaN 的 canonical JSON SHA-256。

    该摘要只用于 generation identity 等 JSON 文档。NPZ 候选文件的身份必须
    直接对 path.read_bytes() 求 SHA-256，不能把压缩包内容重新转成 JSON
    后冒充文件摘要。

    Args:
        payload: 由 JSON 原生标量、mapping、list 或 tuple 构成的有限 mapping。

    Returns:
        str: 64 位小写 SHA-256 十六进制字符串。

    Raises:
        ValueError: payload 含非有限数或不可 JSON 序列化的值。
    """

    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()  # identity 的稳定内容摘要


def stable_digest(payload: Mapping[str, Any]) -> str:
    r"""提供与 pregrasp schema 相同语义的 canonical JSON digest 别名。"""

    return canonical_json_sha256(payload)  # 统一公开一个纯 CPU digest 入口


def _validate_sha256(value: Any, field_name: str) -> str:
    r"""严格接受小写 64 位 SHA-256 字符串，不替 caller 修复大小写。"""

    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a 64-character lowercase SHA-256")
    return value


def _fixed_float(value: Any, *, expected: float, field_name: str) -> float:
    r"""验证固定协议浮点常数，拒绝 bool、NaN、无穷或另一套门限。"""

    if isinstance(value, bool):
        raise ValueError(f"{field_name} must equal fixed protocol value {expected}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must equal fixed protocol value {expected}") from error
    if not math.isfinite(parsed) or not math.isclose(parsed, expected, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"{field_name} must equal fixed protocol value {expected}")
    return expected


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    r"""把 identity 的嵌套字段限定为 mapping，避免字符串迭代造成假通过。"""

    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


_REFINEMENT_REQUIRED_KEYS = frozenset(
    {
        "elite_physical_candidates",
        "elite_proposal_counts_descending",
        "full_physics_candidates_per_round",
        "joint_pca_dimensions",
        "object_position_dimensions",
        "per_asset_manual_parameters",
        "position_center_feedback",
        "proposals_per_round_per_failed_asset",
        "random_stream_key",
        "rounds_max",
        "settle_height_feedback",
        "strict_mode_exploitation",
        "type",
    }
)
_REFINEMENT_CENTER_COUNTS = (24, 20, 16, 12, 8, 8, 6, 6, 4, 4, 4, 4, 3, 3, 3, 3)


def _json_mapping_copy(value: Any, field_name: str) -> dict[str, Any]:
    r"""以 canonical JSON round-trip 固化嵌套 profile，避免 caller 后续改写 identity。"""

    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    try:
        normalized = json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{field_name} must contain finite JSON values") from error
    if not isinstance(normalized, dict):
        raise ValueError(f"{field_name} must be a mapping")
    return normalized


def _fixed_int(value: Any, *, expected: int, field_name: str) -> int:
    r"""验证 profile 中不可漂移的整数预算，区别 bool、float 和实际计数。"""

    if not isinstance(value, Integral) or isinstance(value, bool) or int(value) != expected:
        raise ValueError(f"{field_name} must equal fixed protocol value {expected}")
    return expected


def _validate_refinement_identity(value: Any, field_name: str = "refinement") -> dict[str, Any]:
    r"""逐字段验证现有 strict low-rank CEM 的完整 refinement identity。"""

    refinement = _json_mapping_copy(value, field_name)
    if set(refinement) != _REFINEMENT_REQUIRED_KEYS:
        missing = sorted(_REFINEMENT_REQUIRED_KEYS - set(refinement))
        extra = sorted(set(refinement) - _REFINEMENT_REQUIRED_KEYS)
        raise ValueError(f"{field_name} keys drifted; missing={missing}, extra={extra}")
    _fixed_int(
        refinement["elite_physical_candidates"], expected=16, field_name=f"{field_name}.elite_physical_candidates"
    )
    counts = refinement["elite_proposal_counts_descending"]
    if counts != list(_REFINEMENT_CENTER_COUNTS):
        raise ValueError(f"{field_name}.elite_proposal_counts_descending must preserve the legacy allocation")
    _fixed_int(
        refinement["full_physics_candidates_per_round"],
        expected=128,
        field_name=f"{field_name}.full_physics_candidates_per_round",
    )
    _fixed_int(refinement["joint_pca_dimensions"], expected=4, field_name=f"{field_name}.joint_pca_dimensions")
    _fixed_int(
        refinement["object_position_dimensions"],
        expected=3,
        field_name=f"{field_name}.object_position_dimensions",
    )
    if refinement["per_asset_manual_parameters"] is not False:
        raise ValueError(f"{field_name}.per_asset_manual_parameters must be false")
    if refinement["position_center_feedback"] != "minus_physx_contact_normal_times_1p10_depth_plus_0p25mm":
        raise ValueError(f"{field_name}.position_center_feedback drifted")
    _fixed_int(
        refinement["proposals_per_round_per_failed_asset"],
        expected=128,
        field_name=f"{field_name}.proposals_per_round_per_failed_asset",
    )
    if refinement["random_stream_key"] != "source_content_and_physical_geometry_sha256":
        raise ValueError(f"{field_name}.random_stream_key drifted")
    _fixed_int(refinement["rounds_max"], expected=3, field_name=f"{field_name}.rounds_max")
    if refinement["settle_height_feedback"] != (
        "if_initial_penetration_le_0p5mm_lower_by_clamped_displacement_minus_4p5mm"
    ):
        raise ValueError(f"{field_name}.settle_height_feedback drifted")
    if refinement["type"] != "per_asset_elite_mixture_low_rank_gaussian_cem":
        raise ValueError(f"{field_name}.type must identify the existing low-rank CEM")
    exploitation = _json_mapping_copy(refinement["strict_mode_exploitation"], f"{field_name}.strict_mode_exploitation")
    if set(exploitation) != {"activation", "joint_std_rad", "max_proposals_per_round", "position_std_m"}:
        raise ValueError(f"{field_name}.strict_mode_exploitation keys drifted")
    if exploitation["activation"] != "one_to_seven_strict_or_normalized_gate_violation_le_0p35":
        raise ValueError(f"{field_name}.strict_mode_exploitation.activation drifted")
    _fixed_float(
        exploitation["joint_std_rad"],
        expected=0.0005,
        field_name=f"{field_name}.strict_mode_exploitation.joint_std_rad",
    )
    _fixed_int(
        exploitation["max_proposals_per_round"],
        expected=96,
        field_name=f"{field_name}.strict_mode_exploitation.max_proposals_per_round",
    )
    position_std = exploitation["position_std_m"]
    if position_std != [5e-05, 5e-05, 2.5e-05]:
        raise ValueError(f"{field_name}.strict_mode_exploitation.position_std_m drifted")
    return refinement


def build_revalidation_generation_identity(
    *,
    candidate_npz_sha256: str,
    source_catalog_index_sha256: str,
    strict_gate_digest: str,
    physics_identity_digest: str,
    candidate_count: int = REVALIDATION_CANDIDATE_COUNT,
    projection_margin_fraction: float = REVALIDATION_PROJECTION_MARGIN_FRACTION,
    refinement_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    r"""构造 Allegro128 局部关节限位修订的独立 generation identity。

    该协议从原 catalog 的每资产八个 parent-ranked candidates 出发，仅对新
    non-thumb j1 upper limit 做预先计算的 clip 投影：

    q_projected_j = clip(q_original_j, lower_new_j + 0.101 Delta_j,
    upper_new_j - 0.101 Delta_j).

    其中本输入合同只认证“投影候选来自旧 Top-8、目标余量为 10.1%”这层
    语义；新资产上下限与真实接触动力学必须由上层物理 runner 提供。原
    strict 1 s physics gate 的 digest 在 identity 中原样绑定，strict margin
    仍固定为 10%。本协议没有 Sobol、CEM、随机 seed 或 candidate search。

    Args:
        candidate_npz_sha256: 候选 NPZ 原始 bytes 的 SHA-256。
        source_catalog_index_sha256: parent Top-8 catalog index 的内容摘要。
        strict_gate_digest: 原 strict 1 s gate 的固定摘要。
        physics_identity_digest: 原 strict 物理参数 identity 的固定摘要。
        candidate_count: 每资产候选数；本协议固定为 8。
        projection_margin_fraction: 投影后的归一化余量；本协议固定为 0.101。
        refinement_identity: 可选的既有 strict low-rank CEM profile；为 None 时
            必须逐值保留已发布 v1 identity，为 mapping 时启用独立三轮续跑协议。

    Returns:
        dict[str, Any]: 可嵌入新 canonical lock selection 的 JSON-safe identity。

    Raises:
        ValueError: 任一摘要非法，或试图传入不同候选基数/投影门限。
    """

    candidate_sha = _validate_sha256(candidate_npz_sha256, "candidate_npz_sha256")
    source_sha = _validate_sha256(source_catalog_index_sha256, "source_catalog_index_sha256")
    strict_sha = _validate_sha256(strict_gate_digest, "strict_gate_digest")
    physics_sha = _validate_sha256(physics_identity_digest, "physics_identity_digest")
    try:
        parsed_count = int(candidate_count)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("candidate_count must be fixed at 8 for revalidation") from error
    if (
        not isinstance(candidate_count, Integral)
        or isinstance(candidate_count, bool)
        or parsed_count != candidate_count
    ):
        raise ValueError("candidate_count must be fixed at 8 for revalidation")
    if parsed_count != REVALIDATION_CANDIDATE_COUNT:
        raise ValueError("candidate_count must be fixed at 8 for revalidation")
    projection_margin = _fixed_float(
        projection_margin_fraction,
        expected=REVALIDATION_PROJECTION_MARGIN_FRACTION,
        field_name="projection_margin_fraction",
    )

    # 顶层 digest 字段便于 runner/task binding 直接核对；嵌套段落同时把科学语义
    # 写入可审计 identity，防止只有一串摘要而看不出它复用了哪条物理门。
    base_identity = {
        "artifact_type": REVALIDATION_GENERATION_ARTIFACT_TYPE,
        "kind": REVALIDATION_GENERATION_KIND,
        "schema_version": REVALIDATION_GENERATION_SCHEMA_VERSION,
        "protocol": REVALIDATION_GENERATION_PROTOCOL,
        "candidate_source": REVALIDATION_CANDIDATE_SOURCE,
        "candidate_count": REVALIDATION_CANDIDATE_COUNT,
        "candidate_count_per_asset": REVALIDATION_CANDIDATE_COUNT,
        "projection_margin_fraction": projection_margin,
        "strict_gate_margin_fraction": REVALIDATION_STRICT_MARGIN_FRACTION,
        "candidate_npz_sha256": candidate_sha,
        "source_catalog_index_sha256": source_sha,
        "strict_gate_digest": strict_sha,
        "physics_identity_digest": physics_sha,
        "strict_physics_gate": {
            "name": "original-strict-1s-physics-gate",
            "window_seconds": REVALIDATION_STRICT_PHYSICS_WINDOW_SECONDS,
            "margin_fraction": REVALIDATION_STRICT_MARGIN_FRACTION,
            "gate_digest": strict_sha,
        },
        "projection": {
            "operation": "clip-to-new-limit-with-margin",
            "allowed_joint_class": "non-thumb-j1-upper",
            "max_delta_rad": REVALIDATION_MAX_PROJECTION_DELTA_RAD,
        },
        "search": {
            "source": REVALIDATION_CANDIDATE_SOURCE,
            "uses_sobol": False,
            "uses_cem": False,
            "sobol": False,
            "cem": False,
            "randomized": False,
        },
        "ranking": REVALIDATION_RANKING,
        "publication": {
            "candidate_count_per_asset": REVALIDATION_CANDIDATE_COUNT,
            "ranking": REVALIDATION_RANKING,
            "physical_certification_required": True,
        },
        "candidate_status": "candidates-only-not-physically-certified",
    }
    if refinement_identity is None:
        return base_identity  # v1 的字典和 canonical digest 保持逐值不变

    refinement = _validate_refinement_identity(refinement_identity)
    # 新身份保留同一初始 Top-8/source/physics/gate，同时把“只对不足8项续跑”
    # 和旧 CEM 的中心/elite 分配写入 identity；这些字段会进入 stable_digest。
    return {
        **base_identity,
        "artifact_type": REVALIDATION_REFINEMENT_GENERATION_ARTIFACT_TYPE,
        "kind": REVALIDATION_REFINEMENT_GENERATION_KIND,
        "schema_version": REVALIDATION_REFINEMENT_GENERATION_SCHEMA_VERSION,
        "protocol": REVALIDATION_REFINEMENT_GENERATION_PROTOCOL,
        "seed": REVALIDATION_SCENE_SEED,
        "initial": {
            "candidate_source": REVALIDATION_CANDIDATE_SOURCE,
            "candidate_count": REVALIDATION_CANDIDATE_COUNT,
            "uses_sobol": False,
            "seed": REVALIDATION_SCENE_SEED,
        },
        "search": {
            "source": REVALIDATION_CANDIDATE_SOURCE,
            "uses_sobol": False,
            "uses_cem": True,
            "sobol": False,
            "cem": True,
            "randomized": True,
        },
        "refinement_rounds": 3,
        "refinement": refinement,
        "refinement_scope": "assets-with-fewer-than-8-initial-strict-passes",
        "cem_continuation": {
            "initial_elite_count": 8,
            "initial_center_assignment": "round_robin",
            "later_elite_count": 16,
            "later_center_assignment": "legacy_cem_proposal_counts_descending",
        },
        "ranking": REVALIDATION_REFINED_RANKING,
        "publication": {
            "candidate_count_per_asset": REVALIDATION_CANDIDATE_COUNT,
            "ranking": REVALIDATION_REFINED_RANKING,
            "physical_certification_required": True,
        },
        "source_hashes": {
            "candidate_npz_sha256": candidate_sha,
            "source_catalog_index_sha256": source_sha,
            "strict_gate_digest": strict_sha,
            "physics_identity_digest": physics_sha,
        },
    }


def _reject_c82_markers(document: Mapping[str, Any]) -> None:
    r"""拒绝把历史 c82 protocol/algorithm 字段塞进修订 identity。"""

    for field_name in ("protocol", "algorithm", "generation_protocol", "kind"):
        value = document.get(field_name)
        if isinstance(value, str) and "c82" in value.lower():
            raise ValueError("revalidation identity must not masquerade as legacy c82 protocol")


def validate_revalidation_generation_identity(document: Mapping[str, Any]) -> dict[str, Any]:
    r"""验证并返回一个修订 generation identity 原样 mapping。

    验证包含独立 artifact/kind/schema、候选来源、固定 8/asset、10.1% projection、
    原 strict 1 s/10% physics gate、无 Sobol/CEM，以及四个持久化摘要。允许未来
    添加与这些字段不冲突的 provenance 字段，但固定协议字段不能被覆盖。

    Args:
        document: 从 canonical lock 或 JSON 恢复的 identity mapping。

    Returns:
        dict[str, Any]: 通过验证的原 dict；非 dict mapping 会转成浅 dict。

    Raises:
        ValueError: identity 缺字段、摘要非法、协议常数不符或包含旧 c82 标记。
    """

    if not isinstance(document, Mapping):
        raise ValueError("revalidation generation identity must be a mapping")
    _reject_c82_markers(document)

    refined = document.get("artifact_type") == REVALIDATION_REFINEMENT_GENERATION_ARTIFACT_TYPE
    if not refined and ("refinement" in document or "refinement_rounds" in document):
        raise ValueError("legacy v1 identity cannot be upgraded by attaching refinement fields")
    expected_artifact = (
        REVALIDATION_REFINEMENT_GENERATION_ARTIFACT_TYPE if refined else REVALIDATION_GENERATION_ARTIFACT_TYPE
    )
    expected_kind = REVALIDATION_REFINEMENT_GENERATION_KIND if refined else REVALIDATION_GENERATION_KIND
    expected_schema = (
        REVALIDATION_REFINEMENT_GENERATION_SCHEMA_VERSION if refined else REVALIDATION_GENERATION_SCHEMA_VERSION
    )
    expected_protocol = REVALIDATION_REFINEMENT_GENERATION_PROTOCOL if refined else REVALIDATION_GENERATION_PROTOCOL
    expected_ranking = REVALIDATION_REFINED_RANKING if refined else REVALIDATION_RANKING
    expected_strings = {
        "artifact_type": expected_artifact,
        "kind": expected_kind,
        "schema_version": expected_schema,
        "protocol": expected_protocol,
        "candidate_source": REVALIDATION_CANDIDATE_SOURCE,
        "ranking": expected_ranking,
        "candidate_status": "candidates-only-not-physically-certified",
    }
    for field_name, expected in expected_strings.items():
        if document.get(field_name) != expected:
            raise ValueError(f"{field_name} must equal {expected!r} for revalidation protocol")

    candidate_count = document.get("candidate_count")
    if (
        not isinstance(candidate_count, Integral)
        or isinstance(candidate_count, bool)
        or (candidate_count != REVALIDATION_CANDIDATE_COUNT)
    ):
        raise ValueError("candidate_count must be fixed at 8 for revalidation")
    if refined and document.get("candidate_count_per_asset") not in (None, REVALIDATION_CANDIDATE_COUNT):
        raise ValueError("candidate_count_per_asset must be fixed at 8 when present")
    _fixed_float(
        document.get("projection_margin_fraction"),
        expected=REVALIDATION_PROJECTION_MARGIN_FRACTION,
        field_name="projection_margin_fraction",
    )
    _fixed_float(
        document.get("strict_gate_margin_fraction"),
        expected=REVALIDATION_STRICT_MARGIN_FRACTION,
        field_name="strict_gate_margin_fraction",
    )
    for field_name in (
        "candidate_npz_sha256",
        "source_catalog_index_sha256",
        "strict_gate_digest",
        "physics_identity_digest",
    ):
        _validate_sha256(document.get(field_name), field_name)

    strict_gate = _require_mapping(document.get("strict_physics_gate"), "strict_physics_gate")
    if strict_gate.get("name") != "original-strict-1s-physics-gate":
        raise ValueError("strict_physics_gate must identify the original strict 1 s gate")
    _fixed_float(
        strict_gate.get("window_seconds"),
        expected=REVALIDATION_STRICT_PHYSICS_WINDOW_SECONDS,
        field_name="strict_physics_gate.window_seconds",
    )
    _fixed_float(
        strict_gate.get("margin_fraction"),
        expected=REVALIDATION_STRICT_MARGIN_FRACTION,
        field_name="strict_physics_gate.margin_fraction",
    )
    if strict_gate.get("gate_digest") != document.get("strict_gate_digest"):
        raise ValueError("strict_physics_gate.gate_digest must match strict_gate_digest")

    projection = _require_mapping(document.get("projection"), "projection")
    if projection.get("operation") != "clip-to-new-limit-with-margin":
        raise ValueError("projection operation must be clip-to-new-limit-with-margin")
    if projection.get("allowed_joint_class") != "non-thumb-j1-upper":
        raise ValueError("projection is restricted to non-thumb j1 upper joints")
    _fixed_float(
        projection.get("max_delta_rad"),
        expected=REVALIDATION_MAX_PROJECTION_DELTA_RAD,
        field_name="projection.max_delta_rad",
    )

    search = _require_mapping(document.get("search"), "search")
    if search.get("source") != REVALIDATION_CANDIDATE_SOURCE:
        raise ValueError("search source must be original-top8-with-projection")
    if search.get("uses_sobol") is not False:
        raise ValueError("revalidation protocol cannot enable Sobol search")
    if "sobol" in search and search.get("sobol") is not False:
        raise ValueError("revalidation protocol cannot enable Sobol search")
    expected_cem = refined
    if search.get("uses_cem") is not expected_cem:
        raise ValueError(f"revalidation search uses_cem must be {expected_cem}")
    if "cem" in search and search.get("cem") is not expected_cem:
        raise ValueError(f"revalidation search cem must be {expected_cem}")
    if search.get("randomized") is not refined:
        raise ValueError(f"revalidation search randomized must be {refined}")

    publication = _require_mapping(document.get("publication"), "publication")
    if publication.get("candidate_count_per_asset") != REVALIDATION_CANDIDATE_COUNT:
        raise ValueError("publication candidate count must be fixed at 8")
    if publication.get("ranking") != expected_ranking:
        raise ValueError("publication ranking does not match generation protocol")
    if publication.get("physical_certification_required") is not True:
        raise ValueError("revalidation publication must require physical certification")

    if refined:
        _fixed_int(document.get("seed"), expected=REVALIDATION_SCENE_SEED, field_name="seed")
        _fixed_int(
            document.get("refinement_rounds"),
            expected=3,
            field_name="refinement_rounds",
        )
        initial = _require_mapping(document.get("initial"), "initial")
        if initial.get("candidate_source") != REVALIDATION_CANDIDATE_SOURCE:
            raise ValueError("initial candidate source must be original-top8-with-projection")
        _fixed_int(initial.get("candidate_count"), expected=8, field_name="initial.candidate_count")
        _fixed_int(initial.get("seed"), expected=REVALIDATION_SCENE_SEED, field_name="initial.seed")
        if initial.get("uses_sobol") is not False:
            raise ValueError("revalidation initial proposal cannot use Sobol")
        if document.get("refinement_scope") != "assets-with-fewer-than-8-initial-strict-passes":
            raise ValueError("refinement_scope must target only initial strict failures")
        _validate_refinement_identity(document.get("refinement"))
        continuation = _require_mapping(document.get("cem_continuation"), "cem_continuation")
        if continuation.get("initial_elite_count") != 8:
            raise ValueError("initial CEM continuation must use eight parent elites")
        if continuation.get("initial_center_assignment") != "round_robin":
            raise ValueError("initial CEM continuation must use round-robin center assignment")
        if continuation.get("later_elite_count") != 16:
            raise ValueError("later CEM continuation must use sixteen elites")
        if continuation.get("later_center_assignment") != "legacy_cem_proposal_counts_descending":
            raise ValueError("later CEM continuation must preserve legacy center allocation")
        source_hashes = _require_mapping(document.get("source_hashes"), "source_hashes")
        for field_name in (
            "candidate_npz_sha256",
            "source_catalog_index_sha256",
            "strict_gate_digest",
            "physics_identity_digest",
        ):
            _validate_sha256(source_hashes.get(field_name), f"source_hashes.{field_name}")
            if source_hashes.get(field_name) != document.get(field_name):
                raise ValueError(f"source_hashes.{field_name} must match top-level digest")

    return dict(document) if not isinstance(document, dict) else document


def _string_array(array: np.ndarray, *, field_name: str) -> np.ndarray:
    r"""把 NPZ 中的 fixed-width Unicode/bytes 字符串规约为 Unicode 数组。"""

    if array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{field_name} must be a string array")
    if array.dtype.kind == "S":
        values = np.asarray([value.decode("utf-8") for value in array.reshape(-1)], dtype=str)
        return values.reshape(array.shape)
    return np.asarray(array, dtype=str)


def _numeric_array(array: np.ndarray, *, field_name: str) -> np.ndarray:
    r"""规约 real numeric array，并在任何 shape 判断前拒绝 NaN/Inf。"""

    if array.dtype.kind not in {"f", "i", "u"} or array.dtype.kind == "b":
        raise ValueError(f"{field_name} must be a real numeric array")
    if not np.isfinite(array).all():
        raise ValueError(f"{field_name} must contain only finite values")
    return np.asarray(array, dtype=np.float64)


def _validate_unique_ids(array: np.ndarray, *, field_name: str) -> np.ndarray:
    r"""验证非空字符串 ID 且拒绝重复 row key。"""

    if array.ndim != 1:
        raise ValueError(f"{field_name} must have shape [asset_count]")
    values = _string_array(array, field_name=field_name).reshape(-1)
    text_values = np.asarray([str(value) for value in values], dtype=str)
    if any(not value for value in text_values):
        raise ValueError(f"{field_name} must contain non-empty IDs")
    if len(set(text_values.tolist())) != len(text_values):
        raise ValueError(f"{field_name} must not contain duplicate IDs")
    return text_values


def _validate_entry_digests(array: np.ndarray) -> np.ndarray:
    r"""验证 parent entry digest 的持久化格式与 row provenance。"""

    if array.ndim != 1:
        raise ValueError("parent_entry_digest must have shape [asset_count]")
    values = _string_array(array, field_name="parent_entry_digest").reshape(-1)
    for value in values:
        _validate_sha256(str(value), "parent_entry_digest")
    return np.asarray(values, dtype=str)


def _validate_joint_names(array: np.ndarray) -> np.ndarray:
    r"""验证 canonical 16-slot joint names 唯一且非空。"""

    if array.ndim != 1:
        raise ValueError("canonical_joint_names must have shape [16]")
    values = _string_array(array, field_name="canonical_joint_names").reshape(-1)
    if len(values) != 16 or any(not str(value) for value in values):
        raise ValueError("canonical_joint_names must contain 16 non-empty names")
    if len(set(str(value) for value in values)) != 16:
        raise ValueError("canonical_joint_names must be unique")
    return np.asarray(values, dtype=str)


def _validate_projection(
    original_q: np.ndarray,
    projected_q: np.ndarray,
    joint_names: np.ndarray,
) -> None:
    r"""验证投影只向上作用于 non-thumb j1，并受本轮最大差值锚点约束。"""

    delta = projected_q - original_q  # $\Delta q=q^{projected}-q^{original}$，单位 rad
    changed = delta != 0.0  # 本轮 unchanged slots 必须逐位保持 parent state
    allowed_slots = np.asarray(
        [str(name).endswith("_j1") and not str(name).startswith("thumb_") for name in joint_names],
        dtype=np.bool_,
    )
    if np.any(changed[..., ~allowed_slots]):
        raise ValueError("projection may change only non-thumb j1 upper joints")
    allowed_delta = delta[..., allowed_slots]
    if np.any(allowed_delta < 0.0):
        raise ValueError("projection must be upward for expanded upper limits")
    if allowed_delta.size and float(np.max(allowed_delta)) > REVALIDATION_MAX_PROJECTION_DELTA_RAD + 1.0e-12:
        raise ValueError("projection delta exceeds fixed 0.06453248217813295 rad bound")


def _validate_external_masks(
    active_joint_masks: Any,
    *,
    requested_ids: Sequence[str],
    file_ids: np.ndarray,
    file_mask: np.ndarray,
) -> np.ndarray:
    r"""把 caller 的 canonical active masks 对齐到 requested new_asset_id 顺序。"""

    file_index = {str(asset_id): index for index, asset_id in enumerate(file_ids.tolist())}
    if isinstance(active_joint_masks, Mapping):
        selected: list[np.ndarray] = []
        for asset_id in requested_ids:
            if asset_id not in active_joint_masks:
                raise ValueError(f"active_joint_masks missing asset ID {asset_id!r}")
            candidate = np.asarray(active_joint_masks[asset_id])
            if candidate.shape != (16,) or candidate.dtype.kind != "b":
                raise ValueError(f"active_joint_masks[{asset_id!r}] must be bool shape (16,)")
            selected.append(candidate)
        expected = np.stack(selected, axis=0)
    else:
        expected = np.asarray(active_joint_masks)
        if expected.dtype.kind != "b" or expected.ndim != 2 or expected.shape[1:] != (16,):
            raise ValueError("active_joint_masks must be bool shape [asset_count,16]")
        if expected.shape[0] == len(requested_ids):
            pass  # 与 caller 的 asset_ids 同序
        elif expected.shape[0] == file_ids.shape[0]:
            expected = expected[[file_index[asset_id] for asset_id in requested_ids]]
        else:
            raise ValueError("active_joint_masks row count must match requested or file assets")
    actual = file_mask[[file_index[asset_id] for asset_id in requested_ids]]
    if not np.array_equal(expected, actual):
        raise ValueError("active_joint_masks disagree with candidate file")
    return np.asarray(actual, dtype=np.bool_)


def _validate_unique_candidate_pairs(projected_q: np.ndarray, positions: np.ndarray) -> None:
    r"""拒绝每资产 padding/repeated candidate，保持八个 projected(q,pos) 组合唯一。"""

    for asset_index in range(projected_q.shape[0]):
        pairs = np.concatenate(
            (
                projected_q[asset_index].reshape(REVALIDATION_CANDIDATE_COUNT, -1),
                positions[asset_index].reshape(REVALIDATION_CANDIDATE_COUNT, -1),
            ),
            axis=1,
        )
        if np.unique(pairs, axis=0).shape[0] != REVALIDATION_CANDIDATE_COUNT:
            raise ValueError(f"asset row {asset_index} contains repeated/padded projected candidates")


@dataclass(frozen=True)
class RevalidationCandidates:
    r"""按 new_asset_id 排列的只读、尚未 PhysX 认证的候选集合。

    projected_q_rad 与 original_q_rad 的形状都是 [A,8,16]，其中
    A 是 caller 请求的任意非空资产子集；position/orientation 分别是
    [A,8,3] 与 [A,8,4]。此数据结构刻意没有 q_target_rad 字段：
    revalidation 输入只表达待物理验证的 state 候选，不能从 state 推断 PD target。
    """

    projected_q_rad: np.ndarray
    original_q_rad: np.ndarray
    object_position_h_m: np.ndarray
    object_orientation_h_wxyz: np.ndarray
    parent_asset_id: np.ndarray
    parent_entry_digest: np.ndarray
    new_asset_id: np.ndarray
    active_joint_mask: np.ndarray
    canonical_joint_names: np.ndarray
    candidate_source: str = REVALIDATION_CANDIDATE_SOURCE
    physically_certified: bool = False
    ranking: str = REVALIDATION_RANKING

    def __post_init__(self) -> None:
        r"""复制并冻结 ndarray，防止 caller 修改已校验的输入视图。"""

        array_fields = (
            "projected_q_rad",
            "original_q_rad",
            "object_position_h_m",
            "object_orientation_h_wxyz",
            "parent_asset_id",
            "parent_entry_digest",
            "new_asset_id",
            "active_joint_mask",
            "canonical_joint_names",
        )
        for field_name in array_fields:
            value = np.asarray(getattr(self, field_name)).copy()
            value.setflags(write=False)
            object.__setattr__(self, field_name, value)
        if self.candidate_source != REVALIDATION_CANDIDATE_SOURCE:
            raise ValueError("RevalidationCandidates candidate_source is fixed")
        if self.physically_certified is not False:
            raise ValueError("revalidation candidates cannot be marked physically certified")
        if self.ranking != REVALIDATION_RANKING:
            raise ValueError("revalidation candidates must preserve parent rank")

    @property
    def asset_count(self) -> int:
        r"""返回当前 caller 子集的资产数 A。"""

        return int(self.new_asset_id.shape[0])

    @property
    def candidate_count(self) -> int:
        r"""返回每资产固定候选数 C=8。"""

        return int(self.projected_q_rad.shape[1])


def load_revalidation_candidates(
    path: str | Path,
    *,
    generation_identity: Mapping[str, Any],
    asset_ids: Sequence[str],
    active_joint_masks: Any,
    joint_names: Sequence[str],
) -> RevalidationCandidates:
    r"""加载并验证 revalidation NPZ，再按 new_asset_id 重排任意资产子集。

    文件必须包含固定的九个数组：original/projected canonical state、object
    position/orientation、active mask、parent/new IDs、parent entry digest 和
    canonical joint names。loader 先对原始 NPZ bytes 求 SHA-256，再验证
    generation identity；之后才读取数组并检查 finite、shape、ghost、投影、
    upright quaternion、candidate 唯一性以及 caller 提供的外部 mask/name。

    q_target_rad、target 或 qtarget 相关数组一律拒绝，因为它们会
    让待认证 state 与 PD preload target 的关系变得含糊。返回对象的数组均为
    read-only，且 physically_certified 固定为 False；真实 1 s strict
    gate 仍由 Isaac runner 完成。

    Args:
        path: 候选 NPZ 文件路径。
        generation_identity: 由 build_revalidation_generation_identity
            构造并通过 validation 的 identity。
        asset_ids: 要求输出的 new_asset_id，顺序就是返回顺序，允许任意非空子集。
        active_joint_masks: 与 asset_ids 同序的 bool [A,16]、file-order
            bool [N,16]，或按 new_asset_id 索引的 mapping。
        joint_names: caller canonical 16 joint names，必须逐项匹配 NPZ。

    Returns:
        RevalidationCandidates: 按 caller asset_ids 排列的只读候选。

    Raises:
        ValueError: 文件身份、协议、数组内容或外部 canonical binding 不一致。
        FileNotFoundError: path 不存在。
    """

    identity = validate_revalidation_generation_identity(generation_identity)
    candidate_path = Path(path).expanduser()
    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)

    actual_file_sha = hashlib.sha256(candidate_path.read_bytes()).hexdigest()  # NPZ 原始 bytes identity
    if actual_file_sha != identity["candidate_npz_sha256"]:
        raise ValueError("candidate NPZ SHA-256 does not match generation identity")

    requested_ids = tuple(str(asset_id) for asset_id in asset_ids)
    if not requested_ids:
        raise ValueError("asset_ids must be a non-empty subset")
    if any(not asset_id for asset_id in requested_ids):
        raise ValueError("asset_ids must contain non-empty IDs")
    if len(set(requested_ids)) != len(requested_ids):
        raise ValueError("asset_ids must not contain duplicate IDs")

    caller_names = np.asarray([str(name) for name in joint_names], dtype=str)
    if caller_names.shape != (16,) or any(not name for name in caller_names):
        raise ValueError("joint_names must contain 16 non-empty names")
    if len(set(caller_names.tolist())) != 16:
        raise ValueError("joint_names must be unique")

    try:
        with np.load(candidate_path, allow_pickle=False) as archive:
            keys = frozenset(archive.files)
            extra_keys = keys - _REQUIRED_NPZ_KEYS
            if extra_keys:
                target_keys = [
                    key for key in sorted(extra_keys) if any(marker in key.lower() for marker in _TARGET_KEY_MARKERS)
                ]
                if target_keys:
                    raise ValueError(f"candidate NPZ contains ambiguous qtarget arrays: {target_keys}")
                raise ValueError(f"candidate NPZ contains unexpected arrays: {sorted(extra_keys)}")
            missing_keys = _REQUIRED_NPZ_KEYS - keys
            if missing_keys:
                raise ValueError(f"candidate NPZ is missing required arrays: {sorted(missing_keys)}")
            arrays = {key: archive[key].copy() for key in _REQUIRED_NPZ_KEYS}
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError):
            raise
        raise ValueError(f"cannot read candidate NPZ: {error}") from error

    original_q = _numeric_array(arrays["original_q_rad"], field_name="original_q_rad")
    projected_q = _numeric_array(arrays["projected_q_rad"], field_name="projected_q_rad")
    positions = _numeric_array(arrays["object_position_h_m"], field_name="object_position_h_m")
    orientations = _numeric_array(
        arrays["object_orientation_h_wxyz"],
        field_name="object_orientation_h_wxyz",
    )
    file_mask = arrays["active_joint_mask"]
    if file_mask.dtype.kind != "b":
        raise ValueError("active_joint_mask must be a bool array")
    file_mask = np.asarray(file_mask, dtype=np.bool_)
    parent_ids = _validate_unique_ids(arrays["parent_asset_id"], field_name="parent_asset_id")
    new_ids = _validate_unique_ids(arrays["new_asset_id"], field_name="new_asset_id")
    parent_digests = _validate_entry_digests(arrays["parent_entry_digest"])
    file_joint_names = _validate_joint_names(arrays["canonical_joint_names"])

    # 所有 row/候选轴在这里一次性锁死，避免后面索引广播把错误 shape 静默吞掉。
    asset_count = new_ids.shape[0]
    expected_count = REVALIDATION_CANDIDATE_COUNT
    if not 1 <= asset_count <= REVALIDATION_MAX_ASSET_COUNT:
        raise ValueError("candidate NPZ asset count must lie in [1,128]")
    if original_q.shape != (asset_count, expected_count, 16):
        raise ValueError("original_q_rad must have shape [asset_count,8,16]")
    if projected_q.shape != original_q.shape:
        raise ValueError("projected_q_rad must have shape [asset_count,8,16]")
    if positions.shape != (asset_count, expected_count, 3):
        raise ValueError("object_position_h_m must have shape [asset_count,8,3]")
    if orientations.shape != (asset_count, expected_count, 4):
        raise ValueError("object_orientation_h_wxyz must have shape [asset_count,8,4]")
    if file_mask.shape != (asset_count, 16):
        raise ValueError("active_joint_mask must have shape [asset_count,16]")
    if parent_ids.shape != (asset_count,) or parent_digests.shape != (asset_count,):
        raise ValueError("asset IDs and parent_entry_digest must have one value per asset")
    if not np.array_equal(file_joint_names, caller_names):
        raise ValueError("canonical joint names disagree with caller binding")
    if not np.all(orientations == np.asarray(REVALIDATION_UPRIGHT_QUATERNION_WXYZ)):
        raise ValueError("object orientation must be exact upright quaternion wxyz=(1,0,0,0)")

    # inactive canonical slots 是 ghost；原 state 和 projection 两者都必须 exact zero。
    inactive = np.broadcast_to(~file_mask[:, None, :], original_q.shape)
    if np.any(original_q[inactive]) or np.any(projected_q[inactive]):
        raise ValueError("inactive canonical joint states must be exactly zero ghost")
    _validate_projection(original_q, projected_q, file_joint_names)
    _validate_unique_candidate_pairs(projected_q, positions)

    file_index = {asset_id: index for index, asset_id in enumerate(new_ids.tolist())}
    missing_ids = [asset_id for asset_id in requested_ids if asset_id not in file_index]
    if missing_ids:
        raise ValueError(f"asset_ids missing from candidate NPZ new_asset_id: {missing_ids}")
    selected_indices = np.asarray([file_index[asset_id] for asset_id in requested_ids], dtype=np.int64)
    selected_masks = _validate_external_masks(
        active_joint_masks,
        requested_ids=requested_ids,
        file_ids=new_ids,
        file_mask=file_mask,
    )

    return RevalidationCandidates(
        projected_q_rad=projected_q[selected_indices],
        original_q_rad=original_q[selected_indices],
        object_position_h_m=positions[selected_indices],
        object_orientation_h_wxyz=orientations[selected_indices],
        parent_asset_id=parent_ids[selected_indices],
        parent_entry_digest=parent_digests[selected_indices],
        new_asset_id=new_ids[selected_indices],
        active_joint_mask=selected_masks,
        canonical_joint_names=file_joint_names,
    )


__all__ = [
    "REVALIDATION_CANDIDATE_COUNT",
    "REVALIDATION_CANDIDATE_SOURCE",
    "REVALIDATION_GENERATION_ARTIFACT_TYPE",
    "REVALIDATION_GENERATION_KIND",
    "REVALIDATION_GENERATION_PROTOCOL",
    "REVALIDATION_GENERATION_SCHEMA_VERSION",
    "REVALIDATION_RANKING",
    "REVALIDATION_REFINED_RANKING",
    "REVALIDATION_REFINEMENT_GENERATION_ARTIFACT_TYPE",
    "REVALIDATION_REFINEMENT_GENERATION_KIND",
    "REVALIDATION_REFINEMENT_GENERATION_PROTOCOL",
    "REVALIDATION_REFINEMENT_GENERATION_SCHEMA_VERSION",
    "REVALIDATION_SCENE_SEED",
    "RevalidationCandidates",
    "build_revalidation_generation_identity",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "load_revalidation_candidates",
    "stable_digest",
    "validate_revalidation_generation_identity",
]
