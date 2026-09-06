r"""任意resolved cohort的strict Top-8 pregrasp生成与cache identity。

该协议逐值保留strict-v5 proposal、physics与hard-gate数值，只把manifest-local row随机流替换为
source/physical identity随机流。一次生成必须覆盖resolved lock中的全部成员后才提交index；资产基数由lock给出，
不固定为历史MVP80的80项。Object、scale、physics digest与旧v5相同，因此差异只来自搜索随机流与发布基数语义。
"""

from __future__ import annotations

import json
import os
from typing import Any

from anymani.pregrasp.schema import stable_digest

from .strict_good_pregrasp_identity import (
    STRICT_GOOD_PREGRASP_CEM_CANDIDATES,
    STRICT_GOOD_PREGRASP_CEM_ELITES,
    STRICT_GOOD_PREGRASP_CEM_ROUNDS,
    STRICT_GOOD_PREGRASP_GENERATION_IDENTITY,
    STRICT_GOOD_PREGRASP_OBJECT_SCALE,
    STRICT_GOOD_PREGRASP_PHYSICS_DIGEST,
    STRICT_GOOD_PREGRASP_PHYSICS_IDENTITY,
    STRICT_GOOD_PREGRASP_PHYSICS_TOP_K,
    STRICT_GOOD_PREGRASP_REQUIRE_STRICT,
    STRICT_GOOD_PREGRASP_SEED,
    STRICT_GOOD_PREGRASP_SOBOL_CANDIDATES,
)

# 目录是数据位置，进入实际run/evaluation identity；它不改变exact key的物理或搜索定义。
# 开发/验收可使用独立目录，避免为未见资产补建条目时改变既有训练目录的index摘要。
# 未显式设置时逐值保留训练默认位置，完整resume仍按原身份严格检查，不放宽恢复闸门。
COHORT_GOOD_PREGRASP_CATALOG_ROOT = os.environ.get(
    "ANYMANI_HETERO_GOOD_PREGRASP_CATALOG_ROOT",
    "outputs/pregrasp/catalogs/heterogeneous_rotation/strict-v1/dexcube/scale-1p1",
)  # 相同physical/object/scale/physics/generation identity仍可复用同一Top-8内容
if not COHORT_GOOD_PREGRASP_CATALOG_ROOT.strip():
    raise ValueError("cohort good-pregrasp catalog root must not be blank")
COHORT_GOOD_PREGRASP_EVIDENCE_ROOT = "outputs/pregrasp/search/heterogeneous_rotation/strict-v1/dexcube/scale-1p1"
"""每次cohort invocation在其ID/lock-SHA子目录保存candidate级证据，不覆盖其他cohort。"""
COHORT_GOOD_PREGRASP_OBJECT_SCALE = STRICT_GOOD_PREGRASP_OBJECT_SCALE  # 无量纲DexCube scale，固定1.1
COHORT_GOOD_PREGRASP_SEED = STRICT_GOOD_PREGRASP_SEED  # 全局proposal种子20260902
COHORT_GOOD_PREGRASP_SOBOL_CANDIDATES = STRICT_GOOD_PREGRASP_SOBOL_CANDIDATES  # 每资产初始256项
COHORT_GOOD_PREGRASP_PHYSICS_TOP_K = STRICT_GOOD_PREGRASP_PHYSICS_TOP_K  # 初始geometry Top-32
COHORT_GOOD_PREGRASP_CEM_ROUNDS = STRICT_GOOD_PREGRASP_CEM_ROUNDS  # 最多3轮refinement
COHORT_GOOD_PREGRASP_CEM_CANDIDATES = STRICT_GOOD_PREGRASP_CEM_CANDIDATES  # 每轮每失败资产128项
COHORT_GOOD_PREGRASP_CEM_ELITES = STRICT_GOOD_PREGRASP_CEM_ELITES  # 物理quality最高16项拟合低秩分布
COHORT_GOOD_PREGRASP_REQUIRE_STRICT = STRICT_GOOD_PREGRASP_REQUIRE_STRICT  # runtime逐Top-8复验hard gate
COHORT_GOOD_PREGRASP_PHYSICS_IDENTITY = STRICT_GOOD_PREGRASP_PHYSICS_IDENTITY  # 120 Hz/material/solver真值
COHORT_GOOD_PREGRASP_PHYSICS_DIGEST = STRICT_GOOD_PREGRASP_PHYSICS_DIGEST  # 上述physics identity的SHA-256


def cohort_good_pregrasp_generation_identity() -> dict[str, Any]:
    r"""返回带cohort基数与physical-stream语义的完整strict生成协议。

    先做JSON round-trip取得只含基础类型的深副本，防止修改本协议时原位污染历史MVP80 identity。Generation digest
    必须覆盖random-stream key和全选发布规则，否则同一exact cache key可能接受不同候选分布。
    """

    identity = json.loads(json.dumps(STRICT_GOOD_PREGRASP_GENERATION_IDENTITY))  # 独立JSON-safe协议副本
    identity["algorithm"] = "cohort-strict-good-pregrasp-v1"  # 与row-keyed strict-v5明确分域
    identity["refinement"]["random_stream_key"] = "source_content_and_physical_geometry_sha256"  # 跨manifest复用
    identity["publication"] = {
        "top_k_per_asset": 8,  # 每个exact hand-object-scale key固定八个ranked候选
        "require_all_selected_assets": True,  # 任一member不足Top-8则不提交本批index
        "cardinality_source": "resolved_cohort_lock",  # $A$来自冻结lock，而非硬编码80
    }
    return identity


COHORT_GOOD_PREGRASP_GENERATION_IDENTITY = cohort_good_pregrasp_generation_identity()  # JSON-safe协议真源
COHORT_GOOD_PREGRASP_GENERATION_DIGEST = stable_digest(COHORT_GOOD_PREGRASP_GENERATION_IDENTITY)  # exact key字段

__all__ = [
    "COHORT_GOOD_PREGRASP_CATALOG_ROOT",
    "COHORT_GOOD_PREGRASP_CEM_CANDIDATES",
    "COHORT_GOOD_PREGRASP_CEM_ELITES",
    "COHORT_GOOD_PREGRASP_CEM_ROUNDS",
    "COHORT_GOOD_PREGRASP_EVIDENCE_ROOT",
    "COHORT_GOOD_PREGRASP_GENERATION_DIGEST",
    "COHORT_GOOD_PREGRASP_GENERATION_IDENTITY",
    "COHORT_GOOD_PREGRASP_OBJECT_SCALE",
    "COHORT_GOOD_PREGRASP_PHYSICS_DIGEST",
    "COHORT_GOOD_PREGRASP_PHYSICS_IDENTITY",
    "COHORT_GOOD_PREGRASP_PHYSICS_TOP_K",
    "COHORT_GOOD_PREGRASP_REQUIRE_STRICT",
    "COHORT_GOOD_PREGRASP_SEED",
    "COHORT_GOOD_PREGRASP_SOBOL_CANDIDATES",
    "cohort_good_pregrasp_generation_identity",
]
