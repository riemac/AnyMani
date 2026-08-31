r"""rl_games build 前的 canonical evidence 与 geometry provider 装配。

Task层只交付ordered source/canonical assets与environment routing。本模块拥有下游学习侧的artifact
校验、N040 source realization、provider identity和network config注入；它不定义MDP，也不修改冻结
schema-5 artifact。
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from anymani.assets.bank.path_utils import resolve_anymani_root
from anymani.distill.diagnostics.recording.rl import record_optional_rl_phase
from anymani.distill.methods.density_material_jacobian.artifact import load_se3_retained_encoder_artifact
from anymani.distill.representations.sources.geometry_source import AnchorBankCfg, GeometrySourceCfg
from anymani.distill.rl.canonical_evidence import build_canonical_evidence_bank
from anymani.distill.rl.runtime.retained_geometry import RetainedGeometryProvider

N040_PPO_SOURCE_CFG = GeometrySourceCfg(
    home_points_per_owner=64,
    home_surface_oversample_factor=8,
    static_sampling_seed=0,
    anchors=AnchorBankCfg(
        bank_size=8,
        anchors_per_finger=10,
        radius_m=0.05,
        radial_decay_scale_m=0.025,
        surface_fraction=0.5,
    ),
)
"""N040 snapshot的exact static evidence realization；PPO固定使用anchor bank $A^{(0)}$。"""


def _resolve_artifact_path(configured_path: str) -> Path:
    r"""把YAML中的repo-relative artifact path解析成绝对路径。"""

    path = Path(configured_path).expanduser()
    return path.resolve() if path.is_absolute() else (resolve_anymani_root() / path).resolve()


def _build_heterogeneous_n040_provider(
    retained_cfg: dict[str, Any],
) -> RetainedGeometryProvider:
    r"""从正式heterogeneous source rows构造冻结N040 q-dependent provider。

    Args:
        retained_cfg (dict[str, Any]): YAML中的artifact path与SHA；不接受任意source config覆盖。

    Returns:
        RetainedGeometryProvider: CPU materialized artifact/evidence provider；rl_games随后整体移到policy device。
    """

    from anymani.assets.canonical_runtime import CANONICAL_HAND_SCHEMA_V1
    from anymani.tasks.gm.config.heterogeneous_asset.asset_runtime import (
        HETEROGENEOUS_CANONICAL_ARTIFACTS,
        HETEROGENEOUS_GROUP_MANIFEST_DIGEST,
        HETEROGENEOUS_HAND_SPAWN_CFG,
        HETEROGENEOUS_SOURCE_ASSETS,
        PPO_DATASET,
    )

    if set(retained_cfg) != {"artifact_path", "artifact_sha256"}:
        raise ValueError("retained_geometry config must contain exactly artifact_path and artifact_sha256")
    artifact_path = _resolve_artifact_path(str(retained_cfg["artifact_path"]))
    artifact = load_se3_retained_encoder_artifact(
        artifact_path,
        expected_sha256=str(retained_cfg["artifact_sha256"]),
    )

    record_optional_rl_phase(
        "retained_geometry_evidence",
        "start",
        asset_count=len(HETEROGENEOUS_CANONICAL_ARTIFACTS),
        artifact_sha256=artifact.artifact_sha256,
    )
    evidence_bank = build_canonical_evidence_bank(
        HETEROGENEOUS_HAND_SPAWN_CFG,
        HETEROGENEOUS_CANONICAL_ARTIFACTS,
        source_assets=HETEROGENEOUS_SOURCE_ASSETS,
        source_cfg=N040_PPO_SOURCE_CFG,
        device="cpu",
    )
    provider = RetainedGeometryProvider(
        artifact=artifact,
        evidence_bank=evidence_bank,
        dataset_digest=PPO_DATASET.source_sha256,
        manifest_digest=HETEROGENEOUS_GROUP_MANIFEST_DIGEST,
        canonical_schema_digest=CANONICAL_HAND_SCHEMA_V1.digest,
        evidence_source_config=asdict(N040_PPO_SOURCE_CFG),
    )
    record_optional_rl_phase(
        "retained_geometry_evidence",
        "complete",
        asset_count=len(provider.asset_ids),
        identity_digest=provider.identity["identity_digest"],
        evidence_tensor_digest=provider.identity["evidence_tensor_digest"],
    )
    return provider


def attach_masked_runtime_evidence(agent_cfg: dict[str, Any]) -> None:
    r"""按rl_games network config注入canonical evidence或geometry provider。

    Heterogeneous route只有YAML显式声明`retained_geometry`时才使用N040；声明后任何artifact/source
    failure均直接终止，不允许静默回退hash-Z。没有该字段的旧N000 alias继续构造确定性interface surrogate。

    Args:
        agent_cfg (dict[str, Any]): Hydra/YAML解析后的rl_games root config；函数原位注入runtime modules。
    """

    params = agent_cfg["params"]
    if params["algo"]["name"] != "anymani_masked_ppo":
        return
    network_cfg = params["network"]
    network_name = network_cfg["name"]
    if network_name == "anymani_canonical_masked":
        from anymani.distill.rl.canonical_evidence import build_canonical_evidence_bank
        from anymani.tasks.gm.canonical_unified_env_cfg import CANONICAL_ARTIFACTS, CANONICAL_HAND_SPAWN_CFG

        bank = build_canonical_evidence_bank(CANONICAL_HAND_SPAWN_CFG, CANONICAL_ARTIFACTS)
        network_cfg["canonical_evidence_bank"] = bank
        params["config"]["canonical_evidence_asset_ids"] = bank.asset_ids
        params["config"]["canonical_physical_geometry_hashes"] = bank.physical_geometry_hashes
        print(f"[INFO] Canonical evidence bank: rows={len(bank.asset_ids)} assets={bank.asset_ids}")
        return
    if network_name in {"anymani_heterogeneous_n000_masked", "anymani_heterogeneous_n040_history30"}:
        retained_cfg = network_cfg.get("retained_geometry")
        if retained_cfg is not None:
            if not isinstance(retained_cfg, dict):
                raise TypeError("retained_geometry network config must be a mapping")
            provider = _build_heterogeneous_n040_provider(retained_cfg)
            network_cfg["retained_geometry_provider"] = provider
            params["config"]["anymani_identity"] = provider.identity
            print(
                "[INFO] Retained N040 provider: "
                f"rows={len(provider.asset_ids)} owners=21 width={provider.width} "
                f"identity={provider.identity['identity_digest']}"
            )
            return

        if network_name == "anymani_heterogeneous_n040_history30":
            raise ValueError("N040 History30 network requires explicit retained_geometry artifact config")

        # Old hash-Z path remains an explicitly separate infrastructure baseline.
        from anymani.distill.rl.frozen_z import build_frozen_z_provider_from_canonical_artifacts
        from anymani.tasks.gm.config.heterogeneous_asset.asset_runtime import (
            HETEROGENEOUS_CANONICAL_ARTIFACTS,
            HETEROGENEOUS_GROUP_MANIFEST_DIGEST,
            PPO_DATASET,
        )

        record_optional_rl_phase("frozen_z", "start", asset_count=len(HETEROGENEOUS_CANONICAL_ARTIFACTS))
        provider = build_frozen_z_provider_from_canonical_artifacts(
            HETEROGENEOUS_CANONICAL_ARTIFACTS,
            dataset_digest=PPO_DATASET.source_sha256,
            manifest_digest=HETEROGENEOUS_GROUP_MANIFEST_DIGEST,
            seed=0,
            width=128,
        )
        network_cfg["frozen_z_provider"] = provider
        params["config"]["anymani_identity"] = provider.identity
        record_optional_rl_phase(
            "frozen_z",
            "complete",
            asset_count=provider.z_table.shape[0],
            identity_digest=provider.identity["identity_digest"],
        )
        print(
            "[INFO] Frozen Z provider: "
            f"rows={provider.z_table.shape[0]} owners={provider.z_table.shape[1]} width={provider.width} "
            f"identity={provider.identity['identity_digest']}"
        )
        return
    raise ValueError(f"anymani_masked_ppo does not recognize network builder {network_name!r}")


__all__ = ["N040_PPO_SOURCE_CFG", "attach_masked_runtime_evidence"]
