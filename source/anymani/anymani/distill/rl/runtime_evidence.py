r"""rl_games build 前的 AnyMani runtime evidence 装配。

tasks 只交付 environment-owned canonical manifests；本模块在训练/回放进程内把 manifest lower 成
policy-owned只读输入。旧 five-mother route 仍使用 raw geometry evidence + learned encoder；正式
heterogeneous infra route 只构造 frozen $Z$，不 import SSL method/encoder。
"""

from __future__ import annotations

from typing import Any

from anymani.distill.diagnostics.recording.rl import record_optional_rl_phase


def attach_masked_runtime_evidence(agent_cfg: dict[str, Any]) -> None:
    r"""按 rl_games network name 注入 canonical evidence 或 frozen $Z$ provider。

    Args:
        agent_cfg (dict[str, Any]): Hydra/YAML 解析后的 rl_games root config；函数原位增加仅供
            ``runner.load()`` 使用的 ``nn.Module``/bank 对象，并写入 JSON-safe identity metadata。
    """

    params = agent_cfg["params"]  # rl_games params root
    if params["algo"]["name"] != "anymani_masked_ppo":
        return  # 普通 single-asset/temporal PPO 不需要 canonical runtime evidence
    network_cfg = params["network"]
    network_name = network_cfg["name"]
    if network_name == "anymani_canonical_masked":
        from anymani.distill.rl.canonical_evidence import build_canonical_evidence_bank
        from anymani.tasks.gm.canonical_unified_env_cfg import CANONICAL_ARTIFACTS, CANONICAL_HAND_SPAWN_CFG

        bank = build_canonical_evidence_bank(CANONICAL_HAND_SPAWN_CFG, CANONICAL_ARTIFACTS)
        network_cfg["canonical_evidence_bank"] = bank  # runner build 后成为 learned geometry encoder input
        params["config"]["canonical_evidence_asset_ids"] = bank.asset_ids
        params["config"]["canonical_physical_geometry_hashes"] = bank.physical_geometry_hashes
        print(f"[INFO] Canonical evidence bank: rows={len(bank.asset_ids)} assets={bank.asset_ids}")
        return
    if network_name == "anymani_heterogeneous_n000_masked":
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
            seed=0,  # frozen interface seed 独立于 PPO seed=42
            width=128,
        )
        network_cfg["frozen_z_provider"] = provider  # actor子模块；persistent buffers 进入 checkpoint
        params["config"]["anymani_identity"] = provider.identity  # agent.yaml/run provenance 的完整 row identity
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


__all__ = ["attach_masked_runtime_evidence"]
