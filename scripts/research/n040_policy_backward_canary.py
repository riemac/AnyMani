r"""验证compiled policy adapter backward只更新task/temporal参数而不触碰冻结N040。"""

from __future__ import annotations

import json

import torch
from anymani.assets.bank.path_utils import resolve_anymani_root
from anymani.distill.methods.density_material_jacobian.artifact import load_se3_retained_encoder_artifact
from anymani.distill.models.input_adapters.geometry import StaticGeometryEvidence
from anymani.distill.models.policy import CanonicalEvidenceBank
from anymani.distill.rl.heterogeneous_masked_ppo import (
    HETEROGENEOUS_N040_HISTORY_OBS_DIM,
    HeterogeneousN040HistoryPpoBuilder,
)
from anymani.distill.rl.masked_ppo import AnyManiMaskedContinuousModel
from anymani.distill.rl.runtime.retained_geometry import RetainedGeometryProvider
from anymani.distill.tests.performance.test_canonical_retained_geometry_encoder_latency import (
    _canonical_single_structure_evidence,
)


def _bank(evidence: StaticGeometryEvidence) -> CanonicalEvidenceBank:
    r"""把single-structure evidence转换成一行provider bank。"""

    names = (
        "anchors",
        "home_surface_points",
        "home_surface_mask",
        "palm_normal",
        "space_screws",
        "q_home",
        "entity_role",
        "entity_joint_index",
        "joint_entity_index",
        "shortest_path",
        "parent_direction",
        "child_direction",
        "entity_valid_mask",
        "joint_valid_mask",
        "anchor_valid_mask",
    )
    return CanonicalEvidenceBank(
        evidence=StaticGeometryEvidence(
            **{
                name: (getattr(evidence, name).unsqueeze(0) if getattr(evidence, name) is not None else None)
                for name in names
            }
        ),
        asset_ids=("backward-fixture",),
        physical_geometry_hashes=("backward-physical",),
    )


def main() -> int:
    r"""执行两次compiled forward/backward并报告梯度隔离。"""

    device = torch.device("cuda:0")
    artifact = load_se3_retained_encoder_artifact(
        resolve_anymani_root()
        / "logs/ssl/geometry_ssl_density_material_jacobian_se3_v0_8_1_extended512_matched"
        / "20260830T164445Z/retained_encoder.pt",
        expected_sha256="cda44cc9eae5ca28a1a735176ef4764805559d13e235c52477b6ac438b20ddea",
    )
    provider = RetainedGeometryProvider(
        artifact=artifact,
        evidence_bank=_bank(_canonical_single_structure_evidence(torch.device("cpu"))),
        dataset_digest="backward-dataset",
        manifest_digest="backward-manifest",
        canonical_schema_digest="backward-schema",
        evidence_source_config={"fixture": True},
    )
    builder = HeterogeneousN040HistoryPpoBuilder()
    builder.load(
        {
            "retained_geometry_provider": provider,
            "parallel_geometry_temporal": True,
            "compile_policy_adapter": True,
            "temporal_encoder": "stack_mlp",
            "heterogeneous_policy": {
                "owner_feature_dim": 1,
                "joint_feature_dim": 6,
                "temporal_feature_dim": 32,
                "geometry_entity_width": 128,
                "hidden_width": 128,
                "layers": 1,
                "attention_heads": 4,
                "feedforward_width": 256,
                "dropout": 0.0,
                "initial_log_std": -0.5,
            },
        }
    )
    model = AnyManiMaskedContinuousModel(builder).build(
        {
            "actions_num": 16,
            "input_shape": (HETEROGENEOUS_N040_HISTORY_OBS_DIM,),
            "value_size": 1,
            "normalize_input": False,
            "normalize_value": False,
        }
    ).to(device)
    duplicate_compiled_keys = tuple(
        key for key in model.state_dict() if "_policy_forward" in key or "_orig_mod" in key
    )
    if duplicate_compiled_keys:
        raise RuntimeError(f"compiled callable leaked duplicate checkpoint namespaces: {duplicate_compiled_keys}")
    batch_size = 64
    obs = torch.zeros(batch_size, HETEROGENEOUS_N040_HISTORY_OBS_DIM, device=device)
    history = obs[:, : 30 * 16 * 4].reshape(batch_size, 30, 16, 4)
    history[:, :, :, :3].normal_(mean=0.0, std=0.2)
    history[:, :, :, 3].bernoulli_(0.4)
    limits = obs[:, 30 * 16 * 4 : 30 * 16 * 4 + 32].reshape(batch_size, 16, 2)
    limits[:, :, 0] = -1.0
    limits[:, :, 1] = 1.0
    obs[:, -17] = 0.0
    obs[:, -16:] = 1.0
    prev_actions = torch.zeros(batch_size, 16, device=device)

    model.train()
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        output = model({"obs": obs, "prev_actions": prev_actions, "is_train": True})
        loss = output["prev_neglogp"].mean() + 0.01 * output["entropy"].mean() + output["values"].square().mean()
        loss.backward()
        if not torch.isfinite(loss):
            raise RuntimeError("compiled policy backward produced non-finite loss")
    frozen_grad_count = sum(parameter.grad is not None for parameter in provider.encoder.parameters())
    trainable_grad_count = sum(
        int(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all().item()))
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if frozen_grad_count != 0 or trainable_grad_count == 0:
        raise RuntimeError(
            f"invalid gradient partition frozen_grad_count={frozen_grad_count} trainable_grad_count={trainable_grad_count}"
        )
    print(
        json.dumps(
            {
                "batch_size": batch_size,
                "loss": float(loss.detach().item()),
                "frozen_n040_grad_count": frozen_grad_count,
                "trainable_finite_grad_count": trainable_grad_count,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
