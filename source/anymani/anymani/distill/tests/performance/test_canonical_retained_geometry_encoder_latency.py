r"""RTX 5070 Ti 上 canonical retained geometry encoder 的 40 ms 子预算。"""

from __future__ import annotations

import math
import statistics

import pytest
import torch
from anymani.assets.bank.path_utils import resolve_anymani_root
from anymani.distill.methods.density_material_jacobian.artifact import load_se3_retained_encoder_artifact
from anymani.distill.models.geometry_ssl import GeometrySSLModel, GeometrySSLModelCfg
from anymani.distill.models.input_adapters.geometry import (
    GeometryLatents,
    ImplicitGeometryEncoder,
    StaticGeometryEvidence,
)
from anymani.distill.models.input_adapters.se3_invariant_encoder import (
    SE3InvariantAnchorFrontendCfg,
    SE3InvariantGeometryEncoder,
    SE3InvariantGeometryEncoderCfg,
)
from anymani.distill.models.policy import CanonicalEvidenceBank, EmbodimentPolicyInput
from anymani.distill.rl.heterogeneous_masked_ppo import (
    HETEROGENEOUS_N040_HISTORY_OBS_DIM,
    HeterogeneousN040HistoryPpoBuilder,
)
from anymani.distill.rl.masked_ppo import AnyManiMaskedContinuousModel
from anymani.distill.rl.runtime.retained_geometry import RetainedGeometryProvider

pytestmark = pytest.mark.performance


def _canonical_single_structure_evidence(device: torch.device) -> StaticGeometryEvidence:
    r"""构造 LEAP-like 4×(4 JOINT+TIP) 静态证据；所有 tensor 在计时前已驻留 GPU。"""

    dtype = torch.float32
    owner_count, joint_count = 21, 16
    anchors = torch.empty(40, 3, device=device, dtype=dtype).uniform_(-0.05, 0.05)
    home_surface = torch.empty(owner_count, 64, 3, device=device, dtype=dtype).uniform_(-0.08, 0.12)
    screws = torch.zeros(joint_count, 6, device=device, dtype=dtype)
    screws[:, 2] = 1.0
    screws[:, 3] = torch.linspace(-0.04, 0.04, joint_count, device=device)
    screws[:, 4] = torch.linspace(0.01, -0.10, joint_count, device=device)

    parent = [-1]
    entity_role = [0]
    entity_joint_index = [-1]
    joint_entity_index: list[int] = []
    for finger in range(4):
        previous = 0
        for depth in range(4):
            entity = 1 + 5 * finger + depth
            parent.append(previous)
            entity_role.append(1)
            entity_joint_index.append(4 * finger + depth)
            joint_entity_index.append(entity)
            previous = entity
        parent.append(previous)
        entity_role.append(2)
        entity_joint_index.append(-1)

    graph = torch.full((owner_count, owner_count), 8, dtype=torch.long)
    graph.fill_diagonal_(0)
    adjacency = [[] for _ in range(owner_count)]
    for child, parent_index in enumerate(parent):
        if parent_index >= 0:
            adjacency[child].append(parent_index)
            adjacency[parent_index].append(child)
    for source in range(owner_count):
        frontier = [source]
        distance = {source: 0}
        for node in frontier:
            for neighbor in adjacency[node]:
                if neighbor not in distance:
                    distance[neighbor] = distance[node] + 1
                    frontier.append(neighbor)
        for target, value in distance.items():
            graph[source, target] = min(value, 8)

    parent_direction = torch.full_like(graph, 8)
    child_direction = torch.full_like(graph, 8)
    for source in range(owner_count):
        distance = 0
        ancestor = source
        while ancestor >= 0:
            parent_direction[source, ancestor] = min(distance, 8)
            child_direction[ancestor, source] = min(distance, 8)
            ancestor = parent[ancestor]
            distance += 1

    return StaticGeometryEvidence(
        anchors=anchors,
        home_surface_points=home_surface,
        home_surface_mask=torch.ones(owner_count, 64, device=device, dtype=torch.bool),
        palm_normal=torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype),
        space_screws=screws,
        q_home=torch.zeros(joint_count, device=device, dtype=dtype),
        entity_role=torch.tensor(entity_role, device=device, dtype=torch.long),
        entity_joint_index=torch.tensor(entity_joint_index, device=device, dtype=torch.long),
        joint_entity_index=torch.tensor(joint_entity_index, device=device, dtype=torch.long),
        shortest_path=graph.to(device),
        parent_direction=parent_direction.to(device),
        child_direction=child_direction.to(device),
    )


@pytest.mark.parametrize("encoder_kind", ("legacy", "se3"))
def test_canonical_retained_encoder_p95_is_at_most_40_ms(encoder_kind: str) -> None:
    r"""N031/N040 retained encoder 均以 $B=4096$、20 warmup、50 CUDA Events 验收。

    计时排除 H2D、source、decoder、policy 和 environment，只包含从 GPU-resident $q$/static evidence 到
    graph-backbone final-norm $Z$ 的完整 retained path。两种 frontend 参数布局完全一致，因此共同使用
    `582343` 参数锚点；N040 的 line projection 只能改变前端几何运算，不能偷换网络容量。
    """

    if not torch.cuda.is_available():
        pytest.skip("canonical retained-encoder performance contract requires CUDA")
    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(device)
    if "RTX 5070 Ti" not in device_name:
        pytest.skip(f"performance contract is bound to RTX 5070 Ti, found {device_name}")

    torch.manual_seed(20260813)
    torch.cuda.manual_seed_all(20260813)
    legacy_config = GeometrySSLModelCfg().encoder  # N031 width-128/layers-4 canonical 容量
    if encoder_kind == "legacy":
        model = ImplicitGeometryEncoder(legacy_config).to(device).eval()
    else:
        model = SE3InvariantGeometryEncoder(
            SE3InvariantGeometryEncoderCfg(
                frontend=SE3InvariantAnchorFrontendCfg(
                    relation_width=legacy_config.frontend.relation_width,
                    home_width=legacy_config.frontend.home_width,
                    screw_width=legacy_config.frontend.screw_width,
                    role_width=legacy_config.frontend.role_width,
                    length_scale_m=legacy_config.frontend.length_scale_m,
                ),
                backbone=legacy_config.backbone,
            )
        ).to(device).eval()
    evidence = _canonical_single_structure_evidence(device)
    q = torch.empty(4096, 16, device=device, dtype=torch.float32).uniform_(-0.7, 0.7)

    with torch.no_grad():
        single_sample_reference = model(q[:1], evidence)
        reference = model(q, evidence)
        repeated = model(q, evidence)
    assert reference.entities.shape == (4096, 21, 128)
    torch.testing.assert_close(repeated.entities, reference.entities, atol=0.0, rtol=0.0)
    torch.testing.assert_close(reference.entities[:1], single_sample_reference.entities, atol=2.0e-6, rtol=2.0e-6)

    torch.cuda.reset_peak_memory_stats(device)
    stream = torch.cuda.current_stream(device)
    with torch.no_grad():
        for _ in range(20):
            model(q, evidence)
        torch.cuda.synchronize(device)
        elapsed_ms: list[float] = []
        for _ in range(50):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            model(q, evidence)
            end.record(stream)
            end.synchronize()
            elapsed_ms.append(float(start.elapsed_time(end)))

    ordered = sorted(elapsed_ms)
    p95_ms = ordered[math.ceil(0.95 * len(ordered)) - 1]
    median_ms = statistics.median(ordered)
    max_ms = ordered[-1]
    peak_memory_mib = torch.cuda.max_memory_allocated(device) / (1024.0**2)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        {
            "device": device_name,
            "encoder_kind": encoder_kind,
            "batch_size": 4096,
            "warmup": 20,
            "events": 50,
            "median_ms": median_ms,
            "p95_ms": p95_ms,
            "max_ms": max_ms,
            "parameters": parameter_count,
            "peak_memory_mib": peak_memory_mib,
        }
    )

    assert parameter_count == 582343
    assert p95_ms <= 40.0, f"retained encoder p95={p95_ms:.3f} ms exceeds 40 ms sub-budget"


def _single_row_bank(evidence: StaticGeometryEvidence) -> CanonicalEvidenceBank:
    r"""把single-structure fixture加上asset row轴，供正式runtime provider消费。"""

    fields = {}
    for name in (
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
    ):
        value = getattr(evidence, name)
        fields[name] = value.unsqueeze(0) if value is not None else None  # `[...]->[A=1,...]`
    return CanonicalEvidenceBank(
        evidence=StaticGeometryEvidence(**fields),
        asset_ids=("canonical-performance-fixture",),
        physical_geometry_hashes=("canonical-performance-physical",),
    )


def test_n040_history30_full_actor_p95_is_below_48_ms() -> None:
    r"""完整learned actor以$B=4096$、20 warmups、50 CUDA Events验收48 ms门。

    计时输入均已驻留GPU，覆盖正式N040 artifact、逐JOINT History30 TCN、current/limit task injection、
    一层graph policy adapter、shared LayerNorm+Linear与global logstd输出。它排除H2D、Isaac/PhysX、
    ContactSensor和central critic update。
    """

    if not torch.cuda.is_available():
        pytest.skip("full heterogeneous actor performance contract requires CUDA")
    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(device)
    if "RTX 5070 Ti" not in device_name:
        pytest.skip(f"performance contract is bound to RTX 5070 Ti, found {device_name}")

    torch.manual_seed(20260831)
    torch.cuda.manual_seed_all(20260831)
    artifact_path = (
        resolve_anymani_root()
        / "logs/ssl/geometry_ssl_density_material_jacobian_se3_v0_8_1_extended512_matched"
        / "20260830T164445Z/retained_encoder.pt"
    )
    artifact = load_se3_retained_encoder_artifact(
        artifact_path,
        expected_sha256="cda44cc9eae5ca28a1a735176ef4764805559d13e235c52477b6ac438b20ddea",
    )
    evidence_bank = _single_row_bank(_canonical_single_structure_evidence(torch.device("cpu")))
    provider = RetainedGeometryProvider(
        artifact=artifact,
        evidence_bank=evidence_bank,
        dataset_digest="performance-dataset",
        manifest_digest="performance-manifest",
        canonical_schema_digest="performance-canonical-schema",
        evidence_source_config={"fixture": "40-anchor-64-home"},
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
    batch_size = 4096
    obs = torch.zeros(batch_size, HETEROGENEOUS_N040_HISTORY_OBS_DIM, device=device)
    history = obs[:, : 30 * 16 * 4].reshape(batch_size, 30, 16, 4)
    history[:, :, :, 0].uniform_(-0.7 / torch.pi, 0.7 / torch.pi)  # q/pi
    history[:, :, :, 1].uniform_(-0.7 / torch.pi, 0.7 / torch.pi)  # target/pi
    history[:, :, :, 2].uniform_(-1.0, 1.0)  # previous policy action
    history[:, :, :, 3].bernoulli_(0.4)  # owner fingertip contact bits
    limits = obs[:, 30 * 16 * 4 : 30 * 16 * 4 + 32].reshape(batch_size, 16, 2)
    limits[:, :, 0] = -1.0
    limits[:, :, 1] = 1.0
    obs[:, -17] = 0.0  # 所有env共享single-structure evidence row
    obs[:, -16:] = 1.0

    network = model.a2c_network.eval()

    def profile_cuda(callable_) -> dict[str, float]:
        r"""以10 warmups + 30 events报告同一GPU-resident component。"""

        with torch.inference_mode():
            for _ in range(10):
                callable_()
            torch.cuda.synchronize(device)
            samples = []
            stream = torch.cuda.current_stream(device)
            for _ in range(30):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(stream)
                callable_()
                end.record(stream)
                end.synchronize()
                samples.append(float(start.elapsed_time(end)))
        ordered_samples = sorted(samples)
        return {
            "median_ms": statistics.median(ordered_samples),
            "p95_ms": ordered_samples[math.ceil(0.95 * len(ordered_samples)) - 1],
        }

    rows = torch.zeros(batch_size, dtype=torch.long, device=device)
    latest = history[:, -1]
    provider_profile = profile_cuda(lambda: network.retained_geometry_provider.resolve(rows, latest[:, :, 0] * torch.pi))
    temporal_profile = profile_cuda(lambda: network.temporal_encoder(history, obs[:, -16:] > 0.5))
    static = network.retained_geometry_provider.resolve(rows, latest[:, :, 0] * torch.pi)
    temporal = network.temporal_encoder(history, static.joint_valid_mask)
    margin_lo = latest[:, :, 0] - limits[:, :, 0]
    margin_hi = limits[:, :, 1] - latest[:, :, 0]
    joint_features = torch.cat((latest, margin_lo.unsqueeze(-1), margin_hi.unsqueeze(-1)), dim=-1)
    owner_features = torch.zeros(batch_size, 21, 1, device=device)
    owner_features[:, 17:21, 0] = latest[:, :4, 3]
    policy_input = EmbodimentPolicyInput(
        owner_features=owner_features,
        joint_features=joint_features,
        owner_valid_mask=static.owner_valid_mask,
        joint_valid_mask=static.joint_valid_mask,
        shortest_path=static.shortest_path,
        parent_direction=static.parent_direction,
        child_direction=static.child_direction,
        asset_row=rows,
        geometry_entities=static.geometry_entities,
        temporal_features=temporal,
    )
    policy_profile = profile_cuda(lambda: network.policy(policy_input))

    with torch.inference_mode():
        output = network({"obs": obs})
        assert output[0].shape == output[1].shape == (batch_size, 16)
        for _ in range(20):
            network({"obs": obs})
        torch.cuda.synchronize(device)
        elapsed_ms: list[float] = []
        stream = torch.cuda.current_stream(device)
        for _ in range(50):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            network({"obs": obs})
            end.record(stream)
            end.synchronize()
            elapsed_ms.append(float(start.elapsed_time(end)))

    ordered = sorted(elapsed_ms)
    p95_ms = ordered[math.ceil(0.95 * len(ordered)) - 1]
    result = {
        "device": device_name,
        "batch_size": batch_size,
        "warmup": 20,
        "events": 50,
        "median_ms": statistics.median(ordered),
        "p95_ms": p95_ms,
        "max_ms": ordered[-1],
        "trainable_policy_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "frozen_n040_parameters": sum(parameter.numel() for parameter in provider.encoder.parameters()),
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / (1024.0**2),
        "provider_profile": provider_profile,
        "temporal_profile": temporal_profile,
        "policy_profile": policy_profile,
    }
    print(result)
    assert p95_ms < 48.0, f"full N040 History30 actor p95={p95_ms:.3f} ms violates strict <48 ms gate"


def test_ssl_only_decoder_cost_is_reported_outside_retained_budget() -> None:
    r"""按正式 $B=4,G=21,N_Q=64,N_\sigma=3,E=48$ 报告 disposable readers 前向成本。"""

    if not torch.cuda.is_available():
        pytest.skip("SSL-only decoder profile requires CUDA")
    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(device)
    if "RTX 5070 Ti" not in device_name:
        pytest.skip(f"performance contract is bound to RTX 5070 Ti, found {device_name}")

    model = GeometrySSLModel(GeometrySSLModelCfg()).to(device).eval()
    latents = GeometryLatents(entities=torch.randn(4, 21, 128, device=device))
    query_features = torch.randn(4, 21, 64, 64, device=device)
    bandwidths = torch.tensor([0.004, 0.016, 0.064], device=device).expand(4, -1)
    owner_index = torch.arange(48, device=device) % 21
    query_index = torch.arange(48, device=device) % 64
    joint_index = torch.arange(48, device=device) % 16

    def forward_decoders() -> None:
        model.decode_latents(
            latents,
            query_features,
            bandwidths=bandwidths,
            entity_valid_mask=torch.ones(4, 21, device=device, dtype=torch.bool),
            joint_entity_index=torch.arange(1, 17, device=device),
            owner_index=owner_index,
            query_index=query_index,
            joint_index=joint_index,
        )

    torch.cuda.reset_peak_memory_stats(device)
    stream = torch.cuda.current_stream(device)
    with torch.no_grad():
        for _ in range(20):
            forward_decoders()
        torch.cuda.synchronize(device)
        elapsed_ms: list[float] = []
        for _ in range(50):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            forward_decoders()
            end.record(stream)
            end.synchronize()
            elapsed_ms.append(float(start.elapsed_time(end)))

    decoder_parameter_count = sum(parameter.numel() for parameter in model.parameters()) - sum(
        parameter.numel() for parameter in model.encoder.parameters()
    )
    print(
        {
            "scope": "ssl_only_decoders_forward",
            "device": device_name,
            "logical_batch": 4,
            "median_ms": statistics.median(elapsed_ms),
            "p95_ms": sorted(elapsed_ms)[math.ceil(0.95 * len(elapsed_ms)) - 1],
            "max_ms": max(elapsed_ms),
            "parameters": decoder_parameter_count,
            "peak_memory_mib": torch.cuda.max_memory_allocated(device) / (1024.0**2),
        }
    )
    assert decoder_parameter_count == 298753
    assert all(key.startswith("encoder.") for key in model.retained_state_dict())
