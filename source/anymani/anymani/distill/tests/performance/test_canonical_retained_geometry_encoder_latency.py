r"""RTX 5070 Ti 上 canonical retained geometry encoder 的 40 ms 子预算。"""

from __future__ import annotations

import math
import statistics

import pytest
import torch
from anymani.distill.models.geometry_ssl import GeometrySSLModel, GeometrySSLModelCfg
from anymani.distill.models.input_adapters.geometry import (
    GeometryLatents,
    ImplicitGeometryEncoder,
    StaticGeometryEvidence,
)

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


def test_canonical_retained_encoder_p95_is_at_most_40_ms() -> None:
    r"""$B=4096$、20 warmup、50 CUDA Events；排除 H2D/source/decoder/policy/env。"""

    if not torch.cuda.is_available():
        pytest.skip("canonical retained-encoder performance contract requires CUDA")
    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(device)
    if "RTX 5070 Ti" not in device_name:
        pytest.skip(f"performance contract is bound to RTX 5070 Ti, found {device_name}")

    torch.manual_seed(20260813)
    torch.cuda.manual_seed_all(20260813)
    model = ImplicitGeometryEncoder(GeometrySSLModelCfg().encoder).to(device).eval()
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

    assert parameter_count == 317383
    assert p95_ms <= 40.0, f"retained encoder p95={p95_ms:.3f} ms exceeds 40 ms sub-budget"


def test_ssl_only_decoder_cost_is_reported_outside_retained_budget() -> None:
    r"""按正式 $B=4,G=21,N_Q=64,N_\sigma=3,E=42$ 报告 disposable readers 前向成本。"""

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
    owner_index = torch.arange(42, device=device) % 21
    query_index = torch.arange(42, device=device) % 64
    joint_index = torch.arange(42, device=device) % 16

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
    assert decoder_parameter_count == 513154
    assert all(key.startswith("encoder.") for key in model.retained_state_dict())
