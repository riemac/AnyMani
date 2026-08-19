r"""Graph-biased encoder-only Transformer 的 lookup、加性偏置与全连接 attention 合同。"""

from __future__ import annotations

import hashlib

import pytest
import torch
from anymani.distill.models.backbones.geometry_transformer import (
    GraphBiasedTransformer,
    GraphBiasedTransformerCfg,
)
from anymani.distill.models.geometry_ssl import GeometrySSLModel, GeometrySSLModelCfg

pytestmark = pytest.mark.contract


def test_canonical_model_parameter_and_state_key_identity_is_frozen() -> None:
    """配置重排不得静默改变 canonical 参数量、模块 namespace 或 retained 边界。"""

    model = GeometrySSLModel(GeometrySSLModelCfg())
    state_key_digest = hashlib.sha256("\n".join(model.state_dict()).encode()).hexdigest()

    assert sum(parameter.numel() for parameter in model.parameters()) == 590856
    assert sum(parameter.numel() for parameter in model.encoder.parameters()) == 350407
    assert len(model.state_dict()) == 92
    assert state_key_digest == "5c46820b59eb76dc6be7ca14cc1e6cc7a2c373c1ad724994d209f2a494bfe650"
    assert all(key.startswith("encoder.") for key in model.retained_state_dict())


def test_graph_bias_is_exact_sum_of_shortest_parent_and_child_head_lookups() -> None:
    r"""每个实体对/注意力头的 bias 必须是三张离散关系表的逐项和。"""

    model = GraphBiasedTransformer(
        GraphBiasedTransformerCfg(
            hidden_width=4,
            layers=1,
            attention_heads=2,
            feedforward_width=8,
            max_graph_distance=3,
        )
    ).to(dtype=torch.float64)
    with torch.no_grad():
        bucket = torch.arange(4, dtype=torch.float64)
        model.shortest_path_bias.weight.copy_(torch.stack((bucket, 10.0 + bucket), dim=-1))
        model.parent_direction_bias.weight.copy_(torch.stack((100.0 + bucket, 110.0 + bucket), dim=-1))
        model.child_direction_bias.weight.copy_(torch.stack((1000.0 + bucket, 1010.0 + bucket), dim=-1))

    shortest = torch.tensor([[0, 1, 3], [1, 0, 2], [3, 2, 0]], dtype=torch.long)
    parent = torch.tensor([[0, 1, 2], [3, 0, 1], [3, 3, 0]], dtype=torch.long)
    child = parent.transpose(0, 1).contiguous()
    actual = model._graph_bias(shortest, parent, child)
    expected = (
        model.shortest_path_bias(shortest)
        + model.parent_direction_bias(parent)
        + model.child_direction_bias(child)
    ).permute(2, 0, 1)

    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    assert actual.shape == (2, 3, 3)


def test_max_distance_entity_still_contributes_through_full_attention() -> None:
    r"""图距离只提供有限加性 bias；末桶实体不能被误实现为 hard attention mask。"""

    model = GraphBiasedTransformer(
        GraphBiasedTransformerCfg(
            hidden_width=4,
            layers=1,
            attention_heads=1,
            feedforward_width=8,
            dropout=0.0,
            max_graph_distance=3,
        )
    ).to(dtype=torch.float64)
    layer = model.layers[0]
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        layer.attention_norm.weight.fill_(1.0)
        layer.feedforward_norm.weight.fill_(1.0)
        model.final_norm.weight.fill_(1.0)
        layer.qkv.weight[8:12].copy_(torch.eye(4, dtype=torch.float64))  # 只令 V=LN(token)
        layer.attention_output.weight.copy_(torch.eye(4, dtype=torch.float64))

    graph = torch.tensor([[0, 3, 3], [3, 0, 3], [3, 3, 0]], dtype=torch.long)
    baseline = torch.zeros(1, 3, 4, dtype=torch.float64)
    changed = baseline.clone()
    changed[0, 2] = torch.tensor([1.0, -1.0, 0.5, -0.5], dtype=torch.float64)

    baseline_output = model(baseline, graph, graph, graph)
    changed_output = model(changed, graph, graph, graph)

    assert torch.linalg.vector_norm(changed_output[0, 0] - baseline_output[0, 0]) > 1.0e-6
