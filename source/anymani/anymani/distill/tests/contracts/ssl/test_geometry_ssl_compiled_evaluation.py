from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from anymani.distill.methods.multi_anchor_gaussian_implicit_field import evaluation as evaluation_module


class _ReusedOutputModel:
    r"""模拟 CUDA Graph 在连续 forward 间复用同一输出 storage 的模型。"""

    def __init__(self) -> None:
        self.buffer = torch.zeros(4, 1)  # 四个 `(asset,q)` rows 共享的可复用输出 storage
        self.calls = 0  # forward 序号决定本次 prediction 的可辨识标量值

    def __call__(self, *_args: Any, **_kwargs: Any) -> SimpleNamespace:
        r"""覆写共享 buffer，并返回持有同一 tensor 引用的 prediction。"""

        self.calls += 1  # full/query/same/cross/joint 五次调用应依次得到 1--5
        self.buffer.fill_(float(self.calls))  # 模拟下一次 CUDAGraph replay 覆盖前一次输出
        return SimpleNamespace(kappa=self.buffer, density=self.buffer)


def test_ablation_metrics_are_consumed_before_reused_output_storage_is_overwritten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""每次消融 forward 必须立即归约，不能在后续 compiled 调用后再读取旧 tensor。

    真实 ``torch.compile(mode="reduce-overhead")`` 可让多次调用返回指向同一 CUDA Graph
    output pool 的 tensor；若先保存五个 prediction 再统一计算 metric，五组结果会全部读取最后
    一次 JOINT shuffle 的值。本回归测试用 CPU 共享 buffer 复现同一生命周期语义。
    """

    model = _ReusedOutputModel()  # 无 CUDA 依赖的 output-lifetime 反例
    batch = SimpleNamespace(
        q=torch.zeros(4, 1),  # `[B,N_J]=[4,1]`，只提供 fixed evidence API 所需 batch 轴
        q_index=torch.arange(4),  # 每条 record 的固定 q 配对键
        asset_ids=("a", "a", "b", "b"),  # 同资产与跨资产 permutation 都有合法双射
        evidence=object(),
        evidence_row_index=None,
        joint_coordinate_sign=None,
        queries=SimpleNamespace(query_points_h=torch.zeros(4, 1, 1, 3)),
        field_targets=SimpleNamespace(bandwidths=torch.ones(1)),
        sensitivity_targets=SimpleNamespace(
            owner_index=torch.zeros(4, 1, dtype=torch.long),
            query_index=torch.zeros(4, 1, dtype=torch.long),
            joint_index=torch.zeros(4, 1, dtype=torch.long),
        ),
    )

    # 让三种受控 ablation 都调用同一个复用-buffer 模型；permutation 数值本身不参与本反例。
    monkeypatch.setattr(
        evaluation_module,
        "geometry_ssl_ablation_forward",
        lambda current_model, *args, **kwargs: current_model(*args, **kwargs),
    )
    monkeypatch.setattr(
        evaluation_module,
        "same_asset_q_permutation",
        lambda _asset_ids, *, device: torch.arange(4, device=device),
    )
    monkeypatch.setattr(
        evaluation_module,
        "cross_asset_permutation",
        lambda _asset_ids, *, device: torch.tensor((2, 3, 0, 1), device=device),
    )

    def _metrics(prediction: SimpleNamespace, _batch: Any) -> dict[str, tuple[float, ...]]:
        r"""读取当前 buffer 值；延迟到第五次 forward 后调用会把所有结果误记为 5。"""

        values = tuple(float(value) for value in prediction.kappa[:, 0])  # 当前 forward 的四行标量
        return {name: values for name in ("density", "kappa", "derived_field")}

    monkeypatch.setattr(evaluation_module, "geometry_ssl_reconstruction_metrics_per_sample", _metrics)

    model_stub: Any = model  # 测试替身只实现该函数实际消费的 callable surface
    batch_stub: Any = batch  # SimpleNamespace 精确提供 fixed evidence 所读取的 typed 字段
    evidence: Any = evaluation_module.fixed_evaluation_ablation_evidence(model_stub, (batch_stub,))
    first = evidence["records"][0]["metrics"]  # 第一个 `(asset,q)` 的五组配对结果

    assert first["full"]["kappa"] == 1.0
    assert first["query_only"]["kappa"] == 2.0
    assert first["same_asset_q_shuffle"]["kappa"] == 3.0
    assert first["cross_asset_shuffle"]["kappa"] == 4.0
    assert first["joint_token_shuffle"]["kappa"] == 5.0
