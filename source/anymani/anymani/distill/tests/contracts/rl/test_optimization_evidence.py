r"""优化审计包的CPU往返、预算与不可覆盖合同。"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from anymani.distill.diagnostics.recording.rl.optimization_evidence import (
    read_optimization_evidence,
    write_optimization_evidence,
)


def test_audit_roundtrip_keeps_detached_data_and_rng(tmp_path) -> None:
    r"""保存纯数据，可在无模型/模拟器时安全读取，且不依赖训练buffer后续变化。"""
    value = torch.tensor([1.0, 2.0], requires_grad=True)
    path = tmp_path / "audit.pt"
    write_optimization_evidence(
        path,
        {
            "artifact_type": "palm-rotation-optimization-audit",
            "model": {"weight": value},
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
        },
    )
    with torch.no_grad():
        value.zero_()
    restored = read_optimization_evidence(path)
    torch.testing.assert_close(restored["model"]["weight"], torch.tensor([1.0, 2.0]))
    assert not restored["model"]["weight"].requires_grad
    assert isinstance(restored["numpy_rng"][1], torch.Tensor)
    with pytest.raises(FileExistsError):
        write_optimization_evidence(path, {})


def test_audit_budget_failure_does_not_publish(tmp_path) -> None:
    r"""大张量不能突破约定额度，也不留下看似完成的审计文件。"""
    path = tmp_path / "audit.pt"
    with pytest.raises(ValueError, match="byte budget"):
        write_optimization_evidence(path, {"x": torch.zeros(100)}, max_bytes=16)
    assert not path.exists()


def test_audit_rejects_live_modules(tmp_path) -> None:
    r"""不能把可执行模型对象pickle成审计数据。"""
    with pytest.raises(TypeError, match="only accepts data"):
        write_optimization_evidence(tmp_path / "audit.pt", {"model": torch.nn.Linear(1, 1)})
