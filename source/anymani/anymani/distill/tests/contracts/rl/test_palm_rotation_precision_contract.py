from __future__ import annotations

import torch
from anymani.distill.rl.runtime.palm_rotation_precision import enforce_palm_rotation_precision


def test_precision_contract_overrides_runner_tf32_side_effect() -> None:
    r"""Runner即使先开启TF32，MVP入口也必须在模型构造前恢复严格FP32合同。"""

    original_matmul = bool(torch.backends.cuda.matmul.allow_tf32)  # 保存测试进程原global状态
    original_cudnn = bool(torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True  # 模拟rl_games Runner.__init__副作用
        torch.backends.cudnn.allow_tf32 = True
        flags = enforce_palm_rotation_precision()
        assert flags == {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False}
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_matmul  # 不污染同进程其它precision contracts
        torch.backends.cudnn.allow_tf32 = original_cudnn


def test_precision_contract_can_explicitly_enable_tf32_without_changing_tensor_dtype() -> None:
    r"""TF32候选必须同时开启两项backend flags；普通FP32 tensors本身不改dtype。"""

    original_matmul = bool(torch.backends.cuda.matmul.allow_tf32)
    original_cudnn = bool(torch.backends.cudnn.allow_tf32)
    try:
        flags = enforce_palm_rotation_precision(allow_tf32=True)
        assert flags == {"cuda_matmul_allow_tf32": True, "cudnn_allow_tf32": True}
        assert torch.ones(1).dtype == torch.float32  # TF32只是内部乘法精度，不是storage dtype
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_matmul
        torch.backends.cudnn.allow_tf32 = original_cudnn
